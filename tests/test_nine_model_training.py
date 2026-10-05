"""Preparation-only synthetic contracts; never load released models or GPUs."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from test_continuous_qat import config, make

from w1a1_eagle.block_qat import (
    BlockDrafter,
    BlockQATConfig,
    block_loss,
    block_optimizer,
    block_train_step,
)
from w1a1_eagle.block_training import (
    BlockCursor,
    export_block_checkpoint,
    load_block_checkpoint,
    save_block_checkpoint,
    transition_a8_to_a1,
)
from w1a1_eagle.continuous_qat import rng_state


class NineModelEagleTests(unittest.TestCase):
    def test_direct_a1_exact_positive_step_resume(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            uninterrupted = make(Path(a), config(activation_bits=(1,), max_steps=3))
            uninterrupted.run(require_smoke=False)
            split = make(Path(b), config(activation_bits=(1,), max_steps=1))
            split.run(require_smoke=False)
            continued = make(Path(b), config(activation_bits=(1,), max_steps=3))
            continued.resume()
            continued.run(require_smoke=False)
            self.assertEqual([lane.name for lane in continued.lanes], ["A1"])
            self.assertEqual(continued.step, 3)
            self.assertEqual(continued.tokens, uninterrupted.tokens)
            self.assertEqual(continued.cursor, uninterrupted.cursor)
            for name in continued.lanes[0].linears:
                for key, tensor in continued.lanes[0].linears[name].state_dict().items():
                    if isinstance(tensor, torch.Tensor):
                        self.assertTrue(
                            torch.equal(
                                tensor, uninterrupted.lanes[0].linears[name].state_dict()[key]
                            ),
                            (name, key),
                        )
            actual = continued.lanes[0].optimizer.state_dict()
            expected = uninterrupted.lanes[0].optimizer.state_dict()
            self.assertEqual(actual["param_groups"], expected["param_groups"])
            for ident, state in actual["state"].items():
                for key, tensor in state.items():
                    self.assertTrue(torch.equal(tensor, expected["state"][ident][key]))


def block_fixture(family="dspark", bits=8, profile="ffn15_fusion"):
    cfg = BlockQATConfig(
        family,
        bits,
        hidden_size=8,
        intermediate_size=12,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        vocab_size=19,
        mask_token_id=18,
        profile=profile,
        conditioning="captured_prefix",
    )
    generator = torch.Generator().manual_seed(31)
    tensors = {
        "token_embd.weight": torch.randn(19, 8, generator=generator) * 0.2,
        "output.weight": torch.randn(19, 8, generator=generator) * 0.2,
        "fc.weight": torch.randn(8, 40, generator=generator) * 0.1,
        "enc.output_norm.weight": torch.ones(8),
        "output_norm.weight": torch.ones(8),
    }
    if family == "dspark":
        tensors["markov_w1.weight"] = torch.randn(19, 3, generator=generator) * 0.1
        tensors["markov_w2.weight"] = torch.randn(19, 3, generator=generator) * 0.1
    for i in range(5):
        shapes = {
            "attn_norm": (8,),
            "attn_q_norm": (4,),
            "attn_k_norm": (4,),
            "ffn_norm": (8,),
            "attn_q": (16, 8),
            "attn_k": (8, 8),
            "attn_v": (8, 8),
            "attn_output": (8, 16),
            "ffn_gate": (12, 8),
            "ffn_up": (12, 8),
            "ffn_down": (8, 12),
        }
        for name, shape in shapes.items():
            tensors[f"blk.{i}.{name}.weight"] = (
                torch.ones(shape)
                if len(shape) == 1
                else torch.randn(shape, generator=generator) * 0.1
            )
    batch = SimpleNamespace(
        context_features=torch.randn(3, 5, 8, generator=generator),
        prefix_tokens=(1, 2, 3, 4),
        input_tokens=torch.tensor([4] + [18] * 6),
        positions=torch.arange(3, 10),
        labels=torch.tensor([5, 6, 7, 8, 9, 10, 11]),
        predecessor_ids=torch.tensor([4, 5, 6, 7, 8, 9, 10]),
        loss_mask=torch.ones(7, dtype=torch.bool),
        attention_allowed=torch.ones(7, 10, dtype=torch.bool),
        teacher_logits=torch.randn(7, 19, generator=generator),
    )
    return cfg, tensors, batch


class BlockTrainingTests(unittest.TestCase):
    def test_all_profiles_hard_forward_gradients_and_private_ownership(self):
        for family in ("dspark", "dflash"):
            for bits in (1, 8):
                for profile in ("ffn15", "ffn15_fusion"):
                    cfg, tensors, batch = block_fixture(family, bits, profile)
                    model = BlockDrafter(tensors, cfg)
                    optimizer = block_optimizer(model)
                    report, output = block_train_step(model, optimizer, batch, update=False)
                    self.assertFalse(report["optimizer_updated"])
                    self.assertEqual(report["supervised_tokens"], 7)
                    self.assertEqual(output.logits.shape, (7, 19))
                    self.assertEqual(len(model.binary_linears()), 15 + (profile == "ffn15_fusion"))
                    self.assertNotEqual(
                        model.output.data_ptr(), tensors["output.weight"].data_ptr()
                    )
                    self.assertNotEqual(
                        model.token_embd.data_ptr(), tensors["token_embd.weight"].data_ptr()
                    )
                    self.assertFalse(model.output.requires_grad)
                    self.assertTrue(all(p.grad is not None for p in model.parameters()))

    def test_full_bidirectional_seven_compute_is_return_cap_invariant(self):
        cfg, tensors, batch = block_fixture()
        model = BlockDrafter(tensors, cfg)
        short, long = model(batch, return_proposals=3), model(batch, return_proposals=7)
        self.assertTrue(torch.equal(short.logits, long.logits))
        batch.attention_allowed[0, -1] = False
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            model(batch)

    def test_later_loss_retains_earlier_state_and_kv_gradients(self):
        cfg, tensors, batch = block_fixture()
        model = BlockDrafter(tensors, cfg)
        output = model(batch)
        loss = output.logits[-1].square().sum()
        earlier = output.layer_states[0]
        ck, cv = output.context_kv[0]
        gradients = torch.autograd.grad(loss, (earlier, ck, cv), retain_graph=True)
        self.assertTrue(all(torch.isfinite(g).all() and (g != 0).any() for g in gradients))
        loss.backward()
        self.assertTrue((model.fc.latent_sign.grad != 0).any())

    def test_native_greedy_static_prefix_divergence_masks_all_later_labels(self):
        cfg, tensors, batch = block_fixture()
        cfg = replace(cfg, conditioning="native_greedy")
        model = BlockDrafter(tensors, cfg)
        output = model(batch)
        batch.predecessor_ids = output.predecessor_ids.clone()
        batch.predecessor_ids[2] = (batch.predecessor_ids[2] + 1) % 19
        _, counts = block_loss(output, batch, cfg)
        self.assertEqual(counts["supervised_tokens"], 2)
        self.assertEqual(counts["prefix_mismatch_tokens"], 5)

    def test_current_prefix_teacher_callback_full_l1_and_teacher_is_frozen(self):
        cfg, tensors, batch = block_fixture()
        cfg = replace(cfg, conditioning="native_greedy", objective="full_probability_l1")
        model = BlockDrafter(tensors, cfg)
        output = model(batch)
        seen = []

        def teacher(prefix):
            seen.append(prefix)
            return torch.arange(19).float() * 0.1

        loss, counts = block_loss(output, batch, cfg, teacher_callback=teacher)
        self.assertEqual(counts["supervised_tokens"], 7)
        self.assertEqual(len(seen), 7)
        for slot, prefix in enumerate(seen):
            self.assertEqual(prefix[:4], batch.prefix_tokens)
            self.assertEqual(prefix[4:], tuple(output.logits[:slot].argmax(-1).tolist()))
        loss.backward()
        batch.teacher_logits = torch.randn(7, 3)
        with self.assertRaisesRegex(ValueError, "full-vocabulary"):
            block_loss(output, batch, replace(cfg, conditioning="captured_prefix"))


class BlockCheckpointTests(unittest.TestCase):
    source = {
        "base_gguf_sha256": "a" * 64,
        "data_manifest_sha256": "b" * 64,
        "bundle_sha256": "c" * 64,
        "synthetic": True,
    }

    def test_positive_step_resume_rng_cursor_optimizer_export(self):
        import json
        import random

        import numpy as np

        cfg, tensors, batch = block_fixture()
        model = BlockDrafter(tensors, cfg)
        optimizer = block_optimizer(model)
        block_train_step(model, optimizer, batch)
        cursor = BlockCursor(
            step=1, block_index=4, supervised_tokens=7, presented_tokens=7, unique_blocks=("p:0",)
        )
        with tempfile.TemporaryDirectory() as folder:
            receipt = save_block_checkpoint(model, optimizer, cursor, self.source, Path(folder))
            expected_rng = rng_state("cpu")
            random.random()
            np.random.rand()
            torch.rand(3)
            clone = BlockDrafter(tensors, cfg)
            other = block_optimizer(clone)
            durable = json.loads((Path(folder) / "latest.json").read_text())
            restored = load_block_checkpoint(clone, other, self.source, durable)
            self.assertEqual(restored, cursor)
            actual_rng = rng_state("cpu")
            self.assertEqual(actual_rng["python"], expected_rng["python"])
            self.assertTrue(torch.equal(actual_rng["torch"], expected_rng["torch"]))
            np.testing.assert_array_equal(actual_rng["numpy"][1], expected_rng["numpy"][1])
            block_train_step(model, optimizer, batch)
            block_train_step(clone, other, batch)
            for name, module in model.binary_linears().items():
                for key, value in module.state_dict().items():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(
                            torch.equal(value, clone.binary_linears()[name].state_dict()[key])
                        )
            for key, state in optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, other.state_dict()["state"][key][name]))
            export = export_block_checkpoint(clone, self.source, Path(folder) / "export")
            self.assertFalse(export["native_graph_admitted"])
            manifest = json.loads(Path(export["manifest"]).read_text())
            self.assertEqual(len(manifest["projections"]), 16)
            arrays = np.load(export["npz"])
            self.assertEqual(
                set(arrays), {p + s for p in manifest["projections"] for s in (".latent", ".scale")}
            )
            self.assertEqual(receipt["sha256"], durable["sha256"])

    def test_corrupt_optimizer_rejected_before_model_mutation(self):
        from w1a1_eagle.continuous_qat import sha256

        cfg, tensors, batch = block_fixture()
        model = BlockDrafter(tensors, cfg)
        optimizer = block_optimizer(model)
        block_train_step(model, optimizer, batch)
        with tempfile.TemporaryDirectory() as folder:
            receipt = save_block_checkpoint(
                model, optimizer, BlockCursor(step=1), self.source, Path(folder)
            )
            payload = torch.load(receipt["path"], weights_only=False)
            key = next(iter(payload["optimizer"]["state"]))
            payload["optimizer"]["state"][key]["exp_avg_sq"].fill_(-1)
            torch.save(payload, receipt["path"])
            receipt["sha256"] = sha256(Path(receipt["path"]))
            fresh = BlockDrafter(tensors, cfg)
            opt = block_optimizer(fresh)
            before = [p.clone() for p in fresh.parameters()]
            with self.assertRaisesRegex(ValueError, "second moment"):
                load_block_checkpoint(fresh, opt, self.source, receipt)
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, fresh.parameters())))

    def test_transition_preserves_exact_latents_scales_and_resets_moments(self):
        cfg, tensors, batch = block_fixture()
        model = BlockDrafter(tensors, cfg)
        optimizer = block_optimizer(model)
        block_train_step(model, optimizer, batch)
        result, opt, receipt = transition_a8_to_a1(model, source_checkpoint_sha256="d" * 64)
        self.assertFalse(receipt["exact_resume"])
        self.assertEqual(receipt["optimizer"], "reset")
        self.assertEqual(opt.state_dict()["state"], {})
        self.assertEqual(result.config.activation_bits, 1)
        for name, module in model.binary_linears().items():
            target = result.binary_linears()[name]
            self.assertEqual(target.contract.activation_bits, 1)
            self.assertTrue(torch.equal(module.latent_sign, target.latent_sign))
            self.assertTrue(torch.equal(module.initial_scale, target.initial_scale))
            self.assertTrue(torch.equal(module.scale_offset, target.scale_offset))
            self.assertNotEqual(module.latent_sign.data_ptr(), target.latent_sign.data_ptr())
        block_train_step(result, opt, batch)


class TypedDiagnosticTests(unittest.TestCase):
    def test_large_sign_counts_remain_exact_integers(self):
        from w1a1_eagle.recurrent_qat import typed_diagnostics

        report = typed_diagnostics(
            {"sign_flips": torch.tensor(2**28 + 3, dtype=torch.int64)},
            {"loss": torch.tensor(0.25)},
            device=torch.device("cpu"),
        )
        self.assertIs(type(report["sign_flips"]), int)
        self.assertEqual(report["sign_flips"], 2**28 + 3)
        self.assertEqual(report["loss"], 0.25)
        with self.assertRaisesRegex(ValueError, "exact integers"):
            typed_diagnostics({"count": torch.tensor(2.0)}, {}, device=torch.device("cpu"))


class LauncherContractTests(unittest.TestCase):
    def test_zero_update_smoke_exercises_actual_tiny_models_without_moments(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from train_nine_model_qat import smoke_block

        for family in ("dspark", "dflash"):
            for bits in (1, 8):
                cfg, tensors, batch = block_fixture(family, bits)
                model = BlockDrafter(tensors, cfg)
                optimizer = block_optimizer(model)
                report = smoke_block(model, batch, optimizer)
                self.assertEqual(report["optimizer_updates"], 0)
                self.assertEqual(optimizer.state_dict()["state"], {})
                self.assertGreater(report["later_state_gradient_norm"], 0)
                self.assertGreater(report["later_k_gradient_norm"], 0)
                self.assertGreater(report["later_v_gradient_norm"], 0)

    def test_admission_refuses_synthetic_wrong_device_or_stale_source(self):
        import json
        import sys
        from unittest.mock import patch

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from train_nine_model_qat import CHECKS, require_admission

        source = {"source.py": "d" * 64}
        record = {
            "schema": "nine_model_training_admission_v1",
            "status": "PASS",
            "artifact_kind": "production",
            "bundle_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "compute_capability": [12, 0],
            "optimizer_updates": 0,
            "gpu_uuid": "GPU-actual",
            "candidate": "dspark_a8",
            "source_files": source,
            "checks": dict.fromkeys(CHECKS, "PASS"),
        }
        with (
            tempfile.TemporaryDirectory() as folder,
            patch("train_nine_model_qat.training_source_identity", return_value=source),
        ):
            path = Path(folder) / "admission.json"
            path.write_text(json.dumps(record))
            require_admission(
                path, "a" * 64, "b" * 64, candidate="dspark_a8", gpu_uuid="GPU-actual"
            )
            for key, value in [
                ("artifact_kind", "synthetic"),
                ("compute_capability", [7, 5]),
                ("source_files", {}),
                ("gpu_uuid", "GPU-other"),
                ("candidate", "dflash_a8"),
            ]:
                bad = dict(record)
                bad[key] = value
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "SM120"):
                    require_admission(
                        path, "a" * 64, "b" * 64, candidate="dspark_a8", gpu_uuid="GPU-actual"
                    )


class LauncherLifecycleTests(unittest.TestCase):
    def transaction(
        self,
        folder,
        *,
        resume=False,
        max_steps=2,
        stop_after=None,
        fail_resource=False,
        precision_stage="direct",
    ):
        import contextlib
        import json
        import sys
        from unittest.mock import patch

        import train_nine_model_qat as launcher

        cfg, tensors, batch = block_fixture(bits=1 if precision_stage == "a8_to_a1" else 8)
        if precision_stage == "a8_to_a1":
            cfg = replace(cfg, activation_bits=8)
        model = BlockDrafter(tensors, cfg)
        batch.chain_id = "chain"
        batch.block_index = 0

        class DataCursor:
            def __init__(self, epoch=0, **kwargs):
                self.epoch = epoch

            def payload(self):
                return {"epoch": self.epoch}

        class Dataset:
            def cursor(self, **_):
                return DataCursor()

            def next_block(self, cursor, **_):
                return batch, DataCursor(epoch=cursor.epoch + 1)

        source = dict(BlockCheckpointTests.source)
        root = Path(folder)
        config = root / "config.json"
        config.write_text(json.dumps({"synthetic": "launcher_fixture"}))
        args = SimpleNamespace(
            config=config,
            run_dir=root,
            bundle_sha256="c" * 64,
            stage_name="dspark_a8/train",
            resume=resume,
            smoke_zero_updates=False,
            prepare_only=False,
        )
        spec = {
            "candidate": "dspark_a8",
            "family": "dspark",
            "checkpoint_every": 1,
            "limits": {"max_steps": max_steps, "max_seconds": 60},
            "precision_stage": precision_stage,
            "a8_warmup_steps": 1,
        }
        calls = 0

        def step(*pos, **kwargs):
            nonlocal calls
            calls += 1
            result = block_train_step(*pos, **kwargs)
            if stop_after is not None and calls >= stop_after:
                (root / "STOP").touch()
            return result

        def resource(*_):
            if fail_resource and calls >= 1:
                raise RuntimeError("synthetic resource floor failure")
            return {"fixture_only": True}

        @contextlib.contextmanager
        def teacher(*_):
            yield None

        with (
            patch.dict(
                sys.modules, {"w1a1_eagle.block_data": SimpleNamespace(BlockCursor=DataCursor)}
            ),
            patch.object(launcher, "block_inputs", return_value=(model, Dataset(), source)),
            patch.object(launcher, "resources", side_effect=resource),
            patch.object(launcher, "native_teacher", teacher),
            patch.object(launcher, "block_train_step", side_effect=step),
        ):
            return launcher.run_block(args, spec, {"hardware": "CPU synthetic fixture"})

    def test_success_publishes_hash_bound_export_from_committed_endpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.transaction(folder)
            self.assertEqual(result["completion_reason"], "approved_budget_complete")
            self.assertEqual(result["counters"]["step"], 2)
            self.assertEqual(set(result["checkpoint"]), {"path", "sha256"})
            for kind in ("checkpoint", "manifest"):
                locator = result["exports"]["dspark_a8"][kind]
                self.assertEqual(set(locator), {"path", "sha256"})
                self.assertTrue(Path(locator["path"]).is_file())

    def test_stop_retains_checkpoint_and_never_publishes_export(self):
        import json

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(InterruptedError, "STOP"):
                self.transaction(folder, stop_after=1, max_steps=3)
            latest = json.loads((Path(folder) / "checkpoints/latest.json").read_text())
            self.assertEqual(latest["cursor"]["step"], 1)
            self.assertFalse((Path(folder) / "final-export").exists())
            (Path(folder) / "STOP").unlink()
            resumed = self.transaction(folder, resume=True, max_steps=3)
            self.assertEqual(resumed["counters"]["step"], 3)

    def test_resource_failure_preserves_last_committed_checkpoint(self):
        import json

        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "resource floor"):
                self.transaction(folder, max_steps=3, fail_resource=True)
            latest = json.loads((Path(folder) / "checkpoints/latest.json").read_text())
            self.assertEqual(latest["cursor"]["step"], 1)
            self.assertFalse((Path(folder) / "final-export").exists())

    def test_stage_transition_charges_source_and_exercises_final_a1(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.transaction(folder, precision_stage="a8_to_a1")
            self.assertEqual(result["counters"]["step"], 2)
            self.assertEqual(result["counters"]["stage"], "a1_final")
            self.assertEqual(result["counters"]["supervised_tokens"], 14)


class CalibratedInitializationTests(unittest.TestCase):
    def test_sparse_overlay_preserves_reference_magnitudes_and_changes_inertia_only_by_probe(self):
        from w1a1_eagle.qat_initialization import apply_binary_initialization
        from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

        reference = torch.tensor([[0.2, -0.3]])
        first = RowBinaryLinear(reference, torch.tensor([0.5]), W1AxContract(8))
        second = RowBinaryLinear(reference, torch.tensor([0.5]), W1AxContract(8))
        fit = (torch.tensor([[0.2, -0.3]]), torch.tensor([0.7]))
        apply_binary_initialization(
            {"fc": first}, {"fc": fit}, policy="preserve_reference_magnitudes"
        )
        apply_binary_initialization(
            {"fc": second}, {"fc": (torch.tensor([[1.0, -1.0]]), fit[1])}, policy="unit_probe"
        )
        input = torch.ones(1, 2)
        self.assertTrue(torch.equal(first(input), second(input)))
        for module in (first, second):
            optimizer = torch.optim.SGD([module.latent_sign], lr=0.5)
            module(input).sum().backward()
            optimizer.step()
        self.assertTrue(bool(first.latent_sign[0, 0] < 0))
        self.assertTrue(bool(second.latent_sign[0, 0] > 0))

    def test_fusion_only_block_overlay_retains_other_fifteen_reference_latents(self):
        cfg, tensors, batch = block_fixture()
        reference = BlockDrafter(tensors, cfg)
        latent = -reference.fc.latent_sign.detach().clone()
        initialized = BlockDrafter(
            tensors,
            cfg,
            binary_initializer={"fc": (latent, reference.fc.initial_scale.detach().clone())},
        )
        self.assertTrue(torch.equal(initialized.fc.latent_sign, latent))
        for name, module in reference.binary_linears().items():
            if name != "fc":
                self.assertTrue(
                    torch.equal(module.latent_sign, initialized.binary_linears()[name].latent_sign)
                )
        with self.assertRaisesRegex(ValueError, "magnitudes"):
            BlockDrafter(
                tensors,
                cfg,
                binary_initializer={
                    "fc": (torch.where(latent < 0, -1.0, 1.0), reference.fc.initial_scale.clone())
                },
            )

    def test_negative_zero_is_refused_as_calibrated_negative_bit(self):
        from w1a1_eagle.qat_initialization import apply_binary_initialization
        from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

        module = RowBinaryLinear(torch.zeros(1, 1), torch.ones(1), W1AxContract(1))
        with self.assertRaisesRegex(ValueError, "negative-zero"):
            apply_binary_initialization(
                {"fc": module}, {"fc": (torch.tensor([[-0.0]]), torch.ones(1))}
            )


class ExplicitHardSignAdapterTests(unittest.TestCase):
    def test_declared_hard_sign_conversion_preserves_scaled_forward_and_reference_magnitude(self):
        from w1a1_eagle.qat_initialization import apply_binary_initialization
        from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

        for bits in (1, 8):
            reference = RowBinaryLinear(
                torch.tensor([[0.2, -0.3]]), torch.tensor([0.5]), W1AxContract(bits)
            )
            fit = (torch.tensor([[-1.0, 1.0]]), torch.tensor([0.7]))
            with self.assertRaisesRegex(ValueError, "magnitudes"):
                apply_binary_initialization(
                    {"fc": reference}, {"fc": fit}, policy="preserve_reference_magnitudes"
                )
            report = apply_binary_initialization(
                {"fc": reference},
                {"fc": fit},
                policy="preserve_reference_magnitudes",
                encoding="hard_signs",
            )
            self.assertTrue(torch.equal(reference.latent_sign, torch.tensor([[-0.2, 0.3]])))
            self.assertTrue(torch.equal(reference.initial_scale, fit[1]))
            self.assertEqual(report["fc"]["policy"], "preserve_reference_magnitudes")
            with self.assertRaisesRegex(ValueError, "negative-zero"):
                apply_binary_initialization(
                    {"fc": reference},
                    {"fc": (torch.tensor([[-0.0, 1.0]]), fit[1])},
                    policy="preserve_reference_magnitudes",
                    encoding="hard_signs",
                )

    def test_actual_saved_eagle_fit_npzs_install_without_changing_recipe_magnitudes(self):
        import json

        import numpy as np
        import train_nine_model_qat as launcher

        from w1a1_eagle.qat_initialization import apply_binary_initialization
        from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

        directory = Path(
            "/Users/pippo/github/binary-eagle-decoding/results/nine-model-qat-preparation/eagle-fusion-fixed-half-20261004"
        )
        if not directory.is_dir():
            self.skipTest("external real calibration artifact not materialized on this host")
        contract = json.loads((directory / "initialization-contract.json").read_text())
        for bits in (1, 8):
            record = contract["artifacts"][f"fusion-a{bits}"]
            locator = {key: record[key] for key in ("path", "sha256", "activation_bits")}
            locator["latent_initialization"] = {
                key: record["latent_initialization"][key]
                for key in ("policy", "reference_kind", "reference_sha256")
            }
            initializer = launcher.calibration(locator, W1AxContract(bits))
            latent, scale = initializer["fc"]
            module = RowBinaryLinear(
                torch.full_like(latent, 0.5), torch.ones_like(scale), W1AxContract(bits)
            )
            report = apply_binary_initialization(
                {"fc": module}, initializer, policy="preserve_reference_magnitudes"
            )
            launcher.validate_initializer_reference(report, locator, "eagle")
            self.assertTrue(torch.equal(module.latent_sign, latent))
            self.assertTrue(torch.equal(module.initial_scale, scale))
            self.assertTrue(bool((module.latent_sign.abs() == 0.5).all()))
            legacy = directory.parent / "eagle-fusion-20261004" / f"fusion-a{bits}.npz"
            with np.load(legacy, allow_pickle=False) as old:
                raw = (torch.from_numpy(old["fc.latent"]), torch.from_numpy(old["fc.scale"]))
                converter = RowBinaryLinear(
                    torch.full_like(latent, 0.5), torch.ones_like(scale), W1AxContract(bits)
                )
                apply_binary_initialization(
                    {"fc": converter},
                    {"fc": raw},
                    policy="preserve_reference_magnitudes",
                    encoding="hard_signs",
                )
                self.assertTrue(torch.equal(converter.latent_sign, module.latent_sign))
                self.assertTrue(torch.equal(converter.initial_scale, module.initial_scale))


class EagleCurriculumIntegrationTests(unittest.TestCase):
    def test_existing_runner_adapter_smokes_both_stages_zero_updates(
        self,
    ):
        from unittest.mock import patch

        import train_nine_model_qat as launcher
        from test_qat_curriculum_runner import TrainProvider

        from w1a1_eagle.continuous_qat import ContinuousConfig

        provider = TrainProvider()
        provider.bounded_rounds = provider.rounds
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = root / "config.json"
            config.write_text("{}")
            continuous = ContinuousConfig(
                activation_bits=(8,),
                checkpoint_every=1,
                min_free_disk_bytes=0,
                development_lifecycle="standalone",
                warmup_steps=0,
            )
            spec = {
                "family": "eagle",
                "candidate": "eagle_a1",
                "eagle_config": {"path": str(config), "sha256": launcher.sha256(config)},
                "prepared": {"run_dir": str(root / "original"), "ready_sha256": "a" * 64},
                "curriculum": {
                    "stages": [
                        {"activation_bits": 8, "gpu_seconds": 1000, "max_updates": 1},
                        {"activation_bits": 1, "gpu_seconds": 1000, "max_updates": 1},
                    ],
                    "optimizer_transition": "fresh",
                },
                "curriculum_runner": {
                    "checkpoint_every": 1,
                    "min_free_disk_bytes": 0,
                    "warmup_updates": 0,
                    "update_upper_bound_seconds": 0.01,
                },
            }
            args = SimpleNamespace(
                config=config,
                run_dir=root / "smoke",
                bundle_sha256="c" * 64,
                stage_name="eagle_a1/train",
                resume=False,
                smoke_zero_updates=True,
                prepare_only=False,
            )
            args.run_dir.mkdir()
            authenticated = {"binding": {"artifacts": {}}, "files": SimpleNamespace()}
            # Real tiny graph and existing runner; external source authentication
            # and hardware are isolated fixture boundaries, never production proof.
            with (
                patch("train_continuous_w1ax.load_config", return_value=({}, continuous)),
                patch("prepared_continuous_provider.authenticate", return_value=authenticated),
                patch(
                    "prepared_continuous_provider.PreparedProvider", side_effect=lambda *_: provider
                ),
                patch.object(launcher, "resources", return_value={"hardware": "CPU synthetic"}),
            ):
                result = launcher.run_eagle_curriculum(args, spec, {"compute_capability": [7, 5]})
            self.assertEqual([phase["activation_bits"] for phase in result["smoke"]], [8, 1])
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertEqual(result["training_memory"]["status"], "PENDING")
            self.assertTrue(Path(result["checkpoint"]["path"]).is_file())

    def test_existing_runner_exact_resume_before_after_reset_with_crash_budget(self):
        import json

        from test_qat_curriculum_runner import compare_models, stages
        from test_qat_curriculum_runner import make as curriculum_make

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            full = curriculum_make(root / "full", stages((8, 1), updates=2), crash_safe_budget=True)
            full.run(require_smoke=False)
            partial = curriculum_make(
                root / "partial", stages((8, 1), updates=2), crash_safe_budget=True
            )
            partial.run(require_smoke=False, max_new_updates=2)
            resumed = curriculum_make(
                root / "partial", stages((8, 1), updates=2), crash_safe_budget=True
            )
            resumed.resume()
            resumed.run(require_smoke=False)
            compare_models(full, resumed)
            ledger = json.loads((root / "partial" / "budget-used.json").read_text())
            self.assertIsNone(ledger["active_attempt"])
            self.assertGreater(ledger["training_seconds"], 0)
            self.assertEqual(resumed.state.global_updates, 4)


class FinalPrecisionExposureTests(unittest.TestCase):
    transaction = LauncherLifecycleTests.transaction

    def test_zero_a1_updates_cannot_grant_completed_training(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "required final A1 exposure"):
                self.transaction(folder, precision_stage="a8_to_a1", max_steps=1)
            self.assertFalse((Path(folder) / "final-export").exists())

    def test_stage_counts_expose_charged_warm_source_and_final_precision(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.transaction(folder, precision_stage="a8_to_a1", max_steps=3)
            self.assertEqual(result["counters"]["stage_updates"], 2)
            self.assertEqual(result["counters"]["stage_supervised_tokens"], 14)
            self.assertEqual(result["counters"]["step"], 3)
            self.assertEqual(result["counters"]["supervised_tokens"], 21)


class FixedFiniteA8Tests(unittest.TestCase):
    def test_subnormal_codes_follow_native_double_divide_then_multiply_f32_rne(self):
        from w1a1_eagle.recurrent_qat import fixed_activation_codes, hard_activation

        row = torch.tensor(
            [[1e-37, -1e-37, 5e-38, -5e-38, 0.0, -0.0]], dtype=torch.float32, requires_grad=True
        )
        codes, scale, _ = fixed_activation_codes(row, 8)
        reference = torch.round(
            (
                (row.detach().double() / row.detach().abs().amax(-1, keepdim=True).double()) * 127
            ).float()
        )
        self.assertTrue(torch.equal(codes, reference))
        self.assertEqual(codes[0, 0], 127)
        self.assertEqual(codes[0, 1], -127)
        self.assertEqual(codes[0, -1], 0)
        values, _, _ = hard_activation(row, 8)
        self.assertTrue(torch.isfinite(values).all())
        values.sum().backward()
        self.assertTrue(torch.equal(row.grad, torch.ones_like(row)))
        zeros = torch.tensor([[0.0, -0.0]], dtype=torch.float32)
        code, beta, _ = fixed_activation_codes(zeros, 8)
        self.assertTrue(torch.equal(code, torch.zeros_like(zeros)))
        self.assertEqual(beta, 0)

    def test_normal_domain_is_bit_identical_to_original_fixed_quantizer(self):
        from w1a1_eagle.recurrent_qat import fixed_activation_codes

        x = torch.randn(17, 128, generator=torch.Generator().manual_seed(7))
        old_scale = x.abs().amax(-1, keepdim=True) / 127
        old_codes = torch.round(x * (127 / x.abs().amax(-1, keepdim=True))).clamp(-127, 127)
        codes, scale, _ = fixed_activation_codes(x, 8)
        self.assertTrue(torch.equal(codes, old_codes))
        self.assertTrue(torch.equal(scale, old_scale))

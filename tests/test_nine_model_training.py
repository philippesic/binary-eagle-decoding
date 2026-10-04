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

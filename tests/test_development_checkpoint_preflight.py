"""Tiny real CPU producer/replay and fail-before-capture admission checks."""

import inspect
import json
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import check_continuous_w1ax_readiness as gate  # noqa: E402
import export_recurrent_binary as exporter  # noqa: E402
import w1ax_continuous_stages as stages  # noqa: E402
from test_continuous_qat import FixtureProvider, config  # noqa: E402
from test_recurrent_provider import dense_drafter, provider_round  # noqa: E402

from w1a1_eagle.continuous_qat import ContinuousTrainer, build_lanes  # noqa: E402
from w1a1_eagle.learned_activation import LearnedActivationBank  # noqa: E402
from w1a1_eagle.native_step import NativeStepAdapter  # noqa: E402
from w1a1_eagle.qat_state import deployment_state_sha256  # noqa: E402
from w1a1_eagle.recurrent_provider import forward_torch_round  # noqa: E402
from w1a1_eagle.recurrent_qat import install_joint_linears  # noqa: E402

PROFILES = json.loads((ROOT / "configs/qat_optimization_profiles.json").read_text())["profiles"]
DEPLOYMENT_KEYS = {"activation_quantization", "fusion_correction", "affine_weights"}


def tiny_models():
    model = dense_drafter(native_shape=True)
    model.config.num_attention_heads, model.config.num_key_value_heads = 32, 8
    model.midlayer.self_attn.q_proj = nn.Linear(8, 64, bias=False)
    model.midlayer.self_attn.k_proj = nn.Linear(8, 16, bias=False)
    model.midlayer.self_attn.v_proj = nn.Linear(8, 16, bias=False)
    model.midlayer.self_attn.o_proj = nn.Linear(64, 4, bias=False)
    with torch.no_grad():
        model.embed_tokens.weight.copy_(torch.linspace(-0.4, 0.6, 20).reshape(5, 4))
    return model, nn.Linear(4, 4)


class TinyProvider(FixtureProvider):
    def load_models(self):
        return tiny_models()


def logits(linears):
    from dataclasses import replace

    model, _ = tiny_models()
    for name, module in linears.items():
        parent, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        setattr(model.get_submodule(parent), leaf, module)
    batch = replace(provider_round(), raw_target_features=torch.tensor([[0.2, -0.1, 0.3, 0.4] * 3]))
    with torch.no_grad():
        return forward_torch_round(batch, NativeStepAdapter(model), 3)


class DevelopmentCheckpointPreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.serial = 0

    def produce(self, profile="baseline"):
        self.serial += 1
        directory = self.root / str(self.serial)
        provider = TinyProvider()
        cfg = config(**{k: v for k, v in PROFILES[profile].items() if k in DEPLOYMENT_KEYS})
        trainer = ContinuousTrainer(provider, build_lanes(provider, cfg), cfg, directory)
        with torch.no_grad():
            for lane in trainer.lanes:
                for module in lane.linears.values():
                    module.scale_offset.fill_(0.017)
                    if getattr(module, "affine_binary", None) is not None:
                        module.affine_binary.midpoint.copy_(
                            torch.linspace(-0.21, 0.13, module.out_features)
                        )
                if cfg.activation_quantization == "learned":
                    for q in LearnedActivationBank.from_attached(lane.linears).quantizers.values():
                        q.parameter.fill_(
                            0.1137 if lane.config.contract.activation_bits == 1 else 0.731
                        )
                correction = getattr(lane.linears["fc"], "fusion_correction", None)
                if correction is not None:
                    correction.u.fill_(0.1234567)
                    correction.v.fill_(-0.2134567)
                    if correction.output_bias is not None:
                        correction.output_bias.fill_(0.0234567)
        trainer.save()  # Real publisher hashes both NPZ+JSON; no optimizer step.
        self.assertEqual(trainer.step, 0)
        return trainer, Path(trainer.checkpoint["path"]).parent

    def preflight(self, root):
        with patch.object(
            stages, "host_admission", return_value={"fixture_only": True}
        ) as admission:
            prepared = stages._development_checkpoint_preflight(root, "a" * 64)
        return prepared, admission

    def rewrite_manifest(self, root, mutate, *, republish=False):
        path = root / "A1/joint.json"
        value = json.loads(path.read_text())
        mutate(value)
        path.write_text(json.dumps(value))
        if republish:
            publication = root / "manifest.json"
            value = json.loads(publication.read_text())
            value["exports"]["A1"]["joint.json"] = stages.sha256(path)
            publication.write_text(json.dumps(value))

    def test_real_paired_publication_and_faithful_all_optional_recipe_replay(self):
        names = (
            "baseline",
            "learned-activations",
            "fusion-rank1",
            "fusion-rank4",
            "affine-fusion",
            "affine-all",
            "combined-contract-smoke",
        )
        for name in names:
            with self.subTest(profile=name):
                trainer, root = self.produce(name)
                (configs, identities), admission = self.preflight(root)
                self.assertEqual(set(configs), {8, 1})
                self.assertGreaterEqual(admission.call_args.args[1], 12 * 1024**3)
                for lane in trainer.lanes:
                    stages._development_checkpoint_unchanged(
                        root, lane.config.contract.activation_bits, identities
                    )
                    loaded = install_joint_linears(
                        *tiny_models(), configs[lane.config.contract.activation_bits]
                    )
                    gate._load_checkpoint(
                        root / f"A{lane.config.contract.activation_bits}/joint.npz",
                        root / f"A{lane.config.contract.activation_bits}/joint.json",
                        loaded,
                        lane.config.contract.activation_bits,
                        "a" * 64,
                    )
                    self.assertEqual(
                        deployment_state_sha256(lane.linears), deployment_state_sha256(loaded)
                    )
                    torch.testing.assert_close(
                        logits(lane.linears), logits(loaded), rtol=3e-6, atol=3e-5
                    )
                    correction = getattr(loaded["fc"], "fusion_correction", None)
                    if correction is not None:
                        torch.testing.assert_close(
                            correction.u,
                            lane.linears["fc"].fusion_correction.u.half().float(),
                            rtol=0,
                            atol=0,
                        )

    def test_both_publication_hashes_checked_before_any_array_or_capture(self):
        for target in (
            "A8/joint.npz",
            "A8/joint.json",
            "A1/joint.npz",
            "A1/joint.json",
            "manifest.json",
        ):
            with self.subTest(target=target):
                _, root = self.produce("combined-contract-smoke")
                path = root / target
                path.write_bytes(
                    path.read_bytes() + b" "
                ) if target != "manifest.json" else path.unlink()
                with (
                    patch.object(
                        exporter, "load_checkpoint", side_effect=AssertionError("array staging")
                    ),
                    patch.object(stages, "native_capture", side_effect=AssertionError("capture")),
                    patch.object(
                        stages, "host_admission", side_effect=AssertionError("staging admission")
                    ),
                    self.assertRaises(ValueError),
                ):
                    stages._development_checkpoint_preflight(root, "a" * 64)

    def test_publication_inventory_precision_and_learned_json_downgrade_refused(self):
        for change in ("extra_lane", "missing_file", "wrong_precision", "downgrade", "bad_clip"):
            with self.subTest(change=change):
                _, root = self.produce("learned-activations")
                if change in ("extra_lane", "missing_file"):
                    path = root / "manifest.json"
                    value = json.loads(path.read_text())
                    if change == "extra_lane":
                        value["exports"]["A4"] = value["exports"]["A1"]
                    else:
                        del value["exports"]["A1"]["joint.json"]
                    path.write_text(json.dumps(value))
                elif change == "wrong_precision":
                    self.rewrite_manifest(
                        root, lambda v: v.update(activation_bits=4), republish=True
                    )
                elif change == "bad_clip":
                    path = root / "A8/joint.json"
                    value = json.loads(path.read_text())
                    value["activation_quantizers"]["boundaries"]["head"]["clip_ratio"] = 2**-17
                    path.write_text(json.dumps(value))
                    pub = root / "manifest.json"
                    value = json.loads(pub.read_text())
                    value["exports"]["A8"]["joint.json"] = stages.sha256(path)
                    pub.write_text(json.dumps(value))
                else:

                    def downgrade(v):
                        v.pop("activation_quantizers")
                        v.update(
                            schema_version=2,
                            activation_rule="a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive",
                        )

                    self.rewrite_manifest(root, downgrade)
                    # Bare JSON/NPZ is self-consistent; publication must reject it.
                    self.assertEqual(
                        gate.checkpoint_joint_config(
                            root / "A1/joint.json", 1, "a" * 64
                        ).activation_quantization,
                        "fixed",
                    )
                with (
                    patch.object(
                        exporter, "load_checkpoint", side_effect=AssertionError("array staging")
                    ),
                    patch.object(stages, "host_admission", return_value={}),
                    self.assertRaises(ValueError),
                ):
                    stages._development_checkpoint_preflight(root, "a" * 64)

    def test_host_admission_precedes_arrays_and_packs_released(self):
        _, root = self.produce("combined-contract-smoke")
        with (
            patch.object(stages, "host_admission", side_effect=RuntimeError("host unavailable")),
            patch.object(
                exporter, "load_fusion_correction", side_effect=AssertionError("optional arrays")
            ),
            patch.object(exporter, "load_checkpoint", side_effect=AssertionError("arrays")),
            self.assertRaisesRegex(RuntimeError, "host unavailable"),
        ):
            stages._development_checkpoint_preflight(root, "a" * 64)
        original = exporter.load_checkpoint
        references, events = [], []

        def track(*args, **kwargs):
            self.assertIn("admission", events)
            if references:
                self.assertTrue(all(ref() is None for ref in references))
            values = original(*args, **kwargs)
            references.extend(weakref.ref(v["packed"]) for v in values.values())
            events.append("arrays")
            return values

        with (
            patch.object(
                stages, "host_admission", side_effect=lambda *_: events.append("admission")
            ),
            patch.object(exporter, "load_checkpoint", side_effect=track),
        ):
            stages._development_checkpoint_preflight(root, "a" * 64)
        self.assertTrue(all(ref() is None for ref in references))
        self.assertEqual(events, ["admission", "arrays", "arrays"])

    def test_declared_large_shape_increases_reservation_before_array_reads(self):
        _, root = self.produce()
        self.rewrite_manifest(
            root, lambda v: v["projections"]["fc"].update(shape=[2**30, 12]), republish=True
        )

        def deny(_stage, additional):
            self.assertGreater(additional, 12 * 1024**3)
            raise RuntimeError("large staging reservation refused")

        with (
            patch.object(stages, "host_admission", side_effect=deny),
            patch.object(exporter, "load_checkpoint", side_effect=AssertionError("arrays")),
            self.assertRaisesRegex(RuntimeError, "reservation refused"),
        ):
            stages._development_checkpoint_preflight(root, "a" * 64)

    def test_republished_undeclared_npz_array_retains_strict_loader_rejection(self):
        _, root = self.produce()
        checkpoint = root / "A1/joint.npz"
        with np.load(checkpoint, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        arrays["undeclared"] = np.zeros(1, dtype=np.float32)
        np.savez(checkpoint, **arrays)
        self.rewrite_manifest(
            root, lambda v: v.update(checkpoint_sha256=stages.sha256(checkpoint)), republish=True
        )
        publication = root / "manifest.json"
        value = json.loads(publication.read_text())
        value["exports"]["A1"]["joint.npz"] = stages.sha256(checkpoint)
        publication.write_text(json.dumps(value))
        with (
            patch.object(stages, "host_admission", return_value={}),
            self.assertRaisesRegex(ValueError, "exactly nine"),
        ):
            stages._development_checkpoint_preflight(root, "a" * 64)

    def test_changed_lane_or_publication_after_preflight_and_during_staging_refused(self):
        for target in ("A8/joint.npz", "A1/joint.json", "manifest.json"):
            _, root = self.produce()
            (configs, identities), _ = self.preflight(root)
            path = root / target
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                stages._development_checkpoint_unchanged(root, 8, identities)
        _, root = self.produce()
        original = exporter.load_checkpoint

        def mutate(*args, **kwargs):
            values = original(*args, **kwargs)
            path = root / "A1/joint.json"
            path.write_bytes(path.read_bytes() + b" ")
            return values

        with (
            patch.object(stages, "host_admission", return_value={}),
            patch.object(exporter, "load_checkpoint", side_effect=mutate),
            self.assertRaisesRegex(ValueError, "identity changed"),
        ):
            stages._development_checkpoint_preflight(root, "a" * 64)

    def test_evaluator_calls_preflight_and_pair_guards_before_capture_or_model_load(self):
        _, root = self.produce("learned-activations")
        # Corrupt the second lane; no native action or full-model constructor may
        # run before both lane identities have passed actual production preflight.
        path = root / "A1/joint.json"
        path.write_bytes(path.read_bytes() + b" ")
        cfg = {
            "schema": "w1ax_continuous_development_v1",
            "split": "development",
            "sources": {"sha256": {"base_draft_gguf": "a" * 64}},
        }
        with (
            patch.object(stages, "native_capture", side_effect=AssertionError("native action")),
            patch.object(exporter, "export_model", side_effect=AssertionError("export action")),
            self.assertRaises(ValueError),
        ):
            stages._evaluate_development(cfg, root, self.root)
        source = inspect.getsource(stages._evaluate_development)
        self.assertLess(
            source.index("_development_checkpoint_preflight("), source.index("export_model(")
        )
        self.assertLess(
            source.index("_development_checkpoint_preflight("), source.index("native_capture(")
        )
        self.assertIn(
            'qat = replace(checkpoint_configs[bits], device="cuda:0", allow_accelerator=True)',
            source,
        )
        self.assertNotIn("qat = JointQATConfig", source)
        before_load = source[: source.index("provider.load_models_cpu()")]
        self.assertTrue(
            before_load.rstrip().endswith("identities)")
            or "_development_checkpoint_unchanged(checkpoint_dir, bits, checkpoint_identities)"
            in before_load[-180:]
        )
        with self.assertRaisesRegex(ValueError, "A4/curriculum"):
            stages._development_checkpoint_unchanged(root, 4, {"lanes": {}})


if __name__ == "__main__":
    unittest.main()

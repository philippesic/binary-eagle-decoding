"""CPU source identity/receipt fixtures; no real CUDA admission or models."""

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import test_qat_readiness as fixtures

from w1a1_eagle import continuous_runtime, qat_curriculum_runner
from w1a1_eagle.qat_readiness import (
    optimization_requires_receipt,
    validate_optimization_readiness,
)

HELPER = "activation_reuse.py"
SOURCE_ROOT = Path(continuous_runtime.__file__).parent


class ActivationIdentityTests(unittest.TestCase):
    def source_copy(self, root):
        for name in continuous_runtime.MATH_FILES:
            (root / name).write_bytes((SOURCE_ROOT / name).read_bytes())

    def test_helper_byte_change_binds_runtime_with_reuse_disabled(self):
        from test_continuous_qat import config, make

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = make(root / "tiny", config(max_steps=1))
            self.assertTrue(all(lane.adapter.activation_reuse is False for lane in trainer.lanes))
            self.source_copy(root)
            before = continuous_runtime.training_runtime_identity("cpu", source_root=root)
            helper = root / HELPER
            helper.write_bytes(helper.read_bytes() + b"\n# CPU identity byte-change fixture\n")
            after = continuous_runtime.training_runtime_identity("cpu", source_root=root)
        self.assertNotEqual(before, after)
        self.assertEqual(
            [
                name
                for name in before["math_source_sha256"]
                if before["math_source_sha256"][name] != after["math_source_sha256"][name]
            ],
            [HELPER],
        )

    def test_missing_helper_cannot_generate_runtime_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_copy(root)
            (root / HELPER).unlink()
            with self.assertRaises(FileNotFoundError):
                continuous_runtime.training_runtime_identity("cpu", source_root=root)

    def test_curriculum_inventory_binds_same_helper_bytes(self):
        self.assertIn(HELPER, qat_curriculum_runner.EXTRA_MATH)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.source_copy(root)
            for name in qat_curriculum_runner.EXTRA_MATH:
                if not (root / name).exists():
                    (root / name).write_bytes((SOURCE_ROOT / name).read_bytes())
            with (
                mock.patch.object(qat_curriculum_runner, "__file__", str(root / "runner.py")),
                mock.patch.object(
                    qat_curriculum_runner,
                    "training_runtime_identity",
                    side_effect=lambda device: continuous_runtime.training_runtime_identity(
                        device, source_root=root
                    ),
                ),
            ):
                before = qat_curriculum_runner.curriculum_runtime("cpu")
                (root / HELPER).write_bytes(b"# changed CPU helper identity fixture\n")
                after = qat_curriculum_runner.curriculum_runtime("cpu")
            for identity in (before, after):
                self.assertEqual(
                    identity["math_source_sha256"][HELPER],
                    identity["curriculum_math_sha256"][HELPER],
                )
            self.assertNotEqual(before, after)

    def validate(self, mutate=None):
        config = replace(
            fixtures.FixtureConfig(), a1_computation="reference", activation_quantization="fixed"
        )
        self.assertFalse(optimization_requires_receipt(config))
        context = copy.deepcopy(fixtures.CONTEXT)
        context["runtime_identity"]["math_source_sha256"] = (
            continuous_runtime.training_runtime_identity("cpu")["math_source_sha256"]
        )
        with mock.patch.dict(fixtures.CONTEXT, context, clear=True):
            receipt = fixtures.synthetic_receipt(config)
        if mutate:
            mutate(context, receipt)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "CPU_SCHEMA_FIXTURE.json"
            data = json.dumps(receipt).encode()
            path.write_bytes(data)
            config = replace(
                config,
                optimization_readiness={
                    "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
            )
            return validate_optimization_readiness(config, **context)

    def test_complete_default_recipe_receipt_passes(self):
        result = self.validate()
        self.assertIn(HELPER, result["training_runtime"]["math_source_sha256"])

    def test_mutual_helper_omission_rejected_for_default_recipe(self):
        def omit(context, receipt):
            context["runtime_identity"]["math_source_sha256"].pop(HELPER)
            receipt["training_runtime"]["math_source_sha256"].pop(HELPER)

        with self.assertRaisesRegex(ValueError, "requires activation_reuse.py"):
            self.validate(omit)

    def test_changed_current_helper_rejects_prior_receipt(self):
        def change(context, receipt):
            context["runtime_identity"]["math_source_sha256"][HELPER] = "f" * 64

        with self.assertRaisesRegex(ValueError, "training_runtime differs"):
            self.validate(change)

    def test_receipt_helper_omission_rejected_against_complete_context(self):
        with self.assertRaisesRegex(ValueError, "training_runtime differs"):
            self.validate(
                lambda context, receipt: receipt["training_runtime"]["math_source_sha256"].pop(
                    HELPER
                )
            )

    def test_helper_byte_mismatch_rejects_continuous_resume_before_checkpoint_load(self):
        from test_continuous_qat import config, make

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trained = make(root / "run", config(max_steps=1))
            trained.run(require_smoke=False)
            resumed = make(root / "run", config(max_steps=2))
            source = root / "source"
            source.mkdir()
            self.source_copy(source)
            (source / HELPER).write_bytes(b"# changed CPU resume helper fixture\n")
            changed = continuous_runtime.training_runtime_identity("cpu", source_root=source)
            pointer = (root / "run/latest.json").read_bytes()
            with (
                mock.patch(
                    "w1a1_eagle.continuous_qat.training_runtime_identity", return_value=changed
                ),
                mock.patch(
                    "w1a1_eagle.continuous_qat.torch.load",
                    side_effect=AssertionError("checkpoint loaded before identity gate"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "critical training math"):
                    resumed.resume()
            self.assertEqual(resumed.step, 0)
            self.assertEqual(resumed.cursor, 0)
            self.assertEqual((root / "run/latest.json").read_bytes(), pointer)

    def test_launch_helper_mismatch_rejects_before_training_or_publication(self):
        from test_continuous_qat import config, make

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trainer = make(root / "run", config(max_steps=1))
            self.assertTrue(all(lane.adapter.activation_reuse is False for lane in trainer.lanes))
            runtime = copy.deepcopy(fixtures.CONTEXT["runtime_identity"])
            runtime["math_source_sha256"] = continuous_runtime.training_runtime_identity("cpu")[
                "math_source_sha256"
            ]
            trainer.config = replace(
                trainer.config,
                device="cuda:0",
                optimize_cache=True,
                max_cuda_reserved_bytes=9000,
                min_cuda_free_bytes=100,
            )
            context = copy.deepcopy(fixtures.CONTEXT)
            context.update(source_sha256=trainer.source, runtime_identity=runtime)
            with mock.patch.dict(fixtures.CONTEXT, context, clear=True):
                receipt = fixtures.synthetic_receipt(trainer.config)
            data = json.dumps(receipt).encode()
            path = root / "CPU_LAUNCH_SCHEMA_FIXTURE.json"
            path.write_bytes(data)
            trainer.config = replace(
                trainer.config,
                optimization_readiness={
                    "path": str(path),
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
            )
            runtime["math_source_sha256"][HELPER] = "f" * 64
            before = {
                (lane.name, name): parameter.detach().clone()
                for lane in trainer.lanes
                for name, parameter in lane.drafter.named_parameters()
            }
            properties = SimpleNamespace(
                name=context["hardware"]["name"], major=12, minor=0, total_memory=10000
            )
            with (
                mock.patch(
                    "w1a1_eagle.continuous_qat.training_runtime_identity", return_value=runtime
                ),
                mock.patch("torch.cuda.get_device_properties", return_value=properties),
                mock.patch(
                    "w1a1_eagle.qat_readiness.native_checkout_commit",
                    return_value=context["native_commit"],
                ),
                mock.patch(
                    "w1a1_eagle.continuous_qat.joint_train_step",
                    side_effect=AssertionError("trained before identity gate"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "training_runtime differs"):
                    trainer.run(require_smoke=False)
            self.assertEqual(trainer.step, 0)
            self.assertEqual(trainer.cursor, 0)
            self.assertFalse((root / "run/latest.json").exists())
            for lane in trainer.lanes:
                self.assertEqual(lane.optimizer.state, {})
                for name, parameter in lane.drafter.named_parameters():
                    self.assertTrue(parameter.equal(before[(lane.name, name)]))


if __name__ == "__main__":
    unittest.main()

"""CPU-only receipt fixtures and explicit stub CUDA contexts; no GPU evidence."""

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

from w1a1_eagle.qat_readiness import (
    SCHEMA,
    optimization_requires_receipt,
    recipe_identity,
    validate_optimization_readiness,
)


@dataclass(frozen=True)
class FixtureConfig:
    device: str = "cuda:0"
    a1_computation: str = "single_forward"
    activation_quantization: str = "learned"
    fusion_correction: dict | None = None
    sign_lr: float = 0.001
    warmup_steps: int = 100
    max_steps: int = 10
    max_cuda_reserved_bytes: int = 9000
    min_cuda_free_bytes: int = 100
    optimization_readiness: dict | None = None
    curriculum: dict | None = None


# Synthetic values deliberately named as stubs. These objects exercise the
# schema; they are never a real receipt or a claim of GPU measurement.
CONTEXT = {
    "source_sha256": "a" * 64,
    "runtime_identity": {
        "device_type": "cuda",
        "python_version": "CPU_TEST_STUB",
        "torch_version": "CPU_TEST_STUB",
        "numpy_version": "CPU_TEST_STUB",
        "math_source_sha256": {"stub_math.py": "b" * 64},
        "cuda_math": {
            "float32_matmul_precision": "highest",
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    },
    "native_commit": "c" * 40,
    "backend": "cuda",
    "hardware": {
        "device_type": "cuda",
        "name": "CPU_TEST_CUDA_CONTEXT_STUB",
        "compute_capability": [12, 0],
        "total_memory_bytes": 10000,
    },
}


def synthetic_receipt(config):
    return {
        "schema": SCHEMA,
        "evidence_device_type": "cuda",
        "source_sha256": CONTEXT["source_sha256"],
        "training_runtime": copy.deepcopy(CONTEXT["runtime_identity"]),
        "recipe": recipe_identity(config),
        "native_commit": CONTEXT["native_commit"],
        "backend": "cuda",
        "hardware": copy.deepcopy(CONTEXT["hardware"]),
        "gates": {
            "full_model": {
                "passed": True,
                "model_scope": "full_model",
                "forward_calls": 5,
                "backward_calls": 5,
                "finite_gradients": True,
                "later_state_gradient_norm": 0.1,
                "later_key_gradient_norm": 0.2,
                "later_value_gradient_norm": 0.3,
            },
            "native_decisions": {
                "passed": True,
                "executed": True,
                "cases": 5,
                "changed_choice_count": 0,
                "material_choice_changes": 0,
                "max_changed_choice_margin": 0.0,
                "max_state_relative_rms": 0.01,
                "max_logit_relative_rms": 0.02,
            },
            "memory": {
                "passed": True,
                "peak_allocated_bytes": 7000,
                "peak_reserved_bytes": 8000,
                "min_free_bytes": 1000,
            },
            "learned_quantizers": {
                "passed": True,
                "executed": True,
                "exact_pack": True,
                "cases": 5,
                "artifact_sha256": "d" * 64,
            },
            "fusion_correction": {
                "passed": True,
                "raw_fc_executed": True,
                "zero_identity_passed": True,
                "nonzero_forward_passed": True,
                "raw_input_ancestry_passed": True,
                "cases": 5,
                "artifact_sha256": "e" * 64,
            },
        },
    }


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "CPU_FIXTURE_ONLY.json"
        self.config = FixtureConfig(fusion_correction={"enabled": True, "rank": 1})

    def write(self, receipt, config=None):
        data = json.dumps(receipt, allow_nan=True).encode()
        self.path.write_bytes(data)
        return replace(
            config or self.config,
            optimization_readiness={
                "path": str(self.path),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )

    def validate(self, receipt=None, config=None, **context):
        config = config or self.config
        return validate_optimization_readiness(
            self.write(receipt or synthetic_receipt(config), config), **(CONTEXT | context)
        )

    def test_all_curriculum_stages_need_independent_measured_gates(self):
        config = replace(
            self.config,
            curriculum={
                "stages": [
                    {"activation_bits": 8, "gpu_seconds": 10, "max_updates": 1},
                    {"activation_bits": 1, "gpu_seconds": 20, "max_updates": 2},
                ],
                "optimizer_transition": "fresh",
            },
        )
        receipt = synthetic_receipt(config)
        source = receipt["gates"]
        native = {
            name: copy.deepcopy(value)
            for name, value in source.items()
            if name not in ("full_model", "memory")
        }
        stages = [
            {
                "activation_bits": bits,
                "full_model": copy.deepcopy(source["full_model"]),
                "memory": copy.deepcopy(source["memory"]),
                "native_gates": copy.deepcopy(native),
            }
            for bits in (8, 1)
        ]
        source["curriculum_stages"] = {"passed": True, "activation_bits": [8, 1], "stages": stages}
        self.validate(receipt, config=config)
        for mutation in (
            lambda x: x["gates"].pop("curriculum_stages"),
            lambda x: x["gates"]["curriculum_stages"]["stages"].pop(),
            lambda x: x["gates"]["curriculum_stages"].update(activation_bits=[1, 8]),
            lambda x: x["gates"]["curriculum_stages"]["stages"][1]["native_gates"][
                "native_decisions"
            ].update(material_choice_changes=False),
            lambda x: x["gates"]["curriculum_stages"]["stages"][1]["native_gates"][
                "learned_quantizers"
            ].update(exact_pack=False),
        ):
            invalid = copy.deepcopy(receipt)
            mutation(invalid)
            with self.assertRaises(ValueError):
                self.validate(invalid, config=config)

    def test_valid_synthetic_schema_with_explicit_stub_context(self):
        receipt = self.validate()
        self.assertEqual(receipt["schema"], SCHEMA)
        self.assertNotIn("speedup", receipt)

    def test_import_requires_no_torch_or_device_discovery(self):
        import importlib
        import sys

        import w1a1_eagle.qat_readiness as module

        with mock.patch.dict(sys.modules, {"torch": None}):
            importlib.reload(module)
        self.assertNotIn("torch", module.__dict__)

    def test_cpu_bypasses_receipt_and_context_access(self):
        config = replace(self.config, device="cpu", optimization_readiness={"path": "absent"})
        result = validate_optimization_readiness(config, **{k: None for k in CONTEXT})
        self.assertIsNone(result)

    def test_default_versus_new_options_and_recipe_identity(self):
        self.assertFalse(optimization_requires_receipt({"device": "cuda"}))
        for field, value in (
            ("a1_computation", "single_forward"),
            ("activation_quantization", "learned"),
            ("binary_optimization", {"optimizer": "sgd"}),
            ("fusion_correction", {"enabled": True}),
            ("depth_loss_decay", 0.8),
            ("optimize_cache", True),
            ("optimize_head", True),
            ("persistent_sign_diagnostics", True),
            ("stages", [{"activation_bits": 1}]),
        ):
            self.assertTrue(optimization_requires_receipt({field: value}))
        self.assertFalse(optimization_requires_receipt({"fusion_correction": {"enabled": False}}))
        changed = replace(
            self.config,
            device="cuda:3",
            max_steps=300,
            max_cuda_reserved_bytes=10000,
            optimization_readiness={"path": "other"},
        )
        self.assertEqual(recipe_identity(self.config), recipe_identity(changed))
        self.assertNotEqual(
            recipe_identity(self.config), recipe_identity(replace(self.config, sign_lr=0.1))
        )
        self.assertEqual(recipe_identity({"future_option": (1, 2)}), {"future_option": [1, 2]})

    def test_stale_source_runtime_recipe_native_backend_hardware_rejected(self):
        for field in (
            "source_sha256",
            "training_runtime",
            "recipe",
            "native_commit",
            "backend",
            "hardware",
        ):
            receipt = synthetic_receipt(self.config)
            receipt[field] = "stale"
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(receipt)
        with self.assertRaises(ValueError):
            self.validate(
                synthetic_receipt(self.config), config=replace(self.config, warmup_steps=3)
            )
        with self.assertRaises(ValueError):
            self.validate(backend="cpu")
        context = copy.deepcopy(CONTEXT["runtime_identity"])
        context["device_type"] = "cpu"
        with self.assertRaises(ValueError):
            self.validate(runtime_identity=context)
        for field, value in (("cuda_math", {}), ("math_source_sha256", {})):
            context = copy.deepcopy(CONTEXT["runtime_identity"])
            context[field] = value
            with self.assertRaises(ValueError):
                self.validate(runtime_identity=context)

    def test_cpu_estimate_old_schema_and_explicit_fixture_cannot_grant_cuda(self):
        for field, value in (
            ("schema", "old_ready_v1"),
            ("evidence_device_type", "cpu"),
            ("fixture_only", True),
        ):
            receipt = synthetic_receipt(self.config)
            receipt[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(receipt)
        receipt = synthetic_receipt(self.config)
        receipt["gates"]["full_model"]["model_scope"] = "tiny_synthetic"
        with self.assertRaises(ValueError):
            self.validate(receipt)

    def test_missing_conditional_gates_and_fake_pass_flags_rejected(self):
        for gate in synthetic_receipt(self.config)["gates"]:
            receipt = synthetic_receipt(self.config)
            del receipt["gates"][gate]
            with self.subTest(gate=gate), self.assertRaises(ValueError):
                self.validate(receipt)
            for bad in (1, "true", None, False):
                receipt = synthetic_receipt(self.config)
                receipt["gates"][gate]["passed"] = bad
                with self.subTest(gate=gate, bad=bad), self.assertRaises(ValueError):
                    self.validate(receipt)
        config = replace(self.config, activation_quantization="fixed", fusion_correction=None)
        receipt = synthetic_receipt(config)
        del receipt["gates"]["learned_quantizers"]
        del receipt["gates"]["fusion_correction"]
        self.validate(receipt, config)

    def test_numeric_counts_gradients_and_native_limits_fail_closed(self):
        for gate, field in (
            ("full_model", "backward_calls"),
            ("full_model", "later_key_gradient_norm"),
            ("memory", "peak_reserved_bytes"),
            ("native_decisions", "cases"),
        ):
            for bad in (True, False, 0, -1, float("inf"), float("nan"), "5"):
                receipt = synthetic_receipt(self.config)
                receipt["gates"][gate][field] = bad
                with self.subTest(gate=gate, field=field, bad=bad), self.assertRaises(ValueError):
                    self.validate(receipt)
        for field, value in (
            ("material_choice_changes", 1),
            ("max_changed_choice_margin", 0.021),
            ("max_state_relative_rms", 0.101),
            ("max_logit_relative_rms", 0.101),
        ):
            receipt = synthetic_receipt(self.config)
            receipt["gates"]["native_decisions"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(receipt)

    def test_current_memory_caps_are_checked_without_changing_recipe(self):
        with self.assertRaises(ValueError):
            self.validate(config=replace(self.config, max_cuda_reserved_bytes=7999))
        with self.assertRaises(ValueError):
            self.validate(min_cuda_free_bytes=1001)
        receipt = synthetic_receipt(self.config)
        receipt["gates"]["memory"]["peak_allocated_bytes"] = 8001
        with self.assertRaises(ValueError):
            self.validate(receipt)
        self.validate(max_cuda_reserved_bytes=8000, min_cuda_free_bytes=1000)

    def test_missing_locator_bad_hash_duplicate_json_and_tampering_rejected(self):
        with self.assertRaises(ValueError):
            validate_optimization_readiness(self.config, **CONTEXT)
        config = self.write(synthetic_receipt(self.config))
        self.path.write_bytes(self.path.read_bytes() + b" ")
        with self.assertRaises(ValueError):
            validate_optimization_readiness(config, **CONTEXT)
        data = b'{"schema":"qat_optimization_readiness_v1","schema":"old"}'
        self.path.write_bytes(data)
        config = replace(
            self.config,
            optimization_readiness={
                "path": str(self.path),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )
        with self.assertRaisesRegex(ValueError, "invalid readiness"):
            validate_optimization_readiness(config, **CONTEXT)
        self.path.unlink()
        with self.assertRaisesRegex(ValueError, "cannot be read"):
            validate_optimization_readiness(config, **CONTEXT)


if __name__ == "__main__":
    unittest.main()

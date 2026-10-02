"""Independent CPU-only contracts for the measured-readiness math inventory."""

import copy
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from test_qat_readiness import CONTEXT, FixtureConfig, synthetic_receipt

from w1a1_eagle.continuous_runtime import MATH_FILES, training_runtime_identity
from w1a1_eagle.qat_curriculum_runner import EXTRA_MATH, curriculum_runtime
from w1a1_eagle.qat_readiness import validate_optimization_readiness


class IndependentMathInventoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "CPU_FIXTURE_ONLY.json"
        self.config = FixtureConfig()
        self.runtime = self._stub_cuda_runtime(training_runtime_identity("cpu"))

    @staticmethod
    def _stub_cuda_runtime(actual):
        """Bind actual source hashes while keeping all device evidence synthetic."""
        runtime = copy.deepcopy(actual)
        runtime["device_type"] = "cuda"
        runtime["cuda_math"] = {
            "float32_matmul_precision": "highest",
            "matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        }
        return runtime

    def _validate(self, receipt, config=None, runtime=None):
        config = config or self.config
        runtime = runtime or self.runtime
        data = json.dumps(receipt, sort_keys=True, allow_nan=False).encode()
        self.path.write_bytes(data)
        bound_config = replace(
            config,
            optimization_readiness={
                "path": str(self.path),
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )
        return validate_optimization_readiness(
            bound_config,
            source_sha256=CONTEXT["source_sha256"],
            runtime_identity=runtime,
            native_commit=CONTEXT["native_commit"],
            backend="cuda",
            hardware=CONTEXT["hardware"],
        )

    def _receipt(self, config=None, runtime=None):
        config = config or self.config
        runtime = runtime or self.runtime
        receipt = synthetic_receipt(config)
        receipt["training_runtime"] = copy.deepcopy(runtime)
        return receipt

    def test_complete_receipt_binds_the_authentic_runtime_inventory(self):
        self.assertEqual(set(self.runtime["math_source_sha256"]), set(MATH_FILES))
        self.assertEqual(
            set(self._validate(self._receipt())["training_runtime"]["math_source_sha256"]),
            set(MATH_FILES),
        )

    def test_mutually_omitted_declared_math_sources_reject(self):
        # Omit a key from both caller identity and receipt. This must not turn
        # equal incomplete maps into acceptable readiness evidence.
        for name in ("learned_activation.py", "recurrent_qat.py"):
            with self.subTest(name=name):
                runtime = copy.deepcopy(self.runtime)
                runtime["math_source_sha256"].pop(name)
                receipt = self._receipt(runtime=runtime)
                with self.assertRaisesRegex(ValueError, "math-source identity"):
                    self._validate(receipt, runtime=runtime)

    def test_unknown_curriculum_precision_stage_rejects(self):
        config = replace(
            self.config,
            curriculum={
                "stages": [
                    {"activation_bits": 8, "gpu_seconds": 10, "max_updates": 1},
                    {"activation_bits": 2, "gpu_seconds": 20, "max_updates": 2},
                ],
                "optimizer_transition": "fresh",
            },
        )
        runtime = self._stub_cuda_runtime(curriculum_runtime("cpu"))
        self.assertEqual(set(runtime["curriculum_math_sha256"]), set(EXTRA_MATH))
        receipt = self._receipt(config, runtime)
        gates = receipt["gates"]
        full = copy.deepcopy(gates["full_model"])
        native = {
            name: copy.deepcopy(value)
            for name, value in gates.items()
            if name not in ("full_model", "memory", "curriculum_stages")
        }
        stages = [
            {
                "activation_bits": bits,
                "full_model": copy.deepcopy(full),
                "memory": copy.deepcopy(gates["memory"]),
                "native_gates": copy.deepcopy(native),
            }
            for bits in (8, 2)
        ]
        gates["curriculum_stages"] = {
            "passed": True,
            "activation_bits": [8, 2],
            "stages": stages,
        }
        with self.assertRaisesRegex(ValueError, "precision stage"):
            self._validate(receipt, config=config, runtime=runtime)

    def test_changed_runtime_resume_rejects_before_parameter_or_optimizer_mutation(self):
        # Reuse the small CPU-only provider fixture used by the continuous QAT
        # tests; the source identity gate runs before loading the saved payload.
        from test_continuous_qat import config as continuous_config
        from test_continuous_qat import make as make_continuous

        import torch

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trained = make_continuous(root, continuous_config(max_steps=1))
            trained.run(require_smoke=False)
            resumed = make_continuous(root, continuous_config(max_steps=2))
            parameters = [p.detach().clone() for lane in resumed.lanes for p in lane.drafter.parameters()]
            optimizer = copy.deepcopy(
                [lane.optimizer.state_dict() for lane in resumed.lanes]
            )
            changed = copy.deepcopy(trained.runtime_identity)
            changed["math_source_sha256"].pop("recurrent_qat.py")
            with patch(
                "w1a1_eagle.continuous_qat.training_runtime_identity", return_value=changed
            ):
                with self.assertRaisesRegex(ValueError, "critical training math"):
                    resumed.resume()
            for before, lane in zip(parameters, (p for lane in resumed.lanes for p in lane.drafter.parameters())):
                torch.testing.assert_close(lane, before, rtol=0, atol=0)
            self.assertEqual(
                [lane.optimizer.state_dict() for lane in resumed.lanes], optimizer
            )


if __name__ == "__main__":
    unittest.main()

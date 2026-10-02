"""CPU schema probes using actual producer source inventories; no CUDA execution."""
import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_qat_readiness as fixture
from w1a1_eagle.continuous_runtime import MATH_FILES, training_runtime_identity
from w1a1_eagle.qat_curriculum_runner import EXTRA_MATH, curriculum_runtime
from w1a1_eagle.qat_readiness import _source_inventory


class RequiredMathInventoryTests(unittest.TestCase):
    def context(self, curriculum=False):
        context = copy.deepcopy(fixture.CONTEXT)
        runtime = curriculum_runtime("cpu") if curriculum else training_runtime_identity("cpu")
        context["runtime_identity"]["math_source_sha256"] = runtime["math_source_sha256"]
        if curriculum:
            context["runtime_identity"]["curriculum_math_sha256"] = runtime["curriculum_math_sha256"]
        return context

    def validate(self, context, config=None, mutate=None):
        case = fixture.ReadinessTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        config = config or case.config
        with patch.object(fixture, "CONTEXT", context):
            receipt = fixture.synthetic_receipt(config)
            if mutate:
                mutate(receipt)
            return case.validate(receipt, config)

    def test_literal_inventory_matches_actual_producer_registries(self):
        self.assertEqual(_source_inventory("continuous_runtime.py", "MATH_FILES"), MATH_FILES)
        self.assertEqual(_source_inventory("qat_curriculum_runner.py", "EXTRA_MATH"), EXTRA_MATH)

    def test_each_critical_math_source_cannot_be_mutually_omitted(self):
        complete = self.context()
        self.validate(complete)
        default = replace(fixture.FixtureConfig(), a1_computation="reference", activation_quantization="fixed")
        self.validate(complete, default)
        for name in MATH_FILES:
            with self.subTest(source=name):
                context = copy.deepcopy(complete)
                context["runtime_identity"]["math_source_sha256"].pop(name)
                with self.assertRaisesRegex(ValueError, "requires " + name.replace(".", r"\.")):
                    self.validate(context)

    def test_each_extra_curriculum_source_cannot_be_mutually_omitted(self):
        complete = self.context(curriculum=True)
        self.validate(complete)
        for name in EXTRA_MATH:
            with self.subTest(source=name):
                context = copy.deepcopy(complete)
                context["runtime_identity"]["curriculum_math_sha256"].pop(name)
                with self.assertRaisesRegex(ValueError, "curriculum math-source identity requires"):
                    self.validate(context)

    def test_wrong_receipt_hash_with_complete_current_inventory_is_rejected(self):
        def change(receipt):
            receipt["training_runtime"]["math_source_sha256"]["recurrent_qat.py"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "training_runtime differs"):
            self.validate(self.context(), mutate=change)


if __name__ == "__main__":
    unittest.main()

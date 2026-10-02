"""Independent CPU checks for activation helper identity closure."""

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import test_qat_curriculum_runner as curriculum_fixtures  # noqa: E402
import test_qat_readiness as readiness_fixtures  # noqa: E402

from w1a1_eagle.continuous_runtime import MATH_FILES, training_runtime_identity  # noqa: E402
from w1a1_eagle.qat_curriculum_runner import EXTRA_MATH  # noqa: E402

HELPER = "activation_reuse.py"


class ActivationIdentityClosureTests(unittest.TestCase):
    def test_helper_hash_is_bound_when_reuse_adapter_is_disabled(self):
        self.assertIn(HELPER, MATH_FILES)
        self.assertIn(HELPER, EXTRA_MATH)
        source_root = ROOT / "src" / "w1a1_eagle"
        identity = training_runtime_identity("cpu", source_root=source_root)
        actual = hashlib.sha256((source_root / HELPER).read_bytes()).hexdigest()
        self.assertEqual(identity["math_source_sha256"][HELPER], actual)

        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder)
            for name in MATH_FILES:
                (copied / name).write_bytes((source_root / name).read_bytes())
            first = training_runtime_identity("cpu", source_root=copied)
            (copied / HELPER).write_bytes(b"independent helper mutation")
            changed = training_runtime_identity("cpu", source_root=copied)
        self.assertNotEqual(
            first["math_source_sha256"][HELPER], changed["math_source_sha256"][HELPER]
        )

    def _context(self, sources):
        context = copy.deepcopy(readiness_fixtures.CONTEXT)
        context["runtime_identity"]["math_source_sha256"] = sources
        return context

    def _validate_fixture(self, context):
        fixture = readiness_fixtures.ReadinessTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with patch.object(readiness_fixtures, "CONTEXT", context):
            receipt = readiness_fixtures.synthetic_receipt(fixture.config)
            return fixture.validate(receipt)

    def test_complete_receipt_with_helper_source_is_accepted(self):
        context = self._context({"legacy_math.py": "b" * 64, HELPER: "f" * 64})
        validated = self._validate_fixture(context)
        self.assertEqual(validated["training_runtime"], context["runtime_identity"])

    def test_matching_partial_map_omitting_helper_is_rejected(self):
        context = self._context({"legacy_math.py": "b" * 64})
        with self.assertRaisesRegex(ValueError, "requires activation_reuse\\.py"):
            self._validate_fixture(context)

    def test_malformed_helper_digest_is_rejected(self):
        context = self._context({HELPER: "not-a-sha256"})
        with self.assertRaisesRegex(
            ValueError, "activation_reuse\\.py must be a full lowercase hexadecimal hash"
        ):
            self._validate_fixture(context)

    def test_changed_runtime_helper_hash_rejects_resume_before_state_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = curriculum_fixtures.make(root)
            original.run(require_smoke=False, max_new_updates=1)

            resumed = curriculum_fixtures.make(root)
            resumed.runtime = copy.deepcopy(resumed.runtime)
            resumed.runtime["math_source_sha256"][HELPER] = "f" * 64
            resumed.contract = copy.deepcopy(resumed.contract)
            resumed.contract["runtime"] = copy.deepcopy(resumed.runtime)

            updates_before = resumed.state.global_updates
            phase_before = resumed.state.phase_index
            model_before = {
                name: parameter.detach().clone()
                for name, parameter in resumed.drafter.named_parameters()
            }
            optimizer_before = copy.deepcopy(resumed.optimizer.state_dict())
            rng_before = resumed.rng

            with self.assertRaisesRegex(ValueError, "resume changes config/source/math/runtime"):
                resumed.resume()

            self.assertEqual(resumed.state.global_updates, updates_before)
            self.assertEqual(resumed.state.phase_index, phase_before)
            self.assertEqual(resumed.optimizer.state_dict(), optimizer_before)
            for name, parameter in resumed.drafter.named_parameters():
                self.assertTrue((parameter == model_before[name]).all().item(), name)
            self.assertIs(resumed.rng, rng_before)


if __name__ == "__main__":
    unittest.main()

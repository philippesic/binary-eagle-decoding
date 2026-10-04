import copy
import unittest

from precision_q4 import EXPECTED
from validate_native import check_q4_precision


class Q4AncestryTests(unittest.TestCase):
    def fixture(self):
        export = {"draft_sha256": "reference", "source_sha256": "checkpoint", "target_sha256": "frozen-target"}
        precision = {"schema": "dspark_precision_q4_ffn_v1", "passed": True, "source_export_bound": True,
                     "non_ffn_immutable": True, "candidate_sha256": "candidate", "source_gguf_sha256": "reference",
                     "source_export_sha256": "export-pin", "original_checkpoint_sha256": "checkpoint",
                     "target_sha256": "frozen-target", "selected": [{"name": name, "type": "Q4_0"} for name in sorted(EXPECTED)]}
        return export, precision

    def test_bound_reference_to_candidate(self):
        export, precision = self.fixture()
        check_q4_precision(export, precision, "candidate", "export-pin")

    def test_each_ancestry_pin_rejects_changes(self):
        for key in ("candidate_sha256", "source_gguf_sha256", "source_export_sha256",
                    "original_checkpoint_sha256", "target_sha256"):
            export, precision = self.fixture()
            precision[key] = "changed"
            with self.assertRaisesRegex(ValueError, "ancestry changed"):
                check_q4_precision(export, precision, "candidate", "export-pin")

    def test_non_ffn_change_or_incomplete_proof_rejects(self):
        for key in ("passed", "source_export_bound", "non_ffn_immutable"):
            export, precision = self.fixture()
            precision[key] = False
            with self.assertRaisesRegex(ValueError, "exact FFN"):
                check_q4_precision(export, precision, "candidate", "export-pin")

    def test_wrong_duplicate_or_extra_coverage_rejects(self):
        for mode in ("wrong_type", "duplicate", "extra"):
            export, precision = self.fixture()
            if mode == "wrong_type": precision["selected"][0]["type"] = "Q8_0"
            elif mode == "duplicate": precision["selected"][0] = copy.deepcopy(precision["selected"][1])
            else: precision["selected"].append({"name": "output.weight", "type": "Q4_0"})
            with self.assertRaisesRegex(ValueError, "exactly fifteen"):
                check_q4_precision(export, precision, "candidate", "export-pin")


if __name__ == "__main__":
    unittest.main()

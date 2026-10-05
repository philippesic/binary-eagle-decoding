import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "production_initializer", ROOT / "scripts/prepare_eagle_production_initializer.py"
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
sys.path.insert(0, str(ROOT / "src"))


def record(domain, number, **extra):
    return {
        "id": f"{domain}:{number}",
        "domain": domain,
        "group": f"group:{domain}:{number}",
        "topic": None,
        "content_sha256": f"{number:064x}",
        **extra,
    }


def native_round_fixture(kinds):
    from w1a1_eagle.recurrent_trace import LABEL_SOURCE, RoundAnchor

    anchors, rows = {}, {}
    for number, kind in enumerate(kinds):
        prefix = (0, 1, *([3] * number))
        key = ("train-a", number)
        anchors[key] = RoundAnchor("train-a", "train", number, prefix, 3)
        row = {
            "prompt_id": "train-a",
            "split": "train",
            "round_index": number,
            "depth": 0,
            "parent_position": len(prefix) - 1,
            "input_position": len(prefix),
            "label_position": len(prefix) + 1,
            "verifier_row": 0,
            "prefix_token_ids": [*prefix, 3],
            "input_token_id": 3,
            "alignment_valid": True,
            "is_bonus": False,
            "valid": True,
            "verifier_reached": True,
            "invalid_reason": None,
            "label_source": LABEL_SOURCE,
            "verifier_token_id": 4,
            "proposed_token_id": 6,
            "label_supported": True,
        }
        if kind == "terminal":
            row.update(
                valid=False,
                verifier_reached=False,
                invalid_reason="eos",
                verifier_token_id=None,
                proposed_token_id=None,
                label_supported=False,
            )
        elif kind == "unsupported":
            row.update(verifier_token_id=7, label_supported=False)
        elif kind == "corrupt":
            row.update(label_position=len(prefix) + 2)
        rows[key] = (row,)
    return SimpleNamespace(
        capture=SimpleNamespace(anchors=anchors, rows=rows),
        d2t_offsets=(2, 3, 3),
        target_vocab_size=8,
        draft_vocab_size=3,
        allowed_prompt_ids={"train-a"},
        split="train",
        max_depth=5,
    )


class ProductionInitializerTests(unittest.TestCase):
    def test_split_cannot_borrow_group_topic_or_content_aliases(self):
        records = []
        for ordinal, domain in enumerate(helper.DOMAINS):
            base = ordinal * 100 + 1
            first = record(domain, base, topic=f"topic:{domain}")
            records.extend(
                [
                    first,
                    record(domain, base + 1, group=first["group"]),
                    record(domain, base + 2, topic=first["topic"]),
                    record(domain, base + 3, content_sha256=first["content_sha256"]),
                    record(domain, base + 4),
                ]
            )
        chosen = helper.select_prompts(records, 1, 1)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(
            [(row["domain"], row["calibration_split"]) for row in chosen],
            [(domain, split) for domain in helper.DOMAINS for split in ("fit", "validation")],
        )
        self.assertTrue(
            all(row["id"].endswith((":1", ":5", ":101", ":105", ":201", ":205")) for row in chosen)
        )

    def test_missing_code_validation_fails_closed(self):
        records = [
            record(domain, ordinal * 10 + number)
            for ordinal, domain in enumerate(helper.DOMAINS)
            for number in range(1, 3 if domain != "code" else 2)
        ]
        with self.assertRaisesRegex(ValueError, "all three domains"):
            helper.select_prompts(records, 1, 1)

    def test_opaque_identity_and_positive_quotas_required(self):
        with self.assertRaisesRegex(ValueError, "identity absent"):
            helper.select_prompts([{"id": "prose:1", "domain": "prose"}], 1, 1)
        with self.assertRaisesRegex(ValueError, "quotas"):
            helper.select_prompts([], 0, 1)

    def test_terminal_longest_round_selects_longest_admitted_earlier_round(self):
        child = native_round_fixture(["supported", "supported", "terminal"])
        key, audit, skipped = helper.longest_eligible_round(child, "train-a")
        self.assertEqual(key, ("train-a", 1))
        self.assertEqual(audit.ce_mask, (True,))
        self.assertEqual(skipped, [2])

    def test_no_admitted_round_refuses_instead_of_borrowing_terminal_or_unmapped_label(self):
        child = native_round_fixture(["unsupported", "terminal"])
        with self.assertRaisesRegex(ValueError, "no admitted-label round: train-a"):
            helper.longest_eligible_round(child, "train-a")

    def test_corrupt_longer_round_is_not_silently_skipped(self):
        child = native_round_fixture(["supported", "corrupt"])
        with self.assertRaisesRegex(ValueError, "alignment mismatch"):
            helper.longest_eligible_round(child, "train-a")

    def test_scale_only_preserves_half_magnitude_even_negative_correlation(self):
        import numpy as np

        from w1a1_eagle.block_fusion import FusionFitConfig, fit_fusion, project, quantize

        x = np.array([[1, 2, -4], [2, 3, -5]], dtype=np.float32)
        weight = np.ones((1, 3), dtype=np.float32)
        codes, beta = quantize(x, 8)
        teacher = -project(codes, beta, weight, np.array([2], dtype=np.float32))
        result = fit_fusion(
            x,
            teacher,
            weight,
            FusionFitConfig(8, False, 0, reference_kind="eagle_fixed_reference_0.5"),
        )
        self.assertEqual(result["report"]["events"], [])
        np.testing.assert_array_equal(result["latent"], np.full_like(weight, 0.5))
        self.assertEqual(result["scale"][0], 0)
        self.assertEqual(
            result["latent_initialization"]["reference_kind"], "eagle_fixed_reference_0.5"
        )


if __name__ == "__main__":
    unittest.main()

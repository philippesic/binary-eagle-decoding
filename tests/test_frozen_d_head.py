"""CPU correctness fixtures for alignment, support accounting and one bounded fit."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_frozen_d_head_data import audit_rows, bucket_centers, prepare_parity_arrays, sha256, validate_d2t
from scripts.fit_frozen_d_head import head_logits, objective, optimize, parity_gate


def row(round_index=0, depth=0, token=2):
    parent = round_index + 2
    prefix = [0] * (parent + depth + 1) + [1]
    return {"prompt_id": "train-a", "split": "train", "round_index": round_index, "depth": depth,
            "parent_position": parent, "input_position": parent+depth+1, "label_position": parent+depth+2,
            "verifier_row": depth, "prefix_token_ids": prefix, "input_token_id": 1,
            "verifier_token_id": token, "is_bonus": False, "alignment_valid": True,
            "valid": True, "label_supported": token in (0, 2, 4),
            "label_source": "cloned_native_verifier_sampler_at_actual_proposal_prefix"}


class HeadDataTests(unittest.TestCase):
    def audit(self, rows):
        return audit_rows(rows, np.ones((len(rows), 4), dtype=np.float32), {"train-a"},
                          np.array([0, 2, 4], dtype=np.int64), 6)

    def test_denominators_keep_unsupported_and_stratify_per_depth(self):
        rows = [row(i) for i in range(100)] + [row(101, token=3)] + [row(102, depth=1)]
        arrays, report = self.audit(rows)
        self.assertEqual(report["valid_rows"], 102)
        self.assertEqual(report["unsupported_rows"], 1)
        self.assertEqual(report["selected_rows"], 18)
        self.assertEqual(arrays["labels"].tolist(), [1]*18)
        self.assertEqual(np.flatnonzero(arrays["selected_mask"])[:17].tolist(), bucket_centers(100))
        self.assertEqual(arrays["all_labels"][100], -1)

    def test_censored_rows_preserved_not_trained(self):
        censored = row(1)
        censored.update(valid=False, verifier_token_id=None, label_supported=False)
        arrays, report = self.audit([row(), censored])
        self.assertEqual(report["censored_rows"], 1)
        self.assertEqual(arrays["valid_mask"].tolist(), [True, False])

    def test_rejects_bad_alignment_and_wrong_prefix_and_bonus(self):
        for field, value in (("label_position", 999), ("verifier_row", 2), ("alignment_valid", False),
                             ("prefix_token_ids", [1]), ("is_bonus", True), ("prompt_id", "dev-1"),
                             ("label_supported", False), ("depth", 5)):
            bad = row()
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit([bad])

    def test_rejects_duplicate_state_and_nonfinite(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.audit([row(), row()])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            audit_rows([row()], np.full((1, 4), np.nan, dtype=np.float32), {"train-a"}, np.array([2]), 6)

    def test_mapping_must_be_unique_and_in_bounds(self):
        for values in ([2, 2], [-1, 2], [2, 6]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_d2t(np.array(values), 6)

    def test_parity_gather_uses_d2t_order_and_ignores_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            states = np.arange(12, dtype=np.float32).reshape(3, 4)
            states.tofile(base/"states.f32")
            native = np.array([[1., -np.inf, 2., -np.inf, 3., -np.inf],
                               [4., -np.inf, 5., -np.inf, 6., -np.inf]], dtype=np.float32)
            native.tofile(base/"logits.f32")
            np.save(base/"d2t.npy", np.array([4, 0, 2], dtype=np.int64))
            rows = [{"state_row": i, "depth": i, "full_logits_row": i if i < 2 else None,
                     "full_logits_dim": 6, "full_logits_boundary": "before_sampler"} for i in range(3)]
            (base/"rows.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            manifest = {"d2t_path": "d2t.npy", "target_vocab_size": 6,
                        "captures": [{"states_path": "states.f32", "rows_path": "rows.jsonl",
                                      "logits_path": "logits.f32", "state_dim": 4}]}
            (base/"capture.json").write_text(json.dumps(manifest))
            result = prepare_parity_arrays(base/"capture.json", base/"parity")
            self.assertEqual(result["rows"], 2)
            self.assertTrue(np.array_equal(np.load(base/"parity/parity-logits.npy"), [[3., 1., 2.], [6., 4., 5.]]))
            self.assertTrue(np.array_equal(np.load(base/"parity/parity-states.npy"), states[:2]))


class HeadFitTests(unittest.TestCase):
    def test_objective_is_true_label_ce_and_regularization_to_init(self):
        initial = torch.tensor([[.2, -.1], [-.3, .4]], dtype=torch.float32)
        weight = initial.clone().requires_grad_(True)
        x, labels = torch.tensor([[1., -1.]]), torch.tensor([1])
        loss, ce, penalty = objective(x, labels, weight, initial, "fp16")
        self.assertEqual(float(penalty.detach()), 0)
        self.assertAlmostEqual(float(ce.detach()), float(torch.nn.functional.cross_entropy(head_logits(x, weight, "fp16"), labels).detach()))
        loss.backward()
        self.assertTrue(torch.isfinite(weight.grad).all())
        self.assertGreater(float(weight.grad.abs().sum()), 0)

    def test_short_fit_preserves_zero_exports_finite_head_and_exposures(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = torch.tensor([[.2, -.1], [-.3, .4]])
            x, labels = torch.tensor([[1., -1.], [-1., 1.], [.5, .5]]), torch.tensor([1, 0, 1])
            report = optimize(x, labels, initial, "fp16", Path(directory), "cpu", max_steps=3)
            self.assertEqual(report["steps"], 3)
            self.assertEqual(report["exposures"], 9)
            self.assertEqual(report["completed_passes"], 3)
            self.assertEqual([c["step"] for c in report["checkpoints"]], [0, 3])
            zero = np.load(Path(directory) / "head-step-0000.npy")
            self.assertTrue(np.array_equal(zero, initial.half().numpy()))
            final = np.load(Path(directory) / "head-step-0003.npy")
            self.assertEqual(final.dtype, np.float16)
            self.assertFalse(np.array_equal(zero, final))
            self.assertEqual(len((Path(directory)/"steps.jsonl").read_text().splitlines()), 3)

    def test_no_training_after_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "nonfinite"):
            optimize(torch.tensor([[float('nan'), 0.]]), torch.tensor([0]), torch.ones(2, 2),
                     "fp16", Path(directory), "cpu", max_steps=1)

    def test_gate_binds_artifacts_and_rejects_bad_native_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            w = np.array([[1., 2.], [-1., .5]], dtype=np.float16)
            x = np.array([[1., -1.], [.25, .75]], dtype=np.float32)
            logits = head_logits(torch.from_numpy(x), torch.from_numpy(w).float(), "fp16").numpy()
            for name, values in (("head", w), ("states", x), ("logits", logits)):
                np.save(base/f"{name}.npy", values)
            gate = {"initial_head_sha256": sha256(base/"head.npy"), "dataset_sha256": "dataset",
                    "native_export_zero_passed": True, "native_zero_behavior_passed": True,
                    "input_cast": "fp16", "native_input_cast_evidence": "fixture",
                    "states_path": "states.npy", "states_sha256": sha256(base/"states.npy"),
                    "logits_path": "logits.npy", "logits_sha256": sha256(base/"logits.npy")}
            path = base/"gate.json"
            path.write_text(json.dumps(gate))
            result = parity_gate(path, "dataset", base/"head.npy", torch.from_numpy(w).float(), "cpu")
            self.assertTrue(result["passed"])
            self.assertEqual(result["max_absolute_error"], 0)
            with self.assertRaisesRegex(ValueError, "bound"):
                parity_gate(path, "different", base/"head.npy", torch.from_numpy(w).float(), "cpu")
            np.save(base/"logits.npy", logits + 10)
            gate["logits_sha256"] = sha256(base/"logits.npy")
            path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, "parity failed"):
                parity_gate(path, "dataset", base/"head.npy", torch.from_numpy(w).float(), "cpu")


if __name__ == "__main__":
    unittest.main()

"""Synthetic selection, scoped hook and row-metric checks; no GPU run."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import check_target_tap2_intervention as diagnostic  # noqa: E402
from audit_recurrent_binary_capture import sha256  # noqa: E402


class TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity() for _ in range(3)])
        self.last_layer_two_input = None

    def forward(self, input_ids, output_hidden_states, use_cache):
        assert output_hidden_states and not use_cache
        hidden = input_ids.to(torch.float16).unsqueeze(-1)
        states = [hidden]
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if index == 2:
                self.last_layer_two_input = hidden.clone()
            states.append(hidden)
        return SimpleNamespace(hidden_states=tuple(states))


class Tap2InterventionTests(unittest.TestCase):
    def test_scoped_hook_replaces_only_layer_two_input(self):
        model = TinyDecoder()
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        replacement = torch.full((1, 4, 1), 7, dtype=torch.float16)
        changed = diagnostic.forward_with_layer_input(model, tokens, replacement)
        self.assertTrue(torch.equal(changed[2], tokens.unsqueeze(-1).to(torch.float16)))
        self.assertTrue(torch.equal(changed[3], replacement))
        self.assertTrue(torch.equal(model.last_layer_two_input, replacement))
        baseline = model(input_ids=tokens, output_hidden_states=True, use_cache=False)
        self.assertTrue(torch.equal(baseline.hidden_states[3], tokens.unsqueeze(-1)))
        self.assertEqual(len(model.layers[2]._forward_pre_hooks), 0)

    def test_hook_rejects_wrong_shape_and_is_removed(self):
        model = TinyDecoder()
        with self.assertRaisesRegex(ValueError, "shape"):
            diagnostic.forward_with_layer_input(
                model,
                torch.tensor([[1, 2, 3, 4]]),
                torch.ones((1, 3, 1), dtype=torch.float16),
            )
        self.assertEqual(len(model.layers[2]._forward_pre_hooks), 0)

    def test_metrics_preserve_every_row_and_outlier(self):
        native = np.ones((5, 2), dtype=np.float32)
        reference = native.copy()
        reference[3] += 1
        measured = diagnostic.compare_rows(reference, native)
        self.assertEqual(measured["row_count"], 5)
        self.assertEqual([row["position"] for row in measured["rows"]], list(range(5)))
        self.assertEqual(measured["relative_l2_max_position"], 3)
        self.assertAlmostEqual(measured["outlier_relative_l2"], 1)
        self.assertEqual(measured["rows"][0]["rms"], 0)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            diagnostic.compare_rows(np.full((5, 2), np.nan), native)

    def test_frozen_selection_checks_hash_owner_and_complete_prefill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target.gguf"
            draft = root / "draft.gguf"
            target.write_bytes(b"target")
            draft.write_bytes(b"draft")
            capture = {
                "target_sha256": sha256(target),
                "draft_sha256": sha256(draft),
                "cell_manifest_path": "d_d/manifest.json",
            }
            request = {"id": diagnostic.PROMPT_ID, "task_id": "8"}
            tokens = [1, 2, 3, 4, 5]
            selected = {8: (tokens, [4, 5, 6, 7, 8])}
            with (
                mock.patch.object(
                    diagnostic,
                    "validate_sources",
                    return_value=(capture, {8: diagnostic.PROMPT_ID}, [request]),
                ) as validate,
                mock.patch.object(
                    diagnostic, "select_prefill_rows", return_value=selected
                ) as prefill,
                mock.patch.object(diagnostic, "validate_first_round_prefixes") as rounds,
            ):
                result = diagnostic.select_frozen_prompt(root, root / "train", target, draft)
                self.assertEqual(result[2:4], (tokens, selected[8][1]))
                validate.assert_called_once()
                prefill.assert_called_once()
                rounds.assert_called_once()
                target.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hashes"):
                    diagnostic.select_frozen_prompt(root, root / "train", target, draft)
                target.write_bytes(b"target")
                selected[8] = ([1, 2, 3], [4, 5, 6])
                with self.assertRaisesRegex(ValueError, "outlier position"):
                    diagnostic.select_frozen_prompt(root, root / "train", target, draft)


if __name__ == "__main__":
    unittest.main()

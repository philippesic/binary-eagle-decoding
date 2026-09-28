"""CPU-only joins of actual native preparer outputs into a capture bundle."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import test_recurrent_native_rows as native_row_tests  # noqa: E402
from audit_recurrent_binary_capture import audit_capture, sha256  # noqa: E402
from build_recurrent_capture_bundle import build_bundle  # noqa: E402
from prepare_recurrent_native_features import (  # noqa: E402
    RAW_BOUNDARY,
    RAW_SCHEMA,
    RAW_SOURCE,
    prepare_native_features,
)
from prepare_recurrent_native_rows import prepare, write_prepared  # noqa: E402


class RecurrentCaptureBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        # The existing native-row fixture supplies a real preparer-validated
        # two-depth trace with a bounded raw target-logit row.
        native = native_row_tests.NativeRowsTests(
            methodName="test_rejection_after_proposal_keeps_later_teacher_label_and_distinct_logits"
        )
        native.setUp()
        self.addCleanup(native.doCleanups)
        self.native = native
        self.prompt_hash = sha256(native.prompts)
        self.root = native.root
        self.feature_events = self.root / "target_features.jsonl"
        self.feature_values = self.root / "target_features.f32"
        self.rows_dir = self.root / "prepared_rows"
        self.features_dir = self.root / "prepared_features"
        self.output = self.root / "bundle"
        events = []
        prefixes = ([0], [0, 1], [0, 1, 3], [0, 1, 3, 6], [0, 1, 3, 6, 5])
        for index, prefix in enumerate(prefixes):
            phase = "prefill" if index < 2 else "speculative"
            local = index if phase == "prefill" else index - 2
            retained = index < 3
            decoded = {
                "schema": RAW_SCHEMA,
                "event": "decoded_row",
                "feature_row": index,
                "task_id": 9,
                "parent_task_id": -1,
                "slot_id": 0,
                "decode_ordinal": 0 if phase == "prefill" else 1,
                "batch_row_local": local,
                "batch_row_global": local,
                "position": len(prefix) - 1,
                "token_id": prefix[-1],
                "prefix_token_ids": prefix,
                "target_layer_ids": [2, 18, 33],
                "feature_dim": 7680,
                "boundary": RAW_BOUNDARY,
                "source": RAW_SOURCE,
                "phase": phase,
            }
            disposition = {
                "schema": RAW_SCHEMA,
                "event": "disposition",
                "feature_row": index,
                "task_id": 9,
                "slot_id": 0,
                "retained_input": retained,
                "reason": "accepted_prefix" if retained else "rejected_suffix",
            }
            if phase == "speculative":
                decoded.update(round_index=0, spec_input_row=local)
                disposition.update(round_index=0, accepted_drafts=0)
            events.extend((decoded, disposition))
        self.feature_events.write_text("".join(json.dumps(row) + "\n" for row in events))
        np.ones((len(prefixes), 7680), dtype="<f4").tofile(self.feature_values)
        self._make_prepared()

    def _make_prepared(self) -> None:
        # Simulate a completed cell recording all raw source file hashes.
        cell = json.loads(self.native.cell.read_text())
        cell["files"].update(
            {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in (self.native.logits, self.feature_events, self.feature_values)
            }
        )
        self.native.cell.write_text(json.dumps(cell))
        result = prepare(
            heads_path=self.native.heads,
            rounds_path=self.native.rounds,
            task_map_path=self.native.task_map,
            cell_manifest_path=self.native.cell,
            prompts_path=self.native.prompts,
            absolute_map_path=self.native.absolute_map,
            target_vocab_size=8,
            target_logits_path=self.native.logits,
            offset_path=self.native.offsets,
            t2d_path=self.native.t2d,
            expected_prompt_hash=self.prompt_hash,
            expected_prompt_count=2,
        )
        write_prepared(self.rows_dir, result)
        prepare_native_features(
            self.feature_events,
            self.feature_values,
            self.rows_dir / "anchors.jsonl",
            self.native.task_map,
            self.native.prompts,
            self.features_dir,
            cell_manifest_path=self.native.cell,
            expected_prompt_hash=self.prompt_hash,
            expected_prompt_count=2,
            target_vocab_size=8,
        )

    def build(self, *, continuity_report=None):
        return build_bundle(
            self.rows_dir,
            self.features_dir,
            self.native.logits,
            self.native.cell,
            self.native.prompts,
            self.output,
            continuity_report=continuity_report,
            expected_prompt_hash=self.prompt_hash,
            expected_prompt_count=2,
        )

    def test_builds_self_contained_audited_preparation_only_bundle(self):
        result = self.build()
        manifest = json.loads(result["manifest"].read_text())
        self.assertFalse(manifest["training_eligible"])
        self.assertEqual(manifest["readiness"], "preparation_only")
        self.assertIn("cross_round_acceptance_ancestry", manifest["unverified_gates"])
        self.assertEqual(result["audit"]["raw_target_logit_rows"], 1)
        self.assertEqual(result["audit"]["feature_ledger"]["rows"], 2)
        self.assertEqual(
            result["audit"],
            audit_capture(
                self.output / "manifest.json",
                self.output / "train_prompts.jsonl",
                self.prompt_hash,
                expected_prompt_count=2,
            ),
        )
        # The bundle remains independently auditable after raw sources vanish.
        self.native.logits.unlink()
        self.feature_events.unlink()
        self.feature_values.unlink()
        self.assertEqual(
            audit_capture(
                self.output / "manifest.json",
                self.output / "train_prompts.jsonl",
                self.prompt_hash,
                expected_prompt_count=2,
            )["counts"]["supported"],
            1,
        )

    def test_rejects_cross_source_anchors_task_map_and_cell(self):
        report_path = self.features_dir / "report.json"
        original = json.loads(report_path.read_text())
        for key, bad, expected in (
            ("anchors_sha256", "0" * 64, "anchors"),
            ("task_map_sha256", "0" * 64, "task map"),
            ("cell_manifest_sha256", "0" * 64, "cell manifest"),
        ):
            with self.subTest(key=key):
                report = json.loads(json.dumps(original))
                report["sources"][key] = bad
                report_path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, expected):
                    self.build()
                self.assertFalse(self.output.exists())
        report_path.write_text(json.dumps(original))
        cell = json.loads(self.native.cell.read_text())
        cell["requests"][0]["id"] = "train-b"
        self.native.cell.write_text(json.dumps(cell))
        with self.assertRaisesRegex(ValueError, "ownership"):
            self.build()

    def test_rejects_unverified_ownership_and_prompt_hash(self):
        report_path = self.rows_dir / "preparation.json"
        original = json.loads(report_path.read_text())
        report = json.loads(json.dumps(original))
        report["request_prompt_ownership"] = "unverified_plain_task_map_only"
        report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "ownership is unverified"):
            self.build()
        report_path.write_text(json.dumps(original))
        self.native.prompts.write_text(self.native.prompts.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "frozen train prompts"):
            self.build()

    def test_rejects_missing_logits_features_and_truncated_payload(self):
        self.native.logits.unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.build()
        np.zeros(8, dtype="<f4").tofile(self.native.logits)
        (self.features_dir / "feature_rows.jsonl").unlink()
        with self.assertRaisesRegex(ValueError, "missing feature_rows"):
            self.build()
        # Restore from the prepared payload and then truncate numeric content
        # while leaving its report unchanged.
        self.features_dir.mkdir(exist_ok=True)
        (self.features_dir / "feature_rows.jsonl").write_text(
            json.dumps(
                {
                    "feature_row": 0,
                    "prompt_id": "train-a",
                    "position": 0,
                    "prefix_token_ids": [0],
                    "tap_ids": [2, 18, 33],
                    "boundary": "native_target_block_inputs_concat_before_draft_fc",
                    "source": "native_target_features_on_accepted_prefix",
                    "accepted_prefix": True,
                }
            )
            + "\n"
        )
        with self.assertRaisesRegex(ValueError, "feature_rows SHA256"):
            self.build()
        report = json.loads((self.features_dir / "report.json").read_text())
        report["feature_rows_sha256"] = sha256(self.features_dir / "feature_rows.jsonl")
        (self.features_dir / "report.json").write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "raw target features must be F32"):
            self.build()

    def test_rejects_unverified_feature_ownership_and_vocab_mutation(self):
        feature_report_path = self.features_dir / "report.json"
        report = json.loads(feature_report_path.read_text())
        report["task_prompt_ownership"] = "plain_task_map_unverified_without_request_manifest"
        feature_report_path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "ownership is unverified"):
            self.build()
        report["task_prompt_ownership"] = "cell_manifest_request_and_file_hashes_verified"
        feature_report_path.write_text(json.dumps(report))
        row_report_path = self.rows_dir / "preparation.json"
        row_report = json.loads(row_report_path.read_text())
        row_report["draft_vocab_size"] = 4
        row_report_path.write_text(json.dumps(row_report))
        with self.assertRaisesRegex(ValueError, "offset|draft vocabulary"):
            self.build()

    def test_rejects_truncated_target_logits_even_with_resigned_reports(self):
        # This exercises the final structural audit after all source hashes
        # have been updated to agree on the corrupted payload.
        np.zeros(7, dtype="<f4").tofile(self.native.logits)
        cell = json.loads(self.native.cell.read_text())
        cell["files"][self.native.logits.name] = {
            "bytes": self.native.logits.stat().st_size,
            "sha256": sha256(self.native.logits),
        }
        self.native.cell.write_text(json.dumps(cell))
        rows_report_path = self.rows_dir / "preparation.json"
        rows_report = json.loads(rows_report_path.read_text())
        rows_report["source_sha256"]["raw_target_logits"] = sha256(self.native.logits)
        rows_report["source_sha256"]["cell_manifest"] = sha256(self.native.cell)
        rows_report_path.write_text(json.dumps(rows_report))
        feature_report_path = self.features_dir / "report.json"
        feature_report = json.loads(feature_report_path.read_text())
        feature_report["sources"]["cell_manifest_sha256"] = sha256(self.native.cell)
        feature_report_path.write_text(json.dumps(feature_report))
        with self.assertRaisesRegex(ValueError, "target vocabulary"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_optional_continuity_report_must_join_same_raw_sources(self):
        row_report = json.loads((self.rows_dir / "preparation.json").read_text())
        feature_report = json.loads((self.features_dir / "report.json").read_text())
        report = {
            "schema": "recurrent_internal_continuity_v1",
            "status": "internal_accepted_prefix_continuity_verified",
            "execution_device": "cpu",
            "rounds": row_report["rounds"],
            "source_sha256": {
                "rounds": row_report["source_sha256"]["rounds"],
                "feature_events": feature_report["sources"]["metadata_sha256"],
                "task_map": row_report["source_sha256"]["task_map"],
            },
            "initial_target_sample": "unverified",
            "terminal_emission_and_stop": "unverified",
            "request_completeness": "unverified",
            "numeric_feature_values": "unverified",
        }
        path = self.root / "continuity.json"
        path.write_text(json.dumps(report))
        self.build(continuity_report=path)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertIn("internal_continuity", manifest)
        self.assertFalse(manifest["training_eligible"])
        self.assertIn(
            "initial_sample_terminal_emission_and_request_completeness",
            manifest["unverified_gates"],
        )
        self.output.rename(self.root / "accepted_bundle")
        report["source_sha256"]["feature_events"] = "0" * 64
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "continuity feature events"):
            self.build(continuity_report=path)


if __name__ == "__main__":
    unittest.main()

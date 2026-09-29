"""V2 hard-CE ancestry/continuity gates, using synthetic CPU native metadata."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import test_recurrent_native_rows as native_fixture  # noqa: E402
from audit_recurrent_binary_capture import sha256  # noqa: E402
from audit_recurrent_capture_v2 import audit_capture_v2, load_audited_capture_v2  # noqa: E402
from build_recurrent_capture_bundle import build_bundle  # noqa: E402
from build_recurrent_capture_v2 import build_capture_v2  # noqa: E402
from convert_recurrent_capture_v2 import convert_capture_v2  # noqa: E402
from prepare_recurrent_native_features import (  # noqa: E402
    RAW_BOUNDARY,
    RAW_SCHEMA,
    RAW_SOURCE,
    prepare_native_features,
)
from prepare_recurrent_native_rows import prepare, write_prepared  # noqa: E402
from test_recurrent_native_rows import jsonl  # noqa: E402


class CaptureV2Tests(unittest.TestCase):
    def setUp(self):
        native = native_fixture.NativeRowsTests()
        native.setUp()
        self.addCleanup(native.doCleanups)
        self.native = native
        self.root = native.root
        self.capture = self.root / "native"
        self.cell = self.capture / "d_d"
        self.cell.mkdir(parents=True)
        jsonl(native.prompts, [{"id": "train-a"}])
        native.save()
        for path, name in (
            (native.heads, "heads.jsonl"),
            (native.rounds, "forced-rounds.jsonl"),
            (native.task_map, "task_prompt_ids.json"),
        ):
            shutil.copyfile(
                path, (self.capture if name == "task_prompt_ids.json" else self.cell) / name
            )
        self.prompt_hash = sha256(native.prompts)
        self.events = self.cell / "heads.target_features.jsonl"
        self.values = self.root / "feature_values.f32"
        events = []
        # A rejected first proposal has an unsupported later teacher label.
        for i, prefix in enumerate(([0], [0, 1], [0, 1, 3], [0, 1, 3, 6], [0, 1, 3, 6, 5])):
            phase, local = ("prefill", i) if i < 2 else ("speculative", i - 2)
            decoded = {
                "schema": RAW_SCHEMA,
                "event": "decoded_row",
                "feature_row": i,
                "task_id": 9,
                "parent_task_id": -1,
                "slot_id": 0,
                "decode_ordinal": 0 if i < 2 else 1,
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
            disp = {
                "schema": RAW_SCHEMA,
                "event": "disposition",
                "feature_row": i,
                "task_id": 9,
                "slot_id": 0,
                "retained_input": i < 3,
                "reason": "accepted_prefix" if i < 3 else "rejected_suffix",
            }
            if i >= 2:
                decoded.update(round_index=0, spec_input_row=local)
                disp.update(round_index=0, accepted_drafts=0)
            events.extend((decoded, disp))
        jsonl(self.events, events)
        np.ones((5, 7680), dtype="<f4").tofile(self.values)
        trace = self.cell / "rounds.jsonl"
        jsonl(
            trace,
            [
                {
                    "schema": "w1ax_eagle_round_v1",
                    "task_id": 9,
                    "round_index": 0,
                    "status": "complete",
                    "replay": False,
                    "n_proposed": 2,
                    "n_accepted": 0,
                    "n_emitted": 1,
                    "proposed_token_ids": [6, 5],
                    "emitted_token_ids": [4],
                }
            ],
        )
        reqdir = self.cell / "request-000"
        reqdir.mkdir()
        (reqdir / "prompt.json").write_text(json.dumps({"id": "train-a"}))
        (reqdir / "request.json").write_text(json.dumps({"max_tokens": 2}))
        (reqdir / "response.json").write_text(
            json.dumps(
                {
                    "__verbose": {
                        "tokens": [3, 4],
                        "tokens_predicted": 2,
                        "tokens_evaluated": 2,
                        "truncated": False,
                    },
                    "usage": {"completion_tokens": 2, "prompt_tokens": 2},
                    "choices": [{"finish_reason": "length"}],
                }
            )
        )
        cell = json.loads(native.cell.read_text())
        cell.update(target_sha256="a" * 64, draft_sha256="b" * 64, prompt_count=1)
        cell["files"] = {
            p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in (
                self.cell / "heads.jsonl",
                self.cell / "forced-rounds.jsonl",
                self.events,
                native.logits,
                self.values,
            )
        }
        cell["requests"][0].update(
            generated_token_ids=[3, 4],
            **{
                f"{k}_sha256": sha256(reqdir / f"{k}.json")
                for k in ("prompt", "request", "response")
            },
        )
        self.cell_manifest = self.cell / "manifest.json"
        self.cell_manifest.write_text(json.dumps(cell))
        (self.capture / "capture-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "recurrent_binary_native_capture_v1",
                    "complete": True,
                    "train_prompts_sha256": self.prompt_hash,
                    "train_prompt_count": 1,
                    "cell_manifest_sha256": sha256(self.cell_manifest),
                    "task_prompt_ids_sha256": sha256(self.capture / "task_prompt_ids.json"),
                    "target_vocab_size": 8,
                    "target_sha256": "a" * 64,
                    "draft_sha256": "b" * 64,
                    "d2t_sha256": sha256(native.absolute_map),
                    "d2t_raw_sha256": hashlib.sha256(
                        np.asarray(np.load(native.absolute_map), dtype="<i8").tobytes()
                    ).hexdigest(),
                    "requests": cell["requests"],
                }
            )
        )
        rows_dir = self.rows_dir = self.root / "prepared_rows"
        features_dir = self.features_dir = self.root / "prepared_features"
        result = prepare(
            heads_path=self.cell / "heads.jsonl",
            rounds_path=self.cell / "forced-rounds.jsonl",
            task_map_path=self.capture / "task_prompt_ids.json",
            prompts_path=native.prompts,
            absolute_map_path=native.absolute_map,
            target_vocab_size=8,
            cell_manifest_path=self.cell_manifest,
            target_logits_path=native.logits,
            expected_prompt_hash=self.prompt_hash,
            expected_prompt_count=1,
        )
        write_prepared(rows_dir, result)
        prepare_native_features(
            self.events,
            self.values,
            rows_dir / "anchors.jsonl",
            self.capture / "task_prompt_ids.json",
            native.prompts,
            features_dir,
            cell_manifest_path=self.cell_manifest,
            expected_prompt_hash=self.prompt_hash,
            expected_prompt_count=1,
            target_vocab_size=8,
        )
        self.v1 = self.root / "v1"
        build_bundle(
            rows_dir,
            features_dir,
            native.logits,
            self.cell_manifest,
            native.prompts,
            self.v1,
            expected_prompt_hash=self.prompt_hash,
            expected_prompt_count=1,
            expected_target_hash="a" * 64,
            expected_draft_hash="b" * 64,
            expected_map_raw_hash=result[-1]["absolute_map_raw_sha256"],
        )
        self.map_raw_hash = result[-1]["absolute_map_raw_sha256"]
        self.v2 = self.root / "v2"

    def direct_build(self, **kwargs):
        return build_capture_v2(
            self.rows_dir,
            self.features_dir,
            self.native.logits,
            self.capture,
            self.native.prompts,
            self.v2,
            expected_prompt_sha256=self.prompt_hash,
            expected_prompt_count=1,
            expected_target_sha256="a" * 64,
            expected_draft_sha256="b" * 64,
            expected_map_raw_sha256=self.map_raw_hash,
            **kwargs,
        )

    def test_legacy_builder_default_is_an_independent_raw_copy(self):
        copied = self.v1 / "target_logits.f32"
        self.assertEqual(sha256(copied), sha256(self.native.logits))
        self.assertNotEqual(copied.stat().st_ino, self.native.logits.stat().st_ino)
        self.assertNotIn(
            "raw_target_logits_storage", json.loads((self.v1 / "manifest.json").read_text())
        )

    def test_direct_v2_build_never_copies_raw_and_preserves_inode(self):
        # Remove the existing fixture's v1 copy: direct v2 has no such prerequisite.
        shutil.rmtree(self.v1)
        initial = self.native.logits.stat()
        raw_hash = sha256(self.native.logits)
        original_copy = shutil.copyfile
        seen_links = []
        original_link = os.link

        def checked_copy(source, dest, **kw):
            if Path(source).suffix == ".f32" or Path(dest).suffix == ".f32":
                self.fail("direct v2 attempted to copy raw full-vocabulary bytes")
            return original_copy(source, dest, **kw)

        def checked_link(source, dest):
            original_link(source, dest)
            self.assertEqual(Path(source).stat().st_ino, Path(dest).stat().st_ino)
            seen_links.append((source, dest))

        with (
            patch("build_recurrent_capture_bundle.shutil.copyfile", side_effect=checked_copy),
            patch("build_recurrent_capture_bundle.os.link", side_effect=checked_link),
        ):
            result = self.direct_build()
        self.assertEqual(len(seen_links), 1)
        self.assertEqual(sha256(self.native.logits), raw_hash)
        final = self.native.logits.stat()
        self.assertEqual(
            (initial.st_dev, initial.st_ino, initial.st_size, initial.st_nlink),
            (final.st_dev, final.st_ino, final.st_size, final.st_nlink),
        )
        self.assertEqual(list(self.root.glob(".v2-source-audit-*")), [])
        self.assertEqual(self.audit(), result["audit"])

    def test_direct_sharded_v2_preserves_parent_freeze_and_caps(self):
        prompts = self.root / "train_prompts.jsonl"
        shutil.copyfile(self.native.prompts, prompts)
        self.native.prompts = prompts
        shard = self.root / "shard.json"
        shard.write_text(
            json.dumps(
                {
                    "schema": "w1ax_capture_shard_v1",
                    "prompts_path": prompts.name,
                    "prompts_sha256": self.prompt_hash,
                    "prompt_count": 1,
                    "prompt_ids": ["train-a"],
                    "parent": {
                        "split": "train_small",
                        "manifest_sha256": "a" * 64,
                        "prompts_sha256": "b" * 64,
                        "index_sha256": "c" * 64,
                        "count": 2000,
                    },
                    "target_vocab_size": 8,
                    "bytes_per_raw_logit_row": 32,
                    "caps": {
                        "max_prompts": 1,
                        "max_verifier_logit_rows": 1,
                        "max_raw_logit_bytes": 32,
                    },
                }
            )
        )
        cell = json.loads(self.cell_manifest.read_text())
        cell.update(shard_manifest_sha256=sha256(shard), ordered_prompt_ids=["train-a"])
        self.cell_manifest.write_text(json.dumps(cell))
        rp = self.rows_dir / "preparation.json"
        record = json.loads(rp.read_text())
        record["source_sha256"]["cell_manifest"] = sha256(self.cell_manifest)
        rp.write_text(json.dumps(record))
        fp = self.features_dir / "report.json"
        record = json.loads(fp.read_text())
        record["sources"]["cell_manifest_sha256"] = sha256(self.cell_manifest)
        fp.write_text(json.dumps(record))
        cp = self.capture / "capture-manifest.json"
        record = json.loads(cp.read_text())
        record.update(
            cell_manifest_sha256=sha256(self.cell_manifest),
            requests=cell["requests"],
            shard_manifest_sha256=sha256(shard),
        )
        cp.write_text(json.dumps(record))
        result = self.direct_build(shard_manifest_path=shard)
        m = json.loads((self.v2 / "manifest.json").read_text())
        self.assertEqual(m["shard_manifest"]["sha256"], sha256(shard))
        self.assertEqual(result["audit"]["source_sha256"]["shard_manifest"], sha256(shard))
        self.assertEqual(self.audit(), result["audit"])

    def test_direct_hardlink_failure_has_no_raw_copy_fallback(self):
        before = sha256(self.native.logits)
        with patch(
            "build_recurrent_capture_bundle.os.link",
            side_effect=OSError(errno.EXDEV, "cross-device"),
        ):
            with self.assertRaises(OSError):
                self.direct_build()
        self.assertFalse(self.v2.exists())
        self.assertEqual(sha256(self.native.logits), before)
        self.assertEqual(list(self.root.glob(".v2-source-audit-*")), [])

    def convert(self):
        return convert_capture_v2(
            self.v1 / "manifest.json",
            self.capture,
            self.v2,
            expected_prompt_sha256=self.prompt_hash,
            expected_prompt_count=1,
        )

    def audit(self):
        return audit_capture_v2(
            self.v2 / "manifest.json",
            expected_prompt_sha256=self.prompt_hash,
            expected_prompt_count=1,
        )

    def mutate(self, field, fn):
        path = self.v2 / "manifest.json"
        m = json.loads(path.read_text())
        data_path = self.v2 / m[field]["path"]
        data = json.loads(data_path.read_text())
        fn(data)
        data_path.write_text(json.dumps(data))
        m[field]["sha256"] = sha256(data_path)
        path.write_text(json.dumps(m))

    def test_label_only_conversion_reaudits_without_raw_or_native_sources(self):
        before = sha256(self.v1 / "target_logits.f32")
        result = self.convert()
        self.assertEqual(result["omitted_raw_bytes"], 32)
        self.assertEqual(sha256(self.v1 / "target_logits.f32"), before)
        self.assertFalse((self.v2 / "target_logits.f32").exists())
        self.assertEqual(result["audit"]["counts"]["unsupported"], 1)
        self.assertEqual(result["audit"]["counts"]["verifier_reached"], 1)
        self.assertEqual(result["audit"]["response_tokens"], 2)
        self.assertEqual(
            result["audit"]["raw_probability_recomputation"], "unavailable_label_only_storage"
        )
        # Test independence after external inputs disappear. Production preserves them.
        shutil.rmtree(self.v1)
        shutil.rmtree(self.capture)
        self.native.logits.unlink()
        self.assertEqual(self.audit(), result["audit"])
        loaded = load_audited_capture_v2(
            self.v2 / "manifest.json",
            expected_prompt_sha256=self.prompt_hash,
            expected_prompt_count=1,
        )
        round_input = loaded.round_inputs("train-a", 0)
        self.assertEqual(round_input.prefix_token_ids, (0, 1, 3))
        self.assertEqual(round_input.raw_target_features.shape, (2, 7680))
        self.assertEqual(round_input.rows[1]["verifier_token_id"], 7)
        self.assertFalse(round_input.rows[1]["label_supported"])

    def test_promotion_and_raw_retirement_fail_closed(self):
        self.convert()
        path = self.v2 / "manifest.json"
        original = json.loads(path.read_text())
        for key, value in (
            ("training_eligible", True),
            ("raw_retirement_allowed", True),
            ("readiness", "ready"),
            ("target_logits", {}),
        ):
            m = dict(original)
            m[key] = value
            path.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "storage/eligibility"):
                self.audit()
        path.write_text(json.dumps(original))
        m = json.loads(path.read_text())
        m["storage_policy"]["objective"] = "kl"
        path.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "storage/eligibility"):
            self.audit()

    def test_changed_label_source_and_prefix_fail_with_updated_outer_hashes(self):
        self.convert()
        for field, value in (
            ("verifier_token_id", 5),
            ("prefix_token_ids", [0, 1, 4]),
            ("label_source", "argmax"),
            ("label_supported", False),
        ):
            mpath = self.v2 / "manifest.json"
            m = json.loads(mpath.read_text())
            p = self.v2 / m["rows"]["path"]
            original = p.read_bytes()
            rows = [json.loads(line) for line in p.read_text().splitlines()]
            rows[0][field] = value
            jsonl(p, rows)
            m["rows"]["sha256"] = sha256(p)
            mpath.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, "changed original rows"):
                self.audit()
            p.write_bytes(original)
            m["rows"]["sha256"] = sha256(p)
            mpath.write_text(json.dumps(m))

    def test_wrong_response_and_missing_request_fail(self):
        self.convert()
        mpath = self.v2 / "manifest.json"
        m = json.loads(mpath.read_text())
        p = self.v2 / m["requests"][0]["response"]["path"]
        p.unlink()
        with self.assertRaisesRegex(ValueError, "response file missing"):
            self.audit()

    def test_source_mismatch_and_failed_conversion_leave_no_output(self):
        heads = self.cell / "heads.jsonl"
        heads.write_text(heads.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "ancestry differs"):
            self.convert()
        self.assertFalse(self.v2.exists())
        self.assertEqual(list(self.root.glob(".v2-*")), [])
        self.assertTrue((self.v1 / "target_logits.f32").is_file())

    def test_trace_corruption_rejected_by_emission_join_after_hash_update(self):
        self.convert()
        mpath = self.v2 / "manifest.json"
        m = json.loads(mpath.read_text())
        p = self.v2 / m["native_round_trace"]["path"]
        row = json.loads(p.read_text())
        row["emitted_token_ids"] = [5]
        p.write_text(json.dumps(row) + "\n")
        m["native_round_trace"]["sha256"] = sha256(p)
        mpath.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "round trace disagrees"):
            self.audit()

    def test_omitted_raw_counts_and_source_capture_identity_are_checked(self):
        self.convert()
        mpath = self.v2 / "manifest.json"
        m = json.loads(mpath.read_text())
        original = json.loads(json.dumps(m))
        m["omitted_raw_logits"]["rows"] = 2
        mpath.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "omitted raw-logit"):
            self.audit()
        mpath.write_text(json.dumps(original))
        self.mutate("source_capture", lambda value: value.update(cell_manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "source identity"):
            self.audit()

    def test_unclaimed_raw_copy_rejected(self):
        self.convert()
        shutil.copyfile(self.v1 / "target_logits.f32", self.v2 / "renamed_raw_payload.dat")
        with self.assertRaisesRegex(ValueError, "unclaimed files"):
            self.audit()

    def test_output_immutable_and_path_escape_rejected(self):
        self.convert()
        with self.assertRaisesRegex(ValueError, "output must be new"):
            self.convert()
        mpath = self.v2 / "manifest.json"
        m = json.loads(mpath.read_text())
        m["native_heads"]["path"] = "../heads.jsonl"
        mpath.write_text(json.dumps(m))
        with self.assertRaisesRegex(ValueError, "basenames"):
            self.audit()


if __name__ == "__main__":
    unittest.main()

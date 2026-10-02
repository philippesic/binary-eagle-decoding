"""Authenticated historical full-pass adoption; tiny CPU fixtures only."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_SHA = "82f0185ab9ac38bc622749d2ca5e5c5a297a72494dcbf3d16d2b08df249cdade"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import test_retained_native_capture_import as fixtures  # noqa: E402
import w1ax_capture_provider as provider  # noqa: E402
import w1ax_continuous_stages as stages  # noqa: E402


class HistoricalNativeAuditAdoptionTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.RetainedNativeCaptureImportTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.root = f.root
        self.checkout = f.old.parent.parent
        stages._HISTORICAL_SOURCE_PROOFS.clear()
        stages._HISTORICAL_PASS_PROOFS.clear()
        supervisor = {
            "pid": 676,
            "pgid": 676,
            "supervisor_pid": 674,
            "started_at_utc": "2026-10-02T05:28:08.666837+00:00",
        }
        sup_path = self.checkout / "runs/luna-supervisor-a8-a1-native-order-20261002-08/state.json"
        operation = {
            "preflight": {
                "decoded_query": {
                    "all_checks_pass": True,
                    "identity": {
                        "git_head": stages.HISTORICAL_PRODUCER_COMMIT,
                        "git_diff_name_only": ["scripts/train_continuous_w1ax.py"],
                        "git_status_tracked": [" M scripts/train_continuous_w1ax.py"],
                        "launcher_sha256": LAUNCHER_SHA,
                        "stages_sha256": f.adoption["original_stages"]["sha256"],
                    },
                }
            },
            "launch": {
                "decoded_query": {
                    "returncode": 0,
                    "cwd": str(self.checkout),
                    "supervisor_id": "luna-supervisor-a8-a1-native-order-20261002-08",
                    "supervisor_state_path": str(sup_path),
                    "exact_command": [
                        "exec .venv/bin/python scripts/train_continuous_w1ax.py "
                        "--start --allow-cuda --resume --prepare-only "
                        f"--stages-manifest {f.original_path.relative_to(self.checkout)} "
                        f"--run-dir {f.old}",
                    ],
                }
            },
            "first_health": {"decoded_query": {"supervisor": supervisor}},
        }
        completion = {
            "decoded_query": {
                "status": {
                    "phase": "readiness_complete",
                    "captures_done": 2,
                    "captures_total": 2,
                    "optimization_started": False,
                    "models": {},
                },
                "completed_label_manifest_count": 2,
                "supervisor": supervisor,
                "checker_command": [str(f.old / "status.json"), str(sup_path)],
                # A stale heartbeat does not revoke completed full audits.
                "checker_exit_status": 2,
            }
        }
        operation_record = self.write(self.root / "operation.json", operation)
        completion_record = self.write(self.root / "completion.json", completion)
        self.operation_pin = patch.object(
            stages, "HISTORICAL_OPERATION_SHA256", operation_record["sha256"]
        )
        self.completion_pin = patch.object(
            stages, "HISTORICAL_COMPLETION_SHA256", completion_record["sha256"]
        )
        self.operation_pin.start()
        self.completion_pin.start()
        self.addCleanup(self.operation_pin.stop)
        self.addCleanup(self.completion_pin.stop)
        common = {
            k: f.sources["sha256"][k]
            for k in (
                "target_gguf",
                "candidate_d_gguf",
                "base_draft_gguf",
                "absolute_d2t",
                "model_snapshot_manifest",
            )
        }
        ready = {
            "schema": stages.READINESS_SCHEMA,
            "training_eligible": True,
            "objective": "hard_ce",
            "scale_layout": "row",
            "unresolved_gates": [],
            "scope": "joint_body_head_exact_prefix_teacher_forced_training",
            "common_source_sha256": common,
            "native_binary_sha256": f.sources["sha256"]["binary"],
            "native_runtime": f.sources["native_runtime"],
            "precisions": f.adoption["precision_gates"],
            "teacher_capture_manifest_sha256": [
                c["label_manifest"]["sha256"] for c in f.adoption["captures"]
            ],
        }
        ready_path = f.old / "stages/readiness.json"
        ready_record = self.write(ready_path, ready)
        indexes = {}
        for ordinal, capture in enumerate(f.captures):
            manifest = Path(f.adoption["captures"][ordinal]["label_manifest"]["path"])
            spec_path = f.old / f"stages/provider-{ordinal:05d}.json"
            stages.provider_manifest(f.sources, manifest, ready_path, spec_path)
            split = capture["split"]
            indexes[split] = self.write(
                f.old / f"stages/{split}-providers.json",
                {
                    "schema": "w1ax_streaming_train_v2",
                    "split": split,
                    "training_eligible": split == "train",
                    "continuous_readiness": ready_record,
                    "shards": [
                        {
                            "ordinal": 0,
                            "provider_manifest": str(spec_path),
                            "provider_manifest_sha256": stages.sha256(spec_path),
                        }
                    ],
                },
            )
        f.adoption["historical_audit_provenance"] = {
            "schema": stages.HISTORICAL_AUDIT_SCHEMA,
            "operation": operation_record,
            "completion": completion_record,
            "readiness": ready_record,
            "provider_indexes": indexes,
            "audit_reports": [
                stages.file_record(
                    Path(c["label_manifest"]["path"]).parent / "audit.json",
                )
                for c in f.adoption["captures"]
            ],
        }
        self.config = f.publish()
        self.record = self.config["retained_native_capture_import"]

    def write(self, path, value):
        stages.write_json(path, value)
        return stages.file_record(path)

    def publish(self):
        self.config = self.f.publish()
        self.record = self.config["retained_native_capture_import"]

    def proof(self):
        return stages._historical_retained_pass(self.record)

    def adopt(self, ordinal=0):
        path = self.f.receipts / f"capture-{ordinal:05d}.json"
        stages._adopt_retained_historical_audit(self.record, ordinal, path)
        return path

    def load(self, ordinal=0):
        entry = self.f.adoption["captures"][ordinal]["label_manifest"]
        capture = self.f.captures[ordinal]
        return stages.load_native_labels(
            Path(entry["path"]),
            expected_manifest_sha256=entry["sha256"],
            expected_prompt_sha256=capture["prompts_sha256"],
            expected_prompt_count=capture["prompt_count"],
            audit_receipt=stages.file_record(self.f.receipts / f"capture-{ordinal:05d}.json"),
        )

    def test_authentic_complete_pass_adopts_actual_reports_and_preserves_old_bytes(self):
        before = self.f.snapshot_old()
        with patch.object(stages, "audit_native_labels", side_effect=AssertionError("re-audit")):
            for ordinal in range(2):
                receipt_path = self.adopt(ordinal)
                receipt = json.loads(receipt_path.read_text())
                old_report = json.loads(
                    Path(
                        self.f.adoption["historical_audit_provenance"]["audit_reports"][ordinal][
                            "path"
                        ]
                    ).read_text()
                )
                self.assertEqual(receipt["report"], old_report)
                self.assertFalse(receipt["report"]["training_eligible"])
                self.assertEqual(
                    receipt["audit_origin"]["producer_commit"], stages.HISTORICAL_PRODUCER_COMMIT
                )
                self.assertEqual(self.load(ordinal).report, old_report)
                self.adopt(ordinal)
        self.assertEqual(before, self.f.snapshot_old())
        self.assertFalse((self.f.new / "preparation-ready.json").exists())

    def test_source_ast_and_dependencies_are_proven_from_original_git_objects(self):
        with patch.object(stages.subprocess, "run", wraps=stages.subprocess.run) as git:
            source = stages._historical_audit_source_proof()
            stages._historical_audit_source_proof()
        self.assertEqual(git.call_count, 7)
        self.assertIn("scripts/audit_recurrent_response.py", source["files"])
        for call in git.call_args_list:
            self.assertIn(stages.HISTORICAL_PRODUCER_COMMIT, call.args[0][-1])

    def test_changed_semantic_dependency_rejects_historical_adoption(self):
        run = stages.subprocess.run

        def changed(*args, **kwargs):
            result = run(*args, **kwargs)
            if args[0][-1].endswith("scripts/audit_recurrent_response.py"):
                result.stdout += b"# changed semantic dependency\n"
            return result

        with patch.object(stages.subprocess, "run", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "semantic dependency changed"):
                self.proof()
        self.assertFalse(self.f.receipts.exists())

    def test_changed_auditor_ast_rejects_without_normalization(self):
        run = stages.subprocess.run

        def changed(*args, **kwargs):
            result = run(*args, **kwargs)
            if args[0][-1].endswith("scripts/w1ax_continuous_stages.py"):
                result.stdout = result.stdout.replace(
                    b'"execution_device": "cpu"', b'"execution_device": "changed"'
                )
            return result

        with patch.object(stages.subprocess, "run", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "semantic functions/imports/policy"):
                self.proof()

    def test_missing_or_unauthenticated_evidence_rejects(self):
        provenance = self.f.adoption["historical_audit_provenance"]
        original = copy.deepcopy(provenance)
        for kind in ("operation", "completion", "readiness", "audit_reports", "provider_indexes"):
            with self.subTest(kind=kind):
                self.f.adoption["historical_audit_provenance"] = copy.deepcopy(original)
                del self.f.adoption["historical_audit_provenance"][kind]
                self.publish()
                with self.assertRaisesRegex(ValueError, "authenticated full-pass"):
                    self.proof()
        self.f.adoption["historical_audit_provenance"] = original
        original["operation"]["sha256"] = "0" * 64
        self.publish()
        with self.assertRaisesRegex(ValueError, "authenticated full-pass"):
            self.proof()

    def test_ordinary_sidecar_or_status_ordinal_is_not_provenance(self):
        self.f.adoption["historical_audit_provenance"] = {"phase": "readiness_complete"}
        self.publish()
        with self.assertRaisesRegex(ValueError, "authenticated full-pass"):
            self.proof()

    def test_readiness_order_and_all_manifest_digests_must_match(self):
        provenance = self.f.adoption["historical_audit_provenance"]
        record = provenance["readiness"]
        value = json.loads(Path(record["path"]).read_text())
        value["teacher_capture_manifest_sha256"].reverse()
        provenance["readiness"] = self.write(Path(record["path"]), value)
        self.publish()
        with self.assertRaisesRegex(ValueError, "full ordered readiness"):
            self.proof()

    def test_index_coverage_and_exact_readiness_binding_must_match(self):
        provenance = self.f.adoption["historical_audit_provenance"]
        for corruption in ("coverage", "readiness"):
            record = provenance["provider_indexes"]["train"]
            original = json.loads(Path(record["path"]).read_text())
            changed = copy.deepcopy(original)
            if corruption == "coverage":
                changed["shards"] = []
            else:
                changed["continuous_readiness"]["sha256"] = "0" * 64
            provenance["provider_indexes"]["train"] = self.write(Path(record["path"]), changed)
            self.publish()
            with self.subTest(corruption=corruption):
                with self.assertRaisesRegex(ValueError, "index/readiness/full coverage"):
                    self.proof()
            provenance["provider_indexes"]["train"] = self.write(Path(record["path"]), original)

    def test_provider_capture_join_rejects_even_rehashed_metadata(self):
        provenance = self.f.adoption["historical_audit_provenance"]
        index_record = provenance["provider_indexes"]["train"]
        index = json.loads(Path(index_record["path"]).read_text())
        child = index["shards"][0]
        spec_path = Path(child["provider_manifest"])
        spec = json.loads(spec_path.read_text())
        spec["sha256"]["capture_manifest"] = self.f.adoption["captures"][1]["label_manifest"][
            "sha256"
        ]
        self.write(spec_path, spec)
        child["provider_manifest_sha256"] = stages.sha256(spec_path)
        provenance["provider_indexes"]["train"] = self.write(Path(index_record["path"]), index)
        self.publish()
        with self.assertRaisesRegex(ValueError, "provider capture/readiness/source join"):
            self.proof()

    def test_sidecar_prompt_and_manifest_join_rejects_rehashed_report(self):
        provenance = self.f.adoption["historical_audit_provenance"]
        record = provenance["audit_reports"][0]
        report = json.loads(Path(record["path"]).read_text())
        report["training_prompt_count"] += 1
        provenance["audit_reports"][0] = self.write(Path(record["path"]), report)
        self.publish()
        with self.assertRaisesRegex(ValueError, "report manifest/prompt/split"):
            self.proof()

    def test_metadata_identity_rechecked_after_cached_full_proof(self):
        self.proof()
        record = self.f.adoption["historical_audit_provenance"]["readiness"]
        path = Path(record["path"])
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.proof()

    def test_current_payload_and_inventory_are_required_before_publication(self):
        self.proof()
        manifest = Path(self.f.adoption["captures"][0]["label_manifest"]["path"])
        m = json.loads(manifest.read_text())
        for kind, basename in (
            ("payload", m["files"]["features"]["path"]),
            ("request", m["requests"][0]["request"]["path"]),
        ):
            path = manifest.parent / basename
            original = path.read_bytes()
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.subTest(kind=kind):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.adopt()
                self.assertFalse(self.f.receipts.exists())
            path.write_bytes(original)
        (manifest.parent / "unclaimed.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.adopt()
        self.assertFalse(self.f.receipts.exists())

    def test_receipt_origin_and_provenance_digest_are_validated_on_reuse(self):
        path = self.adopt()
        receipt = json.loads(path.read_text())
        receipt["audit_origin"]["provenance_sha256"] = "0" * 64
        self.write(path, receipt)
        with self.assertRaisesRegex(ValueError, "provenance/report/shard"):
            self.load()

    def test_interrupted_historical_publication_leaves_no_receipt(self):
        self.proof()
        with patch.object(stages.os, "replace", side_effect=InterruptedError("stop")):
            with self.assertRaises(InterruptedError):
                self.adopt()
        self.assertEqual(list(self.f.receipts.iterdir()), [])

    def test_stage_resume_and_label_loading_do_not_repeat_authenticated_full_pass(self):
        before = self.f.snapshot_old()
        self.f.new.mkdir()
        with (
            patch("check_continuous_w1ax_readiness.validate_gate_report"),
            patch(
                "check_continuous_w1ax_readiness.run_gate", side_effect=AssertionError("rerun gate")
            ),
            patch.object(provider, "TARGET_GGUF_SHA256", self.f.sources["sha256"]["target_gguf"]),
            patch.object(
                provider, "CANDIDATE_D_SHA256", self.f.sources["sha256"]["candidate_d_gguf"]
            ),
            patch.object(stages, "audit_q4_file"),
            patch.object(stages, "host_admission"),
            patch.object(stages, "stage_progress"),
            patch.object(stages, "native_capture", side_effect=AssertionError("recapture")),
            patch.object(stages, "build_native_labels", side_effect=AssertionError("rebuild")),
            patch.object(stages, "audit_native_labels", side_effect=AssertionError("re-audit")),
        ):
            first = stages._run_stages(self.config, self.f.new)
            self.assertEqual(first, stages._run_stages(self.config, self.f.new))
            for ordinal in range(2):
                self.assertEqual(self.load(ordinal).report, self.load(ordinal).report)
        self.assertEqual(before, self.f.snapshot_old())
        self.assertFalse((self.f.new / "preparation-ready.json").exists())


if __name__ == "__main__":
    unittest.main()

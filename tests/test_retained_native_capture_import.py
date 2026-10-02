"""Retained capture preparation: tiny CPU fixtures, no real models or CUDA."""

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import test_w1ax_continuous_stages as fixtures
import w1ax_continuous_stages as stages
import w1ax_capture_provider as provider


class RetainedNativeCaptureImportTests(unittest.TestCase):
    def setUp(self):
        self.train = fixtures.NativeLabelStagesTests()
        self.train.setUp()
        self.addCleanup(self.train.doCleanups)
        self.root = self.train.f.root.resolve()
        self.old = self.root / "oldrun"
        self.new = self.root / "newrun"
        self.receipts = self.root / "receipts"
        source_names = ("target_gguf", "candidate_d_gguf", "base_draft_gguf",
                        "absolute_d2t", "model_snapshot_manifest", "binary", "q4_0_draft")
        self.sources = {"sha256": {}, "native_runtime": {"fixture": "CPU"}}
        for name in source_names:
            path = self.root / name
            if name == "absolute_d2t":
                path.write_bytes(self.train.f.native.absolute_map.read_bytes())
            elif name == "model_snapshot_manifest":
                path.write_text(json.dumps({"models": {
                    role: {"directory": str(self.root / role)} for role in ("target", "draft")
                }}))
            else:
                path.write_text("tiny CPU source " + name)
            self.sources[name] = str(path)
            self.sources["sha256"][name] = stages.sha256(path)
        self.sources["native_runtime"] = {
            "immutable_manifest": stages.file_record(Path(self.sources["binary"])),
            "libraries": [stages.file_record(Path(self.sources["binary"]))],
        }
        self.captures = []
        for ordinal, split in enumerate(("train", "development")):
            f = self.train if split == "train" else fixtures.NativeLabelStagesTests()
            if split != "train":
                f.setUp()
                self.addCleanup(f.doCleanups)
                # Give the independent dev fixture a distinct prompt identity.
                for path in f.f.root.rglob("*"):
                    if path.is_file() and path.suffix in {".json", ".jsonl"}:
                        path.write_text(path.read_text().replace("train-a", "dev-a"))
            cell = json.loads(f.f.cell_manifest.read_text())
            cell.update(target_sha256=self.sources["sha256"]["target_gguf"],
                        draft_sha256=self.sources["sha256"]["candidate_d_gguf"],
                        binary_sha256=self.sources["sha256"]["binary"],
                        prompts_sha256=stages.sha256(f.f.native.prompts))
            cell["requests"][0]["prompt_sha256"] = stages.sha256(f.f.cell / "request-000/prompt.json")
            f.f.cell_manifest.write_text(json.dumps(cell))
            folder = self.old / f"stages/capture-{ordinal:05d}/labels"
            stages.build_native_labels(f.f.capture, f.f.native.prompts,
                                       Path(self.sources["absolute_d2t"]), folder,
                                       split=split, target_vocab_size=8)
            prompts = folder / "train.jsonl"
            self.captures.append({
                "split": split, "prompts": str(prompts),
                "prompts_sha256": stages.sha256(prompts), "prompt_count": 1,
                "storage_forecast": {"upper_bound_bytes": 1024},
            })
        self.dev = self.root / "development.jsonl"
        self.dev.write_text(Path(self.captures[1]["prompts"]).read_text() * 24)
        self.original = {
            "schema": stages.STAGES_SCHEMA, "sources": self.sources,
            "captures": self.captures,
            "gate_prompts": self.captures[0]["prompts"],
            "gate_prompts_sha256": self.captures[0]["prompts_sha256"],
            "development_prompts": str(self.dev),
            "development_prompts_sha256": stages.sha256(self.dev),
            "max_capture_storage_bytes": 4096, "min_free_disk_bytes": 0,
        }
        self.original_path = self.root / "original-stages.json"
        stages.write_json(self.original_path, self.original)
        self.adoption = {
            "schema": stages.RETAINED_IMPORT_SCHEMA,
            "original_stages": stages.file_record(self.original_path),
            "original_run_dir": str(self.old), "output_run_dir": str(self.new),
            "audit_receipts_dir": str(self.receipts), "historical_audit_provenance": None,
            "captures": [{"ordinal": i, "label_manifest": stages.file_record(
                self.old / f"stages/capture-{i:05d}/labels/manifest.json")}
                for i in range(2)], "precision_gates": {},
        }
        for bits in (8, 1):
            path = self.old / f"stages/gate-a{bits}/gate.json"
            stages.write_json(path, {
                "schema": "w1ax_continuous_precision_gate_v1", "activation_bits": bits,
                "native_binary_sha256": self.sources["sha256"]["binary"],
                "native_runtime": self.sources["native_runtime"],
                "evidence": {"native_capture_manifest": self.adoption["captures"][0]["label_manifest"]},
            })
            self.adoption["precision_gates"][str(bits)] = stages.file_record(path)
        self.import_path = self.root / "import.json"
        self.output = self.root / "retained-stages.json"
        self.config = self.publish()
        stages._VERIFIED_RECORDS.clear()
        stages._OBSERVED_RECORDS.clear()

    def publish(self):
        stages.write_json(self.import_path, self.adoption)
        return {**self.original,
                "retained_native_capture_import": stages.file_record(self.import_path),
                "native_label_audit_receipts_dir": self.adoption["audit_receipts_dir"]}

    def validate(self, config=None):
        with patch("check_continuous_w1ax_readiness.validate_gate_report") as gate:
            result = stages._validate_retained_import(config or self.config, self.new)
            self.assertEqual(gate.call_count, 2)
            return result

    def snapshot_old(self):
        return {str(p.relative_to(self.old)): p.read_bytes()
                for p in self.old.rglob("*") if p.is_file()}

    def test_cpu_config_publication_has_no_eligibility_or_old_writes(self):
        before = self.snapshot_old()
        with (patch("check_continuous_w1ax_readiness.validate_gate_report"),
              patch("torch.cuda.is_available", side_effect=AssertionError("GPU query")),
              patch.object(stages, "native_capture", side_effect=AssertionError("recapture")),
              patch.object(stages, "audit_native_labels", side_effect=AssertionError("premature audit"))):
            result = stages.prepare_retained_config(self.import_path, self.output)
        self.assertEqual(result, self.config)
        self.assertEqual(before, self.snapshot_old())
        self.assertFalse(self.receipts.exists())
        self.assertFalse(self.new.exists())
        for entry in self.adoption["captures"]:
            self.assertFalse(json.loads(Path(entry["label_manifest"]["path"]).read_text())["training_eligible"])

    def test_new_stage_outputs_reference_retained_captures_and_receipts(self):
        before = self.snapshot_old()
        self.new.mkdir()
        with (patch("check_continuous_w1ax_readiness.validate_gate_report"),
              patch("check_continuous_w1ax_readiness.run_gate", side_effect=AssertionError("rerun gate")),
              patch.object(provider, "TARGET_GGUF_SHA256", self.sources["sha256"]["target_gguf"]),
              patch.object(provider, "CANDIDATE_D_SHA256", self.sources["sha256"]["candidate_d_gguf"]),
              patch.object(stages, "audit_q4_file"), patch.object(stages, "host_admission"),
              patch.object(stages, "stage_progress"),
              patch.object(stages, "native_capture", side_effect=AssertionError("recapture")),
              patch.object(stages, "build_native_labels", side_effect=AssertionError("rebuild")),
              patch.object(stages, "audit_native_labels", wraps=stages.audit_native_labels) as audit):
            result = stages._run_stages(self.config, self.new)
            stages._run_stages(self.config, self.new)
            # A1/A8 provider label reads also reuse the same receipts.
            for i, capture in enumerate(self.captures):
                manifest = Path(self.adoption["captures"][i]["label_manifest"]["path"])
                for _ in range(2):
                    stages.load_native_labels(
                        manifest, expected_prompt_sha256=capture["prompts_sha256"],
                        expected_prompt_count=1,
                        expected_manifest_sha256=self.adoption["captures"][i]["label_manifest"]["sha256"],
                        audit_receipt=stages.file_record(self.receipts / f"capture-{i:05d}.json"))
            self.assertEqual(audit.call_count, 2)
        self.assertEqual(before, self.snapshot_old())
        self.assertTrue(result.is_relative_to(self.new))
        self.assertFalse((self.new / "preparation-ready.json").exists())
        self.assertFalse((self.new / "stages/capture-00000").exists())
        for i, capture in enumerate(self.captures):
            spec = json.loads((self.new / f"stages/provider-{i:05d}.json").read_text())
            self.assertEqual(spec["paths"]["capture_manifest"], self.adoption["captures"][i]["label_manifest"]["path"])
            self.assertEqual(spec["sha256"]["capture_manifest"], self.adoption["captures"][i]["label_manifest"]["sha256"])
            self.assertEqual(spec["native_label_audit_receipt"], stages.file_record(self.receipts / f"capture-{i:05d}.json"))
            self.assertEqual(spec["continuous_readiness"], stages.file_record(self.new / "stages/readiness.json"))
            self.assertEqual(spec["training_eligible"], capture["split"] == "train")
        dev = json.loads((self.new / "stages/development.json").read_text())
        self.assertTrue(Path(dev["providers_manifest"]).is_relative_to(self.new))
        self.assertEqual(dev["full_pool_prompt_count"], 1)

    def test_original_config_sha_change_is_rejected(self):
        self.original_path.write_text(self.original_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validate()

    def test_new_config_cannot_change_original_source_or_split(self):
        for mutation in (lambda c: c["sources"]["sha256"].update(binary="0" * 64),
                         lambda c: c["captures"][0].update(split="development"),
                         lambda c: c.update(gate_prompts_sha256="0" * 64)):
            with self.subTest(mutation=mutation):
                config = copy.deepcopy(self.config)
                mutation(config)
                with self.assertRaisesRegex(ValueError, "original stages"):
                    self.validate(config)

    def test_missing_reordered_partial_or_duplicate_capture_fails(self):
        for entries in (self.adoption["captures"][:1], list(reversed(self.adoption["captures"])),
                        [self.adoption["captures"][0]] * 2):
            with self.subTest(entries=entries):
                self.adoption["captures"] = entries
                with self.assertRaises(ValueError):
                    self.validate(self.publish())

    def test_changed_retained_manifest_requires_explicit_digest(self):
        path = Path(self.adoption["captures"][0]["label_manifest"]["path"])
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validate()

    def test_rehashed_wrong_teacher_source_map_split_or_precision_fails(self):
        path = Path(self.adoption["captures"][0]["label_manifest"]["path"])
        original = json.loads(path.read_text())
        for field, value in (("split", "development"), ("binary_sha256", "0" * 64),
                             ("target_sha256", "0" * 64), ("draft_sha256", "0" * 64),
                             ("activation_bits", 8), ("prompt_count", 2),
                             ("prompts_sha256", "0" * 64)):
            with self.subTest(field=field):
                stages.write_json(path, {**original, field: value})
                self.adoption["captures"][0]["label_manifest"] = stages.file_record(path)
                with self.assertRaisesRegex(ValueError, "ancestry"):
                    self.validate(self.publish())
        original["files"]["absolute_d2t"]["sha256"] = "0" * 64
        stages.write_json(path, original)
        self.adoption["captures"][0]["label_manifest"] = stages.file_record(path)
        with self.assertRaisesRegex(ValueError, "ancestry"):
            self.validate(self.publish())

    def test_changed_owned_payload_and_unclaimed_payload_fail(self):
        folder = Path(self.adoption["captures"][0]["label_manifest"]["path"]).parent
        features = folder / "features.npy"
        before = features.read_bytes()
        features.write_bytes(before[:-1] + b"x")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validate()
        features.write_bytes(before)
        (folder / "hidden.bin").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "inventory"):
            self.validate()

    def test_rehashed_wrong_cache_sampler_fails_full_audit_before_readiness(self):
        path = Path(self.adoption["captures"][0]["label_manifest"]["path"])
        m = json.loads(path.read_text())
        request = path.parent / m["requests"][0]["request"]["path"]
        options = json.loads(request.read_text())
        options["cache_prompt"] = True
        stages.write_json(request, options)
        m["requests"][0]["request"]["sha256"] = stages.sha256(request)
        cell = path.parent / m["files"]["source_cell"]["path"]
        cell_value = json.loads(cell.read_text())
        cell_value["requests"][0]["request_sha256"] = stages.sha256(request)
        stages.write_json(cell, cell_value)
        m["files"]["source_cell"]["sha256"] = stages.sha256(cell)
        stages.write_json(path, m)
        self.adoption["captures"][0]["label_manifest"] = stages.file_record(path)
        for bits, record in self.adoption["precision_gates"].items():
            gate_path = Path(record["path"])
            gate = json.loads(gate_path.read_text())
            gate["evidence"]["native_capture_manifest"] = stages.file_record(path)
            stages.write_json(gate_path, gate)
            self.adoption["precision_gates"][bits] = stages.file_record(gate_path)
        self.config = self.publish()
        self.new.mkdir()
        with (patch("check_continuous_w1ax_readiness.validate_gate_report"),
              patch.object(provider, "TARGET_GGUF_SHA256", self.sources["sha256"]["target_gguf"]),
              patch.object(provider, "CANDIDATE_D_SHA256", self.sources["sha256"]["candidate_d_gguf"]),
              patch.object(stages, "audit_q4_file"), patch.object(stages, "host_admission"),
              patch.object(stages, "stage_progress")):
            with self.assertRaisesRegex(ValueError, "sampler/cache"):
                stages._run_stages(self.config, self.new)
        self.assertFalse((self.new / "stages/readiness.json").exists())
        self.assertFalse((self.receipts / "capture-00000.json").exists())

    def test_missing_gate_fails(self):
        self.adoption["precision_gates"].pop("1")
        with self.assertRaisesRegex(ValueError, "both original"):
            self.validate(self.publish())

    def test_rehashed_wrong_gate_runtime_fails(self):
        gate_path = Path(self.adoption["precision_gates"]["1"]["path"])
        gate = json.loads(gate_path.read_text())
        gate["native_binary_sha256"] = "0" * 64
        stages.write_json(gate_path, gate)
        self.adoption["precision_gates"]["1"] = stages.file_record(gate_path)
        with self.assertRaisesRegex(ValueError, "gate runtime/recipe"):
            self.validate(self.publish())

    def test_missing_capture_file_fails_without_recapture(self):
        Path(self.adoption["captures"][0]["label_manifest"]["path"]).unlink()
        with patch.object(stages, "native_capture", side_effect=AssertionError("recapture")):
            with self.assertRaisesRegex(ValueError, "regular file"):
                self.validate()

    def test_current_run_must_match_explicit_new_output(self):
        with self.assertRaisesRegex(ValueError, "separate new output"):
            stages._validate_retained_import(self.config, self.root / "different-run")

    def test_real_gate_validator_is_required(self):
        with self.assertRaisesRegex(ValueError, "gate failed"):
            stages._validate_retained_import(self.config, self.new)

    def test_gate_prompt_ancestry_is_required(self):
        gate_path = Path(self.adoption["precision_gates"]["1"]["path"])
        gate = json.loads(gate_path.read_text())
        gate["evidence"]["native_capture_manifest"] = self.adoption["captures"][1]["label_manifest"]
        stages.write_json(gate_path, gate)
        self.adoption["precision_gates"]["1"] = stages.file_record(gate_path)
        with self.assertRaisesRegex(ValueError, "gate prompt/split"):
            self.validate(self.publish())

    def test_historical_sidecar_provenance_is_not_assumed(self):
        self.adoption["historical_audit_provenance"] = {"trust": "audit.json"}
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.validate(self.publish())

    def test_oldrun_destination_receipt_or_stage_alias_fails(self):
        for field, value in (("output_run_dir", str(self.old / "child")),
                             ("audit_receipts_dir", str(self.old / "receipts"))):
            adoption = copy.deepcopy(self.adoption)
            adoption[field] = value
            self.adoption = adoption
            with self.assertRaisesRegex(ValueError, "separate"):
                self.validate(self.publish())
        self.adoption["output_run_dir"] = str(self.new)
        self.adoption["audit_receipts_dir"] = str(self.receipts)
        self.new.mkdir()
        (self.new / "stages").symlink_to(self.old / "stages", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "aliases"):
            self.validate(self.publish())

    def test_source_payload_change_is_rejected(self):
        Path(self.sources["base_draft_gguf"]).write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validate()

    def test_existing_config_output_cannot_be_overwritten(self):
        self.output.write_text("preserve")
        with patch("check_continuous_w1ax_readiness.validate_gate_report"):
            with self.assertRaisesRegex(ValueError, "new output"):
                stages.prepare_retained_config(self.import_path, self.output)
        self.assertEqual(self.output.read_text(), "preserve")


if __name__ == "__main__":
    unittest.main()

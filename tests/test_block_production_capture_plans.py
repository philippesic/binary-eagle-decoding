"""CPU proposal preparation and fail-closed ancestry/resource tests; no model/GPU."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_block_production_capture_plans as proposals


class ProposalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.stage = self.source / "results/staged"
        self.stage.mkdir(parents=True)
        self.options = self.root / "options"
        self.options.mkdir()
        prompts, indices = [], []
        for row in range(900):
            domain = proposals.capture.DOMAINS[row % 3]
            ident = f"original:{row}"
            prompts.append({"id": ident, "domain": domain, "split": "TRAIN",
                            "messages": [{"role": "user", "content": f"Original prompt {row}"}]})
            indices.append({"id": ident, "domain": domain, "source_split": "TRAIN",
                            "group": f"group:{row}", "input_tokens": 8})
        for name, records in (("train-00000.jsonl", prompts), ("train-00000.index.jsonl", indices)):
            (self.stage / name).write_text("".join(json.dumps(r) + "\n" for r in records))
        sha = proposals.capture.file_sha256
        shard = {"prompts": "train-00000.jsonl", "index": "train-00000.index.jsonl",
                 "prompts_sha256": sha(self.stage / "train-00000.jsonl"),
                 "index_sha256": sha(self.stage / "train-00000.index.jsonl"), "prompts_count": 900}
        (self.stage / "corpus-manifest.json").write_text(json.dumps({"files": {"train": {"shards": [shard]}}}))
        (self.stage / "target-source-pins.json").write_text(json.dumps({
            "tokenizer_metadata_sha256": "b" * 64, "chat_template_sha256": "c" * 64}))
        historical = Path("/historical/original")
        sources = {name: {"path": str(historical / "results/staged" / name), "sha256": sha(self.stage / name)}
                   for name in ("corpus-manifest.json", "target-source-pins.json", "train-00000.jsonl", "train-00000.index.jsonl")}
        client_path = "scripts/capture_block_qat_teacher.py"
        client_pin = {"path": str(historical / client_path), "sha256": sha(proposals.ROOT / client_path)}
        depth_options = []
        for quota in (150, 250):
            selected = []
            for domain in proposals.capture.DOMAINS:
                eligible = [i for i, r in enumerate(indices) if r["domain"] == domain]
                for split, rows in (("calibration_fit", eligible[:32]),
                                    ("calibration_validation", eligible[32:48]), ("train", eligible[48:48 + quota])):
                    selected.extend({"shard": 0, "row": i, "prompt_id": prompts[i]["id"],
                                     "prompt_sha256": proposals.capture.content_hash(prompts[i]["messages"]),
                                     "domain": domain, "split": split} for i in rows)
            native = {"source_revision": proposals.NATIVE_REVISION,
                      "target": {"path": "/historical/target", "sha256": proposals.TARGET_SHA},
                      "client_source": client_pin, "tokenizer_metadata_sha256": "b" * 64,
                      "chat_template_sha256": "c" * 64, "gpu_layers": 999, "expected_compute_capability": [12, 0]}
            option = {"selected_for_execution": False, "artifact_kind": "proposal_ONLY", "source_files": sources,
                      "source_modules": {client_path: client_pin}, "original_train_selection": selected,
                      "capture_plan_draft": {"selection": selected, "native": native}}
            (self.options / f"{quota}-train-per-domain-exact_soft.json").write_text(json.dumps(option))
            depth_options.append({"quota": quota, "generation": 128, "selected": selected,
                                  "source_files": sources, "raw_retained": proposals.envelope(len(selected), 0)["raw_retained_bytes"],
                                  "protected": 1, "prelaunch": 2, "illustrative_extra": 3, "illustrative_allocation": 5})
        self.depth = self.root / "depth.json"
        self.depth.write_text(json.dumps({"status": proposals.STATUS, "options": depth_options}))
        patcher = patch.object(proposals, "DEPTH_SHA", sha(self.depth))
        patcher.start()
        self.addCleanup(patcher.stop)

    def prepare(self, **changes):
        kwargs = dict(options_dir=self.options, depth_evidence=self.depth, source_root=self.source,
                      checkout_root=Path("/home/philip/project"), asset_root=Path("/home/philip/project/data/transport"),
                      output_root=self.root / "output")
        kwargs.update(changes)
        return proposals.prepare(**kwargs)

    def test_both_real_schema_plans_validate_without_binary_or_target(self):
        summaries = self.prepare()
        self.assertEqual([s["train_prompts"] for s in summaries], [450, 750])
        for summary in summaries:
            self.assertFalse(summary["selected_for_execution"])
            self.assertEqual(summary["status"], proposals.STATUS)
            self.assertEqual(summary["geometry"]["max_teacher_shape"], [126, 151936])
            self.assertLess(summary["producer_bound_bytes"], summary["caps"]["max_total_bytes"])
            self.assertEqual(summary["caps"]["max_requests"], summary["all_role_prompts"] + 6)
            directory = self.root / "output" / f"captured-prefix-{summary['train_prompts']}"
            plan = proposals.load_json(directory / "remote-plan.json")
            corpus = proposals.load_json(directory / "remote-corpus.json")
            self.assertNotIn("/historical", json.dumps(plan))
            self.assertNotIn("/historical", json.dumps(corpus))
            self.assertEqual(len(corpus["files"]["train"]["shards"]), 1)
            self.assertEqual(plan["teacher_logits_layout"], "indexed")
            self.assertEqual(plan["objective"], "exact_soft")
            self.assertEqual(summary["disk_gate"]["status"], "PENDING_FRESH_DISK")
            self.assertIsNone(summary["runtime_library_and_build_files"])
        # Byte preservation covers historical receipts, not a rewritten path file.
        self.assertEqual((self.root / "output/historical-depth-evidence.json").read_bytes(), self.depth.read_bytes())
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_altered_role_or_selector_fails(self):
        p = self.options / "150-train-per-domain-exact_soft.json"
        d = proposals.load_json(p)
        d["original_train_selection"][0]["split"] = "train"
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "selectors differ"):
            self.prepare()

    def test_tampered_source_fails_before_plan(self):
        with (self.stage / "train-00000.index.jsonl").open("a") as stream:
            stream.write("{}\n")
        with self.assertRaisesRegex(ValueError, "source bytes differ"):
            self.prepare()

    def test_joined_but_changed_ascending_role_assignment_fails(self):
        option = proposals.load_json(self.options / "150-train-per-domain-exact_soft.json")
        depth = proposals.load_json(self.depth)["options"][0]
        option["original_train_selection"][0]["row"] = 897
        option["capture_plan_draft"]["selection"][0]["row"] = 897
        depth["selected"][0]["row"] = 897
        with self.assertRaisesRegex(ValueError, "ascending eligible"):
            proposals.validate_selection(option, depth, self.stage / "corpus-manifest.json")

    def test_changed_source_pin_cannot_self_authenticate(self):
        p = self.options / "150-train-per-domain-exact_soft.json"
        d = proposals.load_json(p)
        d["source_files"]["train-00000.jsonl"]["sha256"] = "0" * 64
        p.write_text(json.dumps(d))
        with self.assertRaises(ValueError):
            self.prepare()

    def test_tokenizer_metadata_join_fails(self):
        p = self.options / "150-train-per-domain-exact_soft.json"
        d = proposals.load_json(p)
        d["capture_plan_draft"]["native"]["tokenizer_metadata_sha256"] = "d" * 64
        p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError, "tokenizer/template"):
            self.prepare()

    def test_budget_reports_insufficient_750_without_selecting_450(self):
        budget = proposals.envelope(594, 0)["capture_incremental_prelaunch_free_bytes"] + 1024**2
        summaries = self.prepare(observed_free_disk_bytes=budget)
        self.assertEqual(summaries[0]["disk_gate"]["status"], "SNAPSHOT_SUFFICIENT_NOT_ADMITTED")
        self.assertEqual(summaries[1]["disk_gate"]["status"], "INSUFFICIENT")
        self.assertTrue(all(not s["selected_for_execution"] for s in summaries))

    def test_transport_cannot_escape_historical_source_root(self):
        with self.assertRaises(ValueError):
            proposals.relative_source({"path": "/somewhere/other/file", "sha256": "a" * 64}, "/original")

    def test_runtime_evidence_hash_required(self):
        bad = self.root / "wrong-provenance.json"
        bad.write_text("{}")
        with self.assertRaisesRegex(ValueError, "source bytes differ"):
            self.prepare(build_provenance=bad)
        with self.assertRaisesRegex(ValueError, "source bytes differ"):
            self.prepare(verified_transfer=bad)

    def test_cost_geometry_and_peak_match_native_functions(self):
        anchors = proposals.capture.generated_block_anchors(512, 640)
        rows = proposals.capture.block_teacher_indices(anchors)
        self.assertEqual(len(anchors), 18)
        self.assertEqual(len(rows), 126)
        self.assertEqual(rows, list(range(511, 637)))
        cost = proposals.envelope(594, 0)
        self.assertEqual(cost["native_peak_request_bytes"], 640 * 5 * 2560 * 4 + 129 * 151936 * 4)
        self.assertEqual(cost["raw_retained_bytes"], 594 * (640 * 5 * 2560 * 4 + 126 * 151936 * 4))
        self.assertEqual(cost["capture_envelope_bytes"] - cost["capture_retained_bytes"], 594 * 3 * 151936 * 4)

    def test_invalid_free_disk_value_is_rejected(self):
        for bad in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "typed nonnegative"):
                self.prepare(observed_free_disk_bytes=bad)


if __name__ == "__main__":
    unittest.main()

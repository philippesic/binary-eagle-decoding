"""Bounded CPU evidence: source joins, partial slots and opt-in client replay."""
import copy
import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_nine_model_train_data as capture
import prepare_dspark_full_pool_capture as full
from capture_block_qat_teacher import NativeTeacher
from test_block_data import fixture
import test_nine_model_train_capture as fixture_capture
SyntheticNativeTeacher = fixture_capture.SyntheticNativeTeacher
from w1a1_eagle.block_data import (BlockDataset, PARTIAL_LABEL_POLICY, block_teacher_indices,
                                     generated_block_anchors, file_sha256, token_sha256)


class PartialBatchTests(unittest.TestCase):
    def dataset(self, root, generated, indexed=False, policy=True):
        path = fixture(root)
        manifest = json.loads(path.read_text())
        receipt_path = root / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        for i, chain in enumerate(manifest["chains"]):
            n = generated + chain["prompt_length"]
            tokens = np.arange(n, dtype=np.int64)
            tokens[-1] = 2  # exactly one observed EOS decision, never fabricated
            features = np.arange(n*10, dtype=np.float32).reshape(n, 5, 2)
            logits = np.arange(n*32, dtype=np.float32).reshape(n, 32)
            if policy:
                chain["label_policy"] = PARTIAL_LABEL_POLICY
                receipt["chains"][chain["chain_id"]]["label_policy"] = PARTIAL_LABEL_POLICY
            chain["anchors"] = generated_block_anchors(3, n, partial=policy)
            if indexed:
                indices = block_teacher_indices(chain["anchors"], n if policy else None)
                chain["logits_indices"] = indices
                receipt["chains"][chain["chain_id"]]["logits_indices"] = indices
                logits = logits[indices]
            for key, value in (("tokens", tokens), ("features", features), ("logits", logits)):
                p = root / chain[key]["path"]
                np.save(p, value)
                chain[key]["sha256"] = file_sha256(p)
                receipt["chains"][chain["chain_id"]][key + "_sha256"] = chain[key]["sha256"]
        receipt_path.write_text(json.dumps(receipt))
        manifest["producer"]["receipt"]["sha256"] = file_sha256(receipt_path)
        path.write_text(json.dumps(manifest))
        return BlockDataset(path, expected_sha256=file_sha256(path), allow_synthetic=True)

    def test_all_short_eos_and_partial_indices_mask_prefix_predecessors(self):
        for generated in (1, 2, 6, 7, 8, 13, 14, 15, 16):
            for indexed in (False, True):
                with self.subTest(generated=generated, indexed=indexed), tempfile.TemporaryDirectory() as tmp:
                    ds = self.dataset(Path(tmp), generated, indexed=indexed)
                    self.addCleanup(ds.close)
                    anchors = generated_block_anchors(3, 3 + generated, partial=True)
                    consumed = []
                    for b, anchor in enumerate(anchors):
                        batch = ds.load_block("chain0", b, require_teacher=True)
                        valid = min(7, generated + 2 - anchor)
                        self.assertEqual(batch.input_tokens.shape, (7,))
                        self.assertEqual(batch.attention_allowed.shape, (7, anchor+7))
                        np.testing.assert_array_equal(batch.loss_mask, np.arange(7) < valid)
                        self.assertTrue(np.all(batch.labels[valid:] == -1))
                        self.assertTrue(np.all(batch.predecessor_ids[valid:] == -1))
                        self.assertTrue(np.all(batch.teacher_logits[valid:] == 0))
                        self.assertEqual(batch.teacher_prefix_sha256[valid:], (None,)*(7-valid))
                        self.assertEqual(batch.predecessor_ids[0], anchor)
                        np.testing.assert_array_equal(batch.predecessor_ids[1:valid], batch.labels[:valid-1])
                        for slot in range(valid):
                            self.assertEqual(batch.teacher_prefix_sha256[slot], token_sha256(np.arange(anchor+slot+1)))
                            np.testing.assert_array_equal(batch.teacher_logits[slot], np.arange((anchor+slot)*32, (anchor+slot+1)*32))
                        consumed.extend(batch.labels[batch.loss_mask].tolist())
                    self.assertEqual(len(consumed), generated)
                    self.assertEqual(consumed[-1], 2)
                    self.assertEqual(block_teacher_indices(anchors, 3+generated), list(range(2, 2+generated)))

    def test_full_seven_outputs_identical_to_existing_contract(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            old = self.dataset(Path(a), 14, policy=False)
            new = self.dataset(Path(b), 14)
            self.addCleanup(old.close)
            self.addCleanup(new.close)
            for i in (0, 1):
                x, y = old.load_block("chain0", i), new.load_block("chain0", i)
                for name in ("context_features", "input_tokens", "positions", "labels", "loss_mask",
                             "attention_allowed", "teacher_logits", "predecessor_ids"):
                    np.testing.assert_array_equal(getattr(x, name), getattr(y, name))
                self.assertEqual(x.teacher_prefix_sha256, y.teacher_prefix_sha256)
                self.assertEqual(x.prefix_tokens, y.prefix_tokens)

    def test_partial_map_missing_last_loss_row_rejected(self):
        self.assertEqual(generated_block_anchors(3, 4, partial=True), [2])
        self.assertEqual(generated_block_anchors(3, 4), [])
        with tempfile.TemporaryDirectory() as tmp:
            ds = self.dataset(Path(tmp), 8, indexed=True)
            ds.close()
            manifest = json.loads(ds.path.read_text())
            manifest["chains"][0]["anchors"] = [2]  # cannot silently discard tail
            ds.path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "horizon"):
                BlockDataset(ds.path, expected_sha256=file_sha256(ds.path), allow_synthetic=True)


class PartialClient(SyntheticNativeTeacher):
    """Exercise real Python client control flow against a synthetic native API."""
    generate_capture = NativeTeacher.generate_capture
    _save_receipt = NativeTeacher._save_receipt
    _validate_taps = staticmethod(NativeTeacher._validate_taps)
    short = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_tokens = kwargs["max_tokens"]
        self.ancestry = {"target_chat_template_sha256": capture.content_hash("fixture template")}

    def _capture_request(self, request, ancestry):
        p = request["prompt"]
        maximum = self.short or p["max_new_tokens"]
        result = SyntheticNativeTeacher.generate_capture(self, messages=p.get("messages"),
            prompt_text=p.get("text"), template_mode=p["template_mode"], max_new_tokens=maximum,
            max_prompt_tokens=p["max_prompt_tokens"], tap_ids=request["tap_ids"],
            logits_mode=request["logits_mode"], chain_ancestry=ancestry)
        result["id"] = request["id"]
        result["prompt"]["max_new_tokens"] = p["max_new_tokens"]
        result["generation"]["max_new_tokens"] = p["max_new_tokens"]
        if self.short:
            result["generation"]["termination"] = "eog"
        (self.root / result["id"]).mkdir()
        return result

    def capture_prefix(self, tokens, taps, *, logits_mode, chain_ancestry, decode_history, logits_indices=None):
        result = super().capture_prefix(tokens, taps, logits_mode=logits_mode,
                                      chain_ancestry=chain_ancestry, decode_history=decode_history)
        if logits_mode == "indexed":
            logits = np.arange(len(logits_indices)*32, dtype=np.float32).reshape(-1, 32)
            p = Path(result["files"]["logits"]["path"])
            logits.tofile(p)
            result["logits_indices"] = logits_indices
            result["logits_shape"] = list(logits.shape)
            result["files"]["logits"].update(shape=list(logits.shape), sha256=file_sha256(p))
        result["id"] = "replay-" + str(len(self.calls))
        (self.root / result["id"]).mkdir()
        self._save_receipt(result)
        return result


class PartialCaptureTests(unittest.TestCase):
    def test_real_client_two_pass_short_eos_import_and_source_tamper(self):
        fixture_case = fixture_capture.CaptureTests()
        fixture_case.setUp()
        self.addCleanup(fixture_case.doCleanups)
        fixture_case.plan.update(objective="exact_soft", teacher_logits_layout="indexed",
                                 label_policy=PARTIAL_LABEL_POLICY, capture_goldens=False)
        fixture_case.plan["caps"]["max_requests"] = 18
        fixture_case.plan["caps"]["max_request_bytes"] = 100000
        fixture_case.plan["caps"]["max_shard_bytes"] = 100000
        fixture_case.plan["caps"]["max_eagle_golden_tokens"] = 24
        path = fixture_case.root / "partial-plan.json"
        path.write_text(json.dumps(fixture_case.plan))
        with patch.object(PartialClient, "short", 1):
            report = capture.run_capture(path, file_sha256(path), fixture_case.root / "out", execute=True,
                    teacher_factory=PartialClient,
                    device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                    rss_query=lambda: 1, available_query=lambda: 10**9)
        self.assertEqual(report["status"], "PASS", report.get("failure"))
        manifest = report["manifests"]["dspark"]
        ds = BlockDataset(manifest["path"], expected_sha256=manifest["sha256"])
        self.addCleanup(ds.close)
        batch = ds.load_block("chain-000000", 0, require_teacher=True)
        self.assertEqual(batch.loss_mask.tolist(), [True]+[False]*6)
        source_pin = ds.native_receipts["chain-000000"]["generation_source"]
        source = Path(source_pin["path"])
        original = json.loads(source.read_text())
        self.assertEqual(original["generation"]["termination"], "eog")
        self.assertEqual(original["logits_mode"], "none")
        original["tokens"][-1] += 1
        source.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "changed after admission"):
            ds.load_block("chain-000000", 0, require_teacher=True)
        with self.assertRaisesRegex(ValueError, "source pin"):
            BlockDataset(manifest["path"], expected_sha256=manifest["sha256"])

    def test_controller_publication_binds_original_tensors_and_refuses_fake_restore(self):
        case = fixture_capture.CaptureTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.plan.update(objective="exact_soft", teacher_logits_layout="indexed",
                         label_policy=PARTIAL_LABEL_POLICY, capture_goldens=False)
        case.plan["caps"].update(max_requests=18, max_request_bytes=100000,
                                max_shard_bytes=100000, max_eagle_golden_tokens=24)
        pin = capture.write_json(case.root / "partial-plan.json", case.plan)
        variant = {"kind": "fresh_greedy", "corpus_variant": full.VARIANT}
        logical = {"schema": "block_logical_shard_plan_v1", "geometry": {"family": "dspark"},
                   "generation_variant": variant,
                   "chains": {"global-" + str(i): r for i, r in enumerate(case.plan["selection"])},
                   "shards": {"s0": ["global-" + str(i) for i in range(9)]},
                   "capture_plans": {"s0": {"remote": pin}}}
        logical_pin = capture.write_json(case.root / "logical.json", logical)
        request = {"schema": "block_shard_restore_request_v1", "plan": logical_pin,
                   "plan_sha256": capture.identity(logical), "shard": "s0",
                   "directory": str(case.root / "physical"), "original_identity": None,
                   "generation_variant": variant, "capture_bytes_bound": 8*1024**3}
        request_path = case.root / "request.json"
        request_path.write_text(json.dumps(request))
        output = case.root / "publication.json"
        pin = capture.run_shard_publication(request_path, output, teacher_factory=PartialClient,
                    device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                    rss_query=lambda: 1, available_query=lambda: 10**9)
        publication = json.loads(output.read_text())
        self.assertEqual(pin["sha256"], file_sha256(output))
        self.assertEqual(publication["schema"], "block_shard_publication_v1")
        self.assertEqual(publication["status"], "PASS")
        manifest = json.loads(Path(publication["manifest"]["path"]).read_text())
        self.assertEqual({c["chain_id"] for c in manifest["chains"]}, set(logical["shards"]["s0"]))
        self.assertTrue((case.root / "physical/original-tensor-seal.json").is_file())
        request.update(original_identity={"sealed": "not a restoration proof"}, directory=str(case.root / "second"))
        request_path.write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError, "verified archive/replay"):
            capture.run_shard_publication(request_path, case.root / "second-publication.json")
        self.assertFalse((case.root / "second").exists())

    def test_actual_recomputed_tensor_replay_and_byte_identical_receipt_restore(self):
        import shutil
        import replay_dspark_shard as replay_module
        case = fixture_capture.CaptureTests()
        case.setUp()
        self.addCleanup(case.doCleanups)
        case.plan.update(objective="exact_soft", teacher_logits_layout="indexed",
                         label_policy=PARTIAL_LABEL_POLICY, capture_goldens=False)
        case.plan["caps"].update(max_requests=18, max_request_bytes=100000,
                                max_shard_bytes=100000, max_eagle_golden_tokens=24)
        pin = capture.write_json(case.root / "partial-plan.json", case.plan)
        logical = {"schema": "block_logical_shard_plan_v1", "geometry": {"family": "dspark"},
                   "generation_variant": {"kind": "fresh_greedy"},
                   "chains": {"global-" + str(i): r for i, r in enumerate(case.plan["selection"])},
                   "shards": {"s0": ["global-" + str(i) for i in range(9)]},
                   "capture_plans": {"s0": {"remote": pin}}}
        logical_pin = capture.write_json(case.root / "logical.json", logical)
        physical = case.root / "physical"
        request = {"schema": "block_shard_restore_request_v1", "plan": logical_pin,
                   "plan_sha256": capture.identity(logical), "shard": "s0", "directory": str(physical),
                   "original_identity": None, "generation_variant": logical["generation_variant"],
                   "capture_bytes_bound": 8*1024**3}
        request_path = case.root / "request.json"
        request_path.write_text(json.dumps(request))
        kwargs = {"teacher_factory": PartialClient,
                  "device_query": lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                  "rss_query": lambda: 1, "available_query": lambda: 10**9}
        capture.run_shard_publication(request_path, case.root / "first-publication.json", **kwargs)
        identity = json.loads((physical / "original-tensor-seal.json").read_text())["identity"]
        metastore = case.root / "metastore"
        files = {}
        for src in physical.rglob("*"):
            if src.is_file() and src.suffix in (".json", ".jsonl", ".npy"):
                rel = src.relative_to(physical)
                dst = metastore / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                files[str(rel)] = {"path": str(dst), "sha256": file_sha256(dst)}
        metadata_pin = capture.write_json(case.root / "metadata.json", {
            "schema": "block_shard_original_metadata_v1", "plan_sha256": request["plan_sha256"],
            "shard": "s0", "original_directory": str(physical), "identity": identity,
            "manifest_relative_path": "dspark/manifest.json", "admission_relative_path": "dspark/completed-admission.json",
            "files": files})
        shutil.rmtree(physical)  # synthetic fixture owned exclusively by this test
        request.update(original_identity=identity, original_metadata=metadata_pin,
                       replay_admission=None, replay_probe=True)
        request_path.write_text(json.dumps(request))
        class ReplayFixture(PartialClient):
            def make(self, *args, **kwargs):
                (self.root.parent.parent / "client.py").write_bytes(b"synthetic injected client source fixture")
                return super().make(*args, **kwargs)
        kwargs["teacher_factory"] = ReplayFixture
        capture.run_shard_publication(request_path, case.root / "replay-publication.json", **kwargs)
        publication = json.loads((case.root / "replay-publication.json").read_text())
        join = json.loads(Path(publication["reconstruction_join"]["path"]).read_text())
        self.assertTrue(join["tensor_identity_equal"])
        self.assertTrue(join["historical_receipt_bytes_restored"])
        for cid, expected in identity.items():
            manifest = json.loads((physical / "dspark/manifest.json").read_text())
            chain = next(c for c in manifest["chains"] if c["chain_id"] == cid)
            for kind, digest in expected["artifacts"].items():
                self.assertEqual(file_sha256(Path(chain[kind]["path"])), digest)
            self.assertNotEqual(join["fresh_receipts"][cid]["sha256"], expected["artifacts"]["native_receipt"])
        ds = BlockDataset(publication["manifest"]["path"], expected_sha256=publication["manifest"]["sha256"],
                          admission_path=publication["admission"]["path"], admission_sha256=publication["admission"]["sha256"])
        ds.close()
        # Hardlinked original/fresh tensors charge each physical inode once.
        self.assertLessEqual(replay_module.physical_bytes(physical), capture.tree_bytes(physical))
        shutil.rmtree(physical)
        class FaultyReplay(ReplayFixture):
            def make(self, *args, **kwargs):
                result = super().make(*args, **kwargs)
                file = Path(result["files"]["features"]["path"])
                data = np.fromfile(file, dtype=np.float32)
                data[0] += 0.1
                data.tofile(file)
                result["files"]["features"]["sha256"] = file_sha256(file)
                return result
        kwargs["teacher_factory"] = FaultyReplay
        with self.assertRaisesRegex(ValueError, "differs from original"):
            capture.run_shard_publication(request_path, case.root / "failed-replay-publication.json", **kwargs)
        self.assertFalse((physical / "dspark/manifest.json").exists())
        self.assertTrue(FaultyReplay.instances[-1].closed)


class GroupTests(unittest.TestCase):
    def test_transitive_original_group_or_topic_and_content_not_deduplicated(self):
        records = [{"prompt_id": "a", "source_group": "g1", "source_topic": "t1"},
                   {"prompt_id": "b", "source_group": "g2", "source_topic": "t1"},
                   {"prompt_id": "c", "source_group": "g2", "source_topic": "t2"},
                   {"prompt_id": "d", "source_group": "g3", "source_topic": None}]
        groups = capture.canonical_groups(records)
        self.assertEqual(sorted(map(len, groups.values())), [1, 3])
        self.assertEqual(len({r["group_id"] for r in records[:3]}), 1)

    def test_byte_bounds_keep_whole_multirecord_groups_and_all_records(self):
        rows = [{"prompt_id": str(i), "group_id": str(i//3), "domain": "prose", "split": "train"}
                for i in range(30)]
        groups = full.interleave_groups(rows, calibration=False)
        chunks = full.make_chunks(groups, source_bytes=1024)
        self.assertEqual(sum(map(len, chunks)), 30)
        for group in groups:
            self.assertEqual(sum(all(r in chunk for r in group) for chunk in chunks), 1)


# Opt-in actual original corpus test: no GPU or sealed/development body reads.
@unittest.skipUnless(os.environ.get("DSPARK_PREPARED_PLAN_ROOT"), "actual prepared original source not selected")
class OriginalSourceTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ["DSPARK_PREPARED_PLAN_ROOT"])
        self.contract = json.loads((self.root / "local-contract.json").read_text())
        self.pin = {"path": "local-contract.json", "sha256": file_sha256(self.root / "local-contract.json")}

    def test_actual_all_ten_source_and_counts_lengths_and_imbalance(self):
        rows, summary = capture.validate_full_pool_contract(self.pin, self.root, self.contract["source_corpus"])
        self.assertEqual(len(rows), 10000)
        self.assertEqual(summary["counts"]["train"], {"prose": 3286, "code": 3285, "reasoning": 3285})
        self.assertEqual(summary["canonical_components"], 9969)
        self.assertEqual(summary["multi_record_components"], 25)
        self.assertEqual(summary["multi_record_prompts"], 56)
        self.assertEqual(sum(r["input_tokens"] > 512 for r in rows.values()), 167)
        self.assertEqual(sum(r["split"] == "train" for r in rows.values()), 9856)
        schedule = json.loads((self.root / "full-pool-schedule.json").read_text())
        seen = [p for c in schedule["chunks"] for p in c["prompt_ids"]]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {r["prompt_id"] for r in rows.values()})
        self.assertEqual(sum(len(c["prompt_ids"]) for c in schedule["chunks"] if c["phase"] == "calibration"), 144)

    def mutated_selector(self, change):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        selector = json.loads((self.root / "role-selector.json").read_text())
        change(selector["selection"])
        pin = capture.write_json(root / "selector.json", selector)
        contract = copy.deepcopy(self.contract)
        contract["selector"] = pin
        contract["original_corpus"]["path"] = str(self.root / contract["original_corpus"]["path"])
        corpus_pin = dict(contract["source_corpus"])
        corpus_pin["path"] = str(self.root / corpus_pin["path"])
        contract["source_corpus"] = corpus_pin
        contract_pin = capture.write_json(root / "contract.json", contract)
        return root, pin, contract_pin, corpus_pin

    def test_source_length_tamper_rejected_even_with_rehashed_test_selector(self):
        def change(rows):
            max(rows, key=lambda r: r["input_tokens"])["input_tokens"] = 512
        root, pin, contract, corpus = self.mutated_selector(change)
        with patch.object(capture, "APPROVED_DSPARK_SELECTOR_SHA256", pin["sha256"]):
            with self.assertRaisesRegex(ValueError, "uncropped-length"):
                capture.validate_full_pool_contract(contract, root, corpus)

    def test_cross_role_topic_component_leakage_rejected(self):
        def change(rows):
            counts = Counter(r["group_id"] for r in rows)
            next(r for r in rows if counts[r["group_id"]] > 1)["split"] = "calibration_fit"
        root, pin, contract, corpus = self.mutated_selector(change)
        with patch.object(capture, "APPROVED_DSPARK_SELECTOR_SHA256", pin["sha256"]):
            with self.assertRaisesRegex(ValueError, "cross-role leakage"):
                capture.validate_full_pool_contract(contract, root, corpus)

    def test_selector_sha_tamper_and_wrong_domain_counts_rejected(self):
        root, pin, contract, corpus = self.mutated_selector(lambda rows: rows[0].update(split="calibration_fit"))
        with self.assertRaisesRegex(ValueError, "selector contract"):
            capture.validate_full_pool_contract(contract, root, corpus)
        with patch.object(capture, "APPROVED_DSPARK_SELECTOR_SHA256", pin["sha256"]):
            with self.assertRaisesRegex(ValueError, "counts differ"):
                capture.validate_full_pool_contract(contract, root, corpus)


if __name__ == "__main__":
    unittest.main()

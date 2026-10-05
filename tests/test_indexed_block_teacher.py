"""CPU/synthetic indexed-storage checks; no real target or CUDA capture admission."""
# ruff: noqa: E402

import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import test_block_data as dense_fixtures
import test_nine_model_train_capture as capture_fixtures
from capture_block_qat_teacher import NativeTeacher

from w1a1_eagle.block_data import (
    BLOCK_LOGITS_SELECTION,
    BlockDataset,
    block_teacher_indices,
    file_sha256,
    generated_block_anchors,
)
from w1a1_eagle.block_qat import BlockQATConfig, block_loss


def indexed_fixture(root):
    path = dense_fixtures.fixture(root)
    manifest = json.loads(path.read_text())
    producer_path = root / manifest["producer"]["receipt"]["path"]
    producer = json.loads(producer_path.read_text())
    for chain in manifest["chains"]:
        indices = block_teacher_indices(chain["anchors"])
        logits_path = root / chain["logits"]["path"]
        np.save(logits_path, np.load(logits_path)[indices])
        chain["logits"]["sha256"] = file_sha256(logits_path)
        chain["logits_indices"] = indices
        producer["chains"][chain["chain_id"]].update(
            logits_sha256=chain["logits"]["sha256"], logits_indices=indices
        )
    producer_path.write_text(json.dumps(producer))
    manifest["producer"]["receipt"]["sha256"] = file_sha256(producer_path)
    path.write_text(json.dumps(manifest))
    return path


class IndexedDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dense_root = self.root / "dense"
        self.indexed_root = self.root / "indexed"
        self.dense_root.mkdir()
        self.indexed_root.mkdir()
        self.dense_path = dense_fixtures.fixture(self.dense_root)
        self.indexed_path = indexed_fixture(self.indexed_root)

    def load(self, path):
        return BlockDataset(path, expected_sha256=file_sha256(path), allow_synthetic=True)

    def test_every_anchor_tensor_context_position_and_prefix_is_identical(self):
        dense, indexed = self.load(self.dense_path), self.load(self.indexed_path)
        for cid in dense.chains:
            for block in range(len(dense.chains[cid]["anchors"])):
                a = dense.load_block(cid, block, require_teacher=True)
                b = indexed.load_block(cid, block, require_teacher=True)
                for field in (
                    "context_features",
                    "teacher_logits",
                    "labels",
                    "loss_mask",
                    "positions",
                    "predecessor_ids",
                    "input_tokens",
                    "attention_allowed",
                ):
                    np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
                self.assertEqual(a.prefix_tokens, b.prefix_tokens)
                self.assertEqual(a.teacher_prefix_sha256, b.teacher_prefix_sha256)
        self.assertLess(
            (self.indexed_root / "logits0.npy").stat().st_size,
            (self.dense_root / "logits0.npy").stat().st_size,
        )

    def test_all_divergence_and_padding_masks_have_identical_loss_and_gradients(self):
        dense, indexed = self.load(self.dense_path), self.load(self.indexed_path)
        config = BlockQATConfig(
            "dspark",
            8,
            vocab_size=32,
            mask_token_id=31,
            objective="full_probability_l1",
            depth_decay=0.9,
        )
        torch.set_num_threads(1)
        torch.manual_seed(15)
        logits = torch.randn(7, 32)
        # Include every static first divergence (and rejoining predecessor after it).
        # Exhaust all 128 padding masks; both all-padding and divergence-at-zero refuse.
        for block in range(2):
            batches = [
                dataset.load_block("chain0", block, require_teacher=True)
                for dataset in (dense, indexed)
            ]
            for divergence in range(8):
                predecessor = torch.tensor(batches[0].predecessor_ids)
                if divergence < 7:
                    predecessor[divergence] += 1
                for padding in range(128):
                    mask = torch.tensor([bool(padding & (1 << i)) for i in range(7)])
                    results = []
                    for batch in batches:
                        prediction = logits.clone().requires_grad_()
                        tb = SimpleNamespace(
                            labels=torch.tensor(batch.labels),
                            loss_mask=mask.clone(),
                            predecessor_ids=torch.tensor(batch.predecessor_ids),
                            teacher_logits=torch.tensor(batch.teacher_logits),
                        )
                        output = SimpleNamespace(
                            logits=prediction,
                            predecessor_ids=predecessor,
                            conditioning="native_greedy",
                        )
                        if not any(bool(mask[i]) for i in range(min(divergence, 7))):
                            with self.assertRaisesRegex(ValueError, "no exact-prefix"):
                                block_loss(output, tb, config)
                            continue
                        loss, report = block_loss(output, tb, config)
                        loss.backward()
                        results.append((loss.detach(), prediction.grad, report))
                    if results:
                        self.assertTrue(torch.equal(results[0][0], results[1][0]))
                        self.assertTrue(torch.equal(results[0][1], results[1][1]))
                        self.assertEqual(results[0][2], results[1][2])

    def test_malformed_missing_duplicate_unsorted_wrong_and_receipt_maps_refuse(self):
        original = self.indexed_path.read_text()
        for indices in (
            [2, 2] + list(range(4, 16)),
            list(range(3, 17)),
            list(range(2, 15)),
            list(reversed(range(2, 16))),
            [True] + list(range(3, 16)),
            list(range(2, 15)) + [18],
            None,
        ):
            manifest = json.loads(original)
            manifest["chains"][0]["logits_indices"] = indices
            self.indexed_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.load(self.indexed_path)
        self.indexed_path.write_text(original)
        manifest = json.loads(original)
        producer_path = self.indexed_root / manifest["producer"]["receipt"]["path"]
        producer = json.loads(producer_path.read_text())
        producer["chains"]["chain0"]["logits_indices"] = list(range(3, 17))
        producer_path.write_text(json.dumps(producer))
        manifest["producer"]["receipt"]["sha256"] = file_sha256(producer_path)
        self.indexed_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "exact row map"):
            self.load(self.indexed_path)

    def test_exact_union_overlaps_and_current_generation_boundary(self):
        self.assertEqual(block_teacher_indices([2, 5]), list(range(2, 12)))
        for prompt in (1, 2, 255, 256, 257, 1024):
            for new in range(33):
                anchors = list(range(prompt - 1, prompt + new - 7, 7))
                self.assertEqual(generated_block_anchors(prompt, prompt + new), anchors)
                rows = block_teacher_indices(anchors)
                self.assertEqual(len(rows), 7 * (new // 7))
                self.assertEqual(rows, list(range(prompt - 1, prompt - 1 + len(rows))))

    def test_completed_admission_binds_indexed_artifacts(self):
        dataset = self.load(self.indexed_path)
        path = self.root / "admission.json"
        pin = dataset.write_admission(path)
        restored = BlockDataset(
            self.indexed_path,
            expected_sha256=file_sha256(self.indexed_path),
            allow_synthetic=True,
            admission_path=path,
            admission_sha256=pin,
        )
        np.testing.assert_array_equal(
            dataset.load_block("chain0", 1).teacher_logits,
            restored.load_block("chain0", 1).teacher_logits,
        )


class IndexedNativeImportTests(unittest.TestCase):
    def setUp(self):
        self.base = dense_fixtures.NativeRawImportTests(
            "test_raw_producer_complete_prefix_full_vocab_memmap"
        )
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        for chain in self.base.plan["chains"]:
            record = chain["native_receipt"]
            path = Path(record["path"])
            native = json.loads(path.read_text())
            indices = block_teacher_indices(chain["anchors"])
            logits = native["files"]["logits"]
            raw_path = Path(logits["path"])
            dense = np.fromfile(raw_path, dtype=np.float32).reshape(logits["shape"])
            dense[indices].tofile(raw_path)
            logits.update(shape=[len(indices), 32], sha256=file_sha256(raw_path))
            native.update(
                logits_mode="indexed", logits_shape=logits["shape"], logits_indices=indices
            )
            path.write_text(json.dumps(native))
            record["sha256"] = file_sha256(path)
        self.base.plan_path.write_text(json.dumps(self.base.plan))

    def test_raw_import_map_is_native_bound_and_absolute(self):
        path = self.base.import_()
        dataset = BlockDataset(path, expected_sha256=file_sha256(path))
        self.assertEqual(dataset.chains["chain0"]["logits_indices"], list(range(2, 16)))
        np.testing.assert_array_equal(
            dataset.load_block("chain0", 1).teacher_logits, np.arange(18 * 32).reshape(18, 32)[9:16]
        )

    def test_altered_native_row_positions_and_source_refuse(self):
        for field, value in (
            ("logits_indices", list(range(3, 17))),
            ("producer_source_revision", "wrong-source"),
        ):
            record = self.base.plan["chains"][0]["native_receipt"]
            path = Path(record["path"])
            original = path.read_text()
            native = json.loads(original)
            native[field] = value
            path.write_text(json.dumps(native))
            record["sha256"] = file_sha256(path)
            self.base.plan_path.write_text(json.dumps(self.base.plan))
            with self.assertRaises(ValueError):
                self.base.import_()
            shutil.rmtree(self.base.root / "materialized")
            path.write_text(original)
            record["sha256"] = file_sha256(path)


class IndexedClientTests(unittest.TestCase):
    def test_binds_actual_issued_map_and_rejects_native_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "fixture"
            directory.mkdir()
            np.ones((5, 3, 2), np.float32).tofile(directory / "features.f32")
            np.ones((2, 32), np.float32).tofile(directory / "logits.f32")
            request = {
                "id": "fixture",
                "tokens": [1, 2, 3, 4, 5],
                "tap_ids": [0, 1, 2],
                "logits_mode": "indexed",
                "logits_indices": [1, 3],
            }
            response = request | {
                "schema": "block_native_teacher_request_v1",
                "complete": True,
                "features_shape": [5, 3, 2],
                "logits_shape": [2, 32],
            }
            for rows in ([1, 3], [1, 1], [3, 1], [1, 2], [1], [True, 3]):
                teacher = NativeTeacher.__new__(NativeTeacher)
                teacher.root, teacher.max_tokens, teacher.timeout, teacher.closed = (
                    root,
                    5,
                    5,
                    False,
                )
                teacher.ancestry = {}
                teacher._bind_native_runtime = lambda: {"scope": "synthetic test"}
                teacher._save_receipt = lambda _: None
                teacher.process = SimpleNamespace(
                    poll=lambda: None,
                    stdin=io.StringIO(),
                    stdout=io.StringIO(json.dumps(response | {"logits_indices": rows}) + "\n"),
                )
                selector = SimpleNamespace(
                    register=lambda *a: None, select=lambda *a: [True], close=lambda: None
                )
                with patch("selectors.DefaultSelector", return_value=selector):
                    if rows == [1, 3] and all(type(row) is int for row in rows):
                        self.assertEqual(
                            teacher._capture_request(request, None)["logits_indices"], rows
                        )
                    else:
                        with self.assertRaises(ValueError):
                            teacher._capture_request(request, None)

    def test_generated_client_binds_map_to_actual_complete_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "fixture"
            directory.mkdir()
            np.ones((12, 3, 2), np.float32).tofile(directory / "features.f32")
            np.ones((7, 32), np.float32).tofile(directory / "logits.f32")
            request = {
                "id": "fixture",
                "prompt": {"max_new_tokens": 8},
                "tap_ids": [0, 1, 2],
                "logits_mode": "indexed",
                "logits_selection": BLOCK_LOGITS_SELECTION,
            }
            response = request | {
                "schema": "block_native_teacher_request_v1",
                "complete": True,
                "tokens": [1] * 12,
                "prompt_length": 4,
                "generation": {
                    "mode": "native_target_greedy",
                    "max_new_tokens": 8,
                    "generated_tokens": 8,
                    "stop_eog": True,
                },
                "tokenizer": {
                    "add_special": True,
                    "parse_special": True,
                    "implementation": "llama_tokenize",
                },
                "features_shape": [12, 3, 2],
                "logits_shape": [7, 32],
            }
            for rows in (list(range(3, 10)), list(range(4, 11)), list(range(3, 11)), None):
                teacher = NativeTeacher.__new__(NativeTeacher)
                teacher.root, teacher.max_tokens, teacher.timeout, teacher.closed = (
                    root,
                    12,
                    5,
                    False,
                )
                teacher.ancestry = {}
                teacher._bind_native_runtime = lambda: {"scope": "synthetic test"}
                teacher._save_receipt = lambda _: None
                teacher.process = SimpleNamespace(
                    poll=lambda: None,
                    stdin=io.StringIO(),
                    stdout=io.StringIO(json.dumps(response | {"logits_indices": rows}) + "\n"),
                )
                selector = SimpleNamespace(
                    register=lambda *a: None,
                    select=lambda *a: [True],
                    close=lambda: None,
                )
                with patch("selectors.DefaultSelector", return_value=selector):
                    if rows == list(range(3, 10)):
                        self.assertEqual(
                            teacher._capture_request(request, None)["logits_indices"], rows
                        )
                    else:
                        with self.assertRaises(ValueError):
                            teacher._capture_request(request, None)

    def test_replay_request_validation_requires_explicit_positions(self):
        teacher = NativeTeacher.__new__(NativeTeacher)
        teacher.closed, teacher.max_tokens = False, 8
        teacher.process = SimpleNamespace(poll=lambda: None)
        teacher._capture_request = lambda request, _: request
        for rows in (None, [], [2, 2], [3, 2], [True], [8], [-1], [1.0]):
            with self.assertRaises(ValueError):
                teacher.capture_prefix(
                    [1] * 8, [0, 1, 2], logits_mode="indexed", logits_indices=rows
                )
        self.assertEqual(
            teacher.capture_prefix(
                [1] * 8, [0, 1, 2], logits_mode="indexed", logits_indices=[0, 3]
            )["logits_indices"],
            [0, 3],
        )


class IndexedSyntheticTeacher(capture_fixtures.SyntheticNativeTeacher):
    def make(self, tokens, taps, mode, ancestry):
        result = super().make(tokens, taps, "all" if mode == "indexed" else mode, ancestry)
        if mode == "indexed":
            indices = block_teacher_indices(generated_block_anchors(2, len(tokens)))
            item = result["files"]["logits"]
            path = Path(item["path"])
            np.zeros((len(indices), 32), np.float32).tofile(path)
            item.update(shape=[len(indices), 32], sha256=file_sha256(path))
            result.update(
                logits_mode="indexed",
                logits_shape=item["shape"],
                logits_indices=indices,
                logits_selection=BLOCK_LOGITS_SELECTION,
            )
        return result


class IndexedCostTests(unittest.TestCase):
    def setUp(self):
        self.base = capture_fixtures.CaptureTests("test_exact_soft_full_vocab_contract")
        self.base.setUp()
        self.addCleanup(self.base.doCleanups)
        self.base.plan.update(objective="exact_soft", teacher_logits_layout="indexed")
        self.base.persist()

    def test_cost_exact_retained_union_and_temporary_peak(self):
        import capture_nine_model_train_data as capture

        for count in range(8, 17):
            self.base.plan["generation"]["max_new_tokens"] = count
            self.base.persist()
            _, _, cost = capture.prepare_plan(self.base.path, self.base.pin)
            rows = len(block_teacher_indices(generated_block_anchors(8, 8 + count)))
            self.assertEqual(cost["max_block_retained_teacher_rows"], rows)
            self.assertEqual(cost["max_block_peak_teacher_rows"], count + 1)
            self.assertEqual(cost["max_block_request_bytes"], 24 * 5 * 2 * 4 + (count + 1) * 32 * 4)
            self.assertEqual(cost["max_block_retained_teacher_bytes"], rows * 32 * 4)
            self.assertEqual(cost["max_selected_teacher_rows"], rows * 9)

    def test_indexed_synthetic_end_to_end_plan_capture_import(self):
        import capture_nine_model_train_data as capture

        result = capture.run_capture(
            self.base.path,
            self.base.pin,
            self.base.root / "output",
            execute=True,
            teacher_factory=IndexedSyntheticTeacher,
            device_query=lambda: {
                "name": "fixture CUDA",
                "compute_capability": [7, 5],
                "uuid": "fixture",
            },
            rss_query=lambda: 0,
            available_query=lambda: 10**9,
        )
        self.assertEqual(result["status"], "PASS", result["failure"])
        self.assertEqual(result["production_data_status"], "SYNTHETIC_ONLY")
        for pin in result["manifests"].values():
            dataset = BlockDataset(pin["path"], expected_sha256=pin["sha256"])
            self.assertEqual(dataset.chains["chain-000000"]["logits_indices"], list(range(1, 15)))
            self.assertEqual(dataset.load_block("chain-000000", 1).teacher_logits.shape, (7, 32))


class NativeModelFreeProtocolTests(unittest.TestCase):
    def test_cpp_protocol_union_validation_and_dense_compute_flags(self):
        compiler = shutil.which("c++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        native = ROOT / "third_party/llama.cpp"
        source = native / "tools/block-teacher/block-teacher.cpp"
        text = source.read_text()
        self.assertEqual(text.count("batch.logits[i]=true"), 1)
        self.assertNotIn("batch.logits[i]=false", text)
        self.assertIn('if(mode=="all" || (mode=="indexed"', text)
        self.assertLess(
            text.index("const float * row=llama_get_logits_ith"), text.index('if(mode=="all" ||')
        )
        with tempfile.TemporaryDirectory() as tmp:
            cpp = Path(tmp) / "protocol.cpp"
            cpp.write_text("""#include "indexed-logits.h"
#include <cassert>
int main() {
    for (size_t p : {1,2,255,256,257,1024}) for (size_t m=0;m<=32;++m) {
        auto rows=block_teacher::block_indices(p,p+m);
        assert(rows.size()==7*(m/7));
        for(size_t i=0;i<rows.size();++i) assert(rows[i]==p-1+i);
    }
    assert(block_teacher::explicit_indices({0,3,7},8)==std::vector<size_t>({0,3,7}));
    const std::vector<block_teacher::json> invalid={{}, {1,1}, {3,1}, {-1}, {8}, {true}, {1.0}};
    for (const auto & map : invalid) {
        bool refused=false;
        try {block_teacher::explicit_indices(map,8);} catch(const std::exception &) {refused=true;}
        assert(refused);
    }
}
""")
            binary = Path(tmp) / "protocol"
            subprocess.run(
                [
                    compiler,
                    "-std=c++17",
                    "-I" + str(native / "tools/block-teacher"),
                    "-I" + str(native / "vendor"),
                    str(cpp),
                    "-o",
                    str(binary),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run([str(binary)], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()

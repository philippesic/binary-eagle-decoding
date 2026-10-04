"""Synthetic source/API fixtures; no CUDA or real model admission claimed."""

import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import capture_nine_model_train_data as capture

from w1a1_eagle.block_data import BlockDataset, file_sha256


class SyntheticNativeTeacher:
    """Writes actual raw arrays in the native receipt contract for source API tests."""

    instances = []
    fault = None

    def __init__(self, binary, target, output_root, **kwargs):
        self.root = Path(output_root)
        self.root.mkdir()
        self.closed = False
        self.calls = []
        self.source = kwargs["producer_source_revision"]
        self.binary_sha = file_sha256(binary)
        self.target_sha = file_sha256(target)
        self.instances.append(self)

    def make(self, tokens, taps, mode, ancestry):
        directory = self.root / str(len(self.calls))
        directory.mkdir()
        self.calls.append((tokens, taps, mode))
        arrays = {"features": np.ones((len(tokens), len(taps), 2), dtype=np.float32)}
        if mode != "none":
            arrays["logits"] = np.zeros((len(tokens) if mode == "all" else 1, 32), dtype=np.float32)
        files = {}
        for key, array in arrays.items():
            path = directory / (key + ".f32")
            array.tofile(path)
            files[key] = {
                "path": str(path),
                "sha256": file_sha256(path),
                "shape": list(array.shape),
                "dtype": "float32",
            }
        return {
            "schema": "block_native_teacher_request_v1",
            "complete": True,
            "optimizer_updates": 0,
            "teacher_context_reset_between_requests": True,
            "prefix_contract": "teacher_forced_exact_caller_token_ids",
            "kv_type": "F16",
            "target_precision": "F16",
            "tap_ids": list(taps),
            "tokens": tokens,
            "target_sha256": self.target_sha,
            "producer_binary_sha256": self.binary_sha,
            "producer_source_revision": self.source,
            "client_source_sha256": "e" * 64,
            "producer_host": "synthetic-source-api-fixture",
            "hardware": ["fixture CUDA"],
            "executed_result_buffers": ["CUDA0"],
            "target_storage_buffers": {"CUDA0": 1},
            "chain_ancestry": ancestry,
            "features_shape": list(arrays["features"].shape),
            "logits_mode": mode,
            "logits_shape": [len(tokens) if mode == "all" else 1 if mode == "last" else 0, 32],
            "files": files,
            "decode_history": [
                {"offset": 0, "count": 2, "phase": "prefill", "kv_reused_from_same_chain": False}
            ]
            + [
                {
                    "offset": i,
                    "count": 1,
                    "phase": "target_only_greedy",
                    "kv_reused_from_same_chain": True,
                }
                for i in range(2, len(tokens))
            ],
        }

    def generate_capture(
        self,
        *,
        messages=None,
        prompt_text=None,
        template_mode,
        max_new_tokens,
        tap_ids,
        logits_mode,
        chain_ancestry,
    ):
        if self.fault == "stop":
            signal.raise_signal(signal.SIGTERM)
        tokens = list(range(2 + max_new_tokens))
        result = self.make(tokens, tap_ids, logits_mode, chain_ancestry | {"prompt_length": 2})
        result.update(
            prompt_length=2,
            prompt={"template_mode": template_mode},
            prompt_source_sha256=capture.content_hash(
                messages if messages is not None else prompt_text
            ),
            tokenizer_metadata_sha256="c" * 64,
            chat_template_sha256="d" * 64,
            generation={
                "mode": "native_target_greedy",
                "max_new_tokens": max_new_tokens,
                "generated_tokens": max_new_tokens,
                "stop_eog": True,
                "termination": "max_new_tokens",
            },
        )
        if self.fault == "tokenizer":
            result["tokenizer_metadata_sha256"] = "a" * 64
        if self.fault == "hardware":
            result["executed_result_buffers"] = ["CPU"]
        if self.fault == "short":
            result["tokens"] = [0, 1, 2]
        return result

    def capture_prefix(self, tokens, taps, *, logits_mode, chain_ancestry, decode_history):
        self.asserted_history = decode_history
        result = self.make(tokens, taps, logits_mode, chain_ancestry)
        result["decode_history"] = decode_history
        return result

    def close(self):
        self.closed = True
        if self.fault == "close":
            raise RuntimeError("fixture cleanup failure")


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        SyntheticNativeTeacher.instances = []
        SyntheticNativeTeacher.fault = None
        prompts, indices, selection = [], [], []
        for split in capture.SPLITS:
            for domain in capture.DOMAINS:
                i = len(prompts)
                messages = [{"role": "user", "content": f"original {domain} source row {i}"}]
                prompts.append(
                    {"id": f"source-{i}", "messages": messages, "domain": domain, "split": "train"}
                )
                indices.append(
                    {"id": f"source-{i}", "group": f"group-{i}", "domain": domain, "split": "train"}
                )
                selection.append(
                    {
                        "shard": 0,
                        "row": i,
                        "prompt_id": f"source-{i}",
                        "prompt_sha256": capture.content_hash(messages),
                        "domain": domain,
                        "split": split,
                    }
                )
        self.prompts = prompts
        self.indices = indices
        for name, records in (("prompts", prompts), ("index", indices)):
            (self.root / (name + ".jsonl")).write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
        shard = {name: name + ".jsonl" for name in ("prompts", "index")}
        shard.update(
            {
                name + "_sha256": file_sha256(self.root / (name + ".jsonl"))
                for name in ("prompts", "index")
            }
        )
        corpus = capture.write_json(
            self.root / "corpus.json", {"files": {"train": {"shards": [shard]}}}
        )
        runtime = capture.write_json(
            self.root / "runtime.json", {"artifact_kind": "synthetic_fixture"}
        )
        binary, target = self.root / "binary", self.root / "target"
        binary.write_bytes(b"synthetic native binary pin")
        target.write_bytes(b"synthetic target pin")
        self.plan = {
            "schema": "nine_model_train_capture_plan_v1",
            "corpus": corpus,
            "runtime": runtime,
            "vocab_size": 32,
            "target_width": 2,
            "mask_token_id": 31,
            "input_mode": "native_chat",
            "objective": "hard_ce",
            "native": {
                "binary": {"path": str(binary), "sha256": file_sha256(binary)},
                "target": {"path": str(target), "sha256": file_sha256(target)},
                "source_revision": "a" * 40,
                "tokenizer_metadata_sha256": "c" * 64,
                "chat_template_sha256": "d" * 64,
                "gpu_layers": 999,
                "expected_compute_capability": [7, 5],
            },
            "generation": {"max_new_tokens": 16, "algorithm": "greedy"},
            "caps": {
                "max_requests": 15,
                "max_tokens_per_chain": 24,
                "max_request_bytes": 10000,
                "max_shard_bytes": 10000,
                "max_total_bytes": 2000000,
                "max_source_bytes": 100000,
                "max_host_rss_bytes": 1000000,
                "min_free_disk_bytes": 1,
                "request_timeout_seconds": 10,
                "total_timeout_seconds": 100,
                "max_eagle_golden_tokens": 8,
            },
            "selection": selection,
        }
        self.path = self.root / "plan.json"
        self.persist()

    def persist(self):
        self.path.write_text(json.dumps(self.plan))
        self.pin = file_sha256(self.path)

    def run_(self, **kwargs):
        return capture.run_capture(
            self.path,
            self.pin,
            self.root / "output",
            execute=True,
            teacher_factory=SyntheticNativeTeacher,
            device_query=lambda: {
                "name": "fixture CUDA",
                "compute_capability": [7, 5],
                "uuid": "fixture",
            },
            rss_query=lambda: 0,
            **kwargs,
        )

    def test_plan_only_inventory_and_cost_never_start_teacher(self):
        result = capture.run_capture(self.path, self.pin, self.root / "output")
        self.assertEqual(result["status"], "PENDING")
        self.assertEqual(result["selected_prompt_count"], 9)
        self.assertEqual(result["max_native_requests"], 15)
        self.assertEqual(SyntheticNativeTeacher.instances, [])
        self.assertTrue((self.root / "output/original-source-joins.json").exists())
        with self.assertRaises(FileExistsError):
            capture.run_capture(self.path, self.pin, self.root / "output")

    def test_hard_ce_actual_import_shared_bytes_and_bounded_three_tap_goldens(self):
        result = self.run_()
        self.assertEqual(result["status"], "PASS", result["failure"])
        self.assertEqual(result["artifact_kind"], "synthetic_fixture")
        self.assertEqual(result["production_data_status"], "SYNTHETIC_ONLY")
        self.assertTrue(result["producer_closed"])
        datasets = [
            BlockDataset(p["path"], expected_sha256=p["sha256"])
            for p in result["manifests"].values()
        ]
        self.assertEqual(
            datasets[0].chains["chain-000000"]["features"],
            datasets[1].chains["chain-000000"]["features"],
        )
        self.assertIsNone(datasets[0].load_block("chain-000000", 0).teacher_logits)
        np.testing.assert_array_equal(
            datasets[0].load_block("chain-000000", 0).labels, np.arange(2, 9)
        )
        golden = json.loads(Path(result["eagle_goldens"]["path"]).read_text())
        self.assertEqual(len(golden["cases"]), 3)
        self.assertEqual(golden["tap_ids"], [2, 18, 33])
        calls = SyntheticNativeTeacher.instances[0].calls
        self.assertEqual(sum(len(taps) == 3 and mode == "last" for _, taps, mode in calls), 3)
        self.assertTrue(all(len(tokens) <= 8 for tokens, taps, _ in calls if len(taps) == 3))

    def test_exact_soft_full_vocab_contract(self):
        self.plan["objective"] = "exact_soft"
        self.persist()
        result = self.run_()
        self.assertEqual(result["status"], "PASS", result["failure"])
        pin = result["manifests"]["dspark"]
        batch = BlockDataset(pin["path"], expected_sha256=pin["sha256"]).load_block(
            "chain-000000", 0, require_teacher=True
        )
        self.assertEqual(batch.teacher_logits.shape, (7, 32))

    def test_bad_plan_pin_source_final_content_index_domain_counts_and_caps(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            capture.prepare_plan(self.path, "0" * 64)
        for mutate, error in (
            (lambda: self.plan["selection"][0].update(prompt_sha256="0" * 64), "source-content"),
            (lambda: self.plan["selection"][0].update(domain="reasoning"), "domain"),
            (lambda: self.plan["selection"].pop(), "balanced"),
            (lambda: self.plan["caps"].update(max_source_bytes=1), "source cap"),
            (lambda: self.plan["caps"].update(max_total_bytes=1), "total storage"),
            (lambda: self.plan["caps"].update(max_request_bytes=1), "request/shard"),
            (lambda: self.plan["caps"].update(max_requests=1), "count cap"),
            (lambda: self.plan.update(input_mode="raw_text"), "raw prompt_text"),
        ):
            original = json.loads(json.dumps(self.plan))
            mutate()
            self.persist()
            with self.assertRaisesRegex((ValueError, MemoryError), error):
                capture.prepare_plan(self.path, self.pin)
            self.plan = original
            self.persist()

    def test_original_source_split_and_index_join_refuse(self):
        for field, value in (("split", "final"), ("id", "other")):
            original = self.indices[0].copy()
            self.indices[0][field] = value
            index = self.root / "index.jsonl"
            index.write_text("".join(json.dumps(r) + "\n" for r in self.indices))
            corpus_path = self.root / "corpus.json"
            corpus = json.loads(corpus_path.read_text())
            corpus["files"]["train"]["shards"][0]["index_sha256"] = file_sha256(index)
            corpus_path.write_text(json.dumps(corpus))
            self.plan["corpus"]["sha256"] = file_sha256(corpus_path)
            self.persist()
            with self.assertRaisesRegex(ValueError, "original TRAIN|join differs"):
                capture.prepare_plan(self.path, self.pin)
            self.indices[0] = original

    def test_timeout_and_host_memory_refuse_without_teacher(self):
        ticks = iter((0, 200, 200))
        result = self.run_(clock=lambda: next(ticks))
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["failure"]["type"], "TimeoutError")
        self.assertEqual(SyntheticNativeTeacher.instances, [])

    def test_host_rss_and_free_disk_refuse_before_native_start(self):
        from unittest.mock import patch

        for reason in ("RSS", "disk"):
            out = self.root / f"resource-{reason}"
            with patch.object(
                capture.shutil,
                "disk_usage",
                return_value=type("Disk", (), {"free": 0 if reason == "disk" else 10**9})(),
            ):
                result = capture.run_capture(
                    self.path,
                    self.pin,
                    out,
                    execute=True,
                    teacher_factory=SyntheticNativeTeacher,
                    device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                    rss_query=lambda: 10**9 if reason == "RSS" else 0,
                )
            self.assertEqual(result["status"], "FAIL")
            self.assertIn(reason, result["failure"]["message"])
            self.assertEqual(SyntheticNativeTeacher.instances, [])

    def test_golden_prompt_boundary_not_fabricated_or_truncated(self):
        self.plan["caps"]["max_eagle_golden_tokens"] = 1
        self.persist()
        result = self.run_()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("original prompt and continuation", result["failure"]["message"])
        self.assertTrue(SyntheticNativeTeacher.instances[-1].closed)

    def test_decode_history_preserves_native_partition_and_refuses_gaps(self):
        history = [
            {"offset": 0, "count": 2, "phase": "prefill", "kv_reused_from_same_chain": False},
            {
                "offset": 2,
                "count": 1,
                "phase": "target_only_greedy",
                "kv_reused_from_same_chain": True,
            },
        ]
        self.assertEqual(capture.prefix_history(history, 3), history)
        self.assertEqual(capture.prefix_history(history, 1)[0]["count"], 1)
        history[1]["offset"] = 4
        with self.assertRaisesRegex(ValueError, "contiguous"):
            capture.prefix_history(history, 3)

    def test_duplicate_original_prompt_and_group_refuse(self):
        self.plan["selection"][1] = dict(self.plan["selection"][0])
        self.persist()
        with self.assertRaisesRegex(ValueError, "overlap"):
            capture.prepare_plan(self.path, self.pin)

    def test_raw_text_uses_literal_original_utf8_and_actual_native_api(self):
        import hashlib

        self.plan["input_mode"] = "raw_text"
        for prompt, choice in zip(self.prompts, self.plan["selection"], strict=True):
            prompt["prompt_text"] = "literal original raw prompt \u00e9 " + prompt["id"]
            choice["prompt_sha256"] = hashlib.sha256(prompt["prompt_text"].encode()).hexdigest()
        source = self.root / "prompts.jsonl"
        source.write_text("".join(json.dumps(r) + "\n" for r in self.prompts))
        corpus_path = self.root / "corpus.json"
        corpus = json.loads(corpus_path.read_text())
        corpus["files"]["train"]["shards"][0]["prompts_sha256"] = file_sha256(source)
        corpus_path.write_text(json.dumps(corpus))
        self.plan["corpus"]["sha256"] = file_sha256(corpus_path)
        self.persist()
        result = self.run_()
        self.assertEqual(result["status"], "PASS", result["failure"])

    def test_generation_bad_tokenizer_actual_hardware_stop_and_cleanup(self):
        for fault, error in (
            ("tokenizer", "history"),
            ("hardware", "CUDA"),
            ("stop", "STOP"),
            ("close", "cleanup"),
        ):
            SyntheticNativeTeacher.fault = fault
            out = self.root / f"out-{fault}"
            result = capture.run_capture(
                self.path,
                self.pin,
                out,
                execute=True,
                teacher_factory=SyntheticNativeTeacher,
                device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                rss_query=lambda: 0,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertIn(error, result["failure"]["message"])
            self.assertTrue(SyntheticNativeTeacher.instances[-1].closed)
            self.assertTrue((out / "capture-report.json").exists())


if __name__ == "__main__":
    unittest.main()

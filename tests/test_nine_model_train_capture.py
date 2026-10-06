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
            "prefix_freshness": "caller_current_student_prefix",
            "kv_type": "F16",
            "target_precision": "F16",
            "tap_ids": list(taps),
            "tokens": tokens,
            "target_sha256": self.target_sha,
            "producer_binary_sha256": self.binary_sha,
            "producer_source_revision": self.source,
            "client_source_sha256": file_sha256(self.root.parent.parent / "client.py"),
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
        max_prompt_tokens,
        tap_ids,
        logits_mode,
        chain_ancestry,
    ):
        if self.fault == "stop":
            signal.raise_signal(signal.SIGTERM)
        tokens = list(range(2 + max_new_tokens))
        result = self.make(tokens, tap_ids, logits_mode, chain_ancestry | {"prompt_length": 2})
        result.update(
            prefix_contract="native_tokenized_prompt_then_target_only_greedy",
            prefix_freshness="native_generated_chain",
            prompt_length=2,
            prompt={
                "template_mode": template_mode,
                "max_prompt_tokens": max_prompt_tokens,
                "max_new_tokens": max_new_tokens,
                **({"messages": messages} if messages is not None else {"text": prompt_text}),
            },
            prompt_source_sha256=capture.content_hash(
                messages if messages is not None else prompt_text
            ),
            tokenizer_metadata_sha256="c" * 64,
            chat_template="fixture template",
            chat_template_sha256=capture.content_hash("fixture template"),
            target_chat_template_sha256=capture.content_hash("fixture template"),
            rendered_prompt="fixture rendered prompt",
            rendered_prompt_sha256=capture.content_hash("fixture rendered prompt"),
            tokenizer={
                "implementation": "llama_tokenize",
                "add_special": True,
                "parse_special": True,
            },
            generation={
                "mode": "native_target_greedy",
                "max_new_tokens": max_new_tokens,
                "generated_tokens": max_new_tokens,
                "stop_eog": True,
                "termination": "max_new_tokens",
            },
        )
        if template_mode == "raw_text":
            result.update(
                chat_template="",
                chat_template_sha256=capture.content_hash(""),
                rendered_prompt=prompt_text,
                rendered_prompt_sha256=capture.content_hash(prompt_text),
            )
        if self.fault == "prompt":
            result["prompt_length"] = max_prompt_tokens + 1
        if self.fault == "client":
            result["client_source_sha256"] = "e" * 64
        if self.fault == "tokenizer":
            result["tokenizer_metadata_sha256"] = "a" * 64
        if self.fault == "hardware":
            result["executed_result_buffers"] = ["CPU"]
        if self.fault == "short":
            result["tokens"] = [0, 1, 2]
        if self.fault == "generated_contract":
            result["prefix_contract"] = "teacher_forced_exact_caller_token_ids"
        if self.fault == "rendered":
            result["rendered_prompt"] += " changed"
        if self.fault == "prompt_payload":
            result["prompt"]["messages"][0] = {"role": "user", "content": "changed original"}
        if self.fault == "tokenizer_flags":
            result["tokenizer"]["parse_special"] = False
        return result

    def capture_prefix(self, tokens, taps, *, logits_mode, chain_ancestry, decode_history):
        self.asserted_history = decode_history
        result = self.make(tokens, taps, logits_mode, chain_ancestry)
        result["decode_history"] = decode_history
        if self.fault == "replay_contract":
            result["prefix_contract"] = "native_tokenized_prompt_then_target_only_greedy"
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
        client = self.root / "client.py"
        client.write_bytes(b"synthetic injected client source fixture")
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
                "client_source": {"path": str(client), "sha256": file_sha256(client)},
                "tokenizer_metadata_sha256": "c" * 64,
                "chat_template_sha256": capture.content_hash("fixture template"),
                "gpu_layers": 999,
                "expected_compute_capability": [7, 5],
            },
            "generation": {"max_new_tokens": 16, "algorithm": "greedy"},
            "caps": {
                "max_requests": 15,
                "max_tokens_per_chain": 24,
                "max_prompt_tokens": 8,
                "max_request_bytes": 10000,
                "max_shard_bytes": 10000,
                "max_total_bytes": 2000000,
                "max_source_bytes": 100000,
                "max_source_row_bytes": 4096,
                "max_host_rss_bytes": 1000000,
                "min_host_available_bytes": 1000000,
                "min_free_disk_bytes": 1,
                "request_timeout_seconds": 10,
                "total_timeout_seconds": 100,
                "max_eagle_golden_tokens": 8,
            },
            "selection": selection,
        }
        runtime_path = self.root / "runtime.json"
        native = self.plan["native"]
        runtime_path.write_text(
            json.dumps(
                {
                    "schema": "nine_model_train_capture_runtime_v1",
                    "binary_sha256": native["binary"]["sha256"],
                    "target_sha256": native["target"]["sha256"],
                    "native_source_revision": native["source_revision"],
                    "teacher_client_sha256": native["client_source"]["sha256"],
                    "tokenizer_metadata_sha256": native["tokenizer_metadata_sha256"],
                    "chat_template_sha256": native["chat_template_sha256"],
                }
            )
        )
        self.plan["runtime"]["sha256"] = file_sha256(runtime_path)
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
            available_query=lambda: 10**9,
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
        self.assertEqual(result["min_observed_host_available_bytes"], 10**9)
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
                    available_query=lambda: 10**9,
                )
            self.assertEqual(result["status"], "FAIL")
            self.assertIn(reason, result["failure"]["message"])
            self.assertEqual(SyntheticNativeTeacher.instances, [])

    def test_import_audit_resource_checks_do_not_rescan_retained_tree(self):
        from unittest.mock import patch

        original_import = capture.import_capture_plan
        original_tree = capture.tree_bytes
        counts = {"chain": 0, "audit": 0}

        def imported(*args, **kwargs):
            full, lightweight = kwargs["budget_check"], kwargs["audit_budget_check"]

            def chain_boundary():
                counts["chain"] += 1
                full()

            def audit_slice():
                counts["audit"] += 1
                before = tree.call_count
                lightweight()
                self.assertEqual(tree.call_count, before)

            return original_import(
                *args,
                **(kwargs | {"budget_check": chain_boundary, "audit_budget_check": audit_slice}),
            )

        with (
            patch.object(capture, "tree_bytes", wraps=original_tree) as tree,
            patch.object(capture, "import_capture_plan", side_effect=imported),
            patch.object(capture, "host_available_bytes", return_value=10**9) as available,
        ):
            # Override run_'s fixed available_query to count every resource check.
            result = capture.run_capture(
                self.path,
                self.pin,
                self.root / "output",
                execute=True,
                teacher_factory=SyntheticNativeTeacher,
                device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                rss_query=lambda: 0,
                available_query=available,
            )
        self.assertEqual(result["status"], "PASS", result["failure"])
        self.assertEqual(counts["chain"], 2 * result["selected_prompt_count"])
        self.assertGreater(counts["audit"], counts["chain"] * 3)
        # Every non-audit resource check still performs the complete tree scan;
        # final retained-byte reporting adds one scan without a resource check.
        self.assertEqual(tree.call_count + counts["audit"], available.call_count + 1)

    def test_memory_and_free_disk_floors_reject_inside_read_only_audit(self):
        from unittest.mock import patch

        original_import = capture.import_capture_plan
        for reason in ("available", "RSS", "disk"):
            state = {"inside_audit": False, "audit_checks": 0}

            def imported(*args, **kwargs):
                lightweight = kwargs["audit_budget_check"]

                def audit_slice():
                    state["audit_checks"] += 1
                    state["inside_audit"] = state["audit_checks"] >= 3
                    try:
                        lightweight()
                    finally:
                        state["inside_audit"] = False

                return original_import(*args, **(kwargs | {"audit_budget_check": audit_slice}))

            def failing(which):
                return reason == which and state["inside_audit"]

            with (
                patch.object(capture, "import_capture_plan", side_effect=imported),
                patch.object(
                    capture.shutil,
                    "disk_usage",
                    side_effect=lambda _: type(
                        "Disk", (), {"free": 0 if failing("disk") else 10**9}
                    )(),
                ),
            ):
                output = self.root / f"audit-floor-{reason}"
                result = capture.run_capture(
                    self.path,
                    self.pin,
                    output,
                    execute=True,
                    teacher_factory=SyntheticNativeTeacher,
                    device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                    rss_query=lambda: 10**9 if failing("RSS") else 0,
                    available_query=lambda: 0 if failing("available") else 10**9,
                )
            self.assertGreaterEqual(state["audit_checks"], 3)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(result["failure"]["type"], "MemoryError")
            expected = "MemAvailable" if reason == "available" else reason
            self.assertIn(expected, result["failure"]["message"])
            self.assertTrue(SyntheticNativeTeacher.instances[-1].closed)
            self.assertFalse((output / "dspark/completed-admission.json").exists())

    def test_golden_prompt_boundary_not_fabricated_or_truncated(self):
        self.plan["caps"]["max_eagle_golden_tokens"] = 1
        self.persist()
        result = self.run_()
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("original prompt and continuation", result["failure"]["message"])
        self.assertTrue(SyntheticNativeTeacher.instances[-1].closed)

    def test_runtime_inventory_rejects_stale_source_client_and_arbitrary_json(self):
        path = self.root / "runtime.json"
        original = json.loads(path.read_text())
        for key in ("schema", "teacher_client_sha256", "native_source_revision", "target_sha256"):
            current = original | {key: "stale"}
            path.write_text(json.dumps(current))
            self.plan["runtime"]["sha256"] = file_sha256(path)
            self.persist()
            with self.assertRaisesRegex(ValueError, "runtime schema/native/client"):
                capture.prepare_plan(self.path, self.pin)
        path.write_text(json.dumps({"artifact_kind": "production"}))
        self.plan["runtime"]["sha256"] = file_sha256(path)
        self.persist()
        with self.assertRaisesRegex(ValueError, "runtime schema/native/client"):
            capture.prepare_plan(self.path, self.pin)

    def test_production_current_client_source_must_match_plan_before_start(self):
        import types
        from unittest.mock import patch

        with patch.dict(
            sys.modules,
            {
                "capture_block_qat_teacher": types.SimpleNamespace(
                    NativeTeacher=SyntheticNativeTeacher
                )
            },
        ):
            result = capture.run_capture(
                self.path,
                self.pin,
                self.root / "source-drift",
                execute=True,
                device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                rss_query=lambda: 0,
                available_query=lambda: 10**9,
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "current production NativeTeacher client source", result["failure"]["message"]
        )
        self.assertEqual(SyntheticNativeTeacher.instances, [])

    def test_memavailable_floor_missing_untyped_or_low_refuses_capture(self):
        for name, query in (
            ("low", lambda: 1),
            ("untyped", lambda: "1000000000"),
            ("missing", lambda: (_ for _ in ()).throw(FileNotFoundError("no MemAvailable"))),
        ):
            result = capture.run_capture(
                self.path,
                self.pin,
                self.root / f"available-{name}",
                execute=True,
                teacher_factory=SyntheticNativeTeacher,
                device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                rss_query=lambda: 0,
                available_query=query,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertIn("MemAvailable", result["failure"]["message"])
            self.assertEqual(SyntheticNativeTeacher.instances, [])

    def test_linux_memavailable_parser_requires_explicit_kb_field(self):
        from unittest.mock import patch

        with patch.object(Path, "read_text", return_value="MemTotal: 8 kB\nMemAvailable: 4 kB\n"):
            self.assertEqual(capture.host_available_bytes(), 4096)
        for contents in ("MemTotal: 8 kB", "MemAvailable: 4 MB", "MemAvailable: nan kB"):
            with patch.object(Path, "read_text", return_value=contents):
                with self.assertRaisesRegex(ValueError, "MemAvailable"):
                    capture.host_available_bytes()

    def test_streaming_source_skips_unselected_json_and_bounds_each_row(self):
        path = self.root / "stream-source.jsonl"
        path.write_bytes(
            b"not JSON unselected\n" + b'{"id":"selected"}\n' + b"invalid unread tail\n"
        )
        self.assertEqual(capture.selected_jsonl_rows(path, {1}, 64), {1: {"id": "selected"}})
        path.write_bytes(b"x" * 65 + b"\n" + b'{"id":"selected"}\n')
        with self.assertRaisesRegex(MemoryError, "row storage cap"):
            capture.selected_jsonl_rows(path, {1}, 64)
        self.plan["caps"]["max_source_row_bytes"] = 1
        self.persist()
        with self.assertRaisesRegex(MemoryError, "row storage cap"):
            capture.prepare_plan(self.path, self.pin)

    def test_old_producer_without_prompt_cap_refuses_before_model_construct(self):
        import types
        from unittest.mock import patch

        class MissingPromptTokenGuard:
            def __init__(self, *args, **kwargs):
                raise AssertionError("old producer model must never load")

            def generate_capture(self, **kwargs):
                raise AssertionError("old producer must never decode")

        client = Path(__file__).resolve()
        self.plan["native"]["client_source"] = {"path": str(client), "sha256": file_sha256(client)}
        runtime_path = self.root / "runtime.json"
        runtime = json.loads(runtime_path.read_text())
        runtime["teacher_client_sha256"] = file_sha256(client)
        runtime_path.write_text(json.dumps(runtime))
        self.plan["runtime"]["sha256"] = file_sha256(runtime_path)
        self.persist()
        with patch.dict(
            sys.modules,
            {
                "capture_block_qat_teacher": types.SimpleNamespace(
                    NativeTeacher=MissingPromptTokenGuard
                )
            },
        ):
            result = capture.run_capture(
                self.path,
                self.pin,
                self.root / "old-prompt-api",
                execute=True,
                device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                rss_query=lambda: 0,
                available_query=lambda: 10**9,
            )
        self.assertEqual(result["status"], "FAIL")
        self.assertIn(
            "native producer lacks prompt token cap before decoding", result["failure"]["message"]
        )
        self.assertIsNone(result["producer_closed"])

    def test_prompt_and_generation_caps_fit_the_native_chain_bound(self):
        self.plan["caps"]["max_prompt_tokens"] = 9
        self.persist()
        with self.assertRaisesRegex(ValueError, "prompt token cap plus generation cap"):
            capture.prepare_plan(self.path, self.pin)

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
            ("prompt", "history"),
            ("client", "producer proof"),
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
                available_query=lambda: 10**9,
            )
            self.assertEqual(result["status"], "FAIL")
            self.assertIn(error, result["failure"]["message"])
            self.assertTrue(SyntheticNativeTeacher.instances[-1].closed)
            self.assertTrue((out / "capture-report.json").exists())

    def test_generated_and_replay_contracts_and_native_prompt_bindings_are_distinct(self):
        for fault, error in (
            ("generated_contract", "generated prefix"),
            ("replay_contract", "replay prefix"),
            ("rendered", "history contract"),
            ("prompt_payload", "history contract"),
            ("tokenizer_flags", "history contract"),
        ):
            SyntheticNativeTeacher.fault = fault
            result = capture.run_capture(
                self.path,
                self.pin,
                self.root / fault,
                execute=True,
                teacher_factory=SyntheticNativeTeacher,
                device_query=lambda: {"name": "fixture CUDA", "compute_capability": [7, 5]},
                rss_query=lambda: 0,
                available_query=lambda: 10**9,
            )
            self.assertEqual(result["status"], "FAIL", fault)
            self.assertIn(error, result["failure"]["message"])
            self.assertTrue(SyntheticNativeTeacher.instances[-1].closed)

    def test_importer_rejects_generated_receipt_relabeling_and_runtime_drift(self):
        result = self.run_()
        self.assertEqual(result["status"], "PASS", result["failure"])
        plan = json.loads((self.root / "output/dspark-capture-plan.json").read_text())
        original = json.loads(Path(plan["chains"][0]["native_receipt"]["path"]).read_text())
        for fault in ("prefix_contract", "client_source_sha256", "tokenizer_metadata_sha256"):
            receipt = dict(original)
            receipt[fault] = (
                "teacher_forced_exact_caller_token_ids" if fault == "prefix_contract" else "e" * 64
            )
            pin = capture.write_json(self.root / (fault + "-receipt.json"), receipt)
            modified = dict(plan, chains=[dict(plan["chains"][0], native_receipt=pin)])
            plan_pin = capture.write_json(self.root / (fault + "-plan.json"), modified)
            with self.assertRaisesRegex(ValueError, "replay prefix|pinned runtime"):
                capture.import_capture_plan(
                    plan_pin["path"],
                    expected_sha256=plan_pin["sha256"],
                    output_dir=self.root / (fault + "-import"),
                )

    def test_eog_on_final_allowed_token_and_exact_prompt_history_boundary(self):
        result = self.run_()
        self.assertEqual(result["status"], "PASS", result["failure"])
        receipt = json.loads((self.root / "output/receipts/000000-block.json").read_text())
        record = capture.prepare_plan(self.path, self.pin)[1][0]
        device = {"name": "fixture CUDA", "compute_capability": [7, 5]}
        receipt["generation"]["termination"] = "eog"
        capture.validate_generated(receipt, record, self.plan, device)
        receipt["decode_history"][1]["phase"] = "prefill"
        with self.assertRaisesRegex(ValueError, "prompt boundary"):
            capture.validate_generated(receipt, record, self.plan, device)


if __name__ == "__main__":
    unittest.main()

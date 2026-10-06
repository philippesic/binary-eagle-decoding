"""CPU-only raw-join and lifecycle fixtures; no CUDA/server/SSH claims."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
with patch.object(sys, "argv", ["diagnostic-test", "--helper-checkout", str(ROOT)]):
    spec = importlib.util.spec_from_file_location("eagle_raw_probe", ROOT / "scripts/diagnose_eagle_verifier_parity.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return probe.endpoint.pin(path)


def append(path, values):
    with path.open("a") as stream:
        for row in values:
            stream.write(json.dumps(row) + "\n")


def prompts(index=2):
    return [{"id": probe.CASE if i == index else f"original:{i}",
             "messages": [{"role": "user", "content": f"original prompt{i}"}]} for i in range(24)]


def output(arm):
    ids = list(range(103))
    ids[98] = 777 if arm == "target_only" else 778
    return ids


def points(arm, task=1):
    ids = output(arm)
    rounds = [{"task_id": task, "round_index": i, "status": "complete", "replay": False,
               "emitted_token_ids": [t], "verified_token_ids": [t], "proposed_token_ids": [888],
               "n_accepted": 0} for i, t in enumerate(ids[1:])]
    trace = []
    for p in probe.POSITIONS:
        other = 778 if ids[p] != 778 else 777
        trace.append({"schema": "w1ax_verify_logits_v1", "task_id": task, "round_index": p,
                      "generated_position": p, "row": 0, "base_generated_position": p,
                      "verifier_prefix_draft_ids": [], "mode": "target_only" if arm == "target_only" else "speculative_verify",
                      "replay": False, "sampled": True, "emitted": True, "has_logits": True,
                      "sampled_token_id": ids[p], "emitted_token_id": ids[p], "nan_logit_count": 0,
                      "raw_top5": [{"token_id": t, "logit": 20 - i * 2.0} for i, t in
                                   enumerate([ids[p], other, 900, 901, 902])]})
    return trace, rounds if arm != "target_only" else []


def response(arm):
    return {"generated_token_ids": output(arm), "__verbose": {
        "prompt": "actual rendered prompt", "generation_settings": {"backend_sampling": False}}}


def measurement(arm):
    return {"generated_token_ids": output(arm), "finish_reason": "length",
            "completion_sha256": "a" * 64, "completion_tokens": 103}


class JoinTests(unittest.TestCase):
    def test_original_warmups_and_entire_preceding_order(self):
        seq = probe.request_sequence(prompts(23))
        self.assertEqual(len(seq), 26)
        self.assertEqual([(w, i) for w, i, _ in seq], [(True, 0), (True, 1)] + [(False, i) for i in range(24)])
        self.assertEqual(probe.request_sequence(prompts(2))[-1][2]["id"], probe.CASE)

    def test_missing_or_duplicated_case_rejected(self):
        for values in (prompts()[:23], [{**r, "id": probe.CASE} for r in prompts()], prompts(-1)):
            with self.assertRaises(ValueError):
                probe.request_sequence(values)

    def test_trace_filter_keeps_only_reached_emitted_nonreplay_task(self):
        trace, _ = points("eagle_q4")
        good = trace[2]
        noise = [{**good, key: val} for key, val in (("task_id", 2), ("sampled", False), ("emitted", False), ("replay", True))]
        self.assertEqual(probe.raw_point(trace + noise, 1, 98, output("eagle_q4")), good)
        with self.assertRaisesRegex(ValueError, "one sampled"):
            probe.raw_point(trace + [good], 1, 98, output("eagle_q4"))

    def test_nan_nonfinite_wrong_argmax_and_output_fail(self):
        for mutation in ("nan", "infinite", "id", "sorted", "duplicate"):
            trace, _ = points("eagle_q4")
            row = trace[2]
            if mutation == "nan": row["nan_logit_count"] = 1
            elif mutation == "infinite": row["raw_top5"][1]["logit"] = float("inf")
            elif mutation == "id": row["sampled_token_id"] = 999
            elif mutation == "sorted": row["raw_top5"][1]["logit"] = 50
            else: row["raw_top5"][1]["token_id"] = row["raw_top5"][0]["token_id"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                probe.raw_point(trace, 1, 98, output("eagle_q4"))

    def test_causal_round_stream_and_wrong_retained_prefix(self):
        trace, rounds = points("eagle_q4")
        self.assertEqual(probe.causal(trace[2], rounds, output("eagle_q4"))["base_generated_position"], 98)
        for field, value in (("verifier_prefix_draft_ids", [123]), ("base_generated_position", 97), ("round_index", 96)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "proposal prefix"):
                probe.causal({**trace[2], field: value}, rounds, output("eagle_q4"))
        broken = copy.deepcopy(rounds)
        broken[3]["emitted_token_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "complete output"):
            probe.causal(trace[2], broken, output("eagle_q4"))

    def test_unaccepted_deeper_row_rejected(self):
        trace, rounds = points("eagle_q4")
        rounds[96]["emitted_token_ids"] += rounds[97]["emitted_token_ids"]
        rounds[96]["verified_token_ids"] += rounds[97]["verified_token_ids"]
        del rounds[97]
        bad = {**trace[2], "row": 1, "base_generated_position": 97, "round_index": 97,
               "verifier_prefix_draft_ids": [97]}
        with self.assertRaisesRegex(ValueError, "proposal prefix"):
            probe.causal(bad, rounds, output("eagle_q4"))

    def test_sampler_enable_flags_and_response_mode_fail(self):
        probe.sampler_disabled(["binary", "--spec-draft-backend-sampling"], {})
        for flags, body in ((["-bs"], {}), (["--backend-sampling=true"], {}), ([], {"backend_sampling": False})):
            with self.assertRaises(ValueError): probe.sampler_disabled(flags, body)
        r = response("eagle_q4")
        self.assertEqual(probe.response_join(r, output("eagle_q4")), "actual rendered prompt")
        r["__verbose"]["generation_settings"]["backend_sampling"] = True
        with self.assertRaisesRegex(ValueError, "runtime target"):
            probe.response_join(r, output("eagle_q4"))

    def test_summary_reports_failed_parity_without_numeric_gate(self):
        arms, original = {}, {}
        for arm in probe.ARMS:
            trace, rounds = points(arm)
            row = {"warmup": False, "ordinal": 2, "prompt_id": probe.CASE, "measurement": measurement(arm),
                   "actual_rendered_prompt": "actual rendered prompt",
                   "points": [{"position": p["generated_position"], "raw": p, "causal": probe.causal(p, rounds, output(arm))}
                              for p in trace]}
            arms[arm] = {"records": [row]}
            original[arm, 2] = {**measurement(arm), "actual_rendered_prompt": "actual rendered prompt"}
        summary = probe.summarize(arms, original)
        self.assertEqual(summary["strict_target_only_parity"], "FAILED")
        self.assertEqual(len(summary["shared_first98_generated_ids"]), 98)
        self.assertEqual(summary["margins_at98"]["target_only"]["signed_a8_minus_target_margin"], -2)
        self.assertNotIn("numeric_gate", summary)
        arms["eagle_q4"]["records"][0]["measurement"]["generated_token_ids"][97] = 555
        with self.assertRaisesRegex(ValueError, "complete output"):
            probe.summarize(arms, original)


class NativeSourceTests(unittest.TestCase):
    def fixture(self, root):
        sources = {"common/common.h": "struct common_params_sampling {\n bool backend_sampling = false;\n};\n",
                   "common/arg.cpp": "frozen flag parser",
                   "tools/server/server-schema.cpp": 'field_bool("backend_sampling", params.sampling.backend_sampling)',
                   "tools/server/server-context.cpp": "W1AX_VERIFY_TRACE_JSONL W1AX_VERIFY_TRACE_POSITIONS W1AX_ROUND_TRACE_JSONL use_backend_sampling &= !need_pre_sample_logits"}
        for name, text in sources.items():
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=CPU Fixture", "-c", "user.email=fixture@example.invalid",
                        "commit", "-qm", "Fixture"], check=True)
        return probe.git(root, "rev-parse", "HEAD").decode().strip()

    def test_source_attestation_and_posthash_loss_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sha = self.fixture(root)
            with patch.object(probe, "NATIVE_COMMIT", sha):
                evidence = probe.native_source(root)
                self.assertFalse(evidence["target_backend_sampling_default"])
                path = root / "common/common.h"
                path.write_text(path.read_text().replace("false", "true"))
                with self.assertRaisesRegex(ValueError, "source changed"): probe.native_source(root)
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    probe.endpoint.Files().check(evidence["source"]["common/common.h"])

    def test_wrong_checkout_commit_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.fixture(root)
            with self.assertRaisesRegex(ValueError, "frozen cc9"): probe.native_source(root)


class ReferenceTests(unittest.TestCase):
    def fixture(self, root):
        values = prompts()
        sequence = probe.request_sequence(values)
        protocol = {"max_output_tokens": 128, "seed": 42}
        args = SimpleNamespace(plan=root / "plan.json", reference_progress=root / "progress.json",
                               reference_ancestry=root / "ancestry.json")
        write(args.plan, {"fixture": "original plan"})
        checkpoint = {"records": {"publication": {"fixture": "paused"}},
                      "existing_export": {"model": {"fixture": "original trained bytes"}}}
        ancestry_pin = write(args.reference_ancestry, {"plan": probe.endpoint.pin(args.plan), "checkpoint": checkpoint})
        records, diag = [], []
        for rep in range(6):
            for arm in probe.ARMS:
                for i, prompt in enumerate(values):
                    folder = root / ("diagnostic" if rep == 5 else f"rep-{rep}") / arm / f"prompt-{i:04d}"
                    raw = write(folder / "measurement.json", measurement(arm))
                    write(folder / "request.json", probe.endpoint.request_body(probe.config(protocol), prompt))
                    write(folder / "response.json", response(arm))
                    row = {"cell": arm, "prompt_id": prompt["id"], "repetition": 0 if rep == 5 else rep,
                           "generated_token_ids": output(arm), "raw_result": raw}
                    (diag if rep == 5 else records).append(row)
        progress_pin = write(args.reference_progress, {"clean_records": records, "diagnostic_records": diag})
        args.reference_progress_sha256 = progress_pin["sha256"]
        args.reference_ancestry_sha256 = ancestry_pin["sha256"]
        return args, checkpoint, sequence, protocol

    def test_complete_five_reps_and_diagnostic_join(self):
        with tempfile.TemporaryDirectory() as temp:
            args, checkpoint, sequence, protocol = self.fixture(Path(temp))
            original, pins = probe.reference(args, {}, probe.endpoint.Files(), checkpoint, sequence, protocol)
            self.assertEqual(len(original), 9)
            self.assertEqual(original["target_only", 2]["generated_token_ids"][98], 777)
            self.assertEqual(len(pins), 2 + 9 * 6 * 3)

    def test_progress_and_raw_result_hash_loss_fail_closed(self):
        for mutation in ("progress", "raw"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                args, checkpoint, sequence, protocol = self.fixture(Path(temp))
                target = args.reference_progress if mutation == "progress" else Path(
                    probe.paused.read(args.reference_progress)["clean_records"][0]["raw_result"]["path"])
                target.write_text("changed bytes")
                with self.assertRaisesRegex(ValueError, "artifact changed"):
                    probe.reference(args, {}, probe.endpoint.Files(), checkpoint, sequence, protocol)

    def test_wrong_request_or_changed_run02_export_rejected(self):
        for mutation in ("request", "model"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                args, checkpoint, sequence, protocol = self.fixture(Path(temp))
                if mutation == "model": checkpoint["existing_export"]["model"]["fixture"] = "different model"
                else:
                    path = Path(probe.paused.read(args.reference_progress)["clean_records"][0]["raw_result"]["path"])
                    write(path.parent / "request.json", {"messages": "wrong prefix"})
                with self.assertRaises(ValueError):
                    probe.reference(args, {}, probe.endpoint.Files(), checkpoint, sequence, protocol)


class LifecycleTests(unittest.TestCase):
    def fixture(self, root, failure=None):
        seq = probe.request_sequence(prompts())
        plan = {"gpu_uuid": "GPU-fixture", "runtime": {"binary": {"path": "/frozen/server"}},
                "target": {"path": "/frozen/target"}, "environment": {}}
        protocol = {"max_output_tokens": 128, "seed": 42, "startup_wall_seconds": 90,
                    "context_tokens": 2048, "batch_tokens": 512, "microbatch_tokens": 512,
                    "draft_lengths": {"eagle": 5}}
        proc = SimpleNamespace(pid=123, poll=lambda: None)
        calls, active = [], {"requests": 0}

        def execute(url, body, timeout, directory):
            self.assertEqual(active["requests"], 0)
            active["requests"] += 1
            try:
                if failure == "request": raise RuntimeError("fake request failure")
                write(directory / "request.json", body)
                write(directory / "response.json", response("eagle_q4"))
                write(directory / "measurement.json", measurement("eagle_q4"))
                calls.append(body["messages"][0]["content"])
                if body["messages"] == seq[-1][2]["messages"]:
                    trace, rounds = points("eagle_q4", task=len(calls))
                    append(directory.parent / "verify.jsonl", trace)
                    append(directory.parent / "rounds.jsonl", rounds)
                return measurement("eagle_q4")
            finally:
                active["requests"] -= 1

        auth = Mock(side_effect=ValueError("source hash lost") if failure == "source" else None)
        release, cleanup = Mock(return_value={"dxg_holders": []}), Mock()
        patches = [patch.object(probe.subprocess, "Popen", return_value=proc),
                   patch.object(probe.paused, "process_identity", return_value={"pid": 123, "start_ticks": 456, "boot_id": "fixture"}),
                   patch.object(probe.endpoint, "available_port", return_value=True),
                   patch.object(probe.endpoint, "wait_ready"), patch.object(probe.endpoint, "execute_request", side_effect=execute),
                   patch.object(probe.endpoint, "stop_owned_server", cleanup)]
        return plan, protocol, seq, calls, auth, release, cleanup, patches

    def run_fixture(self, root, failure=None):
        from contextlib import ExitStack
        plan, protocol, seq, calls, auth, release, cleanup, patches = self.fixture(root, failure)
        with ExitStack() as stack:
            for p in patches: stack.enter_context(p)
            if failure == "source-after-spawn": auth.side_effect = [None, ValueError("source hash lost")]
            if failure:
                with self.assertRaises((ValueError, RuntimeError)):
                    probe.run_arm(plan, protocol, {"path": "/frozen/q4"}, "eagle_q4", 18290, seq,
                                  root / "eagle_q4", auth, release, lambda: 600)
            else:
                result = probe.run_arm(plan, protocol, {"path": "/frozen/q4"}, "eagle_q4", 18290, seq,
                                       root / "eagle_q4", auth, release, lambda: 600)
                self.assertEqual(len(result["records"]), 5)
                self.assertEqual(calls, [f"original prompt{i}" for i in (0, 1, 0, 1, 2)])
                self.assertEqual(len(result["records"][-1]["points"]), 5)
                self.assertTrue((root / "eagle_q4/resource-return.json").exists())
                self.assertEqual(probe.paused.read(root / "eagle_q4/process.json")["pgid"], 123)
            if failure == "source": cleanup.assert_not_called()
            else: cleanup.assert_called_once()

    def test_fresh_process_order_join_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp: self.run_fixture(Path(temp))

    def test_request_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as temp: self.run_fixture(Path(temp), "request")

    def test_source_hash_loss_before_spawn(self):
        with tempfile.TemporaryDirectory() as temp: self.run_fixture(Path(temp), "source")

    def test_source_hash_loss_after_spawn_cleanup(self):
        with tempfile.TemporaryDirectory() as temp: self.run_fixture(Path(temp), "source-after-spawn")

    def test_deadline_and_exited_server_cleanup(self):
        from contextlib import ExitStack
        for failure in ("deadline", "exit"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                plan, protocol, seq, calls, auth, release, cleanup, patches = self.fixture(root)
                with ExitStack() as stack:
                    for p in patches: stack.enter_context(p)
                    remaining = Mock(side_effect=TimeoutError("deadline")) if failure == "deadline" else lambda: 600
                    if failure == "exit":
                        stack.enter_context(patch.object(probe.subprocess, "Popen", return_value=SimpleNamespace(pid=123, poll=lambda: 1)))
                    with self.assertRaises((ValueError, TimeoutError)):
                        probe.run_arm(plan, protocol, {"path": "/frozen/q4"}, "eagle_q4", 18290, seq,
                                      root / "eagle_q4", auth, release, remaining)
                    cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()

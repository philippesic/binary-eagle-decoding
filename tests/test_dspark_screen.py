import sys
import unittest
import json
import tempfile
import subprocess
import signal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_dspark_screen import ARMS, command, order, round_summary, stop_owned_server, prior_timing_seconds, check_admission
from benchmark_native_eagle import sha256
from analyze_dspark_screen import summarize
from analyze_dspark_events import analyze, union


def fixture(proposed, accepted, emitted=None, **extra):
    emitted = accepted + 1 if emitted is None else emitted
    return dict(status="complete", replay=False, n_proposed=proposed, n_accepted=accepted,
                n_emitted=emitted, proposed_token_ids=list(range(proposed)),
                emitted_token_ids=list(range(emitted)), **extra)


class ScreenContracts(unittest.TestCase):
    def test_admission_binds_actual_runtime_protocol_and_precision(self):
        config = {key: {"sha256": key} for key in ("binary", "target", "dspark", "dflash", "eagle_q4_0")}
        config["protocol_sha256"] = "protocol"
        admission = dict(passed=True, target_sha256="target", binary_sha256="binary", protocol_sha256="protocol",
                         model_sha256={key: key for key in ("dspark", "dflash", "eagle_q4_0")},
                         memory=True, anchor_first=True, target_immutable=True, cache_contract=True, greedy_semantics=True)
        check_admission(config, admission)
        for key in ("binary_sha256", "protocol_sha256"):
            with self.assertRaisesRegex(ValueError, "runtime/protocol"):
                check_admission(config, {**admission, key: "changed"})
        with self.assertRaisesRegex(ValueError, "model changed: dspark"):
            check_admission(config, {**admission, "model_sha256": {**admission["model_sha256"], "dspark": "Q4-candidate"}})

    def test_cumulative_budget_uses_local_cost_not_double_counted_prior(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp) / "first.json", Path(temp) / "second.json"
            first.write_text(json.dumps(dict(diagnostic=False, inference_s=100, records=[])))
            second.write_text(json.dumps(dict(diagnostic=False, inference_s=200, prior_timing_s=100,
                                             combined_timing_s=300, records=[])))
            cfg = {"prior_timing_receipts": [{"path": str(p), "sha256": sha256(p)} for p in (first, second)]}
            self.assertEqual(prior_timing_seconds(cfg), 300)
            first.write_text(json.dumps(dict(diagnostic=True, inference_s=100, records=[])))
            with self.assertRaises(ValueError):
                prior_timing_seconds(cfg)
    def test_reached_prefix_differs_from_all_proposals(self):
        result = round_summary([fixture(3, 0), fixture(3, 1), fixture(3, 3)], 3)
        self.assertEqual(result["position"]["2"], dict(eligible=3, reached=2, survived=1,
                                                      conditional_survival=0.5))
        self.assertEqual(result["accepted"], 4)
        self.assertEqual(result["proposed"], 9)
        self.assertEqual(result["first_rejection"]["0"], 1)

    def test_truncated_emission_preserves_verifier_counts(self):
        result = round_summary([fixture(3, 3, emitted=1)], 3)
        self.assertEqual(result["accepted"], 3)
        self.assertEqual(result["emitted"], 1)

    def test_replays_and_incomplete_rounds_stay_separate(self):
        replay = fixture(3, 0)
        replay["replay"] = True
        result = round_summary([replay, {"status": "aborted"}, fixture(3, 1)], 3)
        self.assertEqual(result["other_records"], 2)
        self.assertEqual(result["rounds"], 1)

    def test_bad_native_counts_fail(self):
        with self.assertRaises(ValueError):
            round_summary([fixture(3, 4)], 3)
        with self.assertRaises(ValueError):
            round_summary([fixture(7, 2)], 3)
        record = fixture(3, 2)
        record["proposed_token_ids"] = []
        with self.assertRaises(ValueError):
            round_summary([record], 3)

    def test_native_duration_vectors_and_missing_fields(self):
        result = round_summary([fixture(3, 1, draft_step_decode_us=[2, 3], draft_sampler_us=[7], draft_seed_decode_us=11)], 3)
        self.assertEqual(result["cpu_wall_totals_us"]["draft_step_decode_us"], 5)
        self.assertEqual(result["cpu_wall_totals_us"]["draft_seed_decode_us"], 11)
        self.assertIsNone(result["cpu_wall_totals_us"]["process_us"])

    def test_stubborn_owned_server_is_killed_before_supervisor_grace(self):
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);print('ready',flush=True);time.sleep(60)"],
                                start_new_session=True, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(proc.stdout.readline().strip(), "ready")
            stop_owned_server(proc)
            self.assertEqual(proc.returncode, -signal.SIGKILL)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            proc.stdout.close()

    def test_cuda_costs_do_not_add_parent_and_child_spans(self):
        base = dict(kind="node", model_arch="dflash", captured_inventory_only=False,
                    cuda_ms=1, llama_context="draft", context="cuda", stream="main",
                    frame=1, n_outputs=7, op="MUL_MAT")
        nodes = [dict(base, node=0, tensor="body", gpu_begin_ms=0, gpu_end_ms=1),
                 dict(base, node=1, tensor="result_output", gpu_begin_ms=1, gpu_end_ms=2),
                 dict(base, node=2, tensor="dspark_markov_projection-0", gpu_begin_ms=2, gpu_end_ms=3),
                 dict(kind="graph", model_arch="dflash", cuda_ms=3)]
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "server.log"
            log.write_text(''.join("CUDA_EAGLE_EVENT " + json.dumps(n) + "\n" for n in nodes))
            result = analyze(log)
            self.assertEqual(sum(v["cuda_ms_union"] for v in result["costs"].values()), 3)
            self.assertEqual(union([(0, 3), (1, 2), (2, 4)]), 4)
            log.write_text(log.read_text() + 'CUDA_EAGLE_EVENT {"kind":"truncation"}\n')
            with self.assertRaises(ValueError):
                analyze(log)

    def test_every_order_is_paired(self):
        for rep in range(5):
            self.assertEqual(set(order(rep)), set(ARMS))
            self.assertEqual(len(order(rep)), len(ARMS))
        self.assertNotEqual(order(0), order(1))
        for position in range(6):
            self.assertEqual({order(rep)[position] for rep in range(6)}, set(ARMS))
        transitions = [(left, right) for rep in range(6) for left, right in zip(order(rep), order(rep)[1:])]
        self.assertEqual(len(set(transitions)), 30)

    def test_rank_zero_uses_author_driver(self):
        config = {key: {"path": key} for key in ("binary", "target", "dflash", "dspark", "eagle_q4_0")}
        protocol = {"context_tokens": 2048, "eagle_length": 5}
        cmd = command(config, protocol, "dflash_7", 18290)
        self.assertEqual(cmd[cmd.index("--spec-type")+1], "draft-dspark")
        self.assertEqual(cmd[cmd.index("--spec-draft-n-max")+1], "7")
        self.assertEqual(cmd[cmd.index("--spec-draft-p-min")+1], "0")
        self.assertEqual(cmd[cmd.index("--fit")+1], "off")

    def test_analyzer_uses_pooled_rates_and_actual_joins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "protocol.json").write_text(json.dumps(dict(prompt_count=2, repetitions=1, eagle_length=5)))
            records = []
            for arm in ARMS:
                for idx, (tokens, seconds) in enumerate(((10, 1), (30, 3))):
                    path = root / "rep-00" / arm / f"prompt-{idx:02d}"
                    path.mkdir(parents=True)
                    rec = dict(arm=arm, repetition=0, prompt_id=str(idx), warmup=False,
                               completion_tokens=tokens, server_predicted_ms=seconds*1000,
                               request_wall_s=seconds+1, server_response_id=f"{arm}-{idx}",
                               generated_token_ids=[1, 2], artifact_path=str(path.relative_to(root)))
                    records.append(rec)
                    (path / "measurement.json").write_text(json.dumps(rec))
                    (path / "rounds.json").write_text(json.dumps([fixture(3, idx)]))
            (root / "measurements.json").write_text(json.dumps(dict(diagnostic=False, inference_s=1, records=records)))
            result = summarize(root)
            self.assertEqual(result["arms"]["eagle_q4_0"]["pooled_decode_tps"], 10)
            self.assertAlmostEqual(result["arms"]["eagle_q4_0"]["pooled_request_tps"], 40/6)
            self.assertIsNone(result["arms"]["dspark_3"]["native_rounds"]["cpu_wall_totals_us"]["draft_us"])
            # Same aggregate totals cannot cover a missing paired request.
            records[-1]["prompt_id"] = "0"
            (root / "measurements.json").write_text(json.dumps(dict(diagnostic=False, inference_s=1, records=records)))
            with self.assertRaises(ValueError):
                summarize(root)


if __name__ == "__main__":
    unittest.main()

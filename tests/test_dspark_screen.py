import sys
import unittest
import json
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_dspark_screen import ARMS, command, order, round_summary
from analyze_dspark_screen import summarize


def fixture(proposed, accepted, emitted=None, **extra):
    emitted = accepted + 1 if emitted is None else emitted
    return dict(status="complete", replay=False, n_proposed=proposed, n_accepted=accepted,
                n_emitted=emitted, proposed_token_ids=list(range(proposed)),
                emitted_token_ids=list(range(emitted)), **extra)


class ScreenContracts(unittest.TestCase):
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

    def test_every_order_is_paired(self):
        for rep in range(5):
            self.assertEqual(set(order(rep)), set(ARMS))
            self.assertEqual(len(order(rep)), len(ARMS))
        self.assertNotEqual(order(0), order(1))

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

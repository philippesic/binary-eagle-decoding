import copy
import json
import tempfile
import unittest
from pathlib import Path

from audit_completed_pair import ARMS, CASES, audit
from check_export import sha256


class CompletedPairTests(unittest.TestCase):
    def fixture(self, root):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2) + "\n")
        diagnostic, reference = root / "diagnostic", root / "reference"
        protocol = {"context_tokens": 2048, "batch_tokens": 32, "microbatch_tokens": 32,
                    "max_output_tokens": 128, "temperature": 0.0, "seed": 42,
                    "enable_thinking": False, "cache_prompt": False}
        write(reference / "protocol.json", protocol)
        cfg = {key: {"path": "/immutable/"+key, "sha256": key+"-pin"} for key in ("binary", "target", "eagle_q4_0", "dspark")}
        cfg.update(protocol_sha256=sha256(reference / "protocol.json"), diagnostic_arms=list(ARMS), verify_positions=list(range(88, 96)))
        write(reference / "config.json", cfg)
        write(diagnostic / "config.json", cfg)
        write(diagnostic / "readiness.json", {"cases": list(CASES), "output_cap": 128})
        gate = root / "gate.json"
        write(gate, {"max_abs_competing_logit_gap": 0.05, "max_centered_common_top5_difference": 0.05})
        for arm in ARMS:
            traces, rounds = [], []
            command = ["/immutable/binary", "-m", "/immutable/target", "--port", "18290"]
            write(diagnostic / arm / "launch.json", {"command": command})
            write(reference / "rep-00" / arm / "launch.json", {"command": command})
            for case_index, name in enumerate(CASES):
                task = case_index+1
                ids = [2000+i for i in range(128)]
                if name == "prompt-05": ids[90] = 16062 if arm == "eagle_q4_0" else 28071
                request = {"messages": [{"role": "user", "content": name}], "max_tokens": 128,
                           "temperature": 0.0, "seed": 42, "cache_prompt": False,
                           "chat_template_kwargs": {"enable_thinking": False}}
                response = {"__verbose": {"prompt": name, "stop": True, "stop_type": "limit"}}
                for folder in (diagnostic / arm / name, reference / "rep-00" / arm / name):
                    write(folder / "measurement.json", {"generated_token_ids": ids, "finish_reason": "length"})
                    write(folder / "request.json", request)
                    write(folder / "response.json", response)
                cursor, index = 1, 0
                while cursor < len(ids):
                    chunk = ids[cursor:cursor+4]
                    accepted = len(chunk)-1
                    proposed = chunk[:accepted]
                    rounds.append({"task_id": task, "round_index": index, "status": "complete", "replay": False,
                                   "emitted_token_ids": chunk, "verified_token_ids": chunk,
                                   "n_accepted": accepted, "proposed_token_ids": proposed})
                    for position in range(max(cursor, 88), min(cursor+len(chunk), 96)):
                        row = position-cursor
                        selected = ids[position]
                        competing = 28071 if selected == 16062 else 16062
                        top = [{"token_id": selected, "logit": 20.01}, {"token_id": competing, "logit": 20.00},
                               {"token_id": 300, "logit": 19.0}, {"token_id": 301, "logit": 18.0}, {"token_id": 302, "logit": 17.0}]
                        traces.append({"task_id": task, "round_index": index+1, "generated_position": position, "row": row,
                                       "mode": "speculative_verify", "replay": False, "sampled": True, "emitted": True,
                                       "has_logits": True, "nan_logit_count": 0, "raw_top5": top,
                                       "sampled_token_id": selected, "selected_token_id": selected, "emitted_token_id": selected,
                                       "base_generated_position": cursor, "verifier_prefix_draft_ids": proposed[:row]})
                    cursor += len(chunk)
                    index += 1
            for filename, rows in (("verify.jsonl", traces), ("rounds.jsonl", rounds)):
                (diagnostic / arm / filename).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return diagnostic, reference, gate

    def test_complete_reproduction_and_scoped_pair_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.fixture(Path(temp))
            report = audit(*paths)
            self.assertTrue(report["passed"], report.get("failure"))
            self.assertEqual(len(report["complete_reproduction"]), 16)
            self.assertEqual(report["competing_ids"], [16062, 28071])
            self.assertEqual(len(report["full_input_prefix"]["generated_token_ids"]), 90)
            self.assertNotIn("admission", report)

    def test_changed_complete_suffix_or_termination_fails(self):
        for mode in ("suffix", "termination"):
            with tempfile.TemporaryDirectory() as temp:
                paths = self.fixture(Path(temp))
                path = paths[0] / "dspark_3/prompt-04/measurement.json"
                value = json.loads(path.read_text())
                if mode == "suffix": value["generated_token_ids"][-1] += 1
                else: value["finish_reason"] = "stop"
                path.write_text(json.dumps(value))
                report = audit(*paths)
                self.assertFalse(report["passed"])
                self.assertIn("reproduction failed", report["failure"]["message"])

    def test_bad_margin_or_unreached_prefix_fails(self):
        for mode in ("margin", "prefix"):
            with tempfile.TemporaryDirectory() as temp:
                paths = self.fixture(Path(temp))
                path = paths[0] / "dspark_3/verify.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                row = next(r for r in rows if r["task_id"] == 8 and r["generated_position"] == 90)
                if mode == "margin": row["raw_top5"][0]["logit"] = 21.0
                else: row["verifier_prefix_draft_ids"] = [999]
                path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
                self.assertFalse(audit(*paths)["passed"])

    def test_runtime_pin_or_missing_neighbor_capture_fails(self):
        for mode in ("pin", "neighbor"):
            with tempfile.TemporaryDirectory() as temp:
                paths = self.fixture(Path(temp))
                if mode == "pin":
                    path = paths[0] / "config.json"
                    value = json.loads(path.read_text())
                    value["binary"]["sha256"] = "changed"
                    path.write_text(json.dumps(value))
                else:
                    path = paths[0] / "dspark_3/verify.jsonl"
                    rows = [json.loads(line) for line in path.read_text().splitlines()]
                    rows = [r for r in rows if not (r["task_id"] == 8 and r["generated_position"] == 95)]
                    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
                self.assertFalse(audit(*paths)["passed"])


if __name__ == "__main__":
    unittest.main()

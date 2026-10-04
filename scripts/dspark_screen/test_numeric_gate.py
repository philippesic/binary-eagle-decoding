import copy
import json
import tempfile
import unittest
from pathlib import Path

from check_export import sha256
from numeric_gate import CASES, consume, covered_output, produce


class NumericGateTests(unittest.TestCase):
    def fixture(self, root):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2) + "\n")
        protocol = root / "protocol.json"
        write(protocol, {"context_tokens": 2048})
        config = {"binary": {"path": "/pinned/binary", "sha256": "binary-pin"},
                  "target": {"path": "/pinned/target", "sha256": "target-pin"},
                  "eagle_q4_0": {"path": "/pinned/q4", "sha256": "q4-pin"},
                  "dspark": {"path": "/pinned/dspark", "sha256": "ds-pin"},
                  "dflash": {"path": "/pinned/dflash", "sha256": "df-pin"},
                  "protocol_sha256": sha256(protocol)}
        gate = root / "gate.json"
        write(gate, {"first_divergence": 89, "positions": [87, 88, 89, 90], "competing_token_ids": [72499, 9920],
                     "max_abs_competing_logit_gap": 0.05, "max_centered_common_top5_difference": 0.05})
        request = {"temperature": 0, "seed": 42, "max_tokens": 128, "cache_prompt": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        sequences = {}
        for name in CASES:
            target = [1000+i for i in range(128)] if name != "eos" else [14004, 151645]
            if name.endswith("00"):
                target[89] = 72499
            q4 = list(target)
            if name.endswith("00"):
                q4[89] = 9920
                q4[92:95] = [2348, 279, 2331]
            short = list(q4)
            if name.endswith("00"):
                short[94] = 23035
                short[95] = 3685
            sequences[name] = {"target_only": target, "eagle_q4_0": q4, "dspark_3": short}
        def run(folder, arms, position=None):
            cfg = copy.deepcopy(config)
            if position: cfg["verify_positions"] = list(range(position-2, position+2))
            write(folder / "config.json", cfg)
            for arm in arms:
                trace, rounds = [], []
                for case_index, name in enumerate(CASES):
                    ids = sequences[name][arm]
                    case_request = {**request, "messages": [{"role": "user", "content": "prose" if name.endswith("00") else name.replace("warmup", "prompt")}]}
                    dest = folder / arm / name
                    write(dest / "measurement.json", {"generated_token_ids": ids})
                    write(dest / "request.json", case_request)
                    write(dest / "response.json", {"__verbose": {"prompt": case_request["messages"][0]["content"]}})
                    if not position or name == "eos": continue
                    task = case_index+1
                    if arm == "target_only":
                        row, round_index, base, prefix, status = 0, position, position, [], "target"
                    else:
                        cursor, index = 1, 0
                        chosen = None
                        while cursor < len(ids):
                            if name.endswith("00") and cursor == 89:
                                chunk, accepted, proposals = ids[cursor:cursor+1], 0, [34855]
                            elif name.endswith("00") and cursor == 90:
                                chunk, accepted, proposals = ids[cursor:cursor+2], 1, ids[cursor:cursor+1]
                            elif name.endswith("00") and cursor == 92:
                                n = 4 if arm == "dspark_3" else 3
                                chunk = ids[cursor:cursor+n]
                                accepted = n-1
                                proposals = chunk[:3] if arm == "dspark_3" else chunk[:2]+[279]
                            else:
                                count = min(4, len(ids)-cursor)
                                chunk = ids[cursor:cursor+count]
                                accepted, proposals = count-1, chunk[:count-1]
                            r = {"task_id": task, "round_index": index, "status": "complete", "replay": False,
                                 "emitted_token_ids": chunk, "verified_token_ids": chunk, "n_accepted": accepted,
                                 "proposed_token_ids": proposals}
                            rounds.append(r)
                            if cursor <= position < cursor+len(chunk): chosen = (cursor, index, r)
                            cursor += len(chunk)
                            index += 1
                        base, control_index, control = chosen
                        row, round_index = position-base, control_index+1
                        prefix = control["proposed_token_ids"][:row]
                        status = "accepted" if control["n_accepted"] > row else "rejected"
                    chosen_id = ids[position]
                    pair = [72499, 9920] if position == 89 else [2331, 23035]
                    other = pair[1] if chosen_id == pair[0] else pair[0]
                    top = [{"token_id": chosen_id, "logit": 20.01}, {"token_id": other, "logit": 20.00},
                           {"token_id": 300, "logit": 19.0}, {"token_id": 301, "logit": 18.0}, {"token_id": 302, "logit": 17.0}]
                    trace.append({"task_id": task, "round_index": round_index, "generated_position": position, "row": row,
                                  "mode": "target_only" if arm == "target_only" else "speculative_verify", "replay": False,
                                  "sampled": True, "emitted": True, "sampled_token_id": chosen_id, "emitted_token_id": chosen_id,
                                  "has_logits": True, "nan_logit_count": 0, "raw_top5": top,
                                  "base_generated_position": base, "verifier_prefix_draft_ids": prefix, "status": status})
                if position:
                    for filename, values in (("verify.jsonl", trace), ("rounds.jsonl", rounds)):
                        path = folder / arm / filename
                        path.write_text("\n".join(json.dumps(v) for v in values) + "\n" if values else "")
                    write(folder / arm / "launch.json", {"command": ["/pinned/binary", "-m", "/pinned/target"]})
        primary, diag89, diag94, probe = (root / name for name in ("primary", "diag89", "diag94", "probe"))
        run(primary, ("target_only", "eagle_q4_0"))
        run(diag89, ("target_only", "eagle_q4_0"), 89)
        run(diag94, ("eagle_q4_0", "dspark_3"), 94)
        write(probe / "config.json", config)
        manifest = {"cells": []}
        for kind in ("dspark", "dflash"):
            for n in (3, 7):
                cell = {"kind": kind, "maximum": n, "outputs": []}
                for name in CASES:
                    actual_path = probe / f"{kind}_{n}" / name / "measurement.json"
                    write(actual_path, {"generated_token_ids": sequences[name]["dspark_3" if n == 3 else "eagle_q4_0"]})
                    write(actual_path.parent / "request.json", json.loads((primary / "target_only" / name / "request.json").read_text()))
                    cell["outputs"].append({"actual": str(actual_path), "reference": str(primary / "target_only" / name / "measurement.json")})
                manifest["cells"].append(cell)
        write(probe / "manifest.json", manifest)
        inputs = {"manifest": str(probe / "manifest.json"), "primary": str(primary), "protocol": str(protocol),
                  "edges": [{"left": "target_only", "right": "eagle_q4_0", "results": str(diag89), "gate": str(gate)},
                            {"left": "eagle_q4_0", "right": "dspark_3", "results": str(diag94), "gate": str(gate)}]}
        return inputs, manifest

    def test_two_edges_cover_exact_complete_native_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            inputs, manifest = self.fixture(Path(temp))
            receipt = produce(inputs)
            self.assertEqual(len(receipt["covered_outputs"]), 20)
            self.assertEqual(receipt["uncovered_outputs"], [])
            path = Path(temp) / "receipt.json"
            path.write_text(json.dumps(receipt))
            actual = consume({"path": str(path), "sha256": sha256(path)}, manifest, "binary-pin", "target-pin")
            pair = manifest["cells"][0]["outputs"][0]
            req = json.loads((Path(pair["actual"]).parent / "request.json").read_text())
            self.assertTrue(covered_output(actual, pair["actual"], pair["reference"], req,
                                          json.loads(Path(pair["actual"]).read_text())["generated_token_ids"],
                                          json.loads(Path(pair["reference"]).read_text())["generated_token_ids"]))

    def test_one_edge_keeps_uncovered_short_suffixes_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            inputs, _ = self.fixture(Path(temp))
            inputs["edges"] = inputs["edges"][:1]
            receipt = produce(inputs)
            self.assertEqual(len(receipt["uncovered_outputs"]), 4)

    def test_large_margin_or_unreached_row_rejects(self):
        for kind in ("margin", "prefix"):
            with tempfile.TemporaryDirectory() as temp:
                inputs, _ = self.fixture(Path(temp))
                path = Path(inputs["edges"][1]["results"]) / "dspark_3/verify.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if kind == "margin": rows[0]["raw_top5"][0]["logit"] = 21.0
                else: rows[0]["verifier_prefix_draft_ids"] = [999, 279]
                path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
                with self.assertRaises(ValueError): produce(inputs)

    def test_changed_runtime_or_tampered_receipt_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            inputs, manifest = self.fixture(Path(temp))
            receipt = produce(inputs)
            receipt["covered_outputs"] = []
            path = Path(temp) / "receipt.json"
            path.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "reproduce raw"):
                consume({"path": str(path), "sha256": sha256(path)}, manifest, "binary-pin", "target-pin")


if __name__ == "__main__":
    unittest.main()

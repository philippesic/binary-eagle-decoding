"""CPU-only native capture contracts, corruption gates and runner cleanup."""
from __future__ import annotations

import argparse
import copy
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
import analyze_binary_head_factorial as analysis  # noqa: E402
import run_binary_head_capture as runner  # noqa: E402


def row(depth=0, state_row=0):
    return {"schema": "eagle_head_state_v1", "state_row": state_row, "state_dim": 2,
            "state_boundary": analysis.BOUNDARY, "state_dtype": "float32_native_endian",
            "task_id": 9, "round_index": 0, "depth": depth, "parent_position": 0,
            "verifier_row": depth, "input_position": depth+1, "label_position": depth+2,
            "prefix_token_ids": [1, 2]+[3]*depth, "input_token_id": 2 if depth == 0 else 3,
            "alignment_valid": True, "finite": True, "valid": True, "is_bonus": False,
            "forced": True, "verifier_token_id": 3, "proposed_token_id": 3,
            "raw_verifier_argmax_id": 3,
            "draft_argmax_id": 3, "label_supported": True, "target_rank": 1,
            "target_margin": 0.5, "verifier_reached": True, "verifier_sampled_token_id": 3}


def round_row():
    return {"schema": "eagle_forced_round_v1", "task_id": 9, "round_index": 0,
            "prefix_token_ids": [1], "seed_token_id": 2, "pos0": 1,
            "draft_token_ids": [3, 3], "accepted_drafts": 2, "verifier_token_ids": [3, 3]}


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


class FactorialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ["canonical", *analysis.ARMS]:
            cell = self.root/name
            cell.mkdir()
            body, head = analysis.ARMS.get(name, ("Q4_0", "Q4_0"))
            rows = [row(), row(1, 1)]
            jsonl(cell/"heads.jsonl", rows)
            # Distinct bodies, identical states for all heads of the same body.
            (cell/"heads.f32").write_bytes(struct.pack("4f", *([1.0 if body == "D" else 2.0]*4)))
            jsonl(cell/"forced-rounds.jsonl", [round_row()])
            runner.write(cell/"manifest.json", {"task_prompt_ids": {"9": "historical-1"},
                         "binary_sha256": "same-build", "target_sha256": "same-target",
                         "prompts_sha256": "same-prompts",
                         "requests": [{"id": "historical-1", "generated_token_ids": [3, 3]}],
                         "spec": {"body": body, "head": head}})

    def mutate(self, arm, field, value):
        path = self.root/arm/"heads.jsonl"
        rows = runner.read_jsonl(path)
        rows[0][field] = value
        jsonl(path, rows)

    def test_factorial_has_first_later_and_exact_same_body_gates(self):
        report = analysis.analyze(self.root)
        self.assertTrue(report["passed"])
        self.assertEqual(report["arms"]["d_d"]["first"]["valid"], 1)
        self.assertEqual(report["arms"]["d_d"]["later"]["agreement_rate_valid"], 1)

    def test_rejects_same_body_state_changes(self):
        (self.root/"d_f16"/"heads.f32").write_bytes(struct.pack("4f", 1, 1, 1, 1.001))
        with self.assertRaisesRegex(ValueError, "same-body"):
            analysis.analyze(self.root)

    def test_rejects_verifier_label_or_round_boundary_changes(self):
        self.mutate("q4_d", "verifier_token_id", 4)
        with self.assertRaisesRegex(ValueError, "verifier labels differ"):
            analysis.analyze(self.root)
        self.mutate("q4_d", "verifier_token_id", 3)
        forced = round_row()
        forced["pos0"] = 2
        jsonl(self.root/"q4_d"/"forced-rounds.jsonl", [forced])
        with self.assertRaisesRegex(ValueError, "boundaries differ"):
            analysis.analyze(self.root)

    def test_rejects_output_difference(self):
        path = self.root/"d_d"/"manifest.json"
        manifest = json.loads(path.read_text())
        manifest["requests"][0]["generated_token_ids"] = [4]
        runner.write(path, manifest)
        with self.assertRaisesRegex(ValueError, "target output"):
            analysis.analyze(self.root)

    def test_invalid_alignment_bonus_or_payload_stops(self):
        for field, value in [("input_position", 99), ("is_bonus", True),
                             ("finite", False), ("state_row", 7)]:
            with self.subTest(field=field):
                self.mutate("d_d", field, value)
                with self.assertRaises(ValueError):
                    analysis.analyze(self.root)
                jsonl(self.root/"d_d"/"heads.jsonl", [row(), row(1, 1)])
        (self.root/"d_d"/"heads.f32").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "payload size"):
            analysis.analyze(self.root)

    def test_unsupported_labels_keep_valid_denominator(self):
        supported, unsupported = row(), row(1, 1)
        unsupported.update(label_supported=False, target_rank=None, target_margin=None,
                           draft_argmax_id=4)
        metrics = analysis.head_metrics([supported, unsupported])
        self.assertEqual(metrics["valid"], 2)
        self.assertEqual(metrics["unsupported"], 1)
        self.assertEqual(metrics["agreement_rate_valid"], 0.5)
        self.assertEqual(metrics["agreement_rate_supported"], 1)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.prompt = {"id": "historical-1", "messages": [{"role": "user", "content": "hi"}]}
        for name in ("binary", "target", "draft"):
            (self.root/name).write_text(name)
        jsonl(self.root/"prompts.jsonl", [self.prompt])
        self.args = argparse.Namespace(mode="diagnostic", tokens=8, port=18092,
                                       prompts=self.root/"prompts.jsonl", prompt_manifest=None,
                                       prompts_sha256=None,
                                       binary=self.root/"binary", target=self.root/"target",
                                       output=self.root/"out", d2t=None, target_vocab_size=None)
        self.spec = {"body": "D", "head": "D", "draft": str(self.root/"draft")}

    def test_input_scope_and_environment_override_guards(self):
        variants = {name: {"body": body, "head": head, "draft": "x"}
                    for name, (body, head) in analysis.ARMS.items()}
        runner.validate_inputs(self.args, [self.prompt], variants)
        bad = copy.deepcopy(variants)
        bad["d_d"]["env"] = {"EAGLE_FORCE_ROUNDS_JSONL": "old"}
        with self.assertRaisesRegex(ValueError, "override capture"):
            runner.validate_inputs(self.args, [self.prompt], bad)
        with self.assertRaisesRegex(ValueError, "reserved-final"):
            runner.validate_inputs(self.args, [{**self.prompt, "id": "final-001"}], variants)

    def test_train_manifest_hash_and_count_gate(self):
        self.args.mode, self.args.tokens = "train", 128
        self.args.d2t, self.args.target_vocab_size = self.root/"map.npy", 100
        prompts = [{**self.prompt, "id": f"qat-revisit-train-prose-{i:03d}"} for i in range(96)]
        jsonl(self.args.prompts, prompts)
        runner.write(self.root/"manifest.json", {"splits": {"train": {
            "sha256": runner.sha256(self.args.prompts), "prompts": 96}}})
        runner.validate_inputs(self.args, prompts, {"d_d": self.spec})
        self.args.prompts.write_text(self.args.prompts.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "hash/count"):
            runner.validate_inputs(self.args, prompts, {"d_d": self.spec})

    def test_capture_joins_task_and_cleanup_on_http_failure(self):
        self.args.output.mkdir()
        proc = mock.Mock()
        with mock.patch.object(runner.subprocess, "Popen", return_value=proc), \
                mock.patch.object(runner, "wait_ready"), \
                mock.patch.object(runner, "http_json", side_effect=RuntimeError("broken HTTP")), \
                mock.patch.object(runner, "stop_server", return_value={"stopped": True}) as stop:
            with self.assertRaisesRegex(RuntimeError, "broken HTTP"):
                runner.run_cell(self.args, "d_d", self.spec, [self.prompt])
        stop.assert_called_once_with(proc)
        manifest = json.loads((self.args.output/"d_d"/"manifest.json").read_text())
        self.assertTrue(manifest["server_stop"]["stopped"])
        self.assertFalse(manifest["complete"])

    def test_fake_native_capture_request_join_and_payload(self):
        self.args.output.mkdir()
        cell = self.args.output/"d_d"

        def response(*_args):
            captured = row()
            captured["forced"] = False
            jsonl(cell/"heads.jsonl", [captured])
            (cell/"heads.f32").write_bytes(struct.pack("2f", 1, 2))
            jsonl(cell/"forced-rounds.jsonl", [round_row()])
            jsonl(cell/"rounds.jsonl", [{"status": "ok", "n_accepted": 1,
                                       "n_proposed": 1, "n_emitted": 2,
                                       "proposed_token_ids": [3]}])
            return {"generated_token_ids": [3, 4]}

        with mock.patch.object(runner.subprocess, "Popen"), \
                mock.patch.object(runner, "wait_ready"), \
                mock.patch.object(runner, "http_json", side_effect=response), \
                mock.patch.object(runner, "stop_server", return_value={"stopped": True}):
            manifest = runner.run_cell(self.args, "d_d", self.spec, [self.prompt])
        self.assertEqual(manifest["task_prompt_ids"], {"9": "historical-1"})
        self.assertEqual(manifest["requests"][0]["capture_rows"], [0, 1])
        self.assertEqual(manifest["audit"]["state_dim"], 2)
        self.assertIn("--no-spec-draft-backend-sampling", manifest["command"])
        self.assertTrue(manifest["complete"])


if __name__ == "__main__":
    unittest.main()

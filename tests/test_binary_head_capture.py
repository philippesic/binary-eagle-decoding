"""CPU-only native capture contracts, corruption gates and runner cleanup."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_binary_head_factorial as analysis  # noqa: E402
import run_binary_head_capture as runner  # noqa: E402


def row(depth=0, state_row=0):
    return {
        "schema": "eagle_head_state_v1",
        "state_row": state_row,
        "state_dim": 2,
        "state_boundary": analysis.BOUNDARY,
        "state_dtype": "float32_native_endian",
        "task_id": 9,
        "round_index": 0,
        "depth": depth,
        "parent_position": 0,
        "verifier_row": depth,
        "input_position": depth + 1,
        "label_position": depth + 2,
        "prefix_token_ids": [1, 2] + [3] * depth,
        "input_token_id": 2 if depth == 0 else 3,
        "alignment_valid": True,
        "finite": True,
        "valid": True,
        "is_bonus": False,
        "forced": True,
        "verifier_token_id": 3,
        "proposed_token_id": 3,
        "raw_verifier_argmax_id": 3,
        "draft_argmax_id": 3,
        "label_supported": True,
        "target_rank": 1,
        "target_margin": 0.5,
        "verifier_reached": True,
        "verifier_sampled_token_id": 3,
    }


def round_row():
    return {
        "schema": "eagle_forced_round_v1",
        "task_id": 9,
        "round_index": 0,
        "prefix_token_ids": [1],
        "seed_token_id": 2,
        "pos0": 1,
        "draft_token_ids": [3, 3],
        "accepted_drafts": 2,
        "verifier_token_ids": [3, 3],
    }


def jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


class FactorialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ["canonical", *analysis.ARMS]:
            cell = self.root / name
            cell.mkdir()
            body, head = analysis.ARMS.get(name, ("Q4_0", "Q4_0"))
            rows = [row(), row(1, 1)]
            jsonl(cell / "heads.jsonl", rows)
            # Distinct bodies, identical states for all heads of the same body.
            (cell / "heads.f32").write_bytes(
                struct.pack("4f", *([1.0 if body == "D" else 2.0] * 4))
            )
            jsonl(cell / "forced-rounds.jsonl", [round_row()])
            runner.write(
                cell / "manifest.json",
                {
                    "task_prompt_ids": {"9": "historical-1"},
                    "binary_sha256": "same-build",
                    "target_sha256": "same-target",
                    "prompts_sha256": "same-prompts",
                    "requests": [{"id": "historical-1", "generated_token_ids": [3, 3]}],
                    "spec": {"body": body, "head": head},
                },
            )

    def mutate(self, arm, field, value):
        path = self.root / arm / "heads.jsonl"
        rows = runner.read_jsonl(path)
        rows[0][field] = value
        jsonl(path, rows)

    def test_factorial_has_first_later_and_exact_same_body_gates(self):
        report = analysis.analyze(self.root)
        self.assertTrue(report["passed"])
        self.assertEqual(report["arms"]["d_d"]["first"]["valid"], 1)
        self.assertEqual(report["arms"]["d_d"]["later"]["agreement_rate_valid"], 1)

    def test_rejects_same_body_state_changes(self):
        (self.root / "d_f16" / "heads.f32").write_bytes(struct.pack("4f", 1, 1, 1, 1.001))
        with self.assertRaisesRegex(ValueError, "same-body"):
            analysis.analyze(self.root)

    def test_rejects_verifier_label_or_round_boundary_changes(self):
        self.mutate("q4_d", "verifier_token_id", 4)
        with self.assertRaisesRegex(ValueError, "verifier labels differ"):
            analysis.analyze(self.root)
        self.mutate("q4_d", "verifier_token_id", 3)
        forced = round_row()
        forced["pos0"] = 2
        jsonl(self.root / "q4_d" / "forced-rounds.jsonl", [forced])
        with self.assertRaisesRegex(ValueError, "boundaries differ"):
            analysis.analyze(self.root)

    def test_rejects_output_difference(self):
        path = self.root / "d_d" / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["requests"][0]["generated_token_ids"] = [4]
        runner.write(path, manifest)
        with self.assertRaisesRegex(ValueError, "target output"):
            analysis.analyze(self.root)

    def test_invalid_alignment_bonus_or_payload_stops(self):
        for field, value in [
            ("input_position", 99),
            ("is_bonus", True),
            ("finite", False),
            ("state_row", 7),
        ]:
            with self.subTest(field=field):
                self.mutate("d_d", field, value)
                with self.assertRaises(ValueError):
                    analysis.analyze(self.root)
                jsonl(self.root / "d_d" / "heads.jsonl", [row(), row(1, 1)])
        (self.root / "d_d" / "heads.f32").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "payload size"):
            analysis.analyze(self.root)

    def test_unsupported_labels_keep_valid_denominator(self):
        supported, unsupported = row(), row(1, 1)
        unsupported.update(
            label_supported=False, target_rank=None, target_margin=None, draft_argmax_id=4
        )
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
            (self.root / name).write_text(name)
        jsonl(self.root / "prompts.jsonl", [self.prompt])
        self.args = argparse.Namespace(
            mode="diagnostic",
            tokens=8,
            port=18092,
            prompts=self.root / "prompts.jsonl",
            prompt_manifest=None,
            prompts_sha256=None,
            binary=self.root / "binary",
            target=self.root / "target",
            output=self.root / "out",
            d2t=None,
            target_vocab_size=None,
            target_logits_limit=32,
            target_features_limit=128,
        )
        self.spec = {"body": "D", "head": "D", "draft": str(self.root / "draft")}

    def test_input_scope_and_environment_override_guards(self):
        variants = {
            name: {"body": body, "head": head, "draft": "x"}
            for name, (body, head) in analysis.ARMS.items()
        }
        runner.validate_inputs(self.args, [self.prompt], variants)
        bad = copy.deepcopy(variants)
        bad["d_d"]["env"] = {"EAGLE_FORCE_ROUNDS_JSONL": "old"}
        with self.assertRaisesRegex(ValueError, "override capture"):
            runner.validate_inputs(self.args, [self.prompt], bad)
        with self.assertRaisesRegex(ValueError, "reserved-final"):
            runner.validate_inputs(self.args, [{**self.prompt, "id": "final-001"}], variants)

    def test_train_manifest_hash_and_count_gate(self):
        self.args.mode, self.args.tokens = "train", 128
        self.args.d2t, self.args.target_vocab_size = self.root / "map.npy", 100
        prompts = [{**self.prompt, "id": f"qat-revisit-train-prose-{i:03d}"} for i in range(96)]
        jsonl(self.args.prompts, prompts)
        runner.write(
            self.root / "manifest.json",
            {"splits": {"train": {"sha256": runner.sha256(self.args.prompts), "prompts": 96}}},
        )
        runner.validate_inputs(self.args, prompts, {"d_d": self.spec})
        self.args.prompts.write_text(self.args.prompts.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "hash/count"):
            runner.validate_inputs(self.args, prompts, {"d_d": self.spec})

    def test_capture_joins_task_and_cleanup_on_http_failure(self):
        self.args.output.mkdir()
        proc = mock.Mock()
        with (
            mock.patch.object(runner.subprocess, "Popen", return_value=proc),
            mock.patch.object(runner, "wait_ready"),
            mock.patch.object(runner, "http_json", side_effect=RuntimeError("broken HTTP")),
            mock.patch.object(runner, "stop_server", return_value={"stopped": True}) as stop,
        ):
            with self.assertRaisesRegex(RuntimeError, "broken HTTP"):
                runner.run_cell(self.args, "d_d", self.spec, [self.prompt])
        stop.assert_called_once_with(proc)
        manifest = json.loads((self.args.output / "d_d" / "manifest.json").read_text())
        self.assertTrue(manifest["server_stop"]["stopped"])
        self.assertFalse(manifest["complete"])

    def test_fake_native_capture_request_join_and_payload(self):
        self.args.output.mkdir()
        cell = self.args.output / "d_d"

        def response(*_args):
            captured = row()
            captured["forced"] = False
            jsonl(cell / "heads.jsonl", [captured])
            (cell / "heads.f32").write_bytes(struct.pack("2f", 1, 2))
            jsonl(cell / "forced-rounds.jsonl", [round_row()])
            jsonl(
                cell / "rounds.jsonl",
                [
                    {
                        "status": "ok",
                        "n_accepted": 1,
                        "n_proposed": 1,
                        "n_emitted": 2,
                        "proposed_token_ids": [3],
                    }
                ],
            )
            return {"generated_token_ids": [3, 4]}

        with (
            mock.patch.object(runner.subprocess, "Popen"),
            mock.patch.object(runner, "wait_ready"),
            mock.patch.object(runner, "http_json", side_effect=response),
            mock.patch.object(runner, "stop_server", return_value={"stopped": True}),
        ):
            manifest = runner.run_cell(self.args, "d_d", self.spec, [self.prompt])
        self.assertEqual(manifest["task_prompt_ids"], {"9": "historical-1"})
        self.assertEqual(manifest["requests"][0]["capture_rows"], [0, 1])
        self.assertEqual(manifest["audit"]["state_dim"], 2)
        self.assertIn("--no-spec-draft-backend-sampling", manifest["command"])
        self.assertTrue(manifest["complete"])

    def test_recurrent_train_frozen_inputs_and_capture_policy(self):
        self.args.mode, self.args.tokens = "recurrent-train", 128
        self.args.target_vocab_size = runner.TARGET_VOCAB_SIZE
        self.args.d2t = self.root / "d2t.npy"
        mapping = np.arange(32_000, dtype="<i8")
        np.save(self.args.d2t, mapping)
        prompts = [{**self.prompt, "id": f"qat-revisit-train-prose-{i:03d}"} for i in range(96)]
        jsonl(self.args.prompts, prompts)
        prompt_hash = runner.sha256(self.args.prompts)
        runner.write(
            self.root / "manifest.json",
            {"splits": {"train": {"sha256": prompt_hash, "prompts": 96}}},
        )
        variants = {"d_d": self.spec}
        with (
            mock.patch.object(runner, "TRAIN_PROMPTS_SHA256", prompt_hash),
            mock.patch.object(runner, "TARGET_F16_SHA256", runner.sha256(self.args.target)),
            mock.patch.object(runner, "DRAFT_D_SHA256", runner.sha256(self.root / "draft")),
            mock.patch.object(
                runner, "DRAFT_D_D2T_SHA256", hashlib.sha256(mapping.tobytes()).hexdigest()
            ),
        ):
            runner.validate_inputs(self.args, prompts, variants)
            with (
                mock.patch.object(self.args, "target_logits_limit", 65_536),
                mock.patch.object(self.args, "target_features_limit", 131_072),
            ):
                runner.validate_inputs(self.args, prompts, variants)
            self.assertIn("--no-context-shift", runner.server_command(self.args, self.spec))
            for value, name in [
                (0, "target logits"),
                (-1, "target features"),
                (65_537, "target logits"),
                (131_073, "target features"),
            ]:
                with self.subTest(limit=name, value=value):
                    key = (
                        "target_logits_limit"
                        if name == "target logits"
                        else "target_features_limit"
                    )
                    with (
                        mock.patch.object(self.args, key, value),
                        self.assertRaisesRegex(ValueError, name),
                    ):
                        runner.validate_inputs(self.args, prompts, variants)
            for key in (
                "EAGLE_CAPTURE_TARGET_FEATURES",
                "EAGLE_CAPTURE_FULL_LOGITS",
                "GGML_CUDA_DISABLE_GRAPHS",
                "LLAMA_ARG_CONTEXT_SHIFT",
            ):
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(ValueError, "override recurrent"),
                ):
                    runner.validate_inputs(
                        self.args, prompts, {"d_d": {**self.spec, "env": {key: "0"}}}
                    )
            altered = mapping.copy()
            altered[0] = 100_000
            np.save(self.args.d2t, altered)
            with self.assertRaisesRegex(ValueError, "d2t map differs"):
                runner.validate_inputs(self.args, prompts, variants)
        with self.assertRaisesRegex(ValueError, "pinned frozen96"):
            runner.validate_inputs(self.args, prompts, variants)

    def recurrent_response(
        self,
        cell: Path,
        *,
        omit: str | None = None,
        truncate: str | None = None,
        feature_task: int = 9,
    ):
        captured = row()
        captured.update(forced=False, target_logits_row=0)
        jsonl(cell / "heads.jsonl", [captured])
        (cell / "heads.f32").write_bytes(struct.pack("2f", 1, 2))
        jsonl(cell / "forced-rounds.jsonl", [round_row()])
        jsonl(
            cell / "rounds.jsonl",
            [
                {
                    "status": "ok",
                    "n_accepted": 1,
                    "n_proposed": 1,
                    "n_emitted": 2,
                    "proposed_token_ids": [3],
                }
            ],
        )
        files = {
            "heads.target_features.jsonl": lambda: jsonl(
                cell / "heads.target_features.jsonl",
                [
                    {"event": "decoded_row", "feature_row": 0, "task_id": feature_task},
                    {"event": "disposition", "feature_row": 0, "task_id": feature_task},
                ],
            ),
            "heads.target_features.f32": lambda: (cell / "heads.target_features.f32").write_bytes(
                b"\0" * (runner.FEATURE_WIDTH * 4 - (4 if truncate == "features" else 0))
            ),
            "heads.target_logits.f32": lambda: (cell / "heads.target_logits.f32").write_bytes(
                b"\0" * (100 * 4 - (4 if truncate == "logits" else 0))
            ),
        }
        for name, emit in files.items():
            if name != omit:
                emit()
        return {"generated_token_ids": [3, 4]}

    def test_recurrent_capture_hashes_ranges_and_cleanup(self):
        self.args.mode, self.args.tokens = "recurrent-train", 128
        self.args.target_vocab_size = 100
        self.args.output.mkdir()
        cell = self.args.output / "d_d"
        proc = mock.Mock()

        def response(*_args):
            return self.recurrent_response(cell)

        with (
            mock.patch.object(runner.subprocess, "Popen", return_value=proc) as popen,
            mock.patch.object(runner, "wait_ready"),
            mock.patch.object(runner, "http_json", side_effect=response),
            mock.patch.object(runner, "stop_server", return_value={"stopped": True}) as stop,
        ):
            manifest = runner.run_cell(self.args, "d_d", self.spec, [self.prompt])
        stop.assert_called_once_with(proc)
        env = popen.call_args.kwargs["env"]
        self.assertEqual(env["GGML_W1AX_ACT_BITS"], "16")
        self.assertEqual(env["EAGLE_CAPTURE_TARGET_LOGITS"], "1")
        self.assertEqual(env["EAGLE_CAPTURE_TARGET_FEATURES"], "1")
        self.assertEqual(env["EAGLE_CAPTURE_TARGET_FEATURES_LIMIT"], "128")
        self.assertEqual(manifest["requests"][0]["target_feature_event_rows"], [0, 2])
        self.assertEqual(manifest["requests"][0]["target_feature_rows"], [0, 1])
        self.assertEqual(manifest["requests"][0]["target_logit_rows"], [0, 1])
        self.assertEqual(
            manifest["files"]["heads.target_features.f32"]["bytes"], runner.FEATURE_WIDTH * 4
        )
        self.assertIn("heads.target_logits.f32", manifest["files"])
        self.assertTrue(manifest["complete"])

    def test_recurrent_two_request_ranges_survive_logit_cap(self):
        self.args.mode, self.args.tokens = "recurrent-train", 128
        self.args.target_vocab_size = 100
        self.args.target_logits_limit = 1
        self.args.output.mkdir()
        cell = self.args.output / "d_d"
        second_prompt = {**self.prompt, "id": "historical-2"}
        calls = 0

        def response(*_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return self.recurrent_response(cell)
            second_head = row(state_row=1)
            second_head.update(task_id=10, forced=False, target_logits_row=None)
            with (cell / "heads.jsonl").open("a") as stream:
                stream.write(json.dumps(second_head) + "\n")
            with (cell / "heads.f32").open("ab") as stream:
                stream.write(struct.pack("2f", 3, 4))
            second_round = round_row()
            second_round["task_id"] = 10
            with (cell / "forced-rounds.jsonl").open("a") as stream:
                stream.write(json.dumps(second_round) + "\n")
            with (cell / "rounds.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "status": "ok",
                            "n_accepted": 1,
                            "n_proposed": 1,
                            "n_emitted": 2,
                            "proposed_token_ids": [3],
                        }
                    )
                    + "\n"
                )
            with (cell / "heads.target_features.jsonl").open("a") as stream:
                for event in ("decoded_row", "disposition"):
                    stream.write(
                        json.dumps({"event": event, "feature_row": 1, "task_id": 10}) + "\n"
                    )
            with (cell / "heads.target_features.f32").open("ab") as stream:
                stream.write(b"\0" * (runner.FEATURE_WIDTH * 4))
            return {"generated_token_ids": [3, 4]}

        with (
            mock.patch.object(runner.subprocess, "Popen"),
            mock.patch.object(runner, "wait_ready"),
            mock.patch.object(runner, "http_json", side_effect=response),
            mock.patch.object(runner, "stop_server", return_value={"stopped": True}),
        ):
            manifest = runner.run_cell(self.args, "d_d", self.spec, [self.prompt, second_prompt])
        self.assertEqual(manifest["task_prompt_ids"], {"9": "historical-1", "10": "historical-2"})
        self.assertEqual(manifest["requests"][1]["capture_rows"], [1, 2])
        self.assertEqual(manifest["requests"][1]["target_feature_rows"], [1, 2])
        self.assertEqual(manifest["requests"][1]["target_logit_rows"], [1, 1])
        self.assertTrue(manifest["complete"])

    def test_recurrent_capture_rejects_missing_truncated_and_ambiguous_raw_files(self):
        self.args.mode, self.args.tokens = "recurrent-train", 128
        self.args.target_vocab_size = 100
        for case in ("missing-features", "truncated-features", "truncated-logits", "wrong-task"):
            with self.subTest(case=case):
                self.args.output = self.root / case
                self.args.output.mkdir()
                cell = self.args.output / "d_d"
                proc = mock.Mock()

                def response(*_args):
                    return self.recurrent_response(
                        cell,
                        omit="heads.target_features.f32" if case == "missing-features" else None,
                        truncate="features"
                        if case == "truncated-features"
                        else "logits"
                        if case == "truncated-logits"
                        else None,
                        feature_task=10 if case == "wrong-task" else 9,
                    )

                with (
                    mock.patch.object(runner.subprocess, "Popen", return_value=proc),
                    mock.patch.object(runner, "wait_ready"),
                    mock.patch.object(runner, "http_json", side_effect=response),
                    mock.patch.object(
                        runner, "stop_server", return_value={"stopped": True}
                    ) as stop,
                ):
                    with self.assertRaises(ValueError):
                        runner.run_cell(self.args, "d_d", self.spec, [self.prompt])
                stop.assert_called_once_with(proc)
                manifest = json.loads((cell / "manifest.json").read_text())
                self.assertFalse(manifest["complete"])

    def test_recurrent_run_writes_plain_task_map_and_bundle_manifest(self):
        self.args.mode, self.args.tokens = "recurrent-train", 128
        self.args.d2t = self.root / "d2t.npy"
        self.args.d2t.write_bytes(b"synthetic-map")
        self.args.target_vocab_size = 100
        self.args.variants = self.root / "variants.json"
        runner.write(self.args.variants, {"d_d": self.spec})
        raw_names = (
            "heads.jsonl",
            "heads.f32",
            "forced-rounds.jsonl",
            "heads.target_logits.f32",
            "heads.target_features.jsonl",
            "heads.target_features.f32",
        )

        def fake_cell(args, _name, _spec, _prompts):
            cell = args.output / "d_d"
            cell.mkdir()
            runner.write(cell / "manifest.json", {"complete": True})
            return {
                "task_prompt_ids": {"9": self.prompt["id"]},
                "binary_sha256": "binary",
                "target_sha256": "target",
                "draft_sha256": "draft",
                "files": {name: {"bytes": 1, "sha256": name} for name in raw_names},
                "requests": [{"id": self.prompt["id"], "task_id": "9"}],
            }

        with (
            mock.patch.object(runner, "validate_inputs"),
            mock.patch.object(runner, "run_cell", side_effect=fake_cell),
        ):
            runner.run(self.args)
        plain = json.loads((self.args.output / "task_prompt_ids.json").read_text())
        manifest = json.loads((self.args.output / "capture-manifest.json").read_text())
        self.assertEqual(plain, {"9": self.prompt["id"]})
        self.assertEqual(
            manifest["task_prompt_ids_sha256"],
            runner.sha256(self.args.output / "task_prompt_ids.json"),
        )
        self.assertEqual(manifest["target_sha256"], "target")
        self.assertEqual(manifest["raw_files"]["heads.target_features.f32"]["bytes"], 1)
        self.assertTrue(manifest["complete"])

    def test_recurrent_rejects_forced_history_and_final_path_before_io(self):
        self.args.mode = "recurrent-train"
        with mock.patch.object(runner.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "unforced candidate D"):
                runner.run_cell(self.args, "d_d", self.spec, [self.prompt], forced=self.root)
            popen.assert_not_called()
        self.args.prompts = self.root / "final.jsonl"
        with mock.patch.object(runner, "load_prompts") as loader:
            with self.assertRaisesRegex(ValueError, "reserved-final"):
                runner.run(self.args)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()

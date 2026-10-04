import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import probe


class ProbeTests(unittest.TestCase):
    def test_eos_uses_raw_generated_content_and_target_eos_id(self):
        measurement = {"finish_reason": "stop", "generated_token_ids": [14004, 151645]}
        response = {
            "choices": [{"message": {"content": "<think>\n\n</think>\n\nYES"}}],
            "__verbose": {"content": "YES", "tokens": [14004, 151645],
                          "stop": True, "stop_type": "eos"},
        }
        self.assertTrue(probe.one_word_eos_satisfied(response, measurement))

        # Visible chat content may include template markup even when raw completion is exact.
        response["__verbose"]["content"] = "YES!"
        self.assertFalse(probe.one_word_eos_satisfied(response, measurement))
        response["__verbose"]["content"] = "YES"
        measurement["generated_token_ids"] = [14004]
        self.assertFalse(probe.one_word_eos_satisfied(response, measurement))
        response.pop("__verbose")
        self.assertFalse(probe.one_word_eos_satisfied(response, measurement))

    def inputs(self, root):
        def asset(name, value):
            path = root / name
            path.write_text(json.dumps(value))
            return {"path": str(path), "sha256": probe.sha256(path)}
        config = {"binary": asset("binary", "fixture"), "target": asset("target", "fixture")}
        prompt_path = root / "prompts.jsonl"
        prompts = [{"id": str(i), "split": "development", "messages": [{"role": "user", "content": str(i)}]} for i in range(24)]
        prompt_path.write_text("\n".join(json.dumps(p) for p in prompts))
        protocol = {"context_tokens": 2048, "kv_precision": "f16", "target_precision": "f16",
                    "max_output_tokens": 128, "temperature": 0.0, "seed": 42,
                    "enable_thinking": False, "cache_prompt": False, "concurrency": 1,
                    "candidate_lengths": [3, 7], "native_noise_tokens": 7, "warmups_per_cell": 2,
                    "prompt_file": "prompts.jsonl", "prompt_sha256": probe.sha256(prompt_path)}
        config["protocol"] = asset("protocol.json", protocol)
        for kind in ("dspark", "dflash"):
            row = asset(kind + ".gguf", "fixture")
            row["source"] = asset(kind + ".safetensors", "fixture")
            row["conversion_config"] = asset(kind + "-config.json", {})
            row["canonical"] = asset(kind + "-canonical.json", {
                "source_sha256": row["source"]["sha256"], "target_sha256": config["target"]["sha256"],
                "safe_to_borrow_embedding": True, "safe_to_borrow_head": True})
            row["export"] = asset(kind + "-export.json", {
                "passed": True, "source_sha256": row["source"]["sha256"],
                "config_sha256": row["conversion_config"]["sha256"], "draft_sha256": row["sha256"],
                "target_sha256": config["target"]["sha256"], "canonical_comparison_sha256": row["canonical"]["sha256"],
                "canonical_comparison_path": row["canonical"]["path"], "borrows_embedding": True,
                "borrows_head": True, "matrix_storage_types": ["BF16"]})
            config[kind] = row
        return config, protocol, prompts

    def test_preflight_pins_all_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _, _ = self.inputs(root)
            _, _, prompts, pins = probe.preflight(config, root)
            self.assertEqual(len(prompts), 24)
            self.assertEqual(len(pins), 14)

    def test_hash_error_precedes_model_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _, _ = self.inputs(root)
            config["target"]["sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "hash changed"):
                probe.preflight(config, root)

    def test_unsafe_borrowing_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _, _ = self.inputs(root)
            canonical = config["dspark"]["canonical"]
            data = probe.read(canonical["path"])
            data["safe_to_borrow_head"] = False
            Path(canonical["path"]).write_text(json.dumps(data))
            canonical["sha256"] = probe.sha256(Path(canonical["path"]))
            export = config["dspark"]["export"]
            data = probe.read(export["path"])
            data["canonical_comparison_sha256"] = canonical["sha256"]
            Path(export["path"]).write_text(json.dumps(data))
            export["sha256"] = probe.sha256(Path(export["path"]))
            with self.assertRaisesRegex(ValueError, "unsafe target head"):
                probe.preflight(config, root)

    def test_profile_and_sync_env_are_removed(self):
        with patch.dict(probe.os.environ, {"DSPARK_SYNC_COMPONENT_TIMINGS": "1", "GGML_CUDA_EAGLE_EVENTS": "1",
                                           "EAGLE_CAPTURE_PREFIX": "danger", "DSPARK_GRAPH_PROFILE_JSONL": "danger"}):
            env = probe.environment_for(Path("/fixture"), "dspark_3")
        self.assertNotIn("DSPARK_SYNC_COMPONENT_TIMINGS", env)
        self.assertNotIn("GGML_CUDA_EAGLE_EVENTS", env)
        self.assertNotIn("EAGLE_CAPTURE_PREFIX", env)
        self.assertEqual(env["DSPARK_REQUIRE_AUTHOR_LAYOUT"], "1")

    def test_supervisor_cleanup_grace_is_verified_before_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "tmp").mkdir()
            path = root / "state.json"
            receipt = {"stop_grace_seconds": 10, "supervisor_pid": 123, "run_id": "fixture",
                       "command": ["python3", "scripts/dspark_screen/probe.py", "config", "output"]}
            probe.write(path, receipt)
            with patch.dict(probe.os.environ, {"TMPDIR": str(root / "tmp")}):
                with self.assertRaisesRegex(ValueError, "30-second"):
                    probe.supervisor_receipt()
                receipt["stop_grace_seconds"] = 30
                probe.write(path, receipt)
                self.assertEqual(probe.supervisor_receipt()["stop_grace_seconds"], 30)

    def test_readiness_failure_still_records_and_stops_owned_server(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _, _ = self.inputs(root)
            cfg_path = root / "probe.json"
            probe.write(cfg_path, config)
            stopped = []
            def stop(proc, grace_s):
                stopped.append(proc.pid)
                proc.returncode = -15
            screen = SimpleNamespace(command=lambda *args: [config["binary"]["path"]],
                                     wait_ready=lambda *args: (_ for _ in ()).throw(RuntimeError("startup fixture failure")),
                                     stop_owned_server=stop)
            common = SimpleNamespace(environment_manifest=lambda p: {"gpu_capability_query": {"exit_code": 0, "stdout": "NVIDIA GeForce RTX 2080 Ti,7.5"}}, gpu_snapshot=lambda: {},
                                     available_port=lambda *args: True)
            proc = SimpleNamespace(pid=777777, returncode=None)
            with patch.object(probe, "supervisor_receipt", return_value={"unit_test_only": True}), patch.object(probe, "load_helpers", return_value=(screen, common)), patch.object(probe.subprocess, "Popen", return_value=proc):
                with self.assertRaisesRegex(RuntimeError, "startup fixture failure"):
                    probe.run(cfg_path, root / "output", root)
            self.assertEqual(stopped, [777777])
            owned = probe.read(root / "output/target_only/owned_process.json")
            self.assertEqual(owned["pgid"], 777777)
            self.assertTrue(owned["stopped"])
            self.assertTrue((root / "output/failure.json").exists())
            self.assertFalse((root / "output/admission.json").exists())

    def test_complete_transaction_creates_four_cell_manifest_after_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _, _ = self.inputs(root)
            cfg_path = root / "probe.json"
            probe.write(cfg_path, config)
            stopped, requests, manifests = [], [], []
            def stop(proc, grace_s):
                stopped.append(proc.pid)
                proc.returncode = -15
            def rows(path):
                return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            def request(url, body, timeout, directory):
                directory.mkdir()
                probe.write(directory / "request.json", body)
                is_eos = directory.name == "eos"
                ids = [42, 151645] if is_eos else [42]
                response = {"choices": [{"message": {"content": "YES"}}]}
                if is_eos:
                    response["__verbose"] = {"content": "YES", "tokens": ids,
                                              "stop": True, "stop_type": "eos"}
                probe.write(directory / "response.json", response)
                measurement = {"generated_token_ids": ids, "prompt_tokens": 20, "finish_reason": "stop"}
                probe.write(directory / "measurement.json", measurement)
                with (directory.parent / "rounds.jsonl").open("a") as file:
                    file.write(json.dumps({"emitted_token_ids": [42, 151645]}) + "\n")
                requests.append(directory)
                return measurement
            screen = SimpleNamespace(command=lambda cfg, proto, arm, port: [cfg["binary"]["path"], arm],
                                     wait_ready=lambda *args: None, stop_owned_server=stop, rows=rows)
            common = SimpleNamespace(environment_manifest=lambda p: {"gpu_capability_query": {"exit_code": 0, "stdout": "NVIDIA GeForce RTX 2080 Ti,7.5"}},
                                     gpu_snapshot=lambda: {}, available_port=lambda *args: True,
                                     request_body=lambda cfg, prompt: {"messages": prompt["messages"]}, execute_request=request)
            procs = [SimpleNamespace(pid=100+i, returncode=None) for i in range(5)]
            def checked(manifest):
                self.assertEqual(len(stopped), 5)
                manifests.append(manifest)
                return {"passed": True, "unit_test_only": True}
            with patch.object(probe, "supervisor_receipt", return_value={"unit_test_only": True}), patch.object(probe, "load_helpers", return_value=(screen, common)), patch.object(probe.subprocess, "Popen", side_effect=procs), patch.object(probe, "validate", side_effect=checked):
                probe.run(cfg_path, root / "output", root)
            self.assertEqual(len(requests), 25)
            self.assertEqual(len(manifests[0]["cells"]), 4)
            self.assertTrue((root / "output/admission.json").exists())
            for cell in manifests[0]["cells"]:
                self.assertEqual(len(cell["outputs"]), 5)
                for pair in cell["outputs"]:
                    self.assertTrue(Path(pair["actual"]).exists())
                    self.assertTrue(Path(pair["reference"]).exists())

    def test_cancel_during_spawn_persists_pid_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _, _ = self.inputs(root)
            cfg_path = root / "probe.json"
            probe.write(cfg_path, config)
            stopped = []
            proc = SimpleNamespace(pid=777778, returncode=None)
            def spawn(*args, **kwargs):
                probe.interrupted(probe.signal.SIGTERM, None)
                return proc
            def stop(server, grace_s):
                stopped.append(server.pid)
                server.returncode = -15
            screen = SimpleNamespace(command=lambda *args: [config["binary"]["path"]], stop_owned_server=stop)
            common = SimpleNamespace(environment_manifest=lambda p: {"gpu_capability_query": {"exit_code": 0, "stdout": "NVIDIA GeForce RTX 2080 Ti,7.5"}},
                                     gpu_snapshot=lambda: {}, available_port=lambda *args: True)
            with patch.object(probe, "supervisor_receipt", return_value={"unit_test_only": True}), patch.object(probe, "load_helpers", return_value=(screen, common)), patch.object(probe.subprocess, "Popen", side_effect=spawn), patch.object(probe.signal, "signal"):
                with self.assertRaises(SystemExit):
                    probe.run(cfg_path, root / "output", root)
            self.assertEqual(stopped, [777778])
            self.assertTrue(probe.read(root / "output/target_only/owned_process.json")["stopped"])
            self.assertFalse((root / "output/admission.json").exists())


if __name__ == "__main__":
    unittest.main()

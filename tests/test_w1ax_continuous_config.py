"""CPU-only resolver, source pin and recovery checks with tiny artifacts."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import test_w1ax_capture_provider as provider_fixture  # noqa: E402
import w1ax_capture_provider as provider  # noqa: E402
import w1ax_continuous_stages as stages  # noqa: E402


class ContinuousConfigTests(unittest.TestCase):
    def test_prepare_config_microshards_never_opens_sealed_data(self):
        fixture = provider_fixture.ProviderFixture()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root = fixture.root
        legacy = root / "provider.json"
        with (
            patch.object(provider_fixture.prep, "TARGET_GGUF_SHA256", fixture.target_hash),
            patch.object(provider_fixture.prep, "CANDIDATE_D_SHA256", fixture.candidate_hash),
        ):
            fixture._prepare(legacy)
        files = {}
        for split in ("train", "dev"):
            prompts = root / (split + ".jsonl")
            index = root / (split + ".index.jsonl")
            rows = [
                {
                    "id": f"{split}-{i}",
                    "domain": ("code", "prose", "reasoning")[i % 3],
                    "messages": [{"role": "user", "content": f"{split} {i}"}],
                }
                for i in range(24)
            ]
            prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
            index.write_text(
                "".join(json.dumps({"id": r["id"], "input_tokens": 200}) + "\n" for r in rows)
            )
            files[split] = {
                "shards": [
                    {
                        "prompts": prompts.name,
                        "prompts_sha256": stages.sha256(prompts),
                        "prompts_count": len(rows),
                        "index": index.name,
                        "index_sha256": stages.sha256(index),
                    }
                ]
            }
        files["sealed_test"] = {"shards": [{"prompts": "MUST-NOT-READ.jsonl"}]}
        corpus = root / "corpus.json"
        corpus.write_text(
            json.dumps({"schema": "continuous_w1ax_prompt_manifest_v1", "files": files})
        )
        output = root / "resolved-stages.json"
        with (
            patch.object(provider, "TARGET_GGUF_SHA256", fixture.target_hash),
            patch.object(provider, "CANDIDATE_D_SHA256", fixture.candidate_hash),
            patch.object(stages, "audit_q4_file", return_value={"stored_type": "Q4_0"}),
            patch.object(
                stages, "native_runtime_inventory", return_value={"schema": "CPU_fixture"}
            ),
            patch("torch.cuda.is_available", side_effect=AssertionError("GPU query")),
        ):
            result = stages.prepare_config(
                corpus, legacy, fixture.target, fixture.candidate, output, shard_prompts=8
            )
        self.assertEqual(result["split_counts"], {"train": 24, "dev": 24})
        self.assertEqual(len(result["captures"]), 6)
        self.assertEqual(result["development_selected_prompts"], 24)
        self.assertEqual(result["storage_forecast"]["full_vocabulary_target_logits_bytes"], 0)
        self.assertEqual(
            sum(c["storage_forecast"]["unique_input_tokens"] for c in result["captures"]), 9600
        )
        self.assertTrue(all(c["prompt_count"] == 8 for c in result["captures"]))

    def test_refresh_rejects_sealed_path_before_hash_read_or_gpu_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            sealed = root / "sealed_test-00000.jsonl"
            with patch.object(stages, "sha256", side_effect=AssertionError("sealed SHA read")):
                with self.assertRaisesRegex(ValueError, "sealed"):
                    stages.refresh_checkpoint(
                        {}, root / "checkpoint", sealed, root / "out", 8, prompts_sha256="a" * 64
                    )

    def test_opaque_sealed_role_hash_is_rejected_without_prompt_inspection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = root / "manifest.json"
            corpus.write_text(
                json.dumps(
                    {
                        "files": {
                            "sealed_test": {
                                "shards": [
                                    {"prompts": "reserved.jsonl", "prompts_sha256": "a" * 64}
                                ]
                            }
                        }
                    }
                )
            )
            source = {"corpus_manifest": stages.file_record(corpus)}
            with self.assertRaisesRegex(ValueError, "opaque sealed"):
                stages.require_unsealed_prompts(
                    root / "renamed.jsonl", sources=source, expected_sha256="a" * 64
                )

    def test_loaded_library_inventory_rejects_unpinned_or_replaced_bytes(self):
        from run_binary_head_capture import verify_mapped_runtime

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / "libllama.so", root / "libggml-cuda.so"]
            for path in paths:
                path.write_bytes(path.name.encode())
            runtime = {"libraries": [stages.file_record(p) for p in paths]}
            maps = "\n".join(f"000-fff r-xp 00 00:00 0 {p}" for p in paths)
            with patch.object(Path, "read_text", return_value=maps):
                report = verify_mapped_runtime(123, runtime)
                self.assertEqual(len(report["mapped_libraries"]), 2)
                paths[0].write_bytes(b"replaced")
                with self.assertRaisesRegex(ValueError, "changed"):
                    verify_mapped_runtime(123, runtime)
            with patch.object(Path, "read_text", return_value=maps + " (deleted)"):
                with self.assertRaisesRegex(ValueError, "deleted"):
                    verify_mapped_runtime(123, runtime)

    def test_unknown_q4_artifact_fails_before_gguf_or_accelerator_import(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "q4.gguf"
            path.write_bytes(b"not the frozen baseline")
            with patch("torch.cuda.is_available", side_effect=AssertionError("GPU query")):
                with self.assertRaisesRegex(ValueError, "frozen primary"):
                    stages.audit_q4_file(path)

    def test_changed_refresh_source_fails_cpu_inventory_before_capture(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input"
            source.write_bytes(b"a")
            names = (
                "binary",
                "target_gguf",
                "candidate_d_gguf",
                "base_draft_gguf",
                "absolute_d2t",
                "model_snapshot_manifest",
            )
            sources = {name: str(source) for name in names}
            sources["sha256"] = {name: stages.sha256(source) for name in names}
            source.write_bytes(b"b")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                stages.verify_sources(sources)

    def test_recovery_preserves_partial_data_and_refuses_unknown_groups(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            source.write_bytes(b"source")
            names = (
                "binary",
                "target_gguf",
                "candidate_d_gguf",
                "base_draft_gguf",
                "absolute_d2t",
                "model_snapshot_manifest",
            )
            cfg = {
                "schema": stages.STAGES_SCHEMA,
                "sources": {
                    **{n: str(source) for n in names},
                    "sha256": {n: stages.sha256(source) for n in names},
                },
            }
            config_path = root / "config.json"
            config_path.write_text(json.dumps(cfg))
            run = root / "run"
            native = run / "stages/capture-00000/native/d_d"
            native.mkdir(parents=True)
            (run / "resolved_config.json").write_text(json.dumps({"stages": cfg}))
            (run / "status.json").write_text(json.dumps({"pid": 99999999, "models": {}}))
            manifest = native / "manifest.json"
            manifest.write_text(json.dumps({"complete": False}))
            payload = native / "partial.f32"
            payload.write_bytes(b"preserve me")
            with self.assertRaisesRegex(RuntimeError, "stop proof"):
                stages.recover_partial(run, config_path)
            manifest.write_text(json.dumps({"complete": False, "server_pgid": 99999998}))
            result = stages.recover_partial(run, config_path)
            self.assertFalse(result["raw_artifacts_deleted"])
            self.assertFalse(result["processes_started"])
            preserved = Path(result["records"][0]["preserved"])
            self.assertEqual((preserved / "d_d/partial.f32").read_bytes(), b"preserve me")
            self.assertFalse(native.parent.exists())

    def test_owned_evaluation_retention_keeps_partial_summaries(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            foreign = parent / "historical"
            foreign.mkdir()
            for i in range(5):
                path = parent / f"attempt-{i}"
                path.mkdir()
                (path / "ownership.json").write_text(
                    json.dumps({"schema": "continuous_development_owned_v1"})
                )
                os.utime(path, (100 + i, 100 + i))
            stages.prune_owned_evaluations(parent, 3)
            self.assertTrue(foreign.exists())
            self.assertEqual(len(list(parent.glob("attempt-*"))), 3)
            summaries = json.loads((parent / "summaries.json").read_text())
            self.assertEqual(len(summaries), 2)
            self.assertTrue(all(s["status"] == "partial_owned_evaluation" for s in summaries))


if __name__ == "__main__":
    unittest.main()

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_block_binary_native import check
from export_block_binary import FFN_BASES, check_manifest, export_model, pack, sha256
from gguf import GGUFReader, GGUFWriter


class BlockExportTests(unittest.TestCase):
    def fixture(self, root, family="dspark", profile="ffn15_fusion", bits=8):
        base, checkpoint, manifest = (
            root / "base.gguf",
            root / "checkpoint.npz",
            root / "manifest.json",
        )
        writer = GGUFWriter(base, "dflash")
        writer.add_block_count(5)
        writer.add_block_size(7)
        writer.add_target_layers([2, 10, 18, 26, 34])
        writer.add_sample_from_anchor(True)
        writer.add_embedding_length(32)
        writer.add_feed_forward_length(64)
        writer.add_head_count(4)
        writer.add_head_count_kv(2)
        writer.add_context_length(128)
        writer.add_rope_dimension_count(8)
        writer.add_layer_norm_rms_eps(1e-6)
        writer.add_tokenizer_model("llama")
        writer.add_token_list([f"token{i}" for i in range(32)])
        arrays = {}
        entries = {}
        for name in sorted(FFN_BASES | {"fc"}):
            shape = (32, 64) if name.endswith("down") else (64, 32)
            if name == "fc":
                shape = (32, 160)
            dense = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) - 7
            writer.add_tensor(name + ".weight", dense)
            if name == "fc" and profile == "ffn15":
                continue
            entries[name] = {"checkpoint_name": name, "shape": list(shape)}
            arrays[name + ".latent"] = dense
            arrays[name + ".scale"] = np.full(shape[0], 0.03, np.float32)
            arrays[name + ".scale"][0] = 0
        for name in ("token_embd.weight", "output.weight"):
            writer.add_tensor(name, np.ones((32, 32), np.float32))
        for name in ("output_norm.weight", "enc.output_norm.weight"):
            writer.add_tensor(name, np.ones(32, np.float32))
        for i in range(5):
            for name in ("attn_norm", "ffn_norm"):
                writer.add_tensor(f"blk.{i}.{name}.weight", np.ones(32, np.float32))
            for name in ("attn_q_norm", "attn_k_norm"):
                writer.add_tensor(f"blk.{i}.{name}.weight", np.ones(8, np.float32))
            for name, rows in [("attn_q", 32), ("attn_k", 16), ("attn_v", 16), ("attn_output", 32)]:
                writer.add_tensor(
                    f"blk.{i}.{name}.weight", np.eye(rows, 32, dtype=np.float32) * 0.02
                )
        if family == "dspark":
            writer.add_tensor("markov_w1.weight", np.ones((32, 8), np.float32))
            writer.add_tensor("markov_w2.weight", np.ones((32, 8), np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        np.savez(checkpoint, **arrays)
        data = {
            "schema_version": 1,
            "family": family,
            "profile": profile,
            "activation_bits": bits,
            "base_gguf_sha256": sha256(base),
            "checkpoint_sha256": sha256(checkpoint),
            "projections": entries,
        }
        manifest.write_text(json.dumps(data))
        return base, checkpoint, manifest, data

    def test_both_families_profiles_precisions_roundtrip(self):
        for family in ("dspark", "dflash"):
            for profile in ("ffn15", "ffn15_fusion"):
                for bits in (1, 8):
                    with (
                        self.subTest(family=family, profile=profile, bits=bits),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        root = Path(tmp)
                        base, checkpoint, manifest, data = self.fixture(root, family, profile, bits)
                        report = export_model(base, checkpoint, manifest, root / "output.gguf")
                        self.assertTrue(report["serialization_audit_passed"])
                        self.assertEqual(report["native_runtime_gate"], "pending")
                        self.assertEqual(
                            len(report["projections"]), 15 + (profile == "ffn15_fusion")
                        )
                        names = {t.name for t in GGUFReader(root / "output.gguf").tensors}
                        for name in data["projections"]:
                            self.assertNotIn(name + ".weight", names)
                        self.assertIn("output.weight", names)
                        self.assertIn("token_embd.weight", names)

    def test_rejects_bad_contract_and_nonfinite_negative_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, checkpoint, manifest, data = self.fixture(root)
            for key, value in [
                ("activation_bits", True),
                ("activation_bits", 4),
                ("profile", "head"),
                ("family", "eagle"),
            ]:
                bad = copy.deepcopy(data)
                bad[key] = value
                with self.assertRaises(ValueError):
                    check_manifest(bad, data["base_gguf_sha256"], data["checkpoint_sha256"])
            bad = copy.deepcopy(data)
            bad["projections"].pop(next(iter(bad["projections"])))
            with self.assertRaises(ValueError):
                check_manifest(bad, data["base_gguf_sha256"], data["checkpoint_sha256"])
            with np.load(checkpoint) as archive:
                arrays = {name: archive[name] for name in archive.files}
            for value in (-1, np.nan, np.inf):
                arrays["fc.scale"][0] = value
                np.savez(checkpoint, **arrays)
                data["checkpoint_sha256"] = sha256(checkpoint)
                manifest.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "invalid nonnegative"):
                    export_model(base, checkpoint, manifest, root / "bad.gguf")
                self.assertFalse((root / "bad.gguf").exists())

    @unittest.skipUnless(
        os.environ.get("BLOCK_TEST_NATIVE"), "actual native fixture binary not selected"
    )
    def test_actual_native_loader_injection_noise_graph(self):
        for family in ("dspark", "dflash"):
            for profile in ("ffn15", "ffn15_fusion"):
                for bits in (1, 8):
                    with (
                        self.subTest(family=family, profile=profile, bits=bits),
                        tempfile.TemporaryDirectory() as tmp,
                    ):
                        root = Path(tmp)
                        base, checkpoint, manifest, _ = self.fixture(root, family, profile, bits)
                        output = root / "binary.gguf"
                        export_model(base, checkpoint, manifest, output)
                        result = subprocess.run(
                            [os.environ["BLOCK_TEST_NATIVE"], str(output)],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            env={**os.environ, "GGML_W1AX_ACT_BITS": "16"},
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertIn("graph PASS", result.stdout)

    @unittest.skipUnless(
        os.environ.get("BLOCK_TEST_NATIVE"), "actual native fixture binary not selected"
    )
    def test_native_rejects_invalid_contract_and_dense_shadows(self):
        def rewrite(source, destination, changes=None, extra=None):
            reader = GGUFReader(source)
            writer = GGUFWriter(destination, "dflash")
            for key, field in reader.fields.items():
                if key.startswith("GGUF.") or key == "general.architecture":
                    continue
                value = (changes or {}).get(key, field.contents())
                writer.add_key_value(
                    key, value, field.types[0], field.types[-1] if len(field.types) > 1 else None
                )
            for tensor in reader.tensors:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            if extra:
                writer.add_tensor(extra, np.ones((64, 32), np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, checkpoint, manifest, _ = self.fixture(root)
            output = root / "binary.gguf"
            export_model(base, checkpoint, manifest, output)
            cases = [
                ({"dflash.w1ax.version": 2}, None),
                ({"dflash.w1ax.activation_bits": 4}, None),
                ({"dflash.w1ax.profile": "head"}, None),
                ({"dflash.block_size": 8}, None),
                ({"dflash.sample_from_anchor": False}, None),
                ({"dflash.w1ax.tensors": ["fc.weight"]}, None),
                ({}, "blk.0.ffn_up.weight"),
            ]
            for index, (changes, extra) in enumerate(cases):
                bad = root / f"bad{index}.gguf"
                rewrite(output, bad, changes, extra)
                result = subprocess.run(
                    [os.environ["BLOCK_TEST_NATIVE"], str(bad)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("error loading model", result.stderr)

    @unittest.skipUnless(
        os.environ.get("BLOCK_TEST_NATIVE"), "actual native fixture binary not selected"
    )
    def test_source_bound_proof_covers_all_selected_operations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, checkpoint, manifest, _ = self.fixture(root)
            model = root / "binary.gguf"
            export = root / "export.json"
            export.write_text(json.dumps(export_model(base, checkpoint, manifest, model)))
            proof = check(
                Path(os.environ["BLOCK_TEST_NATIVE"]),
                model,
                export,
                gpu_layers=0,
                require_cuda=False,
            )
            self.assertEqual(len(proof["nodes"]), 16)
            self.assertFalse(proof["throughput_or_quality_claim"])
            self.assertTrue(all(node["output_buffer"] == "CPU" for node in proof["nodes"]))
            with self.assertRaisesRegex(ValueError, "CUDA placement"):
                check(
                    Path(os.environ["BLOCK_TEST_NATIVE"]),
                    model,
                    export,
                    gpu_layers=0,
                    require_cuda=True,
                )

    def test_pack_zero_tail_and_little_order(self):
        latent = np.full((1, 33), -1, np.float32)
        latent[0, 0] = -0.0
        latent[0, 32] = 0
        self.assertEqual(pack(latent).view(np.uint32).tolist(), [[1, 1]])


if __name__ == "__main__":
    unittest.main()

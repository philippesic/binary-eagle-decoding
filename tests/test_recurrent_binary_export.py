"""CPU-only synthetic serialization gates for jointly learned binary EAGLE."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH, GroupedBinaryLinear
from w1a1_eagle.recurrent_training import save_training_checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_recurrent_binary import (  # noqa: E402
    PREFIX,
    SOURCE_NAMES,
    GGUFReader,
    GGUFWriter,
    Type,
    export_model,
    gguf_qk_row_order,
    raw_hash,
    sha256,
    tensor_key,
)


class RecurrentBinaryExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.base = self.directory / "f16.gguf"
        self.checkpoint = self.directory / "learned.npz"
        self.manifest = self.directory / "learned.json"
        self.output = self.directory / "binary.gguf"
        self.arrays = {}
        self.shapes = {}
        writer = GGUFWriter(self.base, "eagle3")
        writer.add_string("general.name", "synthetic frozen source")
        writer.add_array("tokenizer.ggml.tokens", ["zero", "one", "two"])
        writer.add_array("test.integer_list", [1, 4, 9])
        for index, (base, name) in enumerate(SOURCE_NAMES.items()):
            rows = 64 if base == "blk.0.attn_q" else 16 if base == "blk.0.attn_k" else 5
            width = 130
            shape = (rows, width)
            self.shapes[base] = shape
            # Unique row signs/scales make Q/K permutation observable.
            row = np.arange(rows, dtype=np.float32)[:, None]
            col = np.arange(width, dtype=np.float32)[None, :]
            latent = (((row * 7 + col * 3 + index) % 11) - 5).astype(np.float32)
            latent[0, 0] = 0
            scales = 0.1 + row * 0.01 + np.arange(2, dtype=np.float32)[None, :] * 0.5
            self.arrays[name + ".latent"] = latent
            self.arrays[name + ".scale"] = scales.astype(np.float32)
            writer.add_tensor(base + ".weight", np.full(shape, index, dtype=np.float16))
        writer.add_tensor("output_norm.weight", np.array([1, 2, 3], dtype=np.float32))
        writer.add_tensor("d2t", np.array([2, 0, 1], dtype=np.int32), raw_dtype=Type.I32)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        self.save_checkpoint()
        self.save_manifest()

    def save_checkpoint(self):
        np.savez(self.checkpoint, **self.arrays)

    def save_manifest(self, *, override=None):
        manifest = {
            "schema_version": 1,
            "base_gguf_sha256": sha256(self.base),
            "training_arithmetic": "native_order",
            "projections": {
                base: {"checkpoint_name": SOURCE_NAMES[base], "shape": list(shape)}
                for base, shape in self.shapes.items()
            },
        }
        if override:
            override(manifest)
        self.manifest.write_text(json.dumps(manifest))

    def export(self):
        return export_model(self.base, self.checkpoint, self.manifest, self.output)

    def test_all_nine_signs_scales_qk_order_and_frozen_tensors(self):
        before = GGUFReader(self.base)
        report = self.export()
        after = GGUFReader(self.output)
        tensors = {tensor.name: tensor for tensor in after.tensors}
        self.assertEqual(len(tensors), 20)
        self.assertEqual(after.fields[PREFIX + "scale_rule"].contents(), "f32_learned_nonnegative")
        self.assertEqual(after.fields[PREFIX + "scale_group_size"].contents(), 128)
        self.assertEqual(
            after.fields[PREFIX + "tensors"].contents(),
            sorted(base + ".weight" for base in SOURCE_NAMES),
        )
        for base, name in SOURCE_NAMES.items():
            latent = self.arrays[name + ".latent"]
            scales = self.arrays[name + ".scale"]
            if base == "blk.0.attn_q":
                latent, scales = gguf_qk_row_order(latent, 32), gguf_qk_row_order(scales, 32)
            elif base == "blk.0.attn_k":
                latent, scales = gguf_qk_row_order(latent, 8), gguf_qk_row_order(scales, 8)
            observed = tensors[base + ".w1a1_packed"]
            self.assertEqual(observed.tensor_type, Type.I32)
            self.assertEqual(observed.data.shape, (self.shapes[base][0], 5))
            bits = np.unpackbits(observed.data.view(np.uint8), axis=1, bitorder="little")
            np.testing.assert_array_equal(bits[:, :130], latent >= 0)
            np.testing.assert_array_equal(bits[:, 130:], 0)
            actual_scales = tensors[base + ".w1a1_scale"]
            self.assertEqual(actual_scales.tensor_type, Type.F32)
            np.testing.assert_array_equal(actual_scales.data, scales)
            self.assertEqual(after.fields[tensor_key(base, "logical_k")].contents(), 130)
            self.assertNotIn(base + ".weight", tensors)
            self.assertEqual(report["projections"][base]["packed_sha256"], raw_hash(observed.data))
        old = {tensor.name: tensor for tensor in before.tensors}
        for name in ("output_norm.weight", "d2t"):
            self.assertEqual(raw_hash(old[name].data), raw_hash(tensors[name].data))
            self.assertEqual(old[name].tensor_type, tensors[name].tensor_type)
        for key in ("general.name", "tokenizer.ggml.tokens", "test.integer_list"):
            self.assertEqual(before.fields[key].contents(), after.fields[key].contents())
        self.assertEqual(report["base_gguf"]["sha256"], sha256(self.base))
        self.assertEqual(report["checkpoint"]["sha256"], sha256(self.checkpoint))
        self.assertEqual(report["output"]["sha256"], sha256(self.output))
        self.assertEqual(report["training_arithmetic"], "native_order")
        self.assertTrue(report["serialization_audit_passed"])

    def test_nonfinite_negative_scale_shape_and_extra_checkpoint_key_are_rejected(self):
        name = SOURCE_NAMES["fc"]
        for value in (np.nan, np.inf, -0.25, -0.0):
            with self.subTest(value=value):
                original = self.arrays[name + ".scale"]
                self.arrays[name + ".scale"] = original.copy()
                self.arrays[name + ".scale"][0, 0] = value
                self.save_checkpoint()
                with self.assertRaisesRegex(ValueError, "scales must be finite and nonnegative"):
                    self.export()
                self.assertFalse(self.output.exists())
                self.arrays[name + ".scale"] = original
        original = self.arrays[name + ".scale"]
        self.arrays[name + ".scale"] = original[:, :1]
        self.save_checkpoint()
        with self.assertRaisesRegex(ValueError, "scale must be F32"):
            self.export()
        self.arrays[name + ".scale"] = original
        self.arrays["unexpected"] = np.array([1], np.float32)
        self.save_checkpoint()
        with self.assertRaisesRegex(ValueError, "exactly nine"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_manifest_name_shape_hash_and_missing_tensor_are_rejected(self):
        self.save_manifest(override=lambda m: m["projections"]["fc"].update(shape=[6, 130]))
        with self.assertRaisesRegex(ValueError, "declared shape"):
            self.export()
        self.save_manifest(
            override=lambda m: m["projections"]["fc"].update(checkpoint_name="wrong")
        )
        with self.assertRaisesRegex(ValueError, "checkpoint tensor name mismatch"):
            self.export()
        self.save_manifest(override=lambda m: m.update(base_gguf_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.export()
        self.save_manifest(override=lambda m: m.update(training_arithmetic="dense"))
        with self.assertRaisesRegex(ValueError, "training arithmetic"):
            self.export()
        del self.arrays[SOURCE_NAMES["fc"] + ".latent"]
        self.save_checkpoint()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "exactly nine"):
            self.export()
        self.assertFalse(self.output.exists())

    def test_extra_layer_or_binary_shadow_rejected(self):
        for extra in ("blk.1.attn_q.weight", "fc.w1a1_packed"):
            with self.subTest(extra=extra):
                path = self.directory / (extra.replace(".", "-") + ".gguf")
                writer = GGUFWriter(path, "eagle3")
                for base, shape in self.shapes.items():
                    writer.add_tensor(base + ".weight", np.zeros(shape, dtype=np.float16))
                writer.add_tensor(extra, np.ones((2, 2), dtype=np.float16))
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                self.base = path
                self.save_manifest()
                with self.assertRaisesRegex(ValueError, "extra decoder layer|binary shadows"):
                    self.export()
                self.assertFalse(self.output.exists())

    def test_cli_writes_audit_and_refuses_overwrite(self):
        audit = self.directory / "audit.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/export_recurrent_binary.py"),
            "--base",
            str(self.base),
            "--checkpoint",
            str(self.checkpoint),
            "--manifest",
            str(self.manifest),
            "--output",
            str(self.output),
            "--audit",
            str(audit),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(audit.read_text())["output"]["sha256"], sha256(self.output))
        with self.assertRaises(FileExistsError):
            self.export()

    def test_training_checkpoint_feeds_exporter(self):
        modules = {}
        for base, name in SOURCE_NAMES.items():
            modules[CANDIDATE_D_BASE_TO_PATH[base]] = GroupedBinaryLinear(
                torch.from_numpy(self.arrays[name + ".latent"].copy()),
                torch.from_numpy(self.arrays[name + ".scale"].copy()),
            )
        self.checkpoint.unlink()
        self.manifest.unlink()
        save_training_checkpoint(modules, sha256(self.base), self.checkpoint, self.manifest)
        report = self.export()
        self.assertEqual(len(report["projections"]), 9)
        self.assertEqual(report["checkpoint"]["sha256"], sha256(self.checkpoint))


if __name__ == "__main__":
    unittest.main()

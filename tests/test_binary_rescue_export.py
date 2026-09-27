"""Byte-level mixed GGUF and fitted-head export gates; no GPU required."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_binary_rescue import (  # noqa: E402
    BASES,
    GLOBAL_KEYS,
    GROUPS,
    PREFIX,
    SUBSETS,
    GGUFWriter,
    Model,
    Type,
    export_model,
    extract_head,
    inventory,
    quantize,
    raw_hash,
    same_tensor,
    tensor_key,
)


def fixture(path, kind, weights, *, corrupt=None, wrong_shape=None):
    writer = GGUFWriter(path, "eagle3")
    writer.add_string("general.name", "source identity")
    writer.add_uint32("general.file_type", 1)
    writer.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    writer.add_array("test.int_array", [1, 2, 3])
    if kind == "BINARY":
        writer.add_uint32(PREFIX + "version", 2)
        writer.add_uint32(PREFIX + "scale_group_size", 128)
        writer.add_array(PREFIX + "groups", list(GROUPS))
        writer.add_array(PREFIX + "tensors", sorted(b + ".weight" for b in BASES))
        for key, value in {
            "bit_order": "little",
            "sign_rule": "nonnegative_is_one",
            "scale_rule": "f32_nonnegative_least_squares",
            "arithmetic": "f32",
        }.items():
            writer.add_string(PREFIX + key, value)
    for base, weight in weights.items():
        if base == wrong_shape:
            weight = weight[:-1]
        if kind == "BINARY":
            packed = np.packbits(weight >= 0, axis=1, bitorder="little").view(np.int32)
            # Deliberately distinct fitted scales; must never be recomputed.
            scale = np.arange(weight.size // 128, dtype=np.float32).reshape(-1, 2) / 100 + 0.1
            writer.add_uint32(tensor_key(base, "logical_k"), weight.shape[1])
            writer.add_string(tensor_key(base, "packed"), base + ".w1a1_packed")
            writer.add_string(tensor_key(base, "scale"), base + ".w1a1_scale")
            writer.add_tensor(base + ".w1a1_packed", packed, raw_dtype=Type.I32)
            writer.add_tensor(base + ".w1a1_scale", scale, raw_dtype=Type.F32)
        else:
            data = weight if kind == Type.F16 else quantize(weight.astype(np.float32), kind)
            if base == corrupt:
                data = data.copy()
                data.view(np.uint8).flat[-1] ^= 1
            writer.add_tensor(base + ".weight", data, raw_dtype=kind)
    writer.add_tensor("output_norm.weight", np.ones(256, np.float32))
    writer.add_tensor("d2t", np.arange(16, dtype=np.int32), raw_dtype=Type.I32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class RescueExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        rng = np.random.default_rng(72)
        self.weights = {
            name: rng.normal(size=(16 if name == "output" else 4, 256)).astype(np.float16)
            for name in BASES
        }
        self.paths = {}
        for label, kind in (
            ("D", "BINARY"),
            ("F16", Type.F16),
            ("Q4_0", Type.Q4_0),
            ("Q8_0", Type.Q8_0),
        ):
            self.paths[label] = self.directory / (label + ".gguf")
            fixture(self.paths[label], kind, self.weights)

    def compose(self, label="fusion", *, base="D", donor="Q8_0", **kwargs):
        path = self.directory / (label + "-out.gguf")
        report = export_model(
            self.paths[base],
            path,
            donor_path=self.paths[donor],
            subsets=tuple(label.split(",")),
            expected_type="BINARY" if donor == "D" else donor,
            source_f16=self.paths["F16"],
            **kwargs,
        )
        return Model(path), report

    def test_four_rescues_and_combo_preserve_every_unaffected_byte_and_metadata(self):
        original = Model(self.paths["D"])
        donor = Model(self.paths["Q8_0"])
        for subset in ("fusion", "attention", "ffn_down", "head", "attention,head"):
            with self.subTest(subset=subset):
                model, report = self.compose(subset)
                selected = {b for s in subset.split(",") for b in SUBSETS[s]}
                self.assertEqual(model.value(PREFIX + "version"), 3)
                self.assertEqual(model.value(PREFIX + "groups"), list(GROUPS))
                self.assertEqual(
                    model.value(PREFIX + "dense_tensors"), sorted(b + ".weight" for b in selected)
                )
                self.assertEqual(model.value(PREFIX + "dense_types"), ["Q8_0"] * len(selected))
                self.assertEqual(
                    set(model.value(PREFIX + "tensors")),
                    {b + ".weight" for b in BASES if b not in selected},
                )
                for key in GLOBAL_KEYS:
                    self.assertEqual(original.value(PREFIX + key), model.value(PREFIX + key))
                for base in BASES:
                    if base in selected:
                        self.assertTrue(
                            same_tensor(
                                model.tensors[base + ".weight"], donor.tensors[base + ".weight"]
                            )
                        )
                        self.assertNotIn(base + ".w1a1_packed", model.tensors)
                        self.assertNotIn(base + ".w1a1_scale", model.tensors)
                        self.assertNotIn(tensor_key(base, "logical_k"), model.fields)
                    else:
                        for tensor in original.projection(base)[0]:
                            self.assertTrue(same_tensor(tensor, model.tensors[tensor.name]))
                for name in ("output_norm.weight", "d2t"):
                    self.assertTrue(same_tensor(original.tensors[name], model.tensors[name]))
                for key in ("general.name", "test.int_array", "tokenizer.ggml.tokens"):
                    self.assertEqual(original.value(key), model.value(key))
                self.assertTrue(report["serialization_audit_passed"])
                self.assertEqual(
                    set(report["reference_quantization_checked_tensors"]),
                    {b + ".weight" for b in selected},
                )
                json.dumps(report)  # manifest values must be plain JSON

    def test_crossed_q4_body_binary_head_and_dense_head_controls(self):
        crossed, report = self.compose("head", base="Q4_0", donor="D")
        self.assertEqual(crossed.value(PREFIX + "tensors"), ["output.weight"])
        self.assertEqual(crossed.value(PREFIX + "dense_types"), ["Q4_0"] * 8)
        self.assertEqual(report["inventory"]["binary_parameters"], self.weights["output"].size)
        for donor in ("F16", "Q4_0"):
            (self.directory / "head-out.gguf").unlink()
            crossed, _ = self.compose("head", donor=donor)
            self.assertEqual(crossed.value(PREFIX + "dense_types"), [donor])

    def test_fitted_head_export_reload_bit_equality_and_initialization_extraction(self):
        path = self.directory / "initial.npy"
        extracted = extract_head(self.paths["F16"], path)
        self.assertEqual(extracted["raw_sha256"], raw_hash(self.weights["output"]))
        np.testing.assert_array_equal(np.load(path), self.weights["output"])
        weight = (self.weights["output"].astype(np.float32) + 0.01).astype(np.float16)
        np.save(path, weight)
        report = export_model(self.paths["D"], self.directory / "fit.gguf", head_npy=path)
        model = Model(self.directory / "fit.gguf")
        self.assertEqual(raw_hash(model.tensors["output.weight"].data), raw_hash(weight))
        self.assertEqual(model.value(PREFIX + "dense_types"), ["F16"])
        self.assertEqual(len(report["unchanged_tensors"]), 18)  # eight pairs plus norm/map
        for invalid in (weight.astype(np.float32), weight[:-1], np.full_like(weight, np.nan)):
            np.save(path, invalid)
            with self.assertRaises(ValueError):
                export_model(self.paths["D"], self.directory / "invalid.gguf", head_npy=path)
            self.assertFalse((self.directory / "invalid.gguf").exists())

    def test_inventory_includes_group_scales_dense_exceptions_and_integer_maps(self):
        model, report = self.compose("head")
        result = report["inventory"]
        projections = {p["name"]: p for p in result["projections"]}
        self.assertEqual(projections["output.weight"]["effective_bits"], 8.5)
        self.assertEqual(projections["fc.weight"]["effective_bits"], 1.25)
        selected = sum(w.size for w in self.weights.values())
        self.assertEqual(result["selected_linear_parameters"], selected)
        self.assertEqual(result["model_parameters_excluding_integer_maps"], selected + 256)
        expected_bytes = sum(t.data.nbytes for t in model.tensors.values())
        self.assertEqual(
            result["tensor_payload_bytes_including_scales_and_integer_maps"], expected_bytes
        )
        self.assertEqual(
            result["model_payload_effective_bits"], 8 * expected_bytes / (selected + 256)
        )
        self.assertEqual({e["type"] for e in result["nonselected_tensors"]}, {"F32", "I32"})
        self.assertTrue(all(not p["runtime_verified"] for p in result["projections"]))
        self.assertEqual(inventory(model), result)

    def test_corrupt_donor_and_wrong_type_shape_or_source_are_rejected(self):
        for defect in ("corrupt", "wrong_shape"):
            fixture(self.paths["Q8_0"], Type.Q8_0, self.weights, **{defect: "fc"})
            with self.assertRaises(ValueError):
                self.compose()
            self.assertFalse((self.directory / "fusion-out.gguf").exists())
        fixture(self.paths["Q8_0"], Type.F16, self.weights)
        with self.assertRaisesRegex(ValueError, "expected Q8_0"):
            self.compose()
        with self.assertRaisesRegex(ValueError, "requires expected-type"):
            export_model(
                self.paths["D"],
                self.directory / "missing.gguf",
                donor_path=self.paths["Q4_0"],
                subsets=("head",),
                expected_type="Q4_0",
            )

    def test_binary_donor_cannot_hide_nonoriginal_dense_q4_body(self):
        fixture(self.paths["Q4_0"], Type.Q4_0, self.weights, corrupt="blk.0.attn_k")
        with self.assertRaisesRegex(ValueError, "reference quantization"):
            self.compose("head", base="Q4_0", donor="D")

    def test_no_overwrite_or_overlapping_subset(self):
        self.compose()
        with self.assertRaises(FileExistsError):
            self.compose()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.compose("ffn,ffn_down")

    def test_mislabeled_v3_dense_type_is_rejected_on_reload(self):
        model, _ = self.compose()
        payload = model.path.read_bytes()
        self.assertEqual(payload.count(b"Q8_0"), 1)
        model.path.write_bytes(payload.replace(b"Q8_0", b"Q4_0"))
        with self.assertRaisesRegex(ValueError, "declared dense type"):
            Model(model.path)

    def test_unknown_packed_projection_is_rejected(self):
        payload = self.paths["D"].read_bytes().replace(b"fc.w1a1_packed", b"xx.w1a1_packed")
        self.paths["D"].write_bytes(payload)
        with self.assertRaisesRegex(ValueError, "outside the known EAGLE"):
            Model(self.paths["D"])

    def test_cli_extract_and_compose_manifests(self):
        output = self.directory / "cli.gguf"
        audit = self.directory / "cli.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/export_binary_rescue.py"),
            "--base",
            str(self.paths["D"]),
            "--donor",
            str(self.paths["Q8_0"]),
            "--source-f16",
            str(self.paths["F16"]),
            "--subset",
            "attention",
            "--expected-type",
            "Q8_0",
            "--output",
            str(output),
            "--audit",
            str(audit),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertTrue(json.loads(audit.read_text())["serialization_audit_passed"])
        extraction = [
            sys.executable,
            str(ROOT / "scripts/export_binary_rescue.py"),
            "--extract-head",
            str(self.paths["F16"]),
            "--output",
            str(self.directory / "head.npy"),
            "--audit",
            str(self.directory / "head.json"),
        ]
        subprocess.run(extraction, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads((self.directory / "head.json").read_text())["type"], "F16")


if __name__ == "__main__":
    unittest.main()

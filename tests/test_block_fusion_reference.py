"""Real GGUF byte parsing on small synthetic model-source fixtures, CPU only."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, os.environ.get("EAGLE_GGUF_PY", str(ROOT / "third_party/llama.cpp/gguf-py")))

from extract_block_fusion_reference import extract_reference  # noqa: E402
from gguf import GGMLQuantizationType, GGUFWriter  # noqa: E402

from w1a1_eagle.block_data import file_sha256  # noqa: E402
from w1a1_eagle.block_fusion import (  # noqa: E402
    canonical_norm_epsilon,
    norm_epsilon_bits,
    validate_norm_descriptor,
)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "source.gguf"
        self.weight = np.arange(20, dtype=np.float16).reshape(2, 10) / np.float16(20)
        self.gamma = np.array([1.0, 0.5], dtype=np.float32)
        self.write_model()

    def write_model(
        self, family="dflash", epsilon=1e-6, wrong_type=False, pruned=False, bf16=False
    ):
        writer = GGUFWriter(self.model, "dflash")
        if wrong_type:
            writer.add_float64("dflash.attention.layer_norm_rms_epsilon", epsilon)
        else:
            writer.add_float32("dflash.attention.layer_norm_rms_epsilon", epsilon)
        writer.add_array("tokenizer.ggml.tokens", ["token" + str(i) for i in range(32)])
        if bf16:
            self.bf16_weight_bits = (self.weight.astype(np.float32).view(np.uint32) >> 16).astype(
                np.uint16
            )
            self.bf16_gamma_bits = (self.gamma.view(np.uint32) >> 16).astype(np.uint16)
            writer.add_tensor(
                "fc.weight", self.bf16_weight_bits, raw_dtype=GGMLQuantizationType.BF16
            )
            writer.add_tensor(
                "enc.output_norm.weight", self.bf16_gamma_bits, raw_dtype=GGMLQuantizationType.BF16
            )
        else:
            writer.add_tensor("fc.weight", self.weight)
            writer.add_tensor("enc.output_norm.weight", self.gamma)
        if family == "dspark":
            writer.add_tensor("markov_w1.weight", np.ones((32, 2), dtype=np.float16))
        if pruned:
            writer.add_tensor("d2t", np.arange(8, dtype=np.int32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def extract(self, **kwargs):
        return extract_reference(
            self.model,
            model_sha256=file_sha256(self.model),
            family="dflash",
            output_dir=self.root / "reference",
            **kwargs,
        )

    def validate(self, metadata, *, supplied=1e-6):
        return validate_norm_descriptor(
            metadata,
            family="dflash",
            weights_sha256=metadata["weights_sha256"],
            norm_sha256=metadata["norm_sha256"],
            supplied_epsilon=supplied,
        )

    def test_actual_dense_payloads_and_epsilon_f32_roundtrip(self):
        metadata = self.extract()
        np.testing.assert_array_equal(
            np.load(self.root / "reference/fc-weight.npy"), self.weight.astype(np.float32)
        )
        np.testing.assert_array_equal(np.load(self.root / "reference/fc-norm.npy"), self.gamma)
        self.assertEqual(metadata["fc_source"]["shape"], [2, 10])
        self.assertEqual(metadata["epsilon_f32_bits"], norm_epsilon_bits(1e-6))
        self.assertEqual(self.validate(metadata), float(np.float32(1e-6)))
        self.assertEqual(json.loads((self.root / "reference/reference.json").read_text()), metadata)

    def test_bf16_raw_bitshift_exact_without_f16_roundtrip(self):
        self.write_model(bf16=True)
        metadata = self.extract()
        weight = (self.bf16_weight_bits.astype(np.uint32) << 16).view(np.float32)
        gamma = (self.bf16_gamma_bits.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(
            np.load(self.root / "reference/fc-weight.npy").view(np.uint32), weight.view(np.uint32)
        )
        np.testing.assert_array_equal(
            np.load(self.root / "reference/fc-norm.npy").view(np.uint32), gamma.view(np.uint32)
        )
        self.assertEqual(metadata["fc_source"]["kind"], "BF16")
        self.assertEqual(metadata["norm_source"]["kind"], "BF16")
        self.validate(metadata)

    def test_epsilon_one_ulp_difference_refuses(self):
        metadata = self.extract()
        wrong = np.nextafter(np.float32(1e-6), np.float32(np.inf))
        with self.assertRaisesRegex(ValueError, "FLOAT32 bits"):
            self.validate(metadata, supplied=float(wrong))

    def test_same_native_f32_value_not_broad_tolerance(self):
        metadata = self.extract()
        same_rounding = float(np.float32(1e-6)) + 1e-16
        self.assertEqual(self.validate(metadata, supplied=same_rounding), float(np.float32(1e-6)))
        with self.assertRaisesRegex(ValueError, "FLOAT32 bits"):
            self.validate(metadata, supplied=1.001e-6)

    def test_wrong_gamma_or_weight_source_refuses(self):
        metadata = self.extract()
        with self.assertRaisesRegex(ValueError, "source differs"):
            validate_norm_descriptor(
                metadata,
                family="dflash",
                weights_sha256="a" * 64,
                norm_sha256=metadata["norm_sha256"],
                supplied_epsilon=1e-6,
            )
        with self.assertRaisesRegex(ValueError, "source differs"):
            validate_norm_descriptor(
                metadata,
                family="dflash",
                weights_sha256=metadata["weights_sha256"],
                norm_sha256="b" * 64,
                supplied_epsilon=1e-6,
            )

    def test_source_family_and_full_vocab_refuse(self):
        self.write_model(family="dspark")
        with self.assertRaisesRegex(ValueError, "family"):
            self.extract()
        self.write_model(pruned=True)
        with self.assertRaisesRegex(ValueError, "full-vocabulary"):
            self.extract()

    def test_wrong_source_hash_refuses(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            extract_reference(
                self.model,
                model_sha256="a" * 64,
                family="dflash",
                output_dir=self.root / "reference",
            )

    def test_native_scalar_dtype_and_invalid_epsilon_refuse(self):
        self.write_model(wrong_type=True)
        with self.assertRaisesRegex(ValueError, "FLOAT32"):
            self.extract()
        for value in (True, np.bool_(False), np.nan, np.inf, 0.0, -1e-6, 1e100, 1e-100):
            with self.assertRaises(ValueError):
                canonical_norm_epsilon(value)

    def test_extraction_cap_and_no_overwrite(self):
        with self.assertRaisesRegex(MemoryError, "bound"):
            self.extract(max_array_bytes=1)
        self.extract()
        with self.assertRaisesRegex(ValueError, "history"):
            self.extract()

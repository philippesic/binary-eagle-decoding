"""CPU-only candidate-D GGUF initialization audit gates."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))

from audit_eagle_w1a1_gguf import SOURCE_NAMES  # noqa: E402
from fit_binary_scales import export_variant, pack_signs  # noqa: E402
from gguf import GGMLQuantizationType as Type  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402
from load_recurrent_binary_init import load_candidate_d_arrays, sha256  # noqa: E402


class RecurrentBinaryInitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        directory = Path(self.temporary.name)
        base = directory / "f16.gguf"
        self.draft = directory / "D.gguf"
        writer = GGUFWriter(base, "eagle3")
        writer.add_tensor("d2t", np.array([2, 4, 6], dtype=np.int64), raw_dtype=Type.I64)
        self.arrays = {}
        for index, name in enumerate(SOURCE_NAMES):
            weight = np.arange(260, dtype=np.float32).reshape(2, 130) - 40 - index
            writer.add_tensor(name + ".weight", weight.astype(np.float16))
            packed = pack_signs(weight)
            scales = np.ones((2, 2), dtype=np.float32) * (index + 1) / 10
            self.arrays[name] = {
                "K": 130,
                "packed": packed,
                "A": np.ones(2, dtype=np.float32),
                "B": scales,
                "C": np.ones(2, dtype=np.float32),
                "D": scales,
            }
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        export_variant(base, self.draft, self.arrays, "D")

    def test_loads_exact_nine_layer_frozen_d(self):
        arrays, report = load_candidate_d_arrays(self.draft, sha256(self.draft))
        self.assertEqual(len(arrays), 9)
        self.assertEqual(report["draft_vocab_size"], 3)
        for name in SOURCE_NAMES:
            np.testing.assert_array_equal(arrays[name][0], self.arrays[name]["packed"])
            np.testing.assert_array_equal(arrays[name][1], self.arrays[name]["D"])
            self.assertEqual(report["layers"][name]["logical_k"], 130)

    def test_rejects_wrong_hash_and_nonzero_tail_bits(self):
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_candidate_d_arrays(self.draft, "0" * 64)
        reader = GGUFReader(self.draft, mode="r+")
        packed = next(tensor for tensor in reader.tensors if tensor.name == "fc.w1a1_packed")
        packed.data[0, -1] |= np.int32(-2147483648)
        del reader
        with self.assertRaisesRegex(ValueError, "tail bits"):
            load_candidate_d_arrays(self.draft, sha256(self.draft))


if __name__ == "__main__":
    unittest.main()

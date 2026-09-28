"""Native CPU loader gates for mixed binary/Q8 EAGLE GGUFs.

Build llama.cpp's shared CPU library first, then run:
    LLAMA_TEST_BUILD_DIR=third_party/llama.cpp/build-cpu \\
        python -m unittest discover -s tests -p test_binary_rescue_loader.py -v

Requires numpy, a C++ compiler, and the vendored gguf-py. Fixtures and the small
loader executable are generated in a temporary directory; no weights are read.
Without an explicit build directory, discovery skips if build-cpu is absent.
An explicitly configured missing library or a compile failure is an error.
"""

import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "third_party" / "llama.cpp"
sys.path.insert(0, str(RUNTIME / "gguf-py"))
import gguf  # noqa: E402

LINEARS = {
    "fc.weight": (32, 96),
    "blk.0.attn_q.weight": (32, 64),
    "blk.0.attn_k.weight": (32, 64),
    "blk.0.attn_v.weight": (32, 64),
    "blk.0.attn_output.weight": (32, 32),
    "blk.0.ffn_gate.weight": (64, 32),
    "blk.0.ffn_down.weight": (32, 64),
    "blk.0.ffn_up.weight": (64, 32),
    "output.weight": (64, 32),
}
LOADER_SOURCE = r"""
#include "llama.h"
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], params);
    const int result = model ? 0 : 1;
    if (model) llama_model_free(model);
    llama_backend_free();
    return result;
}
"""


def write_fixture(path, variant, activation_bits=16):
    """One 32-wide block, with only the requested malformed contract changed."""
    row_scales = variant.startswith("row_")
    learned = variant == "v2_learned" or row_scales
    version = (
        2 if variant in ("v2", "v2_learned", "unknown_scale") or variant.startswith("row_v2")
        else 3
    )
    writer = gguf.GGUFWriter(path, "eagle3")
    for key, value in {
        "context_length": 128,
        "embedding_length": 32,
        "feed_forward_length": 64,
        "block_count": 1,
        "attention.head_count": 1,
        "attention.head_count_kv": 1,
        "vocab_size": 64,
        "target_hidden_size": 32,
    }.items():
        writer.add_uint32(f"eagle3.{key}", value)
    writer.add_float32("eagle3.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_array("eagle3.target_layers", [0, 1, 2])
    writer.add_string("tokenizer.ggml.model", "none")
    writer.add_uint32(
        "eagle3.w1a1.version", version
    )
    writer.add_array("eagle3.w1a1.groups", ["fusion", "attention", "ffn", "head"])
    dense = [] if version == 2 else ["blk.0.ffn_down.weight"]
    packed = [name for name in LINEARS if name not in dense]
    if variant == "overlap":
        packed += dense
    if variant == "omitted":
        packed.remove("fc.weight")
    writer.add_array("eagle3.w1a1.tensors", packed)
    if dense:
        writer.add_array("eagle3.w1a1.dense_tensors", dense)
        writer.add_array("eagle3.w1a1.dense_types", ["F16" if variant == "wrong_type" else "Q8_0"])
    for key, value in {
        "bit_order": "little",
        "sign_rule": "nonnegative_is_one",
        "scale_rule": (
            "f32_learned_nonnegative"
            if learned
            else "unrecognized_scale"
            if variant == "unknown_scale"
            else "f32_nonnegative_least_squares"
        ),
        "arithmetic": "f32",
    }.items():
        writer.add_string(f"eagle3.w1a1.{key}", value)
    writer.add_uint32("eagle3.w1a1.scale_group_size", 0 if row_scales else 128)
    if row_scales:
        writer.add_uint32("eagle3.w1a1.activation_bits", activation_bits)
    for name, (rows, width) in LINEARS.items():
        if name in dense:
            quant_type = gguf.GGMLQuantizationType.Q8_0
            values = gguf.quantize(np.ones((rows, width), np.float32), quant_type)
            writer.add_tensor(name, values, raw_dtype=quant_type)
        else:
            packed_name = name.removesuffix(".weight") + ".w1a1_packed"
            scale_name = name.removesuffix(".weight") + ".w1a1_scale"
            audit = "eagle3.w1a1.tensor." + name.replace(".", "_")
            writer.add_tensor(packed_name, np.full((rows, (width + 31) // 32), -1, np.int32))
            scale_shape = (rows,) if row_scales else (rows, (width + 127) // 128)
            writer.add_tensor(scale_name, np.ones(scale_shape, np.float32))
            writer.add_uint32(audit + ".logical_k", width)
            writer.add_string(audit + ".packed", packed_name)
            writer.add_string(audit + ".scale", scale_name)
    if variant == "shadow":
        writer.add_tensor("fc.weight", np.ones(LINEARS["fc.weight"], np.float16))
    for name in [
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_norm_2.weight",
        "blk.0.ffn_norm.weight",
    ]:
        writer.add_tensor(name, np.ones(32, np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class BinaryRescueNativeLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configured = os.environ.get("LLAMA_TEST_BUILD_DIR")
        build = Path(configured).expanduser().resolve() if configured else RUNTIME / "build-cpu"
        library_dir = build / "bin"
        libraries = list(library_dir.glob("libllama*.dylib")) + list(
            library_dir.glob("libllama.so*")
        )
        if not libraries:
            message = f"Build llama.cpp's shared CPU library first: no libllama in {library_dir}"
            if configured:
                raise RuntimeError(message)
            raise unittest.SkipTest(message)
        cls.temp = tempfile.TemporaryDirectory(prefix="binary-rescue-loader-")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        source = cls.directory / "loader.cpp"
        source.write_text(LOADER_SOURCE)
        cls.loader = cls.directory / "loader"
        command = shlex.split(os.environ.get("CXX", "c++")) + [
            "-std=c++17",
            "-I",
            str(RUNTIME / "include"),
            "-I",
            str(RUNTIME / "ggml" / "include"),
            str(source),
            "-L",
            str(library_dir),
            "-lllama",
            f"-Wl,-rpath,{library_dir}",
            "-o",
            str(cls.loader),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(
                f"Native loader compilation failed:\n{result.stdout}\n{result.stderr}"
            )

    def assert_load(self, variant, expected_error=None, activation_bits=16, metadata_bits=None):
        path = self.directory / f"{variant}.gguf"
        write_fixture(path, variant, activation_bits if metadata_bits is None else metadata_bits)
        result = subprocess.run(
            [str(self.loader), str(path)],
            env={
                **os.environ,
                "GGML_W1AX_ACT_BITS": str(activation_bits),
                "CUDA_VISIBLE_DEVICES": "",
                "GGML_EAGLE_PRUNE_UNUSED_HEAD": "0",
            },
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 1 if expected_error else 0, output)
        if expected_error:
            self.assertIn(expected_error, output)
        elif variant == "valid":
            self.assertIn(
                "dense loaded blk.0.ffn_down.weight (type=q8_0, K=64, rows=32, bytes=2176)", output
            )

    def test_loads_v3_down_only_q8(self):
        self.assert_load("valid")

    def test_loads_unchanged_v2_binary(self):
        self.assert_load("v2")

    def test_loads_learned_nonnegative_v2_binary(self):
        self.assert_load("v2_learned")

    def test_learned_row_scales_allow_all_activation_widths(self):
        for version in (2, 3):
            for bits in (16, 8, 4, 1):
                with self.subTest(version=version, bits=bits):
                    self.assert_load(f"row_v{version}", activation_bits=bits)

    def test_group128_rejects_lower_activation_widths(self):
        for bits in (8, 4, 1):
            with self.subTest(bits=bits):
                self.assert_load("v2_learned", "group128 with explicit A16", activation_bits=bits)

    def test_rejects_row_activation_width_mismatch(self):
        self.assert_load(
            "row_v2", "activation bits metadata does not match",
            activation_bits=8, metadata_bits=4,
        )

    def test_rejects_unknown_scale_provenance(self):
        self.assert_load("unknown_scale", "unsupported or incomplete audit record")

    def test_rejects_wrong_dense_type(self):
        self.assert_load("wrong_type", "dense type or shape does not match audit metadata")

    def test_rejects_dense_packed_overlap(self):
        self.assert_load("overlap", "invalid dense exception")

    def test_rejects_omitted_coverage(self):
        self.assert_load("omitted", "unsupported or incomplete audit record")

    def test_rejects_dense_shadow_of_packed_tensor(self):
        self.assert_load("shadow", "must omit its dense shadow")


if __name__ == "__main__":
    unittest.main()

"""Independent small-row oracle for the real-head binary fixture format."""

import hashlib
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.create_real_head_fixture import (  # noqa: E402
    HEADER,
    MAGIC,
    VERSION,
    build_payload,
    create_fixture,
)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def pack_dense(row):
    words = [0] * ((len(row) + 31) // 32)
    for feature, value in enumerate(row):
        if value >= 0:
            words[feature // 32] |= 1 << (feature % 32)
    return words


class TestRealHeadFixture(unittest.TestCase):
    def setUp(self):
        self.k = 35
        self.dense_weights = [
            [1 if i % 2 == 0 else -1 for i in range(self.k)],
            [-1] * 32 + [1, -1, 1],
            [1] * self.k,
        ]
        self.weights = np.asarray([pack_dense(row) for row in self.dense_weights], dtype="<u4")
        # Padding dirt is deliberately ignored by K-limited XOR/popcount.
        self.weights[:, 1] |= np.uint32(0xFFFF_FFF8)
        self.weight_scales = np.asarray([0.125, 0.5, 1.25], dtype="<f4")
        first = [0.0, -0.0, -1.5, 3.0] + [(-1) ** i * 0.25 for i in range(31)]
        last = [-2.0] * 32 + [0.0, -0.0, 4.0]
        self.activations = np.asarray([first, last], dtype="<f4")

    def test_binary_payload_against_dense_sign_oracle(self):
        indices = np.asarray([0, 4], dtype="<i4")
        payload, summary = build_payload(
            indices, self.activations, self.weights, self.weight_scales, self.k
        )
        self.assertEqual(summary["selected_capture_indices"], [0, 4])
        self.assertEqual(summary["payload_bytes"], len(payload))
        offset = 0

        def take(dtype, shape):
            nonlocal offset
            count = math.prod(shape)
            out = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape)
            offset += count * 4
            return out

        np.testing.assert_array_equal(take("<i4", (2,)), indices)
        np.testing.assert_array_equal(take("<u4", (3, 2)), self.weights)
        np.testing.assert_array_equal(take("<f4", (3,)), self.weight_scales)
        np.testing.assert_array_equal(take("<f4", (2, 35)), self.activations)
        got_packed = take("<u4", (2, 2))
        for token in range(2):
            self.assertEqual(got_packed[token].tolist(), pack_dense(self.activations[token]))
        got_scales = take("<f4", (2,))
        got_dots = take("<i4", (2, 3))
        got_outputs = take("<f4", (2, 3))
        self.assertEqual(offset, len(payload))
        for token, activation in enumerate(self.activations):
            scale = f32(math.fsum(abs(float(value)) for value in activation) / self.k)
            self.assertEqual(float(got_scales[token]), scale)
            for row, dense_weight in enumerate(self.dense_weights):
                expected_dot = sum(
                    1 if (float(x) >= 0) == (weight >= 0) else -1
                    for x, weight in zip(activation, dense_weight, strict=True)
                )
                self.assertEqual(int(got_dots[token, row]), expected_dot)
                expected_output = f32(f32(expected_dot * float(self.weight_scales[row])) * scale)
                self.assertEqual(float(got_outputs[token, row]), expected_output)

    def test_fixture_header_hashes_and_full_length(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            capture = root / "train.pt"
            gguf = root / "head.gguf"
            output = root / "fixture.bin"
            rows = torch.zeros((5, self.k), dtype=torch.bfloat16)
            rows[0] = torch.from_numpy(self.activations[0].copy()).to(torch.bfloat16)
            rows[4] = torch.from_numpy(self.activations[1].copy()).to(torch.bfloat16)
            torch.save({"inputs": rows, "rows": []}, capture)
            gguf.write_bytes(b"synthetic GGUF stand-in")
            with patch(
                "scripts.create_real_head_fixture.load_packed_head",
                return_value=(self.weights, self.weight_scales, self.k, 1),
            ):
                result = create_fixture(capture, gguf, output, tokens=2, expected_shape=(3, self.k))
            raw = output.read_bytes()
            magic, version, header_bytes, k, rows_count, tokens, words, cap_hash, gguf_hash = (
                HEADER.unpack_from(raw)
            )
            self.assertEqual((magic, version, header_bytes), (MAGIC, VERSION, HEADER.size))
            self.assertEqual((k, rows_count, tokens, words), (35, 3, 2, 2))
            self.assertEqual(cap_hash, hashlib.sha256(capture.read_bytes()).digest())
            self.assertEqual(gguf_hash, hashlib.sha256(gguf.read_bytes()).digest())
            self.assertEqual(result["fixture_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(len(raw), HEADER.size + result["payload_bytes"])
            manifest = output.with_name(output.name + ".json")
            self.assertTrue(manifest.is_file())
            self.assertEqual(result["manifest_path"], str(manifest))
            self.assertEqual(
                json.loads(manifest.read_text())["fixture_sha256"], result["fixture_sha256"]
            )
            self.assertEqual(result["versions"]["packed_head_contract_version"], 1)
            self.assertEqual(result["gguf_sha256"], hashlib.sha256(gguf.read_bytes()).hexdigest())
            self.assertEqual(
                result["reference_arithmetic"]["integer_dot"],
                "K - 2 * popcount(masked XOR), exact int32",
            )
            self.assertEqual(
                np.frombuffer(raw, dtype="<i4", count=2, offset=HEADER.size).tolist(), [0, 4]
            )
            with self.assertRaisesRegex(ValueError, "already exist"):
                with patch("scripts.create_real_head_fixture.load_packed_head"):
                    create_fixture(capture, gguf, output, tokens=2, expected_shape=(3, self.k))

    def test_rejects_invalid_shapes_and_nonfinite_inputs(self):
        with self.assertRaisesRegex(ValueError, "invalid shape"):
            build_payload(
                np.asarray([0], dtype=np.int32),
                self.activations,
                self.weights,
                self.weight_scales,
                self.k,
            )
        bad = self.activations.copy()
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            build_payload(np.asarray([0, 1]), bad, self.weights, self.weight_scales, self.k)


if __name__ == "__main__":
    unittest.main()

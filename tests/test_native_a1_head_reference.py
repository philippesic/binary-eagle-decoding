"""CPU-only integer reference for the native A1 head comparison operand."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_continuous_w1ax_readiness as checker
from w1a1_eagle.recurrent_qat import hard_activation


def pack(signs):
    packed_bytes = np.packbits(signs > 0, axis=1, bitorder="little")
    padded = np.zeros((len(signs), ((signs.shape[1] + 31) // 32) * 4), np.uint8)
    padded[:, :packed_bytes.shape[1]] = packed_bytes
    return padded.view("<u4")


def integer_reference(signs, scales, state):
    state = np.asarray(state, dtype=np.float32)
    raw = state.view(np.uint32)
    positive = ((raw & 0x80000000) == 0) | ((raw & 0x7fffffff) == 0)
    codes = [1 if yes else -1 for yes in positive]
    scale = np.float32(np.mean(np.abs(state).astype(np.float64)))
    return np.array([
        np.float32(np.float32(np.float32(sum(int(w) * a for w, a in zip(row, codes)))
                             * np.float32(weight_scale)) * scale)
        for row, weight_scale in zip(signs, scales)
    ], dtype=np.float32)


class NativeA1HeadReferenceTests(unittest.TestCase):
    def setUp(self):
        def forbidden(*args, **kwargs):
            raise AssertionError("accelerator API forbidden in CPU head reference tests")
        for name in ("is_available", "device_count", "get_device_name", "manual_seed_all", "empty_cache"):
            guard = patch.object(torch.cuda, name, side_effect=forbidden)
            guard.start()
            self.addCleanup(guard.stop)

    def signs(self, k, rows=8):
        generator = np.random.default_rng(1701)
        base = np.concatenate((np.ones(k // 2, np.int8), -np.ones(k - k // 2, np.int8)))
        return np.stack([generator.permutation(base) for _ in range(rows)])

    def test_balanced_k2560_zero_dots_remain_exact_zero(self):
        signs = self.signs(2560)
        state = np.full(2560, .1, np.float32)
        scales = np.array([.01, 1., .3, .75, 2., .125, .007, 0.], np.float32)
        actual = checker._replay_head(pack(signs), scales, 2560, torch.from_numpy(state), 1)
        np.testing.assert_array_equal(actual.numpy(), integer_reference(signs, scales, state))
        np.testing.assert_array_equal(actual.numpy(), np.zeros(8, np.float32))
        self.assertEqual(actual.dtype, torch.float32)

    def test_signed_zero_negative_subnormal_and_nonunit_scale_order(self):
        signs = self.signs(2560)
        state = np.resize(np.array([.1, -.3, 0., -0., .25], np.float32), 2560)
        state[0] = -np.nextafter(np.float32(0), np.float32(1))
        scales = np.array([.01, 1., .3, .75, 2., .125, .007, 0.], np.float32)
        # The A1 reference explicitly returns F32 even under a F64 default/input.
        previous = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous)
        torch.set_default_dtype(torch.float64)
        actual = checker._replay_head(pack(signs), scales, 2560, torch.from_numpy(state.astype(np.float64)), 1)
        self.assertEqual(actual.dtype, torch.float32)
        np.testing.assert_array_equal(actual.numpy(), integer_reference(signs, scales, state))
        state.fill(0.)
        state[::2] = -0.
        zeros = checker._replay_head(pack(signs), scales, 2560, torch.from_numpy(state), 1)
        np.testing.assert_array_equal(zeros.numpy(), integer_reference(signs, scales, state))

    def test_chunk_boundary_and_logical_tail(self):
        signs = self.signs(33, rows=4099)
        packed = pack(signs)
        packed[:, -1] |= np.uint32(0xfffffffe)  # unused tail bits must not contribute
        state = np.resize(np.array([-.3, 0., .1], np.float32), 33)
        scales = np.linspace(.01, 1., 4099, dtype=np.float32)
        actual = checker._replay_head(packed, scales, 33, torch.from_numpy(state), 1)
        np.testing.assert_array_equal(actual.numpy(), integer_reference(signs, scales, state))

    def test_other_width_replays_keep_scaled_activation_dot_order(self):
        signs = self.signs(2560)
        state = torch.from_numpy(np.resize(np.array([.1, -.3, .25], np.float32), 2560))
        scales = np.array([.01, 1., .3, .75, 2., .125, .007, 0.], np.float32)
        weight = torch.from_numpy(signs.astype(np.float32))
        for bits in (4, 8, 16):
            with self.subTest(bits=bits):
                expected = torch.nn.functional.linear(hard_activation(state, bits)[0], weight) * torch.from_numpy(scales)
                actual = checker._replay_head(pack(signs), scales, 2560, state, bits)
                self.assertTrue(torch.equal(actual, expected))


if __name__ == "__main__":
    unittest.main()

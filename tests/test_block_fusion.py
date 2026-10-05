import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from w1a1_eagle.block_fusion import (  # noqa: E402
    FusionFitConfig,
    fit_fusion,
    make_fusion_latents,
    project,
    quantize,
    rms_norm_reference,
)
from w1a1_eagle.recurrent_qat import (  # noqa: E402
    RowBinaryLinear,
    W1AxContract,
    hard_activation_with_codes,
)


class FusionTests(unittest.TestCase):
    def test_arithmetic_matches_training_hard_forward(self):
        rng = np.random.default_rng(5)
        raw = rng.normal(size=(9, 32)).astype(np.float32)
        raw[0] = 0
        raw[0, 0] = -0.0
        signs = np.where(rng.normal(size=(4, 32)) < 0, -1, 1).astype(np.float32)
        scales = np.array([0.0, 0.1, 0.25, 3.0], dtype=np.float32)
        for bits in (1, 8):
            dense = torch.nn.Linear(32, 4, bias=False)
            module = RowBinaryLinear(
                dense.weight.detach(), torch.from_numpy(scales), W1AxContract(bits)
            )
            with torch.no_grad():
                module.latent_sign.copy_(torch.from_numpy(signs))
                module.initial_scale.copy_(torch.from_numpy(scales))
                expected = module(torch.from_numpy(raw)).numpy()
                if bits == 8:
                    _, beta_t, _, codes_t = hard_activation_with_codes(torch.from_numpy(raw), bits)
                    expected = (
                        (codes_t.float() @ torch.from_numpy(signs).T)
                        * torch.from_numpy(scales)
                        * beta_t
                    ).numpy()
            codes, beta = quantize(raw, bits)
            np.testing.assert_array_equal(project(codes, beta, signs, scales), expected)

    def test_negative_zero_row_orientation_rescues_at_both_arithmetics(self):
        raw = np.array([[1, 2, 3, -4], [2, 3, 4, -5]], dtype=np.float32)
        weight = np.ones((1, 4), dtype=np.float32)
        for bits in (1, 8):
            codes, beta = quantize(raw, bits)
            teacher = -project(codes, beta, weight, np.array([2], dtype=np.float32))
            control = fit_fusion(raw, teacher, weight, FusionFitConfig(bits))
            rescued = fit_fusion(raw, teacher, weight, FusionFitConfig(bits, True))
            self.assertEqual(control["scale"][0], 0)
            self.assertGreater(rescued["scale"][0], 0)
            self.assertLess(
                rescued["report"]["candidate"]["sse"], control["report"]["candidate"]["sse"]
            )
            np.testing.assert_array_equal(rescued["control_latent"], weight)
            np.testing.assert_array_equal(rescued["latent"], -weight)

    def test_coordinate_updates_are_exported_objective_improvements(self):
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(24, 16)).astype(np.float32)
        weight = rng.normal(size=(3, 16)).astype(np.float32)
        target_sign = np.where(weight < 0, -1, 1).astype(np.float32)
        target_sign[:, 2] *= -1
        codes, beta = quantize(raw, 8)
        teacher = project(codes, beta, target_sign, np.ones(3, dtype=np.float32))
        candidate = fit_fusion(raw, teacher, weight, FusionFitConfig(8, True, 4))
        self.assertTrue(candidate["report"]["events"])
        for event in candidate["report"]["events"]:
            self.assertLess(event["after_sse"], event["before_sse"])
        self.assertLessEqual(
            candidate["report"]["candidate"]["sse"], candidate["report"]["scale_only"]["sse"]
        )

    def test_a1_signed_zero_is_positive(self):
        codes, beta = quantize(np.array([[-0.0, 0.0, -2, 2]], dtype=np.float32), 1)
        np.testing.assert_array_equal(codes, [[1, 1, -1, 1]])
        np.testing.assert_array_equal(beta, [1])

    def test_tiny_a8_finite_reciprocal_fallback_and_rne_ties(self):
        raw = np.array([[1, 3, 254, 0x80000003, 0x80000000]], dtype=np.uint32).view(np.float32)
        codes, beta = quantize(raw, 8)
        np.testing.assert_array_equal(codes, [[0, 2, 127, -2, 0]])
        np.testing.assert_array_equal(beta.view(np.uint32), [2])
        max_subnormal = np.array([[0x007FFFFF, 0x807FFFFF, 0]], dtype=np.uint32).view(np.float32)
        codes, beta = quantize(max_subnormal, 8)
        np.testing.assert_array_equal(codes, [[127, -127, 0]])
        self.assertTrue(np.isfinite(beta).all())

    def test_a8_normal_domain_exact_historical_codes_and_scales(self):
        rng = np.random.default_rng(192)
        raw = rng.normal(size=(32, 32)).astype(np.float32)
        raw *= np.geomspace(1e-30, 1e30, 32).astype(np.float32)[:, None]
        limit = np.max(np.abs(raw), axis=1)
        inv = np.divide(np.float32(127), limit, dtype=np.float32)
        expected = np.clip(np.rint(raw * inv[:, None]), -127, 127).astype(np.int16)
        expected_beta = np.divide(limit, np.float32(127), dtype=np.float32)
        codes, beta = quantize(raw, 8)
        np.testing.assert_array_equal(codes, expected)
        np.testing.assert_array_equal(beta.view(np.uint32), expected_beta.view(np.uint32))

    def test_invalid_inputs_and_config_refuse(self):
        for bits in (True, 4, 16):
            with self.assertRaises(ValueError):
                FusionFitConfig(bits)
        with self.assertRaises(ValueError):
            FusionFitConfig(8, True, 33)
        with self.assertRaises(ValueError):
            quantize(np.array([[np.nan]], dtype=np.float32), 8)

    def test_raw_and_post_norm_have_distinct_diagnostics(self):
        raw = np.array([[3, 4]], dtype=np.float32)
        norm = rms_norm_reference(raw, np.ones(2, dtype=np.float32), 1e-6)
        self.assertFalse(np.array_equal(raw, norm))
        np.testing.assert_allclose(np.mean(norm**2), 1, rtol=1e-6)

    def test_latent_policy_preserves_reference_inertia_and_hard_bits(self):
        reference = np.array([[0.001, -0.2, 1.5, -0.0]], dtype=np.float32)
        signs = np.array([[-1, 1, -1, 1]], dtype=np.int8)
        latent, contract = make_fusion_latents(
            signs,
            reference,
            policy="preserve_reference_magnitudes",
            reference_kind="block_source_weight_magnitudes",
        )
        np.testing.assert_array_equal(np.abs(latent), np.abs(reference))
        np.testing.assert_array_equal(np.where(latent < 0, -1, 1), signs)
        self.assertEqual(contract["source_magnitudes_above_ste_one"], 1)
        self.assertEqual(contract["source_zero_magnitudes"], 1)
        self.assertEqual(len(contract["reference_sha256"]), 64)

    def test_negative_sign_on_exact_zero_refuses_unannounced_magnitude_floor(self):
        with self.assertRaisesRegex(ValueError, "zero"):
            make_fusion_latents(
                np.array([[-1]]),
                np.zeros((1, 1), dtype=np.float32),
                policy="preserve_reference_magnitudes",
                reference_kind="block_source_weight_magnitudes",
            )
        unit, contract = make_fusion_latents(
            np.array([[-1]]),
            np.zeros((1, 1), dtype=np.float32),
            policy="unit_probe",
            reference_kind="block_source_weight_magnitudes",
        )
        self.assertEqual(unit[0, 0], -1)
        self.assertEqual(contract["policy"], "unit_probe")

    def test_eagle_fixed_half_magnitude_is_explicit(self):
        raw = np.array([[1, 2, 3, -4], [2, 3, 4, -5]], dtype=np.float32)
        weight = np.ones((1, 4), dtype=np.float32)
        codes, beta = quantize(raw, 8)
        teacher = -project(codes, beta, weight, np.array([2], dtype=np.float32))
        fitted = fit_fusion(
            raw,
            teacher,
            weight,
            FusionFitConfig(8, True, reference_kind="eagle_fixed_reference_0.5"),
        )
        np.testing.assert_array_equal(fitted["latent"], np.full_like(weight, -0.5))
        np.testing.assert_array_equal(fitted["hard_signs"], -weight)
        np.testing.assert_array_equal(fitted["control_latent"], np.full_like(weight, 0.5))


if __name__ == "__main__":
    unittest.main()

"""CPU numerical and serialization gates for the fixed binary-scale screen."""

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from fit_binary_scales import (  # noqa: E402
    HEADER,
    PROTOCOL,
    SOURCE_NAMES,
    baseline_scales,
    export_variant,
    fit_layer,
    group_features,
    pack_signs,
    projected_ridge_nnls,
    select_prompt_rows,
)
from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter  # noqa: E402


class ScaleFittingTests(unittest.TestCase):
    def test_coupled_nnls_is_not_clipped_unconstrained_solution(self):
        # unconstrained answer [-1, 2]; constrained optimum [0, 1.5]
        # because correlated columns require the positive coefficient to adapt.
        z = torch.tensor([[[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float64)
        y = torch.tensor([[1.0, -1.0, 2.0]], dtype=torch.float64)
        baseline = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
        fit, report = projected_ridge_nnls(z, y, baseline)
        gram = z.transpose(1, 2).bmm(z) / 3
        ridge = PROTOCOL["ridge_relative"] * gram.diagonal(dim1=1, dim2=2).mean(1)
        hessian = gram + torch.diag_embed(ridge[:, None].expand_as(baseline))
        rhs = z.transpose(1, 2).bmm(y[:, :, None]).squeeze(2) / 3 + ridge[:, None] * baseline
        clipped = torch.linalg.solve(hessian, rhs).clamp_min(0)
        self.assertEqual(fit[0, 0], 0)
        self.assertAlmostEqual(fit[0, 1].item(), (rhs[0, 1] / hessian[0, 1, 1]).item(), places=5)
        self.assertLess(
            ((z @ fit[:, :, None]).squeeze(2) - y).square().sum(),
            ((z @ clipped[:, :, None]).squeeze(2) - y).square().sum() - 0.1,
        )
        self.assertTrue(report["converged"].all())
        self.assertFalse(report["fallback"].any())

    def test_degenerate_features_preserve_feasible_anchor(self):
        z = torch.zeros((3, 11, 4), dtype=torch.float32)
        y = torch.ones((3, 11), dtype=torch.float32)
        baseline = torch.tensor([[0, 1, 2, 3]] * 3, dtype=torch.float32)
        fit, report = projected_ridge_nnls(z, y, baseline)
        torch.testing.assert_close(fit, baseline, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(fit).all())
        self.assertTrue(report["converged"].all())

    def test_f32_monotone_comparison_does_not_stall_on_cancellation(self):
        generator = torch.Generator().manual_seed(2)
        z = torch.randn(5, 96, 3, generator=generator)
        z[:, :, 1] += 0.7 * z[:, :, 0]
        z += 2
        y = torch.randn(5, 96, generator=generator) + z[:, :, 1] * 0.5 - z[:, :, 0] * 0.4
        _, report = projected_ridge_nnls(z, y, torch.full((5, 3), 0.2))
        self.assertTrue(report["converged"].all())
        self.assertLessEqual(report["iterations"], PROTOCOL["max_iterations"])

    def test_row_fit_uses_uncentered_postcast_input(self):
        weight = np.array([[1, 3], [-1, -3]], dtype=np.float32)
        x = np.array([[1, 2], [2, 2], [1, 3], [2, 3]], dtype=np.float16).astype(np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            scales, _, details = fit_layer(weight, x, "cpu")
        signed_input = x.sum(1)
        target = x @ weight[0]
        diag = np.mean(signed_input**2)
        rhs = np.mean(signed_input * target)
        ridge = PROTOCOL["ridge_relative"] * diag
        expected = (rhs + ridge * 2) / (diag + ridge)
        self.assertAlmostEqual(float(scales["C"][0]), expected, places=5)
        self.assertGreater(abs(float(scales["C"][0]) - 2), 0.1)
        self.assertTrue(np.all(details["C_sse"] <= details["A_sse"]))
        self.assertTrue(np.all(details["D_sse"] <= details["B_sse"]))
        with self.assertRaisesRegex(ValueError, "actual F16"):
            fit_layer(weight, x + np.float32(0.00001), "cpu")

    def test_group_tail_mean_and_signs_include_positive_zero(self):
        weight = np.array([[0, -0.0, -2, 2, 4]], dtype=np.float32)
        packed = pack_signs(weight)
        self.assertEqual(int(packed[0, 0]), 0b11011)
        np.testing.assert_array_equal(baseline_scales(weight, 2), [[0, 2, 4]])
        x = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.float32)
        sign = torch.from_numpy(np.where(weight >= 0, 1, -1).astype(np.float32))
        np.testing.assert_array_equal(group_features(x, sign, 2), [[[3, 1, 5]]])

    def test_group_fit_recovers_representable_positive_scales(self):
        rng = np.random.default_rng(22)
        x = rng.normal(size=(96, 260)).astype(np.float16).astype(np.float32)
        signs = rng.choice([-1, 1], size=(3, 260)).astype(np.float32)
        # The baseline is already the true optimum; frozen ridge also anchors it.
        weight = signs * np.tile(np.repeat([0.25, 2, 5], [128, 128, 4]), (3, 1)).astype(np.float32)
        with contextlib.redirect_stdout(io.StringIO()):
            scales, _, details = fit_layer(weight, x, "cpu")
        np.testing.assert_allclose(scales["D"], [[0.25, 2, 5]] * 3, rtol=1e-5)
        self.assertTrue(np.all(scales["D"] >= 0))
        self.assertTrue(np.all(details["D_sse"] <= details["B_sse"]))

    def test_even_capture_selection_spans_invocations_and_checks_cast(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            files = []
            for layer, base in enumerate(SOURCE_NAMES):
                for invocation in range(2):
                    seq = 2 * layer + invocation
                    path = directory / f"op-{seq:012}.bin"
                    name = (base + ".weight").encode().ljust(128, b"\0")
                    x = np.arange(40 * invocation, 40 * (invocation + 1), dtype=np.float32)[:, None]
                    path.write_bytes(
                        HEADER.pack(b"W1AXACT1", seq, 1, 2, 40, 16, name) + x.tobytes()
                    )
                    files.append(path.name)
            values, records = select_prompt_rows(
                {"id": "train001", "capture_dir": temp, "files": files}
            )
            for value in values.values():
                self.assertEqual(value.shape, (32, 1))
                self.assertEqual(value[0, 0], 1)
                self.assertEqual(value[-1, 0], 78)
            self.assertEqual(len(records), 18)
            path = directory / files[0]
            data = bytearray(path.read_bytes())
            data[HEADER.size + 4 : HEADER.size + 8] = np.float32(1.0001).tobytes()
            path.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "not actual post-F16"):
                select_prompt_rows({"id": "train001", "capture_dir": temp, "files": files})

    def test_gguf_exports_f32_scales_fixed_signs_and_unchanged_other_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original = directory / "f16.gguf"
            writer = GGUFWriter(original, "eagle3")
            writer.add_string("general.name", "synthetic serialization fixture")
            writer.add_array("eagle3.test_array", [1, 2, 3])
            other = np.arange(5, dtype=np.float32)
            writer.add_tensor("output_norm.weight", other)
            arrays = {}
            for base in SOURCE_NAMES:
                weight = np.arange(260 * 2, dtype=np.float32).reshape(2, 260) - 10
                writer.add_tensor(f"{base}.weight", weight.astype(np.float16))
                arrays[base] = {
                    "K": 260,
                    "packed": pack_signs(weight),
                    "A": baseline_scales(weight, 0),
                    "B": baseline_scales(weight, 128),
                    "C": baseline_scales(weight, 0) * np.float32(1.001),
                    "D": baseline_scales(weight, 128) * np.float32(1.001),
                }
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            for variant in "ABCD":
                artifact = directory / f"{variant}.gguf"
                report = export_variant(original, artifact, arrays, variant)
                self.assertEqual(report["audited_packed_layers"], 9)
                reader = GGUFReader(artifact)
                self.assertEqual(reader.fields["eagle3.w1a1.version"].contents(), 2)
                self.assertEqual(
                    reader.fields["eagle3.w1a1.groups"].contents(),
                    ["fusion", "attention", "ffn", "head"],
                )
                self.assertEqual(
                    reader.fields["eagle3.w1a1.scale_group_size"].contents(),
                    128 if variant in "BD" else 0,
                )
                for tensor in reader.tensors:
                    if tensor.name.endswith(".w1a1_scale"):
                        self.assertEqual(tensor.tensor_type, GGMLQuantizationType.F32)
                        self.assertEqual(tensor.data.shape, (2, 3) if variant in "BD" else (2,))


if __name__ == "__main__":
    unittest.main()

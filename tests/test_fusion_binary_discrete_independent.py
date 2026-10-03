"""Independent CPU checks for direct fusion A8 sign-and-scale fitting."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
FIT_SCRIPT = Path(
    os.environ.get("EAGLE_FUSION_FIT_SCRIPT", REPO / "scripts/fit_fusion_binary_discrete.py")
).resolve()
if not FIT_SCRIPT.is_file():
    raise unittest.SkipTest("feature implementation is still in its isolated worktree")
sys.path.insert(0, str(FIT_SCRIPT.parent))
import create_fusion_binary_discrete_fixture as fixture  # noqa: E402
import fit_fusion_binary_discrete as fitter  # noqa: E402

CONFIG = FIT_SCRIPT.parent.parent / "configs/fusion_binary_discrete_a8.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(manifest: Path, base: Path, output: Path) -> dict:
    return fitter.run(
        SimpleNamespace(
            config=CONFIG,
            manifest=manifest,
            source_weights=None,
            base_gguf=base,
            output_dir=output,
        )
    )


def _change_validation_only(fixture_dir: Path, destination: Path) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((fixture_dir / "synthetic_manifest.json").read_text())
    source_npz = fixture_dir / manifest["operands"]
    with np.load(source_npz, allow_pickle=False) as archive:
        raw, weight, ids = (
            archive[key].copy() for key in ("raw_input", "reference_weight", "raw_join_ids")
        )
    validation = [i for i, row in enumerate(manifest["rows"]) if row["split"] == "validation"]
    raw[validation] = raw[validation] * np.float32(1.75) + np.float32(0.25)
    for i in validation:
        manifest["rows"][i]["raw_input_sha256"] = fitter.array_hash(raw[i].astype("<f4"))
    altered_npz = destination / "altered_validation.npz"
    np.savez(altered_npz, raw_input=raw, reference_weight=weight, raw_join_ids=ids)
    manifest["operands"] = altered_npz.name
    manifest["operands_sha256"] = _sha(altered_npz)
    altered_manifest = destination / "altered_manifest.json"
    altered_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    return altered_manifest, fixture_dir / "synthetic_base.gguf"


class IndependentFusionDiscreteTests(unittest.TestCase):
    def test_a8_rounding_clipping_zero_and_f32_operator_order(self):
        raw = np.array(
            [
                [0.0, -0.0, 0.0, 0.0, 0.0, 0.0],
                [2.5, 3.5, 127.0, -127.0, 63.5, -63.5],
                [127.0, -127.0, 300.0, -300.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        codes, beta = fitter.quantize_a8(raw)
        self.assertEqual(codes.dtype, np.int16)
        np.testing.assert_array_equal(codes[0], [0, 0, 0, 0, 0, 0])
        self.assertEqual(beta[0], np.float32(0))
        np.testing.assert_array_equal(codes[1], [2, 4, 127, -127, 64, -64])
        np.testing.assert_array_equal(codes[2], [54, -54, 127, -127, 0, 0])
        self.assertEqual(beta[1], np.float32(1))
        self.assertEqual(beta[2], np.float32(300 / 127))

        dots = np.array([[16_777_215, -16_777_215]], dtype=np.int32)
        scales = np.array([np.float32(1.0000001), np.float32(0.33333334)])
        row_beta = np.array([np.float32(0.10000001)], dtype=np.float32)
        actual = fitter.native_output(dots, scales, row_beta[:, None])
        expected = np.multiply(
            np.multiply(dots.astype(np.float32), scales, dtype=np.float32),
            row_beta[:, None],
            dtype=np.float32,
        )
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_array_equal(actual, expected)

    def test_exact_stale_flip_conflict_is_rechecked_before_acceptance(self):
        config = fitter.load_config(CONFIG)
        config.update(alternating_passes=1, scans_per_pass=1, max_flips_per_row=2, row_batch=1)
        # Each duplicate-column flip improves the stale baseline, but their
        # joint update worsens it. The fitter must recheck after the first flip.
        codes = np.array([[100, 100, 0], [0, 0, 100], [0, 0, 0]], dtype=np.int16)
        beta = np.ones(3, dtype=np.float32)
        teacher = np.array([[50.0], [400.0], [0.0]], dtype=np.float32)
        weight = np.ones((1, 3), dtype=np.float32)
        initial_signs = np.ones((1, 3), dtype=np.int8)
        dots = fitter.integer_dots(codes, initial_signs)[:, 0]
        scale, solved = fitter.solve_scale(dots, beta, teacher[:, 0], np.float32(1), config)
        baseline = fitter.sse(fitter.native_output(dots, scale, beta), teacher[:, 0])
        first = fitter.sse(fitter.native_output(dots - 2 * codes[:, 0], scale, beta), teacher[:, 0])
        second = fitter.sse(
            fitter.native_output(dots - 2 * codes[:, 1], scale, beta), teacher[:, 0]
        )
        together = fitter.sse(
            fitter.native_output(dots - 2 * codes[:, 0] - 2 * codes[:, 1], scale, beta),
            teacher[:, 0],
        )
        self.assertEqual(solved["neighbor_gain"], 0)
        self.assertLess(first, baseline)
        self.assertLess(second, baseline)
        self.assertGreater(together, baseline)

        signs, scales, detail = fitter.fit(codes, beta, teacher, weight, config)
        self.assertEqual(detail["scans"][0]["proposals"], 2)
        self.assertEqual(detail["scans"][0]["accepted"], 1)
        self.assertEqual(len(detail["flip_events"]), 1)
        event = detail["flip_events"][0]
        self.assertEqual(event["before_sse"], baseline)
        self.assertIn(event["after_sse"], (first, second))
        self.assertIn(signs.tolist(), ([[-1, 1, 1]], [[1, -1, 1]]))
        self.assertTrue(np.isfinite(scales).all() and np.all(scales >= 0))

    def test_zero_weight_signs_are_positive_and_packed_reload_is_exact(self):
        weight = np.array([[0.0, -0.0, -1.0, 1.0]], dtype=np.float32)
        codes = np.array([[127, -127, 2, 3], [-127, 127, 1, -1]], dtype=np.int16)
        beta = np.ones(2, dtype=np.float32)
        teacher = np.zeros((2, 1), dtype=np.float32)
        config = fitter.load_config(CONFIG)
        signs, scales, detail = fitter.fit(codes, beta, teacher, weight, config)
        self.assertEqual(detail["initializer_signs"].tolist(), [[1, 1, -1, 1]])
        np.testing.assert_array_equal(fitter.unpack_signs(fitter.pack_signs(signs), 4), signs)
        self.assertEqual(scales.dtype, np.float32)

    def test_scale_control_is_finite_nonnegative_and_neighbor_converged(self):
        config = fitter.load_config(CONFIG)
        dots = np.array([-100, -50, 25, 75, 127], dtype=np.int32)
        beta = np.array([0.1, 0.2, 0.5, 0.7, 1.0], dtype=np.float32)
        teacher = np.array([-0.2, -0.1, 0.15, 0.3, 0.9], dtype=np.float32)
        scale, detail = fitter.solve_scale(dots, beta, teacher, np.float32(0.01), config)
        self.assertTrue(scale.dtype == np.float32 and np.isfinite(scale) and scale >= 0)
        threshold = max(
            config["improvement_absolute_margin"],
            config["improvement_relative_margin"] * detail["finite_sse"],
        )
        self.assertLessEqual(detail["neighbor_gain"], threshold)
        self.assertLess(detail["continuous_kkt_relative"], 1e-12)
        pred = fitter.native_output(dots, scale, beta)
        self.assertAlmostEqual(fitter.sse(pred, teacher), detail["finite_sse"], delta=1e-12)

    def test_synthetic_run_freezes_before_validation_and_exports_supported_fusion_only(self):
        tmp_path = Path(tempfile.mkdtemp(prefix="fusion-discrete-test-"))
        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
        fixture_dir = tmp_path / "fixture"
        fixture.create_fixture(
            fixture_dir, seed=20261003, prompts=8, rows_per_prompt=3, width=16, outputs=3
        )
        manifest = fixture_dir / "synthetic_manifest.json"
        base = fixture_dir / "synthetic_base.gguf"
        report = _run(manifest, base, tmp_path / "fit-original")
        altered_manifest, altered_base = _change_validation_only(fixture_dir, tmp_path / "altered")
        changed_validation = _run(
            altered_manifest, altered_base, tmp_path / "fit-altered-validation"
        )

        self.assertIs(report["synthetic"], True)
        self.assertEqual(
            report["frozen_candidate_sha256"], changed_validation["frozen_candidate_sha256"]
        )
        self.assertNotEqual(
            report["metrics"]["validation"]["sign_and_scale"],
            changed_validation["metrics"]["validation"]["sign_and_scale"],
        )
        self.assertLessEqual(
            report["metrics"]["train"]["sign_and_scale"]["sse"],
            report["metrics"]["train"]["converged_scale_only"]["sse"],
        )
        self.assertEqual(len(report["train_prompt_hashes"]), 6)
        self.assertEqual(len(report["validation_prompt_hashes"]), 2)
        self.assertTrue(
            set(report["train_prompt_hashes"]).isdisjoint(report["validation_prompt_hashes"])
        )
        self.assertIs(report["export"]["fusion_only"], True)
        self.assertEqual(report["export"]["unchanged_nonfusion_tensors"], 3)
        self.assertEqual(report["export"]["native_validation"], "deferred")
        for row in report["fit"]["control_solver"]:
            threshold = max(
                report["config"]["improvement_absolute_margin"],
                report["config"]["improvement_relative_margin"] * row["finite_sse"],
            )
            self.assertLess(row["continuous_kkt_relative"], 1e-12)
            self.assertLessEqual(row["neighbor_gain"], threshold)

        fitted = tmp_path / "fit-original"
        with np.load(fitted / "fusion_candidate.npz", allow_pickle=False) as candidate:
            signs = candidate["fc.latent"].astype(np.int8)
            scales = candidate["fc.scale"]
            np.testing.assert_array_equal(
                fitter.unpack_signs(candidate["fc.w1a1_packed"], signs.shape[1]), signs
            )
        from gguf import GGUFReader

        base_reader = GGUFReader(base)
        base_tensors = {tensor.name: tensor for tensor in base_reader.tensors}
        exported = GGUFReader(fitted / "fusion_candidate.gguf")
        self.assertEqual(exported.fields["eagle3.w1a1.version"].contents(), 2)
        self.assertEqual(exported.fields["eagle3.w1a1.scale_group_size"].contents(), 0)
        self.assertEqual(exported.fields["eagle3.w1a1.activation_bits"].contents(), 8)
        self.assertEqual(exported.fields["eagle3.w1a1.groups"].contents(), ["fusion"])
        self.assertEqual(exported.fields["eagle3.w1a1.tensors"].contents(), ["fc.weight"])
        observed = {tensor.name: tensor for tensor in exported.tensors}
        self.assertNotIn("fc.weight", observed)
        for name, source in base_tensors.items():
            if name == "fc.weight":
                continue
            self.assertEqual(observed[name].tensor_type, source.tensor_type)
            np.testing.assert_array_equal(observed[name].data, source.data)
        np.testing.assert_array_equal(observed["fc.w1a1_scale"].data, scales)
        np.testing.assert_array_equal(
            fitter.unpack_signs(observed["fc.w1a1_packed"].data, signs.shape[1]), signs
        )
        self.assertEqual(report["script_sha256"], _sha(FIT_SCRIPT))
        self.assertEqual(report["config_sha256"], _sha(CONFIG))
        before = sorted(path.name for path in fitted.iterdir())
        with self.assertRaisesRegex(FileExistsError, "output directory already exists"):
            _run(manifest, base, fitted)
        self.assertEqual(before, sorted(path.name for path in fitted.iterdir()))

    def test_real_mode_fails_closed_even_if_npz_manifest_claims_eligibility(self):
        tmp_path = Path(tempfile.mkdtemp(prefix="fusion-discrete-refusal-test-"))
        self.addCleanup(lambda: shutil.rmtree(tmp_path, ignore_errors=True))
        fixture_dir = tmp_path / "fixture"
        fixture.create_fixture(
            fixture_dir, seed=5, prompts=4, rows_per_prompt=2, width=12, outputs=2
        )
        path = fixture_dir / "synthetic_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["synthetic"] = False
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        config = fitter.load_config(CONFIG)
        with self.assertRaisesRegex(ValueError, "real fitting unavailable"):
            fitter.load_operands(
                path, config, source_weights_path=fixture_dir / "synthetic_source.json"
            )


if __name__ == "__main__":
    unittest.main()

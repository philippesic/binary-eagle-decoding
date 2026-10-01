"""Strict learned scalar metadata and byte-preserving weight export checks."""

import copy
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import test_recurrent_binary_export as export_fixture  # noqa: E402
from export_recurrent_binary import (  # noqa: E402
    LEARNED_ACTIVATION_RULE,
    QUANTIZER_BOUNDARIES,
    QUANTIZER_PREFIX,
    GGUFReader,
    check_activation_quantizers,
    sha256,
)


class LearnedActivationExportTests(export_fixture.RecurrentBinaryExportTests):
    # Reuse the synthetic nine-projection fixture, but avoid rerunning its tests.
    def learned_manifest(self, bits=4):
        for name in self.arrays:
            if name.endswith(".scale") and self.arrays[name].ndim == 2:
                self.arrays[name] = self.arrays[name][:, 0].copy()
        self.save_checkpoint()
        return {
            "schema_version": 3,
            "base_gguf_sha256": sha256(self.base),
            "checkpoint_sha256": sha256(self.checkpoint),
            "scale_layout": "row",
            "activation_bits": bits,
            "activation_rule": LEARNED_ACTIVATION_RULE,
            "weight_rule": "hard_sign_zero_positive_clipped_identity_ste",
            "qk_row_order": "original_checkpoint",
            "export_status": "row_w1ax_requires_native_validation",
            "objective": "hard_ce",
            "projections": {
                base: {"checkpoint_name": name, "shape": list(self.shapes[base])}
                for base, name in __import__("export_recurrent_binary").SOURCE_NAMES.items()
            },
            "activation_quantizers": {
                "version": 1,
                "boundaries": {
                    boundary: {
                        "bits": bits,
                        "threshold_delta": float(np.float32(0.25 if bits == 1 else 0)),
                        "clip_ratio": float(np.float32(1 if bits == 1 else 0.75)),
                    }
                    for boundary in QUANTIZER_BOUNDARIES
                },
            },
        }

    def test_learned_scalar_round_trip(self):
        for bits in (1, 4, 8):
            manifest = self.learned_manifest(bits)
            self.manifest.write_text(json.dumps(manifest))
            report = self.export()
            self.assertEqual(report["activation_quantizers"], manifest["activation_quantizers"])
            reader = GGUFReader(self.output)
            self.assertEqual(reader.fields[QUANTIZER_PREFIX + "version"].contents(), 1)
            self.assertEqual(
                reader.fields[QUANTIZER_PREFIX + "boundaries"].contents(),
                list(QUANTIZER_BOUNDARIES),
            )
            for boundary, params in manifest["activation_quantizers"]["boundaries"].items():
                for key in ("threshold_delta", "clip_ratio"):
                    self.assertEqual(
                        reader.fields[QUANTIZER_PREFIX + boundary + "." + key].contents(),
                        params[key],
                    )
            self.output.unlink()

    def correction_manifest(self, rank=1, bias=False, learned=True):
        from export_recurrent_binary import CORRECTION_ARITHMETIC

        manifest = self.learned_manifest(4)
        manifest["schema_version"] = 4
        if not learned:
            manifest["activation_quantizers"] = None
            manifest["activation_rule"] = (
                "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive"
            )
        descriptor = {
            "version": 1,
            "rank": rank,
            "u_name": "fc.correction_u.weight",
            "v_name": "fc.correction_v.weight",
            "bias_name": "fc.correction_bias" if bias else None,
            "bias_bound": 0.25 if bias else None,
            "arithmetic": CORRECTION_ARITHMETIC,
        }
        self.arrays[descriptor["u_name"]] = np.full(
            (self.shapes["fc"][0], rank), 0.125, dtype=np.float16
        )
        self.arrays[descriptor["v_name"]] = np.full(
            (rank, self.shapes["fc"][1]), -0.25, dtype=np.float16
        )
        if bias:
            self.arrays[descriptor["bias_name"]] = np.full(
                (self.shapes["fc"][0],), 0.125, dtype=np.float32
            )
        self.save_checkpoint()
        manifest["checkpoint_sha256"] = sha256(self.checkpoint)
        manifest["fusion_correction"] = descriptor
        return manifest

    def test_correction_round_trip_fixed_or_learned(self):
        from export_recurrent_binary import CORRECTION_PREFIX, raw_hash

        for rank, bias, learned in ((1, False, False), (4, True, True)):
            manifest = self.correction_manifest(rank, bias, learned)
            self.manifest.write_text(json.dumps(manifest))
            report = self.export()
            reader = GGUFReader(self.output)
            self.assertEqual(report["fusion_correction"], manifest["fusion_correction"])
            self.assertEqual(reader.fields[CORRECTION_PREFIX + "rank"].contents(), rank)
            tensors = {t.name: t for t in reader.tensors}
            for name in (
                manifest["fusion_correction"][k] for k in ("u_name", "v_name", "bias_name")
            ):
                if name:
                    self.assertEqual(raw_hash(tensors[name].data), raw_hash(self.arrays[name]))
            self.output.unlink()

    def test_correction_invalid_arrays_and_descriptor_rejected(self):
        from export_recurrent_binary import check_fusion_correction

        manifest = self.correction_manifest(4, True)
        for key, value in (
            ("rank", 2),
            ("u_name", "untrusted.weight"),
            ("version", 2),
            ("bias_bound", float("nan")),
            ("arithmetic", "other"),
        ):
            case = dict(manifest["fusion_correction"])
            case[key] = value
            with self.assertRaises(ValueError):
                check_fusion_correction(case)
        for name, replacement in (
            ("fc.correction_u.weight", np.zeros((5, 4), dtype=np.float32)),
            ("fc.correction_v.weight", np.zeros((4, 129), dtype=np.float16)),
            ("fc.correction_bias", np.full((5,), 0.5, dtype=np.float32)),
            ("fc.correction_u.weight", np.full((5, 4), np.nan, dtype=np.float16)),
        ):
            original = self.arrays[name]
            self.arrays[name] = replacement
            self.save_checkpoint()
            manifest["checkpoint_sha256"] = sha256(self.checkpoint)
            self.manifest.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                self.export()
            self.arrays[name] = original

    def test_fail_closed_parameters(self):
        valid = self.learned_manifest()["activation_quantizers"]
        for key, value in (
            ("clip_ratio", float("nan")),
            ("clip_ratio", 0),
            ("clip_ratio", 1.1),
            ("clip_ratio", 0.1),
            ("threshold_delta", 0.25),
            ("bits", 8),
            ("clip_ratio", [1]),
        ):
            case = copy.deepcopy(valid)
            case["boundaries"]["qkv"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check_activation_quantizers(case, 4)
        for mutation in (
            lambda x: x["boundaries"].pop("head"),
            lambda x: x["boundaries"].update(extra=x["boundaries"]["head"]),
            lambda x: x["boundaries"]["qkv"].update(extra=1),
            lambda x: x.update(version=2),
        ):
            case = copy.deepcopy(valid)
            mutation(case)
            with self.assertRaises(ValueError):
                check_activation_quantizers(case, 4)
        with self.assertRaises(ValueError):
            check_activation_quantizers(valid, 16)


# Inherited fixture methods are useful; inherited test cases are covered separately.
for name in list(export_fixture.RecurrentBinaryExportTests.__dict__):
    if name.startswith("test_") and name not in LearnedActivationExportTests.__dict__:
        setattr(LearnedActivationExportTests, name, None)

if __name__ == "__main__":
    unittest.main()

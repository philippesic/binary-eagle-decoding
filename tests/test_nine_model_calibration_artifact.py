"""Independent adapter checks against the saved EAGLE A8/A1 fit artifacts.

Set NINE_MODEL_EAGLE_FUSION_DIR to the ignored, source-audited calibration
bundle. Test output reports only pass/fail, never tensor contents.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from train_nine_model_qat import calibration, validate_initializer_reference
from w1a1_eagle.qat_initialization import apply_binary_initialization

from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

FUSION_DIR = Path(os.environ.get("NINE_MODEL_EAGLE_FUSION_DIR", "/missing"))
SOURCE_DIR = FUSION_DIR.parent / "eagle-fusion-20261004"
EXPECTED = {
    8: {
        "candidate": "7a40b375deb2f337a4037aa6d034477d933b66542341fdc15385723f907e0a8f",
        "control": "9b907187f3299e76d0f9435851b4d683dc987c4ee22abb0a696a7f70f095e771",
        "source_candidate": "bfbbcc2e5595fe7426a0e683d6215a9349acd2de1dae94d491c4a8f54a5fc67a",
        "source_control": "2642a292a77a5b687896292a06b741d7fb06306fd9d22260a587bf038f2aec44",
    },
    1: {
        "candidate": "fa2780b1222929b2275138d7368598640c03fe6048efa89d685dff1b8aeae851",
        "control": "f4f4c102f4d8749c716903e3862723ae9968d505d65eae2e4958c14d067af32b",
        "source_candidate": "b6d616ef6871ad5273016e5cdc7a100760c4c93deb35916a979b462c0ab6d62b",
        "source_control": "72de7c9369e3503bf8e1217fa76129d88e0a21dcf5fc896950b07f89c758fd4a",
    },
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _locator(saved: dict) -> dict:
    details = saved["latent_initialization"]
    return {
        "path": saved["path"],
        "sha256": saved["sha256"],
        "activation_bits": saved["activation_bits"],
        "latent_initialization": {
            key: details[key] for key in ("policy", "reference_kind", "reference_sha256")
        },
    }


@unittest.skipUnless(FUSION_DIR.is_dir(), "saved source-audited EAGLE fit bundle unavailable")
class SavedFusionAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads((FUSION_DIR / "initialization-contract.json").read_text())
        if cls.contract.get("schema") != "eagle_fusion_fixed_reference_latents_v1":
            raise ValueError("saved initializer contract schema differs")
        if cls.contract.get("optimizer_updates") != 0:
            raise ValueError("initializer preparation must not contain optimizer updates")

    def test_actual_a8_a1_artifacts_bind_bits_signs_scales_and_reference_magnitudes(self):
        for bits in (8, 1):
            with self.subTest(bits=bits):
                saved = self.contract["artifacts"][f"fusion-a{bits}"]
                source_path = SOURCE_DIR / f"fusion-a{bits}.npz"
                self.assertEqual(_sha(Path(saved["path"])), EXPECTED[bits]["candidate"])
                self.assertEqual(_sha(source_path), EXPECTED[bits]["source_candidate"])
                locator = _locator(saved)
                initialization = calibration(locator, SimpleNamespace(activation_bits=bits))
                fitted_latent, fitted_scale = initialization["fc"]

                with (
                    np.load(source_path, allow_pickle=False) as original,
                    np.load(saved["path"], allow_pickle=False) as calibrated,
                ):
                    original_signs = np.where(original["fc.latent"] < 0, -1.0, 1.0)
                    actual_signs = np.where(calibrated["fc.latent"] < 0, -1.0, 1.0)
                    self.assertTrue(np.array_equal(actual_signs, original_signs))
                    self.assertTrue(np.array_equal(original["fc.scale"], calibrated["fc.scale"]))
                    self.assertTrue(np.all(np.abs(calibrated["fc.latent"]) == np.float32(0.5)))
                    self.assertTrue(np.array_equal(fitted_latent.numpy(), calibrated["fc.latent"]))
                    self.assertTrue(np.array_equal(fitted_scale.numpy(), calibrated["fc.scale"]))

                latent = fitted_latent
                weight = torch.full_like(latent, 0.5)
                target = RowBinaryLinear(weight, torch.ones(len(weight)), W1AxContract(bits))
                linears = {"fc": target}
                for layer in range(5):
                    for projection in ("ffn_gate", "ffn_up", "ffn_down"):
                        name = f"blk.{layer}.{projection}"
                        linears[name] = RowBinaryLinear(
                            torch.tensor([[0.25, -0.4]], dtype=torch.float32),
                            torch.tensor([0.1]),
                            W1AxContract(bits),
                        )
                protected = {
                    name: {key: value.clone() for key, value in module.state_dict().items()}
                    for name, module in linears.items()
                    if name != "fc"
                }
                report = apply_binary_initialization(
                    linears,
                    initialization,
                    policy=saved["latent_initialization"]["policy"],
                )
                validate_initializer_reference(report, locator, "eagle")
                self.assertTrue(torch.equal(target.latent_sign, fitted_latent))
                self.assertTrue(torch.equal(target.initial_scale, fitted_scale))
                for name, snapshot in protected.items():
                    for key, value in snapshot.items():
                        self.assertTrue(torch.equal(value, linears[name].state_dict()[key]))

    def test_historical_unit_fit_requires_explicit_unit_probe(self):
        metadata = self.contract["artifacts"]["fusion-a8"]["latent_initialization"]
        self.assertEqual(metadata["policy"], "preserve_reference_magnitudes")
        historical = SOURCE_DIR / "fusion-a8.npz"
        with np.load(historical, allow_pickle=False) as arrays:
            self.assertTrue(np.all(np.abs(arrays["fc.latent"]) == 1.0))
            latent = arrays["fc.latent"].copy()
            scale = arrays["fc.scale"].copy()
        # The old unit artifact does not match the new reference-magnitude pin;
        # a selected unit_probe must have its own generated artifact/identity.
        self.assertNotEqual(_sha(historical), self.contract["artifacts"]["fusion-a8"]["sha256"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "unit-probe.npz"
            np.savez(path, **{"fc.latent": latent, "fc.scale": scale})
            locator = _locator(self.contract["artifacts"]["fusion-a8"])
            locator.update(path=str(path), sha256=_sha(path))
            locator["latent_initialization"]["policy"] = "unit_probe"
            initialization = calibration(locator, SimpleNamespace(activation_bits=8))
            weight = torch.full_like(initialization["fc"][0], 0.5)
            target = RowBinaryLinear(weight, torch.ones(len(weight)), W1AxContract(8))
            report = apply_binary_initialization(
                {"fc": target}, initialization, policy="unit_probe"
            )
            validate_initializer_reference(report, locator, "eagle")
            self.assertTrue(torch.all(torch.abs(target.latent_sign) == 1))


if __name__ == "__main__":
    unittest.main()

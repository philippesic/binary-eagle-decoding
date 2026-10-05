"""Lightweight CPU guard tests; released-model launches require separate approval."""

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from check_block_qat_cpu_model import input_contract, install_initialization
from test_nine_model_training import block_fixture
from train_nine_model_qat import calibration, smoke_block

from w1a1_eagle.block_qat import BlockDrafter, BlockQATConfig, block_optimizer
from w1a1_eagle.continuous_qat import sha256

MAIN = Path("/Users/pippo/github/binary-eagle-decoding")
CAPTURE = MAIN / "results/nine-model-qat-preparation/development-cpu-pilot-capture-20261004-03"
INDEX = (
    MAIN
    / "results/nine-model-qat-preparation/development-cpu-pilot-fusion-20261004/artifact-index.json"
)
INDEX_SHA = "e59edcd2c235c4fbca800892dde1bc86199db57729177e93371b006a8c490800"
DATA_PINS = {
    "dspark": (
        "77891f9b7027a38c6707ef0722e88e1376df826b382c652a269260676019bbd7",
        "24199f53d50cf95af032c23d3daa532d7f88c551a084974e34902cc80407b1b2",
    ),
    "dflash": (
        "54f1c2bef6b83c7cc2104c56055fa7fbc2a08cc9801b0cb5315f8d7bbbe06040",
        "b7954c392b7f7a47197d51d027007f8e0846c22b8a6d84f42ecbd0828cdb45ee",
    ),
}


def args(family="dspark", **changes):
    manifest_sha, admission_sha = DATA_PINS[family]
    return SimpleNamespace(
        **dict(
            input_mode="native_train",
            data_manifest=CAPTURE / family / "manifest.json",
            data_sha256=manifest_sha,
            data_admission=CAPTURE / family / "completed-admission.json",
            data_admission_sha256=admission_sha,
            initialization_index=INDEX,
            initialization_index_sha256=INDEX_SHA,
        )
        | changes
    )


class BlockCPUCompositionTests(unittest.TestCase):
    def test_synthetic_cannot_claim_real_pins_or_full_soft_targets(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            input_contract(args(input_mode="synthetic"), BlockQATConfig("dspark", 8))
        plain = SimpleNamespace(input_mode="synthetic")
        self.assertIsNone(input_contract(plain, BlockQATConfig("dspark", 8))[0])
        with self.assertRaisesRegex(ValueError, "synthetic"):
            input_contract(plain, BlockQATConfig("dspark", 8, objective="full_probability_l1"))

    def test_missing_pin_refuses_before_model_or_data_load(self):
        with patch("check_block_qat_cpu_model.BlockDataset") as dataset:
            with self.assertRaisesRegex(ValueError, "requires all"):
                input_contract(args(data_admission_sha256=None), BlockQATConfig("dspark", 8))
            dataset.assert_not_called()

    @unittest.skipUnless(INDEX.exists(), "actual pinned CPU artifacts not present")
    def test_wrong_index_pin_refuses_before_loading_npz(self):
        with patch("check_block_qat_cpu_model.calibration") as fit:
            with self.assertRaisesRegex(ValueError, "external SHA"):
                input_contract(
                    args(initialization_index_sha256="0" * 64), BlockQATConfig("dspark", 8)
                )
            fit.assert_not_called()

    @unittest.skipUnless(INDEX.exists(), "actual pinned CPU artifacts not present")
    def test_actual_train_completed_admission_and_all_four_fits_join(self):
        # This reads one small block and sparse FC NPZ per cell, never model weights.
        for family in ("dspark", "dflash"):
            for bits in (8, 1):
                with self.subTest(family=family, bits=bits):
                    config = BlockQATConfig(family, bits)
                    batch, prepared, detail = input_contract(args(family), config)
                    self.assertEqual(batch.context_features.shape[1:], (5, 2560))
                    self.assertEqual(len(batch.prefix_tokens), len(batch.context_features) + 1)
                    self.assertEqual(batch.teacher_logits.shape, (7, 151936))
                    self.assertEqual(int(batch.loss_mask.sum()), 7)
                    self.assertEqual(detail["data_cursor"]["split"], "train")
                    self.assertEqual(detail["initialization"]["activation_bits"], bits)
                    self.assertEqual(set(prepared[0]), {"fc"})
                    self.assertEqual(prepared[0]["fc"][0].shape, (2560, 12800))
                    self.assertEqual(prepared[0]["fc"][1].shape, (2560,))
                    self.assertEqual(detail["production_readiness"], "PENDING")
                    del batch, prepared

    @unittest.skipUnless(INDEX.exists(), "actual pinned CPU artifacts not present")
    def test_wrong_completed_admission_pin_refuses(self):
        with patch("check_block_qat_cpu_model.calibration", return_value={}):
            with self.assertRaisesRegex(ValueError, "admission differs"):
                input_contract(args(data_admission_sha256="0" * 64), BlockQATConfig("dspark", 8))

    def test_sparse_fit_artifact_actual_source_application_and_zero_update_smoke(self):
        for family in ("dspark", "dflash"):
            for bits in (8, 1):
                cfg, tensors, batch = block_fixture(family, bits)
                model = BlockDrafter(tensors, cfg)
                linears = model.binary_linears()
                reference = linears["fc"].latent_sign.detach().clone()
                untouched = {
                    name: {key: value.clone() for key, value in module.state_dict().items()}
                    for name, module in linears.items()
                    if name != "fc"
                }
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "fit.npz"
                    np.savez(
                        path,
                        **{
                            "fc.latent": (-reference).numpy(),
                            "fc.scale": (model.fc.initial_scale.detach() * 0.75).numpy(),
                        },
                    )
                    locator = {
                        "path": str(path),
                        "sha256": sha256(path),
                        "activation_bits": bits,
                        "latent_initialization": {
                            "policy": "preserve_reference_magnitudes",
                            "reference_kind": "block_source_weight_magnitudes",
                            "reference_sha256": hashlib.sha256(
                                reference.abs().numpy().astype("<f4").tobytes()
                            ).hexdigest(),
                        },
                    }
                    installed = install_initialization(model, (calibration(locator, cfg), locator))
                self.assertTrue(installed["application"]["artifact_values_exact"])
                self.assertTrue(torch.equal(model.fc.latent_sign, -reference))
                for name, state in untouched.items():
                    for key, value in state.items():
                        self.assertTrue(torch.equal(linears[name].state_dict()[key], value))
                versions = [p._version for p in model.parameters()]
                optimizer = block_optimizer(model)
                with patch.object(
                    torch.optim.AdamW, "step", side_effect=AssertionError("forbidden")
                ):
                    smoke = smoke_block(model, batch, optimizer)
                self.assertEqual(smoke["optimizer_updates"], 0)
                self.assertEqual(smoke["nonzero_gradient_tensors"], 32)
                self.assertFalse(optimizer.state)
                self.assertEqual(versions, [p._version for p in model.parameters()])
                self.assertTrue(torch.equal(model.output, tensors["output.weight"]))
                self.assertNotEqual(model.output.data_ptr(), tensors["output.weight"].data_ptr())


if __name__ == "__main__":
    unittest.main()

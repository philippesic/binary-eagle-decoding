"""Block-local RNG inventory guards; mocked CUDA checks are CPU evidence only."""

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_block_checkpoint_contract import SOURCE
from test_nine_model_training import block_fixture

import w1a1_eagle.block_training as training
from w1a1_eagle.block_qat import BlockDrafter, block_optimizer
from w1a1_eagle.continuous_qat import rng_state, sha256


def cuda_reference():
    return {
        **rng_state("cpu"),
        "cuda": torch.arange(16, dtype=torch.uint8),
        "cuda_all": [torch.arange(16, dtype=torch.uint8)],
        "cuda_device_count": 1,
        "cuda_device_index": 0,
    }


class BlockRNGContractTests(unittest.TestCase):
    def test_cuda_capture_records_all_device_generators_and_selected_identity(self):
        cuda = torch.arange(16, dtype=torch.uint8)
        with (
            patch.object(training, "rng_state", return_value={**rng_state("cpu"), "cuda": cuda}),
            patch("torch.cuda.device_count", return_value=2),
            patch("torch.cuda.get_rng_state_all", return_value=[cuda, cuda + 1]),
        ):
            actual = training._block_rng_state("cuda:0")
        self.assertEqual(actual["cuda_device_count"], 2)
        self.assertEqual(actual["cuda_device_index"], 0)
        self.assertTrue(torch.equal(actual["cuda"], actual["cuda_all"][0]))
        self.assertNotEqual(actual["cuda_all"][0].data_ptr(), cuda.data_ptr())

    def test_restore_restores_all_cuda_device_states(self):
        state = cuda_reference()
        with (
            patch.object(training, "restore_rng") as common,
            patch("torch.cuda.set_rng_state_all") as all_devices,
        ):
            training._restore_block_rng(state, "cuda:0")
        common.assert_called_once_with(state, "cuda:0")
        all_devices.assert_called_once_with(state["cuda_all"])

    def test_missing_or_malformed_cuda_state_fails_before_model_or_rng_mutation(self):
        reference = cuda_reference()
        corruptions = []
        for field in ("cuda", "cuda_all", "cuda_device_count", "cuda_device_index"):
            value = copy.deepcopy(reference)
            del value[field]
            corruptions.append(value)
        for field, wrong in (
            ("cuda_device_count", 2),
            ("cuda_device_index", 1),
            ("cuda_all", []),
            ("cuda_all", [torch.zeros(15, dtype=torch.uint8)]),
            ("cuda", torch.zeros(16, dtype=torch.int64)),
            ("cuda", torch.zeros(15, dtype=torch.uint8)),
            ("cuda", torch.zeros(16, dtype=torch.uint8)),
            ("torch", torch.zeros(20, dtype=torch.uint8)),
        ):
            value = copy.deepcopy(reference)
            value[field] = wrong
            corruptions.append(value)
        cfg, tensors, _ = block_fixture()
        original = BlockDrafter(tensors, cfg)
        optimizer = block_optimizer(original)
        with tempfile.TemporaryDirectory() as folder:
            receipt = training.save_block_checkpoint(
                original, optimizer, training.BlockCursor(), SOURCE, folder
            )
            valid = torch.load(receipt["path"], weights_only=False)
            for bad in corruptions:
                with self.subTest(keys=list(bad)):
                    payload = copy.deepcopy(valid)
                    payload["rng"] = bad
                    torch.save(payload, receipt["path"])
                    receipt["sha256"] = sha256(Path(receipt["path"]))
                    fresh = BlockDrafter(tensors, cfg)
                    other = block_optimizer(fresh)
                    # Only device identity is mocked. Arithmetic and checkpoints
                    # are actual CPU tensors; no CUDA runtime claim follows.
                    fresh._buffers["token_embd"] = SimpleNamespace(device=torch.device("cuda:0"))
                    before = [p.clone() for p in fresh.parameters()]
                    with (
                        patch.object(training, "_runtime", return_value=valid["runtime"]),
                        patch.object(training, "_block_rng_state", return_value=reference),
                        patch.object(training, "_restore_block_rng") as restore,
                    ):
                        with self.assertRaisesRegex(ValueError, "RNG"):
                            training.load_block_checkpoint(fresh, other, SOURCE, receipt)
                        restore.assert_not_called()
                    self.assertTrue(
                        all(torch.equal(a, b) for a, b in zip(before, fresh.parameters()))
                    )
                    self.assertFalse(other.state)

    def test_cpu_requires_exact_inventory_and_python_numpy_tensor_schemas(self):
        reference = training._block_rng_state("cpu")
        training._validate_block_rng(reference, "cpu", reference)
        corruptions = [
            {**reference, "cuda": torch.zeros(16, dtype=torch.uint8)},
            {**reference, "torch": reference["torch"].float()},
            {**reference, "python": None},
            {**reference, "numpy": ("MT19937", "invalid", 0, 0, 0.0)},
        ]
        for state in corruptions:
            with self.assertRaisesRegex(ValueError, "RNG"):
                training._validate_block_rng(state, "cpu", reference)


if __name__ == "__main__":
    unittest.main()

"""CPU storage calibration and metadata-only CUDA checkpoint admission tests.

Configured shapes use meta tensors with a CUDA device descriptor; no backing
allocation, accelerator discovery, actual transfer, model, or data is used.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import w1a1_eagle.qat_curriculum_runner as runner_module
from w1a1_eagle.qat_curriculum_runner import (
    CurriculumRunner,
    _checkpoint_cpu_copy_bytes,
    _cpu_tree,
)

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = 16 * 1024**2


class CUDATensorMetadata(torch.Tensor):
    @staticmethod
    def __new__(cls, shape, dtype=torch.float32):
        return torch.Tensor._make_subclass(cls, torch.empty(shape, dtype=dtype, device="meta"))

    @property
    def device(self):
        return torch.device("cuda:0")


def tensor_occurrences(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_occurrences(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from tensor_occurrences(child)


def save_stub(folder, model):
    return SimpleNamespace(
        resources=lambda: None,
        _account_occupancy=lambda: None,
        run_dir=folder,
        state=SimpleNamespace(global_updates=1, state_dict=lambda: {"updates": 1}),
        contract={"fixture": True},
        model_phase=0,
        _model_payload=lambda: model,
        optimizer=SimpleNamespace(state_dict=lambda: {}),
        base_lrs=[0.001],
        rng=torch.zeros(8, dtype=torch.uint8),
        cursor=0,
        epoch=0,
        occupancy_seconds=0.0,
        smoke_seconds=0.0,
        transition_seconds=0.0,
        unique_rows=set(),
        unique_prompts=set(),
        device="cuda:0",
        config=SimpleNamespace(min_host_available_bytes=2 * 1024**3),
        last_checkpoint=None,
    )


class CheckpointHostAdmissionTests(unittest.TestCase):
    def test_nested_alias_occurrences_and_mixed_dtypes_are_charged(self):
        cpu = torch.zeros(8, dtype=torch.float64)
        cuda = CUDATensorMetadata((12,), torch.float16)
        payload = {"cpu": cpu, "nested": [cuda, (cuda, CUDATensorMetadata((5,)))], "n": 3}
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")):
            self.assertEqual(_checkpoint_cpu_copy_bytes(payload), (132, 24))

    def test_empty_and_non_tensor_fields_have_no_copy_cost(self):
        for value in (None, 7, "source", {}, [], (), torch.empty(0)):
            self.assertEqual(_checkpoint_cpu_copy_bytes(value), (0, 0))

    def test_actual_cpu_clone_storage_counts_aliases_separately(self):
        tensor = torch.arange(4, dtype=torch.float32, requires_grad=True)
        raw = {"first": tensor, "nested": (tensor, [tensor.view(4)])}
        copied = _cpu_tree(raw)
        tensors = list(tensor_occurrences(copied))
        self.assertEqual(_checkpoint_cpu_copy_bytes(raw), (48, 0))
        self.assertEqual(len({t.untyped_storage().data_ptr() for t in tensors}), 3)
        self.assertEqual(sum(t.untyped_storage().nbytes() for t in tensors), 48)
        for clone in tensors:
            torch.testing.assert_close(clone, tensor)
            self.assertFalse(clone.requires_grad)
            self.assertNotEqual(clone.data_ptr(), tensor.data_ptr())

    def test_configured_adam_order_peak_is_covered_without_backing_storage(self):
        shapes = json.loads((ROOT / "configs/continuous_w1ax.json").read_text())["model_shapes"]
        model, parameters = {}, []
        for index, (rows, columns) in enumerate(shapes):
            sign = CUDATensorMetadata((rows, columns))
            offset = CUDATensorMetadata((rows,))
            model[index] = {
                "latent_sign": sign,
                "scale_offset": offset,
                "initial_scale": CUDATensorMetadata((rows,)),
            }
            parameters.append(sign)
        parameters.extend(v["scale_offset"] for v in model.values())
        optimizer = {
            index: {
                "step": torch.zeros(()),
                "exp_avg": CUDATensorMetadata(p.shape),
                "exp_avg_sq": CUDATensorMetadata(p.shape),
            }
            for index, p in enumerate(parameters)
        }
        payload = {"model": model, "optimizer": optimizer}
        retained, transfer = _checkpoint_cpu_copy_bytes(payload)
        live, peak = 0, 0
        for tensor in tensor_occurrences(payload):
            size = tensor.numel() * tensor.element_size()
            transient = size if tensor.device.type == "cuda" else 0
            peak = max(peak, live + size + transient)
            live += size
        self.assertEqual(retained, 2_619_863_112)
        self.assertEqual(transfer, 327_680_000)
        self.assertEqual(peak, 2_947_020_836)
        self.assertEqual(peak - (retained + WORKSPACE), 310_380_508)
        self.assertGreaterEqual(retained + transfer + WORKSPACE, peak)

    def test_save_rejects_old_boundary_before_copy_or_publication(self):
        tensor = CUDATensorMetadata((64,))
        model = {"first": tensor, "same_tensor_again": tensor}
        retained, transfer = _checkpoint_cpu_copy_bytes(model)
        retained += 8  # The real CPU RNG bytes in save_stub.
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            runner = save_stub(folder, model)
            floor = runner.config.min_host_available_bytes
            with (
                patch.object(
                    runner_module,
                    "linux_host_memory",
                    return_value={"host_available_bytes": floor + retained + WORKSPACE},
                ),
                patch.object(
                    runner_module, "_cpu_tree", side_effect=AssertionError("copy")
                ) as copy,
                patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")),
            ):
                required = floor + retained + transfer + WORKSPACE
                with self.assertRaisesRegex(RuntimeError, f"requires {required} bytes"):
                    CurriculumRunner.save(runner)
                copy.assert_not_called()
            self.assertIsNone(runner.last_checkpoint)
            self.assertFalse((folder / "latest.json").exists())
            self.assertEqual(list((folder / "checkpoints").iterdir()), [])

    def test_exact_new_boundary_preserves_floor_and_workspace(self):
        tensor = CUDATensorMetadata((64,))
        model = {"first": tensor, "same_tensor_again": tensor}
        retained, transfer = _checkpoint_cpu_copy_bytes(model)
        retained += 8
        with tempfile.TemporaryDirectory() as directory:
            runner = save_stub(Path(directory), model)
            floor = runner.config.min_host_available_bytes
            additional = retained + transfer + WORKSPACE
            with (
                patch.object(
                    runner_module,
                    "linux_host_memory",
                    return_value={"host_available_bytes": floor + additional},
                ),
                patch.object(
                    runner_module, "require_host_memory", wraps=runner_module.require_host_memory
                ) as admission,
                patch.object(
                    runner_module, "_cpu_tree", side_effect=AssertionError("admitted copy")
                ),
            ):
                with self.assertRaisesRegex(AssertionError, "admitted copy"):
                    CurriculumRunner.save(runner)
                self.assertEqual(admission.call_args.kwargs["floor_bytes"], floor)
                self.assertEqual(admission.call_args.kwargs["additional_bytes"], additional)
            self.assertIsNone(runner.last_checkpoint)


if __name__ == "__main__":
    unittest.main()

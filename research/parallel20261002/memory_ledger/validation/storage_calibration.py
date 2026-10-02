"""Tiny CPU-only storage alias calibration for the QAT allocation ledger."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from w1a1_eagle.continuous_resources import (
    checkpoint_host_buffer_bytes,
    lane_storage_bytes,
    model_storage_bytes,
)
from w1a1_eagle.qat_curriculum_runner import _cpu_tree


class SharedBank(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bank = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        self.left = nn.Linear(4, 3, bias=False)
        self.right = nn.Linear(4, 3, bias=False)
        self.left.weight = bank
        self.right.weight = bank
        self.register_buffer("bank_view", bank[:2])
        self.register_buffer("offset", torch.zeros((), dtype=torch.float32))


def storage_key(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage()._cdata


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from tensor_leaves(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from tensor_leaves(child)


def unique_storage_bytes(tensors) -> int:
    storages = {}
    for tensor in tensors:
        storage = tensor.untyped_storage()
        storages[storage_key(tensor)] = storage.nbytes()
    return sum(storages.values())


def main() -> None:
    model = SharedBank()
    optimizer = torch.optim.AdamW([model.left.weight], lr=0.01, foreach=False)
    model.left.weight.grad = torch.ones_like(model.left.weight)
    optimizer.step()  # materialize step + first and second moments on CPU
    optimizer.zero_grad(set_to_none=True)

    lane = type("LaneFixture", (), {"drafter": model, "optimizer": optimizer})()
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    expected_model_storage = 3 * 4 * 4 + 4  # shared 3x4 bank + scalar offset
    measured_model_storage = model_storage_bytes(model, device_type="cpu")
    assert measured_model_storage == expected_model_storage

    adam_state = optimizer.state[model.left.weight]
    step, first, second = (
        adam_state["step"],
        adam_state["exp_avg"],
        adam_state["exp_avg_sq"],
    )
    expected_lane_storage = expected_model_storage + sum(
        t.untyped_storage().nbytes() for t in (step, first, second)
    )
    measured_lane_storage = lane_storage_bytes([lane], device_type="cpu")
    assert measured_lane_storage == expected_lane_storage

    model_state = model.state_dict()
    assert storage_key(model_state["left.weight"]) == storage_key(model_state["right.weight"])
    assert storage_key(model_state["left.weight"]) == storage_key(model_state["bank_view"])
    assert storage_key(model_state["left.weight"]) == storage_key(model.left.weight)

    optimizer_state = optimizer.state_dict()
    serialized_first = optimizer_state["state"][0]["exp_avg"]
    assert storage_key(serialized_first) == storage_key(first)

    cloned_state = _cpu_tree(state)
    state_tensors = list(tensor_leaves(state))
    cloned_tensors = list(tensor_leaves(cloned_state))
    occurrence_copy_bytes = sum(t.numel() * t.element_size() for t in state_tensors)
    unique_source_bytes = unique_storage_bytes(state_tensors)
    unique_cloned_bytes = unique_storage_bytes(cloned_tensors)
    assert unique_cloned_bytes == occurrence_copy_bytes
    assert unique_cloned_bytes > unique_source_bytes  # aliases are copied per payload leaf
    assert all(storage_key(a) != storage_key(b) for a, b in zip(state_tensors, cloned_tensors))

    export_lanes = [
        type(
            "ExportLaneFixture",
            (),
            {
                "linears": {
                    "large": type(
                        "LinearFixture",
                        (),
                        {
                            "latent_sign": torch.empty((2, 3)),
                            "scale_offset": torch.empty((2,)),
                        },
                    )(),
                }
            },
        )(),
        type(
            "ExportLaneFixture",
            (),
            {
                "linears": {
                    "small": type(
                        "LinearFixture",
                        (),
                        {
                            "latent_sign": torch.empty((1, 3)),
                            "scale_offset": torch.empty((3,)),
                        },
                    )(),
                }
            },
        )(),
    ]
    expected_export_working_set = 2 * 32 + 24 + 16 * 1024**2
    measured_export_working_set = checkpoint_host_buffer_bytes(export_lanes)
    assert measured_export_working_set == expected_export_working_set

    output = Path("runs/parallel20261002/memory-ledger-validation/storage-state.pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    loaded = torch.load(output, map_location="cpu", weights_only=False)
    loaded_model = loaded["model"]
    assert storage_key(loaded_model["left.weight"]) == storage_key(
        loaded_model["right.weight"]
    )
    assert storage_key(loaded_model["left.weight"]) == storage_key(loaded_model["bank_view"])

    print(
        json.dumps(
            {
                "device": "cpu",
                "model_expected_unique_storage_bytes": expected_model_storage,
                "model_measured_unique_storage_bytes": measured_model_storage,
                "lane_expected_unique_storage_bytes": expected_lane_storage,
                "lane_measured_unique_storage_bytes": measured_lane_storage,
                "state_tensor_occurrences": len(state_tensors),
                "state_unique_storage_bytes": unique_source_bytes,
                "recursive_cpu_clone_unique_storage_bytes": unique_cloned_bytes,
                "recursive_cpu_clone_occurrence_copy_bytes": occurrence_copy_bytes,
                "optimizer_step_bytes": step.untyped_storage().nbytes(),
                "optimizer_exp_avg_bytes": first.untyped_storage().nbytes(),
                "optimizer_exp_avg_sq_bytes": second.untyped_storage().nbytes(),
                "torch_save_roundtrip_preserved_shared_parameter_storage": True,
                "checkpoint_export_buffer_expected_bytes": expected_export_working_set,
                "checkpoint_export_buffer_measured_bytes": measured_export_working_set,
                "run_file": str(output),
                "run_file_bytes": output.stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Allocation arithmetic only: never construct configured-shape tensors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GIB = 1024**3
MIB = 1024**2


def ordered_cpu_copy_peak(entries: list[tuple[int, bool]]) -> dict[str, int]:
    """Retained clones plus one overlapping .cpu() transfer for GPU inputs.

    Existing raw CPU storage is resident before the admission snapshot, so only
    its new clone is additional. GPU inputs require both transfer and clone.
    This is logical live storage, excluding allocator/serialization workspace.
    """
    retained = peak = 0
    for size, transfer in entries:
        peak = max(peak, retained + size * (2 if transfer else 1))
        retained += size
    return {"final_clone_bytes": retained, "additional_peak_bytes": peak}


def configured_ledger(spec: dict) -> dict:
    shapes = spec["model_shapes"]
    weights = sum(out * inp for out, inp in shapes)
    rows = sum(out for out, _ in shapes)
    parameters = weights + rows
    largest = max(out * inp for out, inp in shapes) * 4
    master = 4 * parameters
    initial_scales = 4 * rows
    moments = 8 * parameters
    frozen = spec["frozen_drafter_bytes"]
    persistent = 2 * (master + initial_scales + moments) + frozen
    # Current _core_states insertion order: latent, scale_offset, initial_scale.
    core = [(size, True) for out, inp in shapes for size in (4 * out * inp, 4 * out, 4 * out)]
    # joint_parameter_families orders all signs before all scale offsets; smoke
    # and AdamW initialize state in that order. Ordinary step scalars are CPU.
    states = [
        (size, transfer)
        for counts in ([o * i for o, i in shapes], [o for o, _ in shapes])
        for count in counts
        for size, transfer in ((4, False), (4 * count, True), (4 * count, True))
    ]
    copy = ordered_cpu_copy_peak(core + states)
    current_save_allowance = copy["final_clone_bytes"] + 16 * MIB
    return {
        "weights_per_lane": weights,
        "rows_per_lane": rows,
        "trainable_parameters_per_lane": parameters,
        "head_weights": shapes[-1][0] * shapes[-1][1],
        "largest_f32_tensor_bytes": largest,
        "master_bytes_per_lane": master,
        "initial_scale_bytes_per_lane": initial_scales,
        "adam_moment_bytes_per_lane": moments,
        "paired_persistent_bytes_with_configured_frozen": persistent,
        "one_gradient_bytes": master,
        "one_round_hard_sign_bytes": 4 * weights,
        "before_sign_boolean_bytes": weights,
        "before_scale_snapshot_bytes": 4 * rows,
        "post_backward_known_bytes": persistent + master + weights + 4 * rows,
        "last_head_hard_sign_construction_bytes": 4 * (weights - largest // 4)
        + 3 * largest
        + largest // 4,
        "existing_estimated_peak_bytes": persistent + master + spec["assumed_graph_budget_bytes"],
        "configured_cuda_cap_bytes": spec["training"]["max_cuda_reserved_bytes"],
        "conservative_nonconcurrent_step_envelope_excluding_graph_workspace": persistent
        + master
        + 5 * weights
        + 4 * rows,
        "row_npz_retained_bytes": master,
        "continuous_checkpoint_host_allowance_bytes": 2 * master + largest + 16 * MIB,
        "curriculum_save_core_and_adam": {
            **copy,
            "current_allowance_bytes": current_save_allowance,
            "logical_excess_over_current_allowance_bytes": copy["additional_peak_bytes"]
            - current_save_allowance,
            "proposed_allowance_bytes": copy["final_clone_bytes"] + largest + 16 * MIB,
            "excludes": (
                "RNG, optional recipes, frozen biases and workspace; add their actual entries"
            ),
        },
        "optional_parameters_per_lane": {
            "learned_activation_unique_scalars": 6,
            "affine_fc_rows": shapes[0][0],
            "affine_all_rows": rows,
            "fusion_rank1_factors": sum(shapes[0]),
            "fusion_rank4_factors": 4 * sum(shapes[0]),
            "fusion_output_bias": shapes[0][0],
        },
        "sign_diagnostics_host_two_lanes_bytes": 8 * sum((o * i + 7) // 8 for o, i in shapes),
        "packed_export_one_lane_bytes": sum(o * ((i + 31) // 32) * 4 for o, i in shapes) + 4 * rows,
    }


def main():
    source_names = (
        "configs/continuous_w1ax.json",
        "src/w1a1_eagle/continuous_qat.py",
        "src/w1a1_eagle/continuous_resources.py",
        "src/w1a1_eagle/qat_curriculum_runner.py",
        "src/w1a1_eagle/recurrent_qat.py",
        "src/w1a1_eagle/qat_state.py",
        "src/w1a1_eagle/recurrent_binary.py",
        "src/w1a1_eagle/qat_optimization.py",
        "src/w1a1_eagle/learned_activation.py",
        "src/w1a1_eagle/fusion_correction.py",
        "src/w1a1_eagle/affine_binary.py",
        "scripts/export_recurrent_binary.py",
    )
    spec = json.loads((ROOT / source_names[0]).read_text())
    print(
        json.dumps(
            {
                "schema": "source_bound_allocation_ledger_v1",
                "hardware": "none; shape arithmetic only",
                "source_sha256": {
                    n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in source_names
                },
                "ledger": configured_ledger(spec),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

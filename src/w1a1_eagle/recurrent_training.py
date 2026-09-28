"""CPU reference training step and checkpoint boundary for the nine binary linears.

The caller must supply logits from a faithful recurrent student rollout. This
module never constructs a shortcut head-only forward or captures target labels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .recurrent_binary import CANDIDATE_D_BASE_TO_PATH, GroupedBinaryLinear
from .recurrent_loss import supported_prefix_ce
from .recurrent_trace import TraceAudit

CHECKPOINT_NAMES = {
    "fc": "fc.weight",
    "output": "lm_head.weight",
    "blk.0.attn_q": "midlayer.self_attn.q_proj.weight",
    "blk.0.attn_k": "midlayer.self_attn.k_proj.weight",
    "blk.0.attn_v": "midlayer.self_attn.v_proj.weight",
    "blk.0.attn_output": "midlayer.self_attn.o_proj.weight",
    "blk.0.ffn_gate": "midlayer.mlp.gate_proj.weight",
    "blk.0.ffn_up": "midlayer.mlp.up_proj.weight",
    "blk.0.ffn_down": "midlayer.mlp.down_proj.weight",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_linears(linears: Mapping[str, GroupedBinaryLinear]) -> None:
    if set(linears) != set(CANDIDATE_D_BASE_TO_PATH.values()):
        raise ValueError("training requires exactly the nine candidate-D drafter linears")
    if any(not isinstance(module, GroupedBinaryLinear) for module in linears.values()):
        raise TypeError("all nine linears must use the hard-binary W1A16 forward")
    if any(module.group_size != 128 for module in linears.values()):
        raise ValueError("the joint training contract requires group size 128")
    if len({module.arithmetic for module in linears.values()}) != 1:
        raise ValueError("all nine linears must declare the same training arithmetic")


def train_step(
    linears: Mapping[str, GroupedBinaryLinear],
    logits: Tensor,
    audit: TraceAudit,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Backpropagate one audited unroll and project scales after the update.

    ``logits`` must come from the current student graph and stay attached to
    earlier student states and K/V. This function checks optimizer ownership,
    but actual recurrence and frozen target ownership need model integration.
    """
    _validate_linears(linears)
    expected = {
        id(parameter)
        for module in linears.values()
        for parameter in (module.latent_sign, module.scale_offset)
    }
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(owned) != len(expected) or {id(parameter) for parameter in owned} != expected:
        raise ValueError("optimizer must own only the nine binary sign and scale parameters")
    if logits.device.type != "cpu":
        raise ValueError("reference training step is CPU-only")
    optimizer.zero_grad(set_to_none=True)
    loss = supported_prefix_ce(logits, audit)
    if not loss.requires_grad or not torch.isfinite(loss):
        raise ValueError("recurrent loss must be finite and differentiable")
    loss.backward()
    for parameter in owned:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError("nonfinite binary parameter gradient")
    optimizer.step()
    for module in linears.values():
        module.project_scales_()
        if not torch.isfinite(module.latent_sign).all():
            raise ValueError("nonfinite binary sign parameter after optimizer step")
    return float(loss.detach())


def save_training_checkpoint(
    linears: Mapping[str, GroupedBinaryLinear],
    base_gguf_sha256: str,
    checkpoint_path: Path,
    manifest_path: Path,
) -> dict:
    """Save F32 latent/scale NPZ and strict manifest for the learned exporter.

    The deployable GGUF contains only packed signs/scales; this NPZ is an
    intermediate training artifact and must stay outside Git.
    """
    _validate_linears(linears)
    if len(base_gguf_sha256) != 64 or any(c not in "0123456789abcdef" for c in base_gguf_sha256):
        raise ValueError("base GGUF SHA256 must be lowercase hex")
    checkpoint_path, manifest_path = Path(checkpoint_path), Path(manifest_path)
    if checkpoint_path.exists() or manifest_path.exists():
        raise FileExistsError("checkpoint and manifest must be new paths")
    arrays = {}
    projections = {}
    for base, path in CANDIDATE_D_BASE_TO_PATH.items():
        module = linears[path]
        latent, scales = module.training_arrays()
        name = CHECKPOINT_NAMES[base]
        arrays[name + ".latent"] = latent
        arrays[name + ".scale"] = scales
        projections[base] = {"checkpoint_name": name, "shape": list(latent.shape)}
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(checkpoint_path, **arrays)
    manifest = {
        "schema_version": 1,
        "base_gguf_sha256": base_gguf_sha256,
        "training_arithmetic": next(iter(linears.values())).arithmetic,
        "projections": projections,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "projections": len(projections),
    }

"""Atomic source-bound block QAT checkpoints and native export preparation.

Model weights/teachers stay outside Git. Exact resume is within one arithmetic,
optimizer backend, source/data and runtime contract. A8→A1 is a declared stage
transition with fresh moments, never an exact resume claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .block_qat import BlockQATConfig, block_contract, block_optimizer
from .continuous_qat import atomic_json, restore_rng, rng_state, sha256


@dataclass(frozen=True)
class BlockCursor:
    step: int = 0
    epoch: int = 0
    block_index: int = 0
    supervised_tokens: int = 0
    presented_tokens: int = 0
    elapsed_seconds: float = 0.0
    stage: str = "direct"
    unique_blocks: tuple[str, ...] = ()
    data_cursor: dict | None = None
    stage_updates: int | None = None
    stage_supervised_tokens: int | None = None

    def __post_init__(self):
        for name in ("step", "epoch", "block_index", "supervised_tokens", "presented_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError("checkpoint cursor must have nonnegative integer counters")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("checkpoint elapsed time invalid")
        if self.stage not in {"direct", "a8_warm_start", "a1_final"}:
            raise ValueError("checkpoint precision stage invalid")
        if self.stage_updates is None:
            object.__setattr__(self, "stage_updates", self.step)
        if self.stage_supervised_tokens is None:
            object.__setattr__(self, "stage_supervised_tokens", self.supervised_tokens)
        for name in ("stage_updates", "stage_supervised_tokens"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= getattr(
                self, "step" if name == "stage_updates" else "supervised_tokens"
            ):
                raise ValueError("precision stage exposure counters invalid")
        if self.data_cursor is not None and not isinstance(self.data_cursor, dict):
            raise ValueError("data cursor must be an exact provider payload")
        object.__setattr__(self, "unique_blocks", tuple(self.unique_blocks))
        if len(set(self.unique_blocks)) != len(self.unique_blocks):
            raise ValueError("unique block coverage contains duplicates")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_source(source):
    for field in ("base_gguf_sha256", "data_manifest_sha256", "bundle_sha256"):
        value = source.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("source-bound checkpoint needs valid " + field)
    if source.get("synthetic") is not False and source.get("synthetic") is not True:
        raise ValueError("synthetic provenance must be explicit")


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


def _validate_optimizer(model, optimizer, saved, *, expected_updates, serialized=False):
    """Exact stage-local Adam publication/recovery contract, before mutation."""
    if type(expected_updates) is not int or expected_updates < 0:
        raise ValueError("optimizer stage update count must be a nonnegative integer")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("block checkpoints require the declared AdamW optimizer")
    declared_fused = getattr(model.config, "optimizer_backend", "serial") == "fused_fp32_probe"
    if any(bool(group.get("fused")) != declared_fused for group in optimizer.param_groups):
        raise ValueError("optimizer execution backend differs from declared profile")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    model_parameters = [p for p in model.parameters() if p.requires_grad]
    if len(parameters) != len(model_parameters) or {id(p) for p in parameters} != {
        id(p) for p in model_parameters
    }:
        raise ValueError("optimizer must own exactly current student parameters")
    if any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("optimizer checkpoint masters/moments must be F32")
    expected = optimizer.state_dict()
    if not isinstance(saved, dict) or set(saved) != set(expected):
        raise ValueError("optimizer checkpoint inventory differs")
    groups = saved["param_groups"]
    if not isinstance(groups, list) or len(groups) != len(expected["param_groups"]):
        raise ValueError("optimizer checkpoint family count differs")
    for group, reference in zip(groups, expected["param_groups"]):
        if set(group) != set(reference) or any(group[k] != reference[k] for k in reference):
            raise ValueError("optimizer checkpoint family/options/rates differ")
    ids = [i for group in groups for i in group["params"]]
    states = saved["state"]
    if not isinstance(states, dict) or len(set(ids)) != len(ids):
        raise ValueError("optimizer state/owned IDs invalid")
    if expected_updates == 0:
        if states:
            raise ValueError("zero-update/reset checkpoint requires empty optimizer state")
        return
    if set(states) != set(ids):
        raise ValueError("progressed optimizer checkpoint requires all owned moments")
    for group, live_group in zip(groups, optimizer.param_groups):
        for key, parameter in zip(group["params"], live_group["params"]):
            values = states[key]
            if not isinstance(values, dict) or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("optimizer moments missing/extra")
            step = values["step"]
            step_device = (
                torch.device("cpu") if serialized or not group.get("fused") else parameter.device
            )
            if (
                not isinstance(step, torch.Tensor)
                or step.ndim != 0
                or step.dtype != torch.float32
                or step.device != step_device
                or not bool(torch.isfinite(step))
                or float(step) != expected_updates
            ):
                raise ValueError(
                    "optimizer step must be scalar F32 on declared device at exact stage count"
                )
            for name in ("exp_avg", "exp_avg_sq"):
                value = values[name]
                moment_device = torch.device("cpu") if serialized else parameter.device
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype != torch.float32
                    or value.shape != parameter.shape
                    or value.device != moment_device
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError("optimizer moment shape/dtype/device/finite differs")
                if name == "exp_avg_sq" and bool((value < 0).any()):
                    raise ValueError("optimizer second moment negative")


def _runtime(model):
    device = model.token_embd.device
    result = {
        "torch_version": torch.__version__,
        "device_type": device.type,
        "source": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "block_qat.py",
                "block_training.py",
                "recurrent_qat.py",
                "recurrent_binary.py",
            )
        },
        "optimizer": "serial_fp32_adamw_foreach_false",
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(
            cuda_version=torch.version.cuda,
            device_name=props.name,
            compute_capability=[props.major, props.minor],
        )
    return result


def save_block_checkpoint(model, optimizer, cursor, source, directory):
    validate_source(source)
    _validate_optimizer(
        model, optimizer, optimizer.state_dict(), expected_updates=cursor.stage_updates
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        directory / f"step-{cursor.step:012d}-e{cursor.epoch:06d}-b{cursor.block_index:012d}"
        f"-{cursor.stage}-t{int(cursor.elapsed_seconds * 1e6):016d}.pt"
    )
    if checkpoint.exists():
        raise ValueError("checkpoint exists; exact recovery required")
    payload = {
        "schema": "block_qat_checkpoint_v1",
        "contract": block_contract(model.config),
        "source": source,
        "runtime": _runtime(model),
        "cursor": json.loads(json.dumps(asdict(cursor))),
        "linears": {k: _cpu_tree(m.state_dict()) for k, m in model.binary_linears().items()},
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "rng": rng_state(str(model.token_embd.device)),
    }
    temp = checkpoint.with_suffix(".tmp")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, checkpoint)
    receipt = {
        "schema": payload["schema"],
        "path": str(checkpoint.resolve()),
        "sha256": sha256(checkpoint),
        "contract_sha256": _digest(payload["contract"]),
        "source_sha256": _digest(source),
        "cursor": json.loads(json.dumps(asdict(cursor))),
        "committed": True,
    }
    atomic_json(directory / "latest.json", receipt)
    return receipt


def load_block_checkpoint(model, optimizer, source, receipt):
    validate_source(source)
    path = Path(receipt["path"])
    if path.is_symlink() or sha256(path) != receipt["sha256"]:
        raise ValueError("checkpoint hash/path differs")
    # Hash-bound local checkpoint. This is not a general untrusted pickle importer.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (
        saved.get("schema") != "block_qat_checkpoint_v1"
        or saved.get("contract") != block_contract(model.config)
        or saved.get("source") != source
        or saved.get("runtime") != _runtime(model)
        or receipt.get("contract_sha256") != _digest(saved["contract"])
        or receipt.get("source_sha256") != _digest(source)
        or receipt.get("cursor") != saved.get("cursor")
        or receipt.get("committed") is not True
    ):
        raise ValueError("exact resume source/config/runtime/cursor contract differs")
    cursor = BlockCursor(**saved["cursor"])
    current = model.binary_linears()
    if set(saved["linears"]) != set(current):
        raise ValueError("binary projection checkpoint inventory differs")
    for name, module in current.items():
        reference = module.state_dict()
        state = saved["linears"][name]
        if set(state) != set(reference):
            raise ValueError("binary checkpoint parameter inventory differs")
        for key, value in state.items():
            tensor = reference[key]
            if isinstance(tensor, torch.Tensor):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.dtype != tensor.dtype
                    or value.shape != tensor.shape
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError("binary checkpoint tensor shape/dtype/finite differs")
                if key in {"initial_scale", "frozen_bias"} and not torch.equal(value, tensor.cpu()):
                    raise ValueError("frozen checkpoint scale/bias differs")
            elif value != tensor:
                raise ValueError("binary checkpoint parameter contract differs")
    _validate_optimizer(
        model, optimizer, saved["optimizer"], expected_updates=cursor.stage_updates, serialized=True
    )
    # Validate RNG in isolation and restore the caller's RNG before any mutation.
    original = rng_state(str(model.token_embd.device))
    try:
        restore_rng(saved["rng"], str(model.token_embd.device))
    finally:
        restore_rng(original, str(model.token_embd.device))
    for name, module in current.items():
        module.load_state_dict(saved["linears"][name], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    restore_rng(saved["rng"], str(model.token_embd.device))
    return cursor


def transition_a8_to_a1(model, *, source_checkpoint_sha256: str, stage="a1_final", in_place=False):
    if (
        model.config.activation_bits != 8
        or len(source_checkpoint_sha256) != 64
        or any(c not in "0123456789abcdef" for c in source_checkpoint_sha256)
    ):
        raise ValueError("transition requires committed A8 source checkpoint")
    result = model if in_place else copy.deepcopy(model)
    result.zero_grad(set_to_none=True)
    result.config = replace(model.config, activation_bits=1)
    for layer in result.layers:
        layer.config = result.config
    for module in result.binary_linears().values():
        module.contract = replace(module.contract, activation_bits=1)
    optimizer = block_optimizer(result)
    return (
        result,
        optimizer,
        {
            "schema": "block_precision_transition_v1",
            "from_bits": 8,
            "to_bits": 1,
            "optimizer": "reset",
            "stage": stage,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "latent_magnitude_and_scale_representation": "preserved",
            "exact_resume": False,
        },
    )


def export_block_checkpoint(model, source, output_dir):
    """Write exactly the NPZ/manifest API consumed by export_block_binary.py."""
    validate_source(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "binary.npz"
    if path.exists() or (output_dir / "binary.json").exists():
        raise ValueError("export destination exists; preserve prior artifacts")
    arrays, projections = {}, {}
    for name, module in model.binary_linears().items():
        latent = module.latent_sign.detach().cpu().numpy()
        scale = module.effective_scales().detach().cpu().numpy()
        if not np.isfinite(latent).all() or not np.isfinite(scale).all() or (scale < 0).any():
            raise ValueError("nonfinite/negative native export parameter")
        arrays[name + ".latent"] = latent.astype(np.float32)
        arrays[name + ".scale"] = scale.astype(np.float32)
        projections[name] = {"checkpoint_name": name, "shape": list(latent.shape)}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    manifest = {
        "schema_version": 1,
        "family": model.config.family,
        "profile": model.config.profile,
        "base_gguf_sha256": source["base_gguf_sha256"],
        "checkpoint_sha256": sha256(path),
        "activation_bits": model.config.activation_bits,
        "projections": projections,
    }
    atomic_json(output_dir / "binary.json", manifest)
    return {
        "npz": str(path.resolve()),
        "manifest": str((output_dir / "binary.json").resolve()),
        "sha256": manifest["checkpoint_sha256"],
        "native_graph_admitted": False,
    }


def load_block_gguf(path, expected_sha256, config: BlockQATConfig):
    """Load source BF16/F16/F32 GGUF values without borrowed head/embedding.

    Refuse reduced vocabulary, optional architecture changes and quantized
    floating exceptions. The baseline hash is authenticated once by this loader.
    """
    import sys

    gguf_py = Path(__file__).resolve().parents[2] / "third_party/llama.cpp/gguf-py"
    if str(gguf_py) not in sys.path:
        sys.path.insert(0, str(gguf_py))
    from gguf import GGMLQuantizationType, GGUFReader
    from gguf.quants import dequantize

    path = Path(path)
    if sha256(path) != expected_sha256:
        raise ValueError("block source GGUF hash differs")
    reader = GGUFReader(path)

    def field(name):
        item = reader.fields.get(name)
        return None if item is None else item.contents()

    if (
        field("general.architecture") != "dflash"
        or field("dflash.block_size") != 7
        or field("dflash.sample_from_anchor") is not True
        or list(field("dflash.target_layers") or []) != [2, 10, 18, 26, 34]
    ):
        raise ValueError("source GGUF released author anchor/tap geometry differs")
    forbidden = (
        "d2t",
        "selector_",
        "conv_",
        "shexp",
        "exps",
        "hc_",
        "attn_sinks",
        "attn_post_norm",
        "ffn_post_norm",
        "layer_out_scale",
        "rope_freqs",
    )
    for key in (
        "dflash.attention.sliding_window",
        "dflash.attention.sliding_window_pattern",
        "dflash.hyper_connection.count",
        "dflash.attention.rotation",
    ):
        if field(key) is not None:
            raise ValueError("unsupported architecture/cache metadata: " + key)
    allowed = {
        "token_embd.weight",
        "output.weight",
        "fc.weight",
        "enc.output_norm.weight",
        "output_norm.weight",
        "conf_proj.weight",
        "conf_proj.bias",
    }
    if config.family == "dspark":
        allowed.update({"markov_w1.weight", "markov_w2.weight", "markov_w2.scale"})
    for i in range(5):
        allowed.update(
            f"blk.{i}.{name}.weight"
            for name in (
                "attn_norm",
                "attn_q_norm",
                "attn_k_norm",
                "ffn_norm",
                "attn_q",
                "attn_k",
                "attn_v",
                "attn_output",
                "ffn_gate",
                "ffn_up",
                "ffn_down",
            )
        )
    tensors = {}
    floating = {GGMLQuantizationType.BF16, GGMLQuantizationType.F16, GGMLQuantizationType.F32}
    for tensor in reader.tensors:
        if tensor.name not in allowed or any(part in tensor.name for part in forbidden):
            raise ValueError("unsupported reduced vocabulary/DSV4/selector/sliding architecture")
        if tensor.tensor_type not in floating:
            raise ValueError("source must retain original floating tensors: " + tensor.name)
        values = dequantize(tensor.data, tensor.tensor_type)
        tensors[tensor.name] = torch.from_numpy(np.asarray(values, dtype=np.float32).copy())
    # No shape/metadata inference may silently change the frozen profile.
    for key, value in {
        "dflash.embedding_length": config.hidden_size,
        "dflash.block_count": 5,
        "dflash.feed_forward_length": config.intermediate_size,
        "dflash.attention.head_count": config.num_heads,
        "dflash.attention.head_count_kv": config.num_kv_heads,
        "dflash.attention.key_length": config.head_dim,
        "dflash.attention.value_length": config.head_dim,
        "dflash.rope.freq_base": config.rope_theta,
        "dflash.attention.layer_norm_rms_epsilon": config.norm_eps,
        "tokenizer.ggml.mask_token_id": config.mask_token_id,
    }.items():
        # GGUF arithmetic metadata is stored as F32; canonicalize the
        # declared Python scalar to that deployed representation before compare.
        expected = float(np.float32(value)) if isinstance(value, float) else value
        if field(key) != expected:
            raise ValueError("source GGUF model geometry differs: " + key)
    # Native llama-model.cpp defaults n_rot_full to actual key head width;
    # this optional key is absent in both authenticated released BF16 exports.
    rotary = field("dflash.rope.dimension_count")
    if rotary is None:
        rotary = field("dflash.attention.key_length")
    if rotary != config.head_dim:
        raise ValueError("source GGUF effective rotary dimension differs")
    return tensors

"""Synthetic joint-checkpoint -> GGUF composed forward audit; CPU only."""

from __future__ import annotations

# Imports follow the read-only main submodule lookup in this isolated worktree.
# ruff: noqa: E402
import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[4]
MAIN = Path("/Users/pippo/github/binary-eagle-decoding")
sys.path[:0] = [str(ROOT / "scripts"), str(MAIN / "third_party/llama.cpp/gguf-py")]
from export_recurrent_binary import SOURCE_NAMES, GGUFReader, GGUFWriter, export_model

from w1a1_eagle.affine_binary import AffineBinaryConfig, install_affine_binary
from w1a1_eagle.fusion_correction import FusionCorrectionConfig, install_fusion_correction
from w1a1_eagle.learned_activation import LearnedActivationBank
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    RowBinaryLinear,
    W1AxContract,
    save_joint_checkpoint,
)

ATOL, RTOL = 3e-5, 3e-6
BOUNDARIES = {
    "fc": "fc",
    "output": "head",
    "blk.0.attn_q": "qkv",
    "blk.0.attn_k": "qkv",
    "blk.0.attn_v": "qkv",
    "blk.0.attn_output": "attn_output",
    "blk.0.ffn_gate": "gate_up",
    "blk.0.ffn_up": "gate_up",
    "blk.0.ffn_down": "down",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_control():
    control = json.loads((MAIN / "runs/parallel20261002/control.json").read_text())
    if control["research_stop"]:
        raise RuntimeError("research_stop latch: no new research chunk")


def original_rows(array, base):
    """Independent inverse: native alternating real/imag -> checkpoint halves."""
    heads = 32 if base == "blk.0.attn_q" else 8 if base == "blk.0.attn_k" else None
    if heads is None:
        return array
    rows = array.shape[0]
    order = []
    head_dim = rows // heads
    for h in range(heads):
        for half in range(2):
            for position in range(head_dim // 2):
                order.append(h * head_dim + position * 2 + half)
    return array[np.asarray(order)]


def quantize(x, bits, delta, clip):
    """NumPy hard quantization: explicitly F32 arithmetic, F64 A1 mean."""
    x = np.asarray(x, dtype=np.float32)
    if bits == 1:
        beta = np.abs(x).astype(np.float64).mean(axis=-1, keepdims=True).astype(np.float32)
        shifted = x if delta == 0 else x - np.float32(delta) * beta
        raw = shifted.view(np.uint32)
        neg = ((raw & 0x80000000) != 0) & ((raw & 0x7FFFFFFF) != 0)
        return np.where(neg, -1, 1).astype(np.int32), beta
    qmax = (1 << (bits - 1)) - 1
    limit = np.max(np.abs(x), axis=-1, keepdims=True) * np.float32(clip)
    beta = limit / np.float32(qmax)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inverse = np.where(limit > 0, np.float32(qmax) / limit, np.float32(0))
        normalized = x * inverse
        fallback = (
            x.astype(np.float64) / np.where(limit > 0, limit, 1).astype(np.float64) * qmax
        ).astype(np.float32)
        normalized = np.where(np.isfinite(inverse), normalized, fallback)
    return np.clip(np.rint(normalized), -qmax, qmax).astype(np.int32), beta


def evaluate(path, base, x):
    """Use only GGUF contents and raw inputs: no checkpoint/model access."""
    reader = GGUFReader(path)
    fields = reader.fields
    tensors = {t.name: t for t in reader.tensors}

    def get(key):
        return fields[key].contents()

    bits = int(get("eagle3.w1a1.activation_bits"))
    width = int(get("eagle3.w1a1.tensor." + (base + ".weight").replace(".", "_") + ".logical_k"))
    packed = tensors[base + ".w1a1_packed"].data
    word_bits = (packed.astype(np.uint32)[..., None] >> np.arange(32, dtype=np.uint32)) & 1
    expanded = word_bits.reshape(packed.shape[0], -1)
    if np.any(expanded[:, width:]):
        raise ValueError("nonzero tail bits")
    signs = original_rows((expanded[:, :width].astype(np.int32) * 2 - 1), base)
    alpha = original_rows(tensors[base + ".w1a1_scale"].data, base)
    mu = original_rows(tensors[base + ".w1ax_midpoint"].data, base)
    boundary = BOUNDARIES[base]
    prefix = "eagle3.w1a1.activation_quantizer." + boundary + "."
    codes, beta = quantize(x, bits, get(prefix + "threshold_delta"), get(prefix + "clip_ratio"))
    dot = (codes.astype(np.int64) @ signs.astype(np.int64).T).astype(np.float32)
    total = codes.astype(np.int64).sum(axis=-1, keepdims=True).astype(np.float32)
    # Keep each epilogue operation distinct; combined alpha*beta is forbidden.
    binary = (dot * alpha) * beta
    affine = (total * mu) * beta
    result = binary + affine
    if base == "fc":
        prefix = "eagle3.fusion_correction."
        v = tensors[get(prefix + "v_name")].data.astype(np.float32)
        u = tensors[get(prefix + "u_name")].data.astype(np.float32)
        intermediate = np.asarray(x @ v.T, dtype=np.float32)
        correction = np.asarray(intermediate @ u.T, dtype=np.float32)
        result = result + correction
        bias = get(prefix + "bias_name")
        if bias:
            result = result + tensors[bias].data
    return np.asarray(result, dtype=np.float32)


def raw_inputs(width):
    j = np.arange(width, dtype=np.float32)
    regular = ((j * 7) % 23 - 11) * np.float32(0.17)
    positive = np.ones(width, dtype=np.float32) * np.float32(0.3125)
    ties = np.resize(np.asarray([-1.0, -0.5, -0.0, 0.0, 0.5, 1.0], dtype=np.float32), width)
    zeros = np.zeros(width, dtype=np.float32)
    zeros[::2] = np.float32(-0.0)
    tiny = np.nextafter(np.float32(0), np.float32(1))
    subnormal = np.resize(np.asarray([-tiny, tiny, -0.0, 0.0], dtype=np.float32), width)
    return np.stack([regular, positive, ties, zeros, subnormal])


def create_case(directory, bits, width, rank):
    check_control()
    directory.mkdir(parents=True, exist_ok=True)
    base = directory / "base.gguf"
    writer = GGUFWriter(base, "eagle3")
    linears = {}
    config = JointQATConfig(
        W1AxContract(bits),
        activation_quantization="learned",
        fusion_correction=FusionCorrectionConfig(
            enabled=True, rank=rank, output_bias=True, bias_bound=0.25
        ),
        affine_weights=AffineBinaryConfig(enabled=True, coverage="all"),
    )
    for index, base_name in enumerate(SOURCE_NAMES):
        rows = 128 if base_name == "blk.0.attn_q" else 32 if base_name == "blk.0.attn_k" else 5
        i, j = np.arange(rows)[:, None], np.arange(width)[None, :]
        latent = (((i * 7 + j * 3 + index) % 11) - 5).astype(np.float32) / 8
        latent[0, 0] = np.float32(-0.0)
        scale = np.arange(rows, dtype=np.float32) * np.float32(0.0021) + np.float32(0.0937)
        scale[0] = 0
        writer.add_tensor(base_name + ".weight", np.zeros((rows, width), dtype=np.float16))
        linears[CANDIDATE_D_BASE_TO_PATH[base_name]] = RowBinaryLinear(
            torch.from_numpy(latent), torch.from_numpy(scale), config.contract
        )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    bank = LearnedActivationBank(bits, {p: m.in_features for p, m in linears.items()})
    bank.attach(linears)
    target = nn.Linear(1, 1)
    install_affine_binary(linears, target=target, config=config.affine_weights)
    correction = install_fusion_correction(
        linears["fc"], target=target, config=config.fusion_correction
    )
    with torch.no_grad():
        for n, q in enumerate(bank.quantizers.values()):
            q.parameter.fill_((n - 2) * 0.1875 if bits == 1 else 0.53 + n * 0.061)
        for n, module in enumerate(linears.values()):
            module.affine_binary.midpoint.copy_(
                torch.arange(module.out_features) * 0.0037 - 0.0091 + n * 0.0013
            )
        correction.u.copy_(
            torch.arange(correction.u.numel()).reshape_as(correction.u) * 0.0113 - 0.0251
        )
        correction.v.copy_(
            torch.arange(correction.v.numel()).reshape_as(correction.v).remainder(13) * 0.0077
            - 0.0313
        )
        correction.output_bias.copy_(torch.tensor([-0.13, -0.031, 0.071, 0.113, 0.19]))
    checkpoint, manifest, output = (
        directory / "joint.npz",
        directory / "joint.json",
        directory / "model.gguf",
    )
    save_joint_checkpoint(linears, config, digest(base), checkpoint, manifest)
    export_model(base, checkpoint, manifest, output)
    arrays, comparisons = {}, []
    for base_name, path in CANDIDATE_D_BASE_TO_PATH.items():
        x = raw_inputs(width)
        actual = linears[path](torch.from_numpy(x)).detach().numpy()
        decoded = evaluate(output, base_name, x)
        np.testing.assert_allclose(decoded, actual, atol=ATOL, rtol=RTOL)
        arrays["x::" + base_name], arrays["y::" + base_name] = x, actual
        comparisons.append(
            {
                "base": base_name,
                "max_abs_error": float(np.max(np.abs(decoded - actual))),
                "shape": list(actual.shape),
            }
        )
    # Diagnostic controls: masters must round to F16; correction consumes raw x.
    raw = arrays["x::fc"]
    master_delta = (raw @ correction.v.detach().numpy().T) @ correction.u.detach().numpy().T
    hard_delta = (
        raw @ correction.v.detach().numpy().astype(np.float16).astype(np.float32).T
    ) @ correction.u.detach().numpy().astype(np.float16).astype(np.float32).T
    factor_rounding_delta = float(np.max(np.abs(master_delta - hard_delta)))
    if factor_rounding_delta <= 1e-7:
        raise AssertionError("fixture does not distinguish F16 factors from F32 masters")
    np.savez(directory / "expected.npz", **arrays)
    mutated = directory / "midpoint-mutated.gguf"
    shutil.copyfile(output, mutated)
    reader = GGUFReader(mutated)
    tensor = next(t for t in reader.tensors if t.name == "fc.w1ax_midpoint")
    old = float(tensor.data[0])
    offset = tensor.data_offset
    del tensor, reader
    with mutated.open("r+b") as stream:
        stream.seek(offset)
        stream.write(struct.pack("<f", old + 0.125))
    original = evaluate(output, "fc", arrays["x::fc"])
    changed = evaluate(mutated, "fc", arrays["x::fc"])
    delta = float(np.max(np.abs(changed - original)))
    if delta <= 0.01:
        raise AssertionError("serialized midpoint negative control did not alter forward")
    return {
        "name": directory.name,
        "gguf": str(output),
        "expected": str(directory / "expected.npz"),
        "bits": bits,
        "width": width,
        "rank": rank,
        "comparisons": comparisons,
        "negative_control_max_delta": delta,
        "factor_rounding_max_delta": factor_rounding_delta,
        "hashes": {p.name: digest(p) for p in (base, checkpoint, manifest, output, mutated)},
    }


def run(output):
    check_control()
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        create_case(output / f"a{bits}-k{width}-r{rank}", bits, width, rank)
        for bits in (1, 4, 8)
        for width, rank in ((33, 1), (65, 4), (130, 4))
    ]
    result = {
        "schema": "export_function_audit_v1",
        "hardware": "CPU synthetic; no native backend execution",
        "torch": torch.__version__,
        "gate": {"atol": ATOL, "rtol": RTOL},
        "cases": cases,
        "comparisons": sum(len(c["comparisons"]) for c in cases),
        "max_abs_error": max(p["max_abs_error"] for c in cases for p in c["comparisons"]),
    }
    (output / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)

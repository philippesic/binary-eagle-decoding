#!/usr/bin/env python3
"""Independent NumPy decoder for serialized row-W1Ax EAGLE GGUF functions.

This intentionally reads only the exported GGUF plus raw fixture inputs and
outputs. It does not import the exporter or any training forward implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

BASES = (
    "fc",
    "output",
    "blk.0.attn_q",
    "blk.0.attn_k",
    "blk.0.attn_v",
    "blk.0.attn_output",
    "blk.0.ffn_gate",
    "blk.0.ffn_down",
    "blk.0.ffn_up",
)
BOUNDARY = {
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
Q_PREFIX = "eagle3.w1a1.activation_quantizer."
CORR_PREFIX = "eagle3.fusion_correction."
AFFINE_PREFIX = "eagle3.affine_weights."


def f32(value):
    return np.asarray(value, dtype=np.float32)


def qk_inverse(rows: np.ndarray, heads: int) -> np.ndarray:
    """Invert GGUF's [head, half, head_width] RoPE row permutation."""
    n, k = rows.shape
    if n % (2 * heads):
        raise ValueError(f"row count {n} is not divisible by 2*{heads}")
    width = n // (2 * heads)
    return rows.reshape(heads, width, 2, k).swapaxes(1, 2).reshape(n, k)


def activation(x: np.ndarray, bits: int, delta: float, clip: float):
    """Reconstruct serialized learned A1/A4/A8 or F16 boundary values."""
    x = f32(x)
    if bits == 16:
        return x.astype(np.float16).astype(np.float32), None, f32(1.0)
    if bits == 1:
        # The scale reduction is explicitly F64 then rounded to F32.
        beta = f32(np.mean(np.abs(x).astype(np.float64), axis=-1, keepdims=True))
        threshold = f32(f32(delta) * beta)
        shifted = x if delta == 0 else f32(x - threshold)
        # Preserve negative subnormals; both signed zeroes are positive.
        sign = np.where(np.signbit(shifted) & (shifted != 0), -1, 1).astype(np.int8)
        return sign.astype(np.float32), sign, beta
    if bits not in (4, 8):
        raise ValueError(f"unsupported activation width {bits}")
    qmax = (1 << (bits - 1)) - 1
    maximum = np.max(np.abs(x), axis=-1, keepdims=True)
    limit = f32(maximum * f32(clip))
    beta = f32(limit / f32(qmax))
    inverse = np.zeros_like(limit, dtype=np.float32)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        np.divide(f32(qmax), limit, out=inverse, where=limit > 0)
        normalized = f32(x * inverse)
    overflow = ~np.isfinite(inverse)
    if np.any(overflow):
        denom = np.where(limit > 0, limit, f32(1)).astype(np.float64)
        fallback = f32((x.astype(np.float64) / denom) * qmax)
        normalized = np.where(overflow, fallback, normalized)
    codes = np.clip(np.rint(normalized), -qmax, qmax).astype(np.int8)
    return codes.astype(np.float32), codes, beta


def packed_signs(reader, base: str) -> tuple[np.ndarray, np.ndarray, int]:
    tensors = {t.name: t for t in reader.tensors}
    words_t = tensors[f"{base}.w1a1_packed"]
    scales_t = tensors[f"{base}.w1a1_scale"]
    words = np.asarray(words_t.data, dtype=np.uint32)
    scales = f32(scales_t.data).reshape(-1)
    logical_key = f"eagle3.w1a1.tensor.{(base + '.weight').replace('.', '_')}.logical_k"
    logical_k = int(reader.fields[logical_key].contents())
    if words.ndim != 2 or scales.shape != (words.shape[0],):
        raise ValueError(f"{base}: serialized packed/scales shape mismatch")
    if words.shape[1] != (logical_k + 31) // 32:
        raise ValueError(f"{base}: serialized logical K mismatch")
    shifts = np.arange(32, dtype=np.uint32)
    bits = ((words[..., None] >> shifts) & 1).reshape(words.shape[0], -1)
    # Export uses little bit order and nonnegative_is_one; padding has no effect.
    signs = np.where(bits[:, :logical_k] != 0, 1.0, -1.0).astype(np.float32)
    if base == "blk.0.attn_q":
        signs, scales = qk_inverse(signs, 32), qk_inverse(scales[:, None], 32)[:, 0]
    elif base == "blk.0.attn_k":
        signs, scales = qk_inverse(signs, 8), qk_inverse(scales[:, None], 8)[:, 0]
    return signs, scales, logical_k


def decode_case(gguf_path: Path, bits: int, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    gguf_py = Path(__file__).resolve().parents[4] / "third_party/llama.cpp/gguf-py"
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # type: ignore

    reader = GGUFReader(str(gguf_path))
    fields = {name: field.contents() for name, field in reader.fields.items()}
    quantized = fields.get(Q_PREFIX + "version") == 1
    outputs = {}
    for base in BASES:
        x = f32(inputs[base])
        logical_key = f"eagle3.w1a1.tensor.{(base + '.weight').replace('.', '_')}.logical_k"
        if x.shape[-1] != int(fields[logical_key]):
            raise ValueError(f"{base}: fixture width differs from serialized K")
        if quantized:
            prefix = Q_PREFIX + BOUNDARY[base] + "."
            delta = float(fields[prefix + "threshold_delta"])
            clip = float(fields[prefix + "clip_ratio"])
            values, codes, beta = activation(x, bits, delta, clip)
        else:
            values, codes, beta = activation(x, bits, 0.0, 1.0)
        signs, alpha, _ = packed_signs(reader, base)
        if bits == 16:
            dot = f32(values @ signs.T)
            out = f32(dot * alpha)
        else:
            # Integer dot is exact in int64; serialized row-scale then token-scale.
            dot = np.asarray(codes, dtype=np.int64) @ np.asarray(signs, dtype=np.int64).T
            out = f32(f32(f32(dot) * alpha) * beta)
        affine_key = AFFINE_PREFIX + "version"
        midpoint_name = base + ".w1ax_midpoint"
        if affine_key in fields and any(t.name == midpoint_name for t in reader.tensors):
            midpoint_t = next(t for t in reader.tensors if t.name == midpoint_name)
            midpoint = f32(midpoint_t.data).reshape(-1)
            if base == "blk.0.attn_q":
                midpoint = qk_inverse(midpoint[:, None], 32)[:, 0]
            elif base == "blk.0.attn_k":
                midpoint = qk_inverse(midpoint[:, None], 8)[:, 0]
            sums = np.sum(codes, axis=-1, keepdims=True, dtype=np.int64).astype(np.float32)
            out = f32(out + f32(f32(sums * midpoint) * beta))
        if base == "fc" and CORR_PREFIX + "version" in fields:
            tensors = {t.name: t for t in reader.tensors}
            v = np.asarray(tensors["fc.correction_v.weight"].data, dtype=np.float16).astype(np.float32)
            u = np.asarray(tensors["fc.correction_u.weight"].data, dtype=np.float16).astype(np.float32)
            # Match the contract's two separate F32 reductions and F16-rounded factors.
            latent = f32(x @ v.T)
            correction = f32(latent @ u.T)
            out = f32(out + correction)
            bias_name = fields[CORR_PREFIX + "bias_name"]
            if bias_name:
                bias = f32(tensors[bias_name].data).reshape(-1)
                out = f32(out + bias)
        outputs[base] = out
    return outputs


def validate(index_path: Path) -> dict:
    index = json.loads(index_path.read_text())
    gate = index["gate"]
    atol, rtol = float(gate["atol"]), float(gate["rtol"])
    records = []
    for case in index["cases"]:
        archive = np.load(case["expected"], allow_pickle=False)
        inputs = {base: archive["x::" + base] for base in BASES}
        expected = {base: archive["y::" + base] for base in BASES}
        actual = decode_case(Path(case["gguf"]), int(case["bits"]), inputs)
        max_abs, max_rel, bad = 0.0, 0.0, 0
        for base in BASES:
            got, want = actual[base], f32(expected[base])
            if got.shape != want.shape:
                raise ValueError(f"{case['name']} {base}: shape {got.shape} != {want.shape}")
            difference = np.abs(got - want)
            tolerance = atol + rtol * np.abs(want)
            max_abs = max(max_abs, float(np.max(difference)))
            denom = np.maximum(np.abs(want), atol)
            max_rel = max(max_rel, float(np.max(difference / denom)))
            bad += int(np.count_nonzero(difference > tolerance))
        records.append({"name": case["name"], "bits": case["bits"], "max_abs": max_abs, "max_rel_atol_floor": max_rel, "violations": bad})
    return {"cases": len(records), "gate": gate, "passed": all(r["violations"] == 0 for r in records), "records": records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = validate(args.index)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.report:
        args.report.write_text(rendered + "\n")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Audit every packed EAGLE-3 W1A1 row against the BF16 source checkpoint.

Run with the conversion environment, which provides gguf, safetensors, torch,
and numpy. This is a GGUF serialization audit, not a CUDA execution test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType, GGUFReader
from safetensors import safe_open

SOURCE_NAMES = {
    "fc": "fc.weight",
    "output": "lm_head.weight",
    "blk.0.attn_q": "midlayer.self_attn.q_proj.weight",
    "blk.0.attn_k": "midlayer.self_attn.k_proj.weight",
    "blk.0.attn_v": "midlayer.self_attn.v_proj.weight",
    "blk.0.attn_output": "midlayer.self_attn.o_proj.weight",
    "blk.0.ffn_gate": "midlayer.mlp.gate_proj.weight",
    "blk.0.ffn_down": "midlayer.mlp.down_proj.weight",
    "blk.0.ffn_up": "midlayer.mlp.up_proj.weight",
}


def gguf_qk_row_order(weight: np.ndarray, heads: int) -> np.ndarray:
    """Apply llama.cpp's RoPE Q/K row permutation before packing."""
    rows, logical_k = weight.shape
    if rows % (2 * heads):
        raise ValueError("Q/K rows are not divisible by twice the head count")
    return (
        weight.reshape(heads, 2, rows // heads // 2, logical_k)
        .swapaxes(1, 2)
        .reshape(rows, logical_k)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(source: Path, gguf_path: Path, rows_per_chunk: int = 128) -> dict:
    if rows_per_chunk < 1:
        raise ValueError("rows_per_chunk must be positive")
    reader = GGUFReader(gguf_path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    bases = sorted(
        name.removesuffix(".w1a1_packed") for name in tensors if name.endswith(".w1a1_packed")
    )
    if not bases or len(bases) != len(set(bases)):
        raise ValueError("GGUF has no distinct packed W1A1 tensors")
    if unknown := sorted(set(bases) - SOURCE_NAMES.keys()):
        raise ValueError(f"unknown packed EAGLE tensor(s): {unknown}")

    records = []
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        for base in bases:
            packed = tensors[f"{base}.w1a1_packed"]
            scales = tensors.get(f"{base}.w1a1_scale")
            if scales is None or f"{base}.weight" in tensors:
                raise ValueError(f"{base}: missing scale or dense shadow present")
            if (
                packed.tensor_type != GGMLQuantizationType.I32
                or scales.tensor_type != GGMLQuantizationType.F32
            ):
                raise ValueError(f"{base}: expected I32 packed words and F32 scales")
            source_name = SOURCE_NAMES[base]
            weight = checkpoint.get_tensor(source_name)
            if weight.ndim != 2:
                raise ValueError(f"{source_name}: expected a matrix")
            n_rows, logical_k = weight.shape
            canonical = weight.float().numpy()
            if base == "blk.0.attn_q":
                canonical = gguf_qk_row_order(canonical, 32)
            elif base == "blk.0.attn_k":
                canonical = gguf_qk_row_order(canonical, 8)
            words = (logical_k + 31) // 32
            if packed.data.shape != (n_rows, words) or scales.data.shape != (n_rows,):
                raise ValueError(f"{base}: GGUF and source shapes differ")
            bad_rows = 0
            bad_words = 0
            max_scale_error = 0.0
            for first in range(0, n_rows, rows_per_chunk):
                chunk = canonical[first : first + rows_per_chunk]
                if not np.isfinite(chunk).all():
                    raise ValueError(f"{source_name}: nonfinite source weight")
                signs = chunk >= 0
                if logical_k % 32:
                    signs = np.pad(signs, ((0, 0), (0, 32 - logical_k % 32)))
                expected = np.packbits(signs, axis=1, bitorder="little").view("<i4")
                observed = packed.data[first : first + len(chunk)]
                mismatch = expected != observed
                bad_words += int(np.count_nonzero(mismatch))
                bad_rows += int(np.count_nonzero(np.any(mismatch, axis=1)))
                expected_scales = np.mean(np.abs(chunk), axis=1, dtype=np.float32)
                actual_scales = scales.data[first : first + len(chunk)]
                max_scale_error = max(
                    max_scale_error,
                    float(np.max(np.abs(expected_scales - actual_scales))),
                )
            if bad_words or max_scale_error > 1e-6:
                raise ValueError(
                    f"{base}: {bad_words} packed word mismatches; max scale error {max_scale_error}"
                )
            records.append(
                {
                    "gguf_base": base,
                    "source_tensor": source_name,
                    "rows": n_rows,
                    "logical_k": logical_k,
                    "packed_words_per_row": words,
                    "packed_word_mismatches": bad_words,
                    "packed_row_mismatches": bad_rows,
                    "max_abs_scale_error": max_scale_error,
                }
            )
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "gguf": str(gguf_path.resolve()),
        "gguf_sha256": sha256(gguf_path),
        "packed_tensors": len(records),
        "audited_rows": sum(row["rows"] for row in records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="EAGLE model.safetensors")
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.gguf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary_keys = ("packed_tensors", "audited_rows", "gguf_sha256")
    print(json.dumps({key: result[key] for key in summary_keys}))


if __name__ == "__main__":
    main()

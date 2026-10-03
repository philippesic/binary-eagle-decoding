"""Compare both released BF16 copies with fixed F16 target tensors, in FP32.

This runs before deciding whether conversion may omit either draft copy.
The original checkpoint/config and fixed target are never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

from check_export import sha256


def compare(source: Path, target: Path, llama: Path) -> dict:
    sys.path.insert(0, str(llama / "gguf-py"))
    import numpy as np
    from gguf import GGUFReader, GGMLQuantizationType
    reader = GGUFReader(str(target))
    tensors = {t.name: t for t in reader.tensors}
    with source.open("rb") as f:
        header_bytes = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_bytes))
    report = {"schema": "dspark_frozen_comparison_v1", "source_sha256": sha256(source),
              "target_sha256": sha256(target), "canonical_format": "little-endian FP32 row-major", "tensors": {}}
    for suffix, destination in (("embed_tokens.weight", "token_embd.weight"), ("lm_head.weight", "output.weight")):
        name = next(n for n in header if n.endswith(suffix))
        row = header[name]
        tensor = tensors[destination]
        if row["dtype"] != "BF16" or tensor.tensor_type != GGMLQuantizationType.F16:
            raise ValueError("reference comparison requires released BF16 and frozen F16")
        if row["shape"] != [151936, 2560] or tensor.shape.tolist() != [2560, 151936]:
            raise ValueError("target/released shape or GGUF row order differs")
        count = tensor.n_elements
        values = np.memmap(source, mode="r", dtype="<u2", offset=8 + header_bytes + row["data_offsets"][0], shape=(count,))
        target_values = tensor.data.reshape(-1)
        left_hash, right_hash = hashlib.sha256(), hashlib.sha256()
        stats = dict(elements=count, source_dtype="BF16", target_dtype="F16", shape=row["shape"],
                     differing_elements=0, source_nonfinite=0, target_nonfinite=0,
                     bf16_to_f16_underflow=0, bf16_to_f16_overflow=0, max_abs_difference=0.0, first_difference=None)
        for start in range(0, count, 1024 * 1024):
            stop = min(start + 1024 * 1024, count)
            left = (values[start:stop].astype(np.uint32) << 16).view(np.float32)
            right = target_values[start:stop].astype(np.float32)
            left_hash.update(left.astype("<f4", copy=False).tobytes())
            right_hash.update(right.astype("<f4", copy=False).tobytes())
            finite_left, finite_right = np.isfinite(left), np.isfinite(right)
            stats["source_nonfinite"] += int(np.count_nonzero(~finite_left))
            stats["target_nonfinite"] += int(np.count_nonzero(~finite_right))
            cast = left.astype(np.float16)
            stats["bf16_to_f16_underflow"] += int(np.count_nonzero(finite_left & (left != 0) & (cast == 0)))
            stats["bf16_to_f16_overflow"] += int(np.count_nonzero(finite_left & ~np.isfinite(cast)))
            different = np.flatnonzero(left.view(np.uint32) != right.view(np.uint32))
            stats["differing_elements"] += len(different)
            if len(different) and stats["first_difference"] is None:
                stats["first_difference"] = start + int(different[0])
            finite = finite_left & finite_right
            if np.any(finite):
                stats["max_abs_difference"] = max(stats["max_abs_difference"], float(np.max(np.abs(left[finite] - right[finite]))))
        stats.update(source_canonical_sha256=left_hash.hexdigest(), target_canonical_sha256=right_hash.hexdigest())
        stats["exactly_equivalent"] = (stats["source_canonical_sha256"] == stats["target_canonical_sha256"] and
                                        stats["source_nonfinite"] == stats["target_nonfinite"] == 0)
        report["tensors"][suffix] = stats
    report["safe_to_borrow_embedding"] = report["tensors"]["embed_tokens.weight"]["exactly_equivalent"]
    report["safe_to_borrow_head"] = report["tensors"]["lm_head.weight"]["exactly_equivalent"]
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "target", "llama", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--conversion-config-out", type=Path)
    a = p.parse_args()
    result = compare(a.source, a.target, a.llama)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    if a.conversion_config_out:
        if not a.config:
            p.error("--conversion-config-out requires --config")
        cfg = json.loads(a.config.read_text())
        # Retaining original copies on a mismatch is a preparation choice only;
        # actual residency or any alternate precision track still needs admission.
        cfg["has_embed_tokens"] = not result["safe_to_borrow_embedding"]
        cfg["retain_lm_head"] = not result["safe_to_borrow_head"]
        a.conversion_config_out.write_text(json.dumps(cfg, indent=2) + "\n")

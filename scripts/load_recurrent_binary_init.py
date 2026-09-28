#!/usr/bin/env python3
"""CPU-only audit and loader for the frozen candidate-D binary initialization.

Returns GGUF row-order packed I32 signs and F32 group-128 scales for all nine
selected drafter linears. The trainable installer converts Q/K rows back to
the pinned PyTorch order. This does not load target weights or run inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from export_binary_rescue import Model, Type, raw_hash  # noqa: E402

from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH, unpack_signs  # noqa: E402

D_SHA256 = "10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate_d_arrays(
    path: Path, expected_sha256: str = D_SHA256
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Reject changed coverage, provenance, arithmetic, mapping or payload."""
    path = Path(path)
    actual_hash = sha256(path)
    if actual_hash != expected_sha256:
        raise ValueError("candidate-D GGUF SHA256 differs from frozen initialization")
    model = Model(path)
    prefix = "eagle3.w1a1."
    expected_groups = {"fusion", "attention", "ffn", "head"}
    expected_tensors = {base + ".weight" for base in CANDIDATE_D_BASE_TO_PATH}
    if (
        model.value(prefix + "version") != 2
        or model.value(prefix + "scale_group_size") != 128
        or set(model.value(prefix + "groups")) != expected_groups
        or set(model.value(prefix + "tensors")) != expected_tensors
        or model.value(prefix + "scale_rule") != "f32_nonnegative_least_squares"
        or model.value(prefix + "sign_rule") != "nonnegative_is_one"
        or model.value(prefix + "bit_order") != "little"
        or model.value(prefix + "arithmetic") != "f32"
    ):
        raise ValueError("GGUF is not the frozen nine-linear candidate D")
    mapping = model.tensors.get("d2t")
    if mapping is None or mapping.kind != Type.I64 or mapping.data.ndim != 1:
        raise ValueError("candidate D requires the frozen absolute I64 d2t map")
    absolute_ids = np.asarray(mapping.data, dtype=np.int64)
    if not len(absolute_ids) or np.any(absolute_ids < 0) or np.any(np.diff(absolute_ids) <= 0):
        raise ValueError("candidate-D d2t target IDs must be unique and increasing")
    arrays = {}
    layers = {}
    for base in CANDIDATE_D_BASE_TO_PATH:
        tensors, (width, rows) = model.projection(base)
        if len(tensors) != 2 or tensors[0].kind != Type.I32 or tensors[1].kind != Type.F32:
            raise ValueError(f"{base}: expected binary signs and F32 scales")
        packed = np.array(tensors[0].data, dtype="<i4", copy=True)
        scales = np.array(tensors[1].data, dtype=np.float32, copy=True)
        if packed.shape != (rows, (width + 31) // 32):
            raise ValueError(f"{base}: packed row shape mismatch")
        if scales.shape != (rows, (width + 127) // 128):
            raise ValueError(f"{base}: group-scale shape mismatch")
        unpack_signs(packed, width)  # also checks unused tail bits
        arrays[base] = packed, scales
        layers[base] = {
            "rows": rows,
            "logical_k": width,
            "packed_sha256": raw_hash(packed),
            "scale_sha256": raw_hash(scales),
            "zero_scales": int(np.count_nonzero(scales == 0)),
        }
    return arrays, {
        "schema": "recurrent_binary_candidate_d_init_v1",
        "execution_device": "cpu",
        "gguf_sha256": actual_hash,
        "d2t_sha256": raw_hash(absolute_ids),
        "draft_vocab_size": len(absolute_ids),
        "layers": layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=D_SHA256)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.audit.exists():
        parser.error("audit must be a new file")
    _, report = load_candidate_d_arrays(args.candidate_d, args.expected_sha256)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"gguf_sha256": report["gguf_sha256"], "layers": len(report["layers"])}))


if __name__ == "__main__":
    main()

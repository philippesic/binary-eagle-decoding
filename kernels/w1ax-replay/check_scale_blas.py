#!/usr/bin/env python3
"""Compare sampled native A16 replay outputs with the fitter's F32 BLAS formulas.

The matrices use only the replay's fixed token/output rows. BLAS algorithm
selection can depend on shape, so this is a bounded arithmetic audit, not a
claim about every full calibration reduction. No model or scale is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))
import gguf  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gguf", type=Path, required=True)
    p.add_argument("--capture-dir", type=Path, required=True)
    p.add_argument("--replay-jsonl", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model = {t.name: t for t in gguf.GGUFReader(a.gguf).tensors}
    gguf_hash = sha256(a.gguf)
    records = [json.loads(line) for line in a.replay_jsonl.read_text().splitlines() if line.strip()]
    selected = [r for r in records if r.get("record_type") == "operator_replay"]
    if not selected:
        raise ValueError("no operator replay records")
    for r in selected:
        if r["replay_bits"] != 16 or "sampled_native_outputs" not in r:
            raise ValueError("replay requires --act-bits 16 --emit-reference-values")
        path = a.capture_dir / r["capture"]
        data = path.read_bytes()
        magic, sequence, k, m, n, bits, raw_name = struct.unpack("<8sQQQQI128s", data[:172])
        if magic != b"W1AXACT1" or (sequence, k, m, n) != (
            r["sequence"],
            r["K"],
            r["M"],
            r["source_N"],
        ):
            raise ValueError("capture identity mismatch")
        name = raw_name.split(b"\0")[0].decode()
        if name.endswith(".weight"):
            name = name[:-7] + ".w1a1_packed"
        if name != r["name"] or bits != r["source_bits"]:
            raise ValueError("capture tensor or precision mismatch")
        x = np.frombuffer(data, dtype="<f4", offset=172).reshape(n, k)[r["token_indices"]].copy()
        x = x.astype(np.float16).astype(np.float32)
        rows = np.asarray(r["row_indices"], dtype=np.int64)
        packed = np.asarray(model[name].data).view(np.uint32).reshape(m, (k + 31) // 32)[rows]
        signs = (((packed[:, np.arange(k) // 32] >> (np.arange(k) % 32)) & 1) * 2).astype(
            np.float32
        ) - 1
        scales = (
            np.asarray(model[name.replace(".w1a1_packed", ".w1a1_scale")].data, dtype=np.float32)
            .reshape(m, -1)[rows]
            .copy()
        )
        xt = torch.from_numpy(x).to(a.device)
        st = torch.from_numpy(signs).to(a.device)
        at = torch.from_numpy(scales).to(a.device)
        if r["scale_group_size"] == 0:
            z = (xt @ st.T).T.unsqueeze(2)
        elif r["scale_group_size"] == 128:
            groups = (k + 127) // 128
            pad = groups * 128 - k
            xp = torch.nn.functional.pad(xt, (0, pad))
            sp = torch.nn.functional.pad(st, (0, pad))
            inputs = xp.reshape(len(x), groups, 128).permute(1, 0, 2)
            weights = sp.reshape(len(rows), groups, 128).permute(1, 2, 0)
            z = inputs.bmm(weights).permute(2, 1, 0).contiguous()
        else:
            raise ValueError("unsupported scale group size")
        prediction = z.bmm(at.unsqueeze(2)).squeeze(2).T.cpu().numpy()
        native = np.asarray(r["sampled_native_outputs"], dtype=np.float32).reshape(
            len(x), len(rows)
        )
        if not np.isfinite(prediction).all() or not np.isfinite(native).all():
            raise ValueError("nonfinite arithmetic output")
        delta = prediction.astype(np.float64) - native.astype(np.float64)
        print(
            json.dumps(
                {
                    "record_type": "scale_blas_reduction_diagnostic",
                    "name": name,
                    "capture": r["capture"],
                    "capture_sha256": sha256(path),
                    "gguf_sha256": gguf_hash,
                    "sequence": sequence,
                    "K": k,
                    "M": m,
                    "source_N": n,
                    "N": len(x),
                    "scale_group_size": r["scale_group_size"],
                    "token_indices": r["token_indices"],
                    "row_indices": r["row_indices"],
                    "checked_outputs": int(delta.size),
                    "max_abs_error": float(np.abs(delta).max()),
                    "mean_abs_error": float(np.abs(delta).mean()),
                    "rmse": float(np.sqrt(np.mean(delta * delta))),
                    "max_relative_error_1_plus_abs": float(
                        (np.abs(delta) / (1 + np.abs(native))).max()
                    ),
                    "native_sampled_outputs": native.flatten().tolist(),
                    "blas_sampled_outputs": prediction.flatten().tolist(),
                    "device": a.device,
                    "hardware": torch.cuda.get_device_name() if a.device == "cuda" else "cpu",
                    "torch_version": torch.__version__,
                    "tf32": False,
                    "arithmetic": "fitter F32 matmul/bmm formulas on bounded replay subset",
                    "serving_acceptance_metric": False,
                    "validation": "diagnostic_non_gate",
                }
            )
        )


if __name__ == "__main__":
    main()

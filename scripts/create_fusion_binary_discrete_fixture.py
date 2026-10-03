#!/usr/bin/env python3
"""Create reproducible correlated synthetic operands and tiny frozen GGUF on CPU.

Synthetic data exercises fitting/export only: it is neither captured TRAIN data
nor a deployable research model. Every artifact is written outside Git by caller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from fit_fusion_binary_discrete import array_hash, sha256


def create_fixture(output, *, seed=20261003, prompts=8, rows_per_prompt=24, width=64, outputs=8):
    if (
        prompts < 2
        or min(rows_per_prompt, width, outputs) < 1
        or prompts * rows_per_prompt > 1024
        or width > 1024
        or outputs > 64
    ):
        raise ValueError("fixture dimensions outside bounded synthetic budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    package = Path(os.environ.get("EAGLE_GGUF_PY", root / "third_party/llama.cpp/gguf-py"))
    sys.path.insert(0, str(package))
    from gguf import GGMLQuantizationType as Type
    from gguf import GGUFWriter

    rng = np.random.default_rng(seed)
    n = prompts * rows_per_prompt
    # Low-rank correlated inputs create the covariance opportunity for sign fitting.
    factors = rng.standard_normal((n, 6)).astype(np.float32)
    mixing = rng.standard_normal((6, width)).astype(np.float32)
    x = (factors @ mixing + np.float32(0.1) * rng.standard_normal((n, width))).astype(np.float32)
    if width > 2:
        x[:, 1] = x[:, 0] + np.float32(0.015) * rng.standard_normal(n).astype(np.float32)
        x[:, 2] = -x[:, 0] + np.float32(0.02) * rng.standard_normal(n).astype(np.float32)
    w = (rng.standard_normal((outputs, width)) * 0.3).astype(np.float32)
    # BF16 round-trip source reference, on CPU, to match frozen-weight precision.
    import torch

    w = torch.from_numpy(w).to(torch.bfloat16).float().numpy()
    w[0, 0] = np.float32(0)
    w[0, -1] = np.float32(-0.0)
    base = output / "synthetic_base.gguf"
    writer = GGUFWriter(base, "eagle3")
    writer.add_string("general.name", "synthetic fusion fitter fixture")
    writer.add_string("test.provenance", "synthetic algorithm and serialization only")
    writer.add_array("tokenizer.ggml.tokens", ["zero", "one", "two"])
    writer.add_array("test.integer_list", [1, 4, 9])
    writer.add_tensor("fc.weight", w.astype(np.float16))
    writer.add_tensor("blk.0.attn_q.weight", rng.standard_normal((4, width)).astype(np.float16))
    writer.add_tensor("output_norm.weight", np.array([1, 2, 3], dtype=np.float32))
    writer.add_tensor("d2t", np.array([2, 0, 1], dtype=np.int32), raw_dtype=Type.I32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    row_ids = np.array(
        [f"synthetic-p{i // rows_per_prompt}-row{i % rows_per_prompt}" for i in range(n)]
    )
    operands = output / "synthetic_operands.npz"
    np.savez(operands, raw_input=x, reference_weight=w, raw_join_ids=row_ids)
    description = {
        "seed": seed,
        "prompts": prompts,
        "rows_per_prompt": rows_per_prompt,
        "width": width,
        "outputs": outputs,
        "generator": "correlated Gaussian factors; BF16-rounded source",
        "synthetic_only": True,
    }
    (output / "synthetic_source.json").write_text(
        json.dumps(description, sort_keys=True, indent=2) + "\n"
    )
    rows = [
        {
            "row_id": str(row_ids[i]),
            "prompt_id": f"synthetic-p{i // rows_per_prompt}",
            "prompt_sha256": array_hash(
                np.frombuffer(
                    f"synthetic prompt {i // rows_per_prompt} seed {seed}".encode(), dtype=np.uint8
                )
            ),
            "depth": 0,
            "position": i % rows_per_prompt,
            "split": "train" if i // rows_per_prompt < prompts * 3 // 4 else "validation",
            "raw_input_sha256": array_hash(x[i].astype("<f4")),
        }
        for i in range(n)
    ]
    manifest = {
        "schema_version": 1,
        "projection": "fc",
        "raw_input_stage": "pre_activation_quantization",
        "source_data_split": "train",
        "eligibility": "training_allowed",
        "synthetic": True,
        "source": {
            "frozen_weights_sha256": array_hash(w),
            "base_gguf_sha256": sha256(base),
            "capture_manifest_sha256": sha256(output / "synthetic_source.json"),
        },
        "operands": operands.name,
        "operands_sha256": sha256(operands),
        "rows": rows,
    }
    manifest_path = output / "synthetic_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        **description,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "operands_sha256": sha256(operands),
        "base_gguf": str(base),
        "base_gguf_sha256": sha256(base),
        "script_sha256": sha256(__file__),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261003)
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--rows-per-prompt", type=int, default=24)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--outputs", type=int, default=8)
    a = p.parse_args()
    print(
        json.dumps(
            create_fixture(
                a.output_dir,
                seed=a.seed,
                prompts=a.prompts,
                rows_per_prompt=a.rows_per_prompt,
                width=a.width,
                outputs=a.outputs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

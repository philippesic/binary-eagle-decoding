#!/usr/bin/env python3
"""Probe pinned Qwen3 layer-0 CPU ggml norm and pre-RoPE Q/K/V arithmetic.

The native helper executes only a small ggml CPU graph. No model forward,
accelerator, verifier, or evaluation prompt is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

TARGET_SHA256 = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
LABELS = ("q", "k", "v")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(actual: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    denominator = np.linalg.norm(reference.astype(np.float64), axis=1)
    return {
        "exact_elements": int(np.count_nonzero(actual == reference)),
        "elements": int(actual.size),
        "max_abs": float(np.max(np.abs(delta))),
        "rms": float(np.sqrt(np.mean(delta * delta))),
        "median_relative_row_l2": float(
            np.median(np.linalg.norm(delta, axis=1) / np.maximum(denominator, 1e-12))
        ),
    }


def compile_helper(source: Path, llama_root: Path, cpu_build: Path, binary: Path) -> None:
    include = llama_root / "ggml/include"
    library = cpu_build / "bin"
    subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-O2",
            f"-I{include}",
            str(source),
            f"-L{library}",
            f"-Wl,-rpath,{library}",
            "-lggml-cpu",
            "-lggml-base",
            "-lggml",
            "-o",
            str(binary),
        ],
        check=True,
    )


def probe(
    hf_model: Path,
    target_gguf: Path,
    capture_dir: Path,
    llama_root: Path,
    cpu_build: Path,
) -> dict:
    sys.path.insert(0, str(llama_root / "gguf-py"))
    from gguf import GGMLQuantizationType, GGUFReader  # noqa: PLC0415

    if sha256(target_gguf) != TARGET_SHA256:
        raise ValueError("target GGUF differs from pinned FP16 model")
    cache_path = cpu_build / "CMakeCache.txt"
    cmake_cache = cache_path.read_text()
    for option in ("GGML_CUDA", "GGML_METAL", "GGML_ACCELERATE", "GGML_BLAS"):
        if f"{option}:BOOL=OFF" not in cmake_cache:
            raise ValueError(f"this CPU probe requires {option}=OFF")
    rounds_path = capture_dir / "forced-rounds.jsonl"
    first_round = json.loads(rounds_path.open().readline())
    prefix = first_round["prefix_token_ids"]
    if not isinstance(prefix, list) or not 1 <= len(prefix) <= 128:
        raise ValueError("expected one bounded frozen prompt prefix")
    index_path = hf_model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    reader = GGUFReader(target_gguf, mode="r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}

    def source(name: str) -> torch.Tensor:
        with safe_open(hf_model / index[name], framework="pt", device="cpu") as file:
            return file.get_tensor(name)

    embedding = tensors["token_embd.weight"]
    norm = tensors["blk.0.attn_norm.weight"]
    if (
        embedding.tensor_type != GGMLQuantizationType.F16
        or norm.tensor_type != GGMLQuantizationType.F32
    ):
        raise ValueError("unexpected embedding or norm GGUF type")
    if embedding.data.shape[1] != norm.data.size:
        raise ValueError("embedding and norm widths differ")
    hidden = norm.data.size
    selected_embedding = np.asarray(embedding.data[prefix], dtype=np.float16)
    hf_embedding = source("model.embed_tokens.weight")[prefix].to(torch.float16).numpy()
    if not np.array_equal(selected_embedding, hf_embedding):
        raise ValueError("frozen prompt HF/GGUF F16 embeddings differ")
    hf_norm = source("model.layers.0.input_layernorm.weight").float().numpy()
    if not np.array_equal(norm.data, hf_norm):
        raise ValueError("layer-0 norm source/GGUF differs")
    weights: dict[str, np.ndarray] = {}
    for label in LABELS:
        gguf = tensors[f"blk.0.attn_{label}.weight"]
        if gguf.tensor_type != GGMLQuantizationType.F16 or gguf.data.shape[1] != hidden:
            raise ValueError(f"unexpected {label} GGUF tensor")
        hf = source(f"model.layers.0.self_attn.{label}_proj.weight").to(torch.float16).numpy()
        if not np.array_equal(gguf.data, hf):
            raise ValueError(f"{label} HF F16 source/GGUF differs")
        weights[label] = np.asarray(gguf.data)

    torch.set_num_threads(1)
    x = torch.from_numpy(selected_embedding.copy()).float()
    scale = torch.rsqrt(x.square().mean(dim=1, keepdim=True) + 1e-6)
    hf_normalized = (x * scale * torch.from_numpy(norm.data.copy())).numpy()
    helper = Path(__file__).with_name("native_layer0_projection.cpp")
    with tempfile.TemporaryDirectory(prefix="layer0-projection-") as temporary:
        directory = Path(temporary)
        selected_embedding.astype("<f4").tofile(directory / "embedding.f32")
        np.asarray(norm.data, dtype="<f4").tofile(directory / "attn_norm.f32")
        for label in LABELS:
            np.asarray(weights[label], dtype="<f2").tofile(directory / f"{label}.f16")
        binary = directory / "native_layer0_projection"
        compile_helper(helper, llama_root, cpu_build, binary)
        subprocess.run(
            [
                str(binary),
                str(directory),
                str(len(prefix)),
                str(hidden),
                *(str(weights[label].shape[0]) for label in LABELS),
            ],
            check=True,
        )
        native_norm = np.fromfile(directory / "native_norm.f32", dtype="<f4").reshape(-1, hidden)
        if native_norm.shape != hf_normalized.shape:
            raise ValueError("native norm shape differs")
        comparison = {"norm_ggml_vs_f32": metrics(native_norm, hf_normalized)}
        for label in LABELS:
            native = np.fromfile(directory / f"native_{label}.f32", dtype="<f4").reshape(
                len(prefix), weights[label].shape[0]
            )
            w = torch.from_numpy(np.array(weights[label], copy=True)).float().T
            references = {
                "f32_hf_norm": hf_normalized,
                "f32_native_norm": native_norm,
                "f32_f16cast_native_norm": native_norm.astype(np.float16).astype(np.float32),
            }
            comparison[label] = {
                name: metrics(native, torch.from_numpy(value.copy()).matmul(w).numpy())
                for name, value in references.items()
            }
            comparison[label]["native_output_finite"] = bool(np.isfinite(native).all())

    source_shards = sorted(
        {
            index[name]
            for name in [
                "model.embed_tokens.weight",
                "model.layers.0.input_layernorm.weight",
                *(f"model.layers.0.self_attn.{label}_proj.weight" for label in LABELS),
            ]
        }
    )
    return {
        "schema": "target_layer0_ggml_cpu_projection_probe_v1",
        "execution": "native ggml CPU graph, one thread; F16 GGUF weights, F32 norm and output",
        "hardware": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "ggml_build_options": {
            option: "OFF" for option in ("GGML_CUDA", "GGML_METAL", "GGML_ACCELERATE", "GGML_BLAS")
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "prompt_tokens": len(prefix),
        "prompt_first_round_sha256": hashlib.sha256(
            (json.dumps(first_round, sort_keys=True) + "\n").encode()
        ).hexdigest(),
        "operands": {
            "embedding_rows_hf_f16_exact": True,
            "norm_hf_exact": True,
            "qkv_hf_f16_exact": True,
        },
        "comparison": comparison,
        "source_sha256": {
            "target_gguf": TARGET_SHA256,
            "capture_rounds": sha256(rounds_path),
            "hf_index": sha256(index_path),
            "hf_shards": {shard: sha256(hf_model / shard) for shard in source_shards},
            "native_helper": sha256(helper),
            "ggml_build_cache": sha256(cache_path),
            "ggml_cpu_library": sha256(cpu_build / "bin/libggml-cpu.dylib"),
        },
        "limits": "Layer 0 embedding/norm/QKV pre-RoPE only; no attention or later layers",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--llama-root", type=Path, required=True)
    parser.add_argument("--cpu-build", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists")
    result = probe(
        args.hf_model, args.target_gguf, args.capture_dir, args.llama_root, args.cpu_build
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"prompt_tokens": result["prompt_tokens"], "comparison": result["comparison"]})
    )


if __name__ == "__main__":
    main()

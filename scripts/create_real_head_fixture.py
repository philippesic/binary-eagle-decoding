"""Create a versioned, real EAGLE-head numerical fixture for the CUDA prototype.

Run with the project's GGUF conversion environment (which has torch, numpy and
gguf-py dependencies)::

    results/convert-env/bin/python scripts/create_real_head_fixture.py \
      --capture results/qat-head-capture-20260924/train.pt \
      --gguf models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf \
      --output results/real-head-fixture/w1a1-real-head.bin

All integer fields and arrays are little endian. The 96-byte header is
``<8sIIIIII32s32s``: magic, schema version, header size, K, output rows,
token rows, words per row, capture SHA256, and GGUF SHA256. The payload, in
order, is selected source row indices (i32[T]), packed head (u32[R,W]), head
scales (f32[R]), captured activations converted from BF16 to f32[T,K], expected
packed activations (u32[T,W]), double-sum mean-absolute activation scales
rounded to f32[T], exact integer dots (i32[T,R]), then F32 ordered outputs
(f32[T,R]). No padding or variable-length metadata is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party/llama.cpp/gguf-py"))

MAGIC = b"W1A1RH\x00\x00"
VERSION = 1
HEADER = struct.Struct("<8sIIIIII32s32s")
assert HEADER.size == 96


def sha256_file(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def load_capture(path: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("inputs"), torch.Tensor):
        raise ValueError("capture must contain an inputs tensor")
    inputs = payload["inputs"]
    if inputs.ndim != 2 or inputs.dtype != torch.bfloat16 or inputs.shape[0] < count:
        raise ValueError("capture must contain enough 2D BF16 input rows")
    if not torch.isfinite(inputs).all().item():
        raise ValueError("capture contains nonfinite inputs")
    # Fixed stratified positions avoid both RNG dependence and only taking the
    # first draft call. The final input row is always included.
    indices = np.linspace(0, inputs.shape[0] - 1, num=count, dtype=np.int32)
    if np.unique(indices).size != count:
        raise ValueError("capture row selection repeated an index")
    selected = inputs[torch.from_numpy(indices.astype(np.int64))]
    return indices, selected.float().numpy().astype("<f4", copy=False)


def load_packed_head(path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    try:
        import gguf
    except ImportError as exc:
        raise RuntimeError("GGUF reader dependencies are missing; use results/convert-env") from exc

    reader = gguf.GGUFReader(path)
    fields = {tensor.name: tensor for tensor in reader.tensors}
    if "output.weight" in fields:
        raise ValueError("GGUF contains a dense head instead of packed-only head")
    if "output.w1a1_packed" not in fields or "output.w1a1_scale" not in fields:
        raise ValueError("GGUF lacks packed head or scales")
    packed = fields["output.w1a1_packed"]
    scales = fields["output.w1a1_scale"]
    if packed.tensor_type != gguf.GGMLQuantizationType.I32:
        raise ValueError("packed head GGUF tensor must be I32")
    if scales.tensor_type != gguf.GGMLQuantizationType.F32:
        raise ValueError("head scales GGUF tensor must be F32")
    if packed.data.ndim != 2 or scales.data.ndim != 1:
        raise ValueError("packed head GGUF tensor ranks are invalid")
    rows, words = packed.data.shape
    if scales.data.shape != (rows,) or packed.shape.tolist() != [words, rows]:
        raise ValueError("GGUF packed head and scale dimensions disagree")
    field = reader.get_field("eagle3.w1a1_head.logical_k")
    if field is None:
        raise ValueError("GGUF has no logical K metadata")
    k = int(field.contents())
    if k < 1 or words != (k + 31) // 32:
        raise ValueError("GGUF packed word count disagrees with logical K")
    version_field = reader.get_field("eagle3.w1a1_head.version")
    if version_field is None or int(version_field.contents()) != 1:
        raise ValueError("unsupported GGUF packed-head contract version")
    scale_values = np.asarray(scales.data, dtype="<f4")
    if not np.isfinite(scale_values).all():
        raise ValueError("GGUF contains nonfinite head scales")
    return (
        np.asarray(packed.data, dtype="<i4").view("<u4").copy(),
        scale_values.copy(),
        k,
        int(version_field.contents()),
    )


def build_payload(
    indices: np.ndarray,
    activations: np.ndarray,
    weights: np.ndarray,
    weight_scales: np.ndarray,
    k: int,
) -> tuple[bytes, dict]:
    """Compute expected values independently of the CUDA reduction layout."""
    activations = np.asarray(activations, dtype="<f4")
    weights = np.asarray(weights, dtype="<u4")
    weight_scales = np.asarray(weight_scales, dtype="<f4")
    indices = np.asarray(indices, dtype="<i4")
    if (
        activations.ndim != 2
        or weights.ndim != 2
        or activations.shape[1] != k
        or weights.shape[1] != (k + 31) // 32
        or weight_scales.shape != (weights.shape[0],)
        or indices.shape != (activations.shape[0],)
        or not np.isfinite(activations).all()
        or not np.isfinite(weight_scales).all()
    ):
        raise ValueError("fixture arrays have invalid shape or nonfinite values")
    tokens, _ = activations.shape
    rows, words = weights.shape
    if not (tokens and rows and k and np.unique(indices).size == tokens):
        raise ValueError("fixture needs positive dimensions and unique source indices")

    # The native sign convention treats both signed zeros as positive.
    positive = activations >= np.float32(0)
    activation_words = np.zeros((tokens, words), dtype="<u4")
    for feature in range(k):
        activation_words[:, feature // 32] |= positive[:, feature].astype("<u4") << np.uint32(
            feature % 32
        )
    # float64 accumulation, then one F32 rounding, is the CPU oracle. The
    # existing CUDA packer uses F32 word partials and an F32 reduction tree.
    activation_scales = np.mean(np.abs(activations).astype(np.float64), axis=1).astype("<f4")
    dots = np.empty((tokens, rows), dtype="<i4")
    tail_mask = (1 << (k % 32)) - 1 if k % 32 else 0xFFFFFFFF
    for token in range(tokens):
        mismatches = np.zeros(rows, dtype=np.int32)
        for word in range(words):
            xor = weights[:, word] ^ activation_words[token, word]
            if word == words - 1:
                xor &= np.uint32(tail_mask)
            # uint32.bit_count is unavailable in supported NumPy versions.
            bits = xor.view(np.uint8).reshape(rows, 4)
            mismatches += np.unpackbits(bits, axis=1).sum(axis=1).astype(np.int32)
        dots[token] = k - 2 * mismatches
    outputs = np.empty((tokens, rows), dtype="<f4")
    for token in range(tokens):
        first = dots[token].astype("<f4") * weight_scales
        outputs[token] = first.astype("<f4") * activation_scales[token]
    sections = (
        indices,
        weights,
        weight_scales,
        activations,
        activation_words,
        activation_scales,
        dots,
        outputs,
    )
    payload = b"".join(section.tobytes(order="C") for section in sections)
    return payload, {
        "schema_version": VERSION,
        "k": k,
        "rows": rows,
        "tokens": tokens,
        "words": words,
        "selected_capture_indices": indices.tolist(),
        "payload_bytes": len(payload),
    }


def create_fixture(
    capture: Path,
    gguf_path: Path,
    output: Path,
    *,
    tokens: int = 8,
    expected_shape: tuple[int, int] = (32000, 2560),
) -> dict:
    manifest_path = output.with_name(output.name + ".json")
    if tokens < 1 or output.exists() or manifest_path.exists():
        raise ValueError(
            "tokens must be positive and fixture output/manifest must not already exist"
        )
    weights, scales, k, contract_version = load_packed_head(gguf_path)
    if (weights.shape[0], k) != expected_shape:
        raise ValueError(f"expected real packed head shape {expected_shape}")
    indices, activations = load_capture(capture, tokens)
    if activations.shape[1] != k:
        raise ValueError("capture and GGUF reduction widths differ")
    payload, summary = build_payload(indices, activations, weights, scales, k)
    capture_hash = sha256_file(capture)
    gguf_hash = sha256_file(gguf_path)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        HEADER.size,
        k,
        weights.shape[0],
        tokens,
        weights.shape[1],
        capture_hash,
        gguf_hash,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    manifest_temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(header)
            handle.write(payload)
        fixture_hash = sha256_file(temporary).hex()
        versions = {
            "project_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "llama_cpp_gitlink": subprocess.check_output(
                ["git", "rev-parse", "HEAD:third_party/llama.cpp"], cwd=ROOT, text=True
            ).strip(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "packed_head_contract_version": contract_version,
        }
        code_hashes = {
            str(path.relative_to(ROOT)): sha256_file(path).hex()
            for path in (
                ROOT / "scripts/create_real_head_fixture.py",
                ROOT / "kernels/w1a1_real_head_check.cu",
                ROOT / "kernels/w1a1_cuda.cu",
                ROOT / "kernels/w1a1_cuda.cuh",
            )
        }
        report = {
            **summary,
            "capture_sha256": capture_hash.hex(),
            "gguf_sha256": gguf_hash.hex(),
            "fixture_sha256": fixture_hash,
            "fixture_bytes": temporary.stat().st_size,
            "fixture_path": str(output),
            "manifest_path": str(manifest_path),
            "versions": versions,
            "code_sha256": code_hashes,
            "reference_arithmetic": {
                "activation_source": "captured BF16 converted exactly to F32",
                "zero_sign": "both +0 and -0 map to +1",
                "packed_bit_order": "feature k in word floor(k/32), bit k mod 32",
                "activation_scale": "mean(abs(F32 x)) summed in F64 then rounded once to F32",
                "weight_scale": "use actual F32 output.w1a1_scale from GGUF",
                "integer_dot": "K - 2 * popcount(masked XOR), exact int32",
                "output": "F32(F32(dot * weight_scale) * activation_scale), no bias",
                "cuda_scale_tolerance": "2e-7 + 5e-5*abs(expected)",
                "cuda_output_tolerance": "2e-4 + 5e-5*abs(expected)",
            },
        }
        with manifest_temporary.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output)
        try:
            manifest_temporary.replace(manifest_path)
        except Exception:
            output.unlink(missing_ok=True)
            raise
    finally:
        temporary.unlink(missing_ok=True)
        manifest_temporary.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(create_fixture(args.capture, args.gguf, args.output, tokens=args.tokens)))


if __name__ == "__main__":
    main()

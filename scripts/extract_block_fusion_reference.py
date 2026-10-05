#!/usr/bin/env python3
"""One-time bounded original GGUF FC/gamma/F32-epsilon reference extraction.

Uses the existing streaming header reader: tokenizer arrays are skipped, never
expanded. Only original dense F16/F32 block GGUFs are supported. Extracted NPY
files and externally pinned metadata let each A8/A1 fit avoid rereading a model.
This CPU preparation command does not create a model context or optimize it.
"""

from __future__ import annotations

import argparse
import json
import resource
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fusion_binary_gguf_stream import hash_range, parse_gguf  # noqa: E402

from w1a1_eagle.block_data import file_sha256  # noqa: E402
from w1a1_eagle.block_fusion import canonical_norm_epsilon, norm_epsilon_bits  # noqa: E402


def _stat(path):
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_dev, stat.st_ino)


def extract_reference(
    model, *, model_sha256, family, output_dir, max_array_bytes=512 * 1024 * 1024
):
    model, output = Path(model).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise ValueError("refuse to overwrite reference extraction history")
    if (
        family not in ("dspark", "dflash")
        or type(max_array_bytes) is not int
        or max_array_bytes < 1
    ):
        raise ValueError("declared supported family and positive extraction memory bound required")
    before = _stat(model)
    if file_sha256(model) != model_sha256:
        raise ValueError("original GGUF differs from external model SHA256 pin")
    source = parse_gguf(model)
    tensors = source["tensors"]
    if (
        source["architecture"] != "dflash"
        or "d2t" in tensors
        or ("markov_w1.weight" in tensors) != (family == "dspark")
        or any(k.startswith("dflash.w1ax.") for k in source["fields"])
    ):
        raise ValueError("reference must be original dense full-vocabulary declared block family")
    fc, gamma = tensors.get("fc.weight"), tensors.get("enc.output_norm.weight")
    if (
        fc is None
        or gamma is None
        or fc["kind"] not in (0, 1)
        or gamma["kind"] not in (0, 1)
        or len(fc["dims"]) != 2
        or len(gamma["dims"]) != 1
        or fc["dims"][1] != gamma["dims"][0]
        or fc["dims"][0] != 5 * fc["dims"][1]
        or "fc.bias" in tensors
        or "fc.scale" in tensors
    ):
        raise ValueError("unscaled original five-tap FC and post-FC norm F16/F32 tensors required")
    requested_bytes = 4 * (np.prod(fc["dims"], dtype=np.int64) + gamma["dims"][0])
    if requested_bytes > max_array_bytes:
        raise MemoryError("FC/gamma reference arrays exceed extraction bound")
    epsilon_key = "dflash.attention.layer_norm_rms_epsilon"
    field = source["fields"].get(epsilon_key)
    if field is None or field["kind"] != 6:
        raise ValueError("native block norm epsilon must be original GGUF FLOAT32")
    with model.open("rb") as stream:
        stream.seek(field["start"])
        length = struct.unpack("<Q", stream.read(8))[0]
        if (
            stream.read(length).decode() != epsilon_key
            or struct.unpack("<I", stream.read(4))[0] != 6
        ):
            raise ValueError("epsilon source field changed during extraction")
        encoded_epsilon = stream.read(4)
        epsilon = canonical_norm_epsilon(struct.unpack("<f", encoded_epsilon)[0])
        bits = struct.unpack("<I", encoded_epsilon)[0]
        if norm_epsilon_bits(epsilon) != bits:
            raise ValueError("norm epsilon source bits are not canonical positive F32")
    output.mkdir(parents=True)
    descriptors = {}
    for name, tensor, filename in (
        ("fc.weight", fc, "fc-weight.npy"),
        ("enc.output_norm.weight", gamma, "fc-norm.npy"),
    ):
        shape = tuple(reversed(tensor["dims"]))
        path = output / filename
        original = np.memmap(
            model,
            dtype="<f2" if tensor["kind"] == 1 else "<f4",
            offset=tensor["start"],
            shape=shape,
            mode="r",
        )
        destination = np.lib.format.open_memmap(path, mode="w+", dtype="<f4", shape=shape)
        for first in range(0, len(original), 64):
            chunk = original[first : first + 64].astype(np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError("original model fusion/norm tensor is nonfinite")
            destination[first : first + 64] = chunk
        destination.flush()
        del destination, original
        with model.open("rb") as stream:
            payload_sha = hash_range(stream, tensor["start"], tensor["nbytes"])
        descriptors[name] = {
            "name": name,
            "shape": list(shape),
            "kind": "F16" if tensor["kind"] == 1 else "F32",
            "payload_sha256": payload_sha,
        }
    if _stat(model) != before:
        raise ValueError("original GGUF changed during reference extraction")
    metadata = {
        "schema": "block_fusion_reference_v1",
        "family": family,
        "model_sha256": model_sha256,
        "weights_sha256": file_sha256(output / "fc-weight.npy"),
        "norm_sha256": file_sha256(output / "fc-norm.npy"),
        "epsilon": epsilon,
        "epsilon_f32_bits": bits,
        "epsilon_source_key": epsilon_key,
        "fc_source": descriptors["fc.weight"],
        "norm_source": descriptors["enc.output_norm.weight"],
    }
    metadata_path = output / "reference.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--family", choices=("dspark", "dflash"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-array-bytes", type=int, default=512 * 1024 * 1024)
    args = parser.parse_args()
    metadata = extract_reference(
        args.model,
        model_sha256=args.model_sha256,
        family=args.family,
        output_dir=args.output_dir,
        max_array_bytes=args.max_array_bytes,
    )
    print(
        json.dumps(
            {
                "reference_sha256": file_sha256(args.output_dir / "reference.json"),
                "model_sha256": metadata["model_sha256"],
                "epsilon_f32_bits": metadata["epsilon_f32_bits"],
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024),
                "hardware": "CPU reference extraction only; no model context/CUDA",
            }
        )
    )


if __name__ == "__main__":
    main()

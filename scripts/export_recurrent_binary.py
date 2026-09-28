#!/usr/bin/env python3
"""Export a jointly trained nine-linear W1A16 EAGLE checkpoint on CPU.

The checkpoint is an uncompressed NPZ with exactly two F32 arrays per original
checkpoint weight: ``<source name>.latent`` with shape [rows, K] and
``<source name>.scale`` with shape [rows, ceil(K/128)]. Its JSON manifest has
``schema_version: 1``, ``base_gguf_sha256``, ``training_arithmetic``, and
``projections`` mapping each GGUF base to
``{"checkpoint_name": ..., "shape": [rows, K]}``. All arrays
are in original checkpoint row order. Q and K rows are permuted here into the
GGUF RoPE layout before packing. No accelerator or model execution is used.

This writes truthful ``f32_learned_nonnegative`` metadata. The fork's native
loader recognizes that rule; each export still needs loader and numeric gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/llama.cpp/gguf-py"))
from audit_eagle_w1a1_gguf import SOURCE_NAMES, gguf_qk_row_order  # noqa: E402
from gguf import GGMLQuantizationType as Type  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402

PREFIX = "eagle3.w1a1."
GROUP_SIZE = 128
GROUPS = ("fusion", "attention", "ffn", "head")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_hash(array: np.ndarray) -> str:
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def tensor_key(base: str, suffix: str) -> str:
    return PREFIX + "tensor." + (base + ".weight").replace(".", "_") + "." + suffix


def check_manifest(manifest: dict, base_hash: str) -> dict[str, tuple[str, tuple[int, int]]]:
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "base_gguf_sha256",
        "training_arithmetic",
        "projections",
    }:
        raise ValueError("manifest requires version, base GGUF hash, arithmetic and projections")
    if manifest["schema_version"] != 1 or manifest["base_gguf_sha256"] != base_hash:
        raise ValueError("manifest version or base GGUF hash mismatch")
    if manifest["training_arithmetic"] not in ("native_order", "group_matmul"):
        raise ValueError("unknown declared training arithmetic")
    projections = manifest["projections"]
    if not isinstance(projections, dict) or set(projections) != set(SOURCE_NAMES):
        raise ValueError("manifest must declare exactly nine selected projections")
    checked = {}
    for base, expected_name in SOURCE_NAMES.items():
        item = projections[base]
        if not isinstance(item, dict) or set(item) != {"checkpoint_name", "shape"}:
            raise ValueError(f"{base}: invalid projection declaration")
        shape = item["shape"]
        if item["checkpoint_name"] != expected_name:
            raise ValueError(f"{base}: checkpoint tensor name mismatch")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(type(x) is not int or x <= 0 for x in shape)
        ):
            raise ValueError(f"{base}: invalid declared shape")
        if base == "blk.0.attn_q" and shape[0] % 64:
            raise ValueError("Q row count must be divisible by 64")
        if base == "blk.0.attn_k" and shape[0] % 16:
            raise ValueError("K row count must be divisible by 16")
        checked[base] = expected_name, tuple(shape)
    return checked


def check_base(reader: GGUFReader, expected: dict) -> dict:
    if reader.fields["general.architecture"].contents() != "eagle3":
        raise ValueError("base GGUF must have eagle3 architecture")
    if reader.byte_order != "I":
        raise ValueError("only little-endian base GGUF is supported")
    if any(key.startswith(PREFIX) for key in reader.fields):
        raise ValueError("base GGUF contains existing binary metadata")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError("duplicate tensor names")
    if any(name.endswith((".w1a1_packed", ".w1a1_scale")) for name in tensors):
        raise ValueError("base GGUF contains binary shadows")
    for name in tensors:
        match = re.match(r"^blk\.(\d+)\.", name)
        if match and match.group(1) != "0":
            raise ValueError("base GGUF has an extra decoder layer")
    for base, (_, shape) in expected.items():
        name = base + ".weight"
        tensor = tensors.get(name)
        if tensor is None or tensor.tensor_type != Type.F16:
            raise ValueError(f"{name}: missing original F16 tensor")
        if tuple(map(int, tensor.data.shape)) != shape:
            raise ValueError(f"{name}: base tensor and declared shape differ")
    return tensors


def pack(latent: np.ndarray) -> np.ndarray:
    signs = latent >= 0  # zero sign is positive, including negative zero
    if latent.shape[1] % 32:
        signs = np.pad(signs, ((0, 0), (0, 32 - latent.shape[1] % 32)))
    return np.ascontiguousarray(np.packbits(signs, axis=1, bitorder="little").view("<i4"))


def load_checkpoint(checkpoint: Path, expected: dict) -> dict:
    required = {name + suffix for name, _ in expected.values() for suffix in (".latent", ".scale")}
    arrays = {}
    with np.load(checkpoint, allow_pickle=False) as archive:
        if set(archive.files) != required or len(archive.files) != len(required):
            raise ValueError("checkpoint must contain exactly nine latent/scale pairs")
        for base, (name, shape) in expected.items():
            latent = archive[name + ".latent"]
            scale = archive[name + ".scale"]
            groups = (shape[1] + GROUP_SIZE - 1) // GROUP_SIZE
            if latent.dtype != np.float32 or latent.shape != shape:
                raise ValueError(f"{name}: latent must be F32 with declared shape")
            if scale.dtype != np.float32 or scale.shape != (shape[0], groups):
                raise ValueError(f"{name}: scale must be F32 [rows, ceil(K/128)]")
            if not np.isfinite(latent).all():
                raise ValueError(f"{name}: nonfinite latent")
            if (
                not np.isfinite(scale).all()
                or (scale < 0).any()
                or ((scale == 0) & np.signbit(scale)).any()
            ):
                raise ValueError(f"{name}: scales must be finite and nonnegative")
            if base == "blk.0.attn_q":
                latent, scale = gguf_qk_row_order(latent, 32), gguf_qk_row_order(scale, 32)
            elif base == "blk.0.attn_k":
                latent, scale = gguf_qk_row_order(latent, 8), gguf_qk_row_order(scale, 8)
            arrays[base] = {
                "packed": pack(latent),
                "scale": np.ascontiguousarray(scale),
                "logical_k": shape[1],
                "latent_sha256": raw_hash(archive[name + ".latent"]),
                "source_scale_sha256": raw_hash(archive[name + ".scale"]),
            }
    return arrays


def same_tensor(left, right) -> bool:
    return (
        left.tensor_type == right.tensor_type
        and tuple(map(int, left.shape)) == tuple(map(int, right.shape))
        and raw_hash(left.data) == raw_hash(right.data)
    )


def export_model(base_path: Path, checkpoint: Path, manifest_path: Path, output: Path) -> dict:
    """Validate, write atomically, reread, and return a provenance report."""
    base_path, checkpoint, manifest_path, output = map(
        Path, (base_path, checkpoint, manifest_path, output)
    )
    if output.exists():
        raise FileExistsError(output)
    base_hash = sha256(base_path)
    manifest = json.loads(manifest_path.read_text())
    expected = check_manifest(manifest, base_hash)
    reader = GGUFReader(base_path)
    tensors = check_base(reader, expected)
    arrays = load_checkpoint(checkpoint, expected)
    selected_names = {base + ".weight" for base in SOURCE_NAMES}
    preserved = {name: tensor for name, tensor in tensors.items() if name not in selected_names}
    expected_names = set(preserved) | {
        base + suffix for base in SOURCE_NAMES for suffix in (".w1a1_packed", ".w1a1_scale")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".recurrent-binary-export-", dir=output.parent) as tmp:
        temporary = Path(tmp) / "model.gguf"
        writer = GGUFWriter(temporary, "eagle3")
        for key, field in reader.fields.items():
            if not key.startswith("GGUF.") and key != "general.architecture":
                writer.add_key_value(
                    key,
                    field.contents(),
                    field.types[0],
                    field.types[-1] if len(field.types) > 1 else None,
                )
        writer.add_uint32(PREFIX + "version", 2)
        writer.add_uint32(PREFIX + "scale_group_size", GROUP_SIZE)
        writer.add_array(PREFIX + "groups", list(GROUPS))
        writer.add_array(PREFIX + "tensors", sorted(selected_names))
        writer.add_string(PREFIX + "bit_order", "little")
        writer.add_string(PREFIX + "sign_rule", "nonnegative_is_one")
        writer.add_string(PREFIX + "scale_rule", "f32_learned_nonnegative")
        writer.add_string(PREFIX + "arithmetic", "f32")
        for tensor in reader.tensors:
            base = tensor.name.removesuffix(".weight")
            if tensor.name in selected_names:
                pair = arrays[base]
                writer.add_uint32(tensor_key(base, "logical_k"), pair["logical_k"])
                writer.add_string(tensor_key(base, "packed"), base + ".w1a1_packed")
                writer.add_string(tensor_key(base, "scale"), base + ".w1a1_scale")
                writer.add_tensor(base + ".w1a1_packed", pair["packed"], raw_dtype=Type.I32)
                writer.add_tensor(base + ".w1a1_scale", pair["scale"], raw_dtype=Type.F32)
            else:
                writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        reread = GGUFReader(temporary)
        actual = {tensor.name: tensor for tensor in reread.tensors}
        if len(actual) != len(reread.tensors) or set(actual) != expected_names:
            raise ValueError("export tensor inventory mismatch")
        for name, tensor in preserved.items():
            if not same_tensor(tensor, actual[name]):
                raise ValueError(f"nonselected tensor changed: {name}")
        for base, pair in arrays.items():
            for suffix, kind in (("packed", Type.I32), ("scale", Type.F32)):
                name = base + ".w1a1_" + suffix
                observed = actual[name]
                if observed.tensor_type != kind or raw_hash(observed.data) != raw_hash(
                    pair[suffix]
                ):
                    raise ValueError(f"export mismatch: {name}")
                if observed.data.shape != pair[suffix].shape:
                    raise ValueError(f"export shape mismatch: {name}")
        for key, field in reader.fields.items():
            if not key.startswith("GGUF.") and key != "general.architecture":
                if reread.fields[key].contents() != field.contents():
                    raise ValueError(f"source metadata changed: {key}")
        if reread.fields[PREFIX + "scale_rule"].contents() != "f32_learned_nonnegative":
            raise ValueError("export scale rule mismatch")
        output_hash = sha256(temporary)
        report = {
            "schema_version": 1,
            "base_gguf": {"path": str(base_path), "sha256": base_hash},
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "checkpoint_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "output": {"path": str(output), "sha256": output_hash},
            "scale_rule": "f32_learned_nonnegative",
            "training_arithmetic": manifest["training_arithmetic"],
            "native_loader_gate": "verify loader and full-drafter numeric parity before deployment",
            "serialization_audit_passed": True,
            "projections": {
                base: {
                    "checkpoint_name": expected[base][0],
                    "shape": list(expected[base][1]),
                    "latent_sha256": pair["latent_sha256"],
                    "source_scale_sha256": pair["source_scale_sha256"],
                    "packed_sha256": raw_hash(pair["packed"]),
                    "gguf_scale_sha256": raw_hash(pair["scale"]),
                }
                for base, pair in arrays.items()
            },
            "preserved_tensors": {
                name: {"type": tensor.tensor_type.name, "raw_sha256": raw_hash(tensor.data)}
                for name, tensor in preserved.items()
            },
        }
        temporary.rename(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="original F16 EAGLE GGUF")
    parser.add_argument("--checkpoint", type=Path, required=True, help="F32 latent/scale NPZ")
    parser.add_argument("--manifest", type=Path, required=True, help="shape and name JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve() == args.output.resolve():
        parser.error("audit and output must be distinct new files")
    report = export_model(args.base, args.checkpoint, args.manifest, args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "audit": str(args.audit)}))


if __name__ == "__main__":
    main()

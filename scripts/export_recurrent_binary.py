#!/usr/bin/env python3
"""Export jointly trained nine-linear EAGLE group128/A16 or row W1Ax on CPU.

The checkpoint is an uncompressed NPZ with exactly two F32 arrays per original
checkpoint weight: ``<source name>.latent`` with shape [rows, K] and
``<source name>.scale`` with shape [rows, ceil(K/128)] for legacy group128/A16,
or [rows] for row W1Ax. The schema-v1 group manifest has
``base_gguf_sha256``, ``training_arithmetic``, and
``projections`` mapping each GGUF base to
``{"checkpoint_name": ..., "shape": [rows, K]}``. All arrays
are in original checkpoint row order. Schema-v2 row manifests also pin the
checkpoint hash, activation bits and quantizer rule. Q and K rows are permuted
here into the GGUF RoPE layout before packing. No model execution is used.

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
from gguf import GGUFReader, GGUFValueType, GGUFWriter  # noqa: E402

PREFIX = "eagle3.w1a1."
GROUP_SIZE = 128
GROUPS = ("fusion", "attention", "ffn", "head")
QUANTIZER_BOUNDARIES = ("fc", "qkv", "attn_output", "gate_up", "down", "head")
QUANTIZER_PREFIX = PREFIX + "activation_quantizer."
LEARNED_ACTIVATION_RULE = "learned_scalar_a1_threshold_a4a8_clip_v1"


def check_activation_quantizers(value: dict, bits: int) -> dict:
    """Validate effective F32 scalars, including the shared-boundary inventory."""
    if (
        bits not in (1, 4, 8)
        or not isinstance(value, dict)
        or set(value) != {"version", "boundaries"}
        or type(value["version"]) is not int
        or value["version"] != 1
    ):
        raise ValueError("unsupported learned activation quantizers")
    boundaries = value["boundaries"]
    if not isinstance(boundaries, dict) or set(boundaries) != set(QUANTIZER_BOUNDARIES):
        raise ValueError("learned quantizers require exactly six shared boundaries")
    checked = {}
    for boundary in QUANTIZER_BOUNDARIES:
        item = boundaries[boundary]
        if (
            not isinstance(item, dict)
            or set(item) != {"bits", "threshold_delta", "clip_ratio"}
            or type(item["bits"]) is not int
            or item["bits"] != bits
        ):
            raise ValueError(f"{boundary}: incompatible learned quantizer bits or fields")
        params = {}
        for name in ("threshold_delta", "clip_ratio"):
            scalar = item[name]
            if (
                type(scalar) not in (int, float)
                or not np.isfinite(scalar)
                or abs(scalar) > np.finfo(np.float32).max
            ):
                raise ValueError(f"{boundary}: nonfinite or non-scalar {name}")
            params[name] = float(np.float32(scalar))
            if params[name] != scalar:
                raise ValueError(f"{boundary}: {name} must be an effective F32 scalar")
        delta, clip = params["threshold_delta"], params["clip_ratio"]
        if not 0 < clip <= 1 or (bits == 1 and clip != 1) or (bits != 1 and delta != 0):
            raise ValueError(f"{boundary}: incompatible learned quantizer parameters")
        checked[boundary] = {"bits": bits, **params}
    return {"version": 1, "boundaries": checked}


CORRECTION_PREFIX = "eagle3.fusion_correction."
CORRECTION_ARITHMETIC = "raw_f32_v_f16_dot_f32_u_f16_dot_f32_add_base_f32_bias_f32"


def check_fusion_correction(value: dict) -> dict:
    required = {"version", "rank", "u_name", "v_name", "bias_name", "bias_bound", "arithmetic"}
    if (
        not isinstance(value, dict)
        or set(value) != required
        or type(value["version"]) is not int
        or value["version"] != 1
        or type(value["rank"]) is not int
        or value["rank"] not in (1, 4)
    ):
        raise ValueError("unsupported fusion correction descriptor")
    if (
        value["u_name"] != "fc.correction_u.weight"
        or value["v_name"] != "fc.correction_v.weight"
        or value["arithmetic"] != CORRECTION_ARITHMETIC
    ):
        raise ValueError("incompatible fusion correction names or arithmetic")
    if value["bias_name"] is None:
        if value["bias_bound"] is not None:
            raise ValueError("fusion correction bias bound without bias")
    elif (
        value["bias_name"] != "fc.correction_bias"
        or type(value["bias_bound"]) not in (int, float)
        or not np.isfinite(value["bias_bound"])
        or not 0 < value["bias_bound"] <= np.finfo(np.float32).max
    ):
        raise ValueError("fusion correction bias requires a finite positive bound")
    return dict(value)


def load_fusion_correction(checkpoint: Path, value: dict, shape: tuple) -> dict:
    descriptor = check_fusion_correction(value)
    rank = descriptor["rank"]
    specs = {
        descriptor["u_name"]: (np.float16, (shape[0], rank)),
        descriptor["v_name"]: (np.float16, (rank, shape[1])),
    }
    if descriptor["bias_name"]:
        specs[descriptor["bias_name"]] = (np.float32, (shape[0],))
    arrays = {}
    with np.load(checkpoint, allow_pickle=False) as archive:
        for name, (dtype, expected_shape) in specs.items():
            if name not in archive.files:
                raise ValueError(f"missing fusion correction tensor {name}")
            array = archive[name]
            if (
                array.dtype != dtype
                or array.shape != expected_shape
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"invalid fusion correction tensor {name}")
            if (
                name == descriptor["bias_name"]
                and (np.abs(array) > np.float32(descriptor["bias_bound"])).any()
            ):
                raise ValueError("fusion correction bias exceeds declared bound")
            arrays[name] = np.ascontiguousarray(array)
    return arrays


AFFINE_PREFIX = "eagle3.affine_weights."
AFFINE_ARITHMETIC = "integer_dot_alpha_beta_plus_integer_sum_midpoint_beta_before_bias_f32"


def check_affine_weights(value: dict) -> dict:
    required = {"version", "coverage", "arithmetic", "tensors"}
    if (
        not isinstance(value, dict)
        or set(value) != required
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["coverage"] not in ("fusion", "all")
        or value["arithmetic"] != AFFINE_ARITHMETIC
    ):
        raise ValueError("unsupported affine weight descriptor")
    expected = {"fc"} if value["coverage"] == "fusion" else set(SOURCE_NAMES)
    if (
        not isinstance(value["tensors"], dict)
        or set(value["tensors"]) != expected
        or any(name != base + ".w1ax_midpoint" for base, name in value["tensors"].items())
    ):
        raise ValueError("affine weight coverage or tensor names mismatch")
    return value


def load_affine_weights(checkpoint: Path, descriptor: dict, expected: dict) -> dict:
    check_affine_weights(descriptor)
    result = {}
    with np.load(checkpoint, allow_pickle=False) as archive:
        for base, name in descriptor["tensors"].items():
            if name not in archive.files:
                raise ValueError(f"missing affine midpoint {name}")
            array = archive[name]
            if (
                array.dtype != np.float32
                or array.shape != (expected[base][1][0],)
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"invalid affine midpoint {name}")
            if base in ("blk.0.attn_q", "blk.0.attn_k"):
                array = gguf_qk_row_order(array[:, None], 32 if base.endswith("_q") else 8)[:, 0]
            result[name] = np.ascontiguousarray(array)
    return result


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
    if not isinstance(manifest, dict) or manifest.get("schema_version") not in (1, 2, 3, 4, 5):
        raise ValueError("manifest schema must be v1 group128, v2 row or v3 learned row")
    if manifest["schema_version"] == 1:
        required = {"schema_version", "base_gguf_sha256", "training_arithmetic", "projections"}
        if set(manifest) != required:
            raise ValueError("group manifest requires version, base hash, arithmetic, projections")
        if manifest["training_arithmetic"] not in ("native_order", "group_matmul"):
            raise ValueError("unknown declared training arithmetic")
    else:
        required = {
            "schema_version",
            "base_gguf_sha256",
            "checkpoint_sha256",
            "scale_layout",
            "activation_bits",
            "activation_rule",
            "weight_rule",
            "qk_row_order",
            "export_status",
            "objective",
            "projections",
        }
        if manifest["schema_version"] in (3, 4, 5):
            required.add("activation_quantizers")
        if manifest["schema_version"] in (4, 5):
            required.add("fusion_correction")
            if manifest["schema_version"] == 4 or manifest.get("fusion_correction") is not None:
                check_fusion_correction(manifest.get("fusion_correction"))
        if manifest["schema_version"] == 5:
            required.add("affine_weights")
            check_affine_weights(manifest.get("affine_weights"))
        if set(manifest) != required:
            raise ValueError("row manifest has missing or extra contract fields")
        if (
            manifest["scale_layout"] != "row"
            or manifest["activation_bits"] not in (1, 4, 8, 16)
            or manifest["activation_rule"]
            != (
                LEARNED_ACTIVATION_RULE
                if manifest.get("activation_quantizers") is not None
                else "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive"
            )
            or manifest["weight_rule"] != "hard_sign_zero_positive_clipped_identity_ste"
            or manifest["qk_row_order"] != "original_checkpoint"
            or manifest["export_status"] != "row_w1ax_requires_native_validation"
            or manifest["objective"] not in ("hard_ce", "compact_probability")
            or not isinstance(manifest["checkpoint_sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", manifest["checkpoint_sha256"])
        ):
            raise ValueError("unsupported row checkpoint contract")
        if manifest["schema_version"] == 3 or manifest.get("activation_quantizers") is not None:
            check_activation_quantizers(
                manifest["activation_quantizers"], manifest["activation_bits"]
            )
    if manifest["base_gguf_sha256"] != base_hash:
        raise ValueError("manifest version or base GGUF hash mismatch")
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
    if any(key.startswith((PREFIX, CORRECTION_PREFIX, AFFINE_PREFIX)) for key in reader.fields):
        raise ValueError("base GGUF contains existing binary metadata")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError("duplicate tensor names")
    if any(
        name.endswith((".w1a1_packed", ".w1a1_scale", ".w1ax_midpoint"))
        or name.startswith("fc.correction")
        for name in tensors
    ):
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


def load_checkpoint(
    checkpoint: Path, expected: dict, *, row_scale: bool = False, extra_names: set = frozenset()
) -> dict:
    required = {name + suffix for name, _ in expected.values() for suffix in (".latent", ".scale")}
    required |= extra_names
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
            scale_shape = (shape[0],) if row_scale else (shape[0], groups)
            if scale.dtype != np.float32 or scale.shape != scale_shape:
                raise ValueError(f"{name}: scale must be F32 with shape {scale_shape}")
            if not np.isfinite(latent).all():
                raise ValueError(f"{name}: nonfinite latent")
            if (
                not np.isfinite(scale).all()
                or (scale < 0).any()
                or ((scale == 0) & np.signbit(scale)).any()
            ):
                raise ValueError(f"{name}: scales must be finite and nonnegative")
            if base == "blk.0.attn_q":
                latent = gguf_qk_row_order(latent, 32)
                scale = (
                    gguf_qk_row_order(scale[:, None], 32)[:, 0]
                    if row_scale
                    else gguf_qk_row_order(scale, 32)
                )
            elif base == "blk.0.attn_k":
                latent = gguf_qk_row_order(latent, 8)
                scale = (
                    gguf_qk_row_order(scale[:, None], 8)[:, 0]
                    if row_scale
                    else gguf_qk_row_order(scale, 8)
                )
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
    row_scale = manifest["schema_version"] in (2, 3, 4, 5)
    if row_scale and sha256(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("row checkpoint SHA256 differs from manifest")
    reader = GGUFReader(base_path)
    tensors = check_base(reader, expected)
    correction = manifest.get("fusion_correction")
    correction_arrays = (
        load_fusion_correction(checkpoint, correction, expected["fc"][1])
        if correction is not None
        else {}
    )
    affine = manifest.get("affine_weights")
    affine_arrays = load_affine_weights(checkpoint, affine, expected) if affine is not None else {}
    arrays = load_checkpoint(
        checkpoint,
        expected,
        row_scale=row_scale,
        extra_names=set(correction_arrays) | set(affine_arrays),
    )
    selected_names = {base + ".weight" for base in SOURCE_NAMES}
    preserved = {name: tensor for name, tensor in tensors.items() if name not in selected_names}
    expected_names = set(preserved) | {
        base + suffix for base in SOURCE_NAMES for suffix in (".w1a1_packed", ".w1a1_scale")
    }
    expected_names |= set(correction_arrays) | set(affine_arrays)
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
        writer.add_uint32(PREFIX + "scale_group_size", 0 if row_scale else GROUP_SIZE)
        if row_scale:
            writer.add_uint32(PREFIX + "activation_bits", manifest["activation_bits"])
        if manifest.get("activation_quantizers") is not None:
            quantizers = check_activation_quantizers(
                manifest["activation_quantizers"], manifest["activation_bits"]
            )
            writer.add_uint32(QUANTIZER_PREFIX + "version", 1)
            writer.add_array(QUANTIZER_PREFIX + "boundaries", list(QUANTIZER_BOUNDARIES))
            for boundary, params in quantizers["boundaries"].items():
                for name in ("threshold_delta", "clip_ratio"):
                    writer.add_float32(QUANTIZER_PREFIX + boundary + "." + name, params[name])
        if correction is not None:
            writer.add_uint32(CORRECTION_PREFIX + "version", correction["version"])
            writer.add_uint32(CORRECTION_PREFIX + "rank", correction["rank"])
            for key in ("u_name", "v_name", "arithmetic"):
                writer.add_string(CORRECTION_PREFIX + key, correction[key])
            writer.add_key_value(
                CORRECTION_PREFIX + "bias_name", correction["bias_name"] or "", GGUFValueType.STRING
            )
            writer.add_float32(CORRECTION_PREFIX + "bias_bound", correction["bias_bound"] or 0)
            for name, array in correction_arrays.items():
                writer.add_tensor(
                    name, array, raw_dtype=Type.F16 if array.dtype == np.float16 else Type.F32
                )
        if affine is not None:
            writer.add_uint32(AFFINE_PREFIX + "version", 1)
            writer.add_string(AFFINE_PREFIX + "coverage", affine["coverage"])
            writer.add_string(AFFINE_PREFIX + "arithmetic", affine["arithmetic"])
            bases = sorted(affine["tensors"])
            writer.add_array(AFFINE_PREFIX + "bases", bases)
            writer.add_array(
                AFFINE_PREFIX + "midpoint_tensors", [affine["tensors"][base] for base in bases]
            )
            for name, array in affine_arrays.items():
                writer.add_tensor(name, array, raw_dtype=Type.F32)
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
        if row_scale and (
            reread.fields[PREFIX + "scale_group_size"].contents() != 0
            or reread.fields[PREFIX + "activation_bits"].contents() != manifest["activation_bits"]
        ):
            raise ValueError("row activation or scale metadata mismatch")
        if manifest.get("activation_quantizers") is not None:
            for boundary, params in quantizers["boundaries"].items():
                for name in ("threshold_delta", "clip_ratio"):
                    if (
                        reread.fields[QUANTIZER_PREFIX + boundary + "." + name].contents()
                        != params[name]
                    ):
                        raise ValueError("learned activation metadata mismatch")
        for name, array in (correction_arrays | affine_arrays).items():
            observed = actual[name]
            kind = Type.F16 if array.dtype == np.float16 else Type.F32
            if (
                observed.tensor_type != kind
                or observed.data.shape != array.shape
                or raw_hash(observed.data) != raw_hash(array)
            ):
                raise ValueError("fusion correction tensor round-trip mismatch")
        if correction is not None:
            for key, expected_value in correction.items():
                stored = reread.fields[CORRECTION_PREFIX + key].contents()
                if key == "bias_name":
                    expected_value = expected_value or ""
                if key == "bias_bound":
                    expected_value = float(np.float32(expected_value or 0))
                if stored != expected_value:
                    raise ValueError("fusion correction metadata round-trip mismatch")
        if affine is not None:
            for key in ("version", "coverage", "arithmetic"):
                if reread.fields[AFFINE_PREFIX + key].contents() != affine[key]:
                    raise ValueError("affine metadata round-trip mismatch")
            if reread.fields[AFFINE_PREFIX + "bases"].contents() != sorted(affine["tensors"]):
                raise ValueError("affine coverage round-trip mismatch")
        output_hash = sha256(temporary)
        report = {
            "schema_version": 1,
            "base_gguf": {"path": str(base_path), "sha256": base_hash},
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "checkpoint_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
            "output": {"path": str(output), "sha256": output_hash},
            "scale_rule": "f32_learned_nonnegative",
            "training_arithmetic": manifest.get("training_arithmetic", "dense_matmul_hard_quant"),
            "scale_layout": "row" if row_scale else "group128",
            "activation_bits": manifest["activation_bits"] if row_scale else 16,
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
        if manifest.get("activation_quantizers") is not None:
            report["activation_quantizers"] = quantizers
        if correction is not None:
            report["fusion_correction"] = correction
            report["fusion_correction_tensor_sha256"] = {
                name: raw_hash(array) for name, array in correction_arrays.items()
            }
        if affine is not None:
            report["affine_weights"] = affine
            report["affine_midpoint_sha256"] = {
                name: raw_hash(array) for name, array in affine_arrays.items()
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

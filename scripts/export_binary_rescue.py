#!/usr/bin/env python3
"""Compose audited EAGLE mixed GGUFs without reconstructing binary weights.

Donors are standard GGUFs quantized from the original F16 GGUF (for example
with llama-quantize --pure). Selected donor payloads are checked byte-for-byte
against the bundled GGUF reference quantizer applied to that F16 source.
All unaffected tensor bytes and binary scale/sign metadata are preserved.
This is a serialization/precision audit; execution claims require runtime logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/llama.cpp/gguf-py"))
from gguf import GGMLQuantizationType as Type  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402
from gguf.quants import quantize  # noqa: E402

PREFIX = "eagle3.w1a1."
GROUPS = {
    "fusion": ("fc",),
    "attention": ("blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v", "blk.0.attn_output"),
    "ffn": ("blk.0.ffn_gate", "blk.0.ffn_down", "blk.0.ffn_up"),
    "head": ("output",),
}
SUBSETS = {**GROUPS, "ffn_down": ("blk.0.ffn_down",)}
BASES = tuple(base for group in GROUPS.values() for base in group)
DENSE_TYPES = {Type.Q8_0, Type.Q4_0, Type.F16}
GLOBAL_KEYS = (
    "scale_group_size",
    "bit_order",
    "sign_rule",
    "scale_rule",
    "arithmetic",
)


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


@dataclass
class Tensor:
    name: str
    data: np.ndarray
    kind: Type
    shape: tuple[int, ...]  # GGML dimensions, fastest dimension first

    @classmethod
    def read(cls, tensor):
        return cls(tensor.name, tensor.data, tensor.tensor_type, tuple(map(int, tensor.shape)))


def same_tensor(left: Tensor, right: Tensor) -> bool:
    return (
        left.kind == right.kind
        and left.shape == right.shape
        and raw_hash(left.data) == raw_hash(right.data)
    )


class Model:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.reader = GGUFReader(self.path)
        self.fields = self.reader.fields
        self.tensors = {tensor.name: Tensor.read(tensor) for tensor in self.reader.tensors}
        if len(self.tensors) != len(self.reader.tensors):
            raise ValueError("duplicate tensor names")
        if self.value("general.architecture") != "eagle3":
            raise ValueError("expected eagle3 architecture")
        if self.reader.byte_order != "I":
            raise ValueError("only native little-endian GGUFs are supported")
        self.validate_dispatch()

    def validate_dispatch(self) -> None:
        allowed = {b + suffix for b in BASES for suffix in (".w1a1_packed", ".w1a1_scale")}
        actual = {name for name in self.tensors if name.endswith((".w1a1_packed", ".w1a1_scale"))}
        if actual - allowed:
            raise ValueError("packed/scales tensor outside the known EAGLE projections")
        binary = {b + ".weight" for b in BASES if b + ".w1a1_packed" in self.tensors}
        if not binary and PREFIX + "version" not in self.fields:
            return
        version = self.value(PREFIX + "version")
        groups = self.value(PREFIX + "groups")
        declared = self.value(PREFIX + "tensors")
        if len(groups) != len(set(groups)) or any(g not in GROUPS for g in groups):
            raise ValueError("duplicate or unknown dispatch groups")
        if len(declared) != len(set(declared)) or set(declared) != binary:
            raise ValueError("packed tensor declarations differ from actual packed tensors")
        coverage = {b + ".weight" for g in groups for b in GROUPS[g]}
        if version == 2:
            if coverage != binary:
                raise ValueError("v2 groups do not exactly cover packed tensors")
        elif version == 3:
            dense = self.value(PREFIX + "dense_tensors")
            kinds = self.value(PREFIX + "dense_types")
            if (
                len(dense) != len(set(dense))
                or len(dense) != len(kinds)
                or set(dense) & binary
                or set(dense) | binary != coverage
            ):
                raise ValueError("v3 dense/packed declarations do not disjointly cover groups")
            for name, kind in zip(dense, kinds, strict=True):
                if (
                    name not in self.tensors
                    or self.tensors[name].kind.name != kind
                    or self.tensors[name].kind not in DENSE_TYPES
                ):
                    raise ValueError(f"{name}: declared dense type differs from actual tensor")
        else:
            raise ValueError("expected binary v2 or v3 metadata")

    def value(self, key):
        if key not in self.fields:
            raise ValueError(f"missing metadata: {key}")
        return self.fields[key].contents()

    def projection(self, base: str) -> tuple[list[Tensor], tuple[int, int]]:
        dense = self.tensors.get(base + ".weight")
        packed = self.tensors.get(base + ".w1a1_packed")
        scale = self.tensors.get(base + ".w1a1_scale")
        if dense is not None:
            if packed is not None or scale is not None or dense.kind not in DENSE_TYPES:
                raise ValueError(f"{base}: invalid dense type or binary shadow")
            if len(dense.shape) != 2:
                raise ValueError(f"{base}: expected matrix")
            return [dense], dense.shape
        if packed is None or scale is None or packed.kind != Type.I32 or scale.kind != Type.F32:
            raise ValueError(f"{base}: missing or incorrectly typed packed/scales pair")
        if self.value(PREFIX + "version") not in (2, 3):
            raise ValueError("expected binary v2 or v3 metadata")
        width = int(self.value(tensor_key(base, "logical_k")))
        size = int(self.value(PREFIX + "scale_group_size"))
        if width <= 0 or size not in (0, 128) or len(packed.shape) != 2:
            raise ValueError(f"{base}: invalid binary dimensions/group size")
        rows = packed.shape[1]
        expected_scales = (rows,) if size == 0 else ((width + size - 1) // size, rows)
        if packed.shape[0] != (width + 31) // 32 or scale.shape != expected_scales:
            raise ValueError(f"{base}: packed/scales shape mismatch")
        if not np.isfinite(scale.data).all() or (scale.data < 0).any():
            raise ValueError(f"{base}: invalid scales")
        for suffix, name in (("packed", packed.name), ("scale", scale.name)):
            if self.value(tensor_key(base, suffix)) != name:
                raise ValueError(f"{base}: metadata tensor name mismatch")
        return [packed, scale], (width, rows)


def check_source(tensor: Tensor, source: Tensor) -> None:
    """Reject mislabeled/reconstructed donors; check every byte in bounded chunks."""
    if source.kind != Type.F16 or tensor.shape != source.shape:
        raise ValueError(f"{tensor.name}: source must be same-shape original F16")
    for first in range(0, source.data.shape[0], 128):
        original = source.data[first : first + 128]
        if not np.isfinite(original).all():
            raise ValueError(f"{tensor.name}: nonfinite original")
        expected = (
            original
            if tensor.kind == Type.F16
            else quantize(original.astype(np.float32), tensor.kind)
        )
        if raw_hash(tensor.data[first : first + 128]) != raw_hash(expected):
            raise ValueError(
                f"{tensor.name}: donor differs from original F16 reference quantization"
            )


def copy_field(writer: GGUFWriter, key: str, field) -> None:
    writer.add_key_value(
        key, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None
    )


def inventory(model: Model) -> dict:
    projections = []
    projection_names = set()
    for base in BASES:
        tensors, (width, rows) = model.projection(base)
        projection_names.update(t.name for t in tensors)
        binary = len(tensors) == 2
        payload = sum(t.data.nbytes for t in tensors)
        projections.append(
            {
                "name": base + ".weight",
                "shape": [rows, width],
                "parameters": rows * width,
                "storage": "BINARY" if binary else tensors[0].kind.name,
                "payload_bytes": payload,
                "effective_bits": 8 * payload / (rows * width),
                "expected_operand_path": (
                    "binary signs/F32 scales; A16; ordered F32 accumulation"
                    if binary
                    else "standard Q8_1 activation conversion; MMVQ/MMQ workload-dependent"
                    if tensors[0].kind in (Type.Q8_0, Type.Q4_0)
                    else "standard F16 weight matmul; activation/kernel dispatch workload-dependent"
                ),
                "runtime_verified": False,
            }
        )
    selected_params = sum(p["parameters"] for p in projections)
    binary_params = sum(p["parameters"] for p in projections if p["storage"] == "BINARY")
    exceptions = [
        {
            "name": t.name,
            "type": t.kind.name,
            "shape_ggml": list(t.shape),
            "elements": int(np.prod(t.shape)),
            "payload_bytes": t.data.nbytes,
            "raw_sha256": raw_hash(t.data),
        }
        for t in model.tensors.values()
        if t.name not in projection_names
    ]
    float_exceptions = sum(
        e["elements"] for e in exceptions if e["type"] in ("F16", "F32", "BF16", "F64")
    )
    params = selected_params + float_exceptions
    payload = sum(t.data.nbytes for t in model.tensors.values())
    return {
        "projections": projections,
        "nonselected_tensors": exceptions,
        "selected_linear_parameters": selected_params,
        "binary_parameters": binary_params,
        "binary_selected_linear_share": binary_params / selected_params,
        "model_parameters_excluding_integer_maps": params,
        "binary_model_parameter_share": binary_params / params,
        "tensor_payload_bytes_including_scales_and_integer_maps": payload,
        "selected_linear_payload_effective_bits": (
            8 * sum(p["payload_bytes"] for p in projections) / selected_params
        ),
        "model_payload_effective_bits": 8 * payload / params,
        "file_bytes": model.path.stat().st_size,
        "file_effective_bits_including_metadata_padding": 8 * model.path.stat().st_size / params,
        "parameter_denominator": "logical linear weights plus nonselected floating tensors; "
        "scales are storage overhead; integer maps excluded",
    }


def export_model(
    base_path: Path,
    output: Path,
    *,
    donor_path: Path | None = None,
    subsets: tuple[str, ...] = (),
    expected_type: str | None = None,
    source_f16: Path | None = None,
    head_npy: Path | None = None,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    base = Model(base_path)
    if (donor_path is None) == (head_npy is None):
        raise ValueError("choose exactly one donor GGUF or fitted head NPY")
    if head_npy is not None:
        if subsets or expected_type:
            raise ValueError("head-npy implies subset head and type F16")
        selected = {"output"}
        head = np.load(head_npy, allow_pickle=False)
        if head.dtype != np.float16 or head.ndim != 2 or not np.isfinite(head).all():
            raise ValueError("head-npy must be a finite rank-two F16 array")
        replacements = {"output": [Tensor("output.weight", head, Type.F16, head.shape[::-1])]}
        donor = None
    else:
        if (
            not subsets
            or len(set(subsets)) != len(subsets)
            or any(s not in SUBSETS for s in subsets)
        ):
            raise ValueError("choose distinct named subsets")
        selected_list = [name for subset in subsets for name in SUBSETS[subset]]
        if len(selected_list) != len(set(selected_list)):
            raise ValueError("overlapping subsets")
        selected = set(selected_list)
        if expected_type not in ("BINARY", "F16", "Q4_0", "Q8_0") or source_f16 is None:
            raise ValueError("donor requires expected-type and original source-f16")
        donor = Model(donor_path)
        replacements = {name: donor.projection(name)[0] for name in selected}
        for name, tensors in replacements.items():
            kind = "BINARY" if len(tensors) == 2 else tensors[0].kind.name
            if kind != expected_type:
                raise ValueError(f"{name}: expected {expected_type}, got {kind}")
    expected = dict(base.tensors)
    origins = {name: "base" for name in expected}
    owners = {}
    for name in BASES:
        old, shape = base.projection(name)
        if name in selected:
            new = replacements[name]
            new_shape = donor.projection(name)[1] if donor else tuple(head.shape[::-1])
            if shape != new_shape:
                raise ValueError(f"{name}: replacement shape mismatch")
            for tensor in old:
                del expected[tensor.name]
                del origins[tensor.name]
            for tensor in new:
                expected[tensor.name] = tensor
                origins[tensor.name] = "head_npy" if head_npy else "donor"
            owners[name] = donor if len(new) == 2 else None
        else:
            owners[name] = base if len(old) == 2 else None
    packed = [name for name in BASES if owners[name] is not None]
    dense = [name for name in BASES if owners[name] is None]
    if not packed or not dense:
        raise ValueError("v3 export must contain both binary and dense projections")
    binary_source = owners[packed[0]]
    for name in packed:
        for key in GLOBAL_KEYS:
            if owners[name].value(PREFIX + key) != binary_source.value(PREFIX + key):
                raise ValueError("incompatible binary global metadata")
    if (
        binary_source.value(PREFIX + "bit_order") != "little"
        or binary_source.value(PREFIX + "sign_rule") != "nonnegative_is_one"
        or binary_source.value(PREFIX + "arithmetic") != "f32"
    ):
        raise ValueError("unsupported binary arithmetic/sign metadata")
    checked = []
    if source_f16 is not None:
        source = Model(source_f16)
        for name in BASES:
            source_tensors, shape = source.projection(name)
            if len(source_tensors) != 1 or source_tensors[0].kind != Type.F16:
                raise ValueError("original source must contain nine F16 projections")
            if shape != base.projection(name)[1]:
                raise ValueError("source/base shape mismatch")
            if name in dense and not (head_npy is not None and name == "output"):
                check_source(expected[name + ".weight"], source_tensors[0])
                checked.append(name + ".weight")
        # Exact vocabulary/normalization/mapping identity is part of the source gate.
        for name, tensor in source.tensors.items():
            if name not in {b + ".weight" for b in BASES}:
                if name not in base.tensors or not same_tensor(tensor, base.tensors[name]):
                    raise ValueError(f"{name}: source/base nonselected tensor mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rescue-export-", dir=output.parent) as tmp:
        temporary = Path(tmp) / "model.gguf"
        writer = GGUFWriter(temporary, "eagle3")
        for key, field in base.fields.items():
            if not key.startswith(("GGUF.", PREFIX)) and key != "general.architecture":
                copy_field(writer, key, field)
        writer.add_uint32(PREFIX + "version", 3)
        writer.add_array(PREFIX + "groups", list(GROUPS))
        writer.add_array(PREFIX + "tensors", sorted(name + ".weight" for name in packed))
        dense = sorted(dense)
        writer.add_array(PREFIX + "dense_tensors", [name + ".weight" for name in dense])
        writer.add_array(
            PREFIX + "dense_types", [expected[name + ".weight"].kind.name for name in dense]
        )
        for key in GLOBAL_KEYS:
            copy_field(writer, PREFIX + key, binary_source.fields[PREFIX + key])
        for name in packed:
            for suffix in ("logical_k", "packed", "scale"):
                key = tensor_key(name, suffix)
                copy_field(writer, key, owners[name].fields[key])
        for tensor in expected.values():
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.kind)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        actual = Model(temporary)
        if set(actual.tensors) != set(expected):
            raise ValueError("export tensor-set mismatch")
        for name, tensor in expected.items():
            if not same_tensor(tensor, actual.tensors[name]):
                raise ValueError(f"export byte/type/shape mismatch: {name}")
        report = {
            "schema_version": 1,
            "base": {"path": str(base_path), "sha256": sha256(base_path)},
            "replaced_projections": sorted(selected),
            "unchanged_tensors": sorted(
                name for name, origin in origins.items() if origin == "base"
            ),
            "output_tensors": [
                {
                    "name": name,
                    "origin": origins[name],
                    "type": tensor.kind.name,
                    "shape_ggml": list(tensor.shape),
                    "payload_bytes": tensor.data.nbytes,
                    "raw_sha256": raw_hash(tensor.data),
                }
                for name, tensor in expected.items()
            ],
            "reference_quantization_checked_tensors": checked,
            "dispatch_metadata": {
                key: field.contents()
                for key, field in actual.fields.items()
                if key.startswith(PREFIX)
            },
            "inventory": inventory(actual),
            "serialization_audit_passed": True,
            "execution_audit": "not performed; verify CUDA dispatch and graphs separately",
            "output": {"path": str(output), "sha256": sha256(temporary)},
        }
        for label, path in (
            ("donor", donor_path),
            ("source_f16", source_f16),
            ("head_npy", head_npy),
        ):
            if path is not None:
                report[label] = {"path": str(path), "sha256": sha256(path)}
        temporary.rename(output)
    return report


def extract_head(source_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    model = Model(source_path)
    tensors, _ = model.projection("output")
    if len(tensors) != 1 or tensors[0].kind != Type.F16 or not np.isfinite(tensors[0].data).all():
        raise ValueError("head extraction requires finite original F16 output.weight")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.save(stream, tensors[0].data, allow_pickle=False)
    return {
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "tensor": "output.weight",
        "shape": list(tensors[0].data.shape),
        "type": "F16",
        "raw_sha256": raw_hash(tensors[0].data),
        "output": str(output),
        "sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--donor", type=Path)
    parser.add_argument("--source-f16", type=Path)
    parser.add_argument("--subset", help="comma-separated fusion,attention,ffn_down,head")
    parser.add_argument("--expected-type", choices=("Q8_0", "Q4_0", "F16", "BINARY"))
    parser.add_argument("--head-npy", type=Path, help="strict F16 [draft_vocab,H] fitted head")
    parser.add_argument("--extract-head", type=Path, help="extract original F16 GGUF output.weight")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve() == args.output.resolve():
        parser.error("audit/output paths must be distinct and new")
    if args.extract_head:
        if any(
            (args.base, args.donor, args.source_f16, args.subset, args.expected_type, args.head_npy)
        ):
            parser.error("extract-head cannot be combined with composition options")
        report = extract_head(args.extract_head, args.output)
    else:
        if not args.base:
            parser.error("base is required for composition")
        report = export_model(
            args.base,
            args.output,
            donor_path=args.donor,
            subsets=tuple(args.subset.split(",")) if args.subset else (),
            expected_type=args.expected_type,
            source_f16=args.source_f16,
            head_npy=args.head_npy,
        )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "audit": str(args.audit)}))


if __name__ == "__main__":
    main()

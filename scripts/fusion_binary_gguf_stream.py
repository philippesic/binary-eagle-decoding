"""Bounded little-endian dense GGUF v3 I/O for fusion-only binary export.

This module keeps metadata as file ranges: tokenizer arrays are never expanded
into Python strings or NumPy objects. Original KV records and nonfusion tensor
payloads are copied and hashed in 1 MiB chunks. Only F32/F16/I32/I64 source tensors
and BF16 source tensors plus the project's existing binary metadata are
supported; other formats fail.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

CHUNK_BYTES = 1024 * 1024
SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TENSOR_SIZES = {0: 4, 1: 2, 26: 4, 27: 8, 30: 2}  # GGML F32, F16, I32, I64, BF16
PREFIX = "eagle3.w1a1."


def _read(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF field")
    return data


def _number(stream, code):
    return struct.unpack("<" + code, _read(stream, struct.calcsize("<" + code)))[0]


def _skip(stream, size, file_size):
    if size < 0 or stream.tell() + size > file_size:
        raise ValueError("GGUF field exceeds remaining file")
    stream.seek(size, os.SEEK_CUR)


def _string(stream, file_size, *, decode=False):
    size = _number(stream, "Q")
    if size > 16 * CHUNK_BYTES or stream.tell() + size > file_size:
        raise ValueError("GGUF string exceeds bounded size")
    if decode:
        if size > 4096:
            raise ValueError("GGUF key/tensor name exceeds bounded size")
        return _read(stream, size).decode("utf-8")
    _skip(stream, size, file_size)


def _value(stream, kind, file_size):
    if kind in SCALAR_SIZES:
        _skip(stream, SCALAR_SIZES[kind], file_size)
    elif kind == 8:
        _string(stream, file_size)
    elif kind == 9:
        element, count = _number(stream, "I"), _number(stream, "Q")
        if count > 2_000_000:
            raise ValueError("GGUF array exceeds bounded element count")
        if element in SCALAR_SIZES:
            _skip(stream, count * SCALAR_SIZES[element], file_size)
        elif element == 8:
            for _ in range(count):
                _string(stream, file_size)
        else:
            raise ValueError("unsupported nested/unknown GGUF array element type")
    else:
        raise ValueError("unsupported GGUF metadata type")


def parse_gguf(path):
    """Return small KV/tensor range descriptors, without mapping tensor data."""
    path = Path(path)
    size = path.stat().st_size
    with path.open("rb") as stream:
        if _read(stream, 4) != b"GGUF" or _number(stream, "I") != 3:
            raise ValueError("only little-endian GGUF version 3 is supported")
        tensor_count, kv_count = _number(stream, "Q"), _number(stream, "Q")
        if not 1 <= tensor_count <= 4096 or not 1 <= kv_count <= 4096:
            raise ValueError("GGUF tensor/KV count outside bounded format")
        fields, alignment, architecture = {}, 32, None
        for _ in range(kv_count):
            start = stream.tell()
            key = _string(stream, size, decode=True)
            if key in fields:
                raise ValueError("duplicate GGUF metadata key")
            kind, value_start = _number(stream, "I"), stream.tell()
            _value(stream, kind, size)
            end = stream.tell()
            if key == "general.architecture":
                if kind != 8:
                    raise ValueError("GGUF architecture must be a string")
                stream.seek(value_start)
                architecture = _string(stream, size, decode=True)
                stream.seek(end)
            elif key == "general.alignment":
                if kind != 4:
                    raise ValueError("GGUF alignment must be UINT32")
                stream.seek(value_start)
                alignment = _number(stream, "I")
                stream.seek(end)
            fields[key] = {"start": start, "size": end - start, "kind": kind}
        if alignment < 1 or alignment > 4096 or alignment & (alignment - 1):
            raise ValueError("GGUF alignment must be bounded power of two")
        kv_end, tensors = stream.tell(), {}
        for _ in range(tensor_count):
            name = _string(stream, size, decode=True)
            dims_count = _number(stream, "I")
            if not 1 <= dims_count <= 4:
                raise ValueError("unsupported GGUF tensor dimension count")
            dims = tuple(_number(stream, "Q") for _ in range(dims_count))
            kind, offset = _number(stream, "I"), _number(stream, "Q")
            if name in tensors or kind not in TENSOR_SIZES or min(dims) < 1:
                raise ValueError("duplicate/unsupported GGUF dense tensor")
            count = math.prod(dims)
            nbytes = count * TENSOR_SIZES[kind]
            if nbytes > size or offset % alignment:
                raise ValueError("GGUF tensor size or alignment is invalid")
            tensors[name] = {"dims": dims, "kind": kind, "offset": offset, "nbytes": nbytes}
        data_offset = (stream.tell() + alignment - 1) // alignment * alignment
        spans = []
        for tensor in tensors.values():
            tensor["start"] = data_offset + tensor["offset"]
            if tensor["start"] + tensor["nbytes"] > size:
                raise ValueError("GGUF tensor payload exceeds file bounds")
            spans.append((tensor["start"], tensor["start"] + tensor["nbytes"]))
        spans.sort()
        if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
            raise ValueError("GGUF tensor payloads overlap")
    return {
        "path": path,
        "fields": fields,
        "tensors": tensors,
        "alignment": alignment,
        "architecture": architecture,
        "kv_end": kv_end,
        "data_offset": data_offset,
    }


def hash_range(stream, start, size, *, destination=None):
    stream.seek(start)
    digest = hashlib.sha256()
    while size:
        chunk = _read(stream, min(size, CHUNK_BYTES))
        digest.update(chunk)
        if destination is not None:
            destination.write(chunk)
        size -= len(chunk)
    return digest.hexdigest()


def _encoded_string(value):
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def _kv(key, kind, value):
    if kind == 4:
        body = struct.pack("<I", value)
    elif kind == 8:
        body = _encoded_string(value)
    elif kind == 9:
        body = struct.pack("<IQ", 8, len(value)) + b"".join(_encoded_string(v) for v in value)
    else:
        raise ValueError("unsupported new fusion metadata type")
    return _encoded_string(key) + struct.pack("<I", kind) + body


def _fusion_fields(width):
    return [
        _kv(PREFIX + "version", 4, 2),
        _kv(PREFIX + "scale_group_size", 4, 0),
        _kv(PREFIX + "activation_bits", 4, 8),
        _kv(PREFIX + "groups", 9, ["fusion"]),
        _kv(PREFIX + "tensors", 9, ["fc.weight"]),
        _kv(PREFIX + "bit_order", 8, "little"),
        _kv(PREFIX + "sign_rule", 8, "nonnegative_is_one"),
        _kv(PREFIX + "scale_rule", 8, "f32_nonnegative_least_squares"),
        _kv(PREFIX + "arithmetic", 8, "f32"),
        _kv(PREFIX + "tensor.fc_weight.logical_k", 4, width),
        _kv(PREFIX + "tensor.fc_weight.packed", 8, "fc.w1a1_packed"),
        _kv(PREFIX + "tensor.fc_weight.scale", 8, "fc.w1a1_scale"),
    ]


def _padding(stream, alignment):
    stream.write(b"\0" * ((-stream.tell()) % alignment))


def stream_export(base_path, output_path, packed, scales, width, *, checkpoint=None):
    """Write atomically, then verify raw KV records and every actual tensor byte."""
    base_path, output_path = Path(base_path), Path(output_path)
    if output_path.exists():
        raise FileExistsError("candidate export destination already exists")
    source = parse_gguf(base_path)
    if checkpoint:
        checkpoint("export_source_headers_parsed")
    if source["architecture"] != "eagle3" or any(
        name.startswith((PREFIX, "eagle3.fusion_correction.", "eagle3.affine_weights."))
        for name in source["fields"]
    ):
        raise ValueError("base must be original eagle3 without binary/learned metadata")
    tensors = source["tensors"]
    fusion = tensors.get("fc.weight")
    if fusion is None or fusion["kind"] != 1 or fusion["dims"] != (width, len(scales)):
        raise ValueError("fusion source must have original F16 shape")
    if packed.dtype != np.dtype("<i4") or packed.shape != (len(scales), (width + 31) // 32):
        raise ValueError("packed operand has incompatible shape or precision")
    new_fields = _fusion_fields(width)
    preserved = {name: tensor for name, tensor in tensors.items() if name != "fc.weight"}
    output_tensors, offset = {}, 0
    alignment = source["alignment"]
    for name, tensor in preserved.items():
        output_tensors[name] = {**tensor, "offset": offset}
        offset += (tensor["nbytes"] + alignment - 1) // alignment * alignment
    for name, array, kind in (("fc.w1a1_packed", packed, 26), ("fc.w1a1_scale", scales, 0)):
        output_tensors[name] = {
            "dims": tuple(reversed(array.shape)),
            "kind": kind,
            "offset": offset,
            "nbytes": array.nbytes,
        }
        offset += (array.nbytes + alignment - 1) // alignment * alignment
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".fusion-stream-export-", dir=output_path.parent
    ) as tmp:
        temp = Path(tmp) / "candidate.gguf"
        payload_hashes = {}
        with base_path.open("rb") as original, temp.open("wb") as output:
            output.write(
                struct.pack(
                    "<4sIQQ",
                    b"GGUF",
                    3,
                    len(output_tensors),
                    len(source["fields"]) + len(new_fields),
                )
            )
            hash_range(original, 24, source["kv_end"] - 24, destination=output)
            for record in new_fields:
                output.write(record)
            for name, tensor in output_tensors.items():
                output.write(_encoded_string(name) + struct.pack("<I", len(tensor["dims"])))
                output.write(struct.pack("<" + "Q" * len(tensor["dims"]), *tensor["dims"]))
                output.write(struct.pack("<IQ", tensor["kind"], tensor["offset"]))
            _padding(output, alignment)
            data_start = output.tell()
            for name, tensor in preserved.items():
                if output.tell() != data_start + output_tensors[name]["offset"]:
                    raise ValueError("streamed tensor position differs from descriptor")
                payload_hashes[name] = hash_range(
                    original, tensor["start"], tensor["nbytes"], destination=output
                )
                _padding(output, alignment)
            for array in (packed, scales):
                array.tofile(output)
                _padding(output, alignment)
        if checkpoint:
            checkpoint("export_payload_written")
        observed = parse_gguf(temp)
        if (
            set(observed["tensors"]) != set(output_tensors)
            or observed["architecture"] != "eagle3"
            or observed["alignment"] != alignment
            or set(observed["fields"])
            != set(source["fields"])
            | {
                PREFIX + suffix
                for suffix in (
                    "version",
                    "scale_group_size",
                    "activation_bits",
                    "groups",
                    "tensors",
                    "bit_order",
                    "sign_rule",
                    "scale_rule",
                    "arithmetic",
                    "tensor.fc_weight.logical_k",
                    "tensor.fc_weight.packed",
                    "tensor.fc_weight.scale",
                )
            }
        ):
            raise ValueError("stream export inventory mismatch")
        with base_path.open("rb") as original, temp.open("rb") as reread:
            for name, record in source["fields"].items():
                actual = observed["fields"][name]
                if record["size"] != actual["size"] or hash_range(
                    original, record["start"], record["size"]
                ) != hash_range(reread, actual["start"], actual["size"]):
                    raise ValueError("original metadata raw bytes changed")
            for expected, actual in zip(
                new_fields, list(observed["fields"].values())[len(source["fields"]) :], strict=True
            ):
                if (
                    actual["size"] != len(expected)
                    or hash_range(reread, actual["start"], actual["size"])
                    != hashlib.sha256(expected).hexdigest()
                ):
                    raise ValueError("fusion metadata bytes differ")
            for name, tensor in output_tensors.items():
                actual = observed["tensors"][name]
                if actual["kind"] != tensor["kind"] or actual["dims"] != tensor["dims"]:
                    raise ValueError("export tensor representation changed")
                if name in preserved:
                    if (
                        hash_range(reread, actual["start"], actual["nbytes"])
                        != payload_hashes[name]
                    ):
                        raise ValueError("nonfusion operand payload changed")
                else:
                    expected = packed if name.endswith("packed") else scales
                    if (
                        hash_range(reread, actual["start"], actual["nbytes"])
                        != hashlib.sha256(memoryview(expected).cast("B")).hexdigest()
                    ):
                        raise ValueError("binary signs/scales changed on reload")
        if checkpoint:
            checkpoint("export_stream_reload_verified")
        if output_path.exists():
            raise FileExistsError("candidate export destination appeared during serialization")
        os.replace(temp, output_path)
    return {
        "unchanged_nonfusion_tensors": len(preserved),
        "serialization": "streamed_raw_kv_and_dense_payloads",
        "chunk_bytes": CHUNK_BYTES,
    }

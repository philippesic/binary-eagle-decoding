#!/usr/bin/env python3
"""Seal a CUDA block-0 operand capture against the frozen target ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

LADDER_MANIFEST_SHA256 = "72410d35fae0b1561fca0546e8e3b6e58a30506af75fd0802d10da865db6151d"
LADDER_F32_SHA256 = "242a5a748a2a62e7480363c6bd80688563f63b59a37c3ec1bd884782664a2d9e"
TARGET_GGUF_SHA256 = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
PROMPT_ID = "qat-revisit-train-code-data-validation-03"
TOKENS = 29
HIDDEN = 2560
LAYERS = (*range(19), 33)
SELECTED = {
    "attn_norm-0",
    "Qcur_normed-0",
    "Kcur_normed-0",
    "Qcur-0",
    "Kcur-0",
    "Vcur-0",
    "kqv_out-0",
    "ffn_inp-0",
    "ffn_norm-0",
    "ffn_out-0",
    "l_out-0",
}
CAPTURE_MODES = {
    "all": SELECTED,
    "output_only": {"l_out-0"},
    "attn_norm": {"attn_norm-0", "l_out-0"},
    "qkv_normed": {"Qcur_normed-0", "Kcur_normed-0", "Vcur-0", "l_out-0"},
    "q_norm": {"Qcur_normed-0", "l_out-0"},
    "k_norm": {"Kcur_normed-0", "l_out-0"},
    "k_rope": {"Kcur-0", "l_out-0"},
    "q_deferred_k_norm": {"Qcur-0", "Kcur_normed-0", "l_out-0"},
    "v_only": {"Vcur-0", "l_out-0"},
    "attn_output": {"kqv_out-0", "l_out-0"},
    "ffn": {"ffn_inp-0", "ffn_norm-0", "ffn_out-0", "l_out-0"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"invalid JSONL rows: {path.name}")
    return rows


def sealed_ladder(root: Path, target: Path) -> tuple[list[int], np.ndarray, dict]:
    manifest_path = root / "manifest.json"
    if sha256(manifest_path) != LADDER_MANIFEST_SHA256 or sha256(target) != TARGET_GGUF_SHA256:
        raise ValueError("frozen ladder manifest or target GGUF identity differs")
    manifest = json.loads(manifest_path.read_text())
    ledger = manifest.get("files")
    if (
        manifest.get("schema") != "recurrent_cuda_native_diagnostic_v1"
        or manifest.get("execution_device") != "cuda"
        or manifest.get("training_eligible") is not False
        or manifest.get("prompt_id") != PROMPT_ID
        or manifest.get("source_sha256", {}).get("target") != TARGET_GGUF_SHA256
        or not isinstance(ledger, dict)
    ):
        raise ValueError("frozen ladder metadata differs")
    for name, entry in ledger.items():
        path = root / name
        if (
            Path(name).name != name
            or not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"frozen ladder file differs: {name}")
    rounds = _jsonl(root / "forced-rounds.jsonl")
    prefix = rounds[0].get("prefix_token_ids")
    if (
        not isinstance(prefix, list)
        or len(prefix) != TOKENS
        or any(type(token) is not int or token < 0 for token in prefix)
    ):
        raise ValueError("frozen ladder lacks one 29-token first prefix")
    meta = _jsonl(root / "heads.target_layer_ladder.jsonl")
    if len(meta) != TOKENS:
        raise ValueError("frozen ladder lacks all prefill rows")
    for index, row in enumerate(meta):
        if (
            row.get("row") != index
            or row.get("position") != index
            or row.get("token_id") != prefix[index]
            or row.get("layer_ids") != list(LAYERS)
            or row.get("hidden") != HIDDEN
            or row.get("boundary") != "raw_target_layer_input"
        ):
            raise ValueError(f"frozen ladder row {index} differs from prefill ancestry")
    data_path = root / "heads.target_layer_ladder.f32"
    if sha256(data_path) != LADDER_F32_SHA256:
        raise ValueError("frozen ladder F32 payload differs")
    values = np.memmap(data_path, dtype="<f4", mode="r").reshape(TOKENS, len(LAYERS), HIDDEN)
    if not np.isfinite(values).all():
        raise ValueError("frozen ladder contains nonfinite F32 values")
    return prefix, values, manifest


def prepare(root: Path, target: Path, output: Path) -> dict:
    prefix, _, _ = sealed_ladder(root, target)
    output.mkdir(parents=True, exist_ok=False)
    tokens_path = output / "tokens.i32"
    np.asarray(prefix, dtype="<i4").tofile(tokens_path)
    report = {
        "schema": "target_block0_prepare_v1",
        "prompt_id": PROMPT_ID,
        "tokens": TOKENS,
        "prefix_token_ids": prefix,
        "source_sha256": {
            "ladder_manifest": LADDER_MANIFEST_SHA256,
            "ladder_f32": LADDER_F32_SHA256,
            "target_gguf": TARGET_GGUF_SHA256,
            "tokens_i32": sha256(tokens_path),
            "preparer": sha256(Path(__file__)),
        },
    }
    (output / "prepare.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def _index(root: Path, *, mode: str = "all") -> tuple[dict[str, list[dict]], dict]:
    if mode not in CAPTURE_MODES:
        raise ValueError("unknown block-0 capture mode")
    entries: dict[str, list[dict]] = {}
    file_hashes = {}
    rows = (root / "index.tsv").read_text().splitlines()
    if not rows or len(rows) > 20:
        raise ValueError("block-0 operator index has wrong row count")
    total_bytes = 0
    for line in rows:
        parts = line.split("\t")
        if len(parts) != 12:
            raise ValueError("block-0 operator index has wrong field count")
        name, ordinal_text, size_text, filename = parts[:4]
        if name not in SELECTED or Path(filename).name != filename:
            raise ValueError("block-0 operator index has an unexpected tensor")
        ordinal, size = int(ordinal_text), int(size_text)
        ne = tuple(int(value) for value in parts[4:8])
        nb = tuple(int(value) for value in parts[8:12])
        file = root / filename
        if (
            not 0 <= ordinal < 3
            or size <= 0
            or size > 16 * 1024 * 1024
            or not file.is_file()
            or file.stat().st_size != size
            or np.prod(ne)
            != TOKENS
            * (
                4096
                if name.startswith("Qcur") or name == "kqv_out-0"
                else 1024
                if name.startswith(("Kcur", "Vcur"))
                else HIDDEN
            )
        ):
            raise ValueError(f"block-0 tensor {name} has invalid geometry or bytes")
        entry = {
            "name": name,
            "ordinal": ordinal,
            "file": filename,
            "bytes": size,
            "ne": ne,
            "nb": nb,
        }
        entries.setdefault(name, []).append(entry)
        file_hashes[filename] = sha256(file)
        total_bytes += size
    expected = CAPTURE_MODES[mode]
    if set(entries) != expected or total_bytes > 16 * 1024 * 1024:
        raise ValueError("block-0 tensor capture is incomplete or exceeds cap")
    return entries, {"index.tsv": sha256(root / "index.tsv"), **file_hashes}


def _tensor(root: Path, entry: dict) -> np.ndarray:
    raw = (root / entry["file"]).read_bytes()
    shape = tuple(reversed(entry["ne"]))
    strides = tuple(reversed(entry["nb"]))
    tensor = np.ndarray(shape=shape, dtype="<f4", buffer=raw, strides=strides)
    return np.array(tensor, copy=True)


def audit(root: Path, target: Path, capture: Path, helper: Path, *, mode: str = "all") -> dict:
    prefix, ladder, _ = sealed_ladder(root, target)
    prepared = json.loads((capture / "prepare.json").read_text())
    if (
        prepared.get("schema") != "target_block0_prepare_v1"
        or prepared.get("prefix_token_ids") != prefix
        or prepared["source_sha256"]["tokens_i32"] != sha256(capture / "tokens.i32")
    ):
        raise ValueError("block-0 probe does not use the sealed prefix")
    entries, hashes = _index(capture, mode=mode)
    output_rows = entries["l_out-0"]
    if len(output_rows) != 1:
        raise ValueError("block-0 output is missing or repeated")
    output = _tensor(capture, output_rows[0]).reshape(TOKENS, HIDDEN)
    expected = np.asarray(ladder[:, 1, :])
    delta = output.astype(np.float64) - expected.astype(np.float64)
    exact = int(np.count_nonzero(output.view("<u4") == expected.view("<u4")))
    return {
        "schema": "target_block0_native_cuda_capture_v1",
        "status": "same_native_block_output" if exact == output.size else "block_output_differs",
        "hardware": {"machine": platform.machine(), "precision": "CUDA target F16 GGUF"},
        "prompt_id": PROMPT_ID,
        "prefill_tokens": TOKENS,
        "capture_mode": mode,
        "captured_tensors": {name: len(values) for name, values in entries.items()},
        "block_output": {
            "elements": output.size,
            "exact_elements": exact,
            "max_abs": float(np.max(np.abs(delta))),
            "rms": float(np.sqrt(np.mean(delta**2))),
        },
        "source_sha256": {
            "ladder_manifest": LADDER_MANIFEST_SHA256,
            "ladder_f32": LADDER_F32_SHA256,
            "target_gguf": TARGET_GGUF_SHA256,
            "helper_binary": sha256(helper),
            "helper_source": sha256(Path(__file__).with_name("native_target_block0_capture.cpp")),
            "preparer": sha256(Path(__file__)),
            **hashes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "audit"))
    parser.add_argument("--ladder-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--helper", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--capture-mode", choices=tuple(CAPTURE_MODES), default="all")
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.helper is not None or args.report is not None:
            parser.error("prepare does not accept helper or report")
        result = prepare(args.ladder_dir, args.target_gguf, args.capture_dir)
    else:
        if args.helper is None or args.report is None or args.report.exists():
            parser.error("audit requires helper and a new report path")
        result = audit(
            args.ladder_dir,
            args.target_gguf,
            args.capture_dir,
            args.helper,
            mode=args.capture_mode,
        )
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("schema", "status") if key in result}))


if __name__ == "__main__":
    main()

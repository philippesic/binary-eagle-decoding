"""Prepare/check FFN-only Q4_0 DSpark/DFlash without changing scaffolding.

The quantizer is the existing published runtime. This script does not load a
model for inference, set GPU flags, alter source GGUFs, or implement W1Ax.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

from check_export import sha256

FFN_PATTERN = r"^blk\.[0-4]\.ffn_(gate|up|down)\.weight$"
EXPECTED = {f"blk.{layer}.ffn_{projection}.weight" for layer in range(5)
            for projection in ("gate", "up", "down")}


def command(quantizer: Path, source: Path, output: Path, threads: int = 4):
    if source.resolve() == output.resolve() or threads < 1:
        raise ValueError("requires separate output and positive CPU thread count")
    # The first matching regex wins. Default must be quantized: COPY/BF16
    # bypass the native manual-override branch and cannot implement this policy.
    return [str(quantizer), "--pure", "--leave-output-tensor", "--token-embedding-type", "BF16",
            "--tensor-type", FFN_PATTERN + "=Q4_0", "--tensor-type", r"^.*$=BF16",
            str(source), str(output), "Q4_0", str(threads)]


def policy_type(name, original="BF16"):
    return "Q4_0" if re.fullmatch(FFN_PATTERN, name) else original


def native_extents(shape):
    if not 1 <= len(shape) <= 4 or any(int(value) < 1 for value in shape):
        raise ValueError("invalid native tensor extents")
    return list(shape) + [1] * (4 - len(shape))


def shape_preserved(source, candidate):
    collapsed = list(source)
    while len(collapsed) > 1 and collapsed[-1] == 1:
        collapsed.pop()
    return candidate in (source, collapsed) and native_extents(source) == native_extents(candidate)


def raw_hash(array):
    h = hashlib.sha256()
    raw = array.reshape(-1).view("uint8")
    for start in range(0, len(raw), 8 * 1024 * 1024):
        h.update(raw[start:start + 8 * 1024 * 1024].tobytes())
    return h.hexdigest()


def inspect(source: Path, output: Path, llama: Path, export_report: Path | None = None):
    sys.path.insert(0, str(llama / "gguf-py"))
    from gguf import GGUFReader
    if source.resolve() == output.resolve():
        raise ValueError("source GGUF must not be the candidate")
    before, after = GGUFReader(str(source)), GGUFReader(str(output))
    left, right = {t.name: t for t in before.tensors}, {t.name: t for t in after.tensors}
    if left.keys() != right.keys() or not EXPECTED <= left.keys():
        raise ValueError("tensor names or fifteen-matrix FFN coverage changed")
    if before.get_field("general.architecture").contents() != "dflash" or after.get_field("general.architecture").contents() != "dflash":
        raise ValueError("not a DSpark/DFlash graph")
    # Quantization changes its two bookkeeping fields; all actual layout,
    # target-tap, tokenizer, sampler and ownership metadata must remain exact.
    allowed = {"general.file_type", "general.quantization_version"}
    for name in before.fields.keys() | after.fields.keys():
        if name.startswith("GGUF.") or name in allowed:
            continue
        l, r = before.get_field(name), after.get_field(name)
        if l is None or r is None or l.contents() != r.contents():
            raise ValueError(f"protected metadata changed: {name}")
    selected, preserved = [], []
    for name, tensor in left.items():
        candidate = right[name]
        source_shape, candidate_shape = tensor.shape.tolist(), candidate.shape.tolist()
        if not shape_preserved(source_shape, candidate_shape):
            raise ValueError(f"tensor shape changed: {name}")
        if name in EXPECTED:
            if tensor.tensor_type.name != "BF16" or candidate.tensor_type.name != "Q4_0":
                raise ValueError(f"missing exact BF16->Q4_0 coverage: {name}")
            selected.append({"name": name, "shape": tensor.shape.tolist(),
                             "source_bytes": tensor.n_bytes, "candidate_bytes": candidate.n_bytes,
                             "type": "Q4_0"})
        else:
            source_raw, candidate_raw = raw_hash(tensor.data), raw_hash(candidate.data)
            if tensor.tensor_type != candidate.tensor_type or tensor.n_bytes != candidate.n_bytes or source_raw != candidate_raw:
                raise ValueError(f"non-FFN source bytes/type changed: {name}")
            preserved.append({"name": name, "type": tensor.tensor_type.name,
                              "bytes": tensor.n_bytes, "raw_sha256": source_raw,
                              "source_shape": source_shape, "candidate_shape": candidate_shape,
                              "native_extents": native_extents(source_shape),
                              "trailing_singleton_collapsed": source_shape != candidate_shape})
    source_sha = sha256(source)
    export = json.loads(export_report.read_text()) if export_report else None
    actual_source_bound = bool(export and export.get("passed") and export.get("draft_sha256") == source_sha)
    if export_report and not actual_source_bound:
        raise ValueError("reference export receipt is not bound to this source GGUF")
    if actual_source_bound:
        for row in selected:
            shape = [9728, 2560] if "ffn_down" in row["name"] else [2560, 9728]
            if row["shape"] != shape:
                raise ValueError("released Qwen3-4B FFN dimensions changed")
    return {"schema": "dspark_precision_q4_ffn_v1", "passed": actual_source_bound,
            "source_export_bound": actual_source_bound,
            "source_sha256": source_sha, "candidate_sha256": sha256(output),
            "source_gguf_sha256": source_sha,
            "original_checkpoint_sha256": export.get("source_sha256") if export else None,
            "source_export_sha256": sha256(export_report) if export_report else None,
            "coverage": "five-layer FFN up/gate/down only;15 matrices",
            "quantization": "Q4_0 weight-only; activation/kernel precision requires actual dispatch evidence",
            "selected": sorted(selected, key=lambda row: row["name"]), "preserved": preserved,
            "target_sha256": export.get("target_sha256") if export else None,
            "borrows_embedding": export.get("borrows_embedding") if export else None,
            "borrows_head": export.get("borrows_head") if export else None,
            "target_tied_head_fallback": export.get("target_tied_head_fallback") if export else None,
            "reference_export": str(export_report) if export_report else None,
            "non_ffn_immutable": True,
            "remaining": "fresh actual SM75 model/native trajectory/dispatch admission, then fixed same-device throughput"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("command")
    for arg in ("quantizer", "source", "output", "receipt"):
        prepare.add_argument("--" + arg, type=Path, required=True)
    prepare.add_argument("--threads", type=int, default=4)
    check = modes.add_parser("check")
    for arg in ("source", "output", "llama", "source-export", "receipt"):
        check.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "command":
        result = {"command": command(args.quantizer, args.source, args.output, args.threads),
                  "source_sha256": sha256(args.source), "quantizer_sha256": sha256(args.quantizer),
                  "coverage": "FFN-only15 matrices; preserve all other source bytes/types",
                  "scope": "CPU quantization plan; not a launched run or admission"}
    else:
        result = inspect(args.source, args.output, args.llama, args.source_export)
    args.receipt.write_text(json.dumps(result, indent=2) + "\n")

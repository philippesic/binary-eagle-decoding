#!/usr/bin/env python3
"""Select a deterministic stratified subset of real W1Ax activation captures.

The input files use the ``W1AXACT1`` format emitted by llama.cpp's CUDA
operator. Selected files are symlinked into a new ignored run directory; the
source captures are read only. A JSON manifest records the full invocation
histogram and selected file hashes.

Example::

    python3 scripts/select_w1ax_captures.py \
      --capture-dir runs/w1ax-activation-capture-20260925/activations \
      --output-dir runs/w1ax-operator-replay-20260925/selected
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

MAGIC = b"W1AXACT1"
HEADER = struct.Struct("<8sQQQQI128s")
FILENAME = re.compile(r"op-(\d{12})\.bin\Z")
REQUIRED_TENSORS = frozenset(
    {
        "fc.w1a1_packed",
        "blk.0.attn_q.w1a1_packed",
        "blk.0.attn_k.w1a1_packed",
        "blk.0.attn_v.w1a1_packed",
        "blk.0.attn_output.w1a1_packed",
        "blk.0.ffn_gate.w1a1_packed",
        "blk.0.ffn_down.w1a1_packed",
        "blk.0.ffn_up.w1a1_packed",
        "output.w1a1_packed",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_capture(path: Path) -> dict[str, Any]:
    """Parse and validate one capture header and its payload length."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"capture must be a regular, non-symlink file: {path}")
    match = FILENAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"invalid capture filename: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
    if len(raw) != HEADER.size:
        raise ValueError(f"truncated capture header: {path}")
    magic, sequence, k, m, n, bits, raw_name = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"invalid capture magic in {path}")
    if sequence != int(match.group(1)):
        raise ValueError(f"sequence does not match filename: {path}")
    if min(k, m, n) <= 0:
        raise ValueError(f"capture dimensions must be positive: {path}")
    if bits not in (1, 4, 8, 16):
        raise ValueError(f"unsupported activation precision {bits}: {path}")
    name_bytes = raw_name.split(b"\0", 1)[0]
    if b"\0" in raw_name and any(raw_name[len(name_bytes) :]):
        raise ValueError(f"tensor name is not NUL padded in {path}")
    try:
        name = name_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid tensor name encoding in {path}") from error
    if not name or b"\0" in name_bytes:
        raise ValueError(f"empty or invalid tensor name in {path}")
    expected_size = HEADER.size + n * k * 4
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"invalid capture payload size in {path}: expected {expected_size}, got {actual_size}"
        )
    return {
        "path": path,
        "filename": path.name,
        "sequence": sequence,
        "name": name,
        "K": k,
        "M": m,
        "N": n,
        "bits": bits,
        "bytes": actual_size,
    }


def stratified_indices(count: int, limit: int = 3) -> list[int]:
    """Return ordered bucket-center indices, up to ``limit`` distinct rows."""
    if count < 0 or limit <= 0:
        raise ValueError("count must be nonnegative and limit must be positive")
    selected_count = min(count, limit)
    return [((2 * bucket + 1) * count) // (2 * selected_count) for bucket in range(selected_count)]


def scan_captures(capture_dir: Path) -> list[dict[str, Any]]:
    if not capture_dir.is_dir() or capture_dir.is_symlink():
        raise ValueError(f"capture directory must be a real directory: {capture_dir}")
    paths = sorted(capture_dir.iterdir(), key=lambda item: item.name)
    candidate_paths = [path for path in paths if path.name.startswith("op-") and path.suffix == ".bin"]
    if not candidate_paths:
        raise ValueError(f"no op-*.bin captures found in {capture_dir}")
    captures = [parse_capture(path) for path in candidate_paths]
    names = {capture["name"] for capture in captures}
    missing = sorted(REQUIRED_TENSORS - names)
    unexpected = sorted(names - REQUIRED_TENSORS)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing tensor names: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected tensor names: " + ", ".join(unexpected))
        raise ValueError("; ".join(details))
    return captures


def build_manifest(captures: list[dict[str, Any]], capture_dir: Path) -> dict[str, Any]:
    histogram: Counter[tuple[str, int, int, int, int]] = Counter()
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for capture in captures:
        histogram[(capture["name"], capture["K"], capture["M"], capture["N"], capture["bits"])] += 1
        grouped[(capture["name"], capture["N"])].append(capture)

    histogram_rows = [
        {"name": name, "K": k, "M": m, "N": n, "bits": bits, "count": count}
        for (name, k, m, n, bits), count in sorted(histogram.items())
    ]
    selected = []
    for (name, n), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row["sequence"], row["filename"]))
        for index in stratified_indices(len(rows), limit=3):
            row = rows[index]
            selected.append(
                {
                    "name": name,
                    "N": n,
                    "filename": row["filename"],
                    "sequence": row["sequence"],
                    "K": row["K"],
                    "M": row["M"],
                    "bits": row["bits"],
                    "bytes": row["bytes"],
                    "sha256": sha256(row["path"]),
                }
            )
    return {
        "format": "W1AXACT1",
        "capture_dir": str(capture_dir.resolve()),
        "capture_count": len(captures),
        "required_tensor_names": sorted(REQUIRED_TENSORS),
        "observed_tensor_names": sorted({capture["name"] for capture in captures}),
        "invocation_histogram": histogram_rows,
        "selection": "up to 3 bucket-center sequences per (name, N), sorted by sequence",
        "selected_count": len(selected),
        "selected": selected,
    }


def materialize(captures: list[dict[str, Any]], manifest: dict[str, Any], output_dir: Path) -> Path:
    source_dir = captures[0]["path"].parent.resolve()
    destination = output_dir.expanduser().absolute()
    destination_resolved = destination.resolve(strict=False)
    if destination_resolved == source_dir or source_dir in destination_resolved.parents:
        raise ValueError("output directory must not be inside the raw capture directory")
    if destination_resolved in source_dir.parents:
        raise ValueError("output directory must not contain the raw capture directory")
    if destination.is_symlink():
        raise ValueError(f"output directory must not be a symlink: {destination}")
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError(f"output directory must be new or empty: {destination}")
    else:
        destination.mkdir(parents=True)

    by_name = {capture["filename"]: capture["path"] for capture in captures}
    links: list[Path] = []
    try:
        for item in manifest["selected"]:
            source = by_name[item["filename"]]
            link = destination / item["filename"]
            link.symlink_to(source.resolve())
            links.append(link)
        manifest_path = destination / "manifest.json"
        manifest["output_dir"] = str(destination.resolve())
        manifest["selected_files"] = [item["filename"] for item in manifest["selected"]]
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return manifest_path
    except BaseException:
        for link in links:
            link.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    captures = scan_captures(args.capture_dir.expanduser().resolve())
    manifest = build_manifest(captures, args.capture_dir.expanduser().resolve())
    manifest_path = materialize(captures, manifest, args.output_dir)
    print(
        json.dumps(
            {
                "capture_count": manifest["capture_count"],
                "selected_count": manifest["selected_count"],
                "histogram_rows": len(manifest["invocation_histogram"]),
                "manifest": str(manifest_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

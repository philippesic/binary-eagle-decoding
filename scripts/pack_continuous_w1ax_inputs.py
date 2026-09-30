#!/usr/bin/env python3
"""Deterministic CPU-only launch-input tar; never open/include sealed prompt payloads.

Extract the trusted archive into the remote project root using the user's manual
transfer procedure. No network, SSH, accelerator, capture or training operations.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from prepare_continuous_w1ax_data import opaque_exclusions, require_untracked_destination
from prepare_w1a_data import canonical, sha256

ROOT = Path(__file__).resolve().parents[1]


def safe_name(value: object) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError("manifest shard path must be a single filename")
    if value in {".", ".."} or "\\" in value or "/" in value:
        raise ValueError("unsafe manifest filename")
    return value


def verify_file(path: Path, digest: str) -> None:
    if path.is_symlink():
        raise ValueError("symlink inputs are forbidden")
    if not isinstance(digest, str) or len(digest) != 64 or sha256(path) != digest:
        raise ValueError(f"input hash mismatch: {path.name}")


def add_entry(archive: tarfile.TarFile, name: str, payload: bytes) -> dict:
    parts = PurePosixPath(name).parts
    if name.startswith("/") or ".." in parts:
        raise ValueError("unsafe archive member path")
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(payload))
    return {"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def pack(
    manifest: Path,
    source_lock: Path,
    output: Path,
    exclusion_indexes: list[Path],
    include_reserve: bool = False,
) -> dict:
    if output.exists() or output.with_suffix(output.suffix + ".json").exists():
        raise ValueError("packet output already exists; choose a new ignored file")
    require_untracked_destination(output)
    report = json.loads(manifest.read_text())
    if report.get("schema") != "continuous_w1ax_prompt_manifest_v1":
        raise ValueError("prompt manifest schema mismatch")
    verify_file(source_lock, report["source_lock_sha256"])
    # This also rejects indexes containing messages/text. Read only opaque metadata.
    _, _, _, _, exclusions = opaque_exclusions(exclusion_indexes)
    actual = {(r["file"], r["sha256"], r["rows"]) for r in exclusions}
    expected = {(r["file"], r["sha256"], r["rows"]) for r in report["opaque_exclusions"]}
    if actual != expected:
        raise ValueError("opaque exclusion indexes must match original manifest exactly")
    corpus_name = safe_name(manifest.parent.name)
    corpus_prefix = f"data/continuous-w1ax/{corpus_name}"
    entries = {
        f"{corpus_prefix}/manifest.json": (manifest, sha256(manifest)),
        "configs/continuous_w1ax_sources.json": (source_lock, report["source_lock_sha256"]),
    }
    allowed = {"train", "dev"} | ({"train_reserve"} if include_reserve else set())
    counts = {}
    for split, details in report["files"].items():
        if split not in {"train", "dev", "train_reserve", "sealed_test"}:
            raise ValueError("unknown manifest split")
        count = 0
        for shard in details["shards"]:
            index_name = safe_name(shard["index"])
            if not index_name.endswith(".index.jsonl"):
                raise ValueError("manifest index must be opaque .index.jsonl")
            index_path = manifest.parent / index_name
            verify_file(index_path, shard["index_sha256"])
            # Reject text-bearing indexes before adding any of them to the archive.
            opaque_exclusions([index_path])
            entries[f"{corpus_prefix}/{index_name}"] = (index_path, shard["index_sha256"])
            if split in allowed:
                prompt_name = safe_name(shard["prompts"])
                # Fail closed if a mislabeled train/dev shard points at sealed/test text.
                if not prompt_name.startswith(split + "-") or not prompt_name.endswith(".jsonl"):
                    raise ValueError("prompt filename does not match allowed split")
                prompt_path = manifest.parent / prompt_name
                verify_file(prompt_path, shard["prompts_sha256"])
                entries[f"{corpus_prefix}/{prompt_name}"] = (prompt_path, shard["prompts_sha256"])
                count += shard["prompts_count"]
        if split in allowed:
            if count != details["prompts_count"]:
                raise ValueError("manifest prompt count mismatch")
            counts[split] = count
    for path in exclusion_indexes:
        name = safe_name(path.name)
        digest = sha256(path)
        verify_file(path, digest)
        entries[f"data/continuous-w1ax/exclusions/{digest}/{name}"] = (path, digest)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=".launch-inputs-", dir=output.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        members = []
        with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
            for name, (path, expected_hash) in sorted(entries.items()):
                # Each file remains verified as it is packed; no stale verify/read gap.
                payload = path.read_bytes()
                digest = hashlib.sha256(payload).hexdigest()
                if digest != expected_hash:
                    raise ValueError("input changed during packaging")
                members.append(add_entry(archive, name, payload))
            packet_manifest = {
                "schema": "continuous_w1ax_launch_packet_v1",
                "prompt_manifest_sha256": sha256(manifest),
                "source_lock_sha256": sha256(source_lock),
                "included_prompt_splits": sorted(allowed),
                "prompt_counts": counts,
                "members": members,
                "sealed_prompt_payloads": "excluded;never opened",
                "teacher_readiness": "PROMPTS_ONLY;user-start native capture/audit required",
            }
            add_entry(
                archive,
                "data/continuous-w1ax/launch-packet.json",
                canonical(packet_manifest) + b"\n",
            )
        digest, size = sha256(temporary), temporary.stat().st_size
        result = {
            "schema": "continuous_w1ax_launch_packet_receipt_v1",
            "sha256": digest,
            "bytes": size,
            "prompt_counts": counts,
            "sealed_prompt_files_opened": 0,
            "member_count": len(members) + 1,
            "included_prompt_splits": sorted(allowed),
        }
        os.rename(temporary, output)
        output.with_suffix(output.suffix + ".json").write_bytes(canonical(result) + b"\n")
        return result
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--source-lock", type=Path, default=ROOT / "configs/continuous_w1ax_sources.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclusion-index", type=Path, action="append", default=[])
    parser.add_argument("--include-reserve", action="store_true")
    args = parser.parse_args()
    result = pack(
        args.manifest, args.source_lock, args.output, args.exclusion_index, args.include_reserve
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

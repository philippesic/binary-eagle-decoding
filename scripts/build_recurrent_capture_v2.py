#!/usr/bin/env python3
"""Build hard-CE v2 directly from preparers without duplicating raw logit bytes.

A transient v1 shadow uses a same-filesystem hardlink to the retained native
raw stream for the original preparation audit. Only the new v2 bundle persists;
the native raw stream and all original artifacts remain unchanged. Cross-device
hardlink failure aborts; there is no silent full-logit copy fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from build_recurrent_capture_bundle import build_bundle
from convert_recurrent_capture_v2 import convert_capture_v2


def build_capture_v2(
    rows_dir: Path,
    features_dir: Path,
    target_logits: Path,
    capture_root: Path,
    train_prompts: Path,
    output: Path,
    *,
    expected_prompt_sha256: str,
    expected_prompt_count: int,
    expected_target_sha256: str,
    expected_draft_sha256: str,
    expected_map_raw_sha256: str,
    shard_manifest_path: Path | None = None,
    continuity_report: Path | None = None,
) -> dict:
    output = Path(output)
    target_logits = Path(target_logits)
    original_stat = target_logits.stat()
    if output.exists():
        raise ValueError("v2 output must be new")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after the original inode/link-count invariant survives
    # cleanup. All staging trees are owned by this invocation.
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-publish-", dir=output.parent
    ) as pending:
        staged_output = Path(pending) / "v2-bundle"
        # Put the shadow on the raw file's filesystem. No full-vocabulary
        # logits are copied into the output or a v1 prerequisite.
        with tempfile.TemporaryDirectory(
            prefix=".v2-source-audit-", dir=target_logits.parent
        ) as scratch:
            source = build_bundle(
                rows_dir,
                features_dir,
                target_logits,
                Path(capture_root) / "d_d" / "manifest.json",
                train_prompts,
                Path(scratch) / "v1-shadow",
                continuity_report=continuity_report,
                shard_manifest_path=shard_manifest_path,
                expected_prompt_hash=expected_prompt_sha256,
                expected_prompt_count=expected_prompt_count,
                expected_target_hash=expected_target_sha256,
                expected_draft_hash=expected_draft_sha256,
                expected_map_raw_hash=expected_map_raw_sha256,
                raw_logits_storage="hardlink",
            )
            result = convert_capture_v2(
                source["manifest"],
                capture_root,
                staged_output,
                expected_prompt_sha256=expected_prompt_sha256,
                expected_prompt_count=expected_prompt_count,
            )
        final_stat = target_logits.stat()
        if (
            original_stat.st_dev,
            original_stat.st_ino,
            original_stat.st_size,
            original_stat.st_nlink,
            original_stat.st_mtime_ns,
        ) != (
            final_stat.st_dev,
            final_stat.st_ino,
            final_stat.st_size,
            final_stat.st_nlink,
            final_stat.st_mtime_ns,
        ):
            raise ValueError(
                "v2 transient cleanup changed the original raw inode/size/link count/mtime"
            )
        os.rename(staged_output, output)
        result["manifest"] = str(output / "manifest.json")
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "rows-dir",
        "features-dir",
        "target-logits",
        "capture-root",
        "train-prompts",
        "output",
    ):
        p.add_argument(f"--{name}", type=Path, required=True)
    for name in ("prompt", "target", "draft", "map-raw"):
        p.add_argument(f"--expected-{name}-sha256", required=True)
    p.add_argument("--expected-prompt-count", type=int, required=True)
    p.add_argument("--shard-manifest", type=Path)
    p.add_argument("--continuity-report", type=Path)
    a = p.parse_args()
    result = build_capture_v2(
        a.rows_dir,
        a.features_dir,
        a.target_logits,
        a.capture_root,
        a.train_prompts,
        a.output,
        expected_prompt_sha256=a.expected_prompt_sha256,
        expected_prompt_count=a.expected_prompt_count,
        expected_target_sha256=a.expected_target_sha256,
        expected_draft_sha256=a.expected_draft_sha256,
        expected_map_raw_sha256=a.expected_map_raw_sha256,
        shard_manifest_path=a.shard_manifest,
        continuity_report=a.continuity_report,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "audit"}))


if __name__ == "__main__":
    main()

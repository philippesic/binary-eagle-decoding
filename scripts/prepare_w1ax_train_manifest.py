#!/usr/bin/env python3
"""Pin an existing native bundle for the lazy joint W1Ax provider.

This records hashes and copies the bundle's eligibility status. It never
promotes a preparation-only capture or opens model-weight files. Model and GGUF
bytes are checked by the provider only when `load_models()` is called later.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1ax_capture_provider import (  # noqa: E402
    ANGELSLIM_REVISION,
    CANDIDATE_D_SHA256,
    SCHEMA,
    TARGET_GGUF_SHA256,
    _hash,
    sha256,
)


def prepare(
    capture_manifest: Path,
    prompts: Path,
    absolute_d2t: Path,
    target_gguf: Path,
    candidate_d_gguf: Path,
    base_draft_gguf: Path,
    base_draft_gguf_sha256: str,
    model_snapshot_manifest: Path,
    capture_id: str,
    output: Path,
    *,
    teacher_dir: Path | None = None,
) -> dict:
    """Create the exact provider v1 manifest without changing eligibility."""
    if output.exists():
        raise FileExistsError("provider manifest must be a new path")
    base_hash = _hash(base_draft_gguf_sha256, "base draft GGUF hash")
    if not capture_id:
        raise ValueError("named native capture ID is required")
    paths = {
        "capture_manifest": Path(capture_manifest).resolve(),
        "prompts": Path(prompts).resolve(),
        "target_gguf": Path(target_gguf).resolve(),
        "candidate_d_gguf": Path(candidate_d_gguf).resolve(),
        "base_draft_gguf": Path(base_draft_gguf).resolve(),
        "absolute_d2t": Path(absolute_d2t).resolve(),
        "model_snapshot_manifest": Path(model_snapshot_manifest).resolve(),
    }
    capture = json.loads(paths["capture_manifest"].read_text())
    snapshot = json.loads(paths["model_snapshot_manifest"].read_text())
    if capture.get("schema") != "recurrent_binary_capture_v1":
        raise ValueError("unsupported native capture schema")
    if capture.get("prompts_sha256") != sha256(paths["prompts"]):
        raise ValueError("capture and train prompts SHA256 differ")
    if capture.get("split") != "train":
        raise ValueError("provider manifest requires train capture")
    model_entries = snapshot.get("models", {})
    for role in ("target", "draft"):
        entry = model_entries.get(role, {})
        if not isinstance(entry.get("directory"), str):
            raise ValueError(f"{role} model snapshot directory missing")
        paths[f"{role}_model_dir"] = Path(entry["directory"]).resolve()
    offsets_record = capture.get("offsets", {})
    offsets_path = paths["capture_manifest"].parent / offsets_record.get("path", "")
    if not offsets_path.is_file() or sha256(offsets_path) != offsets_record.get("sha256"):
        raise ValueError("native offset map is absent or changed")
    offsets = np.load(offsets_path, allow_pickle=False)
    absolute = np.load(paths["absolute_d2t"], allow_pickle=False)
    if (
        offsets.ndim != 1
        or absolute.ndim != 1
        or len(offsets) != len(absolute)
        or not np.array_equal(np.arange(len(offsets), dtype=np.int64) + offsets, absolute)
    ):
        raise ValueError("absolute d2t and native offset maps disagree")
    with paths["prompts"].open() as prompt_stream:
        prompts_count = sum(bool(line.strip()) for line in prompt_stream)
    if prompts_count < 1:
        raise ValueError("training prompts are empty")
    teacher = None
    if teacher_dir is not None:
        directory = Path(teacher_dir).resolve()
        teacher = {
            "directory": str(directory),
            "manifest_sha256": sha256(directory / "manifest.json"),
        }
    hashes = {
        "capture_manifest": sha256(paths["capture_manifest"]),
        "prompts": sha256(paths["prompts"]),
        "target_gguf": TARGET_GGUF_SHA256,
        "candidate_d_gguf": CANDIDATE_D_SHA256,
        "base_draft_gguf": base_hash,
        "absolute_d2t": sha256(paths["absolute_d2t"]),
        "model_snapshot_manifest": sha256(paths["model_snapshot_manifest"]),
    }
    manifest = {
        "schema": SCHEMA,
        "training_eligible": capture.get("training_eligible") is True,
        "split": "train",
        "prompt_count": prompts_count,
        "capture_id": capture_id,
        "angelslim_revision": ANGELSLIM_REVISION,
        "paths": {key: str(path) for key, path in paths.items()},
        "sha256": hashes,
        "teacher": teacher,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "capture-manifest",
        "prompts",
        "absolute-d2t",
        "target-gguf",
        "candidate-d-gguf",
        "base-draft-gguf",
        "model-snapshot-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--base-draft-gguf-sha256", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--teacher-dir", type=Path)
    args = parser.parse_args()
    result = prepare(
        args.capture_manifest,
        args.prompts,
        args.absolute_d2t,
        args.target_gguf,
        args.candidate_d_gguf,
        args.base_draft_gguf,
        args.base_draft_gguf_sha256,
        args.model_snapshot_manifest,
        args.capture_id,
        args.output,
        teacher_dir=args.teacher_dir,
    )
    print(
        json.dumps({"output": str(args.output), "training_eligible": result["training_eligible"]})
    )


if __name__ == "__main__":
    main()

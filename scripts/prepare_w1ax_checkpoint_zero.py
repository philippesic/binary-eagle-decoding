#!/usr/bin/env python3
"""Save an untrained row-scale W1Ax checkpoint from pinned dense EAGLE weights.

This is a model initialization and export gate, not training or acceptance
evidence. It loads the official target and drafter on CPU and writes no prompts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from evaluate_pytorch_w1a1 import verify_model_snapshot  # noqa: E402
from w1ax_capture_provider import (  # noqa: E402
    ANGELSLIM_REVISION,
    DRAFT_REPO,
    DRAFT_REVISION,
    TARGET_REPO,
    TARGET_REVISION,
)

from w1a1_eagle.official_loader import load_official_eagle3  # noqa: E402
from w1a1_eagle.recurrent_qat import (  # noqa: E402
    JointQATConfig,
    W1AxContract,
    install_joint_linears,
    save_joint_checkpoint,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    snapshot_path: Path,
    base_gguf: Path,
    expected_base_sha256: str,
    activation_bits: int,
    output_dir: Path,
) -> dict:
    snapshot_path = snapshot_path.resolve()
    base_gguf = base_gguf.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if sha256(base_gguf) != expected_base_sha256:
        raise ValueError("base F16 draft GGUF differs from pinned SHA256")
    snapshot = json.loads(snapshot_path.read_text())
    paths = {}
    for role, repo, revision in (
        ("target", TARGET_REPO, TARGET_REVISION),
        ("draft", DRAFT_REPO, DRAFT_REVISION),
    ):
        entry = snapshot.get("models", {}).get(role, {})
        if entry.get("repo") != repo or entry.get("revision") != revision:
            raise ValueError(f"{role} snapshot revision differs from pin")
        paths[role] = Path(entry["directory"]).resolve()
        verify_model_snapshot(paths[role], entry)
    config = JointQATConfig(
        W1AxContract(activation_bits, "row"),
        device="cpu",
        objective="hard_ce",
        seed=0,
    )
    start = time.monotonic()
    model = load_official_eagle3(
        paths["target"],
        paths["draft"],
        angelslim_revision=ANGELSLIM_REVISION,
        total_token=60,
        depth=5,
        top_k=10,
        threshold=1.0,
        target_load_kwargs={"dtype": torch.float16, "device_map": "cpu"},
    )
    model.eval()
    linears = install_joint_linears(model.eagle_layer, model.base_model, config)
    output_dir.mkdir(parents=True)
    saved = save_joint_checkpoint(
        linears,
        config,
        expected_base_sha256,
        output_dir / "joint.npz",
        output_dir / "joint.json",
    )
    report = {
        "schema": "w1ax_checkpoint_zero_v1",
        "execution_device": "cpu",
        "training_steps": 0,
        "activation_bits": activation_bits,
        "scale_layout": "row",
        "objective": "hard_ce",
        "model_snapshot_manifest_sha256": sha256(snapshot_path),
        "base_gguf_sha256": expected_base_sha256,
        "angelslim_revision": ANGELSLIM_REVISION,
        "projection_count": len(linears),
        "checkpoint_sha256": saved["checkpoint_sha256"],
        "manifest_sha256": saved["manifest_sha256"],
        "elapsed_seconds": time.monotonic() - start,
        "peak_process_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--base-draft-gguf", type=Path, required=True)
    parser.add_argument("--base-draft-gguf-sha256", required=True)
    parser.add_argument("--activation-bits", type=int, choices=(1, 4, 8, 16), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(
        args.model_snapshot_manifest,
        args.base_draft_gguf,
        args.base_draft_gguf_sha256,
        args.activation_bits,
        args.output_dir,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

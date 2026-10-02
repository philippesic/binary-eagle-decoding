"""Research-only deployment preflight; no capture, device query or model loading.

Call preflight_pair before any expensive evaluator action. The manifest IS the
complete deployment recipe when bound to the producer publication; training
profile names are not required. A bare manifest cannot detect semantic relabeling.
This is zero-update replay of effective exported state, not optimizer resume.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import check_continuous_w1ax_readiness as gate
import export_recurrent_binary as exporter

from w1a1_eagle.recurrent_qat import install_joint_linears


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class PreparedCheckpoint:
    checkpoint: Path
    manifest: Path
    bits: int
    base_hash: str
    checkpoint_sha256: str
    manifest_sha256: str
    config: object


def preflight_checkpoint(checkpoint, manifest, bits, base_hash):
    """Validate deployment identity and effective arrays without model allocation."""
    if bits not in (8, 1):
        raise ValueError("paired development supports A8/A1 only; A4 is a separate gap")
    checkpoint, manifest = Path(checkpoint), Path(manifest)
    config = gate.checkpoint_joint_config(manifest, bits, base_hash)
    declared = json.loads(manifest.read_text())
    expected = exporter.check_manifest(declared, base_hash)
    checkpoint_hash = digest(checkpoint)
    if declared["checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("continuous precision checkpoint contract differs")
    correction, affine = declared.get("fusion_correction"), declared.get("affine_weights")
    extras = (
        exporter.load_fusion_correction(checkpoint, correction, expected["fc"][1])
        if correction is not None
        else {}
    )
    midpoints = (
        exporter.load_affine_weights(checkpoint, affine, expected) if affine is not None else {}
    )
    exporter.load_checkpoint(
        checkpoint, expected, row_scale=True, extra_names=set(extras) | set(midpoints)
    )
    quantizers = declared.get("activation_quantizers")
    if (
        quantizers is not None
        and bits != 1
        and any(item["clip_ratio"] < 2**-16 for item in quantizers["boundaries"].values())
    ):
        raise ValueError("deployed clip parameter is outside trainable quantizer contract")
    return PreparedCheckpoint(
        checkpoint, manifest, bits, base_hash, checkpoint_hash, digest(manifest), config
    )


def verify_publication(checkpoint_dir):
    """Bind deployment manifest identity to the existing producer publication."""
    root = Path(checkpoint_dir)
    published = json.loads((root / "manifest.json").read_text())
    if published.get("schema") != "continuous_joint_w1ax_v1" or set(
        published.get("exports", {})
    ) != {"A8", "A1"}:
        raise ValueError("published checkpoint export inventory is incomplete")
    for lane in ("A8", "A1"):
        inventory = published["exports"][lane]
        if set(inventory) != {"joint.npz", "joint.json"}:
            raise ValueError("published checkpoint export file inventory differs")
        for name, expected_digest in inventory.items():
            path = root / lane / name
            if path.is_symlink() or not path.is_file() or digest(path) != expected_digest:
                raise ValueError("published checkpoint export hash mismatch")


def preflight_pair(checkpoint_dir, base_hash):
    """Both lanes must pass before export/native capture or full model load."""
    root = Path(checkpoint_dir)
    verify_publication(root)
    return {
        bits: preflight_checkpoint(
            root / f"A{bits}/joint.npz", root / f"A{bits}/joint.json", bits, base_hash
        )
        for bits in (8, 1)
    }


def construct_and_load(prepared, drafter, target):
    """Faithful pure install→actual loader stage with declared CPU configuration."""
    if (
        digest(prepared.checkpoint) != prepared.checkpoint_sha256
        or digest(prepared.manifest) != prepared.manifest_sha256
    ):
        raise ValueError("checkpoint identity changed after preflight")
    linears = install_joint_linears(drafter, target, prepared.config)
    gate._load_checkpoint(
        prepared.checkpoint, prepared.manifest, linears, prepared.bits, prepared.base_hash
    )
    return linears

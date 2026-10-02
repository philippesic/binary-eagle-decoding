"""Generate an unapplied source-bound production diff from this worktree."""

import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "scripts/w1ax_continuous_stages.py"
OUT = Path(__file__).resolve().parent
HELPER = '''def _development_checkpoint_preflight(checkpoint_dir, base_hash):
    """Validate both deployed recipes/arrays before export or native capture."""
    from check_continuous_w1ax_readiness import checkpoint_joint_config
    from export_recurrent_binary import (
        check_manifest,
        load_affine_weights,
        load_checkpoint,
        load_fusion_correction,
    )

    published = json.loads((Path(checkpoint_dir) / "manifest.json").read_text())
    if published.get("schema") != "continuous_joint_w1ax_v1" or set(
        published.get("exports", {})
    ) != {"A8", "A1"}:
        raise ValueError("published checkpoint export inventory is incomplete")
    for lane in ("A8", "A1"):
        inventory = published["exports"][lane]
        if set(inventory) != {"joint.npz", "joint.json"}:
            raise ValueError("published checkpoint export file inventory differs")
        for name, expected_digest in inventory.items():
            path = Path(checkpoint_dir) / lane / name
            if path.is_symlink() or not path.is_file() or sha256(path) != expected_digest:
                raise ValueError("published checkpoint export hash mismatch")
    configs, identities = {}, {}
    for bits in (8, 1):
        checkpoint = Path(checkpoint_dir) / f"A{bits}/joint.npz"
        manifest_path = Path(checkpoint_dir) / f"A{bits}/joint.json"
        configs[bits] = checkpoint_joint_config(manifest_path, bits, base_hash)
        manifest = json.loads(manifest_path.read_text())
        expected = check_manifest(manifest, base_hash)
        if sha256(checkpoint) != manifest["checkpoint_sha256"]:
            raise ValueError("continuous precision checkpoint contract differs")
        correction, affine = manifest.get("fusion_correction"), manifest.get("affine_weights")
        extras = (
            load_fusion_correction(checkpoint, correction, expected["fc"][1])
            if correction is not None
            else {}
        )
        midpoints = load_affine_weights(checkpoint, affine, expected) if affine is not None else {}
        load_checkpoint(
            checkpoint, expected, row_scale=True, extra_names=set(extras) | set(midpoints)
        )
        quantizers = manifest.get("activation_quantizers")
        if (
            quantizers is not None
            and bits != 1
            and any(item["clip_ratio"] < 2**-16 for item in quantizers["boundaries"].values())
        ):
            raise ValueError("deployed clip parameter is outside trainable quantizer contract")
        identities[bits] = (sha256(checkpoint), sha256(manifest_path))
    return configs, identities


def _development_checkpoint_unchanged(checkpoint_dir, bits, identities):
    if identities[bits] != (
        sha256(Path(checkpoint_dir) / f"A{bits}/joint.npz"),
        sha256(Path(checkpoint_dir) / f"A{bits}/joint.json"),
    ):
        raise ValueError("development checkpoint identity changed after preflight")


'''
source = SOURCE.read_text()
changed = source.replace("def _evaluate_development(", HELPER + "def _evaluate_development(", 1)
# Restrict edits to this function; remove obsolete default-config imports only here.
start = changed.index("def _evaluate_development(")
end = changed.index("\ndef evaluate_development(", start)
block = changed[start:end]
block = block.replace("        JointQATConfig,\n        W1AxContract,\n", "", 1)
needle = '    check_stop(run_dir)\n    prompts = Path(config["native_prompts"])'
replacement = """    check_stop(run_dir)
    checkpoint_configs, checkpoint_identities = _development_checkpoint_preflight(
        checkpoint_dir, sources["sha256"]["base_draft_gguf"]
    )
    prompts = Path(config["native_prompts"])"""
assert needle in block
block = block.replace(needle, replacement, 1)
needle = '        exported = folder / "student.gguf"\n        export_audit = export_model('
replacement = """        exported = folder / "student.gguf"
        _development_checkpoint_unchanged(checkpoint_dir, bits, checkpoint_identities)
        export_audit = export_model("""
assert needle in block
block = block.replace(needle, replacement, 1)
needle = (
    '        qat = JointQATConfig(W1AxContract(bits, "row"), '
    'device="cuda:0", allow_accelerator=True)'
)
replacement = """        _development_checkpoint_unchanged(
            checkpoint_dir, bits, checkpoint_identities
        )
        qat = replace(checkpoint_configs[bits], device="cuda:0", allow_accelerator=True)"""
assert needle in block
block = block.replace(needle, replacement, 1)
changed = changed[:start] + block + changed[end:]
patch = "".join(
    difflib.unified_diff(
        source.splitlines(True),
        changed.splitlines(True),
        fromfile="a/scripts/w1ax_continuous_stages.py",
        tofile="b/scripts/w1ax_continuous_stages.py",
    )
)
(OUT / "development-preflight.patch").write_text(patch)
(OUT / "production_preflight.py").write_text(
    '"""Exact proposed pure helper, extracted for CPU tests; no production edits."""\n\n'
    "import json\nfrom pathlib import Path\n\nfrom adapter import digest as sha256\n\n\n"
    + HELPER.rstrip()
    + "\n"
)
files = (
    "scripts/w1ax_continuous_stages.py",
    "scripts/check_continuous_w1ax_readiness.py",
    "scripts/export_recurrent_binary.py",
    "src/w1a1_eagle/recurrent_qat.py",
    "src/w1a1_eagle/continuous_qat.py",
    "src/w1a1_eagle/fusion_correction.py",
    "src/w1a1_eagle/affine_binary.py",
    "src/w1a1_eagle/learned_activation.py",
    "configs/qat_optimization_profiles.json",
)
(OUT / "source-identity.json").write_text(
    json.dumps(
        {
            "schema": "eval_recipe_proposal_source_identity_v1",
            "source_sha256": {
                path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files
            },
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

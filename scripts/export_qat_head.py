"""Build a derived BF16 EAGLE drafter with a trained head and an evaluation manifest."""

import argparse
import hashlib
import json
import shutil
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_files(directory: Path) -> list[dict]:
    return [
        {
            "path": str(path.relative_to(directory)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    ]


def replace_head(source: Path, head_path: Path, destination: Path) -> None:
    """Copy every source tensor except lm_head.weight, preserving metadata and dtypes."""
    candidate = torch.load(head_path, map_location="cpu", weights_only=True)
    if set(candidate) != {"weight"}:
        raise ValueError("trained head must contain only a weight tensor")
    head = candidate["weight"]
    if not isinstance(head, torch.Tensor) or head.ndim != 2 or head.dtype != torch.bfloat16:
        raise ValueError("trained head must be a rank-two BF16 tensor")
    with safe_open(source, framework="pt", device="cpu") as reader:
        names = set(reader.keys())
        if "lm_head.weight" not in names or "embed_tokens.weight" in names:
            raise ValueError("source is not the pinned untied AngelSlim drafter")
        original = reader.get_tensor("lm_head.weight")
        if head.shape != original.shape:
            raise ValueError(f"trained head shape {tuple(head.shape)} != {tuple(original.shape)}")
        if original.dtype != torch.bfloat16:
            raise ValueError("source head is not BF16")
        metadata = reader.metadata()
        tensors = {name: reader.get_tensor(name) for name in names if name != "lm_head.weight"}
    tensors["lm_head.weight"] = head.contiguous()
    save_file(tensors, destination, metadata=metadata)
    exported = load_file(destination)
    if set(exported) != names or not torch.equal(exported["lm_head.weight"], head):
        raise RuntimeError("derived checkpoint has wrong keys or head")
    for name, tensor in tensors.items():
        if name != "lm_head.weight" and not torch.equal(exported[name], tensor):
            raise RuntimeError(f"derived checkpoint modified {name}")


def export_checkpoint(
    project_root: Path,
    config_path: Path,
    original_manifest_path: Path,
    head_path: Path,
    training_summary_path: Path,
    output_dir: Path,
    manifest_path: Path,
) -> dict:
    config = tomllib.loads(config_path.read_text())
    original_manifest = json.loads(original_manifest_path.read_text())
    source_dir = project_root / "models/hf/Qwen3-4B_eagle3"
    source_entry = original_manifest["models"]["draft"]
    if Path(source_entry["directory"]).resolve() != source_dir.resolve():
        raise ValueError("original draft directory differs from source manifest")
    if config["models"]["draft_dir"] != str(output_dir.relative_to(project_root)):
        raise ValueError("evaluation config draft directory differs from output")
    if config["models"]["target_repo"] != original_manifest["models"]["target"]["repo"]:
        raise ValueError("target repository differs from original manifest")
    if config["models"]["target_revision"] != original_manifest["models"]["target"]["revision"]:
        raise ValueError("target revision differs from original manifest")
    source_files = {item["path"]: item for item in source_entry["files"]}
    actual_files = {item["path"]: item for item in snapshot_files(source_dir)}
    if actual_files != source_files:
        raise ValueError("source draft files differ from original model manifest")
    if "model.safetensors" not in source_files:
        raise ValueError("source manifest lacks model.safetensors")
    if output_dir.exists() or manifest_path.exists():
        raise FileExistsError("derived model directory or manifest already exists")
    summary = json.loads(training_summary_path.read_text())
    if not isinstance(summary, dict):
        raise ValueError("training summary must be a JSON object")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qat-head-export-", dir=output_dir.parent) as tmp:
        temporary = Path(tmp)
        for relative in sorted(source_files):
            source = source_dir / relative
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative == "model.safetensors":
                replace_head(source, head_path, destination)
            else:
                shutil.copy2(source, destination)
        derived_files = snapshot_files(temporary)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output_dir)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "models": {
            "target": original_manifest["models"]["target"],
            "draft": {
                "repo": config["models"]["draft_repo"],
                "revision": config["models"]["draft_revision"],
                "directory": str(output_dir.resolve()),
                "files": derived_files,
            },
        },
        "provenance": {
            "source_model_manifest_sha256": sha256_file(original_manifest_path),
            "source_draft_checkpoint_sha256": source_files["model.safetensors"]["sha256"],
            "trained_head_sha256": sha256_file(head_path),
            "training_summary_sha256": sha256_file(training_summary_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pytorch_w1a1_qat_head.toml"))
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("models/qat-head-pilot-v1"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.config.resolve().parents[1]
    export_checkpoint(
        project_root,
        args.config.resolve(),
        args.original_manifest.resolve(),
        args.head.resolve(),
        args.training_summary.resolve(),
        args.output_dir.resolve(),
        args.manifest.resolve(),
    )
    print(f"wrote derived model and manifest: {args.output_dir}, {args.manifest}")


if __name__ == "__main__":
    main()

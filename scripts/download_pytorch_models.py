"""Download immutable model snapshots and hash their files for W1A1 experiments."""

import argparse
import hashlib
import json
import tomllib
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_files(directory: Path) -> list[dict[str, str | int]]:
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(directory).parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(directory)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pytorch_w1a1.toml"))
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    config_path = args.config.resolve()
    project_root = config_path.parents[1]
    config = tomllib.loads(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    models = config["models"]
    results = {}
    for role in ("target", "draft"):
        repo = models[f"{role}_repo"]
        revision = models[f"{role}_revision"]
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError(f"{role} revision must be a full 40-character commit SHA")
        directory = (project_root / models[f"{role}_dir"]).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=directory)
        results[role] = {
            "repo": repo,
            "revision": revision,
            "directory": str(directory),
            "files": snapshot_files(directory),
        }

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "models": results,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote model manifest: {args.manifest}")


if __name__ == "__main__":
    main()

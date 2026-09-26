#!/usr/bin/env python3
"""Freeze the W1Ax policy and context diagnostic configuration matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY_DEPTHS = (1, 2, 3, 5)
POLICY_FLOORS = (0.0, 0.1, 0.3)
CONTEXT_CAPS = (32, 128)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(value: str, root: Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_jsonl(path: Path, expected_count: int, label: str) -> list[dict[str, Any]]:
    rows = []
    ids: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{label}: invalid JSON on line {line_number}: {error}") from error
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"{label}: each prompt must be an object with a string id")
        if row["id"] in ids:
            raise ValueError(f"{label}: duplicate prompt id {row['id']!r}")
        ids.add(row["id"])
        rows.append(row)
    if len(rows) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} prompts, found {len(rows)}")
    # The final holdout is deliberately excluded from this diagnostic generator.
    if any("qat_final" in prompt_id.lower() or "final" in prompt_id.lower() for prompt_id in ids):
        raise ValueError(f"{label}: prompt IDs indicate a reserved/final prompt set")
    return rows


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, int):
        return str(value)
    raise TypeError(f"unsupported TOML scalar: {value!r}")


def update_table(source: str, table: str, updates: dict[str, Any]) -> str:
    """Update scalar keys in one TOML table without reserializing unrelated settings."""
    table_re = re.compile(rf"(?m)^\[{re.escape(table)}\][ \t]*(?:\r?\n|$)")
    match = table_re.search(source)
    if match is None:
        raise ValueError(f"primary config is missing [{table}]")
    next_table = re.search(r"(?m)^\[[^\]]+\]\s*$", source[match.end() :])
    end = match.end() + next_table.start() if next_table else len(source)
    block = source[match.end() : end]
    for key, value in updates.items():
        line_re = re.compile(rf"(?m)^(\s*){re.escape(key)}\s*=\s*[^\n]*(?:\n|$)")
        replacement = f"{key} = {toml_value(value)}\n"
        if line_re.search(block):
            block = line_re.sub(replacement, block, count=1)
        else:
            if block and not block.endswith("\n"):
                block += "\n"
            block += replacement
    return source[: match.end()] + block + source[end:]


def prepare(
    primary_config: Path,
    development_prompts: Path,
    context_prompts: Path,
    output_dir: Path,
    *,
    root: Path = ROOT,
) -> Path:
    primary_config = primary_config.resolve()
    development_prompts = development_prompts.resolve()
    context_prompts = context_prompts.resolve()
    for path in (primary_config, development_prompts, context_prompts):
        if not path.is_file():
            raise FileNotFoundError(path)
    dev_rows = read_jsonl(development_prompts, 24, "qat_development")
    context_rows = read_jsonl(context_prompts, 9, "context diagnostic")

    original_text = primary_config.read_text()
    primary = tomllib.loads(original_text)
    if primary.get("schema_version") != 1:
        raise ValueError("primary config must use schema_version = 1")
    models, server = primary.get("models"), primary.get("server")
    evaluation = primary.get("evaluation")
    if not isinstance(models, dict) or not isinstance(server, dict) or not isinstance(evaluation, dict):
        raise ValueError("primary config must contain [models], [server], and [evaluation]")
    if evaluation.get("w1ax_matrix") is not True:
        raise ValueError("primary config must enable the eight-path w1ax_matrix")
    if "target" not in models or "ordinary_draft" not in models:
        raise ValueError("primary config is missing target or ordinary draft model entries")
    w1ax = primary.get("w1ax")
    if not isinstance(w1ax, dict) or not isinstance(w1ax.get("draft"), str):
        raise ValueError("primary config must contain [w1ax].draft")
    source_hash = sha256(primary_config)
    prompt_files = {"qat_development": development_prompts, "context_diagnostic": context_prompts}
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(exist_ok=True)
    generated: list[dict[str, Any]] = []

    def emit(name: str, prompt_set: str, updates: dict[str, Any]) -> None:
        text = update_table(original_text, "evaluation", {
            "prompt_file": str(prompt_files[prompt_set]),
            "warmup_requests": 2,
            "repetitions": 5,
            "w1ax_policy_diagnostic": prompt_set == "qat_development",
            "prompt_set": prompt_set,
            **updates,
        })
        # Validate the generated file and prove the run-specific edits kept the
        # resolved model and server settings byte-for-value identical.
        parsed = tomllib.loads(text)
        if parsed["models"] != models or parsed["server"] != server:
            raise AssertionError("config generation changed model or server settings")
        destination = config_dir / name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite frozen config: {destination}")
        destination.write_text(text)
        generated.append({
            "name": destination.name,
            "path": str(destination),
            "sha256": sha256(destination),
            "prompt_set": prompt_set,
            "prompt_file": str(prompt_files[prompt_set]),
            "prompt_file_sha256": sha256(prompt_files[prompt_set]),
            "prompt_count": len(dev_rows) if prompt_set == "qat_development" else len(context_rows),
            "draft_depth": parsed["evaluation"]["max_draft_tokens"],
            "min_draft_probability": parsed["evaluation"]["min_draft_probability"],
            "max_output_tokens": parsed["evaluation"]["max_output_tokens"],
            "w1ax_policy_diagnostic": parsed["evaluation"]["w1ax_policy_diagnostic"],
        })

    for cap in CONTEXT_CAPS:
        emit(
            f"context-cap-{cap}.toml",
            "context_diagnostic",
            {
                "w1ax_matrix": True,
                "max_draft_tokens": 5,
                "min_draft_probability": 0.0,
                "max_output_tokens": cap,
            },
        )
    for depth in POLICY_DEPTHS:
        for floor in POLICY_FLOORS:
            floor_tag = str(floor).replace(".", "p")
            emit(
                f"development-d{depth}-pmin-{floor_tag}.toml",
                "qat_development",
                {
                    "max_draft_tokens": depth,
                    "min_draft_probability": floor,
                    "max_output_tokens": evaluation["max_output_tokens"],
                },
            )

    input_hashes = {"primary_config": source_hash}
    input_files: dict[str, dict[str, Any]] = {}

    def record_input(key: str, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"input {key}: {path}")
        input_hashes[key] = sha256(path)
        input_files[key] = {"path": str(path.resolve()), "sha256": input_hashes[key]}

    record_input("primary_config", primary_config)
    model_paths = {key: value for key, value in models.items()}
    model_paths["w1ax:draft"] = w1ax["draft"]
    for table_name in ("packed_variants", "weight_only_variants", "native_operand_variants"):
        table = primary.get(table_name, {})
        if isinstance(table, dict):
            for variant_name, spec in table.items():
                if isinstance(spec, dict) and isinstance(spec.get("draft"), str):
                    model_paths[f"{table_name}:{variant_name}"] = spec["draft"]
    for key, value in model_paths.items():
        path = resolve_repo_path(value, root)
        record_input(f"model:{key}", path)
    binary_path = resolve_repo_path(server["binary"], root)
    record_input("server_binary", binary_path)
    record_input("prompt:qat_development", development_prompts)
    record_input("prompt:context_diagnostic", context_prompts)
    manifest = {
        "schema": "w1ax_diagnostic_suite_v1",
        "primary_config": str(primary_config),
        "primary_config_sha256": source_hash,
        "model_server_hashes": input_hashes,
        "input_files": input_files,
        "models": models,
        "server": server,
        "prompt_sets": {
            "qat_development": {
                "path": str(development_prompts), "sha256": sha256(development_prompts),
                "count": len(dev_rows), "ids": [row["id"] for row in dev_rows],
            },
            "context_diagnostic": {
                "path": str(context_prompts), "sha256": sha256(context_prompts),
                "count": len(context_rows), "ids": [row["id"] for row in context_rows],
            },
        },
        "configs": generated,
    }
    manifest_path = output_dir / "suite.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite suite manifest: {manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-config", type=Path, required=True)
    parser.add_argument("--development-prompts", type=Path, required=True)
    parser.add_argument("--context-prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.primary_config, args.development_prompts, args.context_prompts, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

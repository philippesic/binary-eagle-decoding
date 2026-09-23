"""Run held-out greedy draft-acceptance sweeps with the pinned PyTorch drafter."""

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from importlib.metadata import version
from itertools import zip_longest
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_prompts(path: Path) -> list[dict]:
    prompts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [item["id"] for item in prompts]
    if not prompts or len(ids) != len(set(ids)):
        raise ValueError("prompt manifest must be nonempty with unique IDs")
    for item in prompts:
        if not isinstance(item.get("messages"), list) or not item["messages"]:
            raise ValueError(f"prompt {item['id']} has no messages")
    return prompts


def verify_model_snapshot(directory: Path, manifest_entry: dict) -> None:
    if directory.resolve() != Path(manifest_entry["directory"]).resolve():
        raise ValueError(f"model directory differs from manifest: {directory}")
    expected = {item["path"]: item for item in manifest_entry["files"]}
    actual = {
        str(path.relative_to(directory)): path
        for path in directory.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(directory).parts
    }
    if set(actual) != set(expected):
        raise ValueError(
            f"model file set differs from manifest: {directory}; "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    for relative_path, path in actual.items():
        recorded = expected[relative_path]
        if path.stat().st_size != recorded["bytes"] or sha256_file(path) != recorded["sha256"]:
            raise ValueError(f"model file hash differs from manifest: {path}")


def git_revision(project_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True
    ).strip()


def run_generation(model, input_ids, evaluation: dict) -> tuple[list[int], list[int]]:
    output, _, _, lengths = model.eagle_generate(
        input_ids,
        temperature=evaluation["temperature"],
        max_new_tokens=evaluation["max_new_tokens"],
        max_length=evaluation["max_length"],
        log=True,
    )
    generated = output[0, input_ids.shape[1] :].detach().cpu().tolist()
    accepted = [int(value) for value in lengths]
    return generated, accepted


def encode_prompt(tokenizer, messages: list[dict], thinking_mode: bool, device):
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=thinking_mode,
    )
    return encoded["input_ids"].to(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pytorch_w1a1.toml"))
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--variants", nargs="*", help="Variant names; default: all")
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    parser.add_argument("--limit-prompts", type=int, help="Short diagnostic subset")
    parser.add_argument(
        "--allow-greedy-mismatch",
        action="store_true",
        help="Continue a Metal development run after recording a target/EAGLE mismatch",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    project_root = config_path.parents[1]
    config = tomllib.loads(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    evaluation = config["evaluation"]
    if evaluation["temperature"] != 0.0:
        raise ValueError("this acceptance runner currently supports greedy decoding only")
    prompt_path = project_root / evaluation["prompt_manifest"]
    prompt_hash = sha256_file(prompt_path)
    if prompt_hash != evaluation["prompt_sha256"]:
        raise ValueError(f"prompt manifest hash changed: {prompt_hash}")
    prompts = load_prompts(prompt_path)
    if args.limit_prompts is not None:
        if args.limit_prompts < 1:
            raise ValueError("--limit-prompts must be positive")
        prompts = prompts[: args.limit_prompts]
    variants = config["variants"]
    if args.variants:
        requested = set(args.variants)
        variants = [item for item in variants if item["name"] in requested]
        if len(variants) != len(requested):
            raise ValueError("one or more requested variant names are missing from config")
    if not variants:
        raise ValueError("no variants selected")
    if args.dry_run:
        print(json.dumps({"prompts": len(prompts), "variants": [v["name"] for v in variants]}))
        return

    if args.model_manifest is None or args.run_id is None:
        parser.error("--model-manifest and --run-id are required for an actual run")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID may contain only letters, numbers, underscores, and hyphens")
    model_manifest = json.loads(args.model_manifest.read_text())
    if model_manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("model manifest was created with a different config")
    for role in ("target", "draft"):
        entry = model_manifest["models"][role]
        if entry["repo"] != config["models"][f"{role}_repo"]:
            raise ValueError(f"{role} model repository mismatch")
        if entry["revision"] != config["models"][f"{role}_revision"]:
            raise ValueError(f"{role} model revision mismatch")
        verify_model_snapshot(project_root / config["models"][f"{role}_dir"], entry)

    import torch
    import transformers

    sys.path.insert(0, str(project_root / "src"))
    from w1a1_eagle import W1A1Config, install_w1a1
    from w1a1_eagle.official_loader import load_official_eagle3

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        gpu_name = torch.cuda.get_device_name(0)
        if "5080" not in gpu_name:
            raise RuntimeError(f"expected the RTX 5080 experiment host, found {gpu_name}")
        hardware_memory = torch.cuda.get_device_properties(0).total_memory
        device_map = "cuda:0"
    else:
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple Metal is not available")
        gpu_name = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        hardware_memory = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
        device_map = "mps"
    torch.manual_seed(evaluation["seed"])

    run_dir = project_root / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / "resolved-config.toml")
    shutil.copyfile(prompt_path, run_dir / "prompts.jsonl")
    shutil.copyfile(args.model_manifest, run_dir / "model-manifest.json")
    environment = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "project_revision": git_revision(project_root),
        "llama_cpp_revision": config.get("runtime", {}).get("llama_cpp_revision"),
        "angelslim_revision": config["models"]["angelslim_revision"],
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "prompt_sha256": prompt_hash,
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "angelslim_package": version("angelslim"),
        "execution_device": args.device,
        "development_check": args.device == "mps" or args.limit_prompts is not None,
        "host_platform": platform.platform(),
        "gpu": gpu_name,
        "cuda_runtime": torch.version.cuda,
        "hardware_memory_bytes": hardware_memory,
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    models = config["models"]
    model = load_official_eagle3(
        project_root / models["target_dir"],
        project_root / models["draft_dir"],
        angelslim_revision=models["angelslim_revision"],
        total_token=evaluation["total_token"],
        depth=evaluation["depth"],
        top_k=evaluation["top_k"],
        threshold=evaluation["threshold"],
        target_load_kwargs={"dtype": torch.bfloat16, "device_map": device_map},
    )
    if model.eagle_layer.total_tokens != evaluation["total_token"] - 1:
        raise RuntimeError("unexpected draft tree node budget")
    model.eval()
    device = next(model.base_model.parameters()).device
    environment["target_dtype"] = str(model.base_model.dtype)
    environment["draft_dtype"] = str(next(model.eagle_layer.parameters()).dtype)
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    quant = config["quantization"]
    quant_config = W1A1Config(
        zero_sign=quant["zero_sign"],
        weight_scale=quant["weight_scale"],
        activation_scale=quant["activation_scale"],
    )
    first_ids = encode_prompt(
        model.tokenizer, prompts[0]["messages"], evaluation["thinking_mode"], device
    )
    parity_tokens = min(32, evaluation["max_new_tokens"])
    target_only, _, _ = model.naive_generate(
        first_ids,
        temperature=0.0,
        max_new_tokens=parity_tokens,
        max_length=evaluation["max_length"],
        log=True,
    )
    target_only_ids = target_only[0, first_ids.shape[1] :].detach().cpu().tolist()
    ordinary = run_generation(model, first_ids, evaluation)
    missing_token = object()
    mismatch = next(
        (
            index
            for index, (target_id, eagle_id) in enumerate(
                zip_longest(
                    target_only_ids[:parity_tokens],
                    ordinary[0][:parity_tokens],
                    fillvalue=missing_token,
                )
            )
            if target_id != eagle_id
        ),
        None,
    )
    parity = {
        "prompt_id": prompts[0]["id"],
        "checked_tokens": parity_tokens,
        "target_only_token_ids": target_only_ids[:parity_tokens],
        "ordinary_eagle_token_ids": ordinary[0][:parity_tokens],
        "first_mismatch_index": mismatch,
        "target_eagle_match": mismatch is None,
    }
    (run_dir / "greedy-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    environment["target_eagle_greedy_match"] = mismatch is None
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    if mismatch is not None and not (args.device == "mps" and args.allow_greedy_mismatch):
        raise RuntimeError("ordinary EAGLE differs from target-only greedy continuation")
    if mismatch is not None:
        print(f"Metal development run: target/EAGLE first differ at token {mismatch + 1}")
    results = []
    parity_checked = False
    with (run_dir / "acceptance.jsonl").open("w") as stream:
        for variant in variants:
            handle = install_w1a1(
                model.eagle_layer,
                variant["groups"],
                quant_config,
                enabled=False,
                target=model.base_model,
            )
            try:
                if not parity_checked:
                    disabled = run_generation(model, first_ids, evaluation)
                    if disabled != ordinary:
                        raise RuntimeError("disabled W1A1 wrapper changed ordinary EAGLE output")
                    parity["disabled_wrapper_token_ids"] = disabled[0][:parity_tokens]
                    (run_dir / "greedy-parity.json").write_text(json.dumps(parity, indent=2) + "\n")
                    parity_checked = True
                handle.set_enabled(True)
                for prompt in prompts:
                    input_ids = encode_prompt(
                        model.tokenizer,
                        prompt["messages"],
                        evaluation["thinking_mode"],
                        device,
                    )
                    generated, accepted = run_generation(model, input_ids, evaluation)
                    proposed_per_round = model.eagle_layer.total_tokens
                    if not accepted or any(
                        value < 0 or value > proposed_per_round for value in accepted
                    ):
                        raise RuntimeError("invalid accepted draft count from official runtime")
                    row = {
                        "variant": variant["name"],
                        "groups": variant["groups"],
                        "prompt_id": prompt["id"],
                        "category": prompt["category"],
                        "prompt_tokens": input_ids.shape[1],
                        "generated_token_ids": generated,
                        "generated_tokens": len(generated),
                        "accepted_per_round": accepted,
                        "accepted_draft_tokens": sum(accepted),
                        "proposed_tree_nodes_per_round": proposed_per_round,
                        "proposed_tree_nodes": proposed_per_round * len(accepted),
                        "rounds": len(accepted),
                    }
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    results.append(row)
            finally:
                handle.uninstall()

    summary = {}
    for variant in variants:
        rows = [row for row in results if row["variant"] == variant["name"]]
        accepted = sum(row["accepted_draft_tokens"] for row in rows)
        proposed = sum(row["proposed_tree_nodes"] for row in rows)
        rounds = sum(row["rounds"] for row in rows)
        summary[variant["name"]] = {
            "prompts": len(rows),
            "accepted_draft_tokens": accepted,
            "proposed_tree_nodes": proposed,
            "rounds": rounds,
            "accepted_per_round": accepted / rounds,
            "accepted_per_proposed_tree_node": accepted / proposed,
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

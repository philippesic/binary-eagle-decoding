"""Bounded, cached-input W1A1 drafter-head reconstruction pilot.

Only the drafter vocabulary head is optimized. The target, EAGLE body, and
held-out acceptance prompts are never loaded by this program.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle.fake_binary import W1A1Config, fake_binary_linear  # noqa: E402
from w1a1_eagle.qat_head import TrainableW1A1Head  # noqa: E402

ARTIFACTS = ("train.pt", "validation.pt", "teacher-head.pt")
ROW_FIELDS = (
    "prompt_id",
    "trajectory",
    "draft_call_index",
    "head_call_index",
    "tree_depth_index",
    "row_in_call",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(manifest: dict, name: str) -> dict:
    """Read either the capture worker's files/artifacts table or named entry."""
    if name == "teacher-head.pt":
        record = manifest.get("teacher_head")
        if isinstance(record, dict) and record.get("path") == name:
            return record
    elif name in ("train.pt", "validation.pt"):
        record = manifest.get("captures", {}).get(name.removesuffix(".pt"))
        if isinstance(record, dict) and record.get("path") == name:
            return record
    for table_name in ("artifacts", "files"):
        table = manifest.get(table_name)
        if isinstance(table, dict) and name in table:
            record = table[name]
            return record if isinstance(record, dict) else {"sha256": record}
        if isinstance(table, list):
            matches = [row for row in table if row.get("path") == name or row.get("name") == name]
            if len(matches) == 1:
                return matches[0]
    direct = manifest.get(name)
    if isinstance(direct, dict):
        return direct
    stem = name.removesuffix(".pt").replace("-", "_")
    for key in (f"{stem}_sha256", f"{stem}_hash"):
        if key in manifest:
            return {"sha256": manifest[key]}
    raise ValueError(f"capture manifest has no hash record for {name}")


def _declared_count(manifest: dict, split: str) -> int:
    for table_name in ("split_counts", "counts", "splits"):
        table = manifest.get(table_name)
        if isinstance(table, dict) and split in table:
            value = table[split]
            if isinstance(value, dict):
                value = value.get("rows", value.get("row_count", value.get("count")))
            if isinstance(value, int) and value > 0:
                return value
    for key in (f"{split}_rows", f"{split}_row_count", f"{split}_count"):
        value = manifest.get(key)
        if isinstance(value, int) and value > 0:
            return value
    record = _artifact_record(manifest, f"{split}.pt")
    value = record.get("rows", record.get("row_count", record.get("count")))
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(f"capture manifest has no positive {split} row count")


def _content_hash(messages: list[dict]) -> str:
    encoded = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_prompt_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"prompt file is empty: {path}")
    for row in rows:
        if not isinstance(row.get("id"), str) or not isinstance(row.get("messages"), list):
            raise ValueError(f"invalid prompt row in {path}")
    return rows


def _validate_prompt_provenance(directory: Path, manifest: dict, heldout_path: Path) -> dict:
    """Audit copied prompt content as well as IDs, not just capture-row labels."""
    prompt_manifest_path = directory / "prompt-manifest.json"
    expected = manifest.get("prompt_manifest_sha256")
    if not isinstance(expected, str) or sha256_file(prompt_manifest_path) != expected:
        raise ValueError("copied prompt manifest SHA256 differs from capture manifest")
    prompt_manifest = json.loads(prompt_manifest_path.read_text())
    if prompt_manifest.get("schema_version") != 1:
        raise ValueError("unsupported QAT prompt manifest schema")
    heldout_hash = sha256_file(heldout_path)
    if manifest.get("heldout_prompt_sha256") != heldout_hash:
        raise ValueError("held-out prompt file differs from capture manifest")
    if prompt_manifest.get("heldout_sha256") != heldout_hash:
        raise ValueError("held-out prompt file differs from QAT prompt manifest")
    heldout = _read_prompt_rows(heldout_path)
    heldout_ids = {row["id"] for row in heldout}
    heldout_content = {_content_hash(row["messages"]) for row in heldout}
    prompts = {}
    all_ids = set(heldout_ids)
    all_content = set(heldout_content)
    for split in ("train", "validation"):
        spec = prompt_manifest["splits"][split]
        if spec.get("path") != f"{split}.jsonl":
            raise ValueError(f"unexpected {split} prompt path in prompt manifest")
        path = directory / f"{split}-prompts.jsonl"
        if not isinstance(spec.get("sha256"), str) or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"{split} copied prompt SHA256 differs from prompt manifest")
        rows = _read_prompt_rows(path)
        if len(rows) != spec.get("prompts"):
            raise ValueError(f"{split} prompt count differs from prompt manifest")
        ids = set()
        for row in rows:
            content = _content_hash(row["messages"])
            if row.get("content_sha256") != content:
                raise ValueError(f"{split} prompt content hash mismatch")
            if row["id"] in all_ids or content in all_content:
                raise ValueError(f"{split} prompt overlaps an earlier split or held-out prompt")
            ids.add(row["id"])
            all_ids.add(row["id"])
            all_content.add(content)
        prompts[split] = ids
    return prompts


def _verify_tensor(name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.device.type != "cpu" or value.dtype != torch.bfloat16 or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be a CPU BF16 tensor with shape {shape}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains nonfinite values")


def _load_split(path: Path, width: int, expected_rows: int) -> tuple[torch.Tensor, list[dict]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {"inputs", "rows"}:
        raise ValueError(f"{path.name} must contain only inputs and rows")
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != expected_rows:
        raise ValueError(f"{path.name} row metadata count differs from manifest")
    _verify_tensor(f"{path.name} inputs", payload["inputs"], (expected_rows, width))
    row_keys = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not set(ROW_FIELDS).issubset(row):
            raise ValueError(f"{path.name} row {index} lacks required metadata")
        if not isinstance(row["prompt_id"], str) or not row["prompt_id"]:
            raise ValueError(f"{path.name} row {index} has invalid prompt ID")
        if row["trajectory"] not in ("ordinary", "head_w1a1"):
            raise ValueError(f"{path.name} row {index} has invalid trajectory")
        if any(not isinstance(row[field], int) or row[field] < 0 for field in ROW_FIELDS[2:]):
            raise ValueError(f"{path.name} row {index} has invalid call/depth index")
        key = tuple(row[field] for field in ROW_FIELDS)
        if key in row_keys:
            raise ValueError(f"{path.name} contains duplicate captured row {key}")
        row_keys.add(key)
    return payload["inputs"], rows


def _heldout_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise ValueError(f"held-out prompt file is missing: {path}")
    prompts = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row.get("id") for row in prompts]
    if not ids or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError("held-out prompt IDs are invalid")
    if len(ids) != len(set(ids)):
        raise ValueError("held-out prompt IDs are duplicated")
    return set(ids)


def load_capture(
    directory: Path, width: int, vocab: int, heldout_path: Path, config_path: Path | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Verify all hashes and split boundaries before returning any trainable data."""
    manifest_path = directory / "capture-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise ValueError("capture manifest must be an object")
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported capture manifest schema")
    if "model_manifest_sha256" in manifest:
        path = directory / "model-manifest.json"
        if sha256_file(path) != manifest["model_manifest_sha256"]:
            raise ValueError("copied model manifest SHA256 differs from capture manifest")
    if "config_sha256" in manifest:
        path = config_path or Path(__file__).resolve().parents[1] / "configs/pytorch_w1a1.toml"
        if sha256_file(path) != manifest["config_sha256"]:
            raise ValueError("pinned QAT capture config SHA256 changed")
    prompt_ids = _validate_prompt_provenance(directory, manifest, heldout_path)
    artifact_hashes = {}
    for name in ARTIFACTS:
        path = directory / name
        record = _artifact_record(manifest, name)
        expected = record.get("sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"capture manifest lacks valid SHA256 for {name}")
        if "bytes" in record and path.stat().st_size != record["bytes"]:
            raise ValueError(f"{name} byte count differs from manifest")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} SHA256 differs from capture manifest")
        artifact_hashes[name] = actual
        if "shape" in record:
            expected_shape = (
                [vocab, width]
                if name == "teacher-head.pt"
                else [_declared_count(manifest, name.removesuffix(".pt")), width]
            )
            if record["shape"] != expected_shape:
                raise ValueError(f"{name} shape differs from capture manifest")
        if "dtype" in record and record["dtype"] != "torch.bfloat16":
            raise ValueError(f"{name} dtype differs from expected BF16")
    train, train_rows = _load_split(
        directory / "train.pt", width, _declared_count(manifest, "train")
    )
    validation, validation_rows = _load_split(
        directory / "validation.pt", width, _declared_count(manifest, "validation")
    )
    teacher_payload = torch.load(
        directory / "teacher-head.pt", map_location="cpu", weights_only=True
    )
    if not isinstance(teacher_payload, dict) or set(teacher_payload) != {"weight"}:
        raise ValueError("teacher-head.pt must contain only weight")
    teacher = teacher_payload["weight"]
    _verify_tensor("teacher head weight", teacher, (vocab, width))
    train_ids = {row["prompt_id"] for row in train_rows}
    validation_ids = {row["prompt_id"] for row in validation_rows}
    if train_ids & validation_ids:
        raise ValueError("training and validation prompt IDs overlap")
    heldout_ids = _heldout_ids(heldout_path)
    if (train_ids | validation_ids) & heldout_ids:
        raise ValueError("capture contains held-out acceptance prompt IDs")
    for split, actual in (("train", train_ids), ("validation", validation_ids)):
        if not actual.issubset(prompt_ids[split]):
            raise ValueError(f"{split} capture-row prompt IDs differ from copied prompt manifest")
    stats = {}
    for row in manifest.get("prompts", []):
        split = row["split"]
        prompt_id = row["prompt_id"]
        if split not in prompt_ids or prompt_id not in prompt_ids[split]:
            raise ValueError("per-prompt capture stats contain unexpected prompt")
        key = (split, prompt_id)
        if key in stats or row["trajectory"] not in ("ordinary", "head_w1a1"):
            raise ValueError("per-prompt capture stats duplicate a prompt or invalid trajectory")
        stats[key] = row["trajectory"]
    if stats:
        for split, rows in (("train", train_rows), ("validation", validation_rows)):
            for row in rows:
                if stats.get((split, row["prompt_id"])) != row["trajectory"]:
                    raise ValueError("captured row trajectory differs from prompt stats")
    provenance = {
        "capture_manifest_sha256": sha256_file(manifest_path),
        "artifact_sha256": artifact_hashes,
        "heldout_prompt_sha256": sha256_file(heldout_path),
        "prompt_manifest_sha256": manifest["prompt_manifest_sha256"],
        "config_sha256": manifest.get("config_sha256"),
        "model_manifest_sha256": manifest.get("model_manifest_sha256"),
        "rows": {"train": len(train_rows), "validation": len(validation_rows)},
        "prompt_counts": {"train": len(train_ids), "validation": len(validation_ids)},
        "trajectories": {
            split: dict(
                sorted(
                    {
                        trajectory: sum(row["trajectory"] == trajectory for row in rows)
                        for trajectory in {row["trajectory"] for row in rows}
                    }.items()
                )
            )
            for split, rows in (("train", train_rows), ("validation", validation_rows))
        },
    }
    return train, validation, teacher, provenance


def _check_forward(head: TrainableW1A1Head, inputs: torch.Tensor) -> None:
    with torch.no_grad():
        sample = inputs[: min(2, len(inputs))]
        exported = head.export_bf16_weight()
        actual = head(sample)
        reference = fake_binary_linear(sample, exported, config=head.config)
        if not torch.equal(actual, reference):
            raise RuntimeError("trainable W1A1 forward differs from inference reference")
        if not torch.equal(exported, head.latent_weight.detach().to(torch.bfloat16)):
            raise RuntimeError("BF16 head export differs from latent weight view")


def _batch_metrics(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> tuple[torch.Tensor, float, float]:
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise RuntimeError("nonfinite student or teacher logits")
    teacher_log_probs = F.log_softmax(teacher_logits.float(), dim=-1)
    student_log_probs = F.log_softmax(student_logits.float(), dim=-1)
    if not torch.isfinite(student_log_probs).all() or not torch.isfinite(teacher_log_probs).all():
        raise RuntimeError("nonfinite log probabilities")
    loss = F.kl_div(student_log_probs, teacher_log_probs.exp(), reduction="batchmean")
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite KL loss")
    k = min(10, teacher_logits.shape[-1])
    teacher_top = teacher_logits.float().topk(k, dim=-1).indices
    student_top = student_logits.float().topk(k, dim=-1).indices
    top1 = (teacher_top[:, 0] == student_top[:, 0]).float().sum().item()
    overlap = (teacher_top[:, :, None] == student_top[:, None, :]).any(dim=-1)
    top10 = overlap.float().sum().item() / k
    return loss, top1, top10


@torch.no_grad()
def validate(
    head: TrainableW1A1Head,
    teacher: torch.Tensor,
    inputs: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict:
    _check_forward(head, inputs[: min(batch_size, len(inputs))].to(device))
    sums = {"kl": 0.0, "top1_agreement": 0.0, "top10_overlap": 0.0}
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        teacher_logits = F.linear(batch, teacher)
        student_logits = head(batch)
        loss, top1, top10 = _batch_metrics(student_logits, teacher_logits)
        n = len(batch)
        sums["kl"] += float(loss) * n
        sums["top1_agreement"] += top1
        sums["top10_overlap"] += top10
    return {name: value / len(inputs) for name, value in sums.items()}


def _atomic_torch_save(payload: dict, path: Path) -> str:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def _checkpoint(
    head: TrainableW1A1Head,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: dict,
    provenance: dict,
    seed: int,
    order: torch.Tensor,
    cursor: int,
) -> dict:
    return {
        "step": step,
        "latent_weight": head.latent_weight.detach().cpu().clone(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if head.latent_weight.is_cuda else None,
        "python_rng_state": random.getstate(),
        "metrics": metrics,
        "provenance": provenance,
        "seed": seed,
        "sampler_order": order.clone(),
        "sampler_cursor": cursor,
        "w1a1": {
            "zero_sign": head.config.zero_sign,
            "weight_scale": head.config.weight_scale,
            "activation_scale": head.config.activation_scale,
            "ste_clip": head.ste_clip,
            "ste_eps": head.ste_eps,
        },
    }


def _restore_best(head: TrainableW1A1Head, path: Path, expected_provenance: dict) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["provenance"] != expected_provenance:
        raise RuntimeError("checkpoint capture provenance mismatch")
    with torch.no_grad():
        head.latent_weight.copy_(checkpoint["latent_weight"].to(head.latent_weight.device))
    return checkpoint


def _write_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _log_progress(path: Path, step: int, metrics: dict, training: dict | None = None) -> None:
    event = {"step": step, "validation": metrics}
    if training is not None:
        event["training"] = training
    with path.open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
        stream.flush()
    print(json.dumps(event, sort_keys=True), flush=True)


def train(args: argparse.Namespace) -> dict:
    if args.max_steps < 1 or not (0 < args.max_minutes <= 45):
        raise ValueError("training is bounded to positive steps and at most 45 minutes")
    if args.max_steps > 500 or args.validation_interval < 1 or args.patience < 1:
        raise ValueError("max steps may not exceed 500; interval and patience must be positive")
    if not (1 <= args.batch_size <= 128 and 1 <= args.validation_batch_size <= 128):
        raise ValueError("batch sizes must be between 1 and 128")
    if not (0 < args.lr <= 3e-4) or args.warmup_steps < 0:
        raise ValueError("learning rate or warmup is outside the bounded pilot")
    if args.vocab_size < 2 or args.hidden_size < 1:
        raise ValueError("invalid head dimensions")
    if not args.dry_run and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory must be empty for this bounded run")
    train_inputs, validation_inputs, frozen_weight, provenance = load_capture(
        args.capture_dir, args.hidden_size, args.vocab_size, args.heldout_prompts, args.config
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    head = TrainableW1A1Head(frozen_weight, W1A1Config()).to(device)
    teacher = frozen_weight.to(device)
    teacher.requires_grad_(False)
    parameters = list(head.parameters())
    if len(parameters) != 1 or parameters[0] is not head.latent_weight:
        raise RuntimeError("unexpected trainable parameters")
    _check_forward(head, train_inputs[: min(2, len(train_inputs))].to(device))
    initial = validate(head, teacher, validation_inputs, args.validation_batch_size, device)
    result = {"status": "dry_run" if args.dry_run else "running", "step0": initial, **provenance}
    if args.dry_run:
        return result
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW([head.latent_weight], lr=args.lr, weight_decay=0.0)
    order = torch.randperm(len(train_inputs))
    cursor = 0
    checkpoint_hashes = {}
    step0 = _checkpoint(head, optimizer, 0, initial, provenance, args.seed, order, cursor)
    checkpoint_hashes["step0.pt"] = _atomic_torch_save(step0, args.output_dir / "step0.pt")
    checkpoint_hashes["best.pt"] = _atomic_torch_save(step0, args.output_dir / "best.pt")
    _log_progress(args.output_dir / "progress.jsonl", 0, initial)
    best_kl = initial["kl"]
    best_step = 0
    no_improvement = 0
    history = [{"step": 0, "validation": initial}]
    last_metrics = initial
    steps_done = 0
    stop_reason = "max_steps"
    deadline = time.monotonic() + args.max_minutes * 60
    started = time.monotonic()
    for step in range(1, args.max_steps + 1):
        if time.monotonic() >= deadline:
            stop_reason = "time_limit"
            break
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(train_inputs))
            cursor = 0
        indices = order[cursor : cursor + args.batch_size]
        cursor += len(indices)
        batch = train_inputs[indices].to(device)
        warmup = min(1.0, step / args.warmup_steps) if args.warmup_steps else 1.0
        optimizer.param_groups[0]["lr"] = args.lr * warmup
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_logits = F.linear(batch, teacher)
        student_logits = head(batch)
        loss, top1, top10 = _batch_metrics(student_logits, teacher_logits)
        loss.backward()
        gradient = head.latent_weight.grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise RuntimeError("nonfinite or missing latent-weight gradient")
        gradient_norm = float(nn.utils.clip_grad_norm_([head.latent_weight], max_norm=1.0))
        if not math.isfinite(gradient_norm):
            raise RuntimeError("nonfinite gradient norm")
        optimizer.step()
        if not torch.isfinite(head.latent_weight).all():
            raise RuntimeError("nonfinite latent head weight")
        steps_done = step
        last_metrics = {
            "train_kl": float(loss.detach()),
            "train_top1_agreement": top1 / len(batch),
            "train_top10_overlap": top10 / len(batch),
            "gradient_norm_preclip": gradient_norm,
            "lr": optimizer.param_groups[0]["lr"],
        }
        if step % args.validation_interval == 0 or step == args.max_steps:
            metrics = validate(head, teacher, validation_inputs, args.validation_batch_size, device)
            history.append({"step": step, "validation": metrics, "training": last_metrics})
            _log_progress(args.output_dir / "progress.jsonl", step, metrics, last_metrics)
            if metrics["kl"] < best_kl:
                best_kl = metrics["kl"]
                best_step = step
                no_improvement = 0
                checkpoint_hashes["best.pt"] = _atomic_torch_save(
                    _checkpoint(
                        head, optimizer, step, metrics, provenance, args.seed, order, cursor
                    ),
                    args.output_dir / "best.pt",
                )
            else:
                no_improvement += 1
            if no_improvement >= args.patience:
                stop_reason = "early_stop"
                break
    checkpoint_hashes["last.pt"] = _atomic_torch_save(
        _checkpoint(
            head, optimizer, steps_done, last_metrics, provenance, args.seed, order, cursor
        ),
        args.output_dir / "last.pt",
    )
    best_checkpoint = _restore_best(head, args.output_dir / "best.pt", provenance)
    _check_forward(head, validation_inputs[: min(2, len(validation_inputs))].to(device))
    exported = head.export_bf16_weight().cpu()
    checkpoint_hashes["best-head.pt"] = _atomic_torch_save(
        {"weight": exported}, args.output_dir / "best-head.pt"
    )
    loaded_export = torch.load(
        args.output_dir / "best-head.pt", map_location="cpu", weights_only=True
    )
    if not torch.equal(loaded_export["weight"], exported):
        raise RuntimeError("BF16 inference export changed on disk")
    with torch.no_grad():
        sample = validation_inputs[: min(2, len(validation_inputs))].to(device)
        if not torch.equal(head(sample), fake_binary_linear(sample, exported.to(device))):
            raise RuntimeError("BF16 inference export differs from selected training forward")
    summary = {
        "status": "completed",
        "created_utc": datetime.now(UTC).isoformat(),
        "device": str(device),
        "torch_version": torch.__version__,
        "steps_done": steps_done,
        "best_step": best_step,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started,
        "step0": initial,
        "best_validation": best_checkpoint["metrics"],
        "history": history,
        "last_training": last_metrics,
        "config": {
            "lr": args.lr,
            "weight_decay": 0.0,
            "batch_size": args.batch_size,
            "validation_batch_size": args.validation_batch_size,
            "warmup_steps": args.warmup_steps,
            "max_steps": args.max_steps,
            "max_minutes": args.max_minutes,
            "validation_interval": args.validation_interval,
            "patience": args.patience,
            "gradient_clip_norm": 1.0,
            "seed": args.seed,
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "ste_clip": head.ste_clip,
            "ste_eps": head.ste_eps,
        },
        "checkpoint_sha256": checkpoint_hashes,
        "progress_sha256": sha256_file(args.output_dir / "progress.jsonl"),
        **provenance,
    }
    _write_json(summary, args.output_dir / "training-summary.json")
    return summary


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/pytorch_w1a1.toml",
    )
    parser.add_argument(
        "--heldout-prompts",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/acceptance_prompts.jsonl",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-minutes", type=float, default=45)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

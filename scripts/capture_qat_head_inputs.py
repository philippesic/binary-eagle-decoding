"""Capture normed EAGLE drafter-head inputs for the bounded head-QAT pilot.

Example on the supervised RTX 5080 host::

    python scripts/capture_qat_head_inputs.py \
        --model-manifest results/model-manifest.json --run-id qat-head-capture-01

Run ``generate_qat_head_prompts.py`` first. This captures ordinary and untrained
head-only W1A1 trajectories; it performs no optimization. The saved BF16 vectors
are exactly the tensors presented to the installed drafter ``lm_head`` module.
"""

import argparse
import json
import platform
import random
import re
import shutil
import sys
import tomllib
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_pytorch_w1a1 import (  # noqa: E402
    encode_prompt,
    git_revision,
    load_prompts,
    sha256_file,
    verify_model_snapshot,
)
from scripts.generate_qat_head_prompts import (  # noqa: E402
    content_hash,
    read_jsonl,
    validate_disjoint,
)

_MISSING = object()


def validate_prompt_manifest(directory: Path, heldout_path: Path) -> tuple[dict, dict]:
    """Check both frozen split hashes, counts, categories, and held-out isolation."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported QAT prompt manifest schema")
    if manifest["heldout_sha256"] != sha256_file(heldout_path):
        raise ValueError("held-out prompt manifest hash changed")
    if manifest["generator_sha256"] != sha256_file(ROOT / "scripts/generate_qat_head_prompts.py"):
        raise ValueError("QAT prompt generator differs from manifest")
    splits = {}
    for split, required, per_category in (("train", 96, 32), ("validation", 24, 8)):
        spec = manifest["splits"][split]
        path = directory / spec["path"]
        if path.resolve().parent != directory.resolve() or path.suffix != ".jsonl":
            raise ValueError(f"unsafe {split} prompt path")
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"{split} prompt file hash changed")
        rows = load_prompts(path)
        counts = {
            category: sum(row["category"] == category for row in rows)
            for category in ("prose", "code", "reasoning")
        }
        if (
            len(rows) != required
            or spec["prompts"] != required
            or any(count != per_category for count in counts.values())
            or spec["categories"] != counts
        ):
            raise ValueError(f"{split} prompt balance changed")
        if any(row.get("content_sha256") != content_hash(row["messages"]) for row in rows):
            raise ValueError(f"{split} prompt content hash changed")
        splits[split] = rows
    validate_disjoint(splits["train"] + splits["validation"], read_jsonl(heldout_path))
    return manifest, splits


def trajectory_for_index(index: int) -> str:
    """Alternating assignments give each category a 50/50 trajectory split."""
    return "ordinary" if index % 2 == 0 else "head_w1a1"


class HeadInputReservoir:
    """Sample individual BF16 rows uniformly within each prompt."""

    def __init__(self, cap: int, seed: int):
        if cap < 1:
            raise ValueError("per-prompt reservoir cap must be positive")
        self.cap = cap
        self.rng = random.Random(seed)
        self.rows = []
        self.seen = 0

    def add(self, tensor, metadata: dict) -> None:
        import torch

        if tensor.dtype != torch.bfloat16 or tensor.ndim != 2:
            raise ValueError("head inputs must be 2D BF16 rows")
        # Transfer one call at a time. Capture is an offline quality experiment,
        # so the synchronization overhead does not enter any latency result.
        cpu = tensor.detach().to(device="cpu", dtype=torch.bfloat16)
        for row_index, vector in enumerate(cpu):
            self.seen += 1
            slot = self.seen - 1 if self.seen <= self.cap else self.rng.randrange(self.seen)
            if slot >= self.cap:
                continue
            item = (vector.clone(), {**metadata, "row_in_call": row_index})
            if slot == len(self.rows):
                self.rows.append(item)
            else:
                self.rows[slot] = item


class HeadCapture:
    """Hook the *installed* drafter lm_head and label its topK call ordinals."""

    def __init__(self, drafter, per_prompt_cap: int, seed: int, expected_width: int):
        self.drafter = drafter
        self.per_prompt_cap = per_prompt_cap
        self.seed = seed
        self.expected_width = expected_width
        self._head_hook = None
        self._original_topk = None
        self._previous_topk_attr = _MISSING
        self._in_topk = False
        self._draft_call_index = -1
        self._head_call_index = 0
        self._prompt_id = None
        self._trajectory = None
        self.reservoir = None

    def __enter__(self):
        from torch import nn

        if not isinstance(self.drafter.lm_head, nn.Module):
            raise TypeError("drafter lm_head is not a module")
        self._original_topk = self.drafter.topK_genrate
        self._previous_topk_attr = self.drafter.__dict__.get("topK_genrate", _MISSING)
        self._head_hook = self.drafter.lm_head.register_forward_pre_hook(self._pre_hook)

        def tracked_topk(*args, **kwargs):
            if self._in_topk:
                raise RuntimeError("nested topK_genrate capture is unsupported")
            self._draft_call_index += 1
            self._head_call_index = 0
            self._in_topk = True
            try:
                return self._original_topk(*args, **kwargs)
            finally:
                self._in_topk = False

        self.drafter.topK_genrate = tracked_topk
        return self

    def __exit__(self, *_):
        if self._previous_topk_attr is _MISSING and "topK_genrate" in self.drafter.__dict__:
            del self.drafter.topK_genrate
        elif self._previous_topk_attr is not _MISSING:
            self.drafter.topK_genrate = self._previous_topk_attr
        if self._head_hook is not None:
            self._head_hook.remove()
        self._head_hook = None
        self._original_topk = None

    def start_prompt(self, prompt_id: str, trajectory: str, index: int) -> None:
        if trajectory not in ("ordinary", "head_w1a1"):
            raise ValueError("unknown trajectory")
        self._prompt_id = prompt_id
        self._trajectory = trajectory
        self._draft_call_index = -1
        self._head_call_index = 0
        self.reservoir = HeadInputReservoir(self.per_prompt_cap, self.seed + index)

    def finish_prompt(self) -> tuple[list, int, int]:
        if self.reservoir is None or not self.reservoir.rows:
            raise RuntimeError(f"no drafter-head inputs captured for {self._prompt_id}")
        rows = self.reservoir.rows
        seen = self.reservoir.seen
        calls = self._draft_call_index + 1
        self.reservoir = None
        self._prompt_id = None
        self._trajectory = None
        return rows, seen, calls

    def _pre_hook(self, module, args):
        if self._prompt_id is None:
            return
        if not self._in_topk:
            raise RuntimeError("lm_head was called outside topK_genrate during capture")
        if len(args) != 1:
            raise RuntimeError("unexpected lm_head input arity")
        input_tensor = args[0]
        if input_tensor.shape[-1] != self.expected_width:
            raise RuntimeError("unexpected drafter-head input width")
        call_index = self._head_call_index
        self._head_call_index += 1
        self.reservoir.add(
            input_tensor.reshape(-1, self.expected_width),
            {
                "prompt_id": self._prompt_id,
                "trajectory": self._trajectory,
                "draft_call_index": self._draft_call_index,
                "head_call_index": call_index,
                # Ordinal of lm_head invocations within topK_genrate. This is
                # the observable tree-depth index in the pinned runtime path.
                "tree_depth_index": call_index,
            },
        )


def save_split(path: Path, records: list, global_cap: int, seed: int) -> dict:
    import torch

    if not records:
        raise ValueError("cannot save empty capture")
    if len(records) > global_cap:
        rng = random.Random(seed)
        records = [records[index] for index in sorted(rng.sample(range(len(records)), global_cap))]
    inputs = torch.stack([row[0] for row in records]).contiguous()
    metadata = [row[1] for row in records]
    if inputs.dtype != torch.bfloat16:
        raise RuntimeError("capture lost BF16 dtype")
    torch.save({"inputs": inputs, "rows": metadata}, path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "rows": len(records),
        "shape": list(inputs.shape),
        "dtype": str(inputs.dtype),
    }


def capture(args) -> dict:
    config_path = args.config.resolve()
    project_root = config_path.parents[1]
    config = tomllib.loads(config_path.read_text())
    if config.get("schema_version") != 1 or config["evaluation"]["temperature"] != 0.0:
        raise ValueError("capture requires pinned greedy evaluation config")
    evaluation = config["evaluation"]
    heldout_path = project_root / evaluation["prompt_manifest"]
    if sha256_file(heldout_path) != evaluation["prompt_sha256"]:
        raise ValueError("held-out prompt hash differs from pinned config")
    prompt_manifest, splits = validate_prompt_manifest(args.prompt_dir, heldout_path)
    model_manifest = json.loads(args.model_manifest.read_text())
    if model_manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("model manifest was created with a different config")
    for role in ("target", "draft"):
        entry = model_manifest["models"][role]
        if (
            entry["repo"] != config["models"][f"{role}_repo"]
            or entry["revision"] != config["models"][f"{role}_revision"]
        ):
            raise ValueError(f"{role} model revision mismatch")
        verify_model_snapshot(project_root / config["models"][f"{role}_dir"], entry)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("invalid run ID")
    if args.per_prompt_cap < 1 or args.global_cap < 1:
        raise ValueError("reservoir caps must be positive")

    import torch
    import transformers

    from w1a1_eagle import W1A1Config, install_w1a1
    from w1a1_eagle.official_loader import load_official_eagle3

    if args.device == "cuda":
        if not torch.cuda.is_available() or "5080" not in torch.cuda.get_device_name(0):
            raise RuntimeError("capture requires the authorized RTX 5080 host")
        device_map = "cuda:0"
        gpu = torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32 = False
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple Metal is unavailable")
        device_map = "mps"
        gpu = platform.processor()
    else:
        raise ValueError("unsupported device")

    torch.manual_seed(evaluation["seed"])
    run_dir = project_root / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / "resolved-config.toml")
    shutil.copyfile(args.model_manifest, run_dir / "model-manifest.json")
    shutil.copyfile(args.prompt_dir / "manifest.json", run_dir / "prompt-manifest.json")
    for split in ("train", "validation"):
        shutil.copyfile(
            args.prompt_dir / prompt_manifest["splits"][split]["path"],
            run_dir / f"{split}-prompts.jsonl",
        )

    model = load_official_eagle3(
        project_root / config["models"]["target_dir"],
        project_root / config["models"]["draft_dir"],
        angelslim_revision=config["models"]["angelslim_revision"],
        total_token=evaluation["total_token"],
        depth=evaluation["depth"],
        top_k=evaluation["top_k"],
        threshold=evaluation["threshold"],
        target_load_kwargs={"dtype": torch.bfloat16, "device_map": device_map},
    )
    model.eval()
    if model.eagle_layer.total_tokens != evaluation["total_token"] - 1:
        raise RuntimeError("unexpected draft tree budget")
    device = next(model.base_model.parameters()).device
    ordinary_head = model.eagle_layer.lm_head
    if (
        not isinstance(ordinary_head, torch.nn.Linear)
        or ordinary_head.weight.dtype != torch.bfloat16
    ):
        raise RuntimeError("expected BF16 ordinary drafter head")
    teacher_weight = ordinary_head.weight.detach().to(device="cpu", dtype=torch.bfloat16).clone()
    if teacher_weight.ndim != 2 or teacher_weight.shape != (32000, 2560):
        raise RuntimeError(f"unexpected pinned head shape: {tuple(teacher_weight.shape)}")
    teacher_path = run_dir / "teacher-head.pt"
    torch.save({"weight": teacher_weight}, teacher_path)

    quant = config["quantization"]
    if quant.get("mode", "binary") != "binary":
        raise ValueError("head pilot requires the pinned W1A1 quantization config")
    handle = install_w1a1(
        model.eagle_layer,
        ["lm_head"],
        W1A1Config(quant["zero_sign"], quant["weight_scale"], quant["activation_scale"]),
        enabled=False,
        target=model.base_model,
    )
    if model.eagle_layer.lm_head is ordinary_head:
        raise RuntimeError("head hook would miss installed W1A1 wrapper")
    captures = {}
    prompt_stats = []
    try:
        with (
            torch.inference_mode(),
            HeadCapture(
                model.eagle_layer, args.per_prompt_cap, evaluation["seed"], teacher_weight.shape[1]
            ) as hook,
        ):
            for split, prompts in splits.items():
                records = []
                for index, prompt in enumerate(prompts):
                    trajectory = trajectory_for_index(index)
                    handle.set_enabled(trajectory == "head_w1a1")
                    hook.start_prompt(
                        prompt["id"], trajectory, index + (0 if split == "train" else 1000)
                    )
                    ids = encode_prompt(
                        model.tokenizer, prompt["messages"], evaluation["thinking_mode"], device
                    )
                    model.eagle_generate(
                        ids,
                        temperature=0.0,
                        max_new_tokens=evaluation["max_new_tokens"],
                        max_length=evaluation["max_length"],
                        log=True,
                    )
                    rows, seen, draft_calls = hook.finish_prompt()
                    records.extend(rows)
                    prompt_stats.append(
                        {
                            "split": split,
                            "prompt_id": prompt["id"],
                            "category": prompt["category"],
                            "trajectory": trajectory,
                            "input_rows_seen": seen,
                            "rows_retained": len(rows),
                            "draft_calls": draft_calls,
                        }
                    )
                captures[split] = save_split(
                    run_dir / f"{split}.pt",
                    records,
                    args.global_cap,
                    evaluation["seed"] + len(captures),
                )
    finally:
        handle.uninstall()
    if not torch.equal(ordinary_head.weight.detach().cpu(), teacher_weight):
        raise RuntimeError("original BF16 drafter head changed during capture")

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "project_revision": git_revision(project_root),
        "capture_script_sha256": sha256_file(Path(__file__)),
        "generator_sha256": prompt_manifest["generator_sha256"],
        "config_sha256": sha256_file(config_path),
        "prompt_manifest_sha256": sha256_file(args.prompt_dir / "manifest.json"),
        "heldout_prompt_sha256": sha256_file(heldout_path),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "model_files": {
            role: [
                {"path": item["path"], "sha256": item["sha256"]}
                for item in model_manifest["models"][role]["files"]
            ]
            for role in ("target", "draft")
        },
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "angelslim_package": version("angelslim"),
        "angelslim_revision": config["models"]["angelslim_revision"],
        "device": args.device,
        "gpu": gpu,
        "cuda_runtime": torch.version.cuda,
        "target_dtype": str(model.base_model.dtype),
        "draft_dtype": str(next(model.eagle_layer.parameters()).dtype),
        "settings": {
            "seed": evaluation["seed"],
            "per_prompt_cap": args.per_prompt_cap,
            "global_cap_per_split": args.global_cap,
            "trajectory_assignment": "alternating index within each split",
            "tree_depth_index": "zero-based lm_head call ordinal within topK_genrate",
            "evaluation": evaluation,
            "quantization": quant,
        },
        "teacher_head": {
            "path": teacher_path.name,
            "sha256": sha256_file(teacher_path),
            "shape": list(teacher_weight.shape),
            "dtype": str(teacher_weight.dtype),
        },
        "captures": captures,
        "prompts": prompt_stats,
    }
    (run_dir / "capture-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return {
        "run_dir": str(run_dir),
        "captures": captures,
        "capture_manifest_sha256": sha256_file(run_dir / "capture-manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pytorch_w1a1.toml")
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, default=ROOT / "data/qat-head-pilot")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("cuda", "mps"), default="cuda")
    parser.add_argument("--per-prompt-cap", type=int, default=256)
    parser.add_argument("--global-cap", type=int, default=32768)
    args = parser.parse_args()
    print(json.dumps(capture(args), indent=2))


if __name__ == "__main__":
    main()

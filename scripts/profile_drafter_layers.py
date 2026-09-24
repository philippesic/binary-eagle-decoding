"""Profile ordinary EAGLE-3 drafter linears on the actual CUDA generation path.

Example (inside a supervised RTX 5080 run)::

    python scripts/profile_drafter_layers.py --model-manifest results/model-manifest.json \
        --run-id drafter-profile-01 --prompt-id prose-01 --warmups 2 --repetitions 5

The output is ``results/<run-id>/layer-profile.json``. CUDA events bracket each
``topK_genrate`` call and each selected ``nn.Linear`` call on the current stream.
The residual includes non-linear graph work, tree selection, launch gaps, and
instrumentation overhead. It is not a native W1A1 speed estimate.
"""

import argparse
import json
import platform
import re
import shutil
import sys
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.evaluate_pytorch_w1a1 import (  # noqa: E402
    encode_prompt,
    git_revision,
    load_prompts,
    sha256_file,
    verify_model_snapshot,
)
from w1a1_eagle.adapter import GROUP_PATHS, DrafterStructureError  # noqa: E402
from w1a1_eagle.official_loader import load_official_eagle3  # noqa: E402

_MISSING = object()


def module_at(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part, None)
        if not isinstance(module, nn.Module):
            raise DrafterStructureError(f"missing drafter module {path!r}")
    return module


def eligible_linears(drafter: nn.Module, target: nn.Module | None = None) -> dict[str, nn.Linear]:
    """Validate the pinned direct-call graph before installing any hooks."""
    if getattr(getattr(drafter, "config", None), "pretraining_tp", None) != 1:
        raise DrafterStructureError("profiling requires drafter.config.pretraining_tp == 1")
    target_module_ids = {id(module) for module in target.modules()} if target else set()
    target_weight_ids = (
        {id(module.weight) for module in target.modules() if isinstance(module, nn.Linear)}
        if target
        else set()
    )
    result = {}
    for paths in GROUP_PATHS.values():
        for path in paths:
            module = module_at(drafter, path)
            if not isinstance(module, nn.Linear):
                raise DrafterStructureError(f"expected ordinary nn.Linear at {path!r}")
            if id(module) in target_module_ids or id(module.weight) in target_weight_ids:
                raise DrafterStructureError(f"target-owned linear at {path!r}")
            result[path] = module
    return result


def summarize_spans(spans: list[dict], draft_ms: list[float]) -> dict:
    """Pure reduction of synchronized event measurements; useful on CPU tests."""
    if not draft_ms or any(value < 0 for value in draft_ms):
        raise ValueError("draft timings must be nonempty and nonnegative")
    if any(row["elapsed_ms"] < 0 for row in spans):
        raise ValueError("linear timings must be nonnegative")
    by_path = defaultdict(lambda: {"calls": 0, "elapsed_ms": 0.0, "shapes": defaultdict(int)})
    by_group = {group: {"calls": 0, "elapsed_ms": 0.0} for group in GROUP_PATHS}
    path_group = {path: group for group, paths in GROUP_PATHS.items() for path in paths}
    for row in spans:
        path = row["path"]
        if path not in path_group:
            raise ValueError(f"unknown profiled path: {path}")
        item = by_path[path]
        item["calls"] += 1
        item["elapsed_ms"] += row["elapsed_ms"]
        shape_key = (
            tuple(row["input_shape"]),
            tuple(row["weight_shape"]),
            row["input_dtype"],
            row["weight_dtype"],
        )
        item["shapes"][shape_key] += 1
        group = by_group[path_group[path]]
        group["calls"] += 1
        group["elapsed_ms"] += row["elapsed_ms"]
    total_ms = sum(draft_ms)
    linear_ms = sum(item["elapsed_ms"] for item in by_group.values())
    groups = {
        name: {
            **item,
            "share_of_draft_event_time": item["elapsed_ms"] / total_ms,
            "binary_eligible_linear": True,
        }
        for name, item in by_group.items()
    }
    paths = {}
    for path, item in by_path.items():
        paths[path] = {
            "calls": item["calls"],
            "elapsed_ms": item["elapsed_ms"],
            "observed_shapes": [
                {
                    "input": list(key[0]),
                    "weight": list(key[1]),
                    "input_dtype": key[2],
                    "weight_dtype": key[3],
                    "calls": count,
                }
                for key, count in sorted(item["shapes"].items())
            ],
        }
    return {
        "draft_invocations": len(draft_ms),
        "draft_event_ms": total_ms,
        "draft_invocation_ms": draft_ms,
        "linear_event_ms": linear_ms,
        "binary_eligible_linear_share": linear_ms / total_ms,
        "remaining_graph_event_ms": total_ms - linear_ms,
        "remaining_graph_share": (total_ms - linear_ms) / total_ms,
        "groups": groups,
        "paths": paths,
    }


def aggregate_repetitions(repetitions: list[dict]) -> dict:
    """Combine measured repetitions using total event time as the denominator."""
    total = sum(item["draft_event_ms"] for item in repetitions)
    if not total:
        raise ValueError("repetitions have no measured draft time")
    groups = {}
    for group in GROUP_PATHS:
        elapsed = sum(item["groups"][group]["elapsed_ms"] for item in repetitions)
        calls = sum(item["groups"][group]["calls"] for item in repetitions)
        groups[group] = {
            "calls": calls,
            "elapsed_ms": elapsed,
            "share_of_draft_event_time": elapsed / total,
            "binary_eligible_linear": True,
        }
    linear = sum(item["elapsed_ms"] for item in groups.values())
    return {
        "draft_event_ms": total,
        "draft_invocations": sum(item["draft_invocations"] for item in repetitions),
        "linear_event_ms": linear,
        "binary_eligible_linear_share": linear / total,
        "remaining_graph_event_ms": total - linear,
        "remaining_graph_share": (total - linear) / total,
        "groups": groups,
    }


class DrafterEventProfiler:
    """Temporary hooks and a temporary method wrapper for one generation run.

    ``event_factory`` and ``synchronize`` can be substituted in CPU tests. The
    production caller supplies CUDA events and a device synchronization.
    """

    def __init__(self, drafter, target, event_factory, synchronize):
        self.drafter = drafter
        self.linears = eligible_linears(drafter, target)
        self.event_factory = event_factory
        self.synchronize = synchronize
        self._hooks = []
        self._pending = []
        self._draft_pending = []
        self._starts = defaultdict(list)
        self._original_topk = None
        self._previous_topk_attr = _MISSING
        self._topk_depth = 0

    def __enter__(self):
        self._original_topk = self.drafter.topK_genrate
        self._previous_topk_attr = self.drafter.__dict__.get("topK_genrate", _MISSING)
        try:
            for path, module in self.linears.items():
                self._hooks.append(module.register_forward_pre_hook(self._pre_hook(path)))
                self._hooks.append(module.register_forward_hook(self._post_hook(path)))

            def timed_topk(*args, **kwargs):
                start, end = self.event_factory(), self.event_factory()
                start.record()
                self._topk_depth += 1
                try:
                    return self._original_topk(*args, **kwargs)
                finally:
                    end.record()
                    self._topk_depth -= 1
                    self._draft_pending.append((start, end))

            self.drafter.topK_genrate = timed_topk
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self._previous_topk_attr is _MISSING and "topK_genrate" in self.drafter.__dict__:
            del self.drafter.topK_genrate
        elif self._previous_topk_attr is not _MISSING:
            self.drafter.topK_genrate = self._previous_topk_attr
        self._previous_topk_attr = _MISSING
        self._original_topk = None
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _pre_hook(self, path):
        def hook(module, args):
            if not self._topk_depth:
                return
            input_tensor = args[0]
            start = self.event_factory()
            start.record()
            self._starts[path].append(
                (
                    start,
                    {
                        "path": path,
                        "input_shape": list(input_tensor.shape),
                        "weight_shape": list(module.weight.shape),
                        "input_dtype": str(input_tensor.dtype),
                        "weight_dtype": str(module.weight.dtype),
                    },
                )
            )

        return hook

    def _post_hook(self, path):
        def hook(_module, _args, _output):
            if not self._topk_depth:
                return
            start, row = self._starts[path].pop()
            end = self.event_factory()
            end.record()
            self._pending.append((start, end, row))

        return hook

    def result(self):
        self.synchronize()
        if any(self._starts.values()):
            raise RuntimeError("an instrumented linear call did not complete")
        spans = []
        for start, end, row in self._pending:
            spans.append({**row, "elapsed_ms": start.elapsed_time(end)})
        draft_ms = [start.elapsed_time(end) for start, end in self._draft_pending]
        if not spans:
            raise RuntimeError("no eligible linear calls were observed")
        if not draft_ms:
            raise RuntimeError("no topK_genrate calls were observed")
        result = summarize_spans(spans, draft_ms)
        missing = set(self.linears) - set(result["paths"])
        if missing:
            raise RuntimeError(f"eligible linears were not called: {sorted(missing)}")
        return result


def run_profile(args):
    config_path = args.config.resolve()
    project_root = config_path.parents[1]
    config = tomllib.loads(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    evaluation = config["evaluation"]
    if evaluation["temperature"] != 0.0:
        raise ValueError("profiling requires greedy decoding")
    prompt_path = project_root / evaluation["prompt_manifest"]
    if sha256_file(prompt_path) != evaluation["prompt_sha256"]:
        raise ValueError("prompt manifest hash changed")
    prompts = load_prompts(prompt_path)
    prompt = next((item for item in prompts if item["id"] == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt ID {args.prompt_id!r} is absent from the manifest")
    if args.warmups < 0 or args.repetitions < 1:
        raise ValueError("warmups must be nonnegative and repetitions positive")
    if args.max_new_tokens < 1 or args.max_new_tokens > evaluation["max_new_tokens"]:
        raise ValueError("max-new-tokens must be within the fixed evaluation maximum")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        raise ValueError("run ID may contain only letters, numbers, underscores, and hyphens")
    manifest = json.loads(args.model_manifest.read_text())
    if manifest["config_sha256"] != sha256_file(config_path):
        raise ValueError("model manifest was created with a different config")
    for role in ("target", "draft"):
        entry = manifest["models"][role]
        if entry["repo"] != config["models"][f"{role}_repo"]:
            raise ValueError(f"{role} model repository mismatch")
        if entry["revision"] != config["models"][f"{role}_revision"]:
            raise ValueError(f"{role} model revision mismatch")
        verify_model_snapshot(project_root / config["models"][f"{role}_dir"], entry)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu_name = torch.cuda.get_device_name(0)
    if "5080" not in gpu_name:
        raise RuntimeError(f"expected the RTX 5080 experiment host, found {gpu_name}")

    run_dir = project_root / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / "resolved-config.toml")
    shutil.copyfile(prompt_path, run_dir / "prompts.jsonl")
    shutil.copyfile(args.model_manifest, run_dir / "model-manifest.json")
    torch.manual_seed(evaluation["seed"])
    models = config["models"]
    model = load_official_eagle3(
        project_root / models["target_dir"],
        project_root / models["draft_dir"],
        angelslim_revision=models["angelslim_revision"],
        total_token=evaluation["total_token"],
        depth=evaluation["depth"],
        top_k=evaluation["top_k"],
        threshold=evaluation["threshold"],
        target_load_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
    )
    model.eval()
    device = next(model.base_model.parameters()).device
    input_ids = encode_prompt(
        model.tokenizer, prompt["messages"], evaluation["thinking_mode"], device
    )
    generation = {
        "temperature": evaluation["temperature"],
        "max_new_tokens": args.max_new_tokens,
        "max_length": evaluation["max_length"],
        "log": True,
    }
    with torch.inference_mode():
        for _ in range(args.warmups):
            model.eagle_generate(input_ids, **generation)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        repetitions = []
        for _ in range(args.repetitions):
            with DrafterEventProfiler(
                model.eagle_layer,
                model.base_model,
                lambda: torch.cuda.Event(enable_timing=True),
                lambda: torch.cuda.synchronize(device),
            ) as profiler:
                output, _, _, accepted = model.eagle_generate(input_ids, **generation)
                result = profiler.result()
            result["generated_tokens"] = output.shape[1] - input_ids.shape[1]
            result["accepted_per_round"] = [int(value) for value in accepted]
            repetitions.append(result)
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command_argv": sys.argv,
        "project_revision": git_revision(project_root),
        "config_sha256": sha256_file(config_path),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "prompt_sha256": sha256_file(prompt_path),
        "prompt_id": prompt["id"],
        "prompt_tokens": input_ids.shape[1],
        "resolved_generation": generation,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "timing_method": "CUDA events on current stream; one device synchronization per repetition",
        "cuda_graph_capture_requested": False,
        "interpretation": (
            "Linear spans are inclusive PyTorch nn.Linear calls. Remaining graph is draft "
            "event time minus disjoint linear spans; it includes non-linear work, "
            "tree selection, host launch gaps, and event instrumentation overhead. "
            "No native binary throughput is inferred."
        ),
        "gpu": gpu_name,
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "angelslim_package": version("angelslim"),
        "python": sys.version,
        "platform": platform.platform(),
        "target_dtype": str(model.base_model.dtype),
        "draft_dtype": str(next(model.eagle_layer.parameters()).dtype),
        "peak_allocated_bytes_whole_generation": torch.cuda.max_memory_allocated(device),
        "aggregate": aggregate_repetitions(repetitions),
        "repetition_results": repetitions,
    }
    (run_dir / "layer-profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "repetitions": len(repetitions)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pytorch_w1a1.toml"))
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    run_profile(parser.parse_args())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""User-started continuous paired A8/A1 QAT; imports/estimates use CPU only."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from w1a1_eagle.continuous_qat import (  # noqa: E402
    ContinuousConfig,
    ContinuousTrainer,
    atomic_json,
    build_lanes,
    memory_estimate,
    sha256,
)
from w1a1_eagle.recurrent_provider import audit_provider_round  # noqa: E402


def load_config(path: Path) -> tuple[dict, ContinuousConfig]:
    spec = json.loads(path.read_text())
    if spec.get("schema") != "continuous_w1ax_experiment_v1":
        raise ValueError("unsupported continuous experiment config")
    kwargs = dict(spec["training"])
    kwargs["seeds"] = tuple(kwargs["seeds"])
    config = ContinuousConfig(**kwargs)
    if config.device != "cuda:0":
        raise ValueError(
            "production experiment is frozen to explicit cuda:0; CPU uses unit fixtures"
        )
    return spec, config


def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"experiment/GPU owner lock already held: {path}") from None
    stream.seek(0)
    stream.truncate()
    stream.write(str(os.getpid()) + "\n")
    stream.flush()
    return stream


def provider_pair(spec, config):
    module_name, separator, factory = spec["factory"].partition(":")
    if not separator or not module_name or not factory:
        raise ValueError("provider factory requires MODULE:FACTORY")
    create = getattr(importlib.import_module(module_name), factory)
    providers = [create(config.qat(bits), Path(spec["manifest"])) for bits in (8, 1)]
    if providers[0].source_metadata != providers[1].source_metadata:
        raise ValueError("A8/A1 providers must bind identical immutable data/resources")
    for provider in providers:
        if provider.training_eligible is not True or not provider.full_body_qat_eligible:
            raise ValueError("audited full-body eligibility required independently for A8 and A1")
    return providers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("start", "stop", "status", "estimate"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/continuous_w1ax.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stages-manifest", type=Path)
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.start:
        parser.error("--resume requires --start")
    if args.estimate:
        spec, config = load_config(args.config)
        estimate = memory_estimate(
            [tuple(shape) for shape in spec["model_shapes"]],
            frozen_bytes=spec["frozen_drafter_bytes"],
            graph_budget_bytes=spec["assumed_graph_budget_bytes"],
        )
        estimate["checkpoint_retention_estimated_bytes"] = (
            estimate["paired_resume_checkpoint_bytes"]
            + 2 * estimate["trainable_parameters_per_model"] * 4
        ) * (config.keep_checkpoints + 1)
        print(json.dumps(estimate, indent=2, sort_keys=True))
        return
    if args.run_dir is None:
        parser.error("--run-dir is required")
    run_dir = args.run_dir.resolve()
    if args.stop:
        if not (run_dir / "status.json").exists():
            parser.error("run directory has no status file")
        (run_dir / "STOP").touch()
        print("Graceful stop requested; wait for status=stopped and supervisor process exit.")
        return
    if args.status:
        print((run_dir / "status.json").read_text(), end="")
        return
    if not args.allow_cuda:
        parser.error("--start requires --allow-cuda; preparation itself never starts a GPU")
    spec, config = load_config(args.config)
    if args.stages_manifest:
        spec["stages"] = json.loads(args.stages_manifest.read_text())
    if args.development_manifest:
        spec["development"] = json.loads(args.development_manifest.read_text())
    if not isinstance(spec.get("stages"), dict):
        parser.error("pinned stages required: use --stages-manifest from CPU prepare-config")
    if args.resume and not (run_dir / "latest.json").exists():
        parser.error("--resume requires an existing paired checkpoint")
    if not args.resume and (run_dir / "status.json").exists():
        parser.error(
            "existing run requires --resume; choose a new run directory for a new experiment"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    # All paired training/capture/development work has one same-host GPU owner.
    # OS lock release is crash-safe; no active process is guessed or killed.
    run_lock = lock(run_dir / ".owner.lock")
    gpu_lock = lock(Path.home() / ".cache/binary-eagle-decoding/cuda-0.owner.lock")
    try:
        resolved_path = run_dir / "resolved_config.json"
        if resolved_path.exists():
            original = json.loads(resolved_path.read_text())
            comparison = json.loads(json.dumps(spec))
            for key in ("max_steps", "max_tokens", "max_seconds", "max_epochs"):
                original["training"].pop(key, None)
                comparison["training"].pop(key, None)
            if original != comparison:
                raise ValueError("resume changes immutable stages/development/source configuration")
            atomic_json(run_dir / f"resume-request-{time.time_ns()}.json", spec)
        else:
            atomic_json(resolved_path, spec)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("manual start requires declared CUDA hardware")
        properties = torch.cuda.get_device_properties(config.device)
        observed = {
            "device_name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_total_bytes": properties.total_memory,
            "resolved_config_sha256": sha256(resolved_path),
        }
        expected = spec["hardware"]
        if (
            observed["device_name"] != expected["device_name"]
            or observed["compute_capability"] != expected["compute_capability"]
        ):
            raise RuntimeError("manual start hardware differs from frozen RTX5080 SM120 contract")
        atomic_json(run_dir / "runtime_environment.json", observed)
        atomic_json(
            run_dir / "status.json",
            {
                "schema": "continuous_joint_w1ax_v1",
                "status": "preparing",
                "heartbeat_unix": time.time(),
                "pid": os.getpid(),
                "models": {},
                "phase": "capture_audit_readiness",
            },
        )
        from w1ax_continuous_stages import run_stages

        # Concrete pinned native capture -> v2 audit -> A8/A1 readiness gates,
        # with no optimization while either contract remains ineligible.
        resolved = run_stages(spec["stages"], run_dir)
        provider_spec = dict(spec["provider"])
        provider_spec["manifest"] = str(resolved)
        providers = provider_pair(provider_spec, config)
        provider = providers[0]
        del providers  # Release the independently audited A1 provider payload.
        minimum = spec["coverage"]
        if len(provider.allowed_prompt_ids) < minimum["min_unique_train_prompts"]:
            raise ValueError("teacher data has too few independent train prompts for declared tier")
        # Streaming audit/coverage count does not materialize all tensors. Keep
        # only two challenging representative rounds for manual CUDA smoke.
        count = 0
        longest = deepest = None
        observed_prompts = set()
        for batch in provider.rounds():
            audit = audit_provider_round(batch, provider)
            count += sum(audit.ce_mask)
            observed_prompts.add(batch.anchor.prompt_id)
            if len(batch.prefix_token_ids) > config.max_prefix_tokens:
                raise ValueError("captured prefix exceeds fixed memory-safe training limit")
            if not any(audit.ce_mask[1:]):
                continue
            if longest is None or len(batch.prefix_token_ids) > len(longest.prefix_token_ids):
                longest = batch
            if deepest is None or len(batch.rows) > len(deepest.rows):
                deepest = batch
        if (
            count < minimum["min_unique_supervised_rows"]
            or len(observed_prompts) < minimum["min_unique_train_prompts"]
        ):
            raise ValueError("audited supervised teacher coverage below declared tier")
        if longest is None or deepest is None:
            raise ValueError("no native teacher round supports later-position gradient smoke")
        atomic_json(
            run_dir / "teacher_coverage.json",
            {
                "unique_train_prompts": len(observed_prompts),
                "unique_supervised_rows": count,
                "source": provider.source_metadata,
                "smoke_longest_prefix": len(longest.prefix_token_ids),
                "smoke_max_depth": len(deepest.rows),
            },
        )
        lanes = build_lanes(provider, config)
        evaluator = None
        development = spec.get("development")
        if development is None or development == {"from_stages": True}:
            development = json.loads((run_dir / "stages/development.json").read_text())
        if development is not None:
            from w1ax_continuous_stages import evaluate_development

            def evaluator(checkpoint, lanes):
                return evaluate_development(development, Path(checkpoint["path"]).parent, run_dir)

        trainer = ContinuousTrainer(
            provider, lanes, config, run_dir, development_evaluator=evaluator
        )
        if args.resume:
            trainer.resume()
            (run_dir / "STOP").unlink(missing_ok=True)
        trainer.status("smoke")
        trainer.smoke(longest)
        trainer.smoke(deepest)
        trainer.save()
        trainer.run()
    except BaseException as error:
        # Includes capture/readiness failures before trainer construction.
        if "trainer" not in locals():
            atomic_json(
                run_dir / "status.json",
                {
                    "schema": "continuous_joint_w1ax_v1",
                    "status": "failed",
                    "heartbeat_unix": time.time(),
                    "pid": os.getpid(),
                    "models": {},
                    "error": f"{type(error).__name__}: {error}",
                },
            )
        elif json.loads((run_dir / "status.json").read_text()).get("status") != "failed":
            trainer.status("failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        gpu_lock.close()
        run_lock.close()


if __name__ == "__main__":
    main()

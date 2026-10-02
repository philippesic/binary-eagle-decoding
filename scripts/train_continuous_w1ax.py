#!/usr/bin/env python3
"""User-started continuous paired A8/A1 QAT; imports/estimates use CPU only."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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
from w1a1_eagle.continuous_runtime import training_runtime_identity  # noqa: E402
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


def record_runtime_observation(run_dir: Path, observed: dict) -> None:
    """Preserve startup provenance; record and validate each resume observation."""
    original_path = run_dir / "runtime_environment.json"
    if original_path.exists():
        atomic_json(run_dir / f"runtime-resume-observation-{time.time_ns()}.json", observed)
        original = json.loads(original_path.read_text())
        for field in (
            "device_name",
            "compute_capability",
            "torch_version",
            "cuda_version",
            "training_runtime",
        ):
            if original.get(field) != observed.get(field):
                raise ValueError("resume changes recorded hardware/CUDA/training runtime identity")
    else:
        atomic_json(original_path, observed)


def resume_kind(run_dir: Path) -> str:
    """Choose durable checkpoint recovery or an explicitly user-retried prep."""
    if (run_dir / "latest.json").exists() or any(
        (run_dir / "checkpoints").glob("step-*/manifest.json")
    ):
        return "checkpoint"
    if not (run_dir / "resolved_config.json").is_file() or not (run_dir / "status.json").is_file():
        raise ValueError("preparation resume requires original resolved config and status")
    status = json.loads((run_dir / "status.json").read_text())
    if status.get("schema") != "continuous_joint_w1ax_v1":
        raise ValueError("preparation resume has an unsupported status contract")
    models = status.get("models")
    if not isinstance(models, dict) or any(
        not isinstance(item, dict) or item.get("step") != 0 for item in models.values()
    ):
        raise ValueError("optimization evidence exists without a complete checkpoint")
    for path in run_dir.glob("metrics.jsonl*"):
        with path.open() as stream:
            for line in stream:
                if line.strip() and json.loads(line).get("step", 0) > 0:
                    raise ValueError("optimization log exists without a complete checkpoint")
    return "preparation"


def require_preparation_resume(run_dir: Path) -> None:
    """Reject recorded optimizer progress before preparation touches CUDA/data."""
    paths = [run_dir / "status.json", run_dir / "latest.json"]
    paths.extend((run_dir / "checkpoints").glob("step-*/manifest.json"))
    for path in paths:
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        step = record.get("step", 0) if path.name == "status.json" else record.get("step")
        models = record.get("models", {})
        if (
            type(step) is not int
            or step != 0
            or record.get("optimization_started") is True
            or not isinstance(models, dict)
            or any(
                not isinstance(model, dict)
                or type(model.get("step")) is not int
                or model["step"] != 0
                for model in models.values()
            )
        ):
            raise ValueError(f"--prepare-only requires zero optimizer progress: {path}")
    for path in run_dir.glob("metrics.jsonl*"):
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                step = json.loads(line).get("step", 0)
                if type(step) is not int or step != 0:
                    raise ValueError(f"--prepare-only found optimization log: {path}")


def require_zero_optimizer_progress(trainer) -> None:
    """Require zero counters and untouched Adam/SGD moment buffers."""
    if (
        type(trainer.step) is not int
        or trainer.step != 0
        or set(trainer.metrics) != {"A8", "A1"}
        or any(
            type(model.get("step")) is not int or model["step"] != 0
            for model in trainer.metrics.values()
        )
    ):
        raise ValueError("--prepare-only requires global and A8/A1 steps to be zero")
    for lane in trainer.lanes:
        for state in lane.optimizer.state.values():
            step = state.get("step", 0)
            if hasattr(step, "item"):
                step = step.item()
            if step != 0:
                raise ValueError(f"--prepare-only found {lane.name} optimizer progress")
            for name, value in state.items():
                if hasattr(value, "is_floating_point"):
                    if bool((value != 0).any()):
                        raise ValueError(
                            f"--prepare-only found {lane.name} nonzero optimizer {name}"
                        )
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    if value != 0:
                        raise ValueError(
                            f"--prepare-only found {lane.name} nonzero optimizer {name}"
                        )
                else:
                    raise ValueError(f"--prepare-only found unsupported optimizer state {name}")


def publish_preparation_ready(trainer, run_dir: Path) -> None:
    """Publish only after all ordinary gates, paired backward smoke and save."""
    require_zero_optimizer_progress(trainer)
    if not trainer.smoke_passed or not trainer.checkpoint or trainer.checkpoint["step"] != 0:
        raise ValueError("preparation requires passed paired smoke and checkpoint zero")
    resources = trainer.resources()
    report_path = run_dir / "preparation-ready.json"
    atomic_json(
        report_path,
        {
            "schema": "continuous_w1ax_preparation_ready_v1",
            "preparation_complete": True,
            "stop_reason": "prepare_only",
            "optimization_started": False,
            "heartbeat_unix": time.time(),
            "step": trainer.step,
            "models": trainer.metrics,
            "source_sha256": trainer.source,
            "training_runtime": trainer.runtime_identity,
            "checkpoint": trainer.checkpoint,
            "teacher_coverage": json.loads((run_dir / "teacher_coverage.json").read_text()),
            "dual_smoke_sha256": sha256(run_dir / "dual_smoke.json"),
            "resources": resources,
        },
    )
    trainer.status(
        "stopped",
        preparation_complete=True,
        stop_reason="prepare_only",
        optimization_started=False,
        preparation_report=str(report_path),
        preparation_report_sha256=sha256(report_path),
        **resources,
    )


def prepared_digest(value: dict) -> str:
    """Same full-provider identity used by the trainer and native actor collector."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def prepared_corpus_inputs(spec: dict, run_dir: Path, prepared_dir: Path, ready_sha: str):
    """Authenticate a completed preparation without loading its optimizer/model state.

    All writes belong to the new run. The old checkpoint is evidence only; the
    new trainer builds independent lanes and performs its own smoke and save.
    """
    if (not isinstance(ready_sha, str) or len(ready_sha) != 64
            or any(c not in "0123456789abcdef" for c in ready_sha)):
        raise ValueError("prepared completion receipt requires a pinned SHA256")
    prepared_dir, run_dir = prepared_dir.resolve(), run_dir.resolve()
    if (prepared_dir == run_dir or prepared_dir in run_dir.parents
            or run_dir in prepared_dir.parents):
        raise ValueError("prepared and new run directories must not overlap")
    artifacts = {}

    def pin(path, expected=None):
        path = Path(path).resolve()
        if path == run_dir or run_dir in path.parents:
            raise ValueError("prepared input overlaps new run output")
        actual = sha256(path)
        if expected is not None and actual != expected:
            raise ValueError(f"prepared artifact SHA256 mismatch: {path}")
        artifacts[str(path)] = actual
        return path

    def read(path, expected=None):
        return json.loads(pin(path, expected).read_text())

    ready = read(prepared_dir / "preparation-ready.json", ready_sha)
    if ready.get("schema") != "continuous_w1ax_preparation_ready_v1":
        raise ValueError("unsupported prepared completion receipt")
    status = read(prepared_dir / "status.json")
    for record in (ready, status):
        if (
            record.get("preparation_complete") is not True
            or record.get("optimization_started") is not False
            or record.get("stop_reason") != "prepare_only"
            or type(record.get("step")) is not int or record["step"] != 0
            or set(record.get("models", {})) != {"A8", "A1"}
            or any(type(m.get("step")) is not int or m["step"] != 0
                   for m in record["models"].values())
        ):
            raise ValueError("prepared corpus requires complete zero-update A8/A1 preparation")
    if (
        status.get("schema") != "continuous_joint_w1ax_v1"
        or status.get("status") != "stopped"
        or status.get("preparation_report_sha256") != ready_sha
        or Path(status.get("preparation_report", "")).resolve()
           != prepared_dir / "preparation-ready.json"
        or status.get("source_sha256") != ready.get("source_sha256")
        or status.get("training_runtime") != ready.get("training_runtime")
        or status.get("models") != ready.get("models")
    ):
        raise ValueError("prepared stopped status does not join completion receipt")
    require_preparation_resume(prepared_dir)
    old = read(prepared_dir / "resolved_config.json")
    # Training controls may change only through the ordinary current-source
    # trainer gates; frozen corpus, model, coverage and development declarations may not.
    def frozen_declarations(value):
        declarations = {k: v for k, v in value.items()
                        if k not in {"training", "preparation_note", "prepared_corpus"}}
        # These two fields describe the new actor/profiling proof. They do not
        # change capture ancestry or enable training controls; ordinary current
        # runtime/model/receipt gates still bind the new training configuration.
        if "optimization_profile" in declarations:
            profile = declarations.pop("optimization_profile")
            if not isinstance(profile, str) or not profile.strip():
                raise ValueError("prepared reuse requires a named optimization profile")
        if "native" in declarations:
            if not isinstance(declarations["native"], dict):
                raise ValueError("prepared reuse requires native actor metadata")
            native = dict(declarations["native"])
            if "expected_commit" in native:
                commit = native.pop("expected_commit")
                if (not isinstance(commit, str) or len(commit) != 40
                        or any(c not in "0123456789abcdef" for c in commit)):
                    raise ValueError("prepared reuse native.expected_commit requires full SHA")
            if native:
                declarations["native"] = native
            else:
                declarations.pop("native")
        return declarations

    if frozen_declarations(old) != frozen_declarations(spec):
        raise ValueError("prepared reuse changes immutable stages/development/source configuration")
    runtime = read(prepared_dir / "runtime_environment.json")
    if runtime.get("training_runtime") != ready.get("training_runtime"):
        raise ValueError("prepared runtime does not join completion receipt")
    checkpoint = ready.get("checkpoint", {})
    if (
        type(checkpoint.get("step")) is not int or checkpoint["step"] != 0
        or status.get("checkpoint") != checkpoint
        or read(prepared_dir / "latest.json") != checkpoint
    ):
        raise ValueError("prepared checkpoint zero does not join receipt/status/latest")
    checkpoint_path = Path(checkpoint["path"]).resolve()
    if prepared_dir not in checkpoint_path.parents:
        raise ValueError("prepared checkpoint must belong to original run")
    pin(checkpoint_path, checkpoint["sha256"])
    manifest = read(checkpoint_path.parent / "manifest.json")
    if (
        manifest.get("schema") != "continuous_joint_w1ax_v1"
        or type(manifest.get("step")) is not int or manifest["step"] != 0
        or manifest.get("sha256") != checkpoint["sha256"]
        or manifest.get("source_sha256") != ready.get("source_sha256")
        or manifest.get("training_runtime") != ready.get("training_runtime")
        or any(type(manifest.get(k)) is not int or manifest[k] != 0 for k in ("epoch", "cursor"))
        or manifest.get("optimizer_rng_cursor_exact") is not True
        or set(manifest.get("exports", {})) != {"A8", "A1"}
    ):
        raise ValueError("prepared checkpoint manifest does not join receipt")
    from dataclasses import asdict

    from w1a1_eagle.continuous_qat import immutable_config

    _, old_config = load_config(prepared_dir / "resolved_config.json")
    if manifest.get("immutable_config") != immutable_config(asdict(old_config)):
        raise ValueError("prepared checkpoint changes original training configuration")
    for lane, exports in manifest["exports"].items():
        if set(exports) != {"joint.npz", "joint.json"}:
            raise ValueError("prepared checkpoint export inventory differs")
        for filename, digest in exports.items():
            pin(checkpoint_path.parent / lane / filename, digest)
    smoke = read(prepared_dir / "dual_smoke.json", ready["dual_smoke_sha256"])
    if not {"A8", "A1"} <= set(smoke):
        raise ValueError("prepared paired smoke is incomplete")
    for lane in ("A8", "A1"):
        for name in ("loss", "later_state_gradient_norm",
                     "later_k_gradient_norm", "later_v_gradient_norm"):
            value = smoke[lane].get(name)
            if (type(value) not in (int, float) or not -float("inf") < value < float("inf")
                    or (name != "loss" and value <= 0)):
                raise ValueError(
                    "prepared paired smoke lacks finite loss/attached state/K/V gradients"
                )
    coverage = read(prepared_dir / "teacher_coverage.json")
    if (coverage != ready.get("teacher_coverage")
            or prepared_digest(coverage.get("source", {})) != ready.get("source_sha256")):
        raise ValueError("prepared coverage/provider source digest does not join receipt")

    stage_dir = prepared_dir / "stages"
    readiness_path = stage_dir / "readiness.json"
    readiness_record = {"path": str(readiness_path), "sha256": sha256(readiness_path)}
    common_keys = ("target_gguf", "candidate_d_gguf", "base_draft_gguf",
                   "absolute_d2t", "model_snapshot_manifest")
    common = {key: spec["stages"]["sources"]["sha256"][key] for key in common_keys}
    from w1ax_continuous_stages import validate_readiness

    for bits in (8, 1):
        validate_readiness(readiness_record, activation_bits=bits, common_hashes=common)
    readiness = read(readiness_path)
    if (readiness.get("native_binary_sha256") != spec["stages"]["sources"]["sha256"]["binary"]
            or readiness.get("native_runtime") != spec["stages"]["sources"].get("native_runtime")):
        raise ValueError("prepared readiness changes frozen native source/runtime")
    # Record every attached metadata artifact, preserving full train/development
    # shard identity. Payload/model ancestry is independently audited by each provider.
    captures = spec["stages"]["captures"]
    capture_hashes = []
    full_records = {}
    development_prompt_ids = {}
    for split, filename in (("train", "train-providers.json"),
                            ("development", "development-providers.json")):
        provider_spec = read(stage_dir / filename)
        planned = [c for c in captures if c["split"] == split]
        records = provider_spec.get("shards", [])
        if (
            provider_spec.get("schema") != "w1ax_streaming_train_v2"
            or provider_spec.get("split") != split
            or provider_spec.get("training_eligible") is not (split == "train")
            or provider_spec.get("continuous_readiness") != readiness_record
            or not planned or len(records) != len(planned)
        ):
            raise ValueError("prepared provider is not the complete frozen capture plan")
        full_records[split] = records
        for ordinal, (record, capture) in enumerate(zip(records, planned)):
            child = read(record["provider_manifest"], record["provider_manifest_sha256"])
            if (
                record.get("ordinal") != ordinal
                or child.get("schema") != "w1ax_native_train_provider_v2"
                or child.get("split") != split
                or child.get("training_eligible") is not (split == "train")
                or child.get("prompt_count") != capture["prompt_count"]
                or child.get("sha256", {}).get("prompts") != capture["prompts_sha256"]
                or child.get("continuous_readiness") != readiness_record
                or any(child["sha256"].get(k) != common[k] for k in common_keys)
            ):
                raise ValueError("prepared shard changes frozen prompts/model/map/readiness")
            prompt_path = pin(child["paths"]["prompts"], capture["prompts_sha256"])
            if split == "development":
                with prompt_path.open() as stream:
                    development_prompt_ids[record["provider_manifest"]] = {
                        json.loads(line)["id"] for line in stream if line.strip()
                    }
            pin(child["paths"]["capture_manifest"], child["sha256"]["capture_manifest"])
            capture_hashes.append(child["sha256"]["capture_manifest"])
    if sorted(capture_hashes) != sorted(readiness.get("teacher_capture_manifest_sha256", [])):
        raise ValueError("prepared readiness does not bind every frozen capture")
    train_path = stage_dir / "train-providers.json"
    source = coverage["source"]
    if (source.get("execution_manifest_sha256") != sha256(train_path)
            or source.get("common_source_sha256") != common):
        raise ValueError("prepared full provider identity differs from teacher coverage")
    development = read(stage_dir / "development.json")
    if (
        development.get("schema") != "w1ax_continuous_development_v1"
        or development.get("split") != "development"
        or development.get("native_prompts") != spec["stages"]["development_prompts"]
        or development.get("native_prompts_sha256") != spec["stages"]["development_prompts_sha256"]
        or development.get("full_pool_prompt_count")
           != sum(c["prompt_count"] for c in captures if c["split"] == "development")
        or development.get("sources", {}).get("sha256") != spec["stages"]["sources"]["sha256"]
        or development.get("sources", {}).get("native_runtime")
           != spec["stages"]["sources"].get("native_runtime")
    ):
        raise ValueError("prepared development changes frozen inputs")
    pin(development["native_prompts"], development["native_prompts_sha256"])
    pool = development["full_pool_manifest"]
    if Path(pool["path"]).resolve() != stage_dir / "development-providers.json":
        raise ValueError("prepared development pool differs from frozen providers")
    pin(pool["path"], pool["sha256"])
    subset = read(development["providers_manifest"])
    with Path(development["native_prompts"]).open() as stream:
        native_ids = {json.loads(line)["id"] for line in stream if line.strip()}
    pool_ids = set().union(*development_prompt_ids.values())
    if not native_ids or not native_ids <= pool_ids:
        raise ValueError("prepared development prompts are absent from frozen capture pool")
    expected_subset = []
    for record in full_records["development"]:
        if development_prompt_ids[record["provider_manifest"]].intersection(native_ids):
            expected_subset.append({**record, "ordinal": len(expected_subset)})
    if (
        subset.get("schema") != "w1ax_streaming_train_v2"
        or subset.get("split") != "development"
        or subset.get("training_eligible") is not False
        or subset.get("continuous_readiness") != readiness_record
        or subset.get("shards") != expected_subset
    ):
        raise ValueError("prepared development subset has unmatched shard ancestry")
    binding = {
        "schema": "continuous_w1ax_prepared_corpus_v1",
        "prepared_run_dir": str(prepared_dir),
        "prepared_ready_sha256": ready_sha,
        "source_sha256": ready["source_sha256"],
        "artifacts": artifacts,
        "provider_manifest": str(train_path),
        "development_manifest": str(stage_dir / "development.json"),
        "checkpoint_policy": "original_zero_checkpoint_evidence_only_new_model_smoke_and_save",
    }
    return binding, train_path, development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("start", "stop", "status", "estimate"):
        action.add_argument("--" + name, action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/continuous_w1ax.json")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="with --start, finish capture/gates/paired backward smoke/checkpoint zero then exit",
    )
    parser.add_argument("--prepared-run-dir", type=Path,
                        help="reuse immutable inputs from a completed zero-update preparation")
    parser.add_argument("--prepared-ready-sha256",
                        help="required exact SHA256 of original preparation-ready.json")
    parser.add_argument("--stages-manifest", type=Path)
    parser.add_argument("--development-manifest", type=Path)
    parser.add_argument("--allow-cuda", action="store_true")
    args = parser.parse_args()
    if args.resume and not args.start:
        parser.error("--resume requires --start")
    if args.prepare_only and not args.start:
        parser.error("--prepare-only requires --start")
    if bool(args.prepared_run_dir) != bool(args.prepared_ready_sha256):
        parser.error("prepared reuse requires both --prepared-run-dir and --prepared-ready-sha256")
    if args.prepared_run_dir and not args.start:
        parser.error("prepared reuse requires --start")
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
    prepared = None
    prepared_development = None
    saved_binding_path = run_dir / "prepared-corpus.json"
    if args.resume and saved_binding_path.exists():
        saved_binding = json.loads(saved_binding_path.read_text())
        if not args.prepared_run_dir:
            args.prepared_run_dir = Path(saved_binding["prepared_run_dir"])
            args.prepared_ready_sha256 = saved_binding["prepared_ready_sha256"]
    elif args.resume and (run_dir / "resolved_config.json").exists():
        if json.loads((run_dir / "resolved_config.json").read_text()).get("prepared_corpus"):
            parser.error("resume requires retained prepared corpus binding")
    if args.prepared_run_dir:
        prepared, prepared_provider, prepared_development = prepared_corpus_inputs(
            spec, run_dir, args.prepared_run_dir, args.prepared_ready_sha256
        )
        if args.resume:
            if (not saved_binding_path.exists()
                    or json.loads(saved_binding_path.read_text()) != prepared):
                parser.error("resume changes immutable prepared corpus binding")
        elif run_dir.exists() and any(run_dir.iterdir()):
            parser.error("prepared reuse requires a new empty run directory")
        spec["prepared_corpus"] = {
            "run_dir": str(args.prepared_run_dir.resolve()),
            "ready_sha256": args.prepared_ready_sha256,
            "binding_sha256": prepared_digest(prepared),
        }
    recovery = None
    if args.resume:
        try:
            recovery = resume_kind(run_dir)
            if args.prepare_only:
                require_preparation_resume(run_dir)
        except ValueError as error:
            parser.error(str(error))
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
        if prepared is not None and not saved_binding_path.exists():
            atomic_json(saved_binding_path, prepared)
        if args.resume:
            (run_dir / "STOP").unlink(missing_ok=True)
        import torch

        from w1a1_eagle.qat_readiness import optimization_requires_receipt

        if optimization_requires_receipt(config):
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
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
            "training_runtime": training_runtime_identity(config.device),
        }
        expected = spec["hardware"]
        if (
            observed["device_name"] != expected["device_name"]
            or observed["compute_capability"] != expected["compute_capability"]
        ):
            raise RuntimeError("manual start hardware differs from frozen RTX5080 SM120 contract")
        record_runtime_observation(run_dir, observed)
        atomic_json(
            run_dir / "status.json",
            {
                "schema": "continuous_joint_w1ax_v1",
                "status": "preparing",
                "heartbeat_unix": time.time(),
                "pid": os.getpid(),
                "models": {},
                "phase": "capture_audit_readiness",
                "optimization_started": False,
            },
        )
        from w1ax_continuous_stages import run_stages

        # Capture new inputs or authenticate the completed immutable preparation.
        # Each provider still independently audits full-body eligibility below.
        resolved = (prepared_provider if prepared is not None
                    else run_stages(spec["stages"], run_dir))
        provider_spec = dict(spec["provider"])
        provider_spec["manifest"] = str(resolved)
        providers = provider_pair(provider_spec, config)
        provider = providers[0]
        if (prepared is not None
                and prepared_digest(provider.source_metadata) != prepared["source_sha256"]):
            raise ValueError(
                "independently audited full provider source differs from prepared binding"
            )
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
        lanes = build_lanes(provider, config, run_dir)
        evaluator = None
        development = spec.get("development")
        if development is None or development == {"from_stages": True}:
            development = (prepared_development if prepared is not None else
                           json.loads((run_dir / "stages/development.json").read_text()))
        if development is not None:
            from w1ax_continuous_stages import evaluate_development

            def evaluator(checkpoint, lanes):
                return evaluate_development(development, Path(checkpoint["path"]).parent, run_dir)

        trainer = ContinuousTrainer(
            provider, lanes, config, run_dir, development_evaluator=evaluator
        )
        if recovery == "checkpoint":
            trainer.resume()
        if args.prepare_only:
            require_zero_optimizer_progress(trainer)
        trainer.status("smoke")
        trainer.smoke(longest)
        trainer.smoke(deepest)
        trainer.save()
        if args.prepare_only:
            publish_preparation_ready(trainer, run_dir)
            return
        trainer.run()
    except InterruptedError as error:
        # Native stage cancellation only raises this after escaped-server
        # process-group cleanup for intentional signals/STOP requests.
        if "trainer" in locals():
            trainer.status(
                "stopped", intentional_native_stage_stop=True, resume_from_last_committed_pair=True
            )
        else:
            atomic_json(
                run_dir / "status.json",
                {
                    "schema": "continuous_joint_w1ax_v1",
                    "status": "stopped",
                    "heartbeat_unix": time.time(),
                    "pid": os.getpid(),
                    "models": {},
                    "intentional_native_stage_stop": True,
                    "reason": str(error),
                },
            )
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
                    "phase": "capture_audit_readiness",
                    "optimization_started": False,
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

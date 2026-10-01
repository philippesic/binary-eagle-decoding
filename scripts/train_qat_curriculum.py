#!/usr/bin/env python3
"""Validate an audited curriculum; only --start --allow-cuda loads a model."""

from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from w1a1_eagle.continuous_qat import atomic_json  # noqa: E402
from w1a1_eagle.qat_curriculum import CurriculumConfig, PrecisionStage  # noqa: E402
from w1a1_eagle.qat_curriculum_runner import (  # noqa: E402
    CurriculumRunner,
    RunnerConfig,
    require_provider,
)
from w1a1_eagle.recurrent_provider import audit_provider_round  # noqa: E402
from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract  # noqa: E402


def load_config(path):
    spec = json.loads(Path(path).read_text())
    if spec.get("schema") != "qat_curriculum_experiment_v1":
        raise ValueError("unsupported curriculum experiment schema")
    curriculum_spec = spec["curriculum"]
    curriculum = CurriculumConfig(
        tuple(PrecisionStage(**stage) for stage in curriculum_spec["stages"]),
        optimizer_transition=curriculum_spec.get("optimizer_transition", "fresh"),
    )
    kwargs = dict(spec["qat"])
    contract = W1AxContract(**kwargs.pop("contract"))
    qat = JointQATConfig(contract, **kwargs)
    config = RunnerConfig(**spec.get("runner", {}))
    if qat.device != "cuda:0" or not qat.allow_accelerator:
        raise ValueError("production CLI requires explicit cuda:0 and allow_accelerator")
    if contract.activation_bits != curriculum.stages[0].activation_bits:
        raise ValueError("QAT precision must equal first curriculum stage")
    if not isinstance(spec.get("hardware"), dict):
        raise ValueError("explicit expected hardware identity required")
    if set(spec["hardware"]) != {"device_name", "compute_capability"}:
        raise ValueError("hardware requires device_name and compute_capability")
    return spec, curriculum, qat, config


def provider_factory(spec):
    name, separator, factory = spec["factory"].partition(":")
    if not separator or not name or not factory:
        raise ValueError("provider factory requires MODULE:FACTORY")
    create = getattr(importlib.import_module(name), factory)
    manifest = Path(spec["manifest"]).resolve()

    def revalidate(qat):
        result = create(qat, manifest)
        require_provider(result)
        return result

    return revalidate


def validate_readiness(factory, curriculum, qat):
    source = None
    for stage in curriculum.stages:
        provider = factory(replace(qat, contract=W1AxContract(stage.activation_bits)))
        if source is None:
            source = provider.source_metadata
        elif source != provider.source_metadata:
            raise ValueError("stage providers must bind identical immutable training sources")
        if not callable(getattr(provider, "load_models_cpu", None)):
            raise ValueError("provider must support CPU model loading before CUDA installation")
    return {
        "schema": "qat_curriculum_readiness_v1",
        "configuration_valid": True,
        "training_source_eligible": True,
        "optimization_started": False,
        "model_loaded": False,
        "cuda_started": False,
        "source": source,
        "curriculum": curriculum.manifest(),
    }


def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+")
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"GPU/run ownership already held: {path}") from None
    stream.seek(0)
    stream.truncate()
    stream.write(str(os.getpid()) + "\n")
    stream.flush()
    return stream


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--allow-cuda", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (args.prepare_only or args.resume or args.allow_cuda) and not args.start:
        parser.error("--prepare-only, --resume and --allow-cuda require --start")
    spec, curriculum, qat, config = load_config(args.config)
    factory = provider_factory(spec["provider"])
    readiness = validate_readiness(factory, curriculum, qat)
    if not args.start:
        print(json.dumps(readiness, sort_keys=True, indent=2))
        return
    if not args.allow_cuda or args.run_dir is None:
        parser.error("--start requires --allow-cuda and --run-dir")
    root = args.run_dir.resolve()
    if args.resume and not (root / "latest.json").is_file():
        parser.error("--resume requires a complete atomic curriculum checkpoint")
    if not args.resume and (
        (root / "latest.json").exists() or (root / "resolved_config.json").exists()
    ):
        parser.error("existing run requires --resume or a new run directory")
    if args.resume:
        original = json.loads((root / "resolved_config.json").read_text())
        status_path = root / "status.json"
        if (
            status_path.is_file()
            and json.loads(status_path.read_text()).get("budget_failed") is True
        ):
            raise ValueError("budget-overrun run requires an explicit new experiment")
        if original != spec:
            raise ValueError("resume changes frozen config/provider/precision/budget contract")
        if args.prepare_only:
            pointer = json.loads((root / "latest.json").read_text())
            if pointer.get("global_updates") != 0:
                raise ValueError("prepare-only refuses existing optimizer progress")
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("explicit start requires declared CUDA hardware")
    properties = torch.cuda.get_device_properties(qat.device)
    observed = {
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
    }
    if observed != spec["hardware"]:
        raise RuntimeError("CUDA hardware differs from declared curriculum contract")
    root.mkdir(parents=True, exist_ok=True)
    run_lock = _lock(root / ".owner.lock")
    try:
        gpu_lock = _lock(Path.home() / ".cache/binary-eagle-decoding/cuda-0.owner.lock")
        try:
            if not args.resume:
                atomic_json(root / "resolved_config.json", spec)
            else:
                (root / "STOP").unlink(missing_ok=True)
            atomic_json(root / "readiness.json", readiness)
            provider = factory(qat)
            trainer = CurriculumRunner(
                provider,
                curriculum,
                qat,
                root,
                config=config,
                start=True,
                allow_cuda=True,
                source_revalidator=factory,
            )
            if args.resume:
                trainer.resume()
            if args.prepare_only or not args.resume:
                batch = max(
                    (
                        b
                        for b in trainer.provider.rounds()
                        if len(b.rows) > 1
                        and any(audit_provider_round(b, trainer.provider).ce_mask[1:])
                    ),
                    key=lambda b: (len(b.prefix_token_ids), len(b.rows)),
                    default=None,
                )
                if batch is None:
                    raise ValueError("no later-proposal train round supports preparation smoke")
                trainer.prepare(batch)
            else:
                trainer.smoke_current()
            if not args.prepare_only:
                trainer.run()
        except BaseException as error:
            if "trainer" in locals():
                trainer.status(
                    "failed",
                    error=f"{type(error).__name__}: {error}",
                    resume_from_last_atomic_checkpoint=True,
                )
            else:
                atomic_json(
                    root / "status.json",
                    {
                        "schema": "qat_curriculum_runner_v1",
                        "status": "failed",
                        "optimization_started": False,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
            raise
        finally:
            gpu_lock.close()
    finally:
        run_lock.close()


if __name__ == "__main__":
    main()

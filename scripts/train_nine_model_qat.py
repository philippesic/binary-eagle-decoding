#!/usr/bin/env python3
"""Source-bound training adapter and zero-update real-model smoke for nine-model QAT.

This command never contacts a host. The operator dispatches it under remote_job.
Real optimizer updates require current SM120 per-candidate admission; SM75 may
only execute --smoke-zero-updates or --prepare-only. Production dependencies
must be materialized and authenticated; missing captures are explicit failures.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import signal
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

from w1a1_eagle.block_qat import (  # noqa: E402
    BlockDrafter,
    BlockQATConfig,
    block_contract,
    block_loss,
    block_optimizer,
    block_train_step,
)
from w1a1_eagle.block_training import (  # noqa: E402
    BlockCursor,
    export_block_checkpoint,
    load_block_checkpoint,
    load_block_gguf,
    save_block_checkpoint,
    transition_a8_to_a1,
)
from w1a1_eagle.continuous_qat import (  # noqa: E402
    ContinuousTrainer,
    atomic_json,
    build_lanes,
    sha256,
)
from w1a1_eagle.continuous_resources import (  # noqa: E402
    linux_host_memory,
    require_host_memory,
)

SCHEMA = "nine_model_qat_training_v1"
CHECKS = ("source", "resource", "kernel", "model", "backward", "memory", "capture_portability")


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec.get("schema") != SCHEMA or spec.get("family") not in {"eagle", "dspark", "dflash"}:
        raise ValueError("unsupported nine-model training schema/family")
    if spec.get("device") != "cuda:0":
        raise ValueError("production profiles require explicit cuda:0")
    if not isinstance(spec.get("candidate"), str) or not spec["candidate"]:
        raise ValueError("candidate name required")
    if spec["family"] != "eagle":
        config = BlockQATConfig(**spec["qat"])
        if config.family != spec["family"]:
            raise ValueError("block family and QAT config differ")
        limits = spec["limits"]
        if not limits or set(limits) - {
            "max_steps",
            "max_supervised_tokens",
            "max_seconds",
            "max_epochs",
        }:
            raise ValueError("explicit bounded training limits required")
        for name, value in limits.items():
            if value is not None and (
                isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
                or (name != "max_seconds" and type(value) is not int)
            ):
                raise ValueError("positive finite training limit required")
        if not any(value is not None for value in limits.values()):
            raise ValueError("at least one hard training cap required")
        if type(spec.get("checkpoint_every")) is not int or spec["checkpoint_every"] < 1:
            raise ValueError("positive checkpoint cadence required")
        if spec.get("precision_stage", "direct") not in {"direct", "a8_to_a1"}:
            raise ValueError("unsupported precision stage")
        if spec.get("precision_stage") == "a8_to_a1" and (
            config.activation_bits != 1
            or type(spec.get("a8_warmup_steps")) is not int
            or spec["a8_warmup_steps"] < 1
        ):
            raise ValueError("A8→A1 requires final A1 and positive charged warmup steps")
    return spec


def require_admission(path, bundle_sha, config_sha):
    record = json.loads(Path(path).read_text())
    if (
        record.get("schema") != "nine_model_training_admission_v1"
        or record.get("status") != "PASS"
        or record.get("bundle_sha256") != bundle_sha
        or record.get("config_sha256") != config_sha
        or record.get("compute_capability") != [12, 0]
        or record.get("optimizer_updates") != 0
        or any(record.get("checks", {}).get(name) != "PASS" for name in CHECKS)
    ):
        raise ValueError("fresh source/config-bound SM120 training admission required")
    return record


def configure_cuda(*, zero_updates):
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("explicit CUDA hardware required")
    props = torch.cuda.get_device_properties("cuda:0")
    capability = [props.major, props.minor]
    if capability not in ([7, 5], [12, 0]) or (not zero_updates and capability != [12, 0]):
        raise ValueError("SM75 development permits zero updates only; real QAT requires SM120")
    torch.cuda.reset_peak_memory_stats("cuda:0")
    return {
        "device_name": props.name,
        "compute_capability": capability,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tf32": False,
        "precision": "F32 student masters/moments and floating reference, F16 KV",
    }


def tensor_batch(batch):
    values = dict(vars(batch))
    for name in (
        "context_features",
        "input_tokens",
        "positions",
        "labels",
        "loss_mask",
        "attention_allowed",
        "predecessor_ids",
        "teacher_logits",
    ):
        value = values.get(name)
        if isinstance(value, np.ndarray):
            # One bounded block copy; mmap stays the immutable provider source.
            values[name] = torch.from_numpy(np.array(value, copy=True))
    return SimpleNamespace(**values)


def resources(spec, stage):
    floors = spec.get("resource_floors", {})
    host = require_host_memory(
        linux_host_memory(),
        floor_bytes=floors.get("host_available_bytes", 2 * 1024**3),
        stage=stage,
    )
    free, total = torch.cuda.mem_get_info("cuda:0")
    reserved = torch.cuda.memory_reserved("cuda:0")
    if free < floors.get("cuda_free_bytes", 1024**3) or reserved > floors.get(
        "cuda_reserved_bytes", 14 * 1024**3
    ):
        raise RuntimeError("CUDA resource floor/cap failure at " + stage)
    return {
        **host,
        "cuda_free_bytes": free,
        "cuda_total_bytes": total,
        "cuda_reserved_bytes": reserved,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated("cuda:0"),
        "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved("cuda:0"),
    }


def calibration(locator, config):
    if locator is None:
        return None
    path = Path(locator["path"])
    if (
        sha256(path) != locator["sha256"]
        or locator.get("activation_bits") != config.activation_bits
    ):
        raise ValueError("calibration artifact must bind current deployed activation arithmetic")
    arrays = np.load(path, allow_pickle=False)
    names = {key.removesuffix(".latent") for key in arrays if key.endswith(".latent")}
    if set(arrays) != {name + suffix for name in names for suffix in (".latent", ".scale")}:
        raise ValueError("calibration latent/scale inventory differs")
    return {
        name: (
            torch.from_numpy(arrays[name + ".latent"].copy()),
            torch.from_numpy(arrays[name + ".scale"].copy()),
        )
        for name in names
    }


def block_inputs(spec, bundle_sha):
    from w1a1_eagle.block_data import BlockDataset

    config = BlockQATConfig(**spec["qat"])
    if spec.get("precision_stage") == "a8_to_a1":
        config = replace(config, activation_bits=8)
    data = spec["data"]
    dataset = BlockDataset(data["path"], expected_sha256=data["sha256"], allow_synthetic=False)
    if (
        dataset.manifest["family"] != config.family
        or dataset.vocab_size != config.vocab_size
        or dataset.target_width != config.hidden_size
        or dataset.manifest["mask_token_id"] != config.mask_token_id
    ):
        raise ValueError("model/data five-tap/full-vocabulary geometry differs")
    source = {
        "base_gguf_sha256": spec["model"]["sha256"],
        "data_manifest_sha256": data["sha256"],
        "bundle_sha256": bundle_sha,
        "synthetic": False,
        "target_sha256": dataset.manifest["producer"]["target_sha256"],
    }
    tensors = load_block_gguf(spec["model"]["path"], spec["model"]["sha256"], config)
    initializer = calibration(spec.get("initialization"), config)
    model = BlockDrafter(tensors, config, binary_initializer=initializer)
    del tensors
    return model.to(spec["device"]), dataset, source


@contextlib.contextmanager
def native_teacher(spec, source, run_dir):
    settings = spec.get("teacher")
    if settings is None:
        yield None
        return
    from capture_block_qat_teacher import NativeTeacher

    if settings["target_sha256"] != source["target_sha256"]:
        raise ValueError("current-prefix teacher target differs from captured target")
    kwargs = {
        key: settings[key]
        for key in ("target_sha256", "max_tokens", "gpu_layers", "producer_source_revision")
    }
    with NativeTeacher(
        settings["binary"], settings["target"], run_dir / "current-prefix-teacher", **kwargs
    ) as teacher:

        def capture(prefix):
            receipt = teacher.capture_prefix(
                list(prefix),
                [2, 10, 18, 26, 34],
                logits_mode="last",
                chain_ancestry={"purpose": "own_prefix_qat_train", "source": source},
            )
            logits = NativeTeacher.array(receipt, "logits")
            return torch.from_numpy(np.array(logits[-1], dtype=np.float32, copy=True))

        yield capture


def smoke_block(model, batch, optimizer, teacher_callback=None, *, require_empty_optimizer=True):
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    # Later slot must backpropagate through previous layer's evolving state/K/V.
    watched = (output.layer_states[0], *output.noise_kv[1])
    gradients = torch.autograd.grad(output.logits[-1].square().sum(), watched, retain_graph=True)
    if any(not bool(torch.isfinite(g).all()) or not bool((g != 0).any()) for g in gradients):
        raise ValueError("later state/K/V gradient gate failed")
    loss, counts = block_loss(output, batch, model.config, teacher_callback=teacher_callback)
    loss.backward()
    parameters = list(model.parameters())
    if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError("selected parameter finite/backward gate failed")
    if require_empty_optimizer and optimizer.state:
        raise ValueError("zero-update smoke unexpectedly allocated optimizer moments")
    report = {
        **counts,
        "loss": float(loss.detach()),
        "optimizer_updates": 0,
        "optimizer_moment_tensors": 0,
        "later_state_gradient_norm": float(gradients[0].norm()),
        "later_k_gradient_norm": float(gradients[1].norm()),
        "later_v_gradient_norm": float(gradients[2].norm()),
        "gradient_tensors": len(parameters),
        "nonzero_gradient_tensors": sum(bool((p.grad != 0).any()) for p in parameters),
        "hard_forward": True,
        "full_private_head": True,
        "selected_projection_count": len(model.binary_linears()),
        "contract": block_contract(model.config),
    }
    optimizer.zero_grad(set_to_none=True)
    return report


def run_block(args, spec, hardware):
    from w1a1_eagle.block_data import BlockCursor as DataCursor

    resources(spec, "before block loading")
    model, dataset, source = block_inputs(spec, args.bundle_sha256)
    optimizer = block_optimizer(model)
    cursor = BlockCursor(
        data_cursor=dataset.cursor(seed=model.config.seed).payload(),
        stage="a8_warm_start" if spec.get("precision_stage") == "a8_to_a1" else "direct",
    )
    if args.resume:
        receipt = json.loads((args.run_dir / "checkpoints/latest.json").read_text())
        if receipt["cursor"]["stage"] == "a1_final":
            model, optimizer, _ = transition_a8_to_a1(model, source_checkpoint_sha256="0" * 64)
        cursor = load_block_checkpoint(model, optimizer, source, receipt)
    first, _ = dataset.next_block(
        DataCursor(**cursor.data_cursor),
        require_teacher=model.config.objective == "full_probability_l1" and not spec.get("teacher"),
    )
    with native_teacher(spec, source, args.run_dir) as teacher:
        if args.prepare_only and cursor.step:
            raise ValueError("prepare-only cannot restore real optimizer progress")
        smoke = smoke_block(
            model, tensor_batch(first), optimizer, teacher, require_empty_optimizer=not args.resume
        )
        footprint = resources(spec, "after actual block forward/backward")
        if args.smoke_zero_updates:
            return {
                "schema": "nine_model_model_smoke_v1",
                "status": "PASS",
                "artifact_kind": "production",
                "bundle_sha256": args.bundle_sha256,
                "config_sha256": sha256(args.config),
                "source": source,
                "hardware": hardware,
                "optimizer_updates": 0,
                "checks": {"model": "PASS", "backward": "PASS", "memory": "PASS"},
                "smoke": smoke,
                "resources": footprint,
            }
        if args.prepare_only:
            committed = save_block_checkpoint(
                model, optimizer, cursor, source, args.run_dir / "checkpoints"
            )
            return {
                "schema": "nine_model_preparation_v1",
                "status": "PASS",
                "optimizer_updates": 0,
                "checkpoint": committed,
                "smoke": smoke,
                "resources": footprint,
            }
        started = time.monotonic()
        elapsed_base = cursor.elapsed_seconds
        stopped = False
        previous_handlers = {}

        def stop(*_):
            nonlocal stopped
            stopped = True

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_handlers[sig] = signal.signal(sig, stop)
        latest = None
        unique = set(cursor.unique_blocks)

        def capped():
            values = {
                "max_steps": cursor.step,
                "max_supervised_tokens": cursor.supervised_tokens,
                "max_seconds": cursor.elapsed_seconds,
                "max_epochs": cursor.epoch,
            }
            return any(
                bound is not None and values[name] >= bound
                for name, bound in spec["limits"].items()
            )

        try:
            while not capped() and not stopped:
                if (args.run_dir / "STOP").exists():
                    stopped = True
                    break
                cursor = replace(cursor, elapsed_seconds=elapsed_base + time.monotonic() - started)
                if capped():
                    break
                resources(spec, "block optimizer update")
                batch, next_data = dataset.next_block(
                    DataCursor(**cursor.data_cursor),
                    require_teacher=model.config.objective == "full_probability_l1"
                    and teacher is None,
                )
                metrics, _ = block_train_step(
                    model, optimizer, tensor_batch(batch), teacher_callback=teacher
                )
                unique.add(f"{batch.chain_id}:{batch.block_index}")
                cursor = replace(
                    cursor,
                    step=cursor.step + 1,
                    epoch=next_data.epoch,
                    block_index=cursor.block_index + 1,
                    supervised_tokens=cursor.supervised_tokens + metrics["supervised_tokens"],
                    presented_tokens=cursor.presented_tokens + metrics["presented_tokens"],
                    elapsed_seconds=elapsed_base + time.monotonic() - started,
                    unique_blocks=tuple(sorted(unique)),
                    data_cursor=next_data.payload(),
                )
                if cursor.stage == "a8_warm_start" and cursor.step >= spec["a8_warmup_steps"]:
                    latest = save_block_checkpoint(
                        model, optimizer, cursor, source, args.run_dir / "checkpoints"
                    )
                    del optimizer
                    model, optimizer, transition = transition_a8_to_a1(
                        model, source_checkpoint_sha256=latest["sha256"]
                    )
                    cursor = replace(cursor, stage="a1_final")
                    atomic_json(args.run_dir / "precision-transition.json", transition)
                    # Transition retains training source cost/cursor; its own exact
                    # checkpoints bind A1 and fresh moments from this point onward.
                elif cursor.step % spec["checkpoint_every"] == 0:
                    latest = save_block_checkpoint(
                        model, optimizer, cursor, source, args.run_dir / "checkpoints"
                    )
                atomic_json(
                    args.run_dir / "status.json",
                    {
                        "schema": SCHEMA,
                        "status": "running",
                        "cursor": asdict(cursor),
                        "metrics": metrics,
                        "heartbeat_unix": time.time(),
                    },
                )
            cursor = replace(cursor, elapsed_seconds=elapsed_base + time.monotonic() - started)
            if latest is None or latest["cursor"] != json.loads(json.dumps(asdict(cursor))):
                latest = save_block_checkpoint(
                    model, optimizer, cursor, source, args.run_dir / "checkpoints"
                )
            if stopped:
                atomic_json(
                    args.run_dir / "status.json",
                    {
                        "schema": SCHEMA,
                        "status": "STOP",
                        "checkpoint": latest,
                        "cursor": asdict(cursor),
                    },
                )
                raise InterruptedError("STOP: checkpoint retained; evaluation forbidden")
            if spec.get("precision_stage") == "a8_to_a1" and cursor.stage != "a1_final":
                raise ValueError("budget ended before final A1 stage; checkpoint retained")
            exported = export_block_checkpoint(model, source, args.run_dir / "final-export")
            return {
                "schema": "nine_model_stage_receipt_v1",
                "stage": args.stage_name,
                "status": "PASS",
                "artifact_kind": "production",
                "bundle_sha256": args.bundle_sha256,
                "config_sha256": sha256(args.config),
                "committed": True,
                "completion_reason": "approved_budget_complete",
                "checkpoint": latest,
                "exports": {
                    spec["candidate"]: {
                        "checkpoint": exported["npz"],
                        "manifest": exported["manifest"],
                        "base_gguf_sha256": source["base_gguf_sha256"],
                    }
                },
                "counters": asdict(cursor),
                "hardware": hardware,
            }
        finally:
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)


def eagle_inputs(spec, args):
    from prepared_continuous_provider import PreparedProvider, authenticate
    from train_prepared_continuous_w1ax import create_current_native_child

    api = importlib.import_module("train_continuous_w1ax")
    locator = spec["eagle_config"]
    if sha256(Path(locator["path"])) != locator["sha256"]:
        raise ValueError("EAGLE configuration hash differs")
    original, config = api.load_config(Path(locator["path"]))
    if config.activation_bits not in {(1,), (8,)}:
        raise ValueError("nine-model campaign EAGLE uses a single direct A1 or A8 lane")
    if config.development_lifecycle != "standalone":
        raise ValueError("EAGLE evaluation must be a fresh standalone process")
    prepared = spec["prepared"]
    authenticated = authenticate(
        api, original, args.run_dir, prepared["run_dir"], prepared["ready_sha256"]
    )
    provider = PreparedProvider(
        authenticated, config.qat(config.activation_bits[0]), create_current_native_child
    )
    lanes = build_lanes(provider, config, args.run_dir)
    return provider, lanes, config


def run_eagle(args, spec, hardware):
    provider, lanes, config = eagle_inputs(spec, args)
    trainer = ContinuousTrainer(provider, lanes, config, args.run_dir)
    if args.resume:
        trainer.resume()
    batches = list(provider.bounded_rounds())
    if not batches:
        raise ValueError("authenticated EAGLE smoke shard empty")
    # The selected full native-prefix fixture gates current model, not historic
    # actor receipts. Full corpus source is already authenticated without tensors.
    smoke = trainer.smoke(max(batches, key=lambda batch: len(batch.rows)))
    del batches
    if any(lane.optimizer.state for lane in lanes) and not args.resume:
        raise ValueError("EAGLE smoke created optimizer moments")
    if args.smoke_zero_updates:
        if trainer.step or any(lane.optimizer.state for lane in lanes):
            raise ValueError("zero-update smoke cannot use a progressed checkpoint")
        return {
            "schema": "nine_model_model_smoke_v1",
            "status": "PASS",
            "artifact_kind": "production",
            "bundle_sha256": args.bundle_sha256,
            "config_sha256": sha256(args.config),
            "source": provider.source_metadata,
            "hardware": hardware,
            "optimizer_updates": 0,
            "checks": {"model": "PASS", "backward": "PASS", "memory": "PASS"},
            "smoke": smoke,
            "resources": resources(spec, "EAGLE actual forward/backward"),
        }
    if args.prepare_only:
        if trainer.step:
            raise ValueError("prepare-only cannot perform/restore real optimizer updates")
        trainer.save()
        return {
            "schema": "nine_model_preparation_v1",
            "status": "PASS",
            "optimizer_updates": 0,
            "checkpoint": trainer.checkpoint,
            "smoke": smoke,
        }
    trainer.run()
    if trainer.stop_requested or not trainer.capped():
        raise InterruptedError(
            "STOP/intermediate development boundary retains checkpoint; no final receipt"
        )
    directory = Path(trainer.checkpoint["path"]).parent
    lane = lanes[0]
    return {
        "schema": "nine_model_stage_receipt_v1",
        "stage": args.stage_name,
        "status": "PASS",
        "artifact_kind": "production",
        "bundle_sha256": args.bundle_sha256,
        "config_sha256": sha256(args.config),
        "committed": True,
        "completion_reason": "approved_budget_complete",
        "checkpoint": trainer.checkpoint,
        "exports": {
            spec["candidate"]: {
                "checkpoint": str(directory / lane.name / "joint.npz"),
                "manifest": str(directory / lane.name / "joint.json"),
                "base_gguf_sha256": provider.base_gguf_sha256,
            }
        },
        "counters": {
            "step": trainer.step,
            "supervised_tokens": trainer.tokens,
            "elapsed_seconds": trainer.elapsed_seconds,
            "epoch": trainer.epoch,
            "cursor": trainer.cursor,
        },
        "hardware": hardware,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "run-dir", "completion-output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle-sha256", required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--allow-cuda", action="store_true")
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke-zero-updates", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if not args.allow_cuda:
        parser.error("explicit operator --allow-cuda required")
    if len(args.bundle_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in args.bundle_sha256
    ):
        parser.error("bundle SHA256 required")
    spec = load_spec(args.config)
    if not (args.smoke_zero_updates or args.prepare_only):
        if args.admission is None:
            parser.error("production updates require --admission")
        require_admission(args.admission, args.bundle_sha256, sha256(args.config))
    if args.resume and args.smoke_zero_updates:
        parser.error("progressed resume is incompatible with zero-update admission smoke")
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    api = importlib.import_module("train_continuous_w1ax")
    run_lock = api.lock(args.run_dir / ".owner.lock")
    gpu_lock = api.lock(Path.home() / ".cache/binary-eagle-decoding/cuda-0.owner.lock")
    try:
        hardware = configure_cuda(zero_updates=args.smoke_zero_updates or args.prepare_only)
        receipt = (run_eagle if spec["family"] == "eagle" else run_block)(args, spec, hardware)
        atomic_json(args.completion_output, receipt)
    except BaseException as error:
        atomic_json(
            args.completion_output,
            {
                "schema": "nine_model_stage_receipt_v1",
                "stage": args.stage_name,
                "bundle_sha256": args.bundle_sha256,
                "status": "STOP" if isinstance(error, InterruptedError) else "FAIL",
                "error": f"{type(error).__name__}: {error}",
                "evaluation_permitted": False,
            },
        )
        raise
    finally:
        gpu_lock.close()
        run_lock.close()


if __name__ == "__main__":
    main()

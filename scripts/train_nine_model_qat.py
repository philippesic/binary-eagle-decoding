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
import re
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
from w1a1_eagle.continuous_budget import TrainingBudget  # noqa: E402
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
        for quota in ("min_a1_updates", "min_a1_supervised_tokens"):
            if quota in spec and (type(spec[quota]) is not int or spec[quota] < 1):
                raise ValueError("final A1 exposure quota must be a positive integer")
        if spec.get("precision_stage") == "a8_to_a1" and (
            limits.get("max_steps") is not None and limits["max_steps"] <= spec["a8_warmup_steps"]
        ):
            raise ValueError("A8→A1 budget must include committed final A1 updates")
    final = declared_qat_configs(spec)[-1]
    final_bits = (
        final.activation_bits
        if isinstance(final, BlockQATConfig)
        else final.contract.activation_bits
    )
    if spec["candidate"] != f"{spec['family']}_a{final_bits}":
        raise ValueError("candidate cell name differs from final family/activation precision")
    return spec


def training_source_identity():
    """Exact source-bound admission inventory; CPU-only, no accelerator query."""
    files = [
        "scripts/train_nine_model_qat.py",
        "scripts/train_continuous_w1ax.py",
        "scripts/prepared_continuous_provider.py",
        "scripts/train_prepared_continuous_w1ax.py",
        "scripts/capture_block_qat_teacher.py",
        "src/w1a1_eagle/block_qat.py",
        "src/w1a1_eagle/block_training.py",
        "src/w1a1_eagle/block_data.py",
        "src/w1a1_eagle/qat_initialization.py",
        "src/w1a1_eagle/qat_admission.py",
        "src/w1a1_eagle/qat_curriculum_runner.py",
        "src/w1a1_eagle/continuous_resources.py",
        "src/w1a1_eagle/continuous_runtime.py",
        "src/w1a1_eagle/nine_model_admission.py",
    ]
    from w1a1_eagle.continuous_runtime import ADMISSION_FILES, MATH_FILES

    files.extend("src/w1a1_eagle/" + name for name in MATH_FILES)
    files.extend("scripts/" + name for name in ADMISSION_FILES)
    # Missing production modules cannot emit a complete readiness inventory.
    return {name: sha256(ROOT / name) for name in sorted(set(files))}


def declared_qat_configs(spec):
    """Exact selected arithmetic for current-package source/config admission."""
    if spec["family"] != "eagle":
        config = BlockQATConfig(**spec["qat"])
        return (
            [replace(config, activation_bits=8), config]
            if spec.get("precision_stage") == "a8_to_a1"
            else [config]
        )
    api = importlib.import_module("train_continuous_w1ax")
    locator = spec["eagle_config"]
    if sha256(Path(locator["path"])) != locator["sha256"]:
        raise ValueError("EAGLE config source changed")
    _, config = api.load_config(Path(locator["path"]))
    if spec.get("precision_stage") == "a8_to_a1":
        initial = config.qat(8)
        from w1a1_eagle.recurrent_qat import W1AxContract

        return [initial, replace(initial, contract=W1AxContract(1))]
    return [config.qat(config.activation_bits[0])]


def require_admission(path, bundle_sha, config_sha, *, candidate=None, gpu_uuid=None):
    record = json.loads(Path(path).read_text())
    if (
        record.get("schema") != "nine_model_training_admission_v1"
        or record.get("status") != "PASS"
        or record.get("artifact_kind") != "production"
        or record.get("bundle_sha256") != bundle_sha
        or record.get("config_sha256") != config_sha
        or record.get("compute_capability") != [12, 0]
        or record.get("optimizer_updates") != 0
        or record.get("source_files") != training_source_identity()
        or not isinstance(record.get("gpu_uuid"), str)
        or not record.get("gpu_uuid")
        or (gpu_uuid is not None and record.get("gpu_uuid") != gpu_uuid)
        or (candidate is not None and record.get("candidate") != candidate)
        or any(record.get("checks", {}).get(name) != "PASS" for name in CHECKS)
    ):
        raise ValueError("fresh source/config-bound SM120 training admission required")
    return record


def canonical_gpu_uuid(value):
    """Canonical physical NVIDIA UUID from a complete Torch/NVML UUID string."""
    match = re.fullmatch(
        r"(?:GPU-)?([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        str(value),
    )
    if match is None:
        raise ValueError("full physical NVIDIA GPU UUID required")
    return "GPU-" + match[1].lower()


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
        "gpu_uuid": canonical_gpu_uuid(props.uuid),
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
    if locator.get("encoding", "policy_latents") not in {"policy_latents", "hard_signs"}:
        raise ValueError("calibration encoding unsupported")
    metadata = locator.get("latent_initialization")
    if not isinstance(metadata, dict) or set(metadata) != {
        "policy",
        "reference_kind",
        "reference_sha256",
    }:
        raise ValueError("calibration requires explicit latent policy/reference kind/SHA")
    if (
        metadata["policy"] not in {"preserve_reference_magnitudes", "unit_probe"}
        or metadata["reference_kind"]
        not in {"eagle_fixed_reference_0.5", "block_source_weight_magnitudes"}
        or len(metadata["reference_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in metadata["reference_sha256"])
    ):
        raise ValueError("calibration latent policy/reference identity invalid")
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
    if names != {"fc"}:
        raise ValueError("selected calibration scope is sparse fusion only")
    return {
        name: (
            torch.from_numpy(arrays[name + ".latent"].copy()),
            torch.from_numpy(arrays[name + ".scale"].copy()),
        )
        for name in names
    }


def validate_initializer_reference(report, locator, family):
    if locator is None:
        return
    metadata = locator["latent_initialization"]
    expected_kind = (
        "eagle_fixed_reference_0.5" if family == "eagle" else "block_source_weight_magnitudes"
    )
    if (
        metadata["reference_kind"] != expected_kind
        or report.get("fc", {}).get("reference_sha256") != metadata["reference_sha256"]
    ):
        raise ValueError("calibrated initializer reference kind/magnitude SHA differs from source")


def block_inputs(spec, bundle_sha):
    from w1a1_eagle.block_data import BlockDataset

    config = BlockQATConfig(**spec["qat"])
    if spec.get("precision_stage") == "a8_to_a1":
        config = replace(config, activation_bits=8)
    data = spec["data"]
    admission = data.get("admission")
    if admission is not None and (
        not isinstance(admission, dict) or set(admission) != {"path", "sha256"}
    ):
        raise ValueError("completed block data admission must pin path and SHA together")
    dataset = BlockDataset(
        data["path"],
        expected_sha256=data["sha256"],
        allow_synthetic=False,
        admission_path=None if admission is None else admission["path"],
        admission_sha256=None if admission is None else admission["sha256"],
    )
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
    if (
        initializer is not None
        and spec["initialization"]["latent_initialization"]["policy"]
        != config.latent_initialization
    ):
        raise ValueError("block initializer magnitude policy differs from QAT contract")
    if (
        spec.get("initialization")
        and spec["initialization"].get("encoding", "policy_latents")
        != config.initialization_encoding
    ):
        raise ValueError("block initializer encoding differs from QAT contract")
    model = BlockDrafter(tensors, config, binary_initializer=initializer)
    validate_initializer_reference(
        getattr(model.fc, "initialization_report", {}), spec.get("initialization"), config.family
    )
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


def smoke_with_training_memory(parameters, spec, hardware, callback, *, optimizer=None):
    """Reserve actual F32 Adam moment shapes through forward/backward on SM120.

    The scratch allocations never attach to the optimizer and never update a
    model. SM75 development reports pending full training-memory admission.
    """
    parameters = [p for p in parameters if p.requires_grad]
    if any(p.dtype != torch.float32 for p in parameters):
        raise ValueError("admitted optimizer masters/moments must be F32")
    reservation = []
    status = "PENDING"
    try:
        if hardware.get("compute_capability") == [12, 0] and not (optimizer and optimizer.state):
            reservation = [torch.zeros_like(p) for p in parameters for _ in range(2)]
            status = "PASS"
        elif hardware.get("compute_capability") == [12, 0] and optimizer and optimizer.state:
            status = "PASS"
        result = callback()
        measured = resources(spec, "actual backward with training moment reservation")
        return result, {
            "status": status,
            "reserved_moment_bytes": sum(t.numel() * t.element_size() for t in reservation),
            "optimizer_moments_attached": 0,
            "optimizer_updates": 0,
            "resources": measured,
        }
    finally:
        del reservation
        if torch.cuda.is_available():
            torch.cuda.synchronize("cuda:0")
            torch.cuda.empty_cache()


def completed_zero_update_smoke_contract(bits, optimizers):
    if any(optimizer.state for optimizer in optimizers):
        raise ValueError("zero-update admission requires empty real optimizer state")
    from w1a1_eagle.recurrent_qat import FIXED_A8_ARITHMETIC_REVISION

    return {
        "fixed_w1a8_arithmetic_revision": FIXED_A8_ARITHMETIC_REVISION,
        "hard_forward": True,
        "optimizer_updates": 0,
        "optimizer_moment_tensors": 0,
        "activation_bits_exercised": list(bits),
    }


def smoke_block(
    model,
    batch,
    optimizer,
    teacher_callback=None,
    *,
    require_empty_optimizer=True,
    resource_observer=None,
):
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
    if resource_observer is not None:
        report["resources_gradients_resident"] = resource_observer()
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
            model, optimizer, _ = transition_a8_to_a1(
                model, source_checkpoint_sha256="0" * 64, in_place=True
            )
        cursor = load_block_checkpoint(model, optimizer, source, receipt)
    first, _ = dataset.next_block(
        DataCursor(**cursor.data_cursor),
        require_teacher=model.config.objective == "full_probability_l1" and not spec.get("teacher"),
    )
    with native_teacher(spec, source, args.run_dir) as teacher:
        if args.prepare_only and cursor.step:
            raise ValueError("prepare-only cannot restore real optimizer progress")
        smoke, training_memory = smoke_with_training_memory(
            model.parameters(),
            spec,
            hardware,
            lambda: smoke_block(
                model,
                tensor_batch(first),
                optimizer,
                teacher,
                require_empty_optimizer=not args.resume,
                resource_observer=lambda: resources(spec, "actual block gradients resident"),
            ),
            optimizer=optimizer,
        )
        if args.smoke_zero_updates and spec.get("precision_stage") == "a8_to_a1":
            model, optimizer, _ = transition_a8_to_a1(
                model, source_checkpoint_sha256="0" * 64, in_place=True
            )
            a1_smoke, a1_memory = smoke_with_training_memory(
                model.parameters(),
                spec,
                hardware,
                lambda: smoke_block(
                    model,
                    tensor_batch(first),
                    optimizer,
                    teacher,
                    resource_observer=lambda: resources(spec, "actual A1 gradients resident"),
                ),
            )
            smoke = {"A8": smoke, "A1": a1_smoke}
            training_memory = {"A8": training_memory, "A1": a1_memory}
        footprint = resources(spec, "after actual block forward/backward")
        if args.smoke_zero_updates:
            return {
                "schema": "nine_model_model_smoke_v1",
                "status": "PASS",
                "artifact_kind": "synthetic" if source["synthetic"] else "production",
                "bundle_sha256": args.bundle_sha256,
                "config_sha256": sha256(args.config),
                "source": source,
                "hardware": hardware,
                "optimizer_updates": 0,
                "checks": {"model": "PASS", "backward": "PASS", "memory": "PASS"},
                "smoke": smoke,
                "smoke_contract": completed_zero_update_smoke_contract(
                    [8, 1]
                    if spec.get("precision_stage") == "a8_to_a1"
                    else [model.config.activation_bits],
                    [optimizer],
                ),
                "training_memory": training_memory,
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
                "smoke_contract": completed_zero_update_smoke_contract(
                    [8, 1]
                    if spec.get("precision_stage") == "a8_to_a1"
                    else [model.config.activation_bits],
                    [optimizer],
                ),
                "training_memory": training_memory,
                "resources": footprint,
            }
        budget = TrainingBudget(
            args.run_dir / "budget-used.json",
            args.bundle_sha256,
            spec["limits"].get("max_seconds"),
            atomic_json,
        )
        budget.begin(cursor.elapsed_seconds)
        cursor = replace(cursor, elapsed_seconds=budget.elapsed())
        stopped = False
        previous_handlers = {}

        def stop(*_):
            nonlocal stopped
            stopped = True

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous_handlers[sig] = signal.signal(sig, stop)
        latest = (
            receipt
            if args.resume
            else save_block_checkpoint(
                model, optimizer, cursor, source, args.run_dir / "checkpoints"
            )
        )
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
                cursor = replace(cursor, elapsed_seconds=budget.elapsed())
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
                    stage_updates=cursor.stage_updates + 1,
                    stage_supervised_tokens=cursor.stage_supervised_tokens
                    + metrics["supervised_tokens"],
                    elapsed_seconds=budget.elapsed(),
                    unique_blocks=tuple(sorted(unique)),
                    data_cursor=next_data.payload(),
                )
                if cursor.stage == "a8_warm_start" and cursor.step >= spec["a8_warmup_steps"]:
                    latest = save_block_checkpoint(
                        model, optimizer, cursor, source, args.run_dir / "checkpoints"
                    )
                    del optimizer
                    model, optimizer, transition = transition_a8_to_a1(
                        model, source_checkpoint_sha256=latest["sha256"], in_place=True
                    )
                    cursor = replace(
                        cursor, stage="a1_final", stage_updates=0, stage_supervised_tokens=0
                    )
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
            cursor = replace(cursor, elapsed_seconds=budget.finish())
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
            if cursor.step <= 0:
                raise ValueError(
                    "training budget completed with zero optimizer updates; checkpoint retained"
                )
            if spec.get("precision_stage") == "a8_to_a1" and (
                cursor.stage != "a1_final"
                or cursor.stage_updates < spec.get("min_a1_updates", 1)
                or cursor.stage_supervised_tokens < spec.get("min_a1_supervised_tokens", 1)
            ):
                raise ValueError(
                    "budget ended before required final A1 exposure; checkpoint retained"
                )
            exported = export_block_checkpoint(model, source, args.run_dir / "final-export")
            return {
                "schema": "nine_model_stage_receipt_v1",
                "stage": args.stage_name,
                "status": "PASS",
                "artifact_kind": "synthetic" if source["synthetic"] else "production",
                "bundle_sha256": args.bundle_sha256,
                "config_sha256": sha256(args.config),
                "committed": True,
                "completion_reason": "approved_budget_complete",
                "checkpoint": {"path": latest["path"], "sha256": latest["sha256"]},
                "exports": {
                    spec["candidate"]: {
                        "checkpoint": {
                            "path": exported["npz"],
                            "sha256": sha256(Path(exported["npz"])),
                        },
                        "manifest": {
                            "path": exported["manifest"],
                            "sha256": sha256(Path(exported["manifest"])),
                        },
                        "base_gguf_sha256": source["base_gguf_sha256"],
                    }
                },
                "counters": asdict(cursor),
                "hardware": hardware,
            }
        finally:
            budget.finish()
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
    initializer = calibration(
        spec.get("initialization"), config.qat(config.activation_bits[0]).contract
    )
    expected_initializer = (spec.get("initialization") or {}).get("sha256")
    if (
        spec.get("initialization")
        and spec["initialization"]["latent_initialization"]["policy"]
        != config.initialization_policy
    ):
        raise ValueError("EAGLE initialization policy differs from immutable config")
    if (
        spec.get("initialization")
        and spec["initialization"].get("encoding", "policy_latents")
        != config.initialization_encoding
    ):
        raise ValueError("EAGLE initialization encoding differs from immutable config")
    if config.initialization_sha256 != expected_initializer:
        raise ValueError("EAGLE immutable config initialization hash differs")
    lanes = build_lanes(provider, config, args.run_dir, initialization=initializer)
    validate_initializer_reference(
        getattr(lanes[0].drafter, "qat_initialization_report", {}),
        spec.get("initialization"),
        "eagle",
    )
    return provider, lanes, config


def run_eagle(args, spec, hardware):
    if spec.get("precision_stage") == "a8_to_a1":
        return run_eagle_curriculum(args, spec, hardware)
    provider, lanes, config = eagle_inputs(spec, args)
    from w1a1_eagle.qat_admission import VerifiedTrainingAdmission

    admission = (
        None
        if args.smoke_zero_updates or args.prepare_only
        else VerifiedTrainingAdmission.from_locator(
            args.admission,
            config_path=args.config,
            bundle_sha256=args.bundle_sha256,
            candidate=spec["candidate"],
        )
    )
    trainer = ContinuousTrainer(provider, lanes, config, args.run_dir, training_admission=admission)
    if args.resume:
        trainer.resume()
    from w1a1_eagle.recurrent_provider import audit_provider_round

    selected = None
    for batch in provider.bounded_rounds():
        audit = audit_provider_round(batch, provider)
        if not any(audit.ce_mask[1:]):
            continue
        if selected is None or len(batch.prefix_token_ids) > len(selected.prefix_token_ids):
            selected = batch
    if selected is None:
        raise ValueError("authenticated EAGLE smoke shard has no later-position labels")
    smoke, training_memory = smoke_with_training_memory(
        lanes[0].drafter.parameters(),
        spec,
        hardware,
        lambda selected=selected: trainer.smoke(selected, allocate_optimizer_state=False),
        optimizer=lanes[0].optimizer,
    )
    del selected
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
            "smoke_contract": completed_zero_update_smoke_contract(
                config.activation_bits, [lane.optimizer for lane in lanes]
            ),
            "training_memory": training_memory,
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
            "bundle_sha256": args.bundle_sha256,
            "config_sha256": sha256(args.config),
            "stage": args.stage_name,
            "artifact_kind": "production",
            "source": provider.source_metadata,
        }
    trainer.run()
    if trainer.stop_requested or not trainer.capped():
        raise InterruptedError(
            "STOP/intermediate development boundary retains checkpoint; no final receipt"
        )
    if trainer.step <= 0:
        raise ValueError(
            "training budget completed with zero optimizer updates; checkpoint retained"
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
        "checkpoint": {"path": trainer.checkpoint["path"], "sha256": trainer.checkpoint["sha256"]},
        "exports": {
            spec["candidate"]: {
                "checkpoint": {
                    "path": str(directory / lane.name / "joint.npz"),
                    "sha256": sha256(directory / lane.name / "joint.npz"),
                },
                "manifest": {
                    "path": str(directory / lane.name / "joint.json"),
                    "sha256": sha256(directory / lane.name / "joint.json"),
                },
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


def run_eagle_curriculum(args, spec, hardware):
    """Use the tested existing reset-only runner, with authenticated lazy data."""
    from prepared_continuous_provider import PreparedProvider, authenticate
    from train_prepared_continuous_w1ax import create_current_native_child

    from w1a1_eagle.qat_admission import VerifiedTrainingAdmission
    from w1a1_eagle.qat_curriculum import CurriculumConfig, PrecisionStage
    from w1a1_eagle.qat_curriculum_runner import CurriculumRunner, RunnerConfig
    from w1a1_eagle.recurrent_provider import audit_provider_round
    from w1a1_eagle.recurrent_qat import save_joint_checkpoint

    api = importlib.import_module("train_continuous_w1ax")
    locator = spec["eagle_config"]
    if sha256(Path(locator["path"])) != locator["sha256"]:
        raise ValueError("EAGLE curriculum configuration SHA differs")
    original, continuous = api.load_config(Path(locator["path"]))
    if continuous.status_every_steps != 1:
        raise ValueError("status cadence probe is supported by direct EAGLE only, not warm runner")
    if continuous.activation_bits != (8,) or continuous.activation_quantization != "fixed":
        raise ValueError("EAGLE warm profile starts fixed-reference A8 single lane")
    curriculum = CurriculumConfig(
        tuple(PrecisionStage(**stage) for stage in spec["curriculum"]["stages"]),
        spec["curriculum"].get("optimizer_transition", "fresh"),
    )
    if tuple(stage.activation_bits for stage in curriculum.stages) != (8, 1):
        raise ValueError("only selected A8→A1 reset transition supported")
    prepared = spec["prepared"]
    auth = authenticate(api, original, args.run_dir, prepared["run_dir"], prepared["ready_sha256"])
    provider = PreparedProvider(auth, continuous.qat(8), create_current_native_child)

    def revalidate(qat):
        # Byte hashes from this process may reuse only unchanged inode/stat
        # identities. New actor eligibility remains separately bound to admission.
        for path, digest in auth["binding"]["artifacts"].items():
            auth["files"].check(Path(path), digest)
        return PreparedProvider(auth, qat, create_current_native_child)

    initializer = calibration(spec.get("initialization"), continuous.qat(8).contract)
    initializer_sha = (spec.get("initialization") or {}).get("sha256")
    if continuous.initialization_sha256 != initializer_sha:
        raise ValueError("curriculum initializer differs from immutable EAGLE config")
    admission = (
        None
        if args.smoke_zero_updates or args.prepare_only
        else VerifiedTrainingAdmission.from_locator(
            args.admission,
            config_path=args.config,
            bundle_sha256=args.bundle_sha256,
            candidate=spec["candidate"],
        )
    )
    runner = CurriculumRunner(
        provider,
        curriculum,
        continuous.qat(8),
        args.run_dir,
        start=True,
        allow_cuda=True,
        config=RunnerConfig(**spec.get("curriculum_runner", {})),
        source_revalidator=revalidate,
        initialization=initializer,
        initialization_sha256=initializer_sha,
        initialization_policy=continuous.initialization_policy,
        initialization_encoding=continuous.initialization_encoding,
        reserve_optimizer_memory=hardware.get("compute_capability") == [12, 0],
        training_admission=admission,
        crash_safe_budget=not (args.smoke_zero_updates or args.prepare_only),
    )
    validate_initializer_reference(
        getattr(runner.drafter, "qat_initialization_report", {}),
        spec.get("initialization"),
        "eagle",
    )
    if args.resume:
        runner.resume()
    selected = None
    for batch in runner.provider.bounded_rounds():
        audit = audit_provider_round(batch, runner.provider)
        if any(audit.ce_mask[1:]) and (
            selected is None or len(batch.prefix_token_ids) > len(selected.prefix_token_ids)
        ):
            selected = batch
    if selected is None:
        raise ValueError("eligible EAGLE curriculum shard lacks later supervision")
    if runner.state.global_updates == 0:
        stages = runner.prepare(selected)
    else:
        # Exact resume uses current stage/backward; no new preparation or reset.
        stages = [runner._smoke_round(selected)]
        runner.smoke_passed = True
    del selected
    if args.smoke_zero_updates or args.prepare_only:
        if runner.state.global_updates or runner.optimizer.state:
            raise ValueError("zero-update curriculum admission requires empty optimizer state")
        pointer = runner.last_checkpoint
        return {
            "schema": "nine_model_model_smoke_v1"
            if args.smoke_zero_updates
            else "nine_model_preparation_v1",
            "status": "PASS",
            "artifact_kind": "production",
            "optimizer_updates": 0,
            "bundle_sha256": args.bundle_sha256,
            "config_sha256": sha256(args.config),
            "source": runner.source,
            "hardware": hardware,
            "smoke": stages,
            "smoke_contract": completed_zero_update_smoke_contract([8, 1], [runner.optimizer]),
            "training_memory": {
                "stages": {
                    str(stage["activation_bits"]): stage["training_memory"] for stage in stages
                },
                "status": "PASS" if hardware.get("compute_capability") == [12, 0] else "PENDING",
                "reservation": "two F32 copies per trainable held through each backward",
                "optimizer_moments_attached": 0,
                "optimizer_updates": 0,
            },
            "checks": {"model": "PASS", "backward": "PASS", "memory": "PASS"},
            "resources": resources(spec, "EAGLE all-stage actual smoke"),
            "checkpoint": {
                "path": str(args.run_dir / "checkpoints" / pointer["file"]),
                "sha256": pointer["sha256"],
            },
        }
    runner.run()
    if runner.stop_requested or not runner.state.complete or runner.budget_failed:
        raise InterruptedError(
            "curriculum STOP/failed cap retains checkpoint; evaluation forbidden"
        )
    directory = args.run_dir / "final-export"
    directory.mkdir(exist_ok=False)
    save_joint_checkpoint(
        runner.linears,
        runner.qat,
        runner.provider.base_gguf_sha256,
        directory / "joint.npz",
        directory / "joint.json",
    )
    pointer = runner.last_checkpoint
    return {
        "schema": "nine_model_stage_receipt_v1",
        "stage": args.stage_name,
        "status": "PASS",
        "artifact_kind": "production",
        "bundle_sha256": args.bundle_sha256,
        "config_sha256": sha256(args.config),
        "committed": True,
        "completion_reason": "approved_budget_complete",
        "checkpoint": {
            "path": str(args.run_dir / "checkpoints" / pointer["file"]),
            "sha256": pointer["sha256"],
        },
        "exports": {
            spec["candidate"]: {
                "checkpoint": {
                    "path": str(directory / "joint.npz"),
                    "sha256": sha256(directory / "joint.npz"),
                },
                "manifest": {
                    "path": str(directory / "joint.json"),
                    "sha256": sha256(directory / "joint.json"),
                },
                "base_gguf_sha256": runner.provider.base_gguf_sha256,
            }
        },
        "counters": {
            "step": runner.state.global_updates,
            "supervised_tokens": sum(stage["supported_rows"] for stage in runner.state.phases),
            "elapsed_seconds": runner.occupancy_seconds,
            "phases": runner.state.phases,
            "transitions": runner.state.transitions,
            "cursor": runner.cursor,
            "epoch": runner.epoch,
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
        require_admission(
            args.admission, args.bundle_sha256, sha256(args.config), candidate=spec["candidate"]
        )
    if args.resume and args.smoke_zero_updates:
        parser.error("progressed resume is incompatible with zero-update admission smoke")
    args.run_dir = args.run_dir.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    api = importlib.import_module("train_continuous_w1ax")
    run_lock = api.lock(args.run_dir / ".owner.lock")
    gpu_lock = api.lock(Path.home() / ".cache/binary-eagle-decoding/cuda-0.owner.lock")
    try:
        hardware = configure_cuda(zero_updates=args.smoke_zero_updates or args.prepare_only)
        if not (args.smoke_zero_updates or args.prepare_only):
            require_admission(
                args.admission,
                args.bundle_sha256,
                sha256(args.config),
                candidate=spec["candidate"],
                gpu_uuid=hardware["gpu_uuid"],
            )
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

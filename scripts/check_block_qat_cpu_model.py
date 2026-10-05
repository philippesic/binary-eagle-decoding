#!/usr/bin/env python3
"""Bounded released-model CPU backward with explicit synthetic or pinned TRAIN inputs.

No GPU is queried; no optimizer update is permitted. This is model/source math,
ownership and resource development evidence, never corpus, quality or CUDA proof.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch  # noqa:E402
from train_nine_model_qat import (  # noqa:E402
    calibration,
    smoke_block,
    tensor_batch,
    validate_initializer_reference,
)

from w1a1_eagle.block_data import BlockDataset  # noqa:E402
from w1a1_eagle.block_qat import BlockDrafter, BlockQATConfig, block_optimizer  # noqa:E402
from w1a1_eagle.block_training import load_block_gguf  # noqa:E402
from w1a1_eagle.continuous_qat import atomic_json, sha256  # noqa:E402
from w1a1_eagle.qat_initialization import apply_binary_initialization  # noqa:E402

MEMORY_CODE = """import json,sys,psutil
p=psutil.Process(int(sys.argv[1]))
m=psutil.virtual_memory()
print(json.dumps({'rss_bytes':p.memory_info().rss,'host_available_bytes':m.available,'host_total_bytes':m.total,'psutil_version':psutil.__version__,'psutil_module':psutil.__file__,'python':sys.executable}))
"""


def memory(python, pid):
    return json.loads(
        subprocess.check_output([python, "-c", MEMORY_CODE, str(pid)], text=True, timeout=10)
    )


def input_contract(args, config):
    """Authenticate bounded current-host data/fit joins before loading model weights."""
    names = (
        "data_manifest",
        "data_sha256",
        "data_admission",
        "data_admission_sha256",
        "initialization_index",
        "initialization_index_sha256",
    )
    supplied = [getattr(args, name, None) is not None for name in names]
    if args.input_mode == "synthetic":
        if any(supplied) or config.objective != "hard_ce":
            raise ValueError("synthetic mode cannot claim pinned TRAIN/calibration or soft targets")
        return None, None, {"input_mode": "synthetic", "production_readiness": "PENDING"}
    if not all(supplied):
        raise ValueError("native_train requires all manifest/admission/initialization index pins")
    index_path = Path(args.initialization_index)
    if sha256(index_path) != args.initialization_index_sha256:
        raise ValueError("initialization index differs from external SHA pin")
    index = json.loads(index_path.read_text())
    if (
        index.get("schema") != "nine_model_cpu_fusion_development_artifact_index_v1"
        or index.get("artifact_kind") != "development_CPU"
        or index.get("status") != "PASS_CPU_DEVELOPMENT_FITS"
        or index.get("optimizer_updates") != 0
    ):
        raise ValueError("selected initialization index is not completed CPU development fit")
    profiles = [
        profile
        for profile in index["profiles"]
        if profile["family"] == config.family
        and type(profile["activation_bits"]) is int
        and profile["activation_bits"] == config.activation_bits
    ]
    if len(profiles) != 1:
        raise ValueError("initialization index must select exactly one family/activation profile")
    locator = profiles[0]["initialization"]
    if (
        locator.get("encoding", "policy_latents") != "policy_latents"
        or locator.get("latent_initialization", {}).get("policy") != "preserve_reference_magnitudes"
        or locator["latent_initialization"]["reference_kind"] != "block_source_weight_magnitudes"
    ):
        raise ValueError("CPU composition selects sparse FC reference-magnitude policy latents")
    initializer = calibration(locator, config)
    dataset = BlockDataset(
        args.data_manifest,
        expected_sha256=args.data_sha256,
        allow_synthetic=False,
        admission_path=args.data_admission,
        admission_sha256=args.data_admission_sha256,
    )
    if (
        dataset.manifest["family"] != config.family
        or dataset.vocab_size != config.vocab_size
        or dataset.target_width != config.hidden_size
        or dataset.manifest["mask_token_id"] != config.mask_token_id
    ):
        raise ValueError("current model/data family/five-tap/full-vocabulary geometry differs")
    cursor = dataset.cursor(split="train", seed=config.seed)
    batch, next_cursor = dataset.next_block(
        cursor, require_teacher=config.objective == "full_probability_l1"
    )
    detail = {
        "input_mode": "native_train",
        "production_readiness": "PENDING",
        "data_manifest": {
            "path": str(Path(args.data_manifest).resolve()),
            "sha256": args.data_sha256,
        },
        "completed_admission": {
            "path": str(Path(args.data_admission).resolve()),
            "sha256": args.data_admission_sha256,
        },
        "initialization_index": {
            "path": str(index_path.resolve()),
            "sha256": args.initialization_index_sha256,
        },
        "initialization": locator,
        "data_cursor": cursor.payload(),
        "next_data_cursor": next_cursor.payload(),
        "target_sha256": dataset.manifest["producer"]["target_sha256"],
        "producer": dataset.manifest["producer"],
        "context_features_shape": list(batch.context_features.shape),
        "teacher_logits_shape": None
        if batch.teacher_logits is None
        else list(batch.teacher_logits.shape),
        "prefix_tokens": list(batch.prefix_tokens),
        "predecessor_ids": batch.predecessor_ids.tolist(),
        "teacher_prefix_sha256": list(batch.teacher_prefix_sha256),
        "scope": (
            "one original TRAIN block, captured-prefix hard CE or exact captured-prefix full L1; "
            "CPU development only"
        ),
    }
    return tensor_batch(batch), (initializer, locator), detail


def install_initialization(model, prepared):
    if prepared is None:
        return {}
    initializer, locator = prepared
    untouched_versions = {
        name: tuple(p._version for p in module.parameters())
        for name, module in model.binary_linears().items()
        if name != "fc"
    }
    report = apply_binary_initialization(
        model.binary_linears(),
        initializer,
        policy=locator["latent_initialization"]["policy"],
        encoding=locator.get("encoding", "policy_latents"),
    )
    validate_initializer_reference(report, locator, model.config.family)
    if (
        not torch.equal(model.fc.latent_sign, initializer["fc"][0])
        or not torch.equal(model.fc.initial_scale, initializer["fc"][1])
        or bool((model.fc.scale_offset != 0).any())
    ):
        raise ValueError("FC calibrated latent/scale application differs from pinned artifact")
    if any(
        tuple(p._version for p in model.binary_linears()[name].parameters()) != versions
        for name, versions in untouched_versions.items()
    ):
        raise ValueError("sparse FC initializer touched an unselected FFN projection")
    report["application"] = {
        "selected_projections": ["fc"],
        "artifact_values_exact": True,
        "ffn_parameter_versions_unchanged": True,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("dspark", "dflash"), required=True)
    parser.add_argument("--activation-bits", type=int, choices=(1, 8), required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-sha256", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--memory-python", required=True)
    parser.add_argument("--max-rss-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--prelaunch-free-bytes", type=int, default=12 * 1024**3)
    parser.add_argument("--live-free-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--input-mode", choices=("synthetic", "native_train"), default="synthetic")
    parser.add_argument(
        "--objective", choices=("hard_ce", "full_probability_l1"), default="hard_ce"
    )
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--data-sha256")
    parser.add_argument("--data-admission", type=Path)
    parser.add_argument("--data-admission-sha256")
    parser.add_argument("--initialization-index", type=Path)
    parser.add_argument("--initialization-index-sha256")
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("preserve prior evidence; report must be new")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = BlockQATConfig(
        args.family, args.activation_bits, conditioning="captured_prefix", objective=args.objective
    )
    observed = memory(args.memory_python, os.getpid())
    base = {
        "schema": "block_qat_actual_cpu_development_v1",
        "artifact_kind": "development_synthetic_inputs"
        if args.input_mode == "synthetic"
        else "development_native_train_cpu_composition",
        "family": args.family,
        "activation_bits": args.activation_bits,
        "base_path": str(args.base.resolve()),
        "base_sha256": args.base_sha256,
        "config": asdict(config),
        "optimizer_updates": 0,
        "teacher_and_features": "synthetic; not native TRAIN capture or quality evaluation"
        if args.input_mode == "synthetic"
        else (
            "pinned native original TRAIN capture; one block CPU composition, "
            "not quality or CUDA readiness"
        ),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": str(torch.__version__),
            "device": "CPU",
            "master_precision": "F32",
            "cache_precision": "F16",
            "source_precision": "original BF16/F16 GGUF values",
        },
        "source_sha256": {
            name: sha256(ROOT / "src/w1a1_eagle" / name)
            for name in (
                "block_qat.py",
                "block_training.py",
                "recurrent_qat.py",
                "qat_initialization.py",
                "block_data.py",
            )
        },
        "guard_source_sha256": sha256(Path(__file__)),
        "launcher_source_sha256": sha256(ROOT / "scripts/train_nine_model_qat.py"),
        "limits": {
            "max_rss_bytes": args.max_rss_bytes,
            "prelaunch_free_bytes": args.prelaunch_free_bytes,
            "live_free_bytes": args.live_free_bytes,
            "max_seconds": args.max_seconds,
        },
        "before": observed,
    }
    phase = "before_source_load"
    peak = observed["rss_bytes"]
    minimum = observed["host_available_bytes"]
    stop = threading.Event()
    samples = []
    publication_lock = threading.Lock()

    def publish(status, **extra):
        with publication_lock:
            atomic_json(
                args.report,
                dict(
                    base,
                    status=status,
                    phase=phase,
                    peak_rss_bytes=peak,
                    minimum_host_available_bytes=minimum,
                    elapsed_seconds=time.monotonic() - started,
                    resource_samples=samples,
                    **extra,
                ),
            )

    if observed["host_available_bytes"] < args.prelaunch_free_bytes:
        publish("FAIL", error="prelaunch host available below declared floor")
        raise RuntimeError("prelaunch resource floor")

    def monitor():
        nonlocal peak, minimum
        while not stop.wait(0.5):
            try:
                sample = memory(args.memory_python, os.getpid())
                peak = max(peak, sample["rss_bytes"])
                minimum = min(minimum, sample["host_available_bytes"])
                if len(samples) < 256:
                    samples.append(
                        {"phase": phase, "elapsed_seconds": time.monotonic() - started, **sample}
                    )
                failed = (
                    sample["rss_bytes"] > args.max_rss_bytes
                    or sample["host_available_bytes"] < args.live_free_bytes
                    or time.monotonic() - started > args.max_seconds
                )
                if failed:
                    publish(
                        "FAIL",
                        error="live RSS/host/time declared bound violated",
                        failed_resource=sample,
                    )
                    os._exit(77)
            except BaseException as error:
                publish("FAIL", error=f"memory observer: {type(error).__name__}: {error}")
                os._exit(78)

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        torch.set_num_threads(args.threads)
        publish("RUNNING")
        phase = "authenticating_pinned_inputs"
        batch, prepared_initialization, data_contract = input_contract(args, config)
        base["input_contract"] = data_contract
        phase = "before_source_load"
        tensors = load_block_gguf(args.base, args.base_sha256, config)
        phase = "constructing_current_student"
        publish("RUNNING", source_tensor_count=len(tensors))
        model = BlockDrafter(tensors, config)
        source_head = tensors["output.weight"]
        source_embedding = tensors["token_embd.weight"]
        if (
            model.output.data_ptr() == source_head.data_ptr()
            or model.token_embd.data_ptr() == source_embedding.data_ptr()
        ):
            raise ValueError("private buffers alias source storage")
        if not torch.equal(model.output, source_head) or not torch.equal(
            model.token_embd, source_embedding
        ):
            raise ValueError("private source values changed")
        del tensors, source_head, source_embedding
        phase = "installing_sparse_calibrated_fc"
        base["initializer_application"] = install_initialization(model, prepared_initialization)
        del prepared_initialization
        optimizer = block_optimizer(model)
        versions = {name: p._version for name, p in model.named_parameters()}
        generator = torch.Generator().manual_seed(123)
        context = 2
        if batch is None:
            batch = SimpleNamespace(
                context_features=torch.randn(context, 5, config.hidden_size, generator=generator)
                * 0.1,
                prefix_tokens=(1, 2, 3),
                input_tokens=torch.tensor([3] + [config.mask_token_id] * 6),
                positions=torch.arange(context, context + 7),
                labels=torch.tensor([10, 11, 12, 13, 14, 15, 16]),
                predecessor_ids=torch.tensor([3, 10, 11, 12, 13, 14, 15]),
                loss_mask=torch.ones(7, dtype=torch.bool),
                attention_allowed=torch.ones(7, context + 7, dtype=torch.bool),
                teacher_logits=None,
            )
        phase = "actual_hard_forward_backward_zero_updates"
        publish(
            "RUNNING",
            trainable_parameters=sum(p.numel() for p in model.parameters()),
            projection_count=len(model.binary_linears()),
            moment_tensors=len(optimizer.state),
        )
        with patch.object(
            torch.optim.AdamW, "step", side_effect=AssertionError("real optimizer update forbidden")
        ):
            smoke = smoke_block(
                model,
                batch,
                optimizer,
                resource_observer=lambda: memory(args.memory_python, os.getpid()),
            )
        if optimizer.state or any(
            p._version != versions[name] for name, p in model.named_parameters()
        ):
            raise ValueError("real optimizer state/update or parameter movement observed")
        own = {id(p) for p in model.parameters() if p.requires_grad}
        bound = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if own != bound:
            raise ValueError("parameter/teacher ownership differs")
        phase = "completed_zero_updates"
        final = memory(args.memory_python, os.getpid())
        rss_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform != "darwin":
            rss_peak *= 1024
        peak = max(peak, final["rss_bytes"], rss_peak)
        minimum = min(minimum, final["host_available_bytes"])
        if peak > args.max_rss_bytes or final["host_available_bytes"] < args.live_free_bytes:
            raise ValueError("final declared memory bound failed")
        publish(
            "PASS",
            smoke=smoke,
            after=final,
            optimizer_moment_tensors=0,
            private_source_values_unchanged=True,
            parameter_versions_unchanged=True,
            trainable_tensor_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
            parameter_ownership=(
                "exact selected student binary signs/scales; private frozen source "
                "buffers, no target parameters"
            ),
        )
    except BaseException as error:
        publish("FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop.set()
        watcher.join(timeout=12)


if __name__ == "__main__":
    main()

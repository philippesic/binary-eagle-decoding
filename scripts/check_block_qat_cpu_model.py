#!/usr/bin/env python3
"""Bounded real released-model CPU backward using explicitly synthetic TRAIN-layout inputs.

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
from train_nine_model_qat import smoke_block  # noqa:E402

from w1a1_eagle.block_qat import BlockDrafter, BlockQATConfig, block_optimizer  # noqa:E402
from w1a1_eagle.block_training import load_block_gguf  # noqa:E402
from w1a1_eagle.continuous_qat import atomic_json, sha256  # noqa:E402

MEMORY_CODE = """import json,sys,psutil
p=psutil.Process(int(sys.argv[1]))
m=psutil.virtual_memory()
print(json.dumps({'rss_bytes':p.memory_info().rss,'host_available_bytes':m.available,'host_total_bytes':m.total,'psutil_version':psutil.__version__,'psutil_module':psutil.__file__,'python':sys.executable}))
"""


def memory(python, pid):
    return json.loads(
        subprocess.check_output([python, "-c", MEMORY_CODE, str(pid)], text=True, timeout=10)
    )


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
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError("preserve prior evidence; report must be new")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = BlockQATConfig(args.family, args.activation_bits, conditioning="captured_prefix")
    observed = memory(args.memory_python, os.getpid())
    base = {
        "schema": "block_qat_actual_cpu_development_v1",
        "artifact_kind": "development_synthetic_inputs",
        "family": args.family,
        "activation_bits": args.activation_bits,
        "base_path": str(args.base.resolve()),
        "base_sha256": args.base_sha256,
        "config": asdict(config),
        "optimizer_updates": 0,
        "teacher_and_features": "synthetic; not native TRAIN capture or quality evaluation",
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
            )
        },
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
        optimizer = block_optimizer(model)
        versions = {name: p._version for name, p in model.named_parameters()}
        generator = torch.Generator().manual_seed(123)
        context = 2
        batch = SimpleNamespace(
            context_features=torch.randn(context, 5, config.hidden_size, generator=generator) * 0.1,
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
        peak = max(
            peak, final["rss_bytes"], int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        )
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

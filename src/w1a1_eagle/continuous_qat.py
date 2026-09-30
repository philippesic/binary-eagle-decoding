"""Continuous paired A8/A1 QAT using audited native-prefix recurrent rollouts.

Both independent optimizers remain live. One audited round is presented to A8
then A1; only one autograd graph is live at a time. This is concurrent model
progress with interleaved microbatches, not simultaneous CUDA kernels. Importing
this module never discovers or initializes an accelerator.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import shutil
import signal
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .recurrent_provider import audit_provider_round, forward_torch_round
from .recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    install_joint_linears,
    joint_train_step,
    save_joint_checkpoint,
    shared_round_hard_signs,
)

SCHEMA = "continuous_joint_w1ax_v1"


@dataclass(frozen=True)
class ContinuousConfig:
    device: str = "cpu"
    sign_lr: float = 0.001
    scale_lr: float = 0.00001
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    seeds: tuple[int, int] = (8101, 1101)
    checkpoint_every: int = 250
    keep_checkpoints: int = 3
    diagnostics_every: int = 100
    development_every: int = 1000
    max_steps: int | None = None
    max_tokens: int | None = None
    max_seconds: float | None = None
    max_epochs: int | None = None
    max_prefix_tokens: int = 2048
    min_free_disk_bytes: int = 8 * 1024**3
    max_cuda_reserved_bytes: int = 12 * 1024**3
    min_cuda_free_bytes: int = 1024**3
    log_max_bytes: int = 8 * 1024**2
    log_backups: int = 3

    def __post_init__(self):
        if torch.device(self.device).type not in {"cpu", "cuda"}:
            raise ValueError("only CPU tests and explicitly user-started CUDA are supported")
        for name in (
            "sign_lr",
            "scale_lr",
            "max_grad_norm",
            "checkpoint_every",
            "keep_checkpoints",
            "diagnostics_every",
            "development_every",
            "max_prefix_tokens",
            "max_cuda_reserved_bytes",
            "log_max_bytes",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.warmup_steps < 0 or self.min_free_disk_bytes < 0 or self.log_backups < 1:
            raise ValueError("invalid warmup, disk or retention bound")
        for name in ("max_steps", "max_tokens", "max_seconds", "max_epochs"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be positive or null")
        if len(self.seeds) != 2 or self.seeds[0] == self.seeds[1]:
            raise ValueError("A8 and A1 need distinct seeds")

    def qat(self, bits: int) -> JointQATConfig:
        return JointQATConfig(
            W1AxContract(bits),
            device=self.device,
            allow_accelerator=self.device != "cpu",
            sign_lr=self.sign_lr,
            scale_lr=self.scale_lr,
            max_grad_norm=self.max_grad_norm,
            seed=self.seeds[0 if bits == 8 else 1],
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def rng_state(device: str) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state: dict, device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def memory_estimate(
    shapes: list[tuple[int, int]], *, frozen_bytes: int = 0, graph_budget_bytes: int = 2 * 1024**3
) -> dict:
    """CPU shape arithmetic; dense F32 AdamW master/moments, no GPU discovery.

    Per lane: 4-byte latent + two 4-byte moments; only active lane gradients
    persist. Kernel temporaries, caching allocator and real graph size vary.
    The manual CUDA smoke, not this estimate, decides admission.
    """
    if not shapes or any(len(s) != 2 or min(s) < 1 for s in shapes):
        raise ValueError("nonempty positive matrix shapes required")
    signs = sum(out * inp for out, inp in shapes)
    rows = sum(out for out, _ in shapes)
    parameters = signs + rows
    persistent = 2 * (12 * parameters + 4 * rows) + frozen_bytes
    active_gradient = 4 * parameters
    return {
        "trainable_parameters_per_model": parameters,
        "dual_persistent_bytes": persistent,
        "one_gradient_bytes": active_gradient,
        "assumed_one_graph_bytes": graph_budget_bytes,
        "estimated_peak_bytes": persistent + active_gradient + graph_budget_bytes,
        "paired_resume_checkpoint_bytes": 24 * parameters + 8 * rows,
        "uncertainty": "graph/temporary budget is assumed; real CUDA smoke mandatory",
    }


@dataclass
class Lane:
    name: str
    config: JointQATConfig
    drafter: nn.Module
    linears: dict
    adapter: Any
    optimizer: torch.optim.Optimizer
    rng: dict


def build_lanes(provider, config: ContinuousConfig) -> list[Lane]:
    """Install on CPU before accelerator transfer; frozen target stays on CPU.

    Provider must expose load_models_cpu for CUDA to avoid transient two dense
    GPU drafter copies. Independent trainable state, moments, scales and RNG.
    Frozen operands are shared only after equality and requires_grad checks.
    """
    loader = getattr(provider, "load_models_cpu", None)
    if config.device != "cpu" and not callable(loader):
        raise ValueError("CUDA provider requires load_models_cpu for bounded dual allocation")
    drafter, target = loader() if callable(loader) else provider.load_models()
    if any(p.device.type != "cpu" for p in drafter.parameters()):
        raise ValueError("dual model installation requires CPU model loading")
    for p in drafter.parameters():
        p.requires_grad_(False)
    for p in target.parameters():
        p.requires_grad_(False)
    first_config = config.qat(8)
    linears = install_joint_linears(drafter, target, replace(first_config, device="cpu"))
    other = copy.deepcopy(drafter)
    other_linears = {path: other.get_submodule(path) for path in linears}
    for module in other_linears.values():
        module.contract = config.qat(1).contract
    # Share only frozen operands while both copies are still on CPU. Native
    # norm rebinding by each adapter may later replace tiny norm parameters;
    # embedding ownership stays shared without a second GPU allocation.
    first_params = dict(drafter.named_parameters())
    for name, parameter in list(other.named_parameters()):
        if not parameter.requires_grad:
            reference = first_params[name]
            if parameter.shape != reference.shape or not torch.equal(parameter, reference):
                raise ValueError("frozen drafter copies disagree")
            parent, _, leaf = name.rpartition(".")
            setattr(other.get_submodule(parent) if parent else other, leaf, reference)
    lanes = []
    for bits, model, modules in ((8, drafter, linears), (1, other, other_linears)):
        lane_config = config.qat(bits)
        model.to(config.device)
        model.eval()
        adapter = provider.make_step_adapter(model)
        if set(adapter.linears) != set(modules) or any(
            adapter.linears[path] is not modules[path] for path in modules
        ):
            raise ValueError("adapter bypasses installed binary linears")
        random.seed(lane_config.seed)
        np.random.seed(lane_config.seed)
        torch.manual_seed(lane_config.seed)
        lanes.append(
            Lane(
                f"A{bits}",
                lane_config,
                model,
                modules,
                adapter,
                torch.optim.AdamW(
                    [
                        {"params": [m.latent_sign for m in modules.values()], "lr": config.sign_lr},
                        {
                            "params": [m.scale_offset for m in modules.values()],
                            "lr": config.scale_lr,
                        },
                    ],
                    weight_decay=0,
                    foreach=False,
                ),
                rng_state(config.device),
            )
        )
    del target
    return lanes


class ObservedAdapter:
    """Keep first proposal state/K/V for bounded later-position gradient gates."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.first = None

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def decode_step(self, *args, **kwargs):
        step = self.adapter.decode_step(*args, **kwargs)
        if self.first is None:
            self.first = step
        return step


def later_gradient(logits, audit, observer) -> dict:
    valid = [i for i, flag in enumerate(audit.ce_mask) if flag]
    if not valid or valid[-1] < 1 or observer.first is None:
        return {
            "later_position_gradient_norm": None,
            "later_state_gradient_norm": None,
            "later_k_gradient_norm": None,
            "later_v_gradient_norm": None,
        }
    first = observer.first
    tensors = [first.pre_norm, first.cache.key, first.cache.value]
    row = valid[-1]
    loss = F.cross_entropy(
        logits[row : row + 1],
        torch.tensor([audit.draft_labels[row]], dtype=torch.long, device=logits.device),
    )
    grads = torch.autograd.grad(loss, tensors, retain_graph=True, allow_unused=True)
    norms = [0.0 if g is None else float(g.detach().norm()) for g in grads]
    if not all(math.isfinite(value) for value in norms):
        raise ValueError("nonfinite later-position recurrent gradient")
    return dict(
        zip(("later_state_gradient_norm", "later_k_gradient_norm", "later_v_gradient_norm"), norms)
    ) | {"later_position_gradient_norm": sum(norms)}


class ContinuousTrainer:
    def __init__(
        self,
        provider,
        lanes: list[Lane],
        config: ContinuousConfig,
        run_dir: Path,
        *,
        development_evaluator=None,
    ):
        if [lane.name for lane in lanes] != ["A8", "A1"]:
            raise ValueError("continuous run requires both independent A8 and A1 lanes")
        if provider.training_eligible is not True or provider.split != "train":
            raise ValueError("substantive audited training eligibility required")
        if getattr(provider, "readiness_scope", None) == "row_a16_hard_ce_100_steps":
            raise ValueError("A16 calibration cannot authorize continuous A8/A1 training")
        owned = [
            {id(p) for group in lane.optimizer.param_groups for p in group["params"]}
            for lane in lanes
        ]
        if owned[0] & owned[1]:
            raise ValueError("models/optimizers must have independent trainable state")
        self.provider, self.lanes, self.config = provider, lanes, config
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.evaluator = development_evaluator
        self.stop_requested = False
        self.step = self.epoch = self.cursor = self.tokens = 0
        self.unique_prompts: set[str] = set()
        self.unique_rows: set[str] = set()
        self.elapsed_seconds = 0.0
        self.metrics = {lane.name: {"step": 0} for lane in lanes}
        self.checkpoint = None
        self.source = hashlib.sha256(
            json.dumps(provider.source_metadata, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.smoke_passed = False

    def resources(self) -> dict:
        free = shutil.disk_usage(self.run_dir).free
        if free < self.config.min_free_disk_bytes:
            raise RuntimeError("disk free space below training safety floor")
        result = {"disk_free_bytes": free}
        if torch.device(self.config.device).type == "cuda":
            available, total = torch.cuda.mem_get_info(self.config.device)
            if available < self.config.min_cuda_free_bytes:
                raise RuntimeError("whole-device free memory below configured CUDA safety floor")
            result.update(cuda_free_bytes=available, cuda_total_bytes=total)
            reserved = torch.cuda.memory_reserved(self.config.device)
            result.update(
                cuda_reserved_bytes=reserved,
                cuda_allocated_bytes=torch.cuda.memory_allocated(self.config.device),
                cuda_peak_bytes=torch.cuda.max_memory_reserved(self.config.device),
            )
            if max(reserved, result["cuda_peak_bytes"]) > self.config.max_cuda_reserved_bytes:
                raise RuntimeError("dual CUDA reserved memory exceeds configured safety ceiling")
        return result

    def status(self, status: str, **extra) -> None:
        atomic_json(
            self.run_dir / "status.json",
            {
                "schema": SCHEMA,
                "status": status,
                "heartbeat_unix": time.time(),
                "pid": os.getpid(),
                "models": self.metrics,
                "step": self.step,
                "epoch": self.epoch,
                "cursor": self.cursor,
                "presented_supervised_tokens": self.tokens,
                "unique_prompts": len(self.unique_prompts),
                "unique_supervised_rows": len(self.unique_rows),
                "source_sha256": self.source,
                "checkpoint": self.checkpoint,
                "scheduling": "A8 then A1 on identical rounds; one live autograd graph",
                **extra,
            },
        )

    def log(self, item: dict) -> None:
        path = self.run_dir / "metrics.jsonl"
        if path.exists() and path.stat().st_size >= self.config.log_max_bytes:
            for ordinal in range(self.config.log_backups, 0, -1):
                old = path.with_name(f"metrics.jsonl.{ordinal}")
                if ordinal == self.config.log_backups:
                    old.unlink(missing_ok=True)
                elif old.exists():
                    os.replace(old, path.with_name(f"metrics.jsonl.{ordinal + 1}"))
            os.replace(path, path.with_name("metrics.jsonl.1"))
        with path.open("a") as stream:
            stream.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")

    def smoke(self, batch) -> dict:
        """Actual forward/backward on both resident models before optimization."""
        audit = audit_provider_round(batch, self.provider)
        if len(batch.rows) < 2 or not any(audit.ce_mask[1:]):
            raise ValueError("dual smoke needs a supported later-position supervision row")
        report = {}
        for lane in self.lanes:
            for group in lane.optimizer.param_groups:
                for parameter in group["params"]:
                    state = lane.optimizer.state[parameter]
                    state.setdefault("step", torch.tensor(0.0))
                    state.setdefault("exp_avg", torch.zeros_like(parameter))
                    state.setdefault("exp_avg_sq", torch.zeros_like(parameter))
        for lane in self.lanes:
            restore_rng(lane.rng, self.config.device)
            lane.optimizer.zero_grad(set_to_none=True)
            observer = ObservedAdapter(lane.adapter)
            device_batch = replace(
                batch, raw_target_features=batch.raw_target_features.to(self.config.device)
            )
            with shared_round_hard_signs(lane.linears):
                logits = forward_torch_round(device_batch, observer, self.provider.draft_vocab_size)
            diagnostics = later_gradient(logits, audit, observer)
            if any(
                diagnostics[key] is None or diagnostics[key] <= 0
                for key in (
                    "later_state_gradient_norm",
                    "later_k_gradient_norm",
                    "later_v_gradient_norm",
                )
            ):
                raise ValueError(f"{lane.name} failed attached later-position state/K/V gate")
            labels = torch.tensor(audit.draft_labels, device=logits.device)
            mask = torch.tensor(audit.ce_mask, device=logits.device)
            loss = F.cross_entropy(logits[mask], labels[mask])
            loss.backward()
            params = [p for group in lane.optimizer.param_groups for p in group["params"]]
            if not torch.isfinite(loss) or any(
                p.grad is None or not torch.isfinite(p.grad).all() or not (p.grad != 0).any()
                for p in params
            ):
                raise ValueError(f"{lane.name} failed all-nine finite/nonzero gradient gate")
            # AdamW moment buffers are resident during smoke, before any update.
            # Reuse these initialized buffers in training instead of allocating
            # moments later and invalidating the admission measurement.
            for parameter in params:
                state = lane.optimizer.state[parameter]
                state.setdefault("step", torch.tensor(0.0))
                state.setdefault("exp_avg", torch.zeros_like(parameter))
                state.setdefault("exp_avg_sq", torch.zeros_like(parameter))
            report[lane.name] = {"loss": float(loss.detach()), **diagnostics}
            lane.optimizer.zero_grad(set_to_none=True)
            del logits, loss, observer, device_batch
        report["resources"] = self.resources()
        self.smoke_passed = True
        atomic_json(self.run_dir / "dual_smoke.json", report)
        return report

    def save(self) -> None:
        resources = self.resources()
        tensor_bytes = sum(
            p.numel() * p.element_size() * 3
            for lane in self.lanes
            for group in lane.optimizer.param_groups
            for p in group["params"]
        )
        if resources["disk_free_bytes"] < tensor_bytes * 2 + self.config.min_free_disk_bytes:
            raise RuntimeError("insufficient disk space for atomic paired checkpoint and export")
        parent = self.run_dir / "checkpoints"
        parent.mkdir(exist_ok=True)
        destination = parent / f"step-{self.step:012d}-e{self.epoch:06d}-r{self.cursor:012d}"
        if destination.exists():
            return
        temp = parent / ("." + destination.name + ".tmp")
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        payload = {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "source": self.source,
            "step": self.step,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "tokens": self.tokens,
            "unique_prompts": sorted(self.unique_prompts),
            "unique_rows": sorted(self.unique_rows),
            "elapsed_seconds": self.elapsed_seconds,
            "metrics": self.metrics,
            "global_rng": rng_state(self.config.device),
            "lanes": {
                lane.name: {
                    "linears": {name: module.state_dict() for name, module in lane.linears.items()},
                    "optimizer": lane.optimizer.state_dict(),
                    "rng": lane.rng,
                }
                for lane in self.lanes
            },
        }
        path = temp / "resume.pt"
        with path.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        for lane in self.lanes:
            folder = temp / lane.name
            folder.mkdir()
            save_joint_checkpoint(
                lane.linears,
                lane.config,
                self.provider.base_gguf_sha256,
                folder / "joint.npz",
                folder / "joint.json",
            )
        manifest = {
            "schema": SCHEMA,
            "step": self.step,
            "source_sha256": self.source,
            "sha256": sha256(path),
            "optimizer_rng_cursor_exact": True,
        }
        atomic_json(temp / "manifest.json", manifest)
        for file in temp.rglob("*"):
            if file.is_file():
                descriptor = os.open(file, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        descriptor = os.open(temp, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temp, destination)
        descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self.checkpoint = {
            "path": str(destination / "resume.pt"),
            "sha256": manifest["sha256"],
            "step": self.step,
        }
        atomic_json(self.run_dir / "latest.json", self.checkpoint)
        for old in sorted(parent.glob("step-*"))[: -self.config.keep_checkpoints]:
            shutil.rmtree(old)

    def resume(self) -> None:
        latest = json.loads((self.run_dir / "latest.json").read_text())
        path = Path(latest["path"])
        if (
            not path.is_relative_to(self.run_dir / "checkpoints")
            or sha256(path) != latest["sha256"]
        ):
            raise ValueError("resume checkpoint path/hash mismatch")
        # Only local trusted self-created checkpoints are accepted after hashing.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        old_config = dict(payload["config"])
        current_config = asdict(self.config)
        for field in ("max_steps", "max_tokens", "max_seconds", "max_epochs"):
            old_config.pop(field)
            current_config.pop(field)
        if (
            payload["schema"] != SCHEMA
            or payload["source"] != self.source
            or old_config != current_config
        ):
            raise ValueError("resume changes immutable model/data/optimizer contract")
        for lane in self.lanes:
            saved = payload["lanes"][lane.name]
            for name, module in lane.linears.items():
                module.load_state_dict(saved["linears"][name], strict=True)
            lane.optimizer.load_state_dict(saved["optimizer"])
            lane.rng = saved["rng"]
        self.step, self.epoch, self.cursor = payload["step"], payload["epoch"], payload["cursor"]
        self.tokens, self.elapsed_seconds = payload["tokens"], payload["elapsed_seconds"]
        self.unique_prompts, self.unique_rows = (
            set(payload["unique_prompts"]),
            set(payload["unique_rows"]),
        )
        self.metrics, self.checkpoint = payload["metrics"], latest
        restore_rng(payload["global_rng"], self.config.device)

    def evaluate_development(self):
        """Release CUDA storage during native evaluation; preserve all state.

        No training/kernel overlaps: models, gradients and Adam moments move to
        CPU, native evaluator runs, then both lanes return to their declared
        device. Frozen parameter aliasing is restored during transfer.
        """
        device = self.config.device

        def move(destination):
            shared = {}
            for lane in self.lanes:
                for module in lane.drafter.modules():
                    for key, value in list(module._parameters.items()):
                        if value is None:
                            continue
                        identity = id(value)
                        if identity not in shared:
                            moved = nn.Parameter(
                                value.detach().to(destination), requires_grad=value.requires_grad
                            )
                            shared[identity] = moved
                            # Preserve optimizer parameter identity for trainable
                            # tensors by changing its storage instead.
                            if value.requires_grad:
                                value.data = moved.data
                                shared[identity] = value
                        module._parameters[key] = shared[identity]
                    for key, value in list(module._buffers.items()):
                        if value is not None:
                            module._buffers[key] = value.to(destination)
                lane.adapter.device = torch.device(destination)
                for state in lane.optimizer.state.values():
                    for key, value in list(state.items()):
                        if isinstance(value, torch.Tensor) and key != "step":
                            state[key] = value.to(destination)

        if torch.device(device).type == "cuda":
            move("cpu")
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        try:
            return self.evaluator(self.checkpoint, self.lanes)
        finally:
            if torch.device(device).type == "cuda":
                move(device)
                self.resources()

    def capped(self) -> bool:
        return any(
            limit is not None and value >= limit
            for value, limit in (
                (self.step, self.config.max_steps),
                (self.tokens, self.config.max_tokens),
                (self.elapsed_seconds, self.config.max_seconds),
                (self.epoch, self.config.max_epochs),
            )
        )

    def run(self, *, require_smoke: bool = True) -> None:
        if require_smoke and not self.smoke_passed:
            raise ValueError("dual resource/math smoke must pass before optimization")
        handlers = {}
        if __import__("threading").current_thread() is __import__("threading").main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT):
                handlers[sig] = signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))
        started = time.monotonic()
        elapsed_base = self.elapsed_seconds
        try:
            while not self.stop_requested and not self.capped():
                yielded = False
                for ordinal, batch in enumerate(self.provider.rounds()):
                    if ordinal < self.cursor:
                        continue
                    yielded = True
                    if self.stop_requested or (self.run_dir / "STOP").exists():
                        self.stop_requested = True
                        break
                    if self.capped():
                        break
                    if len(batch.prefix_token_ids) > self.config.max_prefix_tokens:
                        raise ValueError("accepted prefix exceeds declared memory-safe token cap")
                    audit = audit_provider_round(batch, self.provider)
                    if not any(audit.ce_mask):
                        self.cursor = ordinal + 1
                        continue
                    resource_metrics = self.resources()
                    for lane in self.lanes:
                        restore_rng(lane.rng, self.config.device)
                        begin = time.monotonic()
                        factor = min(1.0, (self.step + 1) / max(1, self.config.warmup_steps))
                        for group, lr in zip(
                            lane.optimizer.param_groups, (self.config.sign_lr, self.config.scale_lr)
                        ):
                            group["lr"] = lr * factor
                        observer = ObservedAdapter(lane.adapter)
                        device_batch = replace(
                            batch,
                            raw_target_features=batch.raw_target_features.to(self.config.device),
                        )
                        with shared_round_hard_signs(lane.linears):
                            logits = forward_torch_round(
                                device_batch, observer, self.provider.draft_vocab_size
                            )
                        diagnostics = (
                            later_gradient(logits, audit, observer)
                            if (self.step % self.config.diagnostics_every == 0)
                            else {}
                        )
                        item = joint_train_step(
                            lane.linears, logits, audit, lane.optimizer, lane.config
                        )
                        lane.rng = rng_state(self.config.device)
                        lane.optimizer.zero_grad(set_to_none=True)
                        if torch.device(self.config.device).type == "cuda":
                            torch.cuda.synchronize(self.config.device)
                        sign_count = sum(
                            module.latent_sign.numel() for module in lane.linears.values()
                        )
                        layer_diagnostics = {}
                        if self.step % self.config.diagnostics_every == 0:
                            for name, module in lane.linears.items():
                                latent = module.latent_sign.detach()
                                scales = module.effective_scales().detach()
                                layer_diagnostics[name] = {
                                    "latent_near_zero_fraction": float(
                                        (latent.abs() < 0.05).float().mean()
                                    ),
                                    "latent_projection_bound_fraction": float(
                                        (latent.abs() >= 1).float().mean()
                                    ),
                                    "zero_scale_fraction": float((scales == 0).float().mean()),
                                    "scale_relative_movement": float(
                                        (
                                            module.scale_offset.detach().abs()
                                            / module.initial_scale.clamp_min(1e-12)
                                        ).mean()
                                    ),
                                    "activation_saturation": None
                                    if lane.name == "A1"
                                    else float(module.last_saturation_fraction),
                                }
                        item["layers"] = layer_diagnostics
                        item["cumulative_sign_flips"] = (
                            self.metrics[lane.name].get("cumulative_sign_flips", 0)
                            + item["sign_flips"]
                        )
                        item["clipped_gradient"] = item["gradient_norm"] > self.config.max_grad_norm
                        if lane.name == "A1":
                            item["saturation_mean"] = None
                        item["learning_diagnostic"] = (
                            "no sign flips after 2000 pairs; inspect manually"
                            if self.step >= 1999 and item["cumulative_sign_flips"] == 0
                            else None
                        )
                        item.update(
                            step=self.step + 1,
                            heartbeat_unix=time.time(),
                            grad_finite=True,
                            step_seconds=time.monotonic() - begin,
                            sign_flip_rate=item["sign_flips"] / sign_count,
                            sign_lr=self.config.sign_lr * factor,
                            scale_lr=self.config.scale_lr * factor,
                            **diagnostics,
                        )
                        self.metrics[lane.name] = item
                        self.status("running", **resource_metrics)
                        del logits, observer, device_batch
                    self.step += 1
                    self.cursor = ordinal + 1
                    self.tokens += sum(audit.ce_mask)
                    self.unique_prompts.add(batch.anchor.prompt_id)
                    self.unique_rows.update(
                        f"{batch.capture_id}:{batch.anchor.prompt_id}:"
                        f"{batch.anchor.round_index}:{i}"
                        for i, flag in enumerate(audit.ce_mask)
                        if flag
                    )
                    self.elapsed_seconds = elapsed_base + time.monotonic() - started
                    self.log(
                        {
                            "step": self.step,
                            "models": self.metrics,
                            "presented_supervised_tokens_per_model": self.tokens,
                            "unique_supervised_rows": len(self.unique_rows),
                            "unique_prompts": len(self.unique_prompts),
                            "epoch": self.epoch,
                        }
                    )
                    if self.step % self.config.checkpoint_every == 0:
                        self.save()
                    if self.step % self.config.development_every == 0:
                        self.save()
                        if self.evaluator is not None:
                            # Explicitly serialized; evaluator may only use dev.
                            self.status("development_evaluation", **self.resources())
                            result = self.evaluate_development()
                            atomic_json(self.run_dir / "development.json", result)
                    self.status("running", **self.resources())
                if self.stop_requested or self.capped():
                    break
                if not yielded and self.cursor == 0:
                    raise ValueError("provider yielded no rounds")
                if self.step == 0:
                    raise ValueError("provider yielded no supported training labels")
                self.epoch += 1
                self.cursor = 0
            self.save()
            self.status("stopped" if self.stop_requested else "completed", **self.resources())
        except BaseException as error:
            self.status(
                "failed",
                error=f"{type(error).__name__}: {error}",
                resume_from_last_committed_pair=True,
            )
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

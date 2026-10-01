"""Opt-in single-model recurrent precision curricula with atomic exact resume.

Imports never inspect accelerators or datasets. CUDA occupancy is conservative
synchronized wall time, including smoke, phase switches, checkpoints and idle
provider work while the model remains resident. CPU fixtures measure CPU wall
seconds against the same bounds; they establish no accelerator performance.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import random
import shutil
import signal
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from .continuous_qat import atomic_json, restore_rng, rng_state, sha256
from .continuous_resources import linux_host_memory, require_host_memory
from .continuous_runtime import training_runtime_identity
from .qat_curriculum import CurriculumConfig, CurriculumState
from .qat_optimization import binary_layout
from .recurrent_loss import supported_prefix_ce
from .recurrent_provider import audit_provider_round, forward_torch_round
from .recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    install_joint_linears,
    joint_optimizer,
    joint_train_step,
    shared_round_hard_signs,
)

SCHEMA = "qat_curriculum_runner_v1"
EXTRA_MATH = (
    "qat_curriculum.py",
    "qat_curriculum_runner.py",
    "qat_optimization.py",
    "learned_activation.py",
    "fusion_correction.py",
)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def curriculum_runtime(device: str) -> dict:
    identity = training_runtime_identity(device)
    root = Path(__file__).parent
    identity["curriculum_math_sha256"] = {
        name: sha256(root / name) for name in EXTRA_MATH if (root / name).is_file()
    }
    if torch.device(device).type == "cuda":
        props = torch.cuda.get_device_properties(device)
        identity["hardware"] = {
            "device_name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_memory": props.total_memory,
        }
    return identity


@dataclass(frozen=True)
class RunnerConfig:
    warmup_updates: int = 100
    checkpoint_every: int = 100
    keep_checkpoints: int = 3
    max_prefix_tokens: int = 2048
    update_upper_bound_seconds: float = 60.0
    min_free_disk_bytes: int = 8 * 1024**3
    min_host_available_bytes: int = 2 * 1024**3
    cpu_load_headroom_bytes: int = 12 * 1024**3
    max_cuda_reserved_bytes: int = 12 * 1024**3
    min_cuda_free_bytes: int = 1024**3

    def __post_init__(self):
        for field in ("checkpoint_every", "keep_checkpoints", "max_prefix_tokens"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"{field} must be a positive integer")
        for field in (
            "warmup_updates",
            "min_free_disk_bytes",
            "min_host_available_bytes",
            "cpu_load_headroom_bytes",
            "max_cuda_reserved_bytes",
            "min_cuda_free_bytes",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise ValueError(f"{field} must be a nonnegative integer")
        if (
            isinstance(self.update_upper_bound_seconds, bool)
            or not math.isfinite(self.update_upper_bound_seconds)
            or self.update_upper_bound_seconds <= 0
        ):
            raise ValueError("update duration reservation must be finite and positive")


def require_provider(provider) -> None:
    if (
        getattr(provider, "training_eligible", None) is not True
        or getattr(provider, "full_body_qat_eligible", None) is not True
        or getattr(provider, "split", None) != "train"
        or getattr(provider, "readiness_scope", None) != "full_body_qat"
    ):
        raise ValueError("audited full-body train eligibility required; calibration refused")
    if not isinstance(getattr(provider, "source_metadata", None), dict):
        raise ValueError("provider must bind immutable source metadata")


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v) for v in value)
    return value


class CurriculumRunner:
    """One current-student graph per round, fresh cache and optimizer per stage.

    CUDA requires both explicit start and accelerator authorization. A CUDA
    source_revalidator must rebuild the stage's provider through its audited
    factory, thereby rehashing files; cached source_metadata alone is insufficient.
    The callback receives JointQATConfig and returns the revalidated provider.
    """

    def __init__(
        self,
        provider,
        curriculum: CurriculumConfig,
        qat: JointQATConfig,
        run_dir: Path,
        *,
        config: RunnerConfig = RunnerConfig(),
        start: bool = False,
        allow_cuda: bool = False,
        source_revalidator=None,
        clock=time.monotonic,
    ):
        if not start:
            raise ValueError("model construction requires explicit start")
        device = torch.device(qat.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("only CPU fixtures or explicitly authorized CUDA are supported")
        if device.type == "cuda" and (not allow_cuda or not qat.allow_accelerator):
            raise ValueError("CUDA curriculum requires explicit start and allow_cuda")
        if qat.objective != "hard_ce" or qat.contract.scale_layout != "row":
            raise ValueError("runner supports audited hard CE with row binary weights")
        if qat.contract.activation_bits != curriculum.stages[0].activation_bits:
            raise ValueError("initial QAT precision differs from first curriculum stage")
        require_provider(provider)
        if device.type == "cuda" and not callable(source_revalidator):
            raise ValueError("CUDA curriculum requires fresh provider source revalidation")
        self.provider, self.curriculum, self.qat_template = provider, curriculum, qat
        self.config, self.device, self.clock = config, qat.device, clock
        self.source_revalidator = source_revalidator
        self.budget_failed = False
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.source = json.loads(json.dumps(provider.source_metadata))
        self.runtime = curriculum_runtime(self.device)
        provider_file = Path(inspect.getfile(type(provider)))
        self.runtime["provider_implementation_sha256"] = sha256(provider_file)
        self.contract = {
            "schema": SCHEMA,
            "curriculum": curriculum.manifest(),
            "qat": asdict(qat),
            "runner": asdict(config),
            "source": self.source,
            "runtime": self.runtime,
        }
        self.state = CurriculumState(
            curriculum, data_contract=self.source, model_contract=self.contract
        )
        self.model_phase = self.cursor = self.epoch = 0
        self.occupancy_seconds = self.smoke_seconds = self.transition_seconds = 0.0
        self.unique_rows: set[str] = set()
        self.unique_prompts: set[str] = set()
        self.smoke_passed = False
        self.stop_requested = False
        self.last_checkpoint = None
        self._accounted_at = None
        self._revalidate(qat)
        self._accounted_at = self.clock()
        self.resources(load=True)
        loader = getattr(self.provider, "load_models_cpu", None)
        if device.type == "cuda" and not callable(loader):
            raise ValueError("CUDA curriculum requires load_models_cpu for bounded installation")
        drafter, target = loader() if callable(loader) else self.provider.load_models()
        if any(p.device.type != "cpu" for m in (drafter, target) for p in m.parameters()):
            raise ValueError("model installation requires both target and drafter loaded on CPU")
        for model in (drafter, target):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.drafter = drafter
        self.qat = qat
        self.linears = install_joint_linears(drafter, target, replace(qat, device="cpu"))
        del target
        self.drafter.to(self.device).eval()
        self.bank = self._activation_bank()
        self.adapter = self._adapter()
        self.optimizer = joint_optimizer(self.linears, self.qat)
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]
        random.seed(qat.seed)
        np.random.seed(qat.seed)
        torch.random.default_generator.manual_seed(qat.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(qat.seed)
        self.rng = rng_state(self.device)
        self.resources()
        self._sync()
        self._account_occupancy()

    def _activation_bank(self):
        if getattr(self.qat, "activation_quantization", "fixed") != "learned":
            return None
        from .learned_activation import LearnedActivationBank

        # Installer may hold a bank on the drafter. Rebind only at construction
        # or a precision switch, before any update. Canonical checkpoint owns
        # six shared parameters once, excluding repeated module alias entries.
        for module in self.linears.values():
            if hasattr(module, "activation_quantizer"):
                del module.activation_quantizer
        bank = LearnedActivationBank(
            self.qat.contract.activation_bits,
            {path: module.in_features for path, module in self.linears.items()},
        ).to(self.device)
        bank.attach(self.linears)
        if hasattr(self.drafter, "qat_activation_bank"):
            self.drafter.qat_activation_bank = bank
        return bank

    def _adapter(self):
        adapter = self.provider.make_step_adapter(self.drafter)
        if set(adapter.linears) != set(self.linears) or any(
            adapter.linears[p] is not m for p, m in self.linears.items()
        ):
            raise ValueError("adapter bypasses current installed projections")
        return adapter

    def _revalidate(self, qat):
        if self.source_revalidator is not None:
            fresh = self.source_revalidator(qat)
            require_provider(fresh)
            if fresh.source_metadata != self.source:
                raise ValueError("source revalidation changed immutable training ancestry")
            self.provider = fresh
        elif self.provider.source_metadata != self.source:
            raise ValueError("training source metadata changed")

    def _sync(self):
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(self.device)

    def _account_occupancy(self):
        self._sync()
        now = self.clock()
        if self._accounted_at is not None:
            seconds = now - self._accounted_at
            if not math.isfinite(seconds) or seconds < 0:
                raise RuntimeError("invalid synchronized occupancy clock")
            self.occupancy_seconds += seconds
        self._accounted_at = now
        if self.occupancy_seconds > sum(s.gpu_seconds for s in self.curriculum.stages):
            self.budget_failed = True
            raise RuntimeError("measured model residency exceeds total curriculum GPU budget")

    def resources(self, *, load=False):
        result = {"disk_free_bytes": shutil.disk_usage(self.run_dir).free}
        if result["disk_free_bytes"] < self.config.min_free_disk_bytes:
            raise RuntimeError("disk safety floor exceeded")
        if torch.device(self.device).type == "cuda":
            result.update(
                require_host_memory(
                    linux_host_memory(),
                    floor_bytes=self.config.min_host_available_bytes,
                    additional_bytes=self.config.cpu_load_headroom_bytes if load else 0,
                    stage="curriculum CPU model load" if load else "curriculum training/checkpoint",
                )
            )
            free, total = torch.cuda.mem_get_info(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            peak = torch.cuda.max_memory_reserved(self.device)
            if free < self.config.min_cuda_free_bytes:
                raise RuntimeError("CUDA whole-device free memory below safety floor")
            if max(reserved, peak) > self.config.max_cuda_reserved_bytes:
                raise RuntimeError("CUDA allocator safety ceiling exceeded")
            result.update(
                cuda_free_bytes=free,
                cuda_total_bytes=total,
                cuda_reserved_bytes=reserved,
                cuda_peak_reserved_bytes=peak,
            )
        return result

    def status(self, status, **extra):
        atomic_json(
            self.run_dir / "status.json",
            {
                "schema": SCHEMA,
                "status": status,
                "pid": os.getpid(),
                "heartbeat_unix": time.time(),
                "global_updates": self.state.global_updates,
                "optimization_started": self.state.global_updates > 0,
                "model_phase": self.model_phase,
                "curriculum": self.state.state_dict(),
                "epoch": self.epoch,
                "cursor": self.cursor,
                "gpu_seconds": self.occupancy_seconds,
                "measurement_device": self.device,
                "timing": "synchronized wall occupancy; CPU fixture time is not CUDA performance",
                "unique_rows": len(self.unique_rows),
                "unique_prompts": len(self.unique_prompts),
                "checkpoint": self.last_checkpoint,
                "budget_failed": self.budget_failed,
                **extra,
            },
        )

    def _core_states(self):
        result = {}
        for path, module in self.linears.items():
            result[path] = {
                k: v
                for k, v in module.state_dict().items()
                if not k.startswith(("activation_quantizer.", "fusion_correction."))
            }
        return result

    def _model_payload(self):
        correction = getattr(self.linears["fc"], "fusion_correction", None)
        if self.bank is not None:
            self.bank.validate_attachment(self.linears)
        return {
            "linears": self._core_states(),
            "activation_bank": None if self.bank is None else self.bank.checkpoint(),
            "fusion_correction": None if correction is None else correction.state_payload(),
        }

    def save(self):
        self.resources()
        self._account_occupancy()
        folder = self.run_dir / "checkpoints"
        folder.mkdir(exist_ok=True)
        name = f"update-{self.state.global_updates:09d}-{uuid.uuid4().hex}.pt"
        path, temporary = folder / name, folder / (name + ".tmp")
        raw_payload = {
            "schema": SCHEMA,
            "contract": self.contract,
            "state": self.state.state_dict(),
            "model_phase": self.model_phase,
            "model": self._model_payload(),
            "optimizer": self.optimizer.state_dict(),
            "base_lrs": self.base_lrs,
            "rng": self.rng,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "occupancy_seconds": self.occupancy_seconds,
            "smoke_seconds": self.smoke_seconds,
            "transition_seconds": self.transition_seconds,
            "unique_rows": sorted(self.unique_rows),
            "unique_prompts": sorted(self.unique_prompts),
        }
        if torch.device(self.device).type == "cuda":

            def tensor_bytes(value):
                if isinstance(value, torch.Tensor):
                    return value.numel() * value.element_size()
                if isinstance(value, dict):
                    return sum(tensor_bytes(v) for v in value.values())
                if isinstance(value, (tuple, list)):
                    return sum(tensor_bytes(v) for v in value)
                return 0

            require_host_memory(
                linux_host_memory(),
                floor_bytes=self.config.min_host_available_bytes,
                additional_bytes=tensor_bytes(raw_payload) + 16 * 1024**2,
                stage="curriculum atomic checkpoint CPU buffers",
            )
        payload = _cpu_tree(raw_payload)
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._account_occupancy()
        pointer = {
            "schema": SCHEMA,
            "file": name,
            "sha256": sha256(path),
            "global_updates": self.state.global_updates,
            "model_phase": self.model_phase,
            "occupancy_seconds": self.occupancy_seconds,
        }
        atomic_json(self.run_dir / "latest.json", pointer)
        self.last_checkpoint = pointer
        retained = sorted(folder.glob("update-*.pt"), key=lambda p: p.stat().st_mtime_ns)
        for old in retained[: -self.config.keep_checkpoints]:
            if old.name != name:
                old.unlink()
        self._account_occupancy()
        return pointer

    def _load_model(self, payload):
        if set(payload) != {"linears", "activation_bank", "fusion_correction"}:
            raise ValueError("model checkpoint inventory differs")
        core = self._core_states()
        if set(payload["linears"]) != set(core):
            raise ValueError("checkpoint projection inventory differs")
        # Validate the full core and tiny canonical recipes before any mutation.
        for path, expected in core.items():
            values = payload["linears"][path]
            if set(values) != set(expected):
                raise ValueError("checkpoint core inventory differs or contains shared aliases")
            for key, value in values.items():
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != expected[key].shape
                    or value.dtype != expected[key].dtype
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError("invalid core checkpoint tensor")
                if key in ("initial_scale", "frozen_bias") and not torch.equal(
                    value.cpu(), expected[key].cpu()
                ):
                    raise ValueError("checkpoint changed frozen binary operands")
                if key == "latent_sign" and bool((value.abs() > 1).any()):
                    raise ValueError("checkpoint latent outside projected interval")
            if bool((values["initial_scale"] + values["scale_offset"] < 0).any()):
                raise ValueError("checkpoint scales outside projected bounds")
        if (self.bank is None) != (payload["activation_bank"] is None):
            raise ValueError("checkpoint activation bank contract differs")
        if self.bank is not None:
            from .learned_activation import LearnedActivationBank

            probe = LearnedActivationBank(
                self.qat.contract.activation_bits,
                {path: m.in_features for path, m in self.linears.items()},
            )
            probe.load_checkpoint(payload["activation_bank"])
        correction = getattr(self.linears["fc"], "fusion_correction", None)
        if (correction is None) != (payload["fusion_correction"] is None):
            raise ValueError("checkpoint fusion correction contract differs")
        if correction is not None:
            from .fusion_correction import FusionCorrection

            probe = FusionCorrection(
                correction.in_features, correction.out_features, correction.config
            )
            probe.load_payload(payload["fusion_correction"])
        with torch.no_grad():
            for path, expected in core.items():
                for key, value in payload["linears"][path].items():
                    expected[key].copy_(value)
        if self.bank is not None:
            self.bank.load_checkpoint(payload["activation_bank"])
            self.bank.validate_attachment(self.linears)
        if correction is not None:
            correction.load_payload(payload["fusion_correction"])

    def resume(self):
        self._account_occupancy()
        reconstruction_seconds = self.occupancy_seconds
        status_path = self.run_dir / "status.json"
        if (
            status_path.is_file()
            and json.loads(status_path.read_text()).get("budget_failed") is True
        ):
            raise ValueError("budget-overrun run requires an explicit new experiment budget")
        self._revalidate(self.qat)
        pointer = json.loads((self.run_dir / "latest.json").read_text())
        if (
            pointer.get("schema") != SCHEMA
            or not isinstance(pointer.get("file"), str)
            or Path(pointer["file"]).name != pointer["file"]
        ):
            raise ValueError("invalid atomic checkpoint pointer")
        path = self.run_dir / "checkpoints" / pointer["file"]
        if sha256(path) != pointer.get("sha256"):
            raise ValueError("checkpoint payload hash differs")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SCHEMA or payload.get("contract") != self.contract:
            raise ValueError("resume changes config/source/math/runtime contract")
        self.state.load_state_dict(payload["state"])
        phase = payload["model_phase"]
        if (
            type(phase) is not int
            or not 0 <= phase < len(self.curriculum.stages)
            or phase not in (self.state.phase_index, self.state.phase_index - 1)
            or pointer["model_phase"] != phase
            or pointer["global_updates"] != self.state.global_updates
            or len(self.state.transitions) != phase
        ):
            raise ValueError("checkpoint model/phase/transition ancestry differs")
        self.model_phase = phase
        self._bind_phase(phase)
        self._load_model(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload["base_lrs"] != self.base_lrs:
            raise ValueError("checkpoint optimizer rates differ")
        for name in ("cursor", "epoch"):
            value = payload[name]
            if type(value) is not int or value < 0:
                raise ValueError("invalid checkpoint provider cursor")
            setattr(self, name, value)
        for name in ("occupancy_seconds", "smoke_seconds", "transition_seconds"):
            value = payload[name]
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid checkpoint timing")
            setattr(self, name, value)
        durable_seconds = pointer.get("occupancy_seconds")
        if (
            isinstance(durable_seconds, bool)
            or not isinstance(durable_seconds, (float, int))
            or not math.isfinite(durable_seconds)
            or durable_seconds < self.occupancy_seconds
        ):
            raise ValueError("invalid checkpoint residency accounting")
        self.occupancy_seconds = durable_seconds + reconstruction_seconds
        self.unique_rows, self.unique_prompts = (
            set(payload["unique_rows"]),
            set(payload["unique_prompts"]),
        )
        self.rng = payload["rng"]
        restore_rng(self.rng, self.device)
        self.last_checkpoint = pointer
        self._sync()
        self._accounted_at = self.clock()
        self._account_occupancy()

    def _bind_phase(self, phase):
        self.qat = replace(
            self.qat_template, contract=W1AxContract(self.curriculum.stages[phase].activation_bits)
        )
        for module in self.linears.values():
            module.contract = self.qat.contract
        self.bank = self._activation_bank()
        self._revalidate(self.qat)
        self.adapter = self._adapter()
        self.optimizer = joint_optimizer(self.linears, self.qat)
        self.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def _transition(self):
        phase = self.state.phase_index
        if self.state.complete or phase == self.model_phase:
            return
        if phase != self.model_phase + 1 or len(self.state.transitions) != self.model_phase:
            raise ValueError("invalid stage transition ancestry")
        self.save()  # Durable phase-boundary source before any contract mutation.
        source = self.last_checkpoint["sha256"]
        begin = self.clock()
        # Parameters and effective scales stay exactly the same objects/values.
        # Only activations and fresh optimizer moments are replaced.
        previous = self.qat.contract.activation_bits
        self._bind_phase(phase)
        self.model_phase = phase
        self.state.record_transition(
            source_bits=previous,
            target_bits=self.qat.contract.activation_bits,
            source_checkpoint_sha256=source,
            target_layout_sha256=_digest(binary_layout(self.linears)),
        )
        self._sync()
        self.transition_seconds += self.clock() - begin
        self.save()

    def _forward(self, batch):
        device_batch = replace(batch, raw_target_features=batch.raw_target_features.to(self.device))
        # forward_torch_round always constructs a new cache from current weights;
        # no recurrent cache or graph survives a backward/optimizer boundary.
        with shared_round_hard_signs(self.linears):
            return forward_torch_round(device_batch, self.adapter, self.provider.draft_vocab_size)

    def prepare(self, batch):
        """Forward/backward every declared precision with strictly zero real updates."""
        if self.state.global_updates != 0:
            raise ValueError("prepare-only refuses resumed optimizer progress")
        original = self._model_payload()
        initial_rng = _cpu_tree(self.rng)
        begin = self.clock()
        report = []
        try:
            for phase, stage in enumerate(self.curriculum.stages):
                self._bind_phase(phase)
                audit = audit_provider_round(batch, self.provider)
                if len(batch.prefix_token_ids) > self.config.max_prefix_tokens:
                    raise ValueError("smoke prefix exceeds memory-safe cap")
                if not any(audit.ce_mask[1:]):
                    raise ValueError("smoke must supervise an attached later proposal")
                self.resources()
                self.optimizer.zero_grad(set_to_none=True)
                moment_probe = []
                if torch.device(self.device).type == "cuda":
                    # Adam-like moment occupancy without an optimizer update.
                    moment_probe = [
                        torch.zeros_like(p)
                        for group in self.optimizer.param_groups
                        for p in group["params"]
                        for _ in range(2)
                    ]
                logits = self._forward(batch)
                loss = supported_prefix_ce(logits, audit)
                if not loss.requires_grad or not bool(torch.isfinite(loss)):
                    raise ValueError("smoke loss must be finite and attached")
                loss.backward()
                parameters = [p for g in self.optimizer.param_groups for p in g["params"]]
                if any(
                    p.grad is not None and not bool(torch.isfinite(p.grad).all())
                    for p in parameters
                ):
                    raise ValueError("smoke gradients are nonfinite")
                if not any(p.grad is not None and bool((p.grad != 0).any()) for p in parameters):
                    raise ValueError("smoke has no trainable gradient")
                self.optimizer.zero_grad(set_to_none=True)
                self._sync()
                report.append(
                    {
                        "activation_bits": stage.activation_bits,
                        "loss": float(loss.detach()),
                        "optimizer_updates": 0,
                    }
                )
                del logits, loss, moment_probe
        finally:
            self._bind_phase(0)
            self._load_model(original)
            self.rng = initial_rng
            restore_rng(self.rng, self.device)
        self._sync()
        self.smoke_seconds += self.clock() - begin
        self.smoke_passed = True
        self.save()
        atomic_json(
            self.run_dir / "preparation-ready.json",
            {
                "schema": SCHEMA,
                "stages": report,
                "optimization_started": False,
                "global_updates": 0,
                "source": self.source,
                "runtime": self.runtime,
                "checkpoint": self.last_checkpoint,
            },
        )
        self.status("prepared", stop_reason="prepare_only", stages_smoked=len(report))
        return report

    def smoke_current(self):
        """Resume admission: attached current-stage backward without an update."""
        batch = next((b for b in self.provider.rounds() if len(b.rows) > 1), None)
        if batch is None:
            raise ValueError("no current train round supports resume smoke")
        audit = audit_provider_round(batch, self.provider)
        if not any(audit.ce_mask[1:]):
            raise ValueError("resume smoke requires later-proposal supervision")
        before_rng = _cpu_tree(self.rng)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            logits = self._forward(batch)
            loss = supported_prefix_ce(logits, audit)
            if not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise ValueError("resume smoke loss must be finite and attached")
            loss.backward()
            parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
            if any(
                p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters
            ):
                raise ValueError("resume smoke gradients must be finite")
            self.resources()
            self._account_occupancy()
            self.smoke_passed = True
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.rng = before_rng
            restore_rng(self.rng, self.device)

    def run(self, *, require_smoke=True, max_new_updates=None):
        if require_smoke and not self.smoke_passed:
            raise ValueError("every-stage forward/backward smoke required before optimization")
        if max_new_updates is not None and (
            type(max_new_updates) is not int or max_new_updates < 1
        ):
            raise ValueError("bounded invocation needs positive max_new_updates")
        starting_updates = self.state.global_updates
        handlers = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                handlers[sig] = signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))
        if self.last_checkpoint is None:
            self.save()
        try:
            while not self.state.complete and not self.stop_requested:
                if (self.run_dir / "STOP").exists():
                    self.stop_requested = True
                    break
                self._transition()
                self._revalidate(self.qat)
                yielded = False
                total_rounds = 0
                for ordinal, batch in enumerate(self.provider.rounds()):
                    total_rounds += 1
                    if ordinal < self.cursor:
                        continue
                    yielded = True
                    if self.stop_requested or (self.run_dir / "STOP").exists():
                        self.stop_requested = True
                        break
                    if self.state.complete or self.state.phase_index != self.model_phase:
                        break
                    if (
                        max_new_updates is not None
                        and self.state.global_updates - starting_updates >= max_new_updates
                    ):
                        self.stop_requested = True
                        break
                    if not self.state.can_start_update(
                        upper_bound_gpu_seconds=self.config.update_upper_bound_seconds
                    ):
                        self.state.finish_phase()
                        self.save()
                        break
                    if len(batch.prefix_token_ids) > self.config.max_prefix_tokens:
                        raise ValueError("accepted prefix exceeds declared memory-safe token cap")
                    audit = audit_provider_round(batch, self.provider)
                    if not any(audit.ce_mask):
                        self.cursor = ordinal + 1
                        continue
                    self.resources()
                    restore_rng(self.rng, self.device)
                    self._sync()
                    begin = self.clock()
                    factor = min(
                        1.0, (self.state.global_updates + 1) / max(1, self.config.warmup_updates)
                    )
                    for group, rate in zip(self.optimizer.param_groups, self.base_lrs):
                        group["lr"] = rate * factor
                    logits = self._forward(batch)
                    metrics = joint_train_step(
                        self.linears, logits, audit, self.optimizer, self.qat
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    self.rng = rng_state(self.device)
                    self._sync()
                    seconds = self.clock() - begin
                    if seconds > self.state.remaining_gpu_seconds:
                        self.budget_failed = True
                    self.state.record_update(
                        activation_bits=self.qat.contract.activation_bits,
                        gpu_seconds=seconds,
                        rows=sum(audit.denominator_mask),
                        tokens=len(batch.rows),
                        supported_rows=sum(audit.ce_mask),
                    )  # Overrun raises before any new committed checkpoint.
                    self.cursor = ordinal + 1
                    self.unique_prompts.add(batch.anchor.prompt_id)
                    self.unique_rows.update(
                        f"{batch.capture_id}:{batch.anchor.prompt_id}:{batch.anchor.round_index}:{i}"
                        for i, flag in enumerate(audit.ce_mask)
                        if flag
                    )
                    del logits
                    self._account_occupancy()
                    metrics.update(
                        global_updates=self.state.global_updates,
                        model_phase=self.model_phase,
                        measured_update_seconds=seconds,
                        warmup_factor=factor,
                    )
                    with (self.run_dir / "metrics.jsonl").open("a") as stream:
                        stream.write(json.dumps(metrics, sort_keys=True, allow_nan=False) + "\n")
                    if (
                        self.state.global_updates % self.config.checkpoint_every == 0
                        or self.state.complete
                        or self.state.phase_index != self.model_phase
                    ):
                        self.save()
                    self.status("running")
                    if self.state.phase_index != self.model_phase:
                        break
                else:
                    if total_rounds == 0:
                        raise ValueError("eligible provider has no train rounds")
                    if self.cursor > total_rounds:
                        raise ValueError("saved cursor exceeds immutable provider inventory")
                    self.cursor = 0
                    self.epoch += 1
                    continue
                if not yielded and not self.stop_requested:
                    raise ValueError("eligible provider yielded no training rounds")
            self.save()
            self.status("completed" if self.state.complete else "stopped")
        except BaseException as error:
            self.optimizer.zero_grad(set_to_none=True)
            try:
                self._account_occupancy()
            except RuntimeError:
                pass
            self.status(
                "failed",
                error=f"{type(error).__name__}: {error}",
                resume_from_last_atomic_checkpoint=True,
                uncommitted_update_may_have_occurred=True,
            )
            raise
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

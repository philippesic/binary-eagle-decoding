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

from .continuous_budget import TrainingBudget
from .continuous_qat import (
    ObservedAdapter,
    atomic_json,
    later_gradient,
    restore_rng,
    rng_state,
    sha256,
)
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
    joint_parameter_families,
    joint_train_step,
    shared_round_hard_signs,
    validate_joint_linears,
)

SCHEMA = "qat_curriculum_runner_v1"
EXTRA_MATH = (
    "qat_curriculum.py",
    "qat_curriculum_runner.py",
    "qat_optimization.py",
    "learned_activation.py",
    "fusion_correction.py",
    "affine_binary.py",
    "qat_initialization.py",
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


def _checkpoint_cpu_copy_bytes(value):
    """Count retained clones and the largest overlapping CUDA transfer.

    _cpu_tree clones every tensor occurrence, including aliases. Its sequential
    detach().cpu().clone() also keeps one transferred CUDA tensor alive while
    allocating that tensor's retained CPU clone. CPU inputs need no transfer.
    Workspace and allocator overhead remain separate from this tensor bound.
    """
    if isinstance(value, torch.Tensor):
        size = value.numel() * value.element_size()
        return size, size if value.device.type == "cuda" else 0
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (tuple, list)):
        children = value
    else:
        return 0, 0
    retained, largest_transfer = 0, 0
    for child in children:
        child_retained, child_transfer = _checkpoint_cpu_copy_bytes(child)
        retained += child_retained
        largest_transfer = max(largest_transfer, child_transfer)
    return retained, largest_transfer


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
        initialization=None,
        initialization_sha256=None,
        initialization_policy="preserve_reference_magnitudes",
        initialization_encoding="policy_latents",
        reserve_optimizer_memory=True,
        training_admission=None,
        crash_safe_budget=False,
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
        self.reserve_optimizer_memory = reserve_optimizer_memory
        if training_admission is not None:
            from .qat_admission import VerifiedTrainingAdmission

            if not isinstance(training_admission, VerifiedTrainingAdmission):
                raise ValueError("typed current-package production admission required")
        self.training_admission = training_admission
        self.budget = None
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
            "initialization_sha256": initialization_sha256,
            "initialization_policy": initialization_policy,
            "initialization_encoding": initialization_encoding,
            "crash_safe_budget": crash_safe_budget,
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
        if crash_safe_budget:
            self.budget = TrainingBudget(
                self.run_dir / "budget-used.json",
                _digest(self.contract),
                sum(stage.gpu_seconds for stage in curriculum.stages),
                atomic_json,
            )
            self.budget.begin(0.0)
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
        from .qat_initialization import apply_binary_initialization

        if initialization is not None and (
            not isinstance(initialization_sha256, str)
            or len(initialization_sha256) != 64
            or any(c not in "0123456789abcdef" for c in initialization_sha256)
        ):
            raise ValueError("calibrated initializer requires immutable SHA256")
        self.drafter.qat_initialization_report = apply_binary_initialization(
            self.linears,
            initialization,
            policy=initialization_policy,
            encoding=initialization_encoding,
        )
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
        if self.budget is not None:
            self.occupancy_seconds = max(self.occupancy_seconds, self.budget.elapsed())
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
                if not k.startswith(
                    ("activation_quantizer.", "fusion_correction.", "affine_binary.")
                )
            }
        return result

    def _affine_bank(self):
        if not any(getattr(m, "affine_binary", None) is not None for m in self.linears.values()):
            return None
        from .affine_binary import AffineBinaryBank

        return AffineBinaryBank.from_attached(self.linears)

    def _model_payload(self):
        correction = getattr(self.linears["fc"], "fusion_correction", None)
        affine = self._affine_bank()
        if self.bank is not None:
            self.bank.validate_attachment(self.linears)
        return {
            "linears": self._core_states(),
            "activation_bank": None if self.bank is None else self.bank.checkpoint(),
            "fusion_correction": None if correction is None else correction.state_payload(),
            "affine_bank": None if affine is None else affine.state_payload(),
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
            retained_bytes, transfer_bytes = _checkpoint_cpu_copy_bytes(raw_payload)
            require_host_memory(
                linux_host_memory(),
                floor_bytes=self.config.min_host_available_bytes,
                additional_bytes=retained_bytes + transfer_bytes + 16 * 1024**2,
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
        if set(payload) != {"linears", "activation_bank", "fusion_correction", "affine_bank"}:
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
        affine = self._affine_bank()
        if (affine is None) != (payload["affine_bank"] is None):
            raise ValueError("checkpoint affine midpoint contract differs")
        if affine is not None:
            from .affine_binary import AffineBinaryBank, AffineBinaryMidpoint

            probe = AffineBinaryBank(
                affine.config,
                {
                    path: AffineBinaryMidpoint(affine.dimensions[path][1], affine.config)
                    for path in affine.declared_paths
                },
                affine.dimensions,
            )
            probe.load_payload(payload["affine_bank"])
        with torch.no_grad():
            for path, expected in core.items():
                for key, value in payload["linears"][path].items():
                    expected[key].copy_(value)
        if self.bank is not None:
            self.bank.load_checkpoint(payload["activation_bank"])
            self.bank.validate_attachment(self.linears)
        if correction is not None:
            correction.load_payload(payload["fusion_correction"])
        if affine is not None:
            affine.load_payload(payload["affine_bank"])

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
        from .qat_state import validate_optimizer_resume

        validate_optimizer_resume(
            self.optimizer,
            payload["optimizer"],
            expected_updates=self.state.phases[phase]["updates"],
        )
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

    def _forward(self, batch, *, adapter=None, execution_metadata=None):
        device_batch = replace(batch, raw_target_features=batch.raw_target_features.to(self.device))
        # forward_torch_round always constructs a new cache from current weights;
        # no recurrent cache or graph survives a backward/optimizer boundary.
        with shared_round_hard_signs(self.linears):
            return forward_torch_round(
                device_batch,
                self.adapter if adapter is None else adapter,
                self.provider.draft_vocab_size,
                optimize_cache=self.qat.optimize_cache,
                optimize_head=self.qat.optimize_head,
                context_chunk_size=self.qat.context_chunk_size,
                execution_metadata=execution_metadata,
            )

    def _smoke_round(self, batch):
        """All-nine and attached later-state gates; never calls optimizer.step."""
        audit = audit_provider_round(batch, self.provider)
        if len(batch.prefix_token_ids) > self.config.max_prefix_tokens:
            raise ValueError("smoke prefix exceeds memory-safe cap")
        if not any(audit.ce_mask[1:]):
            raise ValueError("smoke must supervise an attached later proposal")
        self.resources()
        validate_joint_linears(self.linears, self.qat)
        declared = {}
        for module in self.linears.values():
            parameters = [module.latent_sign, module.scale_offset]
            for name in ("activation_quantizer", "fusion_correction", "affine_binary"):
                attached = getattr(module, name, None)
                if attached is not None:
                    parameters.extend(attached.parameters())
            for parameter in parameters:
                if not parameter.requires_grad:
                    raise ValueError("declared smoke parameter must remain trainable")
                declared[id(parameter)] = parameter
        owned = [p for group in self.optimizer.param_groups for p in group["params"]]
        if len(owned) != len(declared) or {id(p) for p in owned} != set(declared):
            raise ValueError("smoke optimizer ownership differs from declared parameters")
        self.optimizer.zero_grad(set_to_none=True)
        moment_probe = []
        if (
            torch.device(self.device).type == "cuda"
            and self.reserve_optimizer_memory
            and not self.optimizer.state
        ):
            moment_probe = [
                torch.zeros_like(p)
                for group in self.optimizer.param_groups
                for p in group["params"]
                for _ in range(2)
            ]
        observer = ObservedAdapter(self.adapter)
        try:
            execution = {}
            logits = self._forward(batch, adapter=observer, execution_metadata=execution)
            execution.update(
                context_cache_calls=observer.context_cache_calls,
                optimize_cache_requested=self.qat.optimize_cache,
            )
            diagnostics = later_gradient(logits, audit, observer)
            recurrent_keys = (
                "later_state_gradient_norm",
                "later_k_gradient_norm",
                "later_v_gradient_norm",
            )
            if any(
                diagnostics.get(key) is None
                or not math.isfinite(diagnostics[key])
                or diagnostics[key] <= 0
                for key in recurrent_keys
            ):
                raise ValueError("smoke failed attached later-position state/K/V gate")
            loss = supported_prefix_ce(logits, audit)
            if not loss.requires_grad or not bool(torch.isfinite(loss)):
                raise ValueError("smoke loss must be finite and attached")
            loss.backward()
            families = joint_parameter_families(self.linears)
            binary_report = {}
            for path, module in self.linears.items():
                gradients = {}
                for family, parameter in (
                    ("sign", module.latent_sign),
                    ("scale", module.scale_offset),
                ):
                    grad = parameter.grad
                    if (
                        grad is None
                        or not bool(torch.isfinite(grad).all())
                        or not bool((grad != 0).any())
                    ):
                        raise ValueError(f"{path}.{family}: all-nine finite/nonzero gradient gate")
                    gradients[family + "_gradient_norm"] = float(grad.detach().norm())
                binary_report[path] = gradients
            extra_report = {}
            for family in ("activation", "fusion", "midpoint"):
                parameters = families[family]
                if any(
                    p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in parameters
                ):
                    raise ValueError(f"{family}: declared optional-parameter finite-gradient gate")
                extra_report[family] = {
                    "parameter_tensors": len(parameters),
                    "finite_gradient_tensors": len(parameters),
                    "nonzero_gradient_tensors": sum(bool((p.grad != 0).any()) for p in parameters),
                    "gradient_norms": [float(p.grad.detach().norm()) for p in parameters],
                }
            self._sync()
            resources = self.resources()
            return {
                "activation_bits": self.qat.contract.activation_bits,
                "execution": execution,
                "loss": float(loss.detach()),
                "optimizer_updates": 0,
                "all_nine_binary_gradients_passed": True,
                "later_state_kv_passed": True,
                "binary_gradients": binary_report,
                "optional_gradients": extra_report,
                "resources": resources,
                "training_memory": {
                    "status": "PASS" if moment_probe or self.optimizer.state else "PENDING",
                    "reserved_moment_bytes": sum(
                        t.numel() * t.element_size() for t in moment_probe
                    ),
                    "optimizer_moment_tensors": sum(
                        sum(isinstance(v, torch.Tensor) and k != "step" for k, v in state.items())
                        for state in self.optimizer.state.values()
                    ),
                    "resources_gradients_resident": resources,
                },
                **diagnostics,
            }
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            del observer, moment_probe

    def prepare(self, batch):
        """Strong all-stage forward/backward admission, with zero real updates."""
        if self.state.global_updates != 0:
            raise ValueError("prepare-only refuses resumed optimizer progress")
        self.smoke_passed = False
        original = self._model_payload()
        initial_rng = _cpu_tree(self.rng)
        begin = self.clock()
        report = []
        try:
            for phase in range(len(self.curriculum.stages)):
                self._bind_phase(phase)
                report.append(self._smoke_round(batch))
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

    def smoke_current(self, batch=None):
        """Strong resumed-stage admission without changing weights/moments/RNG."""
        self.smoke_passed = False
        if batch is None:
            batch = next((b for b in self.provider.rounds() if len(b.rows) > 1), None)
        if batch is None:
            raise ValueError("no current train round supports resume smoke")
        before_rng = _cpu_tree(self.rng)
        try:
            report = self._smoke_round(batch)
            self._account_occupancy()
            self.smoke_passed = True
            atomic_json(
                self.run_dir / "resume-smoke.json",
                {
                    "schema": SCHEMA,
                    "global_updates": self.state.global_updates,
                    "source": self.source,
                    "runtime": self.runtime,
                    "checkpoint": self.last_checkpoint,
                    "stage": report,
                },
            )
            return report
        finally:
            self.optimizer.zero_grad(set_to_none=True)
            self.rng = before_rng
            restore_rng(self.rng, self.device)

    def require_optimization_readiness(self):
        if self.training_admission is not None:
            self.training_admission.require_for_qat(self.qat)
            return
        from .qat_readiness import curriculum_readiness_config, require_measured_cuda_readiness

        config = curriculum_readiness_config(self.qat_template, self.config, self.curriculum)
        require_measured_cuda_readiness(
            config,
            source_sha256=_digest(self.source),
            runtime_identity=self.runtime,
            device=self.device,
            max_cuda_reserved_bytes=self.config.max_cuda_reserved_bytes,
            min_cuda_free_bytes=self.config.min_cuda_free_bytes,
        )

    def run(self, *, require_smoke=True, max_new_updates=None):
        self.require_optimization_readiness()
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
            if self.budget is not None:
                self.occupancy_seconds = max(self.occupancy_seconds, self.budget.finish())
            for sig, handler in handlers.items():
                signal.signal(sig, handler)

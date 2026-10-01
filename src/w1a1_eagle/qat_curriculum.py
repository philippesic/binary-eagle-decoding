"""Bounded precision/depth curricula and native trajectory-refresh requests.

These APIs prepare new experiments. They neither change the existing paired
A8/A1 run nor claim improved native acceptance. GPU time budgets are measured
training time; capture/audit/export costs are separately accounted here.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .qat_optimization import (
    BinaryOptimizationConfig,
    binary_layout,
    binary_parameter_families,
    make_binary_optimizer,
)
from .recurrent_loss import supported_prefix_ce, supported_prefix_rows
from .recurrent_trace import TraceAudit
from .trajectory_refresh import build_plan, checked_hash, digest


def depth_weighted_supported_ce(
    logits: Tensor, audit: TraceAudit, depths: Sequence[int], *, decay: float = 1.0
) -> Tensor:
    """Normalized supported-label CE; unsupported quality denominator stays intact.

    Depths must come from the audited trace in the same row order. Unsupported
    earlier states remain attached through later logits. decay=1 retains the
    baseline mean CE exactly; 0.8 is an explicit mild early-depth experiment.
    """
    if isinstance(decay, bool) or not math.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError("depth decay must be finite in (0,1]")
    if len(depths) != len(audit.draft_labels) or any(
        type(d) is not int or not 0 <= d < 5 for d in depths
    ):
        raise ValueError("depths must align with audited rows in the supported five-step horizon")
    if decay == 1:
        return supported_prefix_ce(logits, audit)
    supported_logits, supported_labels = supported_prefix_rows(logits, audit)
    mask = torch.tensor(audit.ce_mask, device=logits.device, dtype=torch.bool)
    weights = torch.tensor([decay**d for d in depths], device=logits.device, dtype=torch.float32)[
        mask
    ]
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("depth weighting underflow; preserve every supported depth")
    losses = F.cross_entropy(supported_logits, supported_labels, reduction="none")
    return (weights * losses).sum() / weights.sum()


def depth_exposure(audit: TraceAudit, depths: Sequence[int], *, decay: float = 1.0) -> dict:
    """Separate supported loss weights and all-valid quality exposures."""
    if len(depths) != len(audit.draft_labels) or any(
        type(d) is not int or not 0 <= d < 5 for d in depths
    ):
        raise ValueError("depth exposure rows differ from audited horizon")
    if isinstance(decay, bool) or not math.isfinite(decay) or not 0 < decay <= 1:
        raise ValueError("invalid depth decay")
    if len(audit.ce_mask) != len(depths) or len(audit.denominator_mask) != len(depths):
        raise ValueError("depth exposure audit masks differ")
    return {
        str(d): {
            "quality_rows": sum(
                depth == d and valid for depth, valid in zip(depths, audit.denominator_mask)
            ),
            "supported_rows": sum(
                depth == d and valid for depth, valid in zip(depths, audit.ce_mask)
            ),
            "supported_weight": sum(
                decay**d for depth, valid in zip(depths, audit.ce_mask) if depth == d and valid
            ),
        }
        for d in sorted(set(depths))
    }


@dataclass(frozen=True)
class PrecisionStage:
    activation_bits: int
    gpu_seconds: float
    max_updates: int

    def __post_init__(self):
        if type(self.activation_bits) is not int or self.activation_bits not in (1, 4, 8):
            raise ValueError("precision stages support A8/A4/A1")
        if (
            isinstance(self.gpu_seconds, bool)
            or not math.isfinite(self.gpu_seconds)
            or self.gpu_seconds <= 0
        ):
            raise ValueError("stage GPU budget must be finite and positive")
        if type(self.max_updates) is not int or self.max_updates < 1:
            raise ValueError("stage update budget must be positive")


@dataclass(frozen=True)
class CurriculumConfig:
    stages: tuple[PrecisionStage, ...]
    optimizer_transition: str = "fresh"

    def __post_init__(self):
        bits = tuple(stage.activation_bits for stage in self.stages)
        if bits not in ((1,), (8, 1), (8, 4, 1)):
            raise ValueError("curriculum must be direct A1, A8→A1, or explicit A8→A4→A1")
        if self.optimizer_transition != "fresh":
            raise ValueError("curriculum transfers use explicitly fresh optimizer state")
        if not math.isfinite(sum(stage.gpu_seconds for stage in self.stages)):
            raise ValueError("total curriculum GPU budget overflow")

    def manifest(self) -> dict:
        return {
            "stages": [asdict(stage) for stage in self.stages],
            "optimizer_transition": self.optimizer_transition,
        }

    @classmethod
    def direct_a1(cls, *, gpu_seconds: float, max_updates: int):
        return cls((PrecisionStage(1, gpu_seconds, max_updates),))

    @classmethod
    def a8_to_a1(
        cls,
        *,
        gpu_seconds: float,
        a8_fraction: float = 0.2,
        a8_max_updates: int,
        a1_max_updates: int,
    ):
        if (
            isinstance(a8_fraction, bool)
            or not math.isfinite(a8_fraction)
            or not 0 < a8_fraction < 0.5
        ):
            raise ValueError("short A8 warm-start requires a finite fraction in (0,.5)")
        return cls(
            (
                PrecisionStage(8, gpu_seconds * a8_fraction, a8_max_updates),
                PrecisionStage(1, gpu_seconds * (1 - a8_fraction), a1_max_updates),
            )
        )


class CurriculumState:
    """Strict serializable counters bound to a new experiment's data/model.

    Each update is assigned wholly to its current precision phase. A caller
    reserves a bounded duration before launch and records measured GPU time.
    Budget overruns fail closed and require an explicit new experiment budget;
    they must not be silently credited to the next precision. Warmup uses the
    global update count and must not restart at a precision switch.
    """

    def __init__(self, config: CurriculumConfig, *, data_contract: dict, model_contract: dict):
        self.config = config
        self.data_contract_sha256 = digest(data_contract)
        self.model_contract_sha256 = digest(model_contract)
        self.phase_index = 0
        self.phases = [
            {"updates": 0, "gpu_seconds": 0.0, "rows": 0, "tokens": 0, "supported_rows": 0}
            for _ in config.stages
        ]
        self.overhead_gpu_seconds = {
            key: 0.0 for key in ("capture", "audit", "export", "development")
        }
        self.transitions = []

    @property
    def complete(self):
        return self.phase_index == len(self.config.stages)

    @property
    def activation_bits(self):
        return None if self.complete else self.config.stages[self.phase_index].activation_bits

    @property
    def global_updates(self):
        return sum(phase["updates"] for phase in self.phases)

    @property
    def remaining_gpu_seconds(self):
        if self.complete:
            return 0.0
        return (
            self.config.stages[self.phase_index].gpu_seconds
            - self.phases[self.phase_index]["gpu_seconds"]
        )

    def can_start_update(self, *, upper_bound_gpu_seconds: float) -> bool:
        if (
            isinstance(upper_bound_gpu_seconds, bool)
            or not math.isfinite(upper_bound_gpu_seconds)
            or upper_bound_gpu_seconds <= 0
        ):
            raise ValueError("update duration bound must be finite and positive")
        return not self.complete and upper_bound_gpu_seconds <= self.remaining_gpu_seconds

    def record_update(
        self,
        *,
        activation_bits: int,
        gpu_seconds: float,
        rows: int,
        tokens: int,
        supported_rows: int,
    ):
        if self.complete or activation_bits != self.activation_bits:
            raise ValueError("update precision differs from active curriculum phase")
        if len(self.transitions) != self.phase_index:
            raise ValueError(
                "precision switch requires recorded fresh-optimizer transition before update"
            )
        if (
            isinstance(gpu_seconds, bool)
            or not math.isfinite(gpu_seconds)
            or not 0 < gpu_seconds <= self.remaining_gpu_seconds
        ):
            raise ValueError("measured update exceeds current phase GPU budget")
        if (
            any(type(value) is not int or value < 0 for value in (rows, tokens, supported_rows))
            or supported_rows > rows
        ):
            raise ValueError("invalid curriculum exposure counters")
        phase = self.phases[self.phase_index]
        phase["updates"] += 1
        phase["gpu_seconds"] += gpu_seconds
        phase["rows"] += rows
        phase["tokens"] += tokens
        phase["supported_rows"] += supported_rows
        stage = self.config.stages[self.phase_index]
        if phase["updates"] == stage.max_updates or phase["gpu_seconds"] == stage.gpu_seconds:
            self.phase_index += 1

    def finish_phase(self):
        """Stop early when the next bounded update cannot fit; keep unused budget."""
        if (
            self.complete
            or self.phases[self.phase_index]["updates"] == 0
            or len(self.transitions) != self.phase_index
        ):
            raise ValueError("cannot finish empty or complete phase")
        self.phase_index += 1

    def record_overhead(self, kind: str, gpu_seconds: float):
        if (
            kind not in self.overhead_gpu_seconds
            or isinstance(gpu_seconds, bool)
            or not math.isfinite(gpu_seconds)
            or gpu_seconds < 0
        ):
            raise ValueError("invalid separately measured curriculum overhead")
        self.overhead_gpu_seconds[kind] += gpu_seconds
        if not math.isfinite(self.overhead_gpu_seconds[kind]):
            raise ValueError("curriculum overhead overflow")

    def record_transition(
        self,
        *,
        source_bits: int,
        target_bits: int,
        source_checkpoint_sha256: str,
        target_layout_sha256: str,
    ) -> dict:
        """Register a checked in-place switch with a newly created optimizer.

        The trainer owns the proof that parameters stayed equal and that the
        optimizer has no state. Call this after replacing activation contracts
        and creating that fresh optimizer, before the first destination update.
        """
        checked_hash(source_checkpoint_sha256)
        checked_hash(target_layout_sha256)
        index = self.phase_index
        if (
            self.complete
            or index == 0
            or len(self.transitions) != index - 1
            or self.phases[index]["updates"] != 0
            or (source_bits, target_bits)
            != tuple(stage.activation_bits for stage in self.config.stages[index - 1 : index + 1])
        ):
            raise ValueError("invalid or repeated curriculum transition")
        manifest = {
            "source_bits": source_bits,
            "target_bits": target_bits,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "optimizer_state": "fresh",
            "global_updates": self.global_updates,
            "source_gpu_seconds": self.phases[index - 1]["gpu_seconds"],
            "data_contract_sha256": self.data_contract_sha256,
            "model_contract_sha256": self.model_contract_sha256,
            "target_layout_sha256": target_layout_sha256,
        }
        self.transitions.append(manifest)
        return copy.deepcopy(manifest)

    def state_dict(self):
        return copy.deepcopy(
            {
                "schema": "qat_curriculum_state_v1",
                "config": self.config.manifest(),
                "data_contract_sha256": self.data_contract_sha256,
                "model_contract_sha256": self.model_contract_sha256,
                "phase_index": self.phase_index,
                "phases": self.phases,
                "overhead_gpu_seconds": self.overhead_gpu_seconds,
                "transitions": self.transitions,
            }
        )

    def load_state_dict(self, saved: dict):
        expected = self.state_dict()
        for key in ("schema", "config", "data_contract_sha256", "model_contract_sha256"):
            if saved.get(key) != expected[key]:
                raise ValueError("resume changes curriculum budget/data/model contract")
        index, phases = saved.get("phase_index"), saved.get("phases")
        if (
            type(index) is not int
            or not 0 <= index <= len(self.config.stages)
            or not isinstance(phases, list)
            or len(phases) != len(self.phases)
        ):
            raise ValueError("invalid curriculum phase state")
        for n, (phase, stage) in enumerate(zip(phases, self.config.stages)):
            if not isinstance(phase, dict) or set(phase) != set(self.phases[n]):
                raise ValueError("invalid curriculum exposure inventory")
            if any(
                type(phase[k]) is not int or phase[k] < 0
                for k in ("updates", "rows", "tokens", "supported_rows")
            ):
                raise ValueError("invalid curriculum exposure")
            seconds = phase["gpu_seconds"]
            if (
                isinstance(seconds, bool)
                or not math.isfinite(seconds)
                or not 0 <= seconds <= stage.gpu_seconds
                or phase["updates"] > stage.max_updates
                or phase["supported_rows"] > phase["rows"]
            ):
                raise ValueError("curriculum checkpoint exceeds budget or exposure bounds")
            if (n < index and phase["updates"] == 0) or (n > index and any(phase.values())):
                raise ValueError("curriculum checkpoint phase ancestry differs")
            if n == index and (
                phase["updates"] == stage.max_updates or seconds == stage.gpu_seconds
            ):
                raise ValueError("curriculum checkpoint has unadvanced exhausted phase")
            if phase["updates"] == 0 and any(phase.values()):
                raise ValueError("curriculum exposures lack optimizer updates")
        overhead = saved.get("overhead_gpu_seconds", {})
        if set(overhead) != set(self.overhead_gpu_seconds) or any(
            isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in overhead.values()
        ):
            raise ValueError("invalid curriculum overhead checkpoint")
        transitions = saved.get("transitions")
        if not isinstance(transitions, list) or len(transitions) > min(
            index, len(self.config.stages) - 1
        ):
            raise ValueError("invalid curriculum transition history")
        for n, transition in enumerate(transitions):
            if not isinstance(transition, dict) or set(transition) != {
                "source_bits",
                "target_bits",
                "source_checkpoint_sha256",
                "optimizer_state",
                "global_updates",
                "source_gpu_seconds",
                "data_contract_sha256",
                "model_contract_sha256",
                "target_layout_sha256",
            }:
                raise ValueError("curriculum transition manifest inventory differs")
            if (transition.get("source_bits"), transition.get("target_bits")) != tuple(
                stage.activation_bits for stage in self.config.stages[n : n + 2]
            ):
                raise ValueError("curriculum transition precision ancestry differs")
            checked_hash(transition.get("source_checkpoint_sha256"))
            checked_hash(transition.get("target_layout_sha256"))
            if transition.get("optimizer_state") != "fresh" or transition.get(
                "global_updates"
            ) != sum(p["updates"] for p in phases[: n + 1]):
                raise ValueError("curriculum transition optimizer/counter ancestry differs")
            if (
                transition["source_gpu_seconds"] != phases[n]["gpu_seconds"]
                or transition["data_contract_sha256"] != self.data_contract_sha256
                or transition["model_contract_sha256"] != self.model_contract_sha256
            ):
                raise ValueError("curriculum transition source time/data/model ancestry differs")
        if index < len(phases) and phases[index]["updates"] > 0 and len(transitions) != index:
            raise ValueError("destination updates lack curriculum transition ancestry")
        if index == len(phases) and len(transitions) != len(phases) - 1:
            raise ValueError("completed curriculum lacks transition ancestry")
        self.phase_index, self.phases = index, copy.deepcopy(phases)
        self.overhead_gpu_seconds, self.transitions = (
            copy.deepcopy(overhead),
            copy.deepcopy(transitions),
        )


@torch.no_grad()
def transfer_precision_state_(
    source: Mapping,
    target: Mapping,
    optimizer_config: BinaryOptimizationConfig,
    *,
    curriculum: CurriculumState,
    source_checkpoint_sha256: str,
):
    """Transfer exact latent magnitudes/effective scales, return a fresh optimizer.

    Only activation precision changes: row shape, frozen biases, and all other
    supplied model/data contract fields remain bound in CurriculumState. Calling
    this twice at one transition fails. No source optimizer moments are reused.
    No warmup or exposure counter is reset. Learned activation parameters need
    a separate explicit transfer contract; this API owns binary weights only.
    """
    checked_hash(source_checkpoint_sha256)
    binary_parameter_families(source)
    binary_parameter_families(target)
    index = curriculum.phase_index
    if curriculum.complete or index == 0 or len(curriculum.transitions) != index - 1:
        raise ValueError("transfer requires one completed source phase and no prior transfer")
    source_bits = curriculum.config.stages[index - 1].activation_bits
    target_bits = curriculum.config.stages[index].activation_bits
    if set(source) != set(target):
        raise ValueError("precision transfer projection inventory differs")
    values = {}
    for name in sorted(source):
        src, dst = source[name], target[name]
        if (
            src.contract.activation_bits != source_bits
            or dst.contract.activation_bits != target_bits
            or src.contract.scale_layout != dst.contract.scale_layout
            or src.latent_sign.shape != dst.latent_sign.shape
        ):
            raise ValueError("precision transfer changes non-activation weight contract")
        if src.latent_sign is dst.latent_sign or src.scale_offset is dst.scale_offset:
            raise ValueError("precision transfer requires independent source/target parameters")
        if (
            getattr(src, "_round_hard_signs", None) is not None
            or getattr(dst, "_round_hard_signs", None) is not None
        ):
            raise ValueError("precision transfer cannot mutate an active hard-sign graph cache")
        if (src.frozen_bias is None) != (dst.frozen_bias is None) or (
            src.frozen_bias is not None
            and not torch.equal(src.frozen_bias.detach().cpu(), dst.frozen_bias.detach().cpu())
        ):
            raise ValueError("precision transfer changes frozen bias")
        latent, scale = src.latent_sign.detach(), src.effective_scales().detach()
        raw_scale = src.initial_scale + src.scale_offset
        if (
            not bool(torch.isfinite(latent).all())
            or bool((latent.abs() > 1).any())
            or not bool(torch.isfinite(src.initial_scale).all())
            or bool((src.initial_scale < 0).any())
            or not bool(torch.isfinite(src.scale_offset).all())
            or not bool(torch.isfinite(raw_scale).all())
            or bool((raw_scale < 0).any())
            or not bool(torch.isfinite(scale).all())
            or bool((scale < 0).any())
        ):
            raise ValueError("source precision state outside finite projected bounds")
        if not bool(torch.isfinite(dst.initial_scale).all()) or bool((dst.initial_scale < 0).any()):
            raise ValueError("target initial scales invalid")
        values[name] = (
            latent.to(dst.latent_sign).clone(),
            src.initial_scale.to(dst.initial_scale).clone(),
            src.scale_offset.detach().to(dst.scale_offset).clone(),
        )
    # All projections validated before mutation.
    for name, (latent, initial_scale, offset) in values.items():
        target[name].latent_sign.copy_(latent)
        # Preserve the additive representation, avoiding a subtraction/addition
        # cancellation round trip when the target's initial row scale differs.
        target[name].initial_scale.copy_(initial_scale)
        target[name].scale_offset.copy_(offset)
        target[name].latent_sign.grad = None
        target[name].scale_offset.grad = None
    optimizer = make_binary_optimizer(target, optimizer_config)
    manifest = curriculum.record_transition(
        source_bits=source_bits,
        target_bits=target_bits,
        source_checkpoint_sha256=source_checkpoint_sha256,
        target_layout_sha256=digest(binary_layout(target)),
    )
    return optimizer, manifest


def plan_curriculum_refresh(
    spec: dict,
    *,
    expected_checkpoint_sha256: str,
    expected_export_sha256: str,
    expected_train_prompts_sha256: str,
    expected_native_teacher_contract: dict,
    expected_refresh_round: int,
) -> dict:
    """Use the existing exact-prefix native refresh engine with active bindings.

    A new student prefix produces a new native capture requirement; old rows
    are never relabeled. Result remains training_eligible=False until the
    existing capture auditor/provider admits a new corpus contract. A changed
    corpus must start an explicit new experiment, not exact-resume an old one.
    """
    for value in (
        expected_checkpoint_sha256,
        expected_export_sha256,
        expected_train_prompts_sha256,
    ):
        checked_hash(value)
    if type(expected_refresh_round) is not int or expected_refresh_round < 0:
        raise ValueError("refresh round must be nonnegative")
    if (
        spec.get("student")
        != {
            "checkpoint_sha256": expected_checkpoint_sha256,
            "export_sha256": expected_export_sha256,
        }
        or spec.get("inputs", {}).get("train_prompts", {}).get("sha256")
        != expected_train_prompts_sha256
        or spec.get("native_teacher_contract") != expected_native_teacher_contract
        or spec.get("refresh_round") != expected_refresh_round
    ):
        raise ValueError(
            "refresh request differs from active checkpoint/data/native-teacher contract"
        )
    return build_plan(copy.deepcopy(spec))

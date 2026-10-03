"""Inspect actual A8 recipe attachments and optimizer ownership, without launching.

Execution flags describe requested behavior. Runtime proof is separate, using
observations from an actual forward; learned head training deliberately remains
serial to preserve invocation-local activation-gradient normalization. Neither
these audits nor synthetic fixtures establish native readiness or acceptance.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import torch

from .qat_optimization import BinaryOptimizationConfig, SignFlipDiagnostics
from .recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    joint_parameter_families,
    validate_joint_linears,
)

_RECIPE_FIELDS = (
    "seed",
    "objective",
    "sign_lr",
    "scale_lr",
    "max_grad_norm",
    "a1_computation",
    "activation_quantization",
    "activation_lr",
    "binary_optimization",
    "affine_weights",
    "fusion_correction",
    "depth_loss_decay",
    "optimize_cache",
    "optimize_head",
    "context_chunk_size",
)


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def comparison_training(manifest: Mapping, arm: str) -> dict:
    """Resolve explicit arm overrides; never mix historical checkpoint recipes."""
    if manifest.get("schema") != "qat_a8_comparison_v1":
        raise ValueError("expected qat_a8_comparison_v1")
    if arm not in ("reference", "candidate") or set(manifest.get("arms", {})) != {
        "reference",
        "candidate",
    }:
        raise ValueError("comparison needs exactly reference and candidate")
    training = copy.deepcopy(manifest["shared_training"])
    training.update(copy.deepcopy(manifest["arms"][arm]))
    if training.get("activation_bits") != [8] or training.get("max_seconds") != 7200:
        raise ValueError("comparison must retain A8-only two-hour arm budget")
    if training.get("development_lifecycle") != "standalone":
        raise ValueError("A8 development must run in a separate process")
    if training.get("objective") != "hard_ce":
        raise ValueError("comparison requires matched hard-CE training")
    if training.get("optimize_cache") is not True or training.get("optimize_head") is not True:
        raise ValueError("comparison requires cache/head optimizations in both arms")
    if training.get("fusion_correction") is not None or training.get("depth_loss_decay") != 1:
        raise ValueError("comparison does not admit fusion correction or depth weighting")
    if training.get("curriculum") is not None or training.get("refresh") is not None:
        raise ValueError("comparison does not admit curriculum or trajectory refresh")
    binary = training.get("binary_optimization", {})
    if (
        binary.get("optimizer") != "adamw"
        or binary.get("latent_magnitude") != (0.5 if arm == "reference" else 0.1)
        or any(
            binary.get(k) != v
            for k, v in {
                "sign_lr": 0.001,
                "scale_lr": 0.00001,
                "max_grad_norm": 1.0,
                "sign_gradient_rule": "baseline",
                "scale_gradient_rule": "baseline",
                "clip_policy": "joint",
            }.items()
        )
        or any(training.get(k) != binary[k] for k in ("sign_lr", "scale_lr", "max_grad_norm"))
    ):
        raise ValueError("comparison binary optimization differs from the agreed arm")
    if (
        BinaryOptimizationConfig(**binary).manifest()
        != BinaryOptimizationConfig(latent_magnitude=0.5 if arm == "reference" else 0.1).manifest()
    ):
        raise ValueError("comparison changes baseline AdamW defaults")
    if training.get("seeds") != [8101, 1101] or any(
        training.get(key) is not None for key in ("max_steps", "max_tokens", "max_epochs")
    ):
        raise ValueError("comparison changes matched seeds or introduces another stop budget")
    if training.get("activation_quantization") != ("fixed" if arm == "reference" else "learned"):
        raise ValueError("comparison activation recipe differs from the agreed arm")
    affine = training.get("affine_weights")
    if (arm == "reference" and affine is not None) or (
        arm == "candidate"
        and (
            not isinstance(affine, Mapping)
            or affine.get("enabled") is not True
            or affine.get("coverage") != "all"
        )
    ):
        raise ValueError("comparison must use symmetric reference/all-nine midpoint candidate")
    budget = manifest.get("budget", {})
    if any(
        budget.get(k) != v
        for k, v in {
            "training_seconds_total": 14400,
            "training_seconds_per_arm": 7200,
            "development_seconds_per_transaction": 1200,
            "development_before_training": True,
            "development_at_budget_end": True,
            "sealed_final_allowed": False,
            "a1_held_out": True,
        }.items()
    ):
        raise ValueError("comparison budget/development/split contract differs")
    return _json(training)


def comparison_joint_recipe(manifest: Mapping, arm: str) -> dict:
    training = comparison_training(manifest, arm)
    recipe = {key: training[key] for key in _RECIPE_FIELDS if key in training}
    recipe["seed"] = training["seeds"][0]
    config = JointQATConfig(W1AxContract(8), **recipe)
    return {key: _json(asdict(config)[key]) for key in _RECIPE_FIELDS}


def resolve_comparison_config(base: Mapping, manifest: Mapping, arm: str) -> dict:
    """Produce an ordinary train CLI config while retaining exact source bindings.

    Callers supply authenticated stages/provider/native locators in ``base``.
    This function does not load a model or rewrite a prepared-data manifest.
    """
    if base.get("schema") != "continuous_w1ax_experiment_v1":
        raise ValueError("expected continuous_w1ax_experiment_v1 base")
    result = copy.deepcopy(base)
    result.setdefault("training", {}).update(comparison_training(manifest, arm))
    if result.get("curriculum") is not None or result.get("refresh") is not None:
        raise ValueError("base must not select curriculum or refresh")
    result["comparison"] = {
        "schema": manifest["schema"],
        "arm": arm,
        "initialization": manifest["initialization"],
        "primary_baseline": manifest["primary_baseline"],
        "expected_recipe": comparison_joint_recipe(manifest, arm),
    }
    return _json(result)


def _execution_audit(evidence, *, learned: bool, training: bool, chunk_size: int):
    if evidence is None:
        return {"status": "not_observed", "requested_flags_are_not_runtime_proof": True}
    if not isinstance(evidence, Mapping):
        raise ValueError("execution proof needs actual cache and head observations")
    # This is the forward_torch_round observation plus ObservedAdapter's actual
    # build_context_cache invocation count, rather than requested flags alone.
    if "context_cache_calls" in evidence:
        cache = {
            "requested": True,
            "effective": evidence["context_cache_calls"] > 0,
            "calls": evidence["context_cache_calls"],
            "context_chunk_size": evidence.get("context_chunk_size"),
        }
        head = {
            key: evidence.get(key)
            for key in ("requested", "effective_batched", "path", "reason", "saturation_scope")
        }
    elif set(evidence) == {"cache", "head"}:
        cache, head = evidence["cache"], evidence["head"]
    else:
        raise ValueError("execution proof needs actual cache and head observations")
    if not isinstance(cache, Mapping) or not isinstance(head, Mapping):
        raise ValueError("execution observations must be objects")
    if (
        cache.get("requested") is not True
        or cache.get("effective") is not True
        or type(cache.get("calls")) is not int
        or cache["calls"] < 1
        or cache.get("context_chunk_size") != chunk_size
    ):
        raise ValueError("optimized context-cache path was not observed")
    serial_learned = learned and training
    if (
        head.get("requested") is not True
        or head.get("effective_batched") is not (not serial_learned)
        or head.get("path") != ("serial" if serial_learned else "batched")
        or head.get("reason") != ("trainable_learned_activation" if serial_learned else "batched")
    ):
        raise ValueError("head execution differs from the declared effective path")
    return {"status": "observed", "cache": _json(cache), "head": _json(head)}


def audit_a8_recipe(
    linears: Mapping,
    optimizer: torch.optim.Optimizer,
    config: JointQATConfig,
    expected: Mapping,
    *,
    fresh: bool = False,
    lr_factor: float = 1.0,
    execution_evidence: Mapping | None = None,
    training: bool = True,
) -> dict:
    """Fail closed on mismatched modules, families, optimizer rules or LR schedule.

    ``expected`` is comparison_joint_recipe(...), not model-reported config.
    ``lr_factor`` is the exact caller-known warmup multiplier at this boundary.
    Fresh checks are never applied to positive-step resumes; they enforce the
    deliberate initial latent magnitude and zero midpoint/scale offsets.
    """
    if type(fresh) is not bool or type(training) is not bool:
        raise ValueError("fresh/training controls must be boolean")
    if (
        type(lr_factor) not in (float, int)
        or not math.isfinite(lr_factor)
        or not 0 <= lr_factor <= 1
    ):
        raise ValueError("LR factor must be finite in [0,1]")
    if set(expected) != set(_RECIPE_FIELDS):
        raise ValueError("expected recipe inventory differs")
    actual = {key: _json(asdict(config)[key]) for key in _RECIPE_FIELDS}
    if actual != _json(expected) or config.contract != W1AxContract(8):
        raise ValueError("effective config differs from selected A8 recipe")
    if not config.optimize_cache or not config.optimize_head:
        raise ValueError("comparison requires requested cache/head optimizations")
    validate_joint_linears(linears, config)
    binary = config.binary_optimization
    if binary is None or binary.optimizer != "adamw" or type(optimizer) is not torch.optim.AdamW:
        raise ValueError("comparison requires existing latent-gradient AdamW")
    if getattr(optimizer, "_binary_recipe", None) != binary.manifest():
        raise ValueError("instantiated optimizer recipe differs")
    if (config.sign_lr, config.scale_lr, config.max_grad_norm) != (
        binary.sign_lr,
        binary.scale_lr,
        binary.max_grad_norm,
    ):
        raise ValueError("outer and binary optimizer rates/clipping differ")
    families = joint_parameter_families(linears)
    wanted_counts = {
        "sign": 9,
        "scale": 9,
        "activation": 6 if config.activation_quantization == "learned" else 0,
        "fusion": 0,
        "midpoint": 9 if config.affine_weights is not None else 0,
    }
    if {name: len(values) for name, values in families.items()} != wanted_counts:
        raise ValueError("trainable parameter family coverage differs")
    expected_ids = {id(p) for family in families.values() for p in family}
    trainable_ids = {
        id(p) for module in linears.values() for p in module.parameters() if p.requires_grad
    }
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if (
        trainable_ids != expected_ids
        or len(owned) != len(expected_ids)
        or {id(p) for p in owned} != expected_ids
    ):
        raise ValueError("optimizer must own every declared trainable parameter exactly once")
    rates = {"sign": binary.sign_lr, "scale": binary.scale_lr, "activation": config.activation_lr}
    if config.affine_weights is not None:
        rates["midpoint"] = config.affine_weights.midpoint_lr
    groups = {}
    for group in optimizer.param_groups:
        name = group.get("family")
        if name not in families or name in groups or not families[name]:
            raise ValueError("optimizer family inventory differs")
        if {id(p) for p in group["params"]} != {id(p) for p in families[name]}:
            raise ValueError("optimizer group ownership differs: " + name)
        if (
            group["lr"] != rates[name] * lr_factor
            or tuple(group["betas"]) != tuple(binary.betas)
            or group["eps"] != binary.eps
            or group["weight_decay"] != 0
            or group.get("foreach") is not False
            or group.get("maximize", False) is not False
        ):
            raise ValueError("optimizer group rules/LR differ: " + name)
        groups[name] = {
            "tensors": len(group["params"]),
            "elements": sum(p.numel() for p in group["params"]),
            "lr": group["lr"],
        }
    if set(groups) != {name for name, values in families.items() if values}:
        raise ValueError("optimizer omitted a declared family")
    for name, module in linears.items():
        if module.a1_computation != config.a1_computation:
            raise ValueError("module computation path differs: " + name)
        if not bool(torch.isfinite(module.latent_sign).all()) or bool(
            (module.latent_sign.abs() > 1).any()
        ):
            raise ValueError("latent weights violate finite STE bounds")
        if not bool(torch.isfinite(module.effective_scales()).all()):
            raise ValueError("effective scales are nonfinite")
        if fresh and (
            not bool((module.latent_sign.abs() == binary.latent_magnitude).all())
            or bool((module.scale_offset != 0).any())
        ):
            raise ValueError("fresh latent magnitude/scale initialization differs")
        quantizer = getattr(module, "activation_quantizer", None)
        if quantizer is not None:
            if quantizer.bits != 8 or (fresh and float(quantizer.parameter.detach()) != 1.0):
                raise ValueError("learned A8 quantizer initialization/precision differs")
        affine = getattr(module, "affine_binary", None)
        if affine is not None:
            affine.validate_bound()
            if fresh and bool((affine.midpoint != 0).any()):
                raise ValueError("fresh affine midpoint must be zero")
    return {
        "schema": "qat_a8_effective_recipe_v1",
        "passed": True,
        "activation_bits": 8,
        "projection_count": len(linears),
        "recipe": actual,
        "sign_mechanism": "AdamW_float_latent_surrogate_gradient_zero_crossing",
        "direct_bit_optimizer": False,
        "fresh_initialization_checked": fresh,
        "optimizer_families": groups,
        "execution": _execution_audit(
            execution_evidence,
            learned=config.activation_quantization == "learned",
            training=training,
            chunk_size=config.context_chunk_size,
        ),
    }


class QATUpdateProbe:
    """Observe a real optimizer step without a dense latent-weight shadow.

    Each tensor's maximum-gradient element is selected before step. The report
    records actual displacement there, not a modeled Adam displacement. This
    is bounded sampled movement evidence; it is not the full sign-family L1.
    Hooks run after gradient transformation/clipping and before gradient clear.
    Set ``enabled`` only at admitted diagnostic updates to limit runtime cost.
    ``require_all`` is a decisive prelaunch finite/nonzero/movement gate; later
    trajectory observations can retain zero-gradient tensors without failing.
    """

    def __init__(self, linears: Mapping, optimizer, *, require_all: bool = False):
        if type(require_all) is not bool:
            raise ValueError("require_all must be boolean")
        self.families = {k: list(v) for k, v in joint_parameter_families(linears).items() if v}
        owned = [p for group in optimizer.param_groups for p in group["params"]]
        expected = [p for values in self.families.values() for p in values]
        if len(owned) != len(expected) or {id(p) for p in owned} != {id(p) for p in expected}:
            raise ValueError("update probe optimizer ownership differs")
        self.require_all = require_all
        self.enabled = True
        self.last_report = None
        self._pending = None
        self._hooks = [
            optimizer.register_step_pre_hook(self._before),
            optimizer.register_step_post_hook(self._after),
        ]

    def _before(self, optimizer, args, kwargs):
        if type(self.enabled) is not bool:
            raise ValueError("update probe enabled control must be boolean")
        if not self.enabled:
            self.last_report = None
            return
        self.last_report = None
        self._pending = {}
        for name, parameters in self.families.items():
            rows, indices = [], []
            for p in parameters:
                gradient = p.grad
                if gradient is None or gradient.is_sparse:
                    raise ValueError("missing/dense gradient required: " + name)
                idx = gradient.detach().abs().flatten().argmax()
                indices.append(idx)
                rows.append(
                    torch.stack(
                        (
                            torch.isfinite(gradient).all().to(p.dtype),
                            (gradient != 0).any().to(p.dtype),
                            gradient.detach().norm().to(p.dtype),
                            p.detach().flatten()[idx],
                        )
                    )
                )
            observed = torch.stack(rows).detach().cpu()
            if not bool((observed[:, 0] == 1).all()) or not bool(torch.isfinite(observed).all()):
                raise ValueError("nonfinite update probe gradient/parameter: " + name)
            if self.require_all and not bool((observed[:, 1] == 1).all()):
                raise ValueError("nonzero gradient required for every tensor: " + name)
            self._pending[name] = (indices, observed)

    def _after(self, optimizer, args, kwargs):
        if not self.enabled:
            return
        result = {}
        for name, parameters in self.families.items():
            indices, before = self._pending[name]
            after = torch.stack(
                [p.detach().flatten()[idx] for p, idx in zip(parameters, indices)]
            ).cpu()
            if not bool(torch.isfinite(after).all()):
                raise ValueError("nonfinite updated parameter: " + name)
            delta = (after - before[:, 3]).abs()
            moved = int((delta > 0).sum())
            if self.require_all and moved != len(parameters):
                raise ValueError("measured parameter movement required for every tensor: " + name)
            result[name] = {
                "parameter_tensors": len(parameters),
                "finite_gradient_tensors": len(parameters),
                "nonzero_gradient_tensors": int(before[:, 1].sum()),
                "clipped_gradient_l2": float(before[:, 2].norm()),
                "sampled_moved_tensors": moved,
                "sampled_parameter_displacement_l1": float(delta.double().sum()),
            }
        self.last_report = {
            "schema": "qat_a8_actual_update_v1",
            "passed": True,
            "all_tensor_nonzero_and_movement_gate": self.require_all,
            "movement_scope": "one_actual_max_gradient_element_per_parameter_tensor",
            "gradients_scope": "after_transform_and_clip_before_optimizer_step",
            "families": result,
        }
        self._pending = None

    def close(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._pending = None


class QATRecipeTelemetry:
    """Packed sign history plus small family snapshots; no dense latent shadow.

    Sampled sign counts can miss changes between observations. A flip-back here
    means a return to the initial sign, as in existing SignFlipDiagnostics; no
    correctness or useful-decision attribution is inferred from movement.
    """

    def __init__(self, linears: Mapping, *, contract: Mapping, near_zero: float = 0.01):
        if (
            type(near_zero) not in (float, int)
            or not math.isfinite(near_zero)
            or not 0 < near_zero <= 1
        ):
            raise ValueError("near-zero threshold must be in (0,1]")
        self.near_zero = float(near_zero)
        self.signs = SignFlipDiagnostics(linears, contract=dict(contract))
        self.initial = self._small_families(linears)

    @staticmethod
    def _small_families(linears):
        return {
            name: [p.detach().cpu().clone() for p in parameters]
            for name, parameters in joint_parameter_families(linears).items()
            if name != "sign"
        }

    def observe(self, linears: Mapping, *, step: int) -> dict:
        current = self._small_families(linears)
        if set(current) != set(self.initial) or any(
            len(current[k]) != len(self.initial[k])
            or any(a.shape != b.shape for a, b in zip(current[k], self.initial[k]))
            for k in current
        ):
            raise ValueError("telemetry family layout changed")
        signs = self.signs.observe(linears, step=step)
        total = near = 0
        abs_sum, minimum = 0.0, math.inf
        for module in linears.values():
            magnitude = module.latent_sign.detach().abs()
            total += magnitude.numel()
            near += int((magnitude <= self.near_zero).sum())
            abs_sum += float(magnitude.sum(dtype=torch.float64))
            minimum = min(minimum, float(magnitude.min()))
        return {
            **signs,
            "sign_mechanism": "latent_gradient_zero_crossing",
            "near_zero_threshold": self.near_zero,
            "near_zero_fraction": near / total,
            "minimum_latent_distance_to_zero": minimum,
            "mean_latent_magnitude": abs_sum / total,
            "family_movement_l1_since_initialization": {
                name: sum(
                    float((a - b).abs().double().sum())
                    for a, b in zip(parameters, self.initial[name])
                )
                for name, parameters in current.items()
            },
            "sampled_counts_not_native_acceptance": True,
        }

    def state_dict(self) -> dict:
        return copy.deepcopy(
            {
                "schema": "qat_a8_telemetry_v1",
                "near_zero": self.near_zero,
                "initial": self.initial,
                "signs": self.signs.state_dict(),
            }
        )

    def load_state_dict(self, state: Mapping, linears: Mapping, *, resumed_step: int) -> None:
        if state.get("schema") != "qat_a8_telemetry_v1" or state.get("near_zero") != self.near_zero:
            raise ValueError("telemetry resume contract differs")
        saved, current = state.get("initial"), self._small_families(linears)
        if not isinstance(saved, dict) or set(saved) != set(current):
            raise ValueError("telemetry saved family inventory differs")
        for name, parameters in current.items():
            if not isinstance(saved[name], list) or len(saved[name]) != len(parameters):
                raise ValueError("telemetry saved family count differs")
            for initial, parameter in zip(saved[name], parameters):
                if (
                    not isinstance(initial, torch.Tensor)
                    or initial.shape != parameter.shape
                    or initial.dtype != parameter.dtype
                    or not bool(torch.isfinite(initial).all())
                ):
                    raise ValueError("telemetry saved parameter invalid")
        self.signs.load_state_dict(state["signs"], linears, resumed_step=resumed_step)
        self.initial = copy.deepcopy(saved)


def load_comparison_manifest(path: Path) -> dict:
    value = json.loads(Path(path).read_text())
    for arm in ("reference", "candidate"):
        comparison_joint_recipe(value, arm)
    return value

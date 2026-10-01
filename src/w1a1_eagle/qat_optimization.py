"""Explicit binary optimization recipes; no model loading or training launch.

Defaults retain the continuous trainer's AdamW recipe. Optional gradient rules
are experiment controls, not measured improvements or native readiness claims.
The caller applies transforms after backward, before step, and projection after
step. Parameter families and ordering are checked at every resume boundary.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import Tensor

from .trajectory_refresh import digest


@dataclass(frozen=True)
class BinaryOptimizationConfig:
    latent_magnitude: float = 0.5
    optimizer: str = "adamw"
    sign_lr: float = 1e-3
    scale_lr: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.0
    sign_gradient_rule: str = "baseline"
    scale_gradient_rule: str = "baseline"
    scale_floor: float = 1e-5
    clip_policy: str = "joint"
    max_grad_norm: float = 1.0
    sign_max_grad_norm: float = 1.0
    scale_max_grad_norm: float = 1.0

    def __post_init__(self):
        for name in ("latent_magnitude", "sign_lr", "scale_lr", "eps", "scale_floor",
                     "max_grad_norm", "sign_max_grad_norm", "scale_max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.latent_magnitude > 1:
            raise ValueError("latent initialization must stay in inclusive STE interval")
        if self.optimizer not in ("adamw", "sgd"):
            raise ValueError("optimizer must be adamw or sgd")
        if len(self.betas) != 2 or any(isinstance(x, bool) or not math.isfinite(x) or
                                       not 0 <= x < 1 for x in self.betas):
            raise ValueError("Adam betas must be finite in [0,1)")
        if isinstance(self.momentum, bool) or not math.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("SGD momentum must be finite in [0,1)")
        if self.sign_gradient_rule not in ("baseline", "weight_unit"):
            raise ValueError("unknown sign gradient rule")
        if self.scale_gradient_rule not in ("baseline", "fan_in_rsqrt"):
            raise ValueError("unknown scale gradient rule")
        if self.clip_policy not in ("joint", "per_family"):
            raise ValueError("unknown clipping policy")

    def manifest(self) -> dict:
        result = asdict(self)
        result["betas"] = list(self.betas)
        return result


def binary_parameter_families(linears: Mapping) -> tuple[list, list]:
    if not linears or any(not isinstance(key, str) or not key for key in linears):
        raise ValueError("linears need nonempty stable names")
    signs, scales = [], []
    for name in sorted(linears):
        module = linears[name]
        sign, scale = module.latent_sign, module.scale_offset
        if not isinstance(sign, torch.nn.Parameter) or not isinstance(scale, torch.nn.Parameter):
            raise ValueError("linears must expose trainable latent_sign and scale_offset")
        if sign.ndim != 2 or not sign.requires_grad or not scale.requires_grad:
            raise ValueError("binary parameters must be trainable with matrix signs")
        if scale.shape != (sign.shape[0],) or module.contract.scale_layout != "row":
            raise ValueError("optimization recipes currently support row scales only")
        if sign.device != scale.device or not sign.is_floating_point() or not scale.is_floating_point():
            raise ValueError("binary parameters must share a device and be floating point")
        signs.append(sign)
        scales.append(scale)
    if len({id(p) for p in signs + scales}) != len(signs + scales):
        raise ValueError("aliased binary parameters")
    return signs, scales


def binary_layout(linears: Mapping) -> list[dict]:
    binary_parameter_families(linears)
    return [{"name": name, "sign_shape": list(linears[name].latent_sign.shape),
             "scale_shape": list(linears[name].scale_offset.shape),
             "activation_bits": linears[name].contract.activation_bits,
             "scale_layout": linears[name].contract.scale_layout}
            for name in sorted(linears)]


def _extra_parameters(parameters) -> list:
    result = list(parameters)
    if any(not isinstance(p, torch.nn.Parameter) or not p.requires_grad or not p.is_floating_point() for p in result):
        raise ValueError("additional optimizer parameters must be trainable floating parameters")
    if len({id(p) for p in result}) != len(result):
        raise ValueError("aliased additional optimizer parameters")
    return result


def validate_optimizer_ownership(linears: Mapping, optimizer: torch.optim.Optimizer,
                                 *, additional_parameters=()) -> None:
    signs, scales = binary_parameter_families(linears)
    extras = _extra_parameters(additional_parameters)
    expected = signs + scales + extras
    if len({id(p) for p in expected}) != len(expected):
        raise ValueError("additional optimizer parameters alias binary parameters")
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(expected) or {id(p) for p in actual} != {id(p) for p in expected}:
        raise ValueError("optimizer must own exactly the supplied binary sign/scale parameters")
    families = (signs, scales, extras) if extras else (signs, scales)
    if len(optimizer.param_groups) != len(families) or any(
        [id(p) for p in group["params"]] != [id(p) for p in family]
        for group, family in zip(optimizer.param_groups, families)
    ):
        raise ValueError("optimizer family ordering differs from stable named layout")


@torch.no_grad()
def initialize_latents_(linears: Mapping, config: BinaryOptimizationConfig) -> None:
    """Fresh initialization only: preserve hard signs, zeros positive, and scales.

    Do this before creating an optimizer. Never call it on a resumed checkpoint;
    resumed latent magnitudes encode learned sign inertia.
    """
    signs, _ = binary_parameter_families(linears)
    if any(getattr(module, "_round_hard_signs", None) is not None for module in linears.values()):
        raise ValueError("initialization cannot mutate an active hard-sign graph cache")
    if any(not bool(torch.isfinite(p).all()) for p in signs):
        raise ValueError("latent initialization requires finite parameters")
    for sign in signs:
        sign.copy_(torch.where(sign < 0, -config.latent_magnitude, config.latent_magnitude))


def make_binary_optimizer(linears: Mapping, config: BinaryOptimizationConfig,
                          *, extra_parameters=(), extra_lr: float | None = None) -> torch.optim.Optimizer:
    signs, scales = binary_parameter_families(linears)
    extras = _extra_parameters(extra_parameters)
    if len({id(p) for p in signs + scales + extras}) != len(signs + scales + extras):
        raise ValueError("additional optimizer parameters alias binary parameters")
    if extras:
        if extra_lr is None or isinstance(extra_lr, bool) or not math.isfinite(extra_lr) or extra_lr <= 0:
            raise ValueError("additional optimizer parameters require explicit finite positive LR")
    elif extra_lr is not None:
        raise ValueError("extra LR requires additional optimizer parameters")
    groups = [{"params": signs, "lr": config.sign_lr, "family": "sign"},
              {"params": scales, "lr": config.scale_lr, "family": "scale"}]
    if extras:
        groups.append({"params": extras, "lr": extra_lr, "family": "additional"})
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(groups, betas=config.betas, eps=config.eps,
                                      weight_decay=0, foreach=False)
    else:
        optimizer = torch.optim.SGD(groups, momentum=config.momentum, weight_decay=0,
                                    foreach=False)
    optimizer._binary_recipe = config.manifest()
    optimizer._binary_extra_layout = [{"shape": list(p.shape), "dtype": str(p.dtype)} for p in extras]
    optimizer._binary_extra_lr = extra_lr
    return optimizer


def transform_binary_gradients_(linears: Mapping, config: BinaryOptimizationConfig,
                                *, additional_parameters=()) -> dict:
    """Validate, transform, then clip gradients; retain inclusive clipped STE.

    ``weight_unit`` removes the positive row scale from the sign gradient,
    bounded by scale_floor. At exact zero scale its sign gradient stays zero;
    scale gradients may still revive the row. This adapts an optimizer rule,
    and does not replace the hard forward or change activation gradients.
    """
    signs, scales = binary_parameter_families(linears)
    extras = _extra_parameters(additional_parameters)
    if len({id(p) for p in signs + scales + extras}) != len(signs + scales + extras):
        raise ValueError("additional gradient parameters alias binary parameters")
    modules = [linears[name] for name in sorted(linears)]
    effective = [module.effective_scales().detach() for module in modules]
    if any(not bool(torch.isfinite(p).all()) for p in signs + scales + effective + extras):
        raise ValueError("nonfinite binary parameters or effective scales")
    if any(p.grad is not None and (p.grad.is_sparse or not bool(torch.isfinite(p.grad).all()))
           for p in signs + scales + extras):
        raise ValueError("binary gradients must be finite dense tensors")
    sign_before = _grad_norm(signs)
    scale_before = _grad_norm(scales)
    with torch.no_grad():
        for sign, scale, alpha in zip(signs, scales, effective):
            if sign.grad is not None and config.sign_gradient_rule == "weight_unit":
                multiplier = torch.where(alpha > 0, alpha.clamp_min(config.scale_floor).reciprocal(), 0)
                sign.grad.mul_(multiplier[:, None])
            if scale.grad is not None and config.scale_gradient_rule == "fan_in_rsqrt":
                scale.grad.mul_(1 / math.sqrt(sign.shape[1]))
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in signs + scales + extras):
        raise ValueError("gradient transform overflow")
    sign_transformed, scale_transformed = _grad_norm(signs), _grad_norm(scales)
    extra_norm = _grad_norm(extras)
    if config.clip_policy == "joint":
        norm = float(torch.nn.utils.clip_grad_norm_(signs + scales + extras, config.max_grad_norm,
                                                    error_if_nonfinite=True))
        clipped = norm > config.max_grad_norm
    else:
        torch.nn.utils.clip_grad_norm_(signs, config.sign_max_grad_norm, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(scales, config.scale_max_grad_norm, error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(extras, config.max_grad_norm, error_if_nonfinite=True)
        norm = math.hypot(sign_transformed, scale_transformed, extra_norm)
        clipped = (sign_transformed > config.sign_max_grad_norm or scale_transformed > config.scale_max_grad_norm
                   or extra_norm > config.max_grad_norm)
    return {"gradient_norm": norm, "sign_gradient_norm": sign_before,
            "scale_gradient_norm": scale_before, "transformed_sign_gradient_norm": sign_transformed,
            "transformed_scale_gradient_norm": scale_transformed, "additional_gradient_norm": extra_norm,
            "clipped_gradient": clipped}


def _grad_norm(params: list) -> float:
    norms = [p.grad.detach().norm() for p in params if p.grad is not None]
    return float(torch.stack(norms).double().norm()) if norms else 0.0


@torch.no_grad()
def project_binary_parameters_(linears: Mapping) -> None:
    signs, scales = binary_parameter_families(linears)
    if any(not bool(torch.isfinite(p).all()) for p in signs + scales):
        raise ValueError("optimizer produced nonfinite binary parameters")
    for module in linears.values():
        if not bool(torch.isfinite(module.initial_scale + module.scale_offset).all()):
            raise ValueError("optimizer produced nonfinite raw weight scales")
    for module in linears.values():
        module.latent_sign.clamp_(-1, 1)
        module.project_scales_()


def optimizer_checkpoint(linears: Mapping, optimizer: torch.optim.Optimizer,
                         config: BinaryOptimizationConfig, *, contract: dict, additional_parameters=()) -> dict:
    extras = _extra_parameters(additional_parameters)
    validate_optimizer_ownership(linears, optimizer, additional_parameters=extras)
    if getattr(optimizer, "_binary_recipe", None) != config.manifest():
        raise ValueError("optimizer recipe differs from checkpoint config")
    return {"schema": "binary_optimizer_v1", "recipe": config.manifest(),
            "layout": binary_layout(linears), "contract_sha256": digest(contract),
            "extra_layout": [{"shape": list(p.shape), "dtype": str(p.dtype)} for p in extras],
            "extra_lr": optimizer._binary_extra_lr,
            "optimizer": copy.deepcopy(optimizer.state_dict())}


def load_optimizer_checkpoint(linears: Mapping, optimizer: torch.optim.Optimizer,
                              config: BinaryOptimizationConfig, saved: dict, *, contract: dict,
                              additional_parameters=()) -> None:
    extras = _extra_parameters(additional_parameters)
    validate_optimizer_ownership(linears, optimizer, additional_parameters=extras)
    if (saved.get("schema") != "binary_optimizer_v1" or saved.get("recipe") != config.manifest()
            or saved.get("layout") != binary_layout(linears)
            or saved.get("contract_sha256") != digest(contract)
            or saved.get("extra_layout") != optimizer._binary_extra_layout
            or saved.get("extra_lr") != optimizer._binary_extra_lr
            or getattr(optimizer, "_binary_recipe", None) != config.manifest()):
        raise ValueError("resume changes binary optimizer recipe/layout/model/data contract")
    state = saved.get("optimizer")
    if not isinstance(state, dict) or not isinstance(state.get("state"), dict):
        raise ValueError("invalid optimizer state")
    for values in state["state"].values():
        for value in values.values():
            if isinstance(value, Tensor) and not bool(torch.isfinite(value).all()):
                raise ValueError("nonfinite optimizer checkpoint")
    # Reject corrupted group rates/options even when the outer recipe is intact.
    expected = optimizer.state_dict()["param_groups"]
    groups = state.get("param_groups", [])
    if len(groups) != len(expected):
        raise ValueError("optimizer checkpoint family inventory differs")
    for group, reference in zip(groups, expected):
        if set(group) != set(reference) or any(group[k] != reference[k] for k in reference
                                             if k not in ("lr",)):
            raise ValueError("optimizer checkpoint group options differ")
        if not math.isfinite(group["lr"]) or not 0 < group["lr"] <= reference["lr"]:
            raise ValueError("optimizer checkpoint LR outside configured warmup range")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    ids = [key for group in groups for key in group["params"]]
    if not set(state["state"]).issubset(set(ids)):
        raise ValueError("optimizer checkpoint contains unowned state")
    for key, parameter in zip(ids, parameters):
        values = state["state"].get(key, {})
        required = {"step", "exp_avg", "exp_avg_sq"} if config.optimizer == "adamw" else {"momentum_buffer"}
        if values and set(values) != required:
            raise ValueError("optimizer checkpoint moment inventory differs")
        for field, value in values.items():
            if not isinstance(value, Tensor) or (field != "step" and value.shape != parameter.shape):
                raise ValueError("optimizer checkpoint moment shape differs")
            if field == "step" and (value.numel() != 1 or float(value) < 0 or float(value) != int(value)):
                raise ValueError("invalid optimizer checkpoint update counter")
            if field == "exp_avg_sq" and bool((value < 0).any()):
                raise ValueError("negative optimizer second moment")
    optimizer.load_state_dict(copy.deepcopy(state))


class SignFlipDiagnostics:
    """Opt-in sampled sign history, packed at one bit per mask on CPU.

    Counts concern observations, not unobserved steps. Sampling every update
    gives exact flip/flip-back counts; intervals can miss chatter. Four packed
    masks cost ~0.5 bytes per sign, with a per-module unpack/copy temporary.
    ``sustained_disagreement`` means different from initialization at two
    adjacent observations without an intervening observed sign change.
    """

    def __init__(self, linears: Mapping, *, contract: dict):
        self.layout = binary_layout(linears)
        self.contract_sha256 = digest(contract)
        self.last_step = 0
        self.observations = 0
        self.cumulative_flips = 0
        self.cumulative_flip_backs = 0
        self.masks = {}
        for name in sorted(linears):
            packed = self._signs(linears[name])
            self.masks[name] = {"initial": packed.copy(), "previous": packed.copy(),
                                "ever_flipped": np.zeros_like(packed),
                                "disagreement": np.zeros_like(packed)}

    @staticmethod
    def _signs(module):
        latent = module.latent_sign.detach()
        if not bool(torch.isfinite(latent).all()):
            raise ValueError("sign diagnostics require finite latents")
        return np.packbits((latent < 0).cpu().numpy().reshape(-1), bitorder="little")

    def observe(self, linears: Mapping, *, step: int) -> dict:
        if type(step) is not int or step <= self.last_step or binary_layout(linears) != self.layout:
            raise ValueError("diagnostics require increasing steps and unchanged layout")
        current = {name: self._signs(linears[name]) for name in sorted(linears)}
        flips = flip_backs = net = unique = sustained = total = 0
        for row in self.layout:
            name, masks = row["name"], self.masks[row["name"]]
            count = math.prod(row["sign_shape"])
            new = current[name]
            changed = new ^ masks["previous"]
            disagreement = new ^ masks["initial"]
            backs = changed & masks["disagreement"] & ~disagreement
            continuing = disagreement & masks["disagreement"] & ~changed
            masks["ever_flipped"] |= changed
            for key, value in (("flips", changed), ("backs", backs), ("net", disagreement),
                               ("unique", masks["ever_flipped"]), ("sustained", continuing)):
                number = int(np.unpackbits(value, bitorder="little", count=count).sum())
                if key == "flips": flips += number
                elif key == "backs": flip_backs += number
                elif key == "net": net += number
                elif key == "unique": unique += number
                else: sustained += number
            masks["previous"] = new
            masks["disagreement"] = disagreement
            total += count
        self.cumulative_flips += flips
        self.cumulative_flip_backs += flip_backs
        gap = step - self.last_step
        self.last_step, self.observations = step, self.observations + 1
        return {"sign_flips": flips, "flip_backs": flip_backs, "net_sign_disagreement": net,
                "unique_flipped_signs": unique, "sustained_disagreement": sustained,
                "cumulative_sign_flips": self.cumulative_flips,
                "cumulative_flip_backs": self.cumulative_flip_backs,
                "sign_count": total, "observation_gap_steps": gap,
                "diagnostic_observations": self.observations}

    def state_dict(self) -> dict:
        return copy.deepcopy({"schema": "binary_flip_diagnostics_v1", "layout": self.layout,
                              "contract_sha256": self.contract_sha256, "last_step": self.last_step,
                              "observations": self.observations, "cumulative_flips": self.cumulative_flips,
                              "cumulative_flip_backs": self.cumulative_flip_backs, "masks": self.masks})

    def load_state_dict(self, saved: dict, linears: Mapping, *, resumed_step: int | None = None) -> None:
        if (saved.get("schema") != "binary_flip_diagnostics_v1" or saved.get("layout") != self.layout
                or saved.get("contract_sha256") != self.contract_sha256
                or binary_layout(linears) != self.layout):
            raise ValueError("diagnostic resume changes model/data/layout contract")
        for key in ("last_step", "observations", "cumulative_flips", "cumulative_flip_backs"):
            if type(saved.get(key)) is not int or saved[key] < 0:
                raise ValueError("invalid diagnostic counter")
        if saved["observations"] > saved["last_step"] or saved["cumulative_flip_backs"] > saved["cumulative_flips"]:
            raise ValueError("inconsistent diagnostic counters")
        if resumed_step is None:
            resumed_step = saved["last_step"]
        if type(resumed_step) is not int or resumed_step < saved["last_step"]:
            raise ValueError("diagnostic resume predates the saved observation")
        masks = saved.get("masks", {})
        if set(masks) != set(self.masks):
            raise ValueError("diagnostic mask inventory differs")
        for name, expected in self.masks.items():
            if set(masks[name]) != set(expected):
                raise ValueError("diagnostic mask families differ")
            for key, value in masks[name].items():
                if not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.shape != expected[key].shape:
                    raise ValueError("invalid packed diagnostic mask")
            if resumed_step == saved["last_step"] and not np.array_equal(masks[name]["previous"], self._signs(linears[name])):
                raise ValueError("diagnostic checkpoint differs from restored latent signs")
            if not np.array_equal(masks[name]["disagreement"], masks[name]["initial"] ^ masks[name]["previous"]):
                raise ValueError("diagnostic disagreement mask inconsistent")
            if np.any(masks[name]["disagreement"] & ~masks[name]["ever_flipped"]):
                raise ValueError("diagnostic history cannot explain current disagreement")
            count = next(math.prod(row["sign_shape"]) for row in self.layout if row["name"] == name)
            if count % 8 and any(int(value[-1]) >> (count % 8) for value in masks[name].values()):
                raise ValueError("nonzero diagnostic mask padding")
        self.masks = copy.deepcopy(masks)
        self.last_step, self.observations = saved["last_step"], saved["observations"]
        self.cumulative_flips = saved["cumulative_flips"]
        self.cumulative_flip_backs = saved["cumulative_flip_backs"]

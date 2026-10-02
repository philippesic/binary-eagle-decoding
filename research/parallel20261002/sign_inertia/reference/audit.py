"""Bounded CPU sign-inertia audit using production optimizer and hard A1 layers.

Run with PYTHONPATH=src <python> research/.../reference/audit.py --output <ignored path>.
The summary is small; per-update tensors stay outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from w1a1_eagle.qat_optimization import (
    BinaryOptimizationConfig,
    SignFlipDiagnostics,
    initialize_latents_,
    make_binary_optimizer,
    project_binary_parameters_,
    transform_binary_gradients_,
)
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract

CONTROL = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")
BUDGET = 64
LINEAR_X = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
XOR_X = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
SEQUENCES = [(1, 1, -1), (1, -1, -1), (-1, 1, 1), (-1, -1, 1)]


def control_check():
    control = json.loads(CONTROL.read_text())
    if control["research_stop"] or control.get("reset_observed"):
        raise RuntimeError("Research stop/reset: checkpoint; do not launch")
    return control


def exact_recurrent(signs, scale):
    """Independent Python scalar hard A1 recurrence; no STE or Torch calls."""
    result = []
    for sequence in SEQUENCES:
        hidden = 0.25
        row = []
        for value in sequence:
            alpha = (abs(value) + abs(hidden)) / 2  # production A1 mean-absolute scale
            z = scale * alpha * (signs[0] * value + signs[1] * (1 if hidden >= 0 else -1)) + 0.15
            row.append(z)
            hidden = math.tanh(z)
        result.append(row)
    return result


def fixture(name):
    if name == "linear":
        x = torch.tensor(LINEAR_X, dtype=torch.float32)
        target = torch.tensor([1, -1, 1], dtype=torch.float32)
        return x, torch.sign(x @ target)
    if name == "xor":
        return torch.tensor(XOR_X, dtype=torch.float32), torch.tensor([1, -1, -1, 1])
    return torch.tensor(SEQUENCES, dtype=torch.float32), torch.tensor(
        [[1 if z > 0 else -1 for z in row] for row in exact_recurrent((1, -1), 1)]
    )


def logits(module, name, x):
    if name != "recurrent":
        return module(x).squeeze(-1)
    hidden = x.new_full((len(x),), 0.25)
    result = []
    for depth in range(x.shape[1]):
        z = module(torch.stack((x[:, depth], hidden), dim=-1)).squeeze(-1)
        result.append(z)
        hidden = torch.tanh(z)  # attached current-student recurrent state
    return torch.stack(result, dim=1)


def enumerate_capacity(name, scale):
    x, y = fixture(name)
    rows = []
    for signs in itertools.product((-1, 1), repeat=x.shape[-1] if name != "recurrent" else 2):
        values = (
            exact_recurrent(signs, scale)
            if name == "recurrent"
            else [scale * sum(a * b for a, b in zip(signs, row)) for row in x.tolist()]
        )
        margins = torch.tensor(values) * y
        rows.append(
            {
                "signs": list(signs),
                "min_margin": float(margins.min()),
                "correct": int((margins > 0).sum()),
                "logits": values,
            }
        )
    best = max(rows, key=lambda r: (r["correct"], r["min_margin"]))
    return {
        "states": rows,
        "feasible_states": sum(r["min_margin"] > 0 for r in rows),
        "best": best,
        "total_decisions": y.numel(),
    }


def metrics(z, y):
    margins = z * y
    correct = margins > 0
    prefixes = correct.int().cumprod(-1).sum(-1) if z.ndim == 2 else correct.int().cumprod(0).sum()
    return {
        "correct": int(correct.sum()),
        "min_margin": float(margins.min()),
        "mean_margin": float(margins.mean()),
        "hard_prefix": float(prefixes.float().mean()),
        "ce": float(F.softplus(-2 * margins).mean()),
        "logits": z.tolist(),
        "decision_margins": margins.tolist(),
    }


def run_case(name, scale, magnitude, scale_mode, option):
    x, y = fixture(name)
    width = 2 if name == "recurrent" else x.shape[-1]
    config = replace(BinaryOptimizationConfig(), latent_magnitude=magnitude, **option)
    module = RowBinaryLinear(
        torch.ones(1, width),
        torch.tensor([scale]),
        W1AxContract(1),
        bias=torch.tensor([0.15]) if name == "recurrent" else None,
    )
    modules = {"fixture": module}
    initialize_latents_(modules, config)
    optimizer = make_binary_optimizer(modules, config)
    diagnostics = SignFlipDiagnostics(modules, contract={"fixture": name})
    initial = metrics(logits(module, name, x).detach(), y)
    records = []
    first_useful = None
    first_full = 0 if initial["correct"] == y.numel() else None
    wrong_flip_events = 0
    for step in range(1, BUDGET + 1):
        optimizer.zero_grad(set_to_none=True)
        z = logits(module, name, x)
        loss = F.softplus(-2 * z * y).mean()
        loss.backward()
        if scale_mode == "frozen":
            module.scale_offset.grad = None  # ownership unchanged; scale receives no update
        raw_sign_gradient = module.latent_sign.grad.tolist()
        raw_scale_gradient = (
            None if module.scale_offset.grad is None else module.scale_offset.grad.tolist()
        )
        gradients = transform_binary_gradients_(modules, config)
        post_gradient = module.latent_sign.grad.clone()
        scale_gradient = module.scale_offset.grad
        latent_before = module.latent_sign.detach().clone()
        optimizer.step()
        project_binary_parameters_(modules)
        flips = diagnostics.observe(modules, step=step)
        current = metrics(logits(module, name, x).detach(), y)
        old_signs = [1 if v >= 0 else -1 for v in latent_before.flatten().tolist()]
        current_scale = float(module.effective_scales().detach()[0])
        old_sign_values = (
            exact_recurrent(old_signs, current_scale)
            if name == "recurrent"
            else [current_scale * sum(a * b for a, b in zip(old_signs, row)) for row in x.tolist()]
        )
        counterfactual = metrics(torch.tensor(old_sign_values), y)
        # Hold post-update scales fixed: attribute improvement to the sign event,
        # not coincident scale movement. More flips/CE alone never qualify.
        useful = flips["sign_flips"] and (
            current["correct"] > counterfactual["correct"]
            or (
                current["correct"] == counterfactual["correct"]
                and current["min_margin"] > counterfactual["min_margin"] + 1e-5
            )
        )
        if useful and first_useful is None:
            first_useful = step
        if current["correct"] == y.numel() and first_full is None:
            first_full = step
        if flips["sign_flips"] and (
            current["correct"] < counterfactual["correct"]
            or (
                current["correct"] == counterfactual["correct"]
                and current["min_margin"] < counterfactual["min_margin"] - 1e-5
            )
        ):
            wrong_flip_events += 1
        state = optimizer.state[module.latent_sign]
        records.append(
            {
                "step": step,
                **current,
                **flips,
                **gradients,
                "clipped_sign_gradient": post_gradient.tolist(),
                "clipped_scale_gradient_norm": 0
                if scale_gradient is None
                else float(scale_gradient.norm()),
                "raw_sign_gradient": raw_sign_gradient,
                "raw_scale_gradient": raw_scale_gradient,
                "latent": module.latent_sign.tolist(),
                "latent_step": (module.latent_sign.detach() - latent_before).tolist(),
                "min_latent_distance": float(module.latent_sign.detach().abs().min()),
                "effective_scale": module.effective_scales().tolist(),
                "optimizer_sign_state": {key: value.tolist() for key, value in state.items()},
                "old_sign_post_scale_metrics": counterfactual,
                "useful_flip": bool(useful),
            }
        )
    summary = {
        "fixture": name,
        "initial_scale": scale,
        "latent_magnitude": magnitude,
        "scale_mode": scale_mode,
        "recipe": config.manifest(),
        "first_useful_flip": first_useful,
        "first_all_correct": first_full,
        "wrong_flip_events": wrong_flip_events,
        "full_correct_updates": sum(r["correct"] == y.numel() for r in records),
        "tail16_full_correct_updates": sum(r["correct"] == y.numel() for r in records[-16:]),
        "initial": initial,
        "final": records[-1],
        "budget": BUDGET,
    }
    return {"summary": summary, "trajectory": records}


def run():
    control_check()
    torch.set_num_threads(1)
    options = {
        "baseline": {},
        "weight_unit": {"sign_gradient_rule": "weight_unit"},
        "per_family": {"clip_policy": "per_family"},
        "weight_unit_per_family": {
            "sign_gradient_rule": "weight_unit",
            "clip_policy": "per_family",
        },
    }
    results = []
    # A predeclared controlled nuisance grid, not tuning: 2 tasks x 3 scales x
    # 2 inertias x 2 scale modes x 4 existing controls = 96 CPU trajectories.
    for name in ("linear", "recurrent"):
        for scale in (0.01, 1, 100):
            control_check()
            for magnitude, mode, option_name in itertools.product(
                (0.5, 0.02), ("frozen", "joint"), options
            ):
                case = run_case(name, scale, magnitude, mode, options[option_name])
                case["summary"]["option"] = option_name
                results.append(case)
    # One infeasible control plus SGD/momentum controls, avoiding a second grid.
    for name, option_name, option in (
        ("xor", "baseline", {}),
        ("linear", "sgd", {"optimizer": "sgd"}),
        ("linear", "sgd_momentum", {"optimizer": "sgd", "momentum": 0.9}),
    ):
        control_check()
        case = run_case(name, 1, 0.02, "frozen", option)
        case["summary"]["option"] = option_name
        results.append(case)
    sources = [
        "src/w1a1_eagle/" + name
        for name in (
            "qat_optimization.py",
            "continuous_qat.py",
            "recurrent_qat.py",
            "recurrent_binary.py",
            "recurrent_loss.py",
            "trajectory_refresh.py",
            "adapter.py",
        )
    ]
    sources = [name for name in sources if Path(name).exists()]
    return {
        "schema": "sign_inertia_cpu_v1",
        "hardware": platform.platform(),
        "device": "CPU",
        "precision": "torch.float32",
        "torch": torch.__version__,
        "budget": BUDGET,
        "source_sha256": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        "capacity": {
            name: {str(scale): enumerate_capacity(name, scale) for scale in (0.01, 1, 100)}
            for name in ("linear", "recurrent", "xor")
        },
        "runs": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runs": len(result["runs"]),
                "output": str(args.output),
                "feasible": {
                    name: result["capacity"][name]["1"]["feasible_states"]
                    for name in result["capacity"]
                },
            }
        )
    )

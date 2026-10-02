"""Tiny actual-step equivalence and dispatch allocation census; CPU only."""

from __future__ import annotations

import copy
import difflib
import hashlib
import inspect
import json
import platform
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from research.parallel20261002.step_bookkeeping.reference.snapshot_step import (
    FUNCTION_SHA256,
    joint_train_step_without_snapshot_clone,
    source_pair,
)
from scripts.train_joint_w1ax import tiny_joint_fixture, tiny_rollout
from w1a1_eagle.affine_binary import AffineBinaryConfig, install_affine_binary
from w1a1_eagle.learned_activation import LearnedActivationBank
from w1a1_eagle.qat_optimization import BinaryOptimizationConfig
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    joint_optimizer,
    joint_parameter_families,
    joint_train_step,
)

ROOT = Path(__file__).resolve().parents[4]
CONTROL = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")


def check_control():
    control = json.loads(CONTROL.read_text())
    if (
        control["research_stop"]
        or control["reset_observed"]
        or control["last_reset_unix"] != 1791049896
        or 100 - control["last_weekly_used_percent"] <= 1
    ):
        raise RuntimeError("research stop/reset/allowance gate")


def fixture(optional=False):
    affine = AffineBinaryConfig(enabled=True, coverage="all", mild_l2=0.03) if optional else None
    config = JointQATConfig(
        W1AxContract(4),
        sign_lr=0.1,
        scale_lr=0.01,
        seed=1062,
        activation_quantization="learned" if optional else "fixed",
        activation_lr=0.001,
        affine_weights=affine,
        binary_optimization=BinaryOptimizationConfig(sign_lr=0.1, scale_lr=0.01)
        if optional else None,
    )
    linears, trace = tiny_joint_fixture(config)
    with torch.no_grad():
        for module in linears.values():
            module.latent_sign.copy_(torch.where(module.latent_sign < 0, -0.001, 0.001))
    if optional:
        bank = LearnedActivationBank(4, {p: m.in_features for p, m in linears.items()})
        bank.attach(linears)
        install_affine_binary(linears, target=nn.Module(), config=affine)
        with torch.no_grad():
            for module in linears.values():
                module.affine_binary.midpoint.copy_(torch.linspace(-0.03, 0.03, 4))
    return linears, trace, config


def parameters(linears):
    return [p for family in joint_parameter_families(linears).values() for p in family]


class StepCensus(TorchDispatchMode):
    """Trace snapshot lineage until zero_grad; no profiler memory estimates."""

    def __init__(self, linears):
        super().__init__()
        self.phase = True
        self.lineage = {m.latent_sign.data_ptr(): p for p, m in linears.items()}
        self.master_pointers = set(self.lineage)
        self.ops = Counter()
        self.float_clones = []
        self.snapshots = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        name = str(func)
        self.ops[name] += 1
        if (
            self.phase
            and name == "aten.clone.default"
            and args[0].data_ptr() in self.master_pointers
        ):
            path = self.lineage[args[0].data_ptr()]
            self.lineage[output.data_ptr()] = path
            self.float_clones.append((path, output.numel() * output.element_size()))
        if self.phase and name == "aten.lt.Scalar" and args[0].data_ptr() in self.lineage:
            assert args[1] == 0 and output.dtype == torch.bool
            self.snapshots[self.lineage[args[0].data_ptr()]] = output
            # Clone temporaries die after comparison; do not retain their freed
            # addresses as lineage when the CPU allocator reuses those blocks.
            if args[0].data_ptr() not in self.master_pointers:
                self.lineage.pop(args[0].data_ptr())
        return output

    def summary(self):
        return {
            "sign_snapshot_float_clones": len(self.float_clones),
            "sign_snapshot_float_clone_bytes": sum(size for _, size in self.float_clones),
            "boolean_snapshot_count": len(self.snapshots),
            "boolean_snapshot_bytes": sum(t.numel() * t.element_size()
                                          for t in self.snapshots.values()),
            "total_clone_calls": self.ops["aten.clone.default"],
            "local_scalar_dense_calls": self.ops["aten._local_scalar_dense.default"],
            "ops": dict(sorted(self.ops.items())),
        }


def run(step, linears, trace, config, *, nan_gradient=False):
    optimizer = joint_optimizer(linears, config)
    logits, cache, states = tiny_rollout(linears, "cpu")
    prior = {p: m.latent_sign.detach() < 0 for p, m in linears.items()}
    census = StepCensus(linears)
    zero_grad = optimizer.zero_grad

    def phase_end(*args, **kwargs):
        census.phase = False
        return zero_grad(*args, **kwargs)

    hook = None
    if nan_gradient:
        hook = linears["fc"].latent_sign.register_hook(lambda grad: grad * float("nan"))
    with patch.object(optimizer, "zero_grad", phase_end), patch.object(
        optimizer, "step", wraps=optimizer.step
    ) as optimizer_step, census:
        error = None
        try:
            metrics = step(linears, logits, trace, optimizer, config)
        except ValueError as exc:
            metrics, error = None, str(exc)
        calls = optimizer_step.call_count
    if hook is not None:
        hook.remove()
    assert len(census.snapshots) == 9
    storages = [t.untyped_storage().data_ptr() for t in census.snapshots.values()]
    assert len(set(storages)) == 9
    assert not set(storages) & {m.latent_sign.untyped_storage().data_ptr()
                              for m in linears.values()}
    for path, snapshot in census.snapshots.items():
        torch.testing.assert_close(snapshot, prior[path], rtol=0, atol=0)
    return {
        "metrics": metrics,
        "error": error,
        "optimizer_step_calls": calls,
        "parameters": [p.detach().clone() for p in parameters(linears)],
        "gradients": [None if p.grad is None else p.grad.detach().clone()
                      for p in parameters(linears)],
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "cache_gradients": [t.grad for pair in cache for t in pair],
        "state_gradients": [t.grad for t in states],
        "census": census.summary(),
        "snapshots": census.snapshots,
    }


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_exact(left, right)
    else:
        assert a == b, (a, b)


def comparison(optional=False, *, nan_gradient=False):
    check_control()
    linears, trace, config = fixture(optional)
    other = copy.deepcopy(linears)
    initial = [p.detach().clone() for p in parameters(linears)]
    a = run(joint_train_step, linears, trace, config, nan_gradient=nan_gradient)
    b = run(joint_train_step_without_snapshot_clone, other, trace, config,
            nan_gradient=nan_gradient)
    for key in ("metrics", "error", "optimizer_step_calls", "parameters", "gradients",
                "optimizer_state", "cache_gradients", "state_gradients", "snapshots"):
        assert_exact(a[key], b[key])
    assert a["census"]["sign_snapshot_float_clones"] == 9
    assert b["census"]["sign_snapshot_float_clones"] == 0
    assert a["census"]["boolean_snapshot_bytes"] == b["census"]["boolean_snapshot_bytes"] == 144
    assert a["census"]["local_scalar_dense_calls"] == b["census"]["local_scalar_dense_calls"]
    if nan_gradient:
        assert a["error"] == "nonfinite joint QAT gradient"
        assert a["optimizer_step_calls"] == 0
        assert_exact(a["parameters"], initial)
        assert not a["optimizer_state"]["state"]
    else:
        assert a["optimizer_step_calls"] == 1
        assert a["metrics"]["sign_flips"] > 0
        assert a["metrics"]["scale_l1_movement"] > 0
    return {
        "recipe": "learned_A4_affine_all_binary_optimization" if optional else "fixed_A4",
        "nan_gradient": nan_gradient,
        "exact_metrics": a["metrics"],
        "error": a["error"],
        "optimizer_step_calls": a["optimizer_step_calls"],
        "baseline": a["census"],
        "candidate": b["census"],
    }


def configured_bytes():
    shapes = json.loads((ROOT / "configs/continuous_w1ax.json").read_text())["model_shapes"]
    rows = [{"shape": [o, i], "weights": o * i, "f32_copy_payload_bytes": 4 * o * i,
             "nominal_read_plus_write_bytes": 8 * o * i,
             "retained_bool_bytes": o * i} for o, i in shapes]
    return {
        "per_projection": rows,
        "weights_per_lane": sum(x["weights"] for x in rows),
        "f32_copy_payload_bytes_per_lane_step": sum(x["f32_copy_payload_bytes"] for x in rows),
        "nominal_read_plus_write_bytes_per_lane_step": sum(x["nominal_read_plus_write_bytes"]
                                                         for x in rows),
        "largest_f32_clone_bytes": max(x["f32_copy_payload_bytes"] for x in rows),
        "retained_bool_snapshot_bytes_per_lane_step": sum(x["retained_bool_bytes"] for x in rows),
    }


def main():
    check_control()
    output = {
        "schema": "joint_step_snapshot_audit_v1",
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "device": "CPU", "torch": torch.__version__,
                     "threads": torch.get_num_threads()},
        "function_sha256": FUNCTION_SHA256,
        "source_sha256": hashlib.sha256(Path(inspect.getsourcefile(joint_train_step))
                                        .read_bytes()).hexdigest(),
        "comparisons": [comparison(), comparison(True), comparison(nan_gradient=True)],
        "configured_bytes": configured_bytes(),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    old, new = source_pair()
    patch_path = ROOT / "research/parallel20261002/step_bookkeeping/reference/snapshot-only.patch"
    # Patch coordinates bind to the complete source, for ordinary git apply --check.
    source = Path(inspect.getsourcefile(joint_train_step)).read_text()
    changed = source.replace(old, new)
    patch_path.write_text("".join(difflib.unified_diff(
        source.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile="a/src/w1a1_eagle/recurrent_qat.py", tofile="b/src/w1a1_eagle/recurrent_qat.py",
    )))


if __name__ == "__main__":
    main()

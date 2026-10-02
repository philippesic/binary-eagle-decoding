"""Actual source operation census and exact tiny-step proof, CPU only."""

import ast
import copy
import hashlib
import inspect
import json
import platform
import statistics
import subprocess
import timeit
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode

from scripts.train_joint_w1ax import tiny_joint_fixture, tiny_rollout
from w1a1_eagle import recurrent_qat as qat
from w1a1_eagle.affine_binary import AffineBinaryConfig, install_affine_binary
from w1a1_eagle.learned_activation import LearnedActivationBank
from w1a1_eagle.qat_optimization import BinaryOptimizationConfig

BASE = "e8569fb"
CONTROL = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")


def check_control():
    d = json.loads(CONTROL.read_text())
    if (d["research_stop"] or d["reset_observed"] or d["last_reset_unix"] != 1791049896
            or 100 - d["last_weekly_used_percent"] <= 1):
        raise RuntimeError("research stop/reset/allowance gate")


def baseline_step():
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:src/w1a1_eagle/recurrent_qat.py"], text=True
    )
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "joint_train_step")
    namespace = dict(vars(qat))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-original-step>", "exec"),
         namespace)
    return namespace["joint_train_step"]


def fixture(optional=False):
    affine = AffineBinaryConfig(enabled=True, coverage="all", mild_l2=0.03) if optional else None
    config = qat.JointQATConfig(
        qat.W1AxContract(4), sign_lr=0.1, scale_lr=0.01, seed=1062,
        activation_quantization="learned" if optional else "fixed",
        activation_lr=0.001, affine_weights=affine,
        binary_optimization=BinaryOptimizationConfig(sign_lr=0.1, scale_lr=0.01)
        if optional else None,
    )
    linears, audit = tiny_joint_fixture(config)
    with torch.no_grad():
        for module in linears.values():
            module.latent_sign.copy_(torch.where(module.latent_sign < 0, -0.001, 0.001))
    if optional:
        LearnedActivationBank(4, {p: m.in_features for p, m in linears.items()}).attach(linears)
        install_affine_binary(linears, target=nn.Module(), config=affine)
    return linears, audit, config


class Census(TorchDispatchMode):
    def __init__(self, linears):
        super().__init__()
        self.phase = "before_update"
        self.ops = {p: Counter() for p in ("before_update", "after_update")}
        self.sign_pointers = {m.latent_sign.data_ptr() for m in linears.values()}
        self.sign_clones = 0
        self.extractions = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[self.phase][str(func)] += 1
        if (str(func) == "aten.clone.default" and self.phase == "before_update"
                and args[0].data_ptr() in self.sign_pointers):
            self.sign_clones += 1
        return func(*args, **(kwargs or {}))

    def summary(self):
        return {
            "sign_snapshot_float_clones": self.sign_clones,
            "ops_by_phase": {p: dict(sorted(c.items())) for p, c in self.ops.items()},
            "local_scalar_dense_by_phase": {
                p: c["aten._local_scalar_dense.default"] for p, c in self.ops.items()
            },
            "reporting_host_extractions": self.extractions,
            "note": "CPU tensors already reside on host; .cpu() makes no device transfer",
        }


def run(step, linears, audit, config):
    optimizer = qat.joint_optimizer(linears, config)
    logits, cache, states = tiny_rollout(linears, "cpu")
    census = Census(linears)
    original_step = optimizer.step
    original_tolist = torch.Tensor.tolist

    def mutate(*args, **kwargs):
        value = original_step(*args, **kwargs)
        census.phase = "after_update"
        return value

    def host_list(tensor):
        census.extractions.append({"device": str(tensor.device), "dtype": str(tensor.dtype),
                                   "numel": tensor.numel()})
        return original_tolist(tensor)

    with patch.object(optimizer, "step", mutate), patch.object(torch.Tensor, "tolist", host_list):
        with census:
            metrics = step(linears, logits, audit, optimizer, config)
    families = qat.joint_parameter_families(linears)
    parameters = [p for family in families.values() for p in family]
    return {
        "metrics": metrics,
        "parameters": [p.detach().clone() for p in parameters],
        "gradients": [None if p.grad is None else p.grad.detach().clone() for p in parameters],
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "cache_gradients": [t.grad for pair in cache for t in pair],
        "state_gradients": [t.grad for t in states],
        "census": census.summary(),
    }


def exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact(a, b)
    else:
        assert type(left) is type(right) and left == right, (left, right)


def comparison(optional=False):
    check_control()
    linears, audit, config = fixture(optional)
    candidate = copy.deepcopy(linears)
    baseline = run(baseline_step(), linears, audit, config)
    actual = run(qat.joint_train_step, candidate, audit, config)
    for key in baseline:
        if key != "census":
            exact(baseline[key], actual[key])
    old, new = baseline["census"], actual["census"]
    assert old["sign_snapshot_float_clones"] == 9
    assert new["sign_snapshot_float_clones"] == 0
    assert old["local_scalar_dense_by_phase"]["before_update"] == (
        new["local_scalar_dense_by_phase"]["before_update"])
    assert old["local_scalar_dense_by_phase"]["after_update"] - (
        new["local_scalar_dense_by_phase"]["after_update"]) == (7 if optional else 8)
    assert [x["dtype"] for x in new["reporting_host_extractions"]] == [
        "torch.float32", "torch.int64"]
    return {"recipe": "learned_A4_affine_binary_recipe" if optional else "fixed_A4",
            "exact_metrics": actual["metrics"], "baseline": old, "candidate": new}


def reporting_cost():
    """Bounded CPU conversion cost; excludes forward/update and accelerator latency."""
    check_control()
    values = {f"float_{i}": torch.tensor(i + 0.25) for i in range(6)}
    values.update({f"count_{i}": torch.tensor(i, dtype=torch.int64) for i in range(2)})

    def scalar_conversions():
        return {name: int(value) if value.dtype == torch.int64 else float(value)
                for name, value in values.items()}

    def grouped_conversions():
        return qat._reporting_scalars(values)

    exact(scalar_conversions(), grouped_conversions())
    repetitions, trials = 2000, 5
    old, new = [], []
    for _ in range(trials):
        old.append(timeit.timeit(scalar_conversions, number=repetitions) / repetitions)
        new.append(timeit.timeit(grouped_conversions, number=repetitions) / repetitions)
    return {"scope": "already-computed CPU reporting scalars only; no training latency claim",
            "repetitions_per_trial": repetitions, "trials": trials,
            "baseline_seconds_per_call": old, "candidate_seconds_per_call": new,
            "baseline_median_seconds": statistics.median(old),
            "candidate_median_seconds": statistics.median(new)}


def main():
    check_control()
    torch.set_num_threads(1)
    output = {
        "base_commit": BASE,
        "source_sha256": hashlib.sha256(Path(inspect.getsourcefile(qat)).read_bytes()).hexdigest(),
        "hardware": {"platform": platform.platform(), "machine": platform.machine(),
                     "device": "CPU", "torch": torch.__version__, "threads": 1},
        "comparisons": [comparison(False), comparison(True)],
        "cpu_reporting_cost": reporting_cost(),
    }
    Path(__file__).with_name("summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"hardware": output["hardware"], "results": [
        {"recipe": c["recipe"], "before_scalar_calls": {
            k: c[k]["local_scalar_dense_by_phase"]["before_update"]
            for k in ("baseline", "candidate")}, "after_scalar_calls": {
            k: c[k]["local_scalar_dense_by_phase"]["after_update"]
            for k in ("baseline", "candidate")}}
        for c in output["comparisons"]]}, indent=2))


if __name__ == "__main__":
    main()

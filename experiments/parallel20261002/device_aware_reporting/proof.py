"""Actual reporting-helper CPU proof; routing stubs never allocate off CPU."""

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
from torch.utils._python_dispatch import TorchDispatchMode

from experiments.parallel20261002.step_metrics_feature.probe import (
    check_control,
    exact,
    fixture,
    run,
)
from w1a1_eagle import recurrent_qat as qat

BASE = "93a2ff1"


def pinned_grouped_helper():
    source = subprocess.check_output(
        ["git", "show", f"{BASE}:src/w1a1_eagle/recurrent_qat.py"], text=True
    )
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "_reporting_scalars")
    namespace = dict(vars(qat))
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<pinned-grouped-helper>", "exec"),
         namespace)
    return namespace["_reporting_scalars"]


class Census(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.operations = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.operations[str(func)] += 1
        return func(*args, **(kwargs or {}))


def helper_census():
    values = {f"float_{i}": torch.tensor(i + 0.25) for i in range(6)}
    values.update({f"count_{i}": torch.tensor(2**53 + i, dtype=torch.int64) for i in range(2)})
    counts = {}
    for label, helper in (("grouped_baseline", pinned_grouped_helper()),
                          ("device_aware", qat._reporting_scalars)):
        census = Census()
        with census:
            output = helper(values)
        exact({name: value.item() for name, value in values.items()}, output)
        counts[label] = dict(sorted(census.operations.items()))
    assert counts["grouped_baseline"]["aten.stack.default"] == 2
    assert counts["device_aware"].get("aten.stack.default", 0) == 0
    assert counts["device_aware"].get("aten.detach.default", 0) == 0
    assert counts["device_aware"]["aten._local_scalar_dense.default"] == len(values)
    # The direct path dispatches only scalar reads: no tensor allocation ops.
    assert set(counts["device_aware"]) == {"aten._local_scalar_dense.default"}
    return counts


def step_proof(optional=False):
    check_control()
    linears, audit, config = fixture(optional)
    candidate = copy.deepcopy(linears)
    with patch.object(qat, "_reporting_scalars", pinned_grouped_helper()):
        baseline = run(qat.joint_train_step, linears, audit, config)
    actual = run(qat.joint_train_step, candidate, audit, config)
    for key in baseline:
        if key != "census":
            exact(baseline[key], actual[key])
    # Same safety gates and math; only post-update reporting changes.
    assert baseline["census"]["ops_by_phase"]["before_update"] == (
        actual["census"]["ops_by_phase"]["before_update"])
    assert actual["census"]["reporting_host_extractions"] == []
    return {"recipe": "learned_A4_affine_binary" if optional else "fixed_A4",
            "exact_metrics": actual["metrics"],
            "baseline_census": baseline["census"], "candidate_census": actual["census"]}


def benchmark():
    check_control()
    values = {f"float_{i}": torch.tensor(i + 0.25) for i in range(6)}
    values.update({f"count_{i}": torch.tensor(i, dtype=torch.int64) for i in range(2)})
    grouped = pinned_grouped_helper()

    def original_scalar_conversion():
        return {name: int(value) if value.dtype == torch.int64 else float(value)
                for name, value in values.items()}

    functions = {"original_scalar_reference": original_scalar_conversion,
                 "grouped_baseline": lambda: grouped(values),
                 "device_aware": lambda: qat._reporting_scalars(values)}
    for function in functions.values():
        exact(original_scalar_conversion(), function())
        for _ in range(100):
            function()
    samples = {key: [] for key in functions}
    repetitions, trials = 10000, 9
    # Rotate order each trial to reduce systematic order bias; no runtime gate.
    for trial in range(trials):
        names = list(functions)
        names = names[trial % len(names):] + names[:trial % len(names)]
        for name in names:
            samples[name].append(timeit.timeit(functions[name], number=repetitions)
                                 / repetitions * 1e6)
    return {"scope": "8 already-computed CPU scalars; excludes training and accelerator latency",
            "repetitions_per_trial": repetitions, "trials": trials,
            "microseconds_per_call": samples,
            "median_microseconds_per_call": {key: statistics.median(v)
                                             for key, v in samples.items()}}


def main():
    check_control()
    torch.set_num_threads(1)
    output = {"base_commit": BASE,
              "source_sha256": hashlib.sha256(Path(inspect.getsourcefile(qat)).read_bytes())
              .hexdigest(),
              "hardware": {"device": "CPU", "machine": platform.machine(),
                           "platform": platform.platform(), "torch": torch.__version__,
                           "threads": torch.get_num_threads()},
              "reporting_helper_census": helper_census(),
              "step_proofs": [step_proof(False), step_proof(True)],
              "cpu_reporting_benchmark": benchmark()}
    Path(__file__).with_name("summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({key: output[key] for key in (
        "hardware", "reporting_helper_census", "cpu_reporting_benchmark")}, indent=2))


if __name__ == "__main__":
    main()

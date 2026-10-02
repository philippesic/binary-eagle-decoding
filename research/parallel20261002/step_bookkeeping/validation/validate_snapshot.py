"""Independent tiny CPU fixture for joint_train_step bookkeeping equivalence."""

from __future__ import annotations

import copy
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
CONTROL_PATH = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")
ORIGINAL_RESET_UNIX = 1791049896

from w1a1_eagle.affine_binary import AffineBinaryConfig, install_affine_binary  # noqa: E402
from w1a1_eagle.learned_activation import LearnedActivationBank  # noqa: E402
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH  # noqa: E402
from w1a1_eagle.recurrent_qat import (  # noqa: E402
    JointQATConfig,
    RowBinaryLinear,
    W1AxContract,
    joint_optimizer,
)
from w1a1_eagle.recurrent_trace import TraceAudit  # noqa: E402


class AllocationCensus(TorchDispatchMode):
    """Count dispatcher clone outputs and nonempty CPU allocation outputs."""

    def __init__(self) -> None:
        super().__init__()
        self.clone_outputs: list[dict[str, object]] = []
        self.allocations: list[dict[str, object]] = []
        self.bool_comparisons = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        name = str(func)

        def tensors(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, (tuple, list)):
                for part in value:
                    yield from tensors(part)

        outputs = list(tensors(result))
        if name.startswith("aten.clone"):
            self.clone_outputs.extend(
                {"dtype": str(t.dtype), "numel": t.numel(), "bytes": t.numel() * t.element_size()}
                for t in outputs
            )
        if name.startswith(("aten.empty", "aten.zeros", "aten.ones", "aten.full")):
            self.allocations.extend(
                {
                    "op": name,
                    "dtype": str(t.dtype),
                    "numel": t.numel(),
                    "bytes": t.numel() * t.element_size(),
                }
                for t in outputs
            )
        if name.startswith("aten.lt"):
            self.bool_comparisons += sum(t.dtype == torch.bool for t in outputs)
        return result


def _check_control(checkpoint: str) -> None:
    control = json.loads(CONTROL_PATH.read_text())
    if control.get("research_stop"):
        raise RuntimeError(f"research stop requested before {checkpoint}")
    if control.get("reset_observed") or control.get("last_reset_unix") != ORIGINAL_RESET_UNIX:
        raise RuntimeError(f"allowance reset observed before {checkpoint}")
    remaining = 100 - float(control["last_weekly_used_percent"])
    if remaining <= 1:
        raise RuntimeError(f"weekly allowance at or below 1% before {checkpoint}")


def _linears(seed: int, mode: str):
    torch.random.default_generator.manual_seed(seed)
    linears = {}
    # Equal tiny widths satisfy all six canonical learned-activation boundaries.
    for ordinal, path in enumerate(CANDIDATE_D_BASE_TO_PATH.values()):
        base = (
            torch.tensor(
                [[0.72, -0.39, 0.18], [-0.44, 0.83, -0.27], [0.31, 0.12, -0.91]],
                dtype=torch.float32,
            )
            + ordinal * 0.001
        )
        scales = torch.tensor([0.7, 1.1, 0.9], dtype=torch.float32)
        linears[path] = RowBinaryLinear(base, scales, W1AxContract(1))
    config = JointQATConfig(
        contract=W1AxContract(1),
        device="cpu",
        activation_quantization="learned" if mode == "learned" else "fixed",
        affine_weights=AffineBinaryConfig(enabled=True, coverage="all")
        if mode == "affine"
        else None,
    )
    if mode == "learned":
        bank = LearnedActivationBank(1, {path: 3 for path in linears})
        bank.attach(linears)
    if mode == "affine":
        bank = install_affine_binary(
            linears,
            target=torch.nn.Linear(3, 3),
            config=config.affine_weights,
        )
    return linears, config


def _audit() -> TraceAudit:
    return TraceAudit(
        draft_labels=(-1, 1),
        valid_mask=(True, True),
        supported_mask=(False, True),
        ce_mask=(False, True),
        denominator_mask=(True, True),
        target_to_draft=(0, 1, 2),
        counts={"total": 2, "valid": 2, "unsupported": 1},
        per_depth={},
    )


def _logits(linears, mode: str) -> torch.Tensor:
    # Make each of the nine actual trainable projections participate in the graph.
    x = torch.tensor([[0.13, -0.41, 0.77], [-0.61, 0.28, 0.52]], dtype=torch.float32)
    for index, module in enumerate(linears.values()):
        x = module(x + (index + 1) * 0.031)
    return x


def _tree_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _tree_snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_tree_snapshot(v) for v in value)
    return copy.deepcopy(value)


def _compare_tree(left, right, path="root") -> dict[str, float | int]:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, msg=lambda msg: f"{path}: {msg}")
        return {"tensors": 1, "max_abs": 0.0}
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        out = {"tensors": 0, "max_abs": 0.0}
        for key in left:
            item = _compare_tree(left[key], right[key], f"{path}.{key}")
            out["tensors"] += item["tensors"]
        return out
    if isinstance(left, (list, tuple)):
        assert len(left) == len(right), path
        out = {"tensors": 0, "max_abs": 0.0}
        for index, (a, b) in enumerate(zip(left, right)):
            item = _compare_tree(a, b, f"{path}[{index}]")
            out["tensors"] += item["tensors"]
        return out
    assert left == right, (path, left, right)
    return {"tensors": 0, "max_abs": 0.0}


def _optimizer(linears, config):
    return joint_optimizer(linears, config)


def _run(step: Callable, mode: str, census: bool = False):
    linears, config = _linears(20261002, mode)
    optimizer = _optimizer(linears, config)
    # Seed optimizer state so the comparison verifies real moments and step counters.
    torch.random.default_generator.manual_seed(20261003)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter]["step"] = torch.tensor(3.0)
            optimizer.state[parameter]["exp_avg"] = torch.full_like(parameter, 0.017)
            optimizer.state[parameter]["exp_avg_sq"] = torch.full_like(parameter, 0.029)
    logits = _logits(linears, mode)
    collector = AllocationCensus()
    if census:
        with collector:
            metrics = step(linears, logits, _audit(), optimizer, config)
    else:
        metrics = step(linears, logits, _audit(), optimizer, config)
    values = {
        "metrics": metrics,
        "parameters": {path: module.state_dict() for path, module in linears.items()},
        "gradients": {
            f"{path}.{name}": p.grad
            for path, module in linears.items()
            for name, p in module.named_parameters()
        },
        "optimizer": optimizer.state_dict(),
    }
    counts = {
        "clone_outputs": collector.clone_outputs,
        "clone_count": len(collector.clone_outputs),
        "clone_float_count": sum(x["dtype"] == "torch.float32" for x in collector.clone_outputs),
        "clone_bool_count": sum(x["dtype"] == "torch.bool" for x in collector.clone_outputs),
        "float_clone_numel_9": sum(
            x["dtype"] == "torch.float32" and x["numel"] == 9 for x in collector.clone_outputs
        ),
        "float_clone_numel_3": sum(
            x["dtype"] == "torch.float32" and x["numel"] == 3 for x in collector.clone_outputs
        ),
        "bool_comparison_outputs": collector.bool_comparisons,
        "allocations": collector.allocations,
    }
    return _tree_snapshot(values), counts


def _nan_gradient_rejection(step: Callable) -> None:
    linears, config = _linears(20261002, "fixed")
    optimizer = _optimizer(linears, config)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            optimizer.state[parameter]["step"] = torch.tensor(3.0)
            optimizer.state[parameter]["exp_avg"] = torch.full_like(parameter, 0.017)
            optimizer.state[parameter]["exp_avg_sq"] = torch.full_like(parameter, 0.029)
    parameters_before = _tree_snapshot(
        {
            f"{path}.{name}": p
            for path, module in linears.items()
            for name, p in module.named_parameters()
        }
    )
    optimizer_before = _tree_snapshot(optimizer.state_dict())
    linears["fc"].latent_sign.register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    try:
        step(linears, _logits(linears, "fixed"), _audit(), optimizer, config)
    except ValueError as error:
        assert "nonfinite joint QAT gradient" in str(error)
    else:
        raise AssertionError("NaN gradient did not fail joint_train_step")
    parameters_after = {
        f"{path}.{name}": p
        for path, module in linears.items()
        for name, p in module.named_parameters()
    }
    _compare_tree(parameters_before, parameters_after, "parameters_after_rejection")
    _compare_tree(optimizer_before, optimizer.state_dict(), "optimizer_after_rejection")


def main():
    _check_control("main")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    report = {
        "environment": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
            "device": "CPU",
            "machine": os.uname().machine,
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "cases": {},
    }
    from research.parallel20261002.step_bookkeeping.reference.snapshot_step import (
        joint_train_step_without_snapshot_clone,
    )
    from w1a1_eagle.recurrent_qat import joint_train_step as production_step

    for mode in ("fixed", "learned", "affine"):
        _check_control(f"{mode} fixture")
        reference, reference_census = _run(production_step, mode, census=True)
        candidate, candidate_census = _run(
            joint_train_step_without_snapshot_clone, mode, census=True
        )
        equality = _compare_tree(reference, candidate)
        assert reference["metrics"] == candidate["metrics"], mode
        assert reference_census["float_clone_numel_9"] == 9
        assert candidate_census["float_clone_numel_9"] == 0
        assert (
            reference_census["float_clone_numel_3"] == candidate_census["float_clone_numel_3"] == 9
        )
        assert reference_census["clone_float_count"] - candidate_census["clone_float_count"] == 9
        assert (
            reference_census["bool_comparison_outputs"]
            == candidate_census["bool_comparison_outputs"]
        )
        assert reference_census["clone_bool_count"] == candidate_census["clone_bool_count"]
        report["cases"][mode] = {
            "result": "passed exact metrics, gradients, parameters, and optimizer state",
            "metrics": reference["metrics"],
            "equal_tensor_count": equality["tensors"],
            "reference_census": reference_census,
            "candidate_census": candidate_census,
        }
    _check_control("NaN gradient fixture")
    _nan_gradient_rejection(production_step)
    _nan_gradient_rejection(joint_train_step_without_snapshot_clone)
    report["nan_gradient"] = "both steps fail before parameter or optimizer state changes"
    source = torch.tensor([-1.0, 1.0], dtype=torch.float32)
    prior_bool = source.detach() < 0
    assert prior_bool.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    source.add_(2)
    assert prior_bool.tolist() == [True, False]
    report["independent_boolean_snapshot"] = "passed source mutation does not alter bool result"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

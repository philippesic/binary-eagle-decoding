"""Independent CPU checks for the immediate learned-activation reuse prototype."""

from __future__ import annotations

import gc
import json
import platform
import weakref
from pathlib import Path

import torch

from research.parallel20261002.activation_reuse.reference.prototype import (
    immediate_group,
    install_reuse,
    quantize,
)
from research.parallel20261002.auxiliary_vjp.reference.audit import (
    ATOL,
    RTOL,
    batch,
    combined,
    parameters,
    snapshot,
)
from w1a1_eagle.recurrent_provider import forward_torch_round
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    joint_optimizer,
    joint_train_step,
    shared_round_hard_signs,
)
from w1a1_eagle.recurrent_trace import TraceAudit

CONTROL = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")
RUN = Path("runs/parallel20261002/activation_reuse/validation-independent-01")


def check_control() -> dict:
    state = json.loads(CONTROL.read_text())
    used = state.get("last_weekly_used_percent", 0)
    if (
        state["research_stop"]
        or state.get("reset_observed")
        or state.get("last_reset_unix") != state.get("initial_reset_unix")
        or used >= 100 - state.get("threshold_remaining_percent", 1)
    ):
        raise RuntimeError("Research stopped by supervisor control")
    return {
        "research_stop": state["research_stop"],
        "reset_observed": state.get("reset_observed"),
        "last_reset_unix": state.get("last_reset_unix"),
        "remaining_percent": 100 - used,
    }


def _max_error(actual: torch.Tensor, wanted: torch.Tensor) -> float:
    return float((actual - wanted).abs().max()) if actual.numel() else 0.0


def _assert_tensor(actual: torch.Tensor, wanted: torch.Tensor, label: str) -> float:
    torch.testing.assert_close(actual, wanted, atol=ATOL, rtol=RTOL, msg=label)
    return _max_error(actual, wanted)


def saved_storage_census(adapter, b) -> tuple[dict, dict]:
    """Read the two tensors saved directly by each learned STE node."""
    calls = []
    for module in adapter.linears.values():
        quantizer = getattr(module, "activation_quantizer", None)
        if quantizer is None or hasattr(quantizer, "_validation_original_forward"):
            continue
        original = quantizer.forward

        def observe(self, input, *, _original=original, **kwargs):
            result = _original(input, **kwargs)
            node = result.values.grad_fn
            saved = () if node is None else node.saved_tensors
            calls.append(
                {
                    "boundary": self.boundary,
                    "node_type": None if node is None else type(node).__name__,
                    "storages": [
                        {
                            "storage": tensor.untyped_storage().data_ptr(),
                            "dtype": str(tensor.dtype),
                            "shape": tuple(tensor.shape),
                            "bytes": tensor.numel() * tensor.element_size(),
                        }
                        for tensor in saved
                    ],
                }
            )
            return result

        quantizer._validation_original_forward = original
        from types import MethodType

        quantizer.forward = MethodType(observe, quantizer)

    result = snapshot(b, adapter, cache=True, chunk=2, shared=True)
    grouped = {}
    for call in calls:
        if call["boundary"] in ("qkv", "gate_up"):
            grouped.setdefault(call["boundary"], []).append(call)
    summary = {}
    for boundary, group_calls in grouped.items():
        unique = {}
        for call in group_calls:
            for index, storage in enumerate(call["storages"]):
                unique.setdefault(storage["storage"], storage)
        summary[boundary] = {
            "quantizer_invocations": len(group_calls),
            "attached_ste_invocations": sum(call["node_type"] is not None for call in group_calls),
            "saved_tensors_per_invocation": sorted(len(call["storages"]) for call in group_calls),
            "ste_node_types": sorted({call["node_type"] for call in group_calls}, key=str),
            "unique_support_derivative_storages": len(unique),
            "unique_support_derivative_bytes": sum(row["bytes"] for row in unique.values()),
            "unique_saved_by_dtype": _storage_dtype_counts(unique.values()),
        }
    return result, summary


def _storage_dtype_counts(rows):
    result = {}
    for row in rows:
        result[row["dtype"]] = result.get(row["dtype"], 0) + 1
    return result


def graph_and_storage(bits: int) -> dict:
    check_control()
    b = batch(parent=2, depth=3, terminal=True)
    baseline_adapter = combined(bits, "single_forward")
    reuse_adapter = install_reuse(combined(bits, "single_forward"))
    baseline, baseline_storage = saved_storage_census(baseline_adapter, b)
    reused, reused_storage = saved_storage_census(reuse_adapter, b)
    # The read-only auxiliary fixture's algebraic projection/correction is an
    # independent handwritten VJP oracle; it does not use the production
    # learned-activation or affine projection backward.
    oracle = snapshot(b, combined(bits, oracle=True), serial=True)
    errors = {name: _assert_tensor(reused[name], baseline[name], name) for name in baseline}
    oracle_errors = {
        name: _assert_tensor(baseline[name], oracle[name], f"oracle {name}") for name in baseline
    }
    groups = reuse_adapter.reuse_events
    # Three serial proposal steps contain exactly one QKV and one gate/up group
    # apiece. Each group must compute one unique quantization and hit each sibling.
    assert len(groups) == 6, groups
    expected = [("qkv", 2, 1), ("gate_up", 1, 1)] * 3
    observed = [(row["label"], row["hits"], row["misses"]) for row in groups]
    assert observed == expected, observed
    assert all(row["retained"] == 0 for row in groups), groups
    for boundary, expected in (
        ("qkv", {"invocations": (11, 11), "attached_calls": (9, 9), "unique_storages": (18, 6)}),
        ("gate_up", {"invocations": (6, 6), "attached_calls": (6, 6), "unique_storages": (12, 6)}),
    ):
        got_invocations = tuple(
            census[boundary]["quantizer_invocations"]
            for census in (baseline_storage, reused_storage)
        )
        assert got_invocations == expected["invocations"], (
            bits,
            boundary,
            got_invocations,
            expected,
        )
        got_attached_calls = tuple(
            census[boundary]["attached_ste_invocations"]
            for census in (baseline_storage, reused_storage)
        )
        assert got_attached_calls == expected["attached_calls"], (
            bits,
            boundary,
            got_attached_calls,
            expected,
        )
        got_storage = tuple(
            census[boundary]["unique_support_derivative_storages"]
            for census in (baseline_storage, reused_storage)
        )
        assert got_storage == expected["unique_storages"], (bits, boundary, got_storage, expected)
    return {
        "bits": bits,
        "max_abs_vjp_or_output_error": max(errors.values(), default=0.0),
        "max_abs_independent_hand_vjp_error": max(oracle_errors.values(), default=0.0),
        "compared_outputs_and_vjps": len(errors),
        "group_events": groups,
        "baseline_saved_storage": baseline_storage,
        "reuse_saved_storage": reused_storage,
    }


def one_clipped_update(bits: int = 8) -> dict:
    check_control()
    baseline = combined(bits, "single_forward")
    reused = install_reuse(combined(bits, "single_forward"))
    config = JointQATConfig(
        W1AxContract(bits),
        activation_quantization="learned",
        a1_computation="single_forward",
        max_grad_norm=0.07,
        fusion_correction=baseline.linears["fc"].fusion_correction.config,
        affine_weights=baseline.linears["fc"].affine_binary.config,
    )
    audit = TraceAudit(
        (3, 3, 3, -1),
        (True, True, True, False),
        (True, True, True, False),
        (True, True, True, False),
        (True, True, True, False),
        tuple(range(7)),
        {},
        {},
    )
    observations = []
    for adapter in (baseline, reused):
        check_control()
        optimizer = joint_optimizer(adapter.linears, config)
        b = batch(parent=2, depth=3, terminal=True)
        with shared_round_hard_signs(adapter.linears):
            logits = forward_torch_round(
                b,
                adapter,
                7,
                optimize_cache=True,
                optimize_head=False,
                context_chunk_size=2,
            )
            metric = joint_train_step(adapter.linears, logits, audit, optimizer, config)
        named = parameters(adapter)
        states = {
            name: {
                key: value.detach().clone()
                for key, value in optimizer.state[param].items()
                if isinstance(value, torch.Tensor)
            }
            for name, param in named.items()
        }
        observations.append(
            {
                "logits": logits.detach(),
                "metrics": metric,
                "parameters": {name: p.detach().clone() for name, p in named.items()},
                "clipped_gradients": {name: p.grad.detach().clone() for name, p in named.items()},
                "optimizer_state": states,
            }
        )
    base, actual = observations
    assert set(base["metrics"]) == set(actual["metrics"])
    metric_errors = {}
    for key, expected in base["metrics"].items():
        got = actual["metrics"][key]
        if isinstance(expected, int):
            assert got == expected, (key, got, expected)
            metric_errors[key] = 0.0
        else:
            metric_errors[key] = abs(float(got) - float(expected))
            assert metric_errors[key] <= ATOL + RTOL * abs(float(expected)), (key, got, expected)
    update_errors = {"logits": _assert_tensor(actual["logits"], base["logits"], "update logits")}
    for family in ("parameters", "clipped_gradients"):
        for name, wanted in base[family].items():
            update_errors[f"{family}.{name}"] = _assert_tensor(actual[family][name], wanted, name)
    for name, tensors in base["optimizer_state"].items():
        for key, wanted in tensors.items():
            update_errors[f"optimizer_state.{name}.{key}"] = _assert_tensor(
                actual["optimizer_state"][name][key], wanted, f"{name}.{key}"
            )
    norm = float(torch.sqrt(sum(p.grad.square().sum() for p in parameters(reused).values())))
    assert norm <= config.max_grad_norm + 1e-6
    return {
        "bits": bits,
        "gradient_norm_after_clipping": norm,
        "clip_limit": config.max_grad_norm,
        "max_abs_update_or_state_error": max(update_errors.values(), default=0.0),
        "compared_tensors": len(update_errors),
        "metric_errors": metric_errors,
        "group_events": reused.reuse_events,
    }


def stale_controls() -> dict:
    """Prove mutation and grad-mode boundaries cannot reuse a stale result."""
    check_control()
    adapter = combined(8, "single_forward")
    quantizer = adapter.linears["midlayer.mlp.gate_proj"].activation_quantizer
    base = torch.tensor([[0.2, -0.7, 0.4, 0.9, -0.1, 0.6, -0.8, 0.3]], requires_grad=True)
    value = base * 1.0
    with immediate_group("mutation") as state:
        before = quantize(quantizer, value)
        with torch.no_grad():
            value.add_(0.25)
        after = quantize(quantizer, value)
    assert state.hits == 0 and state.misses == 2 and not state.entries
    assert not torch.equal(before.values, after.values)

    with immediate_group("grad_mode") as state:
        with torch.no_grad():
            detached = quantize(quantizer, value)
        attached = quantize(quantizer, value)
    assert state.hits == 0 and state.misses == 2 and not state.entries
    assert not detached.values.requires_grad
    assert attached.values.requires_grad
    grad = torch.autograd.grad(attached.values.sum(), quantizer.parameter)[0]
    assert torch.isfinite(grad)
    return {
        "mutation": {"hits": 0, "misses": 2, "outputs_changed": True, "entries_after_exit": 0},
        "no_grad_to_grad": {
            "hits": 0,
            "misses": 2,
            "first_output_attached": False,
            "second_output_attached": True,
            "parameter_gradient_finite": bool(torch.isfinite(grad)),
            "entries_after_exit": 0,
        },
    }


def cache_lifetime_control() -> dict:
    check_control()
    adapter = combined(8, "single_forward")
    quantizer = adapter.linears["midlayer.mlp.gate_proj"].activation_quantizer
    value = torch.tensor([[0.2, -0.7, 0.4, 0.9, -0.1, 0.6, -0.8, 0.3]])
    with immediate_group("lifetime") as state:
        result = quantize(quantizer, value)
        result_ref = weakref.ref(result)
        assert result_ref() is result
    entries_after_exit = len(state.entries)
    del result
    gc.collect()
    assert entries_after_exit == 0 and result_ref() is None
    return {
        "entries_after_group_exit": entries_after_exit,
        "cached_result_released_after_caller_reference_drop": True,
    }


def run() -> dict:
    torch.set_num_threads(1)
    check_control()
    result = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "CPU",
            "torch_threads": torch.get_num_threads(),
        },
        "control_at_start": check_control(),
        "graph_and_storage": [graph_and_storage(bits) for bits in (1, 4, 8)],
        "one_clipped_update": one_clipped_update(8),
        "stale_controls": stale_controls(),
        "cache_lifetime_control": cache_lifetime_control(),
        "control_at_end": check_control(),
    }
    return result


if __name__ == "__main__":
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "result.json"
    path.write_text(json.dumps(run(), indent=2, sort_keys=True) + "\n")
    print(path)

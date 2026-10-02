"""Independent CPU-only curriculum checkpoint probes using synthetic fixtures.

Run from the repository root with the environment recorded in the report.
Every case checks the shared research-stop control file before doing work.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

import torch
import numpy as np

from test_qat_curriculum_runner import make, stages
import w1a1_eagle.qat_curriculum_runner as runner_module
from w1a1_eagle.qat_curriculum_runner import audit_provider_round, joint_train_step


REPO = Path(__file__).resolve().parents[3]
CONTROL = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")
RUNS = REPO / "runs/parallel20261002/independent-validation-torch214"


def check_control(case: str) -> None:
    status = json.loads(CONTROL.read_text())
    if status.get("research_stop"):
        raise SystemExit(f"research_stop set before {case}: {status.get('stop_reason')}")
    if status.get("monitor_interrupt"):
        raise SystemExit(f"monitor interrupt set before {case}")


def same(left, right) -> bool:
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            same(left[k], right[k]) for k in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            same(a, b) for a, b in zip(left, right)
        )
    return left == right


def max_tensor_difference(left, right) -> float:
    differences = []
    def visit(a, b):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or a.shape != b.shape:
                raise AssertionError("model payload tensor inventories differ")
            if a.numel():
                differences.append(float((a.double() - b.double()).abs().max()))
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise AssertionError("model payload mapping inventories differ")
            for key in a:
                visit(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise AssertionError("model payload sequence inventories differ")
            for x, y in zip(a, b):
                visit(x, y)
    visit(left, right)
    return max(differences, default=0.0)


def perform_next_update(reader):
    batch = next(reader.provider.rounds())
    audit = audit_provider_round(batch, reader.provider)
    logits = reader._forward(batch)
    metrics = joint_train_step(reader.linears, logits, audit, reader.optimizer, reader.qat)
    return metrics, reader._model_payload()


def read_payload(run: Path):
    pointer_path = run / "latest.json"
    pointer = json.loads(pointer_path.read_text())
    path = run / "checkpoints" / pointer["file"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return pointer, path, payload


def rewrite_payload(run: Path, mutate) -> None:
    pointer, path, payload = read_payload(run)
    mutate(payload, pointer)
    torch.save(payload, path)
    pointer["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    pointer_path = run / "latest.json"
    pointer_path.write_text(json.dumps(pointer, sort_keys=True) + "\n")


def case_exact_midphase(run: Path) -> dict:
    check_control("exact midphase round-trip")
    curriculum = stages((8, 1), updates=3)
    original = make(run, curriculum, recipe=True)
    original.run(require_smoke=False, max_new_updates=1)
    original.save()
    pointer, _, payload = read_payload(run)
    resumed = make(run, curriculum, recipe=True)
    resumed.resume()
    checks = {
        "midphase_pointer_updates": pointer["global_updates"],
        "optimizer_has_moments": bool(payload["optimizer"]["state"]),
        "core_and_optional_model_exact": same(payload["model"], resumed._model_payload()),
        "activation_bank_exact": same(
            payload["model"]["activation_bank"], resumed.bank.checkpoint()
        ),
        "optimizer_exact": same(payload["optimizer"], resumed.optimizer.state_dict()),
        "rng_exact": same(payload["rng"], resumed.rng),
        "cursor_epoch_exact": (payload["cursor"], payload["epoch"])
        == (resumed.cursor, resumed.epoch),
    }
    if not all(checks[k] for k in checks if isinstance(checks[k], bool)):
        raise AssertionError(checks)
    return checks


def case_boundary(run: Path) -> dict:
    check_control("phase-boundary checkpoint")
    curriculum = stages((8, 1), updates=3)
    original = make(run, curriculum, recipe=True)
    original.run(require_smoke=False, max_new_updates=1)
    original.state.finish_phase()
    boundary_pointer = original.save()
    boundary_payload = torch.load(
        run / "checkpoints" / boundary_pointer["file"], map_location="cpu", weights_only=False
    )
    resumed = make(run, curriculum, recipe=True)
    resumed.resume()
    before_transition = {
        "state_phase": resumed.state.phase_index,
        "model_phase": resumed.model_phase,
        "transition_count": len(resumed.state.transitions),
        "activation_bits": resumed.qat.contract.activation_bits,
        "optimizer_state_count": len(resumed.optimizer.state),
        "boundary_model_bits": boundary_payload["model_phase"],
    }
    resumed._transition()
    after_transition = {
        "state_phase": resumed.state.phase_index,
        "model_phase": resumed.model_phase,
        "transition_count": len(resumed.state.transitions),
        "activation_bits": resumed.qat.contract.activation_bits,
        "optimizer_state_count": len(resumed.optimizer.state),
    }
    if before_transition != {
        "state_phase": 1,
        "model_phase": 0,
        "transition_count": 0,
        "activation_bits": 8,
        "optimizer_state_count": len(boundary_payload["optimizer"]["state"]),
        "boundary_model_bits": 0,
    }:
        raise AssertionError(before_transition)
    if after_transition != {
        "state_phase": 1,
        "model_phase": 1,
        "transition_count": 1,
        "activation_bits": 1,
        "optimizer_state_count": 0,
    }:
        raise AssertionError(after_transition)
    return {"source_boundary": before_transition, "after_transition": after_transition}


def case_activation_damage(run: Path, damage: str) -> dict:
    check_control(f"activation {damage}")
    curriculum = stages((8, 1), updates=3)
    writer = make(run, curriculum, recipe=True)
    writer.run(require_smoke=False, max_new_updates=1)
    writer.save()
    def mutate(payload, _pointer):
        bank = payload["model"]["activation_bank"]["parameters"]
        if damage == "omitted":
            del payload["model"]["activation_bank"]
        elif damage == "partial":
            del bank[next(iter(bank))]
        else:
            raise AssertionError(damage)

    rewrite_payload(run, mutate)
    reader = make(run, curriculum, recipe=True)
    try:
        reader.resume()
    except Exception as error:
        return {"rejected": True, "error": f"{type(error).__name__}: {error}"}
    return {"rejected": False, "error": None}


def case_optimizer_damage(run: Path, damage: str) -> dict:
    check_control(f"optimizer {damage}")
    curriculum = stages((8, 1), updates=3)
    writer = make(run, curriculum, recipe=True)
    writer.run(require_smoke=False, max_new_updates=1)
    writer.save()
    # Hold an exact-resume control loaded from the pristine checkpoint.
    control = make(run, curriculum, recipe=True)
    control.resume()

    def mutate(payload, _pointer):
        state = payload["optimizer"]["state"]
        if not state:
            raise AssertionError("fixture update did not create optimizer moments")
        if damage == "omitted":
            payload["optimizer"]["state"] = {}
        elif damage == "partial":
            first = next(iter(state))
            payload["optimizer"]["state"] = {first: state[first]}
        elif damage == "nonfinite":
            entry = next(iter(state.values()))
            entry["exp_avg"].view(-1)[0] = float("nan")
        elif damage == "wrong_shape":
            entry = next(iter(state.values()))
            entry["exp_avg"] = torch.zeros((1,), dtype=entry["exp_avg"].dtype)
        else:
            raise AssertionError(damage)

    rewrite_payload(run, mutate)

    # Build the exact-resume control from the original checkpoint before applying
    # the same next update to the intentionally damaged optimizer checkpoint.
    control_metrics, control_model = perform_next_update(control)
    reader = make(run, curriculum, recipe=True)
    try:
        reader.resume()
    except Exception as error:
        return {"rejected": True, "error": f"{type(error).__name__}: {error}"}
    states = list(reader.optimizer.state.values())
    nonfinite = any(
        isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
        for entry in states
        for v in entry.values()
    )
    wrong_shapes = []
    for parameter, entry in reader.optimizer.state.items():
        for key in ("exp_avg", "exp_avg_sq"):
            value = entry.get(key)
            if isinstance(value, torch.Tensor) and value.shape != parameter.shape:
                wrong_shapes.append({"state": key, "expected": list(parameter.shape), "actual": list(value.shape)})
    first = states[0] if states else {}
    try:
        update, next_model = perform_next_update(reader)
        next_update = {"completed": True, "finite_loss": bool(np.isfinite(update["loss"]))}
    except Exception as error:
        next_model = None
        next_update = {"completed": False, "error": f"{type(error).__name__}: {error}"}
    return {
        "rejected": False,
        "error": None,
        "restored_state_entries": len(states),
        "nonfinite_moments": nonfinite,
        "wrong_shapes": wrong_shapes,
        "first_entry_keys": sorted(first),
        "step_values": [str(entry.get("step")) for entry in states[:3]],
        "next_step": next_update,
        "next_update_max_abs_model_difference_vs_exact_resume": (
            None if next_model is None else max_tensor_difference(control_model, next_model)
        ),
        "exact_control_loss": control_metrics["loss"],
        "damaged_resume_loss": None if not next_update["completed"] else update["loss"],
    }


def case_transition_publication_failure(run: Path) -> dict:
    check_control("target transition pointer publication failure")
    curriculum = stages((8, 1), updates=3)

    def make_boundary(path):
        runner = make(path, curriculum, recipe=True)
        runner.run(require_smoke=False, max_new_updates=1)
        runner.state.finish_phase()
        runner.save()
        return runner

    faulted = make_boundary(run / "faulted")
    calls = 0
    original_atomic_json = runner_module.atomic_json

    def fail_target_publication(path, value):
        nonlocal calls
        if Path(path).name == "latest.json":
            calls += 1
            if calls == 2:
                raise OSError("injected target latest.json publication failure")
        return original_atomic_json(path, value)

    with patch.object(runner_module, "atomic_json", side_effect=fail_target_publication):
        try:
            faulted._transition()
        except OSError as error:
            injected_error = str(error)
        else:
            raise AssertionError("transition latest.json fault did not fire")
    pointer_after_fault, _, source_payload = read_payload(run / "faulted")
    if pointer_after_fault["model_phase"] != 0 or source_payload["model_phase"] != 0:
        raise AssertionError("latest pointer did not remain at committed source phase")

    replay = make(run / "faulted", curriculum, recipe=True)
    replay.resume()
    replay._transition()

    fault_free = make_boundary(run / "fault_free")
    fault_free._transition()
    comparison = {
        "model_payload_exact": same(fault_free._model_payload(), replay._model_payload()),
        "optimizer_exact": same(fault_free.optimizer.state_dict(), replay.optimizer.state_dict()),
        "state_phase": replay.state.phase_index,
        "model_phase": replay.model_phase,
        "transition_count": len(replay.state.transitions),
        "optimizer_state_count": len(replay.optimizer.state),
    }
    if not comparison["model_payload_exact"] or not comparison["optimizer_exact"]:
        raise AssertionError(comparison)
    return {
        "injected_error": injected_error,
        "latest_after_failure_model_phase": pointer_after_fault["model_phase"],
        "replay_comparison_to_fault_free": comparison,
    }


def case_phase_ancestry(run: Path) -> dict:
    check_control("adversarial phase ancestry")
    curriculum = stages((8, 1), updates=3)
    writer = make(run, curriculum, recipe=True)
    writer.run(require_smoke=False, max_new_updates=1)
    writer.save()

    def mutate(payload, pointer):
        payload["state"]["phase_index"] = 1
        payload["model_phase"] = 1
        pointer["model_phase"] = 1

    rewrite_payload(run, mutate)
    reader = make(run, curriculum, recipe=True)
    try:
        reader.resume()
    except Exception as error:
        return {"rejected": True, "error": f"{type(error).__name__}: {error}"}
    return {"rejected": False, "error": None}


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    RUNS.mkdir(parents=True, exist_ok=True)
    report = {}
    scenarios = [
        ("exact_midphase", case_exact_midphase),
        ("phase_boundary", case_boundary),
        ("transition_publication_failure", case_transition_publication_failure),
        ("activation_omitted", lambda p: case_activation_damage(p, "omitted")),
        ("activation_partial", lambda p: case_activation_damage(p, "partial")),
        ("optimizer_omitted", lambda p: case_optimizer_damage(p, "omitted")),
        ("optimizer_partial", lambda p: case_optimizer_damage(p, "partial")),
        ("optimizer_nonfinite", lambda p: case_optimizer_damage(p, "nonfinite")),
        ("optimizer_wrong_shape", lambda p: case_optimizer_damage(p, "wrong_shape")),
        ("phase_ancestry", case_phase_ancestry),
    ]
    for name, function in scenarios:
        check_control(name)
        run = RUNS / name
        if run.exists():
            shutil.rmtree(run)
        run.mkdir(parents=True)
        report[name] = function(run)
        (run / "result.json").write_text(json.dumps(report[name], indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

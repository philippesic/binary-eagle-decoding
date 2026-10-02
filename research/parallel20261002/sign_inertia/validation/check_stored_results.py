"""Independent consistency check for the owner's stored 99 CPU trajectories."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

RESULTS = Path(
    "/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/sign_inertia/results.json"
)
LINEAR_X = ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))
XOR_X = ((1, 1), (1, -1), (-1, 1), (-1, -1))
SEQUENCES = ((1, 1, -1), (1, -1, -1), (-1, 1, 1), (-1, -1, 1))
SCALES = (0.01, 1.0, 100.0)


def recurrent(signs, scale):
    rows = []
    for sequence in SEQUENCES:
        hidden = 0.25
        values = []
        for value in sequence:
            alpha = (abs(value) + abs(hidden)) / 2
            z = scale * alpha * (signs[0] * value + signs[1] * (1 if hidden >= 0 else -1)) + 0.15
            values.append(z)
            hidden = math.tanh(z)
        rows.append(values)
    return rows


def close(a, b, *, atol=1e-6):
    return abs(float(a) - float(b)) <= atol * max(1.0, abs(float(a)), abs(float(b)))


def check_capacity(data):
    target = (1, -1, 1)
    linear_y = tuple(1 if sum(a * b for a, b in zip(target, row)) > 0 else -1 for row in LINEAR_X)
    xor_y = (1, -1, -1, 1)
    recurrent_y = tuple(tuple(1 if z > 0 else -1 for z in row) for row in recurrent((1, -1), 1.0))
    fixtures = {
        "linear": (LINEAR_X, linear_y, 3),
        "xor": (XOR_X, xor_y, 2),
    }
    feasible_counts = {}
    for name, (inputs, labels, width) in fixtures.items():
        for scale in SCALES:
            expected_states = []
            for signs in itertools.product((-1, 1), repeat=width):
                logits = [scale * sum(a * b for a, b in zip(signs, row)) for row in inputs]
                margins = [z * y for z, y in zip(logits, labels)]
                expected_states.append((signs, logits, margins))
            saved = data["capacity"][name][str(int(scale) if scale.is_integer() else scale)]
            assert len(saved["states"]) == len(expected_states)
            for actual, (signs, logits, margins) in zip(saved["states"], expected_states):
                assert tuple(actual["signs"]) == signs
                assert all(close(a, b) for a, b in zip(actual["logits"], logits))
                assert actual["correct"] == sum(margin > 0 for margin in margins)
                assert close(actual["min_margin"], min(margins))
            count = sum(all(margin > 0 for margin in margins) for _, _, margins in expected_states)
            assert saved["feasible_states"] == count
            feasible_counts[(name, scale)] = count

    for scale in SCALES:
        expected_states = []
        for signs in itertools.product((-1, 1), repeat=2):
            logits = recurrent(signs, scale)
            margins = [
                [z * y for z, y in zip(row, label)] for row, label in zip(logits, recurrent_y)
            ]
            expected_states.append((signs, logits, margins))
        saved = data["capacity"]["recurrent"][str(int(scale) if scale.is_integer() else scale)]
        assert len(saved["states"]) == len(expected_states)
        for actual, (signs, logits, margins) in zip(saved["states"], expected_states):
            assert tuple(actual["signs"]) == signs
            assert len(actual["logits"]) == len(logits)
            for actual_row, expected_row in zip(actual["logits"], logits):
                assert all(close(a, b, atol=1e-5) for a, b in zip(actual_row, expected_row))
            flat = [value for row in margins for value in row]
            assert actual["correct"] == sum(value > 0 for value in flat)
            assert close(actual["min_margin"], min(flat), atol=1e-5)
        count = sum(
            all(value > 0 for row in margins for value in row) for _, _, margins in expected_states
        )
        assert saved["feasible_states"] == count
        feasible_counts[("recurrent", scale)] = count
    return feasible_counts


def check_trajectories(data):
    assert data["schema"] == "sign_inertia_cpu_v1"
    assert data["device"] == "CPU" and data["precision"] == "torch.float32"
    assert len(data["runs"]) == 99
    controlled = [
        run
        for run in data["runs"]
        if run["summary"]["fixture"] in ("linear", "recurrent")
        and run["summary"]["recipe"]["optimizer"] == "adamw"
    ]
    assert len(controlled) == 96
    combos = set()
    useful_total = wrong_total = flips_total = flip_backs_total = 0
    for run in data["runs"]:
        summary = run["summary"]
        trajectory = run["trajectory"]
        assert len(trajectory) == summary["budget"] == 64
        assert [row["step"] for row in trajectory] == list(range(1, 65))
        assert trajectory[-1] == summary["final"]
        final_margins = summary["final"]["decision_margins"]
        decision_count = sum(len(row) if isinstance(row, list) else 1 for row in final_margins)
        useful_steps = []
        wrong_events = 0
        full_updates = 0
        for record in trajectory:
            previous = record["old_sign_post_scale_metrics"]
            useful = record["sign_flips"] > 0 and (
                record["correct"] > previous["correct"]
                or (
                    record["correct"] == previous["correct"]
                    and record["min_margin"] > previous["min_margin"] + 1e-5
                )
            )
            wrong = record["sign_flips"] > 0 and (
                record["correct"] < previous["correct"]
                or (
                    record["correct"] == previous["correct"]
                    and record["min_margin"] < previous["min_margin"] - 1e-5
                )
            )
            assert bool(record["useful_flip"]) == bool(useful)
            useful_steps.extend([record["step"]] if useful else [])
            wrong_events += int(wrong)
            full_updates += int(record["correct"] == decision_count)
            assert record["correct"] == sum(
                value > 0
                for row in record["decision_margins"]
                for value in (row if isinstance(row, list) else [row])
            )
        assert summary["first_useful_flip"] == (useful_steps[0] if useful_steps else None)
        assert summary["wrong_flip_events"] == wrong_events
        assert summary["full_correct_updates"] == full_updates
        assert summary["tail16_full_correct_updates"] == sum(
            row["correct"] == decision_count for row in trajectory[-16:]
        )
        useful_total += len(useful_steps)
        wrong_total += wrong_events
        flips_total += sum(row["sign_flips"] for row in trajectory)
        flip_backs_total += sum(row["flip_backs"] for row in trajectory)
        if (
            summary["fixture"] in ("linear", "recurrent")
            and summary["recipe"]["optimizer"] == "adamw"
        ):
            combos.add(
                (
                    summary["fixture"],
                    summary["initial_scale"],
                    summary["latent_magnitude"],
                    summary["scale_mode"],
                    summary["option"],
                )
            )
    assert len(combos) == 96
    recurrent_example = next(
        run["summary"]
        for run in data["runs"]
        if run["summary"]["fixture"] == "recurrent"
        and run["summary"]["initial_scale"] == 1
        and run["summary"]["latent_magnitude"] == 0.02
        and run["summary"]["scale_mode"] == "joint"
        and run["summary"]["option"] == "baseline"
    )
    return {
        "runs": len(data["runs"]),
        "controlled_runs": len(controlled),
        "unique_controlled_cases": len(combos),
        "useful_flip_events": useful_total,
        "wrong_flip_events": wrong_total,
        "observed_flip_events": flips_total,
        "flip_backs": flip_backs_total,
        "recurrent_scale1_m002_joint_baseline": {
            key: recurrent_example[key]
            for key in (
                "first_useful_flip",
                "wrong_flip_events",
                "full_correct_updates",
                "tail16_full_correct_updates",
            )
        },
    }


if __name__ == "__main__":
    result = json.loads(RESULTS.read_text())
    capacity = check_capacity(result)
    trajectories = check_trajectories(result)
    print(
        json.dumps(
            {
                "capacity_feasible_states": {
                    f"{name}:{scale:g}": count for (name, scale), count in capacity.items()
                },
                "trajectory_checks": trajectories,
            },
            indent=2,
        )
    )

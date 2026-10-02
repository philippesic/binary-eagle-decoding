"""Independent adversarial checks for the staged resume research prototype.

All training data is the existing deterministic tiny CPU fixture. The script
never accesses model weights or project data.
"""

from __future__ import annotations

import copy
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from test_qat_curriculum_runner import make  # noqa: E402

from research.parallel20261002.resume_validation.reference.staged_resume import (  # noqa: E402
    stage_resume,
)
from w1a1_eagle.qat_optimization import BinaryOptimizationConfig  # noqa: E402


def rng_snapshot():
    return (
        random.getstate(),
        copy.deepcopy(np.random.get_state()),
        torch.random.get_rng_state().clone(),
    )


def assert_rng_equal(test, left, right):
    test.assertEqual(left[0], right[0])
    test.assertEqual(left[1][0], right[1][0])
    np.testing.assert_array_equal(left[1][1], right[1][1])
    test.assertEqual(left[1][2:], right[1][2:])
    torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def live_snapshot(runner):
    return {
        "state": copy.deepcopy(runner.state.state_dict()),
        "model_phase": runner.model_phase,
        "cursor": runner.cursor,
        "epoch": runner.epoch,
        "unique_rows": set(runner.unique_rows),
        "unique_prompts": set(runner.unique_prompts),
        "timings": (
            runner.occupancy_seconds,
            runner.smoke_seconds,
            runner.transition_seconds,
        ),
        "model": {
            path: copy.deepcopy(module.state_dict()) for path, module in runner.linears.items()
        },
        "parameter_ids": {
            path: tuple(id(parameter) for parameter in module.parameters())
            for path, module in runner.linears.items()
        },
        "parameters": {
            name: (
                id(parameter),
                parameter.untyped_storage().data_ptr(),
                parameter.requires_grad,
                None if parameter.grad is None else parameter.grad.detach().clone(),
            )
            for name, parameter in runner.drafter.named_parameters()
        },
        "optimizer": copy.deepcopy(runner.optimizer.state_dict()),
        "rng": rng_snapshot(),
        "checkpoint": copy.deepcopy(runner.last_checkpoint),
        "accounted_at": runner._accounted_at,
    }


def assert_live_equal(test, before, runner):
    after = live_snapshot(runner)
    for key in (
        "state",
        "model_phase",
        "cursor",
        "epoch",
        "unique_rows",
        "unique_prompts",
        "timings",
        "parameter_ids",
        "parameters",
        "optimizer",
        "checkpoint",
        "accounted_at",
    ):
        if key == "parameters":
            assert_tree_equal(test, before[key], after[key])
        else:
            test.assertEqual(before[key], after[key], key)
    for path, values in before["model"].items():
        assert_tree_equal(test, values, after["model"][path])
    assert_rng_equal(test, before["rng"], after["rng"])


def fixture_checkpoint(
    root: Path, updates=1, *, optimizer="adamw", momentum=0.0, recipe=False, affine=False
):
    """Produce a checkpoint through the existing small fixture's make()."""
    if optimizer == "adamw":
        source = make(root / "source", recipe=recipe, affine=affine)
    else:
        original = __import__("test_qat_curriculum_runner").JointQATConfig

        def configured(*args, **kwargs):
            kwargs["binary_optimization"] = BinaryOptimizationConfig(
                optimizer="sgd", momentum=momentum
            )
            return original(*args, **kwargs)

        with patch("test_qat_curriculum_runner.JointQATConfig", side_effect=configured):
            source = make(root / "source", recipe=recipe, affine=affine)
    source.run(require_smoke=False, max_new_updates=updates)
    pointer = copy.deepcopy(source.last_checkpoint)
    checkpoint = source.run_dir / "checkpoints" / pointer["file"]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return source, payload, pointer


def logical_curriculum_state(value):
    """Drop measured wall time; compare update, exposure, and ancestry exactly."""
    result = copy.deepcopy(value)
    result.pop("overhead_gpu_seconds")
    for phase in result["phases"]:
        phase.pop("gpu_seconds")
    for transition in result["transitions"]:
        transition.pop("source_gpu_seconds")
        transition.pop("source_checkpoint_sha256")
    return result


class AdversarialStagedResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="resume-validator-")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def target(self, name="target", *, optimizer="adamw", momentum=0.0, recipe=False, affine=False):
        if optimizer == "adamw":
            return make(self.root / name, recipe=recipe, affine=affine)
        original = __import__("test_qat_curriculum_runner").JointQATConfig

        def configured(*args, **kwargs):
            kwargs["binary_optimization"] = BinaryOptimizationConfig(
                optimizer="sgd", momentum=momentum
            )
            return original(*args, **kwargs)

        with patch("test_qat_curriculum_runner.JointQATConfig", side_effect=configured):
            return make(self.root / name, recipe=recipe, affine=affine)

    def test_malformed_owned_states_fail_without_live_or_global_mutation(self):
        _, original, pointer = fixture_checkpoint(
            self.root / "checkpoint", recipe=True, affine=True
        )
        first_id = next(iter(original["optimizer"]["state"]))
        cases = {
            "reordered owned IDs": lambda p: p["optimizer"]["param_groups"][0]["params"].reverse(),
            "unowned state ID": lambda p: p["optimizer"]["state"].__setitem__(9999, {}),
            "bad moment shape": lambda p: p["optimizer"]["state"][first_id]["exp_avg"].resize_(1),
            "bad moment dtype": lambda p: p["optimizer"]["state"][first_id].__setitem__(
                "exp_avg", p["optimizer"]["state"][first_id]["exp_avg"].double()
            ),
            "nonfinite moment": lambda p: p["optimizer"]["state"][first_id]["exp_avg"].fill_(
                float("nan")
            ),
            "negative second moment": lambda p: p["optimizer"]["state"][first_id][
                "exp_avg_sq"
            ].fill_(-1),
            "fractional step": lambda p: p["optimizer"]["state"][first_id]["step"].fill_(0.5),
            "negative step": lambda p: p["optimizer"]["state"][first_id]["step"].fill_(-1),
            "non-scalar step": lambda p: p["optimizer"]["state"][first_id].__setitem__(
                "step", torch.zeros(1)
            ),
            "immutable betas": lambda p: p["optimizer"]["param_groups"][0].__setitem__(
                "betas", (0.2, 0.3)
            ),
            "effective LR": lambda p: p["optimizer"]["param_groups"][0].__setitem__(
                "lr", p["optimizer"]["param_groups"][0]["lr"] * 0.5
            ),
        }
        failures = []
        for index, (name, corrupt) in enumerate(cases.items()):
            with self.subTest(case=name):
                runner = self.target(f"target-{index}", recipe=True, affine=True)
                payload = copy.deepcopy(original)
                corrupt(payload)
                before = live_snapshot(runner)
                with self.assertRaises((ValueError, TypeError)):
                    stage_resume(runner, payload, pointer)
                try:
                    assert_live_equal(self, before, runner)
                except AssertionError as error:
                    failures.append(f"{name}: {error}")
        self.assertFalse(failures, "mutation on invalid payload: " + "; ".join(failures))

    def test_valid_midphase_and_boundary_next_updates_match_uninterrupted(self):
        full = make(self.root / "full")
        full.run(require_smoke=False)
        for count in (1, 2):
            with self.subTest(update=count):
                _, payload, pointer = fixture_checkpoint(self.root / f"partial-{count}", count)
                resumed = self.target(f"resumed-{count}")
                before_staging = live_snapshot(resumed)
                staged = stage_resume(resumed, payload, pointer)
                assert_live_equal(self, before_staging, resumed)
                staged.commit()
                resumed.run(require_smoke=False)
                for path, module in full.linears.items():
                    for key, expected in module.state_dict().items():
                        actual = resumed.linears[path].state_dict()[key]
                        if isinstance(expected, torch.Tensor):
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        else:
                            self.assertEqual(actual, expected)
                self.assertEqual(
                    logical_curriculum_state(resumed.state.state_dict()),
                    logical_curriculum_state(full.state.state_dict()),
                )
                self.assertEqual(resumed.cursor, full.cursor)
                self.assertEqual(resumed.epoch, full.epoch)
                self.assertEqual(resumed.unique_rows, full.unique_rows)
                self.assertEqual(resumed.unique_prompts, full.unique_prompts)
                self.assertEqual(
                    resumed.optimizer.state_dict()["param_groups"],
                    full.optimizer.state_dict()["param_groups"],
                )
                for key, values in full.optimizer.state_dict()["state"].items():
                    actual = resumed.optimizer.state_dict()["state"][key]
                    self.assertEqual(set(actual), set(values))
                    for name, value in values.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(actual[name], value, rtol=0, atol=0)
                        else:
                            self.assertEqual(actual[name], value)
                torch.testing.assert_close(resumed.rng["torch"], full.rng["torch"], rtol=0, atol=0)

    def test_missing_moments_are_legal_when_gradient_is_absent(self):
        _, payload, pointer = fixture_checkpoint(
            self.root / "missing-moment", 1, recipe=True, affine=True
        )
        parameter_id = next(iter(payload["optimizer"]["state"]))
        del payload["optimizer"]["state"][parameter_id]
        runner = self.target("missing-moment-target", recipe=True, affine=True)
        runner.optimizer.zero_grad(set_to_none=True)
        stage_resume(runner, payload, pointer).commit()
        self.assertNotIn(parameter_id, runner.optimizer.state_dict()["state"])

    def test_fresh_phase_empty_moments_and_sgd_momentum_variants(self):
        for optimizer, momentum, updates in (
            ("adamw", 0.0, 2),
            ("sgd", 0.0, 1),
            ("sgd", 0.8, 1),
        ):
            with self.subTest(optimizer=optimizer, momentum=momentum):
                _, payload, pointer = fixture_checkpoint(
                    self.root / f"source-{optimizer}-{momentum}",
                    updates,
                    optimizer=optimizer,
                    momentum=momentum,
                )
                if optimizer == "adamw" and updates == 2:
                    self.assertEqual(payload["optimizer"]["state"], {})
                if optimizer == "sgd" and momentum == 0:
                    self.assertEqual(payload["optimizer"]["state"], {})
                if optimizer == "sgd" and momentum > 0:
                    self.assertTrue(payload["optimizer"]["state"])
                    self.assertTrue(
                        all(
                            set(values) == {"momentum_buffer"}
                            for values in payload["optimizer"]["state"].values()
                        )
                    )
                target = self.target(
                    f"target-{optimizer}-{momentum}", optimizer=optimizer, momentum=momentum
                )
                before_staging = live_snapshot(target)
                staged = stage_resume(target, payload, pointer)
                assert_live_equal(self, before_staging, target)
                staged.commit()


if __name__ == "__main__":
    unittest.main(verbosity=2)

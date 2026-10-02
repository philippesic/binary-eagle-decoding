"""Tiny CPU exact-update and storage-lifetime checks; no real models/data."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from staged_resume import check_control, named_layout, resume_staged, stage_resume
from test_qat_curriculum_runner import compare_models, make

from w1a1_eagle.qat_optimization import BinaryOptimizationConfig
from w1a1_eagle.recurrent_qat import JointQATConfig


def exact_tree(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact_tree(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            exact_tree(x, y)
    else:
        assert a == b


def load_fixture(runner):
    pointer = json.loads((runner.run_dir / 'latest.json').read_text())
    payload = torch.load(runner.run_dir / 'checkpoints' / pointer['file'],
                         map_location='cpu', weights_only=False)
    return payload, pointer


class StagingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        check_control()
        torch.set_num_threads(1)

    def test_next_update_matches_unchanged_runner_midphase_boundary_complete(self):
        for recipe, affine, count in ((False, False, 0), (False, False, 1),
                                      (True, True, 1), (True, True, 2),
                                      (True, True, 3), (True, True, 4), (True, True, 6)):
            check_control()
            with self.subTest(recipe=recipe, affine=affine, count=count):
                with tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    source = make(root, recipe=recipe, affine=affine)
                    if count:
                        source.run(require_smoke=False, max_new_updates=count)
                    else:
                        source.save()
                    original = make(root, recipe=recipe, affine=affine)
                    original.resume()
                    staged = make(root, recipe=recipe, affine=affine)
                    resume_staged(staged)
                    compare_models(original, staged)
                    exact_tree(original.optimizer.state_dict(), staged.optimizer.state_dict())
                    exact_tree(original.state.state_dict(), staged.state.state_dict())
                    exact_tree(original.rng, staged.rng)
                    original.run(require_smoke=False, max_new_updates=1)
                    staged.run(require_smoke=False, max_new_updates=1)
                    compare_models(original, staged)
                    exact_tree(original.optimizer.state_dict(), staged.optimizer.state_dict())
                    exact_tree(original.rng, staged.rng)
                    self.assertEqual(original.state.global_updates, staged.state.global_updates)

    def test_candidate_storage_is_checkpoint_backed_and_frozen_shared(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = make(root, recipe=True, affine=True)
            source.run(require_smoke=False, max_new_updates=1)
            live = make(root, recipe=True, affine=True)
            flags = [(id(p), p.requires_grad) for p in live.drafter.parameters()]
            payload, pointer = load_fixture(live)
            staged = stage_resume(live, payload, pointer)
            self.assertEqual(flags, [(id(p), p.requires_grad) for p in live.drafter.parameters()])
            for path, module in staged.candidate.linears.items():
                self.assertEqual(module.latent_sign.untyped_storage().data_ptr(),
                                 payload['model']['linears'][path]['latent_sign']
                                 .untyped_storage().data_ptr())
                self.assertNotEqual(module.latent_sign.untyped_storage().data_ptr(),
                                    live.linears[path].latent_sign.untyped_storage().data_ptr())
            self.assertIs(staged.candidate.drafter.embed_tokens.weight,
                          live.drafter.embed_tokens.weight)
            for actual, values in zip(staged.candidate.optimizer.state.values(),
                                      payload['optimizer']['state'].values()):
                for key in ('exp_avg', 'exp_avg_sq', 'step'):
                    self.assertEqual(actual[key].untyped_storage().data_ptr(),
                                     values[key].untyped_storage().data_ptr())
            self.assertIsNot(staged.candidate.drafter.qat_affine_bank,
                             live.drafter.qat_affine_bank)
            for path, row in staged.candidate.drafter.qat_affine_bank.midpoints.items():
                self.assertIs(row, staged.candidate.linears[path].affine_binary)

    def test_optional_participation_detects_silent_omission(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = make(root, recipe=True, affine=True)
            source.run(require_smoke=False, max_new_updates=1)
            live = make(root, recipe=True, affine=True)
            payload, pointer = load_fixture(live)
            payload['participation'] = {x['name']: 1 for x in named_layout(
                source.linears, source.optimizer)}
            stage_resume(live, payload, pointer)
            omitted = copy.deepcopy(payload)
            del omitted['optimizer']['state'][0]
            with self.assertRaisesRegex(ValueError, 'participation'):
                stage_resume(live, omitted, pointer)
            del omitted['participation']
            stage_resume(live, omitted, pointer)  # grad=None is otherwise admissible.

    def test_actual_none_gradient_omits_state_and_receipt_preserves_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = make(root, recipe=True, affine=True)
            for group, rate in zip(source.optimizer.param_groups, source.base_lrs):
                group['lr'] = rate * (1 / source.config.warmup_updates)
            first = source.optimizer.param_groups[0]['params'][0]
            first.grad = torch.zeros_like(first)
            source.optimizer.step()  # Only one tiny parameter participates; no model change.
            source.optimizer.zero_grad(set_to_none=True)
            source.state.record_update(activation_bits=8, gpu_seconds=0.001,
                                       rows=0, tokens=0, supported_rows=0)
            self.assertEqual(len(source.optimizer.state), 1)
            source.save()
            live = make(root, recipe=True, affine=True)
            payload, pointer = load_fixture(live)
            layout = named_layout(source.linears, source.optimizer)
            payload['participation'] = {x['name']: int(i == 0) for i, x in enumerate(layout)}
            stage_resume(live, payload, pointer).commit()
            exact_tree(source.optimizer.state_dict(), live.optimizer.state_dict())

    def test_binary_recipe_adamw_and_sgd_next_update_matches(self):
        for algorithm, momentum in (('adamw', 0.0), ('sgd', 0.0), ('sgd', 0.8)):
            check_control()

            def configured(*args, **kwargs):
                kwargs['binary_optimization'] = BinaryOptimizationConfig(
                    optimizer=algorithm, momentum=momentum)
                return JointQATConfig(*args, **kwargs)

            with self.subTest(algorithm=algorithm, momentum=momentum):
                with tempfile.TemporaryDirectory() as folder, patch(
                        'test_qat_curriculum_runner.JointQATConfig', side_effect=configured):
                    root = Path(folder)
                    source = make(root, recipe=True, affine=True)
                    source.run(require_smoke=False, max_new_updates=1)
                    original = make(root, recipe=True, affine=True)
                    original.resume()
                    staged = make(root, recipe=True, affine=True)
                    resume_staged(staged)
                    original.run(require_smoke=False, max_new_updates=1)
                    staged.run(require_smoke=False, max_new_updates=1)
                    compare_models(original, staged)
                    exact_tree(original.optimizer.state_dict(), staged.optimizer.state_dict())
                    exact_tree(original.rng, staged.rng)


if __name__ == '__main__':
    unittest.main()

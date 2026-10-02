"""Tiny actual-API CPU curriculum oracle; no live recipe or core changes.

Run with PYTHONPATH=src:tests .venv/bin/python research/.../audit.py.
Synthetic checkpoints live only in a TemporaryDirectory. Printed JSON is a
small research summary, not a model/capture artifact.
"""

from __future__ import annotations

import json
import platform
import tempfile
from pathlib import Path

import torch
import numpy as np

from test_qat_curriculum_runner import compare_models, make
from w1a1_eagle.learned_activation import learned_activation, learned_activation_reference
from w1a1_eagle.recurrent_qat import joint_parameter_families

CONTROL = Path('/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json')


def check_control():
    if CONTROL.is_file():
        control = json.loads(CONTROL.read_text())
        if control.get('research_stop') or control.get('monitor_interrupt'):
            raise SystemExit('research_stop/monitor_interrupt: checkpoint without further experiments')


def scalar_values(runner):
    return {name: float(q.parameter.detach()) for name, q in runner.bank.quantizers.items()}


def exact_tree(left, right):
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype
        assert np.array_equal(left, right)
    elif isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            exact_tree(a, b)
    else:
        assert left == right


def transition_probe(*, sentinels=False):
    check_control()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runner = make(root, recipe=True, affine=True)
        model = runner.drafter
        results = []
        for destination in (4, 1):
            check_control()
            runner.stop_requested = False
            runner.run(require_smoke=False, max_new_updates=1)
            learned_from_step = scalar_values(runner)
            if sentinels:
                # Distinct valid synthetic states make resets material and
                # visible; these are diagnostics, not learned real-model values.
                with torch.no_grad():
                    for i, quantizer in enumerate(runner.bank.quantizers.values()):
                        quantizer.parameter.fill_(0.55 + i * 0.05)
                runner.bank.project_()
            before = scalar_values(runner)
            families = joint_parameter_families(runner.linears)
            retained = {key: [(p, p.detach().clone()) for p in families[key]]
                        for key in ('sign', 'scale', 'fusion', 'midpoint')}
            old_activation = list(families['activation'])
            old_optimizer = runner.optimizer
            old_core = {p: {k: v.clone() for k, v in m.items()}
                        for p, m in runner._core_states().items()}
            old_steps = sorted({int(s['step']) for s in old_optimizer.state.values()})
            assert len(old_optimizer.state) == 36 and old_steps == [1]
            source = runner.qat.contract.activation_bits
            runner.state.finish_phase()
            runner._transition()
            assert runner.drafter is model and runner.optimizer is not old_optimizer
            assert runner.optimizer.state == {}
            exact_tree(old_core, runner._core_states())
            current = joint_parameter_families(runner.linears)
            for key, entries in retained.items():
                assert len(entries) == len(current[key])
                for (p, value), actual in zip(entries, current[key]):
                    assert p is actual and torch.equal(value, actual)
            assert not {id(p) for p in old_activation} & {id(p) for p in current['activation']}
            after = scalar_values(runner)
            assert set(after.values()) == ({0.0} if destination == 1 else {1.0})
            runner.bank.validate_attachment(runner.linears)
            ownership = [p for g in runner.optimizer.param_groups for p in g['params']]
            assert len(ownership) == 36 and len({id(p) for p in ownership}) == 36
            resumed = make(root, recipe=True, affine=True)
            resumed.resume()
            compare_models(runner, resumed)
            exact_tree(runner.optimizer.state_dict(), resumed.optimizer.state_dict())
            exact_tree(runner.rng, resumed.rng)
            assert resumed.model_phase == runner.model_phase
            x = torch.tensor([[-1., -.61, -.2, .0, .23, .59, .91, 1.]])
            isolated_reset = None
            if destination == 4:
                # Same A4 arithmetic/input, only scalar differs. This is an
                # isolated reset counterfactual, not an authorized transfer rule.
                prior = learned_activation_reference(x, 4, torch.tensor(before['head']))
                reset = learned_activation_reference(x, 4, torch.tensor(after['head']))
                isolated_reset = {
                    'max_abs_output_difference': float((prior.values-reset.values).abs().max()),
                    'changed_integer_codes': int((prior.codes != reset.codes).sum()),
                    'prior_clip': before['head'], 'reset_clip': after['head'],
                }
            results.append({
                'source_bits': source, 'destination_bits': destination,
                'after_actual_tiny_step': learned_from_step,
                'before_transition': before, 'after_transition': after,
                'old_optimizer_tensors': len(old_optimizer.state),
                'old_adam_steps': old_steps, 'new_optimizer_tensors': len(runner.optimizer.state),
                'retained_exact': {k: len(v) for k, v in retained.items()},
                'retained_core_buffers_exact': True,
                'new_activation_tensors': len(current['activation']),
                'canonical_attachment_passed': True, 'post_transition_resume_exact': True,
                'isolated_same_a4_reset': isolated_reset,
            })
        return results


def gradient_probe():
    """Distinguish hard finite differences from the declared surrogate VJP.

    LSQ uses q-u inside, q outside, and A1 uses a threshold surrogate. These
    are intentionally not exact derivatives of discontinuous hard quantization.
    A local frozen-q surrogate independently checks the implemented VJP by FD.
    """
    check_control()
    results = []
    x = torch.tensor([[-1.0, -.53, -.13, .0, .19, .47, .83]], dtype=torch.float32)
    upstream = torch.tensor([[.2, -.3, .7, -.5, 1.1, .4, -.8]])
    eps = 1e-4
    for bits, value in ((8, .71), (4, .71), (1, .23)):
        p = torch.tensor(value, requires_grad=True)
        hard = learned_activation(x, bits, p)
        (hard.values * upstream).sum().backward()
        analytic = float(p.grad)
        ref = learned_activation_reference(x, bits, p.detach())
        if bits == 1:
            shifted = x.double() - value * ref.scale.double()
            support = shifted.abs() <= ref.scale.double()
            normalizer = x.numel() ** .5
            derivative = -ref.scale.double() * support / normalizer
            def proxy(c):
                return ref.values.double() + (c-value)*derivative
        else:
            qmax = (1 << (bits-1))-1
            maximum = x.abs().amax(-1, keepdim=True).double()
            support = x.abs().double() <= maximum*value
            normalizer = (x.numel()*qmax) ** .5
            # Frozen hard codes; inside includes LSQ's -x log(c) term.
            def proxy(c):
                base = ref.codes.double()*maximum*c/qmax
                inside = base - x.double()*torch.log(torch.tensor(c, dtype=torch.float64))
                return torch.where(support, inside, base)/normalizer
        fd = float(((proxy(value+eps)-proxy(value-eps))*upstream.double()).sum()/(2*eps))
        plus = learned_activation_reference(x, bits, torch.tensor(value+eps)).values
        minus = learned_activation_reference(x, bits, torch.tensor(value-eps)).values
        hard_fd = float(((plus-minus)*upstream).sum()/(2*eps))
        assert abs(fd-analytic) < 2e-6
        results.append({'bits': bits, 'parameter': value, 'autograd_surrogate_vjp': analytic,
                        'frozen_code_surrogate_finite_difference': fd,
                        'absolute_error': abs(fd-analytic), 'hard_forward_finite_difference': hard_fd,
                        'hard_fd_is_not_the_ste_contract': True})
    return results


def audit():
    check_control()
    torch.set_num_threads(1)
    return {'schema': 'parallel_curriculum_transition_audit_v1',
            'hardware': platform.machine(), 'device': 'cpu', 'torch': torch.__version__,
            'floating_parameters': 'F32, tiny frozen provider operands include FP16',
            'ordinary_tiny_training': transition_probe(),
            'material_synthetic_reset_probe': transition_probe(sentinels=True),
            'surrogate_finite_difference': gradient_probe(),
            'no_acceptance_latency_or_gpu_claim': True}


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, sort_keys=True, allow_nan=False))

"""Independent CPU acceptance checks for the bounded transition audit."""

import importlib.util
import unittest
from pathlib import Path


def audit_module():
    source = Path(__file__).resolve().parents[1] / 'research/parallel20261002/curriculum_transition/audit.py'
    spec = importlib.util.spec_from_file_location('parallel_curriculum_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CurriculumTransitionAuditTests(unittest.TestCase):
    def test_actual_phase_switch_resets_activation_but_retains_binary_state(self):
        results = audit_module().transition_probe(sentinels=True)
        assert [(r['source_bits'], r['destination_bits']) for r in results] == [(8, 4), (4, 1)]
        assert all(r['post_transition_resume_exact'] and r['new_optimizer_tensors'] == 0 for r in results)
        assert results[0]['isolated_same_a4_reset']['max_abs_output_difference'] > .1


    def test_declared_surrogate_vjp_has_independent_finite_difference_oracle(self):
        results = audit_module().gradient_probe()
        assert all(r['absolute_error'] < 2e-6 for r in results)
        assert results[-1]['hard_forward_finite_difference'] == 0
        assert results[-1]['autograd_surrogate_vjp'] != 0

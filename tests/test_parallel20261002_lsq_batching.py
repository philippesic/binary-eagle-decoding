"""Protected QAT CPU regressions for learned-head reference equivalence.

The historical discrepancy and candidate source pins remain in the LSQ report,
raw records, and proposal files. These tests exercise the current provider.
"""

import hashlib
import math
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from research.parallel20261002.lsq_batching.reference.audit import (
    ATOL,
    RTOL,
    compare,
    run_audit,
    run_case,
    run_ragged,
)


class ProtectedQATValidation(unittest.TestCase):
    def setUp(self):
        # These synthetic correctness checks are QAT work; they remain runnable
        # after supporting research stops and without its machine-local record.
        control = patch("research.parallel20261002.lsq_batching.reference.audit.check_control")
        control.start()
        self.addCleanup(control.stop)


class LSQBatchingAuditTests(ProtectedQATValidation):
    @classmethod
    def setUpClass(cls):
        with patch("research.parallel20261002.lsq_batching.reference.audit.check_control"):
            cls.report = run_audit()
            cls.results = cls.report["cases"]

    def test_report_pins_the_imported_provider_source(self):
        import w1a1_eagle.recurrent_provider as provider

        source = Path(provider.__file__).resolve()
        self.assertEqual(self.report["source_paths"]["recurrent_provider.py"], str(source))
        self.assertEqual(
            self.report["source_sha256"]["recurrent_provider.py"],
            hashlib.sha256(source.read_bytes()).hexdigest(),
        )

    def test_actual_native_head_batch_preserves_logits_loss_and_other_vjps(self):
        for entry in self.results:
            for policy in (
                "batched",
                "chunk2",
                "serial_fallback",
                "chain_domain_batched",
                "chain_domain_chunk2",
            ):
                with self.subTest(entry=entry, policy=policy):
                    values = entry[policy]
                    self.assertEqual(values["logits_max_abs"], 0.0)
                    self.assertEqual(values["loss_abs"], 0.0)
                    for key in (
                        "input_vjp_max_abs",
                        "sign_vjp_max_abs",
                        "scale_vjp_max_abs",
                        "other_activation_vjp_max_abs",
                        "other_parameter_step_max_abs",
                    ):
                        self.assertLessEqual(values[key], ATOL)

    def test_trainable_head_preserves_serial_parameter_vjp_and_update(self):
        for entry in self.results:
            for policy in ("batched", "chunk2"):
                with self.subTest(entry=entry, policy=policy):
                    self.assertIsNotNone(entry[policy]["head_gradient_ratio"])
                    self.assertAlmostEqual(entry[policy]["head_gradient_ratio"], 1.0, delta=1e-5)
                    self.assertLessEqual(entry[policy]["head_parameter_step_abs"], ATOL)

    def test_repeated_row_mean_loss_holds_serial_vjp_constant(self):
        for bits in (1, 4, 8):
            values = [
                e["batched"]["head_gradient_reference"]
                for e in self.results
                if e["fixture"] == "repeat" and e["bits"] == bits
            ]
            self.assertGreater(abs(values[0]), 1e-4)
            for value in values:
                self.assertAlmostEqual(value, values[0], delta=ATOL)

    def test_terminal_padding_does_not_enter_normalization_domain(self):
        for e in self.results:
            if e["invalid_terminal"]:
                continue
            padded = next(
                p
                for p in self.results
                if all(p[k] == e[k] for k in ("bits", "depth", "fixture", "computation"))
                and p["invalid_terminal"]
            )
            self.assertEqual(padded["batched"], e["batched"])

    def test_both_declared_remedies_restore_parameter_update(self):
        for entry in self.results:
            for policy in ("serial_fallback", "chain_domain_batched", "chain_domain_chunk2"):
                self.assertLessEqual(entry[policy]["head_parameter_step_abs"], ATOL)
                self.assertAlmostEqual(
                    entry[policy]["head_gradient_reference"],
                    entry[policy]["head_gradient_candidate"],
                    delta=ATOL,
                )

    def test_ragged_valid_chain_lengths_use_each_chains_domain(self):
        torch.set_num_threads(1)
        for bits in (1, 4, 8):
            serial, batched = [run_ragged(bits, p) for p in ("serial", "batched")]
            delta = compare(serial, batched)
            self.assertEqual(delta["logits_max_abs"], 0.0)
            self.assertEqual(delta["loss_abs"], 0.0)
            self.assertLessEqual(delta["input_vjp_max_abs"], ATOL)
            key = next(k for k in serial["grads"] if k.startswith("lm_head.activation_quantizer."))
            expected_serial, expected_batched = torch.zeros(()), torch.zeros(())
            for depth in (1, 2, 4):
                individual = run_case(bits, depth, invalid=True)
                weighted = individual["grads"][key] * depth / 7
                expected_serial += weighted
                expected_batched += weighted
            torch.testing.assert_close(serial["grads"][key], expected_serial, rtol=RTOL, atol=ATOL)
            torch.testing.assert_close(
                batched["grads"][key], expected_batched, rtol=RTOL, atol=ATOL
            )
            # One accidental minibatch-wide denominator has a different contract.
            self.assertFalse(
                torch.isclose(
                    batched["grads"][key], expected_serial / math.sqrt(7), rtol=RTOL, atol=ATOL
                )
            )


class LiveProviderFallbackTests(ProtectedQATValidation):
    def test_current_provider_preserves_serial_vjps_and_update(self):
        from research.parallel20261002.lsq_batching.reference.audit import batch, tiny_native
        from w1a1_eagle.recurrent_provider import forward_torch_round
        from w1a1_eagle.recurrent_qat import joint_optimizer, shared_round_hard_signs

        for bits in (1, 4, 8):
            for depth in (1, 2, 4):
                reference = run_case(bits, depth, True)
                adapter, cfg = tiny_native(bits)
                optimizer = joint_optimizer(adapter.linears, cfg)
                b = batch(depth, invalid=True)
                with shared_round_hard_signs(adapter.linears):
                    logits = forward_torch_round(b, adapter, 3, optimize_head=True)
                    loss = torch.nn.functional.cross_entropy(
                        logits[:depth], torch.full((depth,), 2, dtype=torch.long)
                    )
                    loss.backward()
                torch.testing.assert_close(logits, reference["logits"], rtol=0, atol=0)
                torch.testing.assert_close(
                    b.raw_target_features.grad, reference["raw_vjp"], rtol=RTOL, atol=ATOL
                )
                for name, parameter in adapter.drafter.named_parameters():
                    if parameter.requires_grad:
                        torch.testing.assert_close(
                            parameter.grad, reference["grads"][name], rtol=RTOL, atol=ATOL
                        )
                optimizer.step()
                for name, parameter in adapter.drafter.named_parameters():
                    if parameter.requires_grad:
                        torch.testing.assert_close(
                            parameter, reference["params"][name], rtol=RTOL, atol=ATOL
                        )
                self.assertEqual(adapter.head_saturation_scope, "last_valid_row")

    def test_current_provider_retains_batched_inference_and_fixed_quantizer_training(self):
        from unittest.mock import patch

        from research.parallel20261002.lsq_batching.reference.audit import batch, tiny_native
        from w1a1_eagle.recurrent_provider import forward_torch_round

        adapter, _ = tiny_native(4)
        with (
            patch.object(adapter, "decode_head", wraps=adapter.decode_head) as head,
            torch.no_grad(),
        ):
            forward_torch_round(batch(4, invalid=True), adapter, 3, optimize_head=True)
            self.assertEqual(head.call_count, 1)
        for module in adapter.linears.values():
            del module.activation_quantizer
        with patch.object(adapter, "decode_head", wraps=adapter.decode_head) as head:
            forward_torch_round(batch(4, invalid=True), adapter, 3, optimize_head=True)
            self.assertEqual(head.call_count, 1)


class ObservedFallbackTests(ProtectedQATValidation):
    def test_observer_delegates_trainable_head_and_keeps_first_attached_state(self):
        from unittest.mock import patch

        from research.parallel20261002.lsq_batching.reference.audit import batch, tiny_native
        from w1a1_eagle.continuous_qat import ObservedAdapter
        from w1a1_eagle.recurrent_provider import forward_torch_round

        adapter, _ = tiny_native(4)
        observed = ObservedAdapter(adapter)
        self.assertIs(observed.linears, adapter.linears)
        with patch.object(adapter, "decode_head", wraps=adapter.decode_head) as head:
            logits = forward_torch_round(
                batch(4, invalid=True), observed, 3, optimize_head=True
            )
            self.assertEqual(head.call_count, 0)
        self.assertIsNotNone(observed.first)
        later = torch.nn.functional.cross_entropy(logits[3:4], torch.tensor([2]))
        vjp = torch.autograd.grad(later, observed.first.pre_norm)[0]
        self.assertGreater(float(vjp.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

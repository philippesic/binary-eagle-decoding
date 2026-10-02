"""CPU synthetic invariance and distinguishing controls for the LSQ batching audit."""

import math
import unittest

import torch

from research.parallel20261002.lsq_batching.reference.audit import (
    ATOL,
    RTOL,
    compare,
    run_audit,
    run_case,
    run_ragged,
)


class LSQBatchingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = run_audit()["cases"]

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

    def test_nonzero_parameter_vjp_exposes_invocation_normalizer(self):
        for entry in self.results:
            with self.subTest(entry=entry):
                self.assertIsNotNone(entry["batched"]["head_gradient_ratio"])
                self.assertAlmostEqual(
                    entry["batched"]["head_gradient_ratio"], math.sqrt(entry["depth"]), delta=1e-5
                )
                self.assertAlmostEqual(
                    entry["chunk2"]["head_gradient_ratio"],
                    math.sqrt(min(entry["depth"], 2)),
                    delta=1e-5,
                )
                if entry["depth"] > 1:
                    self.assertGreater(entry["batched"]["head_parameter_step_abs"], 1e-5)

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
                expected_batched += weighted / math.sqrt(depth)
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


class IsolatedProviderProposalTests(unittest.TestCase):
    def test_source_bound_patch_preserves_serial_vjps_and_update(self):
        from research.parallel20261002.lsq_batching.reference.audit import batch, tiny_native
        from research.parallel20261002.lsq_batching.reference.proposal import load_proposed_forward
        from w1a1_eagle.recurrent_qat import joint_optimizer, shared_round_hard_signs

        proposed = load_proposed_forward()
        for bits in (1, 4, 8):
            for depth in (1, 2, 4):
                reference = run_case(bits, depth, True)
                adapter, cfg = tiny_native(bits)
                optimizer = joint_optimizer(adapter.linears, cfg)
                b = batch(depth, invalid=True)
                with shared_round_hard_signs(adapter.linears):
                    logits = proposed(b, adapter, 3, optimize_head=True)
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

    def test_proposal_retains_batched_inference_and_fixed_quantizer_training(self):
        from unittest.mock import patch

        from research.parallel20261002.lsq_batching.reference.audit import batch, tiny_native
        from research.parallel20261002.lsq_batching.reference.proposal import load_proposed_forward

        proposed = load_proposed_forward()
        adapter, _ = tiny_native(4)
        with (
            patch.object(adapter, "decode_head", wraps=adapter.decode_head) as head,
            torch.no_grad(),
        ):
            proposed(batch(4, invalid=True), adapter, 3, optimize_head=True)
            self.assertEqual(head.call_count, 1)
        for module in adapter.linears.values():
            del module.activation_quantizer
        with patch.object(adapter, "decode_head", wraps=adapter.decode_head) as head:
            proposed(batch(4, invalid=True), adapter, 3, optimize_head=True)
            self.assertEqual(head.call_count, 1)


class ObservedProposalTests(unittest.TestCase):
    def test_observer_delegates_trainable_head_and_keeps_first_attached_state(self):
        from unittest.mock import patch

        from research.parallel20261002.lsq_batching.reference.audit import batch, tiny_native
        from research.parallel20261002.lsq_batching.reference.proposal import load_proposed_forward
        from w1a1_eagle.continuous_qat import ObservedAdapter

        adapter, _ = tiny_native(4)
        observed = ObservedAdapter(adapter)
        self.assertIs(observed.linears, adapter.linears)
        with patch.object(adapter, "decode_head", wraps=adapter.decode_head) as head:
            logits = load_proposed_forward()(
                batch(4, invalid=True), observed, 3, optimize_head=True
            )
            self.assertEqual(head.call_count, 0)
        self.assertIsNotNone(observed.first)
        later = torch.nn.functional.cross_entropy(logits[3:4], torch.tensor([2]))
        vjp = torch.autograd.grad(later, observed.first.pre_norm)[0]
        self.assertGreater(float(vjp.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

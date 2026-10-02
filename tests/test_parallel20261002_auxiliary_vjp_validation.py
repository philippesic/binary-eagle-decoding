"""Independent handwritten combined auxiliary-VJP checks (CPU only)."""

import unittest

import torch

from w1a1_eagle.affine_binary import affine_binary_projection, shared_affine_input_sums
from w1a1_eagle.fusion_correction import FusionCorrection, FusionCorrectionConfig
from w1a1_eagle.recurrent_qat import hard_activation_with_codes


def close(actual, expected, *, name):
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7, msg=name)


class CombinedAuxiliaryVJPTests(unittest.TestCase):
    def test_two_affine_consumers_plus_fusion_hand_vjp_all_activation_widths(self):
        """Shared Q sum, affine midpoint, F16 factor STE, and raw path all add."""
        source = torch.tensor(
            [[0.73, -0.41, 0.19, 0.88], [-0.62, 0.27, 0.51, -0.34]],
            dtype=torch.float32,
            requires_grad=True,
        )
        loss_weights = (
            torch.tensor([[0.2, -0.7, 0.4], [-0.1, 0.8, 0.3]]),
            torch.tensor([[-0.5, 0.6], [0.9, -0.2]]),
        )
        for bits in (1, 4, 8):
            with self.subTest(bits=bits):
                x = source.detach().clone().requires_grad_()
                q, beta, _, codes = hard_activation_with_codes(x, bits)
                c, b = codes, beta

                # Two projections share the exact quantizer result and sum node.
                signs_a = torch.tensor([[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 1.0],
                                        [1.0, 1.0, -1.0, -1.0]])
                alpha_a = torch.tensor([0.31, 0.22, 0.47], requires_grad=True)
                midpoint_a = torch.tensor([0.13, -0.21, 0.09], requires_grad=True)
                bias_a = torch.tensor([0.04, -0.03, 0.12], requires_grad=True)
                signs_b = torch.tensor([[-1.0, -1.0, 1.0, 1.0], [1.0, -1.0, -1.0, 1.0]])
                alpha_b = torch.tensor([0.26, 0.39], requires_grad=True)
                midpoint_b = torch.tensor([-0.17, 0.11], requires_grad=True)
                bias_b = torch.tensor([-0.02, 0.07], requires_grad=True)

                correction = FusionCorrection(4, 3, FusionCorrectionConfig(True, 1, True, seed=17))
                with torch.no_grad():
                    correction.v.copy_(torch.tensor([[0.27123, -0.19317, 0.34721, 0.12673]]))
                    correction.u.copy_(torch.tensor([[0.18327], [-0.29149], [0.22711]]))
                    correction.output_bias.copy_(torch.tensor([0.03, -0.06, 0.02]))
                vh = correction.v.detach().half().float()
                uh = correction.u.detach().half().float()
                bias_c = correction.effective_bias()

                with shared_affine_input_sums():
                    first = affine_binary_projection(
                        q, signs_a, alpha_a, midpoint_a, codes=c, beta=b, bias=bias_a
                    )
                    second = affine_binary_projection(
                        q, signs_b, alpha_b, midpoint_b, codes=c, beta=b, bias=bias_b
                    )
                    output = correction.add_to(x, first)
                    objective = (output * loss_weights[0]).sum() + (second * loss_weights[1]).sum()
                objective.backward()

                # Independent VJP of the declared affine surrogate. For each
                # consumer: dQ = G S alpha + sum_o(G_o mu_o), with shared Q
                # accumulating both terms before the quantizer identity STE.
                qsum = q.detach().sum(-1, keepdim=True)
                da = loss_weights[0]
                db = loss_weights[1]
                grad_q_a = (da * alpha_a.detach()) @ signs_a + da @ midpoint_a.detach()[:, None]
                grad_q_b = (db * alpha_b.detach()) @ signs_b + db @ midpoint_b.detach()[:, None]
                expected_q = grad_q_a + grad_q_b
                expected_x = expected_q.clone()

                # F16 hard factor values in the forward; identity derivative
                # to each F32 master is the factor's ordinary matrix VJP.
                z = x.detach() @ vh.T
                grad_u = da.T @ z
                grad_z = da @ uh
                grad_v = grad_z.T @ x.detach()
                grad_x_correction = grad_z @ vh
                expected_x += grad_x_correction
                close(x.grad, expected_x, name="combined raw/input-sum accumulation")
                close(correction.u.grad, grad_u, name="U gradient through F16 STE")
                close(correction.v.grad, grad_v, name="V gradient through F16 STE")
                close(correction.output_bias.grad, da.sum(0), name="correction bias gradient")
                close(alpha_a.grad, (da * (q.detach() @ signs_a.T)).sum(0), name="alpha A")
                close(alpha_b.grad, (db * (q.detach() @ signs_b.T)).sum(0), name="alpha B")
                close(midpoint_a.grad, (da * qsum).sum(0), name="midpoint A")
                close(midpoint_b.grad, (db * qsum).sum(0), name="midpoint B")
                close(bias_a.grad, da.sum(0), name="affine bias A")
                close(bias_b.grad, db.sum(0), name="affine bias B")

                # The half values differ from F32 masters so this exercises
                # the specified straight-through rule rather than an exact cast.
                self.assertTrue(torch.any(correction.u.detach() != uh))
                self.assertTrue(torch.any(correction.v.detach() != vh))
                self.assertGreater(float(grad_x_correction.abs().sum()), 0.0)

    def test_detached_correction_is_a_failing_negative_control(self):
        correction = FusionCorrection(3, 2, FusionCorrectionConfig(True, 1, seed=4))
        with torch.no_grad():
            correction.u.copy_(torch.tensor([[0.2], [-0.3]]))
            correction.v.copy_(torch.tensor([[0.4, -0.1, 0.3]]))
        raw = torch.tensor([[0.6, -0.2, 0.7]], requires_grad=True)
        prediction = correction(raw)
        prediction.sum().backward()
        good = (raw.grad.clone(), correction.u.grad.clone(), correction.v.grad.clone())
        self.assertTrue(all(float(g.abs().sum()) > 0 for g in good))

        correction.zero_grad(set_to_none=True)
        bad_raw = raw.detach().clone().requires_grad_()
        bad = torch.nn.functional.linear(
            torch.nn.functional.linear(bad_raw, correction.v.detach().half().float()),
            correction.u.detach().half().float(),
        )
        bad.sum().backward()
        self.assertGreater(float(bad_raw.grad.abs().sum()), 0.0)
        self.assertIsNone(correction.u.grad)
        self.assertIsNone(correction.v.grad)

    def test_shared_boundary_input_sum_cache_is_identity_safe(self):
        x = torch.tensor([[0.5, -0.25, 0.75]], requires_grad=True)
        values, beta, _, codes = hard_activation_with_codes(x, 4)
        signs = torch.tensor([[1.0, -1.0, 1.0]])
        alpha = torch.tensor([0.4])
        midpoint = torch.tensor([0.2])
        with shared_affine_input_sums():
            # Identical operands within the scope must reuse a single sum VJP.
            from w1a1_eagle.affine_binary import affine_input_sum

            first_sum = affine_input_sum(values, codes=codes, beta=beta)
            second_sum = affine_input_sum(values, codes=codes, beta=beta)
            self.assertIs(first_sum, second_sum)
            y1 = affine_binary_projection(
                values, signs, alpha, midpoint, codes=codes, beta=beta, input_sum=first_sum
            )
            y2 = affine_binary_projection(
                values, signs, alpha, midpoint, codes=codes, beta=beta, input_sum=second_sum
            )
            (2 * y1.sum() - 0.5 * y2.sum()).backward()
        # The same attached sum contributes with the net coefficient 1.5.
        expected = (1.5 * (0.4 * signs[0] + 0.2)).expand_as(x)
        close(x.grad, expected, name="shared sum consumer accumulation")


if __name__ == "__main__":
    unittest.main()

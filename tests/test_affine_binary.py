"""CPU-only affine binary algebra, attached gradients and strict ownership."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from w1a1_eagle.affine_binary import (
    ARITHMETIC, AffineBinaryBank, AffineBinaryConfig, AffineBinaryMidpoint,
    affine_binary_projection, affine_input_sum, install_affine_binary,
    shared_affine_input_sums, validate_affine_optimizer,
)
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract, hard_activation, hard_sign_ste


def fixture(config=AffineBinaryConfig(enabled=True)):
    linears = {path: RowBinaryLinear(torch.tensor([[.5, -.5, .5], [-.5, .5, .5]]),
                                    torch.tensor([.2, .3]), W1AxContract(1))
               for path in CANDIDATE_D_BASE_TO_PATH.values()}
    target = nn.Linear(3, 2)
    return linears, target, install_affine_binary(linears, target=target, config=config)


def boundary(x, bits):
    values, beta, _ = hard_activation(x, bits)
    if bits == 16:
        return values, None, None
    if bits == 1:
        raw = x.float().contiguous().view(torch.int32)
        negative = ((raw & -2147483648) != 0) & ((raw & 2147483647) != 0)
        codes = torch.where(negative, -1.0, 1.0)
    else:
        qmax = 2 ** (bits - 1) - 1
        codes = torch.where(beta > 0, torch.round(x.detach() / beta), torch.zeros_like(x)).clamp(-qmax, qmax)
    return values, codes, beta


class AffineBinaryTests(unittest.TestCase):
    def test_config_validation(self):
        for kwargs in ({"enabled": 1}, {"coverage": "head"}, {"midpoint_lr": float("nan")},
                       {"midpoint_lr": 0}, {"midpoint_lr": True}, {"mild_l2": -1},
                       {"mild_l2": float("inf")}, {"midpoint_bound": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AffineBinaryConfig(**kwargs)

    def test_disabled_inventory_and_fusion_all(self):
        linears, target, off = fixture(AffineBinaryConfig())
        self.assertEqual(off.declared_paths, ())
        self.assertEqual(list(off.parameters()), [])
        self.assertEqual(off.native_payload(), (None, {}))
        self.assertFalse(any(hasattr(m, "affine_binary") for m in linears.values()))
        self.assertEqual(fixture()[2].declared_paths, ("fc",))
        linears, _, bank = fixture(AffineBinaryConfig(enabled=True, coverage="all"))
        self.assertEqual(set(bank.declared_paths), set(linears))
        self.assertEqual(len(list(bank.parameters())), 9)
        for path, row in bank.midpoints.items():
            self.assertIs(row, linears[path].affine_binary)
            self.assertEqual(row.midpoint.dtype, torch.float32)
            self.assertEqual(torch.count_nonzero(row.midpoint).item(), 0)

    def test_install_atomic_target_alias_reinstall_and_inventory(self):
        linears, target, _ = fixture(AffineBinaryConfig())
        with self.assertRaisesRegex(ValueError, "nine"):
            install_affine_binary({"fc": linears["fc"]}, target=target)
        target.register_buffer("alias", linears["fc"].latent_sign.detach().view(-1))
        with self.assertRaisesRegex(ValueError, "aliases target"):
            install_affine_binary(linears, target=target, config=AffineBinaryConfig(enabled=True))
        self.assertFalse(any(hasattr(m, "affine_binary") for m in linears.values()))
        linears, target, bank = fixture()
        with self.assertRaisesRegex(ValueError, "already"):
            install_affine_binary(linears, target=target, config=bank.config)

    def test_default_midpoint_projection_and_explicit_bound(self):
        row = AffineBinaryMidpoint(2, AffineBinaryConfig(enabled=True, mild_l2=.25))
        with torch.no_grad():
            row.midpoint.copy_(torch.tensor([-100., 100.]))
        row.project_()
        self.assertEqual(row.midpoint.tolist(), [-100., 100.])
        self.assertEqual(row.regularization_loss().item(), 5000.)
        bounded = AffineBinaryMidpoint(2, AffineBinaryConfig(enabled=True, midpoint_bound=.1))
        with torch.no_grad():
            bounded.midpoint.copy_(row.midpoint)
        bounded.project_()
        torch.testing.assert_close(bounded.midpoint, torch.tensor([-.1, .1]))
        with torch.no_grad():
            row.midpoint[0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            row.project_()
        row = AffineBinaryMidpoint(2, AffineBinaryConfig()).half()
        with self.assertRaisesRegex(ValueError, "F32"):
            row.project_()

    def test_hard_formula_and_zero_midpoint_identity_all_widths(self):
        tiny = torch.tensor(1, dtype=torch.int32).view(torch.float32).item()
        x = torch.tensor([[1., -.3, 0., -0., -tiny], [0., -0., 0., 0., 0.]])
        signs = torch.tensor([[1., -1., 1., -1., 1.], [-1., 1., 1., 1., -1.]])
        alpha, midpoint, bias = torch.tensor([.17, .29]), torch.tensor([.13, -.07]), torch.tensor([.1, -.2])
        for bits in (1, 4, 8, 16):
            values, codes, beta = boundary(x, bits)
            operand = values if codes is None else codes
            dot, summed = F.linear(operand, signs), operand.sum(dim=-1, keepdim=True)
            expected = dot * alpha + summed * midpoint if beta is None else (dot * alpha) * beta + (summed * midpoint) * beta
            for single in (False, True):
                actual = affine_binary_projection(values, signs, alpha, midpoint, codes=codes, beta=beta, bias=bias, single_forward=single)
                self.assertTrue(torch.equal(actual, expected + bias))
                identity = affine_binary_projection(values, signs, alpha, torch.zeros_like(midpoint), codes=codes, beta=beta, bias=bias, single_forward=single)
                base = dot * alpha if beta is None else (dot * alpha) * beta
                self.assertTrue(torch.equal(identity.view(torch.int32), (base + bias).view(torch.int32)))

    def test_attached_vjp_reference_single_and_zero_scale(self):
        for bits in (1, 4, 8, 16):
            gradients = []
            outputs = []
            for single in (False, True):
                x = torch.tensor([[1.7, -.8, .3], [-.2, .6, 1.1]], requires_grad=True)
                latent = torch.tensor([[.2, -.3, .4], [-.3, .1, .4]], requires_grad=True)
                alpha = torch.tensor([0., .3], requires_grad=True)
                midpoint = torch.tensor([.2, -.1], requires_grad=True)
                values, codes, beta = boundary(x, bits)
                out = affine_binary_projection(values, hard_sign_ste(latent), alpha, midpoint, codes=codes, beta=beta, single_forward=single)
                (out * torch.tensor([[.2, -.7], [.6, .3]])).sum().backward()
                outputs.append(out.detach())
                gradients.append([t.grad.clone() for t in (x, latent, alpha, midpoint)])
                self.assertGreater(float(alpha.grad[0].abs()), 0)
                self.assertGreater(float(midpoint.grad.abs().sum()), 0)
                self.assertGreater(float(x.grad.abs().sum()), 0)
            self.assertTrue(torch.equal(outputs[0], outputs[1]))
            for first, second in zip(*gradients):
                torch.testing.assert_close(first, second, rtol=1e-5, atol=2e-7)

    def test_midpoint_gradient_uses_actual_q_even_if_codes_differ(self):
        values = torch.tensor([[1., -2., 3.]], requires_grad=True)
        codes, beta = torch.tensor([[1., 1., 1.]]), torch.tensor([[.5]])
        midpoint = torch.tensor([.2], requires_grad=True)
        out = affine_binary_projection(values, torch.ones(1, 3), torch.zeros(1), midpoint, codes=codes, beta=beta)
        out.sum().backward()
        self.assertEqual(midpoint.grad.item(), 2.)
        torch.testing.assert_close(values.grad, torch.full_like(values, .2))
        self.assertAlmostEqual(out.item(), .3, places=6)

    def test_learned_quantizer_sum_gradient_and_zero_activation(self):
        for zero in (False, True):
            x = torch.tensor([[0., 0., 0.] if zero else [.8, -.2, .6]], requires_grad=True)
            factor = torch.tensor(1.2, requires_grad=True)
            values, codes, beta = boundary(x, 1)
            values = values * factor
            beta = beta * factor.detach()
            midpoint = torch.tensor([.3], requires_grad=True)
            output = affine_binary_projection(values, torch.ones(1, 3), torch.zeros(1), midpoint, codes=codes, beta=beta)
            output.sum().backward()
            torch.testing.assert_close(x.grad, torch.full_like(x, .36))
            if zero:
                self.assertEqual(midpoint.grad.item(), 0)
                self.assertEqual(output.item(), 0)
            else:
                self.assertGreater(abs(factor.grad.item()), 0)

    def test_shared_boundary_one_code_sum_one_q_sum_one_gemm_per_projection(self):
        x = torch.tensor([[1., -.2, .3]], requires_grad=True)
        signs, alpha, midpoint = torch.ones(2, 3), torch.ones(2), torch.ones(2)
        values, codes, beta = boundary(x, 1)
        original_sum = torch.Tensor.sum
        calls = []
        def count_sum(tensor, *args, **kwargs):
            calls.append(tensor)
            return original_sum(tensor, *args, **kwargs)
        with shared_affine_input_sums(), patch.object(torch.Tensor, "sum", count_sum), patch("w1a1_eagle.affine_binary.F.linear", wraps=F.linear) as linear:
            # Source keys intentionally share mathematically equivalent Q nodes.
            outputs = []
            for _ in range(3):
                v, c, b = boundary(x, 1)
                outputs.append(affine_binary_projection(v, signs, alpha, midpoint, codes=c, beta=b, cache_key=(x, 1, "fixed")))
            self.assertEqual(len(calls), 2)
            self.assertEqual(linear.call_count, 3)
        sum(o.sum() for o in outputs).backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 12.))
        with shared_affine_input_sums():
            first = affine_input_sum(values, codes=codes, beta=beta)
            self.assertIs(first, affine_input_sum(values, codes=codes, beta=beta))
            with self.assertRaisesRegex(ValueError, "nested"):
                with shared_affine_input_sums():
                    pass
        with shared_affine_input_sums():
            self.assertIsNot(first, affine_input_sum(values, codes=codes, beta=beta))

    def test_mutation_version_separates_sum_and_quantizers(self):
        x = torch.tensor([[1., -1., 1.]])
        values, codes, beta = boundary(x, 1)
        with shared_affine_input_sums():
            first = affine_input_sum(values, codes=codes, beta=beta, cache_key=(x, 1, "fixed"))
            different = affine_input_sum(values, codes=-codes, beta=beta, cache_key=(x, 1, "learned", 0))
            self.assertIsNot(first, different)
            x.add_(1)
            second = affine_input_sum(values, codes=codes, beta=beta, cache_key=(x, 1, "fixed"))
            self.assertIsNot(first, second)

    def test_sum_and_projection_fail_closed_and_inference_no_stale_cache(self):
        values, codes, beta = boundary(torch.tensor([[1., -.3, .4]]), 1)
        for c, b in ((codes, None), (None, beta), (codes[:, :2], beta),
                     (codes + .1, beta), (codes, -beta), (codes, beta.double())):
            with self.subTest(c=c, b=b), self.assertRaises(ValueError):
                affine_input_sum(values, codes=c, beta=b)
        with self.assertRaisesRegex(ValueError, "source"):
            affine_input_sum(values, codes=codes, beta=beta, cache_key=("id", 1))
        with torch.inference_mode(), shared_affine_input_sums():
            x = torch.tensor([[1., -1., 1.]])
            first = affine_input_sum(x)
            x.add_(1)
            second = affine_input_sum(x)
            self.assertEqual(first.hard.item(), 1.)
            self.assertEqual(second.hard.item(), 4.)
            self.assertIsNot(first, second)

    def test_a16_shared_single_input_reduction(self):
        values = torch.tensor([[1.0001, -.3, .4]]).half().float()
        original_sum = torch.Tensor.sum
        calls = []
        def count_sum(tensor, *args, **kwargs):
            calls.append(tensor)
            return original_sum(tensor, *args, **kwargs)
        with shared_affine_input_sums(), patch.object(torch.Tensor, "sum", count_sum):
            for _ in range(2):
                affine_binary_projection(values, torch.ones(2, 3), torch.ones(2), torch.ones(2))
            self.assertEqual(len(calls), 1)

    def test_later_state_kv_gradients(self):
        first = torch.tensor([[.7, -.2, .4]], requires_grad=True)
        midpoint = torch.tensor([.13, -.07, .04], requires_grad=True)
        signs = torch.tensor([[1., -1., 1.], [-1., 1., 1.], [1., 1., -1.]])
        def project(x):
            q, codes, beta = boundary(x, 1)
            return affine_binary_projection(q, signs, torch.tensor([.2, .3, .4]), midpoint, codes=codes, beta=beta)
        state = project(first)
        state.retain_grad()
        key, value = project(state), project(state * .7)
        key.retain_grad(); value.retain_grad()
        last = project(state + key * .3 + value * .2)
        (last * torch.tensor([[.2, -.7, .5]])).sum().backward()
        for tensor in (first, state, key, value, midpoint):
            self.assertIsNotNone(tensor.grad)
            self.assertGreater(float(tensor.grad.abs().sum()), 0)

    def test_optimizer_exact_joint_ownership_and_target_views(self):
        linears, target, bank = fixture()
        base = [p for m in linears.values() for p in (m.latent_sign, m.scale_offset)]
        optimizer = torch.optim.AdamW([{"params": base}, bank.parameter_group(target=target)])
        validate_affine_optimizer(optimizer, bank, target=target, base_parameters=base)
        incomplete = torch.optim.AdamW(base)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_affine_optimizer(incomplete, bank, target=target, base_parameters=base)
        target.register_buffer("affine_alias", bank.rows[0].midpoint.detach().view(-1))
        with self.assertRaisesRegex(ValueError, "aliases target"):
            validate_affine_optimizer(optimizer, bank, target=target, base_parameters=base)

    def test_strict_payload_atomic_restore_and_native_inventory(self):
        _, _, bank = fixture(AffineBinaryConfig(enabled=True, coverage="all"))
        with torch.no_grad():
            for row in bank.rows:
                row.midpoint.fill_(.1)
        payload = bank.state_payload()
        restored = fixture(bank.config)[2]
        restored.load_payload(payload)
        self.assertEqual(restored.identity(), bank.identity())
        for a, b in zip(bank.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(a, b))
        for mutation in ("extra", "hash", "dtype", "config"):
            bad = copy.deepcopy(payload)
            path = next(iter(bad["state"]))
            if mutation == "extra": bad["extra"] = 1
            elif mutation == "hash": bad["state"][path][0] += 1
            elif mutation == "dtype": bad["state"][path] = bad["state"][path].half()
            else: bad["identity"]["config"]["coverage"] = "fusion"
            with self.assertRaises(ValueError): restored.load_payload(bad)
            self.assertTrue(torch.equal(restored.rows[0].midpoint, bank.rows[0].midpoint))
        descriptor, tensors = bank.native_payload()
        self.assertEqual(descriptor["arithmetic"], ARITHMETIC)
        self.assertEqual(set(descriptor["tensors"]), set(CANDIDATE_D_BASE_TO_PATH))
        self.assertEqual(set(tensors), {base + ".w1ax_midpoint" for base in CANDIDATE_D_BASE_TO_PATH})
        self.assertTrue(all(t.dtype == torch.float32 and t.shape == (2,) for t in tensors.values()))
        self.assertEqual(bank.manifest_payload()["storage_bytes"], 9 * 2 * 4)

    def test_tiny_synthetic_optimizer_exact_resume(self):
        config = AffineBinaryConfig(enabled=True, mild_l2=.01)
        bank = fixture(config)[2]
        optimizer = torch.optim.AdamW(bank.parameters(), lr=.01, weight_decay=0)
        def step(b, opt):
            opt.zero_grad()
            ((b.rows[0].midpoint - torch.tensor([.2, -.1])).square().sum() + b.regularization_loss()).backward()
            opt.step(); b.project_()
        step(bank, optimizer)
        state, optstate = bank.state_payload(), copy.deepcopy(optimizer.state_dict())
        resumed = fixture(config)[2]
        resumed.load_payload(state)
        ropt = torch.optim.AdamW(resumed.parameters(), lr=.01, weight_decay=0)
        ropt.load_state_dict(optstate)
        step(bank, optimizer); step(resumed, ropt)
        self.assertTrue(torch.equal(bank.rows[0].midpoint, resumed.rows[0].midpoint))
        for key in ("exp_avg", "exp_avg_sq", "step"):
            self.assertTrue(torch.equal(optimizer.state[bank.rows[0].midpoint][key], ropt.state[resumed.rows[0].midpoint][key]))


if __name__ == "__main__":
    unittest.main()

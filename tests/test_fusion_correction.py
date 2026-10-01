"""CPU checks of fusion residual arithmetic, ownership, resume and synthetic fits."""

import copy
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.fit_fusion_correction import (
    fit_manifest,
    fit_reduced_rank,
    load_joined_calibration,
    row_sha256,
    sha256,
)
from w1a1_eagle.fusion_correction import (
    ARITHMETIC,
    FusionCorrection,
    FusionCorrectionConfig,
    correction_parameter_group,
    install_fusion_correction,
    validate_correction_optimizer,
)
from w1a1_eagle.recurrent_qat import JointQATConfig, RowBinaryLinear, W1AxContract


def synthetic_manifest(directory: Path, rank: int = 1):
    rng = np.random.default_rng(37)
    x = rng.normal(size=(24, 6)).astype(np.float32)
    u = rng.normal(size=(5, rank)).astype(np.float32) * 0.03
    v = rng.normal(size=(rank, 6)).astype(np.float32) * 0.2
    binary = rng.normal(size=(24, 5)).astype(np.float32)
    reference = binary + x @ v.T @ u.T + np.float32(0.01)
    source = {
        "base_weights_sha256": "a" * 64,
        "reference_weights_sha256": "b" * 64,
        "quantizer_sha256": "c" * 64,
    }
    rows = []
    for i in range(len(x)):
        prompt = f"synthetic-{i // 4}"
        rows.append(
            {
                "row_id": f"row-{i}",
                "prompt_id": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "domain": ("prose", "code", "reasoning")[i // 4 % 3],
                "depth": i % 4,
                "position": i % 4,
                "split": "train" if i < 16 else "validation",
                "source_quantizer_sha256": source["quantizer_sha256"],
                "raw_input_sha256": row_sha256(x[i]),
                "binary_output_sha256": row_sha256(binary[i]),
                "reference_output_sha256": row_sha256(reference[i]),
            }
        )
    join_ids = np.array([r["row_id"] for r in rows])
    arrays = {
        "raw_input": x,
        "binary_output": binary,
        "reference_output": reference,
        "raw_join_ids": join_ids,
        "binary_join_ids": join_ids.copy(),
        "reference_join_ids": join_ids.copy(),
    }
    operands = directory / "operands.npz"
    np.savez(operands, **arrays)
    manifest = {
        "version": 1,
        "projection": "fc",
        "raw_input_stage": "pre_activation_quantization",
        "source_data_split": "train",
        "eligibility": "training_allowed",
        "synthetic": True,
        "source": source,
        "operands": operands.name,
        "operands_sha256": sha256(operands),
        "rows": rows,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest, arrays


class FusionCorrectionTests(unittest.TestCase):
    def test_default_off_has_no_parameters_and_returns_base_object(self):
        module = FusionCorrection(6, 5)
        x, base = torch.randn(2, 6), torch.randn(2, 5)
        self.assertEqual(list(module.parameters()), [])
        self.assertIs(module.add_to(x, base), base)
        self.assertEqual(module.native_payload(), (None, {}))
        self.assertEqual(module.state_payload()["state"], {})
        torch.testing.assert_close(module(x), torch.zeros(2, 5), rtol=0, atol=0)

    def test_initial_zero_and_useful_first_u_gradient(self):
        for rank in (1, 4):
            module = FusionCorrection(6, 5, FusionCorrectionConfig(True, rank))
            same = FusionCorrection(6, 5, FusionCorrectionConfig(True, rank))
            torch.testing.assert_close(module.v, same.v, rtol=0, atol=0)
            self.assertGreater(float(module.v.detach().abs().sum()), 0)
            x = torch.arange(12, dtype=torch.float32).reshape(2, 6).requires_grad_()
            output = module(x)
            torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
            output.sum().backward()
            self.assertGreater(float(module.u.grad.abs().sum()), 0)
            torch.testing.assert_close(module.v.grad, torch.zeros_like(module.v), rtol=0, atol=0)
            torch.testing.assert_close(x.grad, torch.zeros_like(x), rtol=0, atol=0)
            with torch.no_grad():
                module.u.fill_(0.1)
            module.zero_grad()
            x.grad = None
            module(x).sum().backward()
            self.assertGreater(float(module.v.grad.abs().sum()), 0)
            self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_hard_f16_factors_f32_intermediate_and_ste(self):
        module = FusionCorrection(6, 5, FusionCorrectionConfig(True, 4))
        with torch.no_grad():
            module.u.copy_(torch.arange(20).reshape(5, 4) * 0.011 + 0.00001)
        x = torch.arange(12, dtype=torch.float32).reshape(2, 6) * 0.013
        expected = F.linear(F.linear(x, module.v.half().float()), module.u.half().float())
        self.assertGreater(float((module.u - module.u.half().float()).detach().abs().sum()), 0)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(x)
        self.assertEqual(output.dtype, torch.float32)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.sum().backward()
        expected_u_grad = F.linear(x, module.v.half().float()).sum(0).expand_as(module.u)
        torch.testing.assert_close(module.u.grad, expected_u_grad, rtol=0, atol=0)
        # This input would change under an F16 intermediate cast.
        wrong = F.linear(
            F.linear(x, module.v.half().float()).half().float(), module.u.half().float()
        )
        self.assertFalse(torch.equal(expected, wrong))

    def test_raw_path_distinguishes_same_a1_representation(self):
        fc = RowBinaryLinear(torch.ones(1, 2), torch.ones(1), W1AxContract(1))
        x = torch.tensor([[1.0, 3.0], [3.0, 1.0]])
        base = fc(x)
        self.assertEqual(float(base[0].detach()), float(base[1].detach()))
        correction = install_fusion_correction(
            fc, target=nn.Linear(2, 1), config=FusionCorrectionConfig(True)
        )
        with torch.no_grad():
            correction.u.fill_(1)
            correction.v.copy_(torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(fc(x), base + torch.tensor([[1.0], [3.0]]), rtol=0, atol=0)
        torch.testing.assert_close(fc(input=x), fc(x), rtol=0, atol=0)

    def test_hook_zero_equivalence_and_deepcopy_independence(self):
        fc = RowBinaryLinear(torch.randn(5, 6), torch.ones(5) * 0.2, W1AxContract(1))
        x = torch.randn(2, 3, 6)
        base = fc(x)
        original = install_fusion_correction(
            fc, target=nn.Linear(6, 5), config=FusionCorrectionConfig(True, 4)
        )
        torch.testing.assert_close(fc(x), base, rtol=0, atol=0)
        other = copy.deepcopy(fc)
        with torch.no_grad():
            other.fusion_correction.u.fill_(0.2)
        self.assertFalse(torch.equal(other(x), fc(x)))
        other(x).sum().backward()
        self.assertIsNone(original.u.grad)
        self.assertGreater(float(other.fusion_correction.u.grad.abs().sum()), 0)
        self.assertNotEqual(original.v.data_ptr(), other.fusion_correction.v.data_ptr())
        with self.assertRaisesRegex(ValueError, "already attached"):
            install_fusion_correction(fc, target=nn.Linear(6, 5))

    def test_bias_bound_and_native_tensor_contract(self):
        module = FusionCorrection(6, 5, FusionCorrectionConfig(True, 4, True, 0.03))
        with torch.no_grad():
            module.output_bias.fill_(2)
        torch.testing.assert_close(module(torch.randn(2, 6)), torch.full((2, 5), 0.03))
        descriptor, arrays = module.native_payload()
        self.assertEqual(descriptor["arithmetic"], ARITHMETIC)
        self.assertEqual(arrays["fc.correction_u.weight"].dtype, torch.float16)
        self.assertEqual(arrays["fc.correction_u.weight"].shape, (5, 4))
        self.assertEqual(arrays["fc.correction_v.weight"].shape, (4, 6))
        self.assertEqual(arrays["fc.correction_bias"].dtype, torch.float32)
        self.assertEqual(float(module.output_bias[0].detach()), 2)  # Export preserves masters.
        module.project_()
        self.assertLessEqual(float(module.output_bias.detach().abs().max()), 0.030000001)

    def test_guards_and_rng_isolation(self):
        state = torch.random.get_rng_state().clone()
        FusionCorrection(6, 5, FusionCorrectionConfig(True, 4, seed=123))
        torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
        for kwargs in (
            {"rank": 2},
            {"rank": True},
            {"bias_bound": float("nan")},
            {"output_bias": True},
            {"seed": -1},
            {"enabled": 1},
        ):
            with self.assertRaises(ValueError):
                FusionCorrectionConfig(**kwargs)
        module = FusionCorrection(6, 5, FusionCorrectionConfig(True))
        with self.assertRaisesRegex(ValueError, "width"):
            module(torch.tensor(1.0))
        with self.assertRaisesRegex(ValueError, "finite"):
            module(torch.full((6,), float("nan")))
        with torch.no_grad():
            module.u.fill_(1e6)
        with self.assertRaisesRegex(ValueError, "overflows F16"):
            module(torch.ones(6))
        module.u.data.zero_()
        module.half()
        with self.assertRaisesRegex(ValueError, "remain F32"):
            module(torch.ones(6))

    def test_target_aliases_rejected_before_mutation(self):
        fc = RowBinaryLinear(torch.ones(5, 6), torch.ones(5), W1AxContract(1))
        with self.assertRaisesRegex(ValueError, "target owns"):
            install_fusion_correction(fc, target=fc, config=FusionCorrectionConfig(True))
        target = nn.Module()
        target.register_buffer("view", fc.latent_sign.detach().view(-1))
        with self.assertRaisesRegex(ValueError, "alias target storage"):
            install_fusion_correction(fc, target=target, config=FusionCorrectionConfig(True))
        self.assertFalse(hasattr(fc, "fusion_correction"))

    def test_optimizer_ownership(self):
        module = FusionCorrection(6, 5, FusionCorrectionConfig(True, 4, True))
        target = nn.Linear(6, 5)
        group = correction_parameter_group(module, target=target, lr=0.01)
        optimizer = torch.optim.AdamW([group])
        validate_correction_optimizer(optimizer, [module], target=target)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_correction_optimizer(
                optimizer, [module], base_parameters=[module.u], target=target
            )
        wrong = torch.optim.AdamW([module.u], lr=0.01)
        with self.assertRaisesRegex(ValueError, "exactly"):
            validate_correction_optimizer(wrong, [module], target=target)
        self.assertIsNone(
            correction_parameter_group(FusionCorrection(6, 5), target=target, lr=0.01)
        )

    def test_exact_state_optimizer_resume_and_fail_closed_identity(self):
        config = FusionCorrectionConfig(True, 4, True)
        module = FusionCorrection(6, 5, config)
        optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
        x = torch.arange(12).float().reshape(2, 6) * 0.03

        def step(correction, opt):
            opt.zero_grad(set_to_none=True)
            (correction(x) - 0.2).square().mean().backward()
            opt.step()

        step(module, optimizer)
        payload, optimizer_state = module.state_payload(), copy.deepcopy(optimizer.state_dict())
        restored = FusionCorrection(6, 5, config)
        restored.load_payload(payload)
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.01)
        restored_optimizer.load_state_dict(optimizer_state)
        step(module, optimizer)
        step(restored, restored_optimizer)
        for first, second in zip(module.parameters(), restored.parameters(), strict=True):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        before = restored.state_payload()
        malformed = copy.deepcopy(payload)
        malformed["state"]["v"][0, 0] += 0.1
        with self.assertRaisesRegex(ValueError, "hash differs"):
            restored.load_payload(malformed)
        for name, tensor in before["state"].items():
            torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            different = FusionCorrection(6, 5, FusionCorrectionConfig(True, 4, True, seed=1))
            different.load_payload(payload)
        malformed = copy.deepcopy(payload)
        malformed["state"]["u"] = malformed["state"]["u"].half()
        with self.assertRaisesRegex(ValueError, "invalid fusion master"):
            restored.load_payload(malformed)

    def test_later_only_loss_reaches_correction_earlier_state_and_kv(self):
        from scripts.train_joint_w1ax import tiny_joint_fixture, tiny_rollout

        config = JointQATConfig(W1AxContract(16), seed=3)
        linears, _ = tiny_joint_fixture(config)
        correction = install_fusion_correction(
            linears["fc"], target=nn.Linear(4, 4), config=FusionCorrectionConfig(True, 4)
        )
        with torch.no_grad():
            correction.u.fill_(0.1)
        logits, cache, states = tiny_rollout(linears, "cpu")
        F.cross_entropy(logits[-1:], torch.tensor([1])).backward()
        for gradient in (
            correction.u.grad,
            correction.v.grad,
            cache[0][0].grad,
            cache[0][1].grad,
            states[0].grad,
        ):
            self.assertGreater(float(gradient.abs().sum()), 0)


class FusionFitTests(unittest.TestCase):
    def test_synthetic_rank1_rank4_fit_and_heldout_strata(self):
        for rank in (1, 4):
            with tempfile.TemporaryDirectory() as directory:
                path, _, _ = synthetic_manifest(Path(directory), rank)
                _, report = fit_manifest(path, FusionCorrectionConfig(True, rank, True), ridge=1e-4)
                for split in ("train", "validation"):
                    self.assertLess(
                        report["metrics"][split]["corrected_mse"],
                        report["metrics"][split]["base_mse"] * 0.001,
                    )
                self.assertEqual(report["metrics"]["validation"]["prompts"], 2)
                self.assertTrue(report["strata"])
                self.assertFalse(report["native_acceptance_measured"])

    def test_validation_operands_never_change_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest, arrays = synthetic_manifest(Path(directory), 1)
            first, _ = fit_manifest(path, FusionCorrectionConfig(True, 1, True))
            arrays["reference_output"][16:] += 10
            for index in range(16, 24):
                manifest["rows"][index]["reference_output_sha256"] = row_sha256(
                    arrays["reference_output"][index]
                )
            archive = Path(directory) / "operands.npz"
            np.savez(archive, **arrays)
            manifest["operands_sha256"] = sha256(archive)
            path.write_text(json.dumps(manifest))
            second, report = fit_manifest(path, FusionCorrectionConfig(True, 1, True))
            for a, b in zip(first.parameters(), second.parameters(), strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertGreater(report["metrics"]["validation"]["corrected_mse"], 90)

    def test_manifest_rejects_bad_eligibility_prompt_leak_and_quantizer(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest, _ = synthetic_manifest(Path(directory))
            mutations = [
                ("eligibility", "preparation_only"),
                ("source_data_split", "final"),
                ("raw_input_stage", "post_activation_quantization"),
                ("synthetic", False),
            ]
            for key, value in mutations:
                invalid = copy.deepcopy(manifest)
                invalid[key] = value
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError):
                    load_joined_calibration(path)
            invalid = copy.deepcopy(manifest)
            for row in invalid["rows"][16:20]:
                row["prompt_sha256"] = invalid["rows"][0]["prompt_sha256"]
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "disjoint prompt content"):
                load_joined_calibration(path)
            invalid = copy.deepcopy(manifest)
            invalid["rows"][0]["source_quantizer_sha256"] = "d" * 64
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "quantizer differs"):
                load_joined_calibration(path)

    def test_archive_and_row_join_hashes_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path, manifest, arrays = synthetic_manifest(Path(directory))
            arrays["reference_join_ids"] = arrays["reference_join_ids"][::-1].copy()
            archive = Path(directory) / "operands.npz"
            np.savez(archive, **arrays)
            with self.assertRaisesRegex(ValueError, "archive SHA256"):
                load_joined_calibration(path)
            manifest["operands_sha256"] = sha256(archive)
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "row joins disagree"):
                load_joined_calibration(path)
            arrays["reference_join_ids"] = arrays["raw_join_ids"].copy()
            arrays["raw_input"][0, 0] += 1
            np.savez(archive, **arrays)
            manifest["operands_sha256"] = sha256(archive)
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "row operand SHA256"):
                load_joined_calibration(path)

    def test_regression_guards_and_synthetic_cli(self):
        with self.assertRaisesRegex(ValueError, "ridge"):
            fit_reduced_rank(
                torch.ones(4, 6), torch.ones(4, 5), FusionCorrectionConfig(True), ridge=0
            )
        with tempfile.TemporaryDirectory() as directory:
            import sys

            path, _, _ = synthetic_manifest(Path(directory))
            output = Path(directory) / "fit"
            script = Path(__file__).resolve().parents[1] / "scripts" / "fit_fusion_correction.py"
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--manifest",
                    str(path),
                    "--output-dir",
                    str(output),
                    "--output-bias",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("validation", result.stdout)
            report = json.loads((output / "fit_report.json").read_text())
            checkpoint = torch.load(output / "fusion_correction.pt", weights_only=True)
            restored = FusionCorrection(6, 5, FusionCorrectionConfig(True, 1, True))
            restored.load_payload(checkpoint["correction"])
            self.assertEqual(report["checkpoint_sha256"], sha256(output / "fusion_correction.pt"))
            self.assertEqual(report["execution"], "synthetic_fit")


if __name__ == "__main__":
    unittest.main()

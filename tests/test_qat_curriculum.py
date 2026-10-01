"""Synthetic CPU curriculum/ancestry checks, with no real data optimizer steps."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from w1a1_eagle.qat_curriculum import (
    CurriculumConfig, CurriculumState, PrecisionStage, depth_exposure,
    depth_weighted_supported_ce, plan_curriculum_refresh, transfer_precision_state_,
)
from w1a1_eagle.qat_optimization import BinaryOptimizationConfig, make_binary_optimizer
from w1a1_eagle.recurrent_loss import supported_prefix_ce
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract
from w1a1_eagle.recurrent_trace import TraceAudit
from w1a1_eagle.trajectory_refresh import file_record


def audit():
    return TraceAudit(draft_labels=(0, -1, 1), valid_mask=(True, True, True),
                      supported_mask=(True, False, True), ce_mask=(True, False, True),
                      denominator_mask=(True, True, True), target_to_draft=(0, 1),
                      counts={}, per_depth={})


def modules(bits):
    return {"linear": RowBinaryLinear(torch.tensor([[.1, -.2], [-.3, .4]]),
                                       torch.tensor([.2, .5]), W1AxContract(bits),
                                       bias=torch.tensor([.1, .2]))}


class CurriculumTests(unittest.TestCase):
    def test_depth_loss_exact_default_and_attached_later_gradient(self):
        state = torch.tensor(1.0, requires_grad=True)
        logits = torch.stack((torch.tensor([1.0, 0.0]), torch.stack((state, state * 2)),
                              torch.stack((state * 2, state))))
        self.assertTrue(torch.equal(depth_weighted_supported_ce(logits, audit(), (0, 1, 2)),
                                   supported_prefix_ce(logits, audit())))
        weighted = depth_weighted_supported_ce(logits, audit(), (0, 1, 2), decay=.8)
        loss = F.cross_entropy(logits[[0, 2]], torch.tensor([0, 1]), reduction="none")
        torch.testing.assert_close(weighted, (loss[0] + .64 * loss[1]) / 1.64)
        weighted.backward()
        self.assertNotEqual(state.grad.item(), 0)
        exposures = depth_exposure(audit(), (0, 1, 2), decay=.8)
        self.assertEqual(exposures["1"]["quality_rows"], 1)
        self.assertEqual(exposures["1"]["supported_rows"], 0)
        self.assertEqual(exposures["2"]["supported_weight"], .8 ** 2)
        for depths, decay in (((0, 2), .8), ((0, 1, 5), .8), ((0, 1, 2), float("nan")), ((0, 1, 2), 0)):
            with self.assertRaises(ValueError):
                depth_weighted_supported_ce(logits, audit(), depths, decay=decay)

    def test_budget_resume_and_staged_options(self):
        config = CurriculumConfig.a8_to_a1(gpu_seconds=10, a8_max_updates=2, a1_max_updates=8)
        state = CurriculumState(config, data_contract={"corpus": "v1"}, model_contract={"model": "v1"})
        self.assertEqual(state.activation_bits, 8)
        self.assertFalse(state.can_start_update(upper_bound_gpu_seconds=3))
        state.record_update(activation_bits=8, gpu_seconds=1, rows=3, tokens=7, supported_rows=2)
        with self.assertRaises(ValueError):
            state.record_update(activation_bits=8, gpu_seconds=2, rows=3, tokens=7, supported_rows=2)
        state.record_update(activation_bits=8, gpu_seconds=1, rows=3, tokens=7, supported_rows=2)
        self.assertEqual(state.activation_bits, 1)
        self.assertEqual(state.global_updates, 2)
        state.record_overhead("capture", 2)
        restored = CurriculumState(config, data_contract={"corpus": "v1"}, model_contract={"model": "v1"})
        restored.load_state_dict(json.loads(json.dumps(state.state_dict(), sort_keys=True, allow_nan=False)))
        self.assertEqual(restored.state_dict(), state.state_dict())
        changed = CurriculumState(config, data_contract={"corpus": "v2"}, model_contract={"model": "v1"})
        with self.assertRaises(ValueError):
            changed.load_state_dict(state.state_dict())
        state.record_transition(source_bits=8, target_bits=1, source_checkpoint_sha256="a" * 64,
                                target_layout_sha256="b" * 64)
        state.record_update(activation_bits=1, gpu_seconds=8, rows=3, tokens=7, supported_rows=2)
        self.assertTrue(state.complete)
        with self.assertRaises(ValueError):
            state.record_update(activation_bits=1, gpu_seconds=.1, rows=3, tokens=7, supported_rows=2)
        ladder = CurriculumConfig((PrecisionStage(8, 1, 1), PrecisionStage(4, 1, 1), PrecisionStage(1, 8, 8)))
        self.assertEqual([stage.activation_bits for stage in ladder.stages], [8, 4, 1])
        for fraction in (0, .5, 1, float("nan")):
            with self.assertRaises(ValueError):
                CurriculumConfig.a8_to_a1(gpu_seconds=10, a8_fraction=fraction, a8_max_updates=2, a1_max_updates=8)

    def test_budgeted_transfer_exact_magnitudes_scales_fresh_moments(self):
        config = CurriculumConfig.a8_to_a1(gpu_seconds=10, a8_max_updates=1, a1_max_updates=8)
        state = CurriculumState(config, data_contract={"corpus": "v1"}, model_contract={"model": "v1"})
        source, target = modules(8), modules(1)
        old_optimizer = make_binary_optimizer(source, BinaryOptimizationConfig())
        for p in source["linear"].parameters():
            p.grad = torch.ones_like(p)
        old_optimizer.step()
        with torch.no_grad():
            source["linear"].scale_offset[0] = -.2
            target["linear"].initial_scale[:] = torch.tensor([.7, .8])
        self.assertTrue(old_optimizer.state)
        with self.assertRaises(ValueError):
            transfer_precision_state_(source, target, BinaryOptimizationConfig(), curriculum=state, source_checkpoint_sha256="a" * 64)
        state.record_update(activation_bits=8, gpu_seconds=2, rows=3, tokens=7, supported_rows=2)
        optimizer, manifest = transfer_precision_state_(source, target, BinaryOptimizationConfig(), curriculum=state, source_checkpoint_sha256="a" * 64)
        self.assertFalse(optimizer.state)
        self.assertEqual(manifest["global_updates"], 1)
        torch.testing.assert_close(source["linear"].latent_sign, target["linear"].latent_sign, rtol=0, atol=0)
        torch.testing.assert_close(source["linear"].effective_scales(), target["linear"].effective_scales(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            transfer_precision_state_(source, target, BinaryOptimizationConfig(), curriculum=state, source_checkpoint_sha256="a" * 64)
        restored = CurriculumState(config, data_contract={"corpus": "v1"}, model_contract={"model": "v1"})
        restored.load_state_dict(state.state_dict())
        self.assertEqual(restored.state_dict(), state.state_dict())

    def test_transfer_prevalidates_frozen_and_nonfinite_state(self):
        config = CurriculumConfig.a8_to_a1(gpu_seconds=10, a8_max_updates=1, a1_max_updates=8)
        state = CurriculumState(config, data_contract={}, model_contract={})
        state.record_update(activation_bits=8, gpu_seconds=2, rows=1, tokens=1, supported_rows=1)
        source, target = modules(8), modules(1)
        before = target["linear"].latent_sign.detach().clone()
        target["linear"].frozen_bias[0] = .5
        with self.assertRaises(ValueError):
            transfer_precision_state_(source, target, BinaryOptimizationConfig(), curriculum=state, source_checkpoint_sha256="a" * 64)
        torch.testing.assert_close(before, target["linear"].latent_sign)
        target["linear"].frozen_bias.copy_(source["linear"].frozen_bias)
        with torch.no_grad():
            source["linear"].latent_sign[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            transfer_precision_state_(source, target, BinaryOptimizationConfig(), curriculum=state, source_checkpoint_sha256="a" * 64)
        self.assertFalse(state.transitions)

    def refresh_fixture(self, directory):
        def write(name, value):
            path = directory / name
            path.write_text(json.dumps(value) if not isinstance(value, str) else value)
            return file_record(path)
        prompts = write("prompts.jsonl", "\n".join(json.dumps({"id": d}) for d in ("prose", "code", "reasoning")))
        checkpoint, exported = write("checkpoint.bin", "synthetic checkpoint"), write("export.bin", "synthetic export")
        identity = {"checkpoint_sha256": checkpoint["sha256"], "export_sha256": exported["sha256"]}
        contract = {"target_gguf_sha256": "c" * 64, "tokenizer_sha256": "d" * 64,
                    "absolute_d2t_sha256": "e" * 64, "native_revision": "0" * 40,
                    "execution_policy_sha256": "f" * 64, "target_vocab_size": 100}
        policy = {"schema": "w1ax_refresh_policy_v1", "split": "train", "train_prompts_sha256": prompts["sha256"],
                  "development_sample_sha256": "b" * 64,
                  "train_sample": [{"prompt_id": d, "domain": d, "split": "train"} for d in ("prose", "code", "reasoning")],
                  "caps": {"max_rounds": 2, "max_student_rows": 3, "max_new_label_rows": 3, "max_new_feature_rows": 6, "max_prefix_tokens": 8},
                  "learning_curve": {"min_completed_steps": 200, "min_relative_ce_improvement": .05,
                                     "max_acceptance_regression": .02, "min_changed_prefix_fraction": .2}}
        rows = [{"id": d, "prompt_id": d, "split": "train", "prefix_token_ids": [1, 2, 4],
                 "feature_prefix_token_ids": [1, 2], **identity} for d in ("prose", "code", "reasoning")]
        index = write("index.jsonl", "")
        artifact = write("native.bin", "synthetic native-source marker")
        teacher = {"schema": "w1ax_refresh_teacher_index_v1", "split": "train", "native_teacher_contract": contract,
                   "train_prompts_sha256": prompts["sha256"], "index": index, "artifacts": {"native": artifact}}
        curve = {"schema": "w1ax_refresh_learning_curve_v1", "split": "development", "development_sample_sha256": "b" * 64,
                 "checkpoint_sha256": checkpoint["sha256"],
                 "previous": {"checkpoint_sha256": "a" * 64, "completed_steps": 100, "hard_ce": 1, "native_acceptance": .5},
                 "current": {"checkpoint_sha256": checkpoint["sha256"], "completed_steps": 200, "hard_ce": .8, "native_acceptance": .5}}
        spec = {"schema": "w1ax_refresh_request_v1", "refresh_round": 0, "student": identity, "native_teacher_contract": contract,
                "inputs": {"policy": write("policy.json", policy), "student_rows": write("rows.jsonl", "\n".join(json.dumps(row) for row in rows)),
                           "checkpoint": checkpoint, "student_export": exported, "teacher_manifest": write("teacher.json", teacher),
                           "train_prompts": prompts, "learning_curve": write("curve.json", curve)}}
        bindings = {"expected_checkpoint_sha256": checkpoint["sha256"], "expected_export_sha256": exported["sha256"],
                    "expected_train_prompts_sha256": prompts["sha256"], "expected_native_teacher_contract": contract,
                    "expected_refresh_round": 0}
        return spec, bindings

    def test_refresh_new_prefix_native_ancestry_and_stale_guards(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, bindings = self.refresh_fixture(Path(temp))
            plan = plan_curriculum_refresh(spec, **bindings)
            self.assertFalse(plan["training_eligible"])
            self.assertTrue(plan["capture_queue_ready"])
            self.assertEqual(plan["counts"]["new_label_rows"], 3)
            self.assertEqual(plan["counts"]["new_feature_rows"], 6)
            labels = [row for row in plan["capture_requests"] if row["kind"] == "label"]
            self.assertTrue(all(row["prefix_token_ids"] == [1, 2, 4] for row in labels))
            for key in ("expected_checkpoint_sha256", "expected_export_sha256", "expected_train_prompts_sha256"):
                changed = {**bindings, key: "9" * 64}
                with self.assertRaises(ValueError):
                    plan_curriculum_refresh(spec, **changed)
            changed = copy.deepcopy(spec)
            changed["native_teacher_contract"]["execution_policy_sha256"] = "9" * 64
            with self.assertRaises(ValueError):
                plan_curriculum_refresh(changed, **bindings)
            Path(spec["inputs"]["student_rows"]["path"]).write_text("tampered")
            with self.assertRaises(ValueError):
                plan_curriculum_refresh(spec, **bindings)


if __name__ == "__main__":
    unittest.main()

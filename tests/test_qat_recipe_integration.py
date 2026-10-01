"""End-to-end tiny optimizer/checkpoint tests for optional QAT recipes."""
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from test_continuous_qat import config, make

from w1a1_eagle.continuous_qat import ContinuousConfig
from w1a1_eagle.continuous_runtime import MATH_FILES
from w1a1_eagle.learned_activation import LearnedActivationBank, native_order_linear
from w1a1_eagle.qat_state import recipe_state, validate_resume_state
from w1a1_eagle.recurrent_qat import (
    JointQATConfig, RowBinaryLinear, W1AxContract, joint_parameter_families,
)


def optimized(**kwargs):
    return config(activation_quantization="learned", a1_computation="single_forward",
                  optimize_cache=True, optimize_head=True,
                  fusion_correction={"enabled": True, "rank": 1, "output_bias": True},
                  binary_optimization={"latent_magnitude": .1, "sign_lr": .01,
                                       "scale_lr": .001}, **kwargs)


class RecipeIntegrationTests(unittest.TestCase):
    def test_all_parameter_families_are_independent_and_shared_once(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder), optimized(max_steps=1))
            first, second = trainer.lanes
            for lane in trainer.lanes:
                families = joint_parameter_families(lane.linears)
                self.assertEqual([len(x) for x in families.values()], [9, 9, 6, 3])
                owned = [p for g in lane.optimizer.param_groups for p in g["params"]]
                self.assertEqual(len(owned), len({id(p) for p in owned}))
                LearnedActivationBank.from_attached(lane.linears).validate_attachment(lane.linears)
                self.assertTrue(all(p.requires_grad for p in owned))
                self.assertFalse(lane.drafter.embed_tokens.weight.requires_grad)
            self.assertIs(first.drafter.embed_tokens.weight, second.drafter.embed_tokens.weight)
            self.assertIsNot(first.linears["fc"].fusion_correction.u,
                             second.linears["fc"].fusion_correction.u)
            self.assertEqual(first.linears["fc"].activation_quantizer.bits, 8)
            self.assertEqual(second.linears["fc"].activation_quantizer.bits, 1)

    def test_learned_single_forward_matches_explicit_surrogate_gradients(self):
        for bits in (1, 4, 8):
            from w1a1_eagle.learned_activation import LearnedActivationQuantizer
            baseline = RowBinaryLinear(torch.tensor([[.1,-.2,.3],[-.4,.2,.2]]),
                                       torch.tensor([.2,.7]), W1AxContract(bits))
            baseline.activation_quantizer = LearnedActivationQuantizer(bits,"fc",3)
            baseline.activation_quantizer.parameter.data.fill_(.2 if bits == 1 else .7)
            candidate = copy.deepcopy(baseline)
            candidate.a1_computation = "single_forward"
            inputs = [torch.tensor([[.1,-.4,.3],[.7,.2,-.3]],requires_grad=True) for _ in range(2)]
            outputs = [m(x) for m,x in zip((baseline,candidate),inputs)]
            torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
            with torch.no_grad():
                q = baseline.activation_quantizer(inputs[0])
                expected = native_order_linear(q, torch.where(baseline.latent_sign < 0,-1.,1.),
                                               baseline.effective_scales())
            torch.testing.assert_close(outputs[1],expected,rtol=0,atol=0)
            for output in outputs: (output * torch.tensor([.4,-.7])).sum().backward()
            torch.testing.assert_close(inputs[0].grad,inputs[1].grad,rtol=1e-4,atol=1e-6)
            for a,b in zip(baseline.parameters(),candidate.parameters()):
                torch.testing.assert_close(a.grad,b.grad,rtol=1e-4,atol=1e-6)

    def test_combined_recipe_exact_resume_and_schema4_export_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            direct = make(root/"direct",optimized(max_steps=4))
            direct.run(require_smoke=False)
            partial = make(root/"resume",optimized(max_steps=2))
            partial.run(require_smoke=False)
            resumed = make(root/"resume",optimized(max_steps=4))
            resumed.resume()
            resumed.run(require_smoke=False)
            self.assertEqual(direct.step,resumed.step)
            for a,b in zip(direct.lanes,resumed.lanes):
                for name in a.linears:
                    for key,value in a.linears[name].state_dict().items():
                        other = b.linears[name].state_dict()[key]
                        if isinstance(value,torch.Tensor):
                            torch.testing.assert_close(value,other,rtol=0,atol=0)
                        else: self.assertEqual(value,other)
                for ga,gb in zip(a.optimizer.param_groups,b.optimizer.param_groups):
                    for pa,pb in zip(ga["params"],gb["params"]):
                        for key,value in a.optimizer.state[pa].items():
                            torch.testing.assert_close(value,b.optimizer.state[pb][key],rtol=0,atol=0)
                manifest = json.loads((Path(resumed.checkpoint["path"]).parent/a.name/"joint.json").read_text())
                self.assertEqual(manifest["schema_version"],4)
                self.assertEqual(manifest["activation_quantizers"]["boundaries"]["fc"]["bits"],
                                 a.config.contract.activation_bits)
                self.assertEqual(manifest["fusion_correction"]["rank"],1)

    def test_resume_rejects_inconsistent_tied_alias_before_mutating(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder),optimized(max_steps=1))
            lane = trainer.lanes[0]
            before = recipe_state(lane.linears)
            states = {name: copy.deepcopy(m.state_dict()) for name,m in lane.linears.items()}
            states["midlayer.self_attn.k_proj"]["activation_quantizer.clip_ratio"].fill_(.5)
            with self.assertRaisesRegex(ValueError,"inconsistent shared"):
                validate_resume_state(lane.linears,states,before)
            torch.testing.assert_close(lane.linears["midlayer.self_attn.q_proj"].activation_quantizer.parameter,
                                       torch.tensor(1.),rtol=0,atol=0)

    def test_changed_recipe_or_frozen_operand_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder),optimized(max_steps=1))
            lane = trainer.lanes[0]
            states = {name: copy.deepcopy(m.state_dict()) for name,m in lane.linears.items()}
            states["fc"]["initial_scale"][0] += 1
            with self.assertRaisesRegex(ValueError,"changed frozen"):
                validate_resume_state(lane.linears,states,recipe_state(lane.linears))
            trainer.run(require_smoke=False)
            changed = make(Path(folder),replace(optimized(max_steps=2),depth_loss_decay=.8))
            with self.assertRaisesRegex(ValueError,"immutable"):
                changed.resume()

    def test_cpu_recipe_smoke_never_steps_or_discovers_cuda(self):
        for optimizer in ("adamw","sgd"):
            with tempfile.TemporaryDirectory() as folder, patch.object(
                torch.cuda,"is_available",side_effect=AssertionError("accelerator query")):
                cfg = replace(optimized(max_steps=1),binary_optimization={"optimizer":optimizer})
                trainer = make(Path(folder),cfg)
                # Choose nonsingular toy binary dots: the width-four default
                # has legal exact zero gate/up scale gradients and fails smoke.
                generator = torch.Generator().manual_seed(0)
                for module in trainer.lanes[1].linears.values():
                    module.latent_sign.data.copy_(torch.randn(module.latent_sign.shape,
                                                             generator=generator).sign()*.1)
                before = [{k:p.detach().clone() for k,p in lane.drafter.named_parameters()}
                          for lane in trainer.lanes]
                with patch.object(torch.optim.Optimizer,"step",side_effect=AssertionError("update")):
                    trainer.smoke(next(iter(trainer.provider.rounds())))
                self.assertEqual(trainer.step,0)
                for lane,saved in zip(trainer.lanes,before):
                    for name,p in lane.drafter.named_parameters():
                        torch.testing.assert_close(p,saved[name],rtol=0,atol=0)
                    self.assertTrue(all(float(state.get("step",0)) == 0
                                        for state in lane.optimizer.state.values()))

    def test_zero_update_boundary_rejects_nonzero_optimizer_moments(self):
        from scripts.train_continuous_w1ax import require_zero_optimizer_progress
        for name in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
            with tempfile.TemporaryDirectory() as folder:
                trainer = make(Path(folder),config(max_steps=1))
                parameter = trainer.lanes[0].optimizer.param_groups[0]["params"][0]
                trainer.lanes[0].optimizer.state[parameter][name] = torch.ones_like(parameter)
                with self.assertRaisesRegex(ValueError,"nonzero optimizer"):
                    require_zero_optimizer_progress(trainer)

    def test_learned_smoke_rejects_actual_zero_scale_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            trainer = make(Path(folder),optimized(max_steps=1))
            with self.assertRaisesRegex(ValueError,"all-nine finite/nonzero"):
                trainer.smoke(next(iter(trainer.provider.rounds())))
            self.assertEqual(trainer.step,0)

    def test_recipe_sources_are_bound_and_guards_reject_invalid_choices(self):
        self.assertTrue({"qat_optimization.py","qat_curriculum.py","learned_activation.py",
                         "fusion_correction.py","qat_state.py"}.issubset(MATH_FILES))
        for kwargs in ({"activation_quantization":"unknown"}, {"a1_computation":"unknown"},
                       {"depth_loss_decay":0}, {"context_chunk_size":0}):
            with self.assertRaises(ValueError): ContinuousConfig(**kwargs)
        with self.assertRaises(ValueError): JointQATConfig(W1AxContract(16),activation_quantization="learned")
        for kwargs in ({"sign_lr": True}, {"warmup_steps": True}, {"max_steps": 1.5},
                       {"depth_loss_decay": True}, {"seeds": (True,2)}):
            with self.assertRaises(ValueError): ContinuousConfig(**kwargs)
        with self.assertRaises(ValueError): W1AxContract(True)
        with self.assertRaises(ValueError): JointQATConfig(W1AxContract(1),activation_lr=True)
        for value in (1,"false"):
            with self.assertRaises(ValueError): JointQATConfig(W1AxContract(1),allow_accelerator=value)


if __name__ == "__main__": unittest.main()

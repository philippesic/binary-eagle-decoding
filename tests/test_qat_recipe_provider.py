"""CPU synthetic v2 readiness and v3 actor proof checks; no GPU discovery."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import check_continuous_w1ax_readiness as gate
import export_recurrent_binary as exporter
import w1ax_capture_provider as provider
import w1ax_continuous_stages as stages
import test_continuous_readiness as legacy_readiness
from w1a1_eagle.fusion_correction import install_fusion_correction
from w1a1_eagle.affine_binary import install_affine_binary
from w1a1_eagle.qat_state import deployment_state_sha256
from w1a1_eagle.learned_activation import BOUNDARY_PATHS, LearnedActivationBank
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract, joint_optimizer, joint_parameter_families


class RecipeProviderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, value):
        path = self.root / name
        if isinstance(value, bytes): path.write_bytes(value)
        else: path.write_text(json.dumps(value))
        return stages.file_record(path)

    def actor(self, *, bits=4, schema=2, affine_coverage="all", learned=True, fusion=True):
        expected = {base: (name, (64 if base == "blk.0.attn_q" else 16 if base == "blk.0.attn_k" else 4, 4))
                    for base, name in exporter.SOURCE_NAMES.items()}
        arrays = {name + suffix: (np.full(shape, .1, np.float32) if suffix == ".latent" else
                                  np.full(shape[0], .2, np.float32))
                  for name, shape in expected.values() for suffix in (".latent", ".scale")}
        quantizers = {"version": 1, "boundaries": {name: {"bits": bits, "threshold_delta": .25 if bits == 1 else 0,
                    "clip_ratio": 1 if bits == 1 else .5} for name in exporter.QUANTIZER_BOUNDARIES}} if schema >= 3 and learned else None
        correction = {"version": 1, "rank": 1, "u_name": "fc.correction_u.weight", "v_name": "fc.correction_v.weight",
                      "bias_name": "fc.correction_bias", "bias_bound": .1, "arithmetic": exporter.CORRECTION_ARITHMETIC} if schema in (4, 5) and fusion else None
        if correction:
            arrays.update({correction["u_name"]: np.full((4, 1), .2, np.float16),
                           correction["v_name"]: np.full((1, 4), .3, np.float16),
                           correction["bias_name"]: np.zeros(4, np.float32)})
        affine = None
        if schema == 5:
            affine = {"version": 1, "coverage": affine_coverage, "arithmetic": exporter.AFFINE_ARITHMETIC,
                      "tensors": {base: base + ".w1ax_midpoint" for base in (expected if affine_coverage == "all" else {"fc"})}}
            for base, name in affine["tensors"].items():
                arrays[name] = np.arange(expected[base][1][0], dtype=np.float32) / 100 + .125
        checkpoint = self.root / "joint.npz"
        np.savez(checkpoint, **arrays)
        manifest = {"schema_version": schema, "base_gguf_sha256": "a" * 64,
                    "checkpoint_sha256": stages.sha256(checkpoint), "scale_layout": "row", "activation_bits": bits,
                    "activation_rule": exporter.LEARNED_ACTIVATION_RULE if quantizers is not None else
                        "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive",
                    "weight_rule": "hard_sign_zero_positive_clipped_identity_ste", "qk_row_order": "original_checkpoint",
                    "export_status": "row_w1ax_requires_native_validation", "objective": "hard_ce",
                    "projections": {base: {"checkpoint_name": name, "shape": list(shape)} for base, (name, shape) in expected.items()}}
        if schema >= 3: manifest["activation_quantizers"] = quantizers
        if schema >= 4: manifest["fusion_correction"] = correction
        if affine: manifest["affine_weights"] = affine
        manifest_record = self.write("joint.json", manifest)
        exported = self.write("student.gguf", b"synthetic export marker")
        extras = set(arrays) - {name + suffix for name, _ in expected.values() for suffix in (".latent", ".scale")}
        values = exporter.load_checkpoint(checkpoint, expected, row_scale=True, extra_names=extras)
        audit = {"schema_version": 1, "base_gguf": {"path": str(self.root / "base.gguf"), "sha256": "a" * 64},
                 "checkpoint": stages.file_record(checkpoint), "checkpoint_manifest": manifest_record, "output": exported,
                 "scale_rule": "f32_learned_nonnegative", "training_arithmetic": "dense_matmul_hard_quant", "scale_layout": "row",
                 "activation_bits": bits, "native_loader_gate": "verify loader and full-drafter numeric parity before deployment",
                 "serialization_audit_passed": True, "preserved_tensors": {},
                 "projections": {base: {"checkpoint_name": expected[base][0], "shape": list(expected[base][1]),
                    "latent_sha256": v["latent_sha256"], "source_scale_sha256": v["source_scale_sha256"],
                    "packed_sha256": exporter.raw_hash(v["packed"]), "gguf_scale_sha256": exporter.raw_hash(v["scale"])} for base, v in values.items()}}
        if quantizers is not None: audit["activation_quantizers"] = quantizers
        if correction:
            names = (correction["u_name"], correction["v_name"], correction["bias_name"])
            audit.update(fusion_correction=correction, fusion_correction_tensor_sha256={name: exporter.raw_hash(arrays[name]) for name in names if name})
        if affine:
            native_midpoints = exporter.load_affine_weights(checkpoint, affine, expected)
            audit.update(affine_weights=affine, affine_midpoint_sha256={name: exporter.raw_hash(value) for name,value in native_midpoints.items()})
        binding = {"checkpoint": stages.file_record(checkpoint), "checkpoint_manifest": manifest_record,
                   "export": exported, "export_audit": self.write("export-audit.json", audit)}
        return binding, manifest, audit, expected

    def test_actor_schema2_3_4_strict_tensor_options(self):
        for schema in (2, 3, 4):
            with self.subTest(schema=schema):
                binding, manifest, audit, _ = self.actor(schema=schema)
                self.assertEqual(provider.validate_actor_export(binding, activation_bits=4, base_hash="a" * 64), manifest)
                changed = copy.deepcopy(audit)
                changed["unexpected"] = True
                binding["export_audit"] = self.write("export-audit.json", changed)
                with self.assertRaises(ValueError):
                    provider.validate_actor_export(binding, activation_bits=4, base_hash="a" * 64)
        binding, manifest, audit, _ = self.actor(schema=4)
        changed = copy.deepcopy(audit)
        changed["fusion_correction_tensor_sha256"]["fc.correction_u.weight"] = "b" * 64
        binding["export_audit"] = self.write("export-audit.json", changed)
        with self.assertRaisesRegex(ValueError, "tensors differ"):
            provider.validate_actor_export(binding, activation_bits=4, base_hash="a" * 64)
        binding, manifest, audit, _ = self.actor(schema=3, bits=1)
        changed = copy.deepcopy(audit)
        changed["activation_quantizers"]["boundaries"]["head"]["threshold_delta"] = .5
        binding["export_audit"] = self.write("export-audit.json", changed)
        with self.assertRaises(ValueError):
            provider.validate_actor_export(binding, activation_bits=1, base_hash="a" * 64)

    def test_loader_all_families_and_prevalidation(self):
        binding, manifest, audit, expected = self.actor(schema=4)
        config = gate.checkpoint_joint_config(binding["checkpoint_manifest"]["path"], 4, "a" * 64)
        modules = {name.removesuffix(".weight"): RowBinaryLinear(torch.full(shape, .5), torch.ones(shape[0]), W1AxContract(4))
                   for name, shape in expected.values()}
        bank = LearnedActivationBank(4, {name: module.in_features for name, module in modules.items()})
        bank.attach(modules)
        install_fusion_correction(modules["fc"], target=torch.nn.Linear(2, 2), config=config.fusion_correction)
        recipe = gate._load_checkpoint(binding["checkpoint"]["path"], binding["checkpoint_manifest"]["path"], modules, 4, "a" * 64)
        self.assertEqual(recipe, gate.checkpoint_recipe(manifest))
        self.assertEqual(float(bank.quantizers["head"].parameter.detach()), .5)
        torch.testing.assert_close(modules["fc"].fusion_correction.u, torch.tensor(np.full((4, 1), .2, np.float16)).float())
        families = joint_parameter_families(modules)
        self.assertEqual({k: len(v) for k, v in families.items()}, {"sign": 9, "scale": 9, "activation": 6, "fusion": 3, "midpoint": 0})
        optimizer = joint_optimizer(modules, config)
        self.assertEqual(sum(len(g["params"]) for g in optimizer.param_groups), 27)
        before = modules["fc"].latent_sign.detach().clone()
        changed = copy.deepcopy(manifest)
        changed["projections"]["fc"]["shape"] = [3, 4]
        record = self.write("joint.json", changed)
        with self.assertRaises(ValueError):
            gate._load_checkpoint(binding["checkpoint"]["path"], record["path"], modules, 4, "a" * 64)
        torch.testing.assert_close(modules["fc"].latent_sign, before)

    def test_a4_readiness_requires_own_versioned_gate(self):
        fixture = legacy_readiness.ReadinessTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        common = fixture.common
        report = fixture.report(4)
        binding, manifest, audit, _ = self.actor(schema=2)
        report.update(schema=gate.RECIPE_SCHEMA, recipe=gate.checkpoint_recipe(manifest), deployment_state_sha256="c" * 64,
                      trainable_parameter_counts={"sign": 9, "scale": 9, "activation": 0, "fusion": 0})
        report["evidence"].update(binding)
        for key in ("native_cell", "native_capture_manifest"):
            path = Path(report["evidence"][key]["path"])
            value = json.loads(path.read_text())
            value["draft_sha256"] = binding["export"]["sha256"]
            if key == "native_cell":
                log = path.parent / "server.log"
                log.write_text("CUDA packed W1A4 BITSERIAL dispatch")
                value["files"] = {"server.log": {"sha256": stages.sha256(log)}}
            path.write_text(json.dumps(value))
            report["evidence"][key] = stages.file_record(path)
        gate.validate_gate_report(report, 4, common)
        gate_record = self.write("gate4.json", report)
        readiness = {"schema": stages.RECIPE_READINESS_SCHEMA, "training_eligible": True, "objective": "hard_ce",
                     "scale_layout": "row", "common_source_sha256": common, "unresolved_gates": [], "precisions": {"4": gate_record}}
        ready_record = self.write("ready.json", readiness)
        stages.validate_readiness(ready_record, activation_bits=4, common_hashes=common)
        with self.assertRaises(ValueError):
            stages.validate_readiness(ready_record, activation_bits=1, common_hashes=common)
        readiness["precisions"] = {"4": self.write("gate4.json", {**report, "schema": gate.SCHEMA})}
        with self.assertRaises(ValueError):
            stages.validate_readiness(self.write("ready.json", readiness), activation_bits=4, common_hashes=common)
        with self.assertRaises(ValueError):
            gate.validate_gate_report(report, 8, common)
        config = gate.checkpoint_joint_config(binding["checkpoint_manifest"]["path"], 4, "a" * 64)
        good_ready = {**readiness, "precisions": {"4": self.write("gate4.json", report)}}
        provider.validate_provider_recipe(config, good_ready)
        with self.assertRaises(ValueError):
            provider.validate_provider_recipe(type("Config", (), {"contract": W1AxContract(4), "activation_quantization": "learned"})(), good_ready)

    def test_refresh_v3_keeps_prefix_actor_binding_and_rejects_stale(self):
        binding, manifest, audit, _ = self.actor(schema=4)
        capture = self.write("capture.json", {"exact_prefix": [1, 2, 4]})
        prompts = self.write("prompts.json", {"split": "train"})
        binary = self.write("binary", b"synthetic native marker")
        common = {"base_draft_gguf": "a" * 64, "target_gguf": "b" * 64}
        receipt = {"schema": "w1ax_exact_prefix_refresh_v3", "training_eligible": False,
                   "changed_prefix_labels_reused": False, "activation_bits": 4,
                   "capture_manifest": capture, "prompts": prompts, "native_binary": binary,
                   "common_source_sha256": common, **{k: binding[k] for k in ("export", "checkpoint", "checkpoint_manifest")}}
        binding["refresh_receipt"] = self.write("receipt.json", receipt)
        args = dict(capture_manifest_sha256=capture["sha256"], captured_draft_sha256=binding["export"]["sha256"],
                    prompts_sha256=prompts["sha256"], common_hashes=common, activation_bits=4, native_binary_sha256=binary["sha256"])
        original = Path(capture["path"]).read_bytes()
        self.assertFalse(provider.validate_captured_drafter(binding, **args)["training_eligible"])
        self.assertEqual(Path(capture["path"]).read_bytes(), original)
        receipt["changed_prefix_labels_reused"] = True
        binding["refresh_receipt"] = self.write("receipt.json", receipt)
        with self.assertRaises(ValueError):
            provider.validate_captured_drafter(binding, **args)

    def test_a4_capture_explicit_marker_with_mocked_execution(self):
        prompts = self.root / "train-prompts.jsonl"
        prompts.write_text(json.dumps({"id": "train-one", "domain": "prose"}) + "\n")
        sources = {"target_gguf": "target", "candidate_d_gguf": "D", "binary": "binary",
                   "sha256": {"target_gguf": provider.TARGET_GGUF_SHA256, "candidate_d_gguf": provider.CANDIDATE_D_SHA256},
                   "native_runtime": {"ld_library_path": "synthetic"}}
        with patch.object(stages, "verify_sources"), patch.object(stages, "host_admission"), \
                patch.object(torch.cuda, "empty_cache"), patch("run_binary_head_capture.run_cell") as run:
            stages.native_capture(sources, prompts, self.root / "native", activation_bits=4, tokens=2)
        args, _, spec, _ = run.call_args.args
        self.assertEqual(args.activation_bits, 4)
        self.assertEqual(args.tokens, 2)
        self.assertIn("CUDA packed W1A4 BITSERIAL dispatch", spec["required_markers"])
        self.assertIsInstance(args.cancellation_guard, stages.NativeCancellationGuard)

    def test_schema5_midpoint_hydration_original_order_and_actor_hashes(self):
        for learned in (False, True):
            binding, manifest, audit, expected = self.actor(schema=5, learned=learned, fusion=False)
            self.assertEqual(provider.validate_actor_export(binding, activation_bits=4, base_hash="a" * 64), manifest)
            config = gate.checkpoint_joint_config(binding["checkpoint_manifest"]["path"], 4, "a" * 64)
            modules = {name.removesuffix(".weight"): RowBinaryLinear(torch.full(shape, .5), torch.ones(shape[0]), W1AxContract(4))
                       for name, shape in expected.values()}
            if learned:
                bank = LearnedActivationBank(4, {name: module.in_features for name,module in modules.items()})
                bank.attach(modules)
            install_affine_binary(modules, target=torch.nn.Linear(2,2), config=config.affine_weights)
            gate._load_checkpoint(binding["checkpoint"]["path"], binding["checkpoint_manifest"]["path"], modules, 4, "a" * 64)
            self.assertEqual(len(joint_parameter_families(modules)["midpoint"]), 9)
            with np.load(binding["checkpoint"]["path"]) as archive:
                for base, tensor in manifest["affine_weights"]["tensors"].items():
                    path = expected[base][0].removesuffix(".weight")
                    torch.testing.assert_close(modules[path].affine_binary.midpoint.detach(), torch.from_numpy(archive[tensor]), rtol=0,atol=0)
            before = deployment_state_sha256(modules)
            modules["fc"].affine_binary.midpoint.data.add_(.125)
            self.assertNotEqual(before, deployment_state_sha256(modules))
            broken = copy.deepcopy(audit)
            broken["affine_midpoint_sha256"]["fc.w1ax_midpoint"] = "b" * 64
            binding["export_audit"] = self.write("export-audit.json", broken)
            with self.assertRaisesRegex(ValueError, "tensors differ"):
                provider.validate_actor_export(binding, activation_bits=4, base_hash="a" * 64)
        binding, manifest, audit, expected = self.actor(schema=5, learned=False, fusion=False, affine_coverage="fusion")
        with np.load(binding["checkpoint"]["path"]) as archive:
            arrays = {name: archive[name].copy() for name in archive.files if name != "fc.w1ax_midpoint"}
        np.savez(binding["checkpoint"]["path"], **arrays)
        manifest["checkpoint_sha256"] = stages.sha256(Path(binding["checkpoint"]["path"]))
        binding["checkpoint"] = stages.file_record(Path(binding["checkpoint"]["path"]))
        binding["checkpoint_manifest"] = self.write("joint.json", manifest)
        audit["checkpoint"],audit["checkpoint_manifest"] = binding["checkpoint"],binding["checkpoint_manifest"]
        binding["export_audit"] = self.write("export-audit.json", audit)
        with self.assertRaisesRegex(ValueError, "missing affine midpoint"):
            provider.validate_actor_export(binding, activation_bits=4, base_hash="a" * 64)
        binding,manifest,audit,_ = self.actor(schema=5,learned=False,fusion=False,affine_coverage="fusion")
        with np.load(binding["checkpoint"]["path"]) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        arrays["fc.w1ax_midpoint"] = arrays["fc.w1ax_midpoint"].astype(np.float16)
        np.savez(binding["checkpoint"]["path"],**arrays)
        binding["checkpoint"] = stages.file_record(Path(binding["checkpoint"]["path"]))
        manifest["checkpoint_sha256"] = binding["checkpoint"]["sha256"]
        binding["checkpoint_manifest"] = self.write("joint.json",manifest)
        audit["checkpoint"],audit["checkpoint_manifest"] = binding["checkpoint"],binding["checkpoint_manifest"]
        binding["export_audit"] = self.write("export-audit.json",audit)
        with self.assertRaisesRegex(ValueError,"invalid affine midpoint"):
            provider.validate_actor_export(binding,activation_bits=4,base_hash="a" * 64)

    def test_affine_head_replay_alpha_zero_and_zero_midpoint_gradient(self):
        for bits in (1,4,8,16):
            binding, manifest, _, expected = self.actor(bits=bits,schema=5,learned=False,fusion=False)
            config = gate.checkpoint_joint_config(binding["checkpoint_manifest"]["path"], bits, "a" * 64)
            modules = {name.removesuffix(".weight"): RowBinaryLinear(torch.full(shape,.5),torch.zeros(shape[0]),W1AxContract(bits))
                       for name,shape in expected.values()}
            install_affine_binary(modules,target=torch.nn.Linear(2,2),config=config.affine_weights)
            module = modules["lm_head"]
            module.affine_binary.midpoint.data.fill_(.25)
            state = torch.tensor([1.,2.,3.,4.])
            packed = exporter.pack(module.latent_sign.detach().numpy())
            actual = gate._replay_head(packed,np.zeros(4,np.float32),4,state,bits,midpoint=module.affine_binary.midpoint)
            torch.testing.assert_close(actual,module(state).flatten(),rtol=0,atol=0)
            self.assertTrue(bool((actual != 0).any()))
            module.zero_grad(set_to_none=True)
            module(torch.tensor([[1.,-1.,1.,-1.]])).sum().backward()
            self.assertIsNotNone(module.affine_binary.midpoint.grad)
            torch.testing.assert_close(module.affine_binary.midpoint.grad,torch.zeros(4))
        fixture = legacy_readiness.ReadinessTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        legacy = fixture.report(8)
        record = legacy["evidence"]["checkpoint_manifest"]
        value = json.loads(Path(record["path"]).read_text())
        value["schema_version"],value["affine_weights"] = 5,manifest["affine_weights"]
        Path(record["path"]).write_text(json.dumps(value))
        legacy["evidence"]["checkpoint_manifest"] = stages.file_record(Path(record["path"]))
        with self.assertRaisesRegex(ValueError,"legacy gate cannot"):
            gate.validate_gate_report(legacy,8,fixture.common)


if __name__ == "__main__": unittest.main()

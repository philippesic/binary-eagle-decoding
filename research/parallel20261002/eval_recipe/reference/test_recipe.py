"""Real tiny producer→construction→loader checks; never call native evaluator."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import check_continuous_w1ax_readiness as gate
import torch
import w1ax_continuous_stages as stages
from adapter import construct_and_load, digest, preflight_checkpoint, preflight_pair
from test_recurrent_provider import dense_drafter
from torch import nn

from w1a1_eagle.learned_activation import LearnedActivationBank
from w1a1_eagle.qat_state import deployment_state_sha256
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    install_joint_linears,
    save_joint_checkpoint,
)

ROOT = Path(__file__).resolve().parents[4]
PROFILES = json.loads((ROOT / "configs/qat_optimization_profiles.json").read_text())["profiles"]
DEPLOYMENT_KEYS = {"activation_quantization", "fusion_correction", "affine_weights"}
NAMES = (
    "baseline",
    "learned-activations",
    "fusion-rank1",
    "fusion-rank4",
    "affine-fusion",
    "affine-all",
    "combined-contract-smoke",
)


def tiny():
    drafter = dense_drafter(native_shape=True)
    drafter.config.num_attention_heads = 32
    drafter.config.num_key_value_heads = 8
    drafter.midlayer.self_attn.q_proj = nn.Linear(8, 64, bias=False)
    drafter.midlayer.self_attn.k_proj = nn.Linear(8, 16, bias=False)
    drafter.midlayer.self_attn.v_proj = nn.Linear(8, 16, bias=False)
    drafter.midlayer.self_attn.o_proj = nn.Linear(64, 4, bias=False)
    with torch.no_grad():
        drafter.embed_tokens.weight.copy_(torch.linspace(-0.4, 0.6, 20).reshape(5, 4))
    return drafter, nn.Linear(4, 4)


def graph_logits(linears):
    from dataclasses import replace

    from test_recurrent_provider import provider_round

    from w1a1_eagle.native_step import NativeStepAdapter
    from w1a1_eagle.recurrent_provider import forward_torch_round

    drafter, _ = tiny()
    for path, module in linears.items():
        parent, leaf = path.rsplit(".", 1) if "." in path else ("", path)
        setattr(drafter.get_submodule(parent), leaf, module)
    batch = replace(provider_round(), raw_target_features=torch.tensor([[0.2, -0.1, 0.3, 0.4] * 3]))
    with torch.no_grad():
        return forward_torch_round(batch, NativeStepAdapter(drafter), 3)


def publish(root):
    # Same export identity envelope emitted by ContinuousTrainer.save;
    # no resume payload/model/update is required for deployment preflight.
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "continuous_joint_w1ax_v1",
                "exports": {
                    lane: {name: digest(root / lane / name) for name in ("joint.npz", "joint.json")}
                    for lane in ("A8", "A1")
                },
            }
        )
    )


def produce(root, bits, name):
    drafter, target = tiny()
    options = {key: value for key, value in PROFILES[name].items() if key in DEPLOYMENT_KEYS}
    config = JointQATConfig(W1AxContract(bits, "row"), **options)
    linears = install_joint_linears(drafter, target, config)
    with torch.no_grad():
        for module in linears.values():
            module.scale_offset.fill_(0.017)
            if getattr(module, "affine_binary", None) is not None:
                module.affine_binary.midpoint.copy_(
                    torch.linspace(-0.21, 0.13, module.out_features)
                )
        if config.activation_quantization == "learned":
            bank = LearnedActivationBank.from_attached(linears)
            for index, quantizer in enumerate(bank.quantizers.values()):
                quantizer.parameter.fill_(0.1137 + index * 0.017 if bits == 1 else 0.731)
        correction = getattr(linears["fc"], "fusion_correction", None)
        if correction is not None:
            correction.u.fill_(0.1234567)
            correction.v.fill_(-0.2134567)
            if correction.output_bias is not None:
                correction.output_bias.fill_(0.0234567)
    checkpoint, manifest = root / "joint.npz", root / "joint.json"
    save_joint_checkpoint(linears, config, "a" * 64, checkpoint, manifest)
    return linears, checkpoint, manifest


class RecipeTests(unittest.TestCase):
    def test_actual_call_chain_source_has_fixed_construction_after_capture(self):
        import inspect

        source = inspect.getsource(stages._evaluate_development)
        self.assertLess(
            source.index("native_capture(sources"), source.index("qat = JointQATConfig")
        )
        self.assertIn('JointQATConfig(W1AxContract(bits, "row")', source)
        self.assertIn("install_joint_linears(drafter, target, qat)", source)
        self.assertNotIn("checkpoint_joint_config", source)

    def test_named_profiles_real_failure_and_exact_reconstruction(self):
        for bits in (8, 1):
            for name in NAMES:
                with self.subTest(bits=bits, name=name), tempfile.TemporaryDirectory() as tmp:
                    source, checkpoint, manifest = produce(Path(tmp), bits, name)
                    original = install_joint_linears(*tiny(), JointQATConfig(W1AxContract(bits)))
                    if name == "baseline":
                        gate._load_checkpoint(checkpoint, manifest, original, bits, "a" * 64)
                        self.assertEqual(
                            deployment_state_sha256(source), deployment_state_sha256(original)
                        )
                    else:
                        with self.assertRaises(ValueError):
                            gate._load_checkpoint(checkpoint, manifest, original, bits, "a" * 64)
                    prepared = preflight_checkpoint(checkpoint, manifest, bits, "a" * 64)
                    loaded = construct_and_load(prepared, *tiny())
                    self.assertEqual(
                        deployment_state_sha256(source), deployment_state_sha256(loaded)
                    )
                    for path in source:
                        x = torch.linspace(-0.81, 0.91, 3 * source[path].in_features).reshape(3, -1)
                        with torch.no_grad():
                            torch.testing.assert_close(
                                source[path](x), loaded[path](x), atol=3e-5, rtol=3e-6
                            )
                    torch.testing.assert_close(
                        graph_logits(source), graph_logits(loaded), atol=3e-5, rtol=3e-6
                    )
                    correction = getattr(loaded["fc"], "fusion_correction", None)
                    if correction is not None:
                        self.assertFalse(
                            torch.equal(source["fc"].fusion_correction.u, correction.u)
                        )
                        torch.testing.assert_close(
                            source["fc"].fusion_correction.u.half().float(),
                            correction.u,
                            atol=0,
                            rtol=0,
                        )

    def test_missing_recipe_or_wrong_precision_prevents_expensive_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for bits in (8, 1):
                produce(root / f"A{bits}", bits, "combined-contract-smoke")
            publish(root)
            manifest = root / "A1/joint.json"
            data = json.loads(manifest.read_text())
            del data["affine_weights"]
            manifest.write_text(json.dumps(data))
            actions = []
            with self.assertRaises(ValueError):
                preflight_pair(root, "a" * 64)
                actions.append("native capture")
            self.assertEqual(actions, [])
            with self.assertRaisesRegex(ValueError, "A4"):
                preflight_checkpoint(root / "A8/joint.npz", root / "A8/joint.json", 4, "a" * 64)

    def test_exact_proposed_pair_preflight_and_construction(self):
        from production_preflight import (
            _development_checkpoint_preflight,
            _development_checkpoint_unchanged,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {}
            for bits in (8, 1):
                sources[bits], _, _ = produce(root / f"A{bits}", bits, "combined-contract-smoke")
            publish(root)
            configs, identities = _development_checkpoint_preflight(root, "a" * 64)
            for bits in (8, 1):
                _development_checkpoint_unchanged(root, bits, identities)
                loaded = install_joint_linears(*tiny(), configs[bits])
                gate._load_checkpoint(
                    root / f"A{bits}/joint.npz",
                    root / f"A{bits}/joint.json",
                    loaded,
                    bits,
                    "a" * 64,
                )
                self.assertEqual(
                    deployment_state_sha256(loaded), deployment_state_sha256(sources[bits])
                )
            manifest = root / "A1/joint.json"
            data = json.loads(manifest.read_text())
            data["activation_quantizers"]["boundaries"]["fc"]["bits"] = 8
            manifest.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                _development_checkpoint_preflight(root, "a" * 64)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                _development_checkpoint_unchanged(root, 1, identities)

    def test_learned_manifest_downgrade_requires_publication_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for bits in (8, 1):
                produce(root / f"A{bits}", bits, "learned-activations")
            with self.assertRaises(FileNotFoundError):
                preflight_pair(root, "a" * 64)
            publish(root)
            manifest = root / "A1/joint.json"
            data = json.loads(manifest.read_text())
            del data["activation_quantizers"]
            data["schema_version"] = 2
            data["activation_rule"] = (
                "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive"
            )
            manifest.write_text(json.dumps(data))
            # Bare NPZ cannot reveal lost learned scalar identity: valid schema2.
            standalone = preflight_checkpoint(root / "A1/joint.npz", manifest, 1, "a" * 64)
            self.assertEqual(standalone.config.activation_quantization, "fixed")
            with self.assertRaisesRegex(ValueError, "export hash mismatch"):
                preflight_pair(root, "a" * 64)

    def test_changed_identity_rejected_before_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, checkpoint, manifest = produce(Path(tmp), 8, "baseline")
            prepared = preflight_checkpoint(checkpoint, manifest, 8, "a" * 64)
            manifest.write_text(manifest.read_text() + " ")
            drafter, target = tiny()
            before = copy.deepcopy(drafter.state_dict())
            with self.assertRaisesRegex(ValueError, "identity changed"):
                construct_and_load(prepared, drafter, target)
            for key, value in drafter.state_dict().items():
                torch.testing.assert_close(value, before[key], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

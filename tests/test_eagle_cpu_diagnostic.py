"""Tiny source-adapter tests and real diagnostic metadata; no released weights."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from check_eagle_qat_cpu_model import (
    UNRESOLVED,
    authenticate,
    capture_ancestry,
    official_classes,
    require_diagnostic_capture,
    smoke,
)
from test_continuous_qat import FixtureProvider
from test_recurrent_provider import dense_drafter
from torch import nn

from w1a1_eagle.continuous_qat import sha256
from w1a1_eagle.native_step import NativeStepAdapter
from w1a1_eagle.qat_initialization import apply_binary_initialization
from w1a1_eagle.recurrent_provider import audit_provider_round
from w1a1_eagle.recurrent_qat import (
    JointQATConfig,
    W1AxContract,
    install_joint_linears,
    joint_optimizer,
)

MAIN = Path("/Users/pippo/github/binary-eagle-decoding")
MANIFEST = MAIN / "results/recurrent-binary-cpu-smoke-20260928-a16/bundle/manifest.json"
PLAN = Path(__file__).resolve().parents[1] / (
    "experiments/nine-model-qat-preparation/eagle-direct-cpu-diagnostic-plan-2026-10-04.json"
)
SOURCE = {
    "path": "/Users/pippo/.cache/uv/git-v0/checkouts/4f55df2cf6edc6bf/0358da9",
    "revision": "0358da9c651e6a7d7ccafea26ced4b9c98d11681",
    "files": {
        "configuration_eagle3_model.py": (
            "248bc33aed309f2779d64a2bcbc436d82d6a66ebd7671ce2bf7a4c1fbd70ddb2"
        ),
        "draft/llama3_eagle3.py": (
            "82a132b04cb914b20aa84d96676ce34e4f9aa7bfbf4edf8225bb42a7c99979a8"
        ),
        "draft/base_model.py": "fed4eaf004e51a8f1ddcdf1c867feb2d89c7f0a123ad6d336e54d973515f9d8d",
    },
}


class EagleCPUDiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(MANIFEST.exists(), "local immutable TRAIN diagnostic absent")
    def test_capture_source_report_cell_drift_and_distinct_actor_roles(self):
        plan, manifest = json.loads(PLAN.read_text()), json.loads(MANIFEST.read_text())
        joined = capture_ancestry(plan, manifest)
        self.assertNotEqual(
            joined["historical_capture_drafter"]["declared_sha256"],
            joined["current_dense_base"]["sha256"],
        )
        self.assertEqual(joined["historical_payload_status"], "historical_payload_SHA_checked_only")
        absent = copy.deepcopy(plan)
        absent["historical_capture_drafter"]["payload"] = None
        self.assertEqual(
            capture_ancestry(absent, manifest)["historical_payload_status"],
            "historicalSHA_unverified_payload_absent",
        )
        for report_name, changed_key in (
            ("features", "features_sha256"),
            ("cell_manifest", "target_sha256"),
        ):
            altered_plan, altered_manifest = copy.deepcopy(plan), copy.deepcopy(manifest)
            original = plan["capture_source_reports"][report_name]
            record = json.loads(Path(original["path"]).read_text())
            record[changed_key] = "0" * 64
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "changed-report.json"
                path.write_text(json.dumps(record))
                locator = {"path": str(path), "sha256": sha256(path)}
                altered_plan["capture_source_reports"][report_name] = locator
                altered_manifest["source_report_sha256"][report_name] = locator["sha256"]
                if report_name == "cell_manifest":
                    altered_plan["capture_source_cell"] = locator
                with self.assertRaisesRegex(ValueError, "ancestry|source report"):
                    capture_ancestry(altered_plan, altered_manifest)
                # Byte drift without repinning is independently refused.
                path.write_text(json.dumps({**record, "complete": False}))
                with self.assertRaisesRegex(ValueError, "SHA differs"):
                    capture_ancestry(altered_plan, altered_manifest)

    @unittest.skipUnless(MANIFEST.exists(), "local immutable TRAIN diagnostic absent")
    def test_actual_scale_only_controls_join_without_model_loading(self):
        plan = json.loads(PLAN.read_text())
        for bits in (8, 1):
            _, _, prompt, manifest, init = authenticate(plan, bits)
            self.assertEqual(prompt["id"], "qat-revisit-train-prose-urban-waterways-01")
            self.assertFalse(manifest["training_eligible"])
            self.assertEqual(init, plan["initializations"][str(bits)])

    @unittest.skipUnless(MANIFEST.exists(), "local immutable TRAIN diagnostic absent")
    def test_candidate_cannot_replace_scale_only_control_and_caps_cannot_expand(self):
        plan = json.loads(PLAN.read_text())
        contract = json.loads(Path(plan["initialization_contract"]["path"]).read_text())
        changed = copy.deepcopy(plan)
        changed["initializations"]["8"]["sha256"] = contract["artifacts"]["fusion-a8"]["sha256"]
        with self.assertRaisesRegex(ValueError, "scale-only"):
            authenticate(changed, 8)
        changed = copy.deepcopy(plan)
        changed["limits"]["max_rss_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "limits differ"):
            authenticate(changed, 8)

    @unittest.skipUnless(MANIFEST.exists(), "local immutable TRAIN diagnostic absent")
    def test_ineligible_original_train_is_preserved_and_never_production(self):
        manifest = json.loads(MANIFEST.read_text())
        require_diagnostic_capture(manifest)
        self.assertFalse(manifest["training_eligible"])
        self.assertEqual(tuple(manifest["unverified_gates"]), UNRESOLVED)
        with self.assertRaisesRegex(ValueError, "cannot grant production"):
            require_diagnostic_capture(manifest, production=True)
        for key, value in (
            ("training_eligible", True),
            ("split", "final"),
            ("unverified_gates", list(UNRESOLVED[:-1])),
        ):
            changed = copy.deepcopy(manifest)
            changed[key] = value
            with self.assertRaisesRegex(ValueError, "preserve"):
                require_diagnostic_capture(changed)

    @unittest.skipUnless(Path(SOURCE["path"]).exists(), "pinned local official source absent")
    def test_official_leaf_import_only_and_wrong_source_pin(self):
        config, drafter = official_classes(SOURCE)
        self.assertEqual(drafter.__name__, "Llama3Eagle3Drafter")
        self.assertEqual(config.__name__, "Eagle3Config")
        wrong = copy.deepcopy(SOURCE)
        wrong["files"]["draft/base_model.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source SHA"):
            official_classes(wrong)
        for module in (
            sys.modules[config.__module__],
            sys.modules[drafter.__bases__[0].__module__],
        ):
            with patch.object(module, "__file__", "/tmp/escaped-official-source.py"):
                with self.assertRaisesRegex(ValueError, "escaped"):
                    official_classes(SOURCE)

    def test_both_direct_widths_calibrated_sparse_fc_and_forbidden_update(self):
        provider = FixtureProvider()
        batch = next(provider.rounds())
        audit = audit_provider_round(batch, provider)
        for bits in (8, 1):
            model = dense_drafter(native_shape=True)
            # The four-wide fixture's original A1 head has an exact zero dot
            # in every row; one source bit avoids that degenerate scale VJP.
            with torch.no_grad():
                model.lm_head.weight[0, 0].mul_(-1)
            for p in model.parameters():
                p.requires_grad_(False)
            target = nn.Linear(4, 4)
            cfg = JointQATConfig(W1AxContract(bits), device="cpu", objective="hard_ce")
            linears = install_joint_linears(model, target, cfg)
            untouched = {
                name: tuple(p._version for p in m.parameters())
                for name, m in linears.items()
                if name != "fc"
            }
            fc = linears["fc"]
            fitted = (fc.latent_sign.detach().clone(), fc.initial_scale.detach().clone() * 0.8)
            apply_binary_initialization(linears, {"fc": fitted})
            self.assertTrue(torch.equal(fc.initial_scale, fitted[1]))
            self.assertEqual(
                untouched,
                {
                    name: tuple(p._version for p in m.parameters())
                    for name, m in linears.items()
                    if name != "fc"
                },
            )
            target_before = {name: p.detach().clone() for name, p in target.named_parameters()}
            optimizer = joint_optimizer(linears, cfg)
            adapter = NativeStepAdapter(model)
            with patch.object(torch.optim.AdamW, "step", side_effect=AssertionError("forbidden")):
                result = smoke(adapter, batch, audit, optimizer)
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertEqual(result["selected_gradient_tensors_finite_nonzero"], 18)
            self.assertTrue(result["parameter_versions_after_initialization_unchanged"])
            self.assertFalse(optimizer.state)
            for name, p in target.named_parameters():
                self.assertTrue(torch.equal(p, target_before[name]))

    def test_nonempty_optimizer_state_is_refused_before_forward(self):
        model = dense_drafter(native_shape=True)
        for p in model.parameters():
            p.requires_grad_(False)
        cfg = JointQATConfig(W1AxContract(8), device="cpu")
        linears = install_joint_linears(model, nn.Linear(4, 4), cfg)
        optimizer = joint_optimizer(linears, cfg)
        optimizer.state[next(iter(linears.values())).latent_sign] = {"step": torch.tensor(1.0)}
        with self.assertRaisesRegex(ValueError, "empty state"):
            smoke(NativeStepAdapter(model), None, None, optimizer)


if __name__ == "__main__":
    unittest.main()

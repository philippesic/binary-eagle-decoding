"""Lazy project provider tests with tiny files and injected model doubles."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import prepare_w1ax_train_manifest as prep  # noqa: E402
import train_joint_w1ax as train_joint  # noqa: E402
import w1ax_capture_provider as provider_module  # noqa: E402
from audit_recurrent_binary_capture import (  # noqa: E402
    CALIBRATION_CHECKS,
    CALIBRATION_ONLY_SCOPE,
    CALIBRATION_PINNED_INPUT_SHA256,
    CALIBRATION_READINESS_SCHEMA,
    validate_calibration_readiness_report,
)
from compact_w1a_teacher import compact  # noqa: E402
from test_recurrent_provider import TinyStep, dense_drafter, provider_round  # noqa: E402

from w1a1_eagle.recurrent_provider import train_from_provider  # noqa: E402
from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract  # noqa: E402
from w1a1_eagle.recurrent_trace import validate_recurrent_trace  # noqa: E402


def passing_checks(directory, digest, provider_inventory=None):
    results = {
        "pinned_inputs_and_response_ancestry": {
            "prompt_hash_match": True,
            "bundle_audit_hash_match": True,
            "model_hashes_match": True,
            "response_requests": 3,
            "response_exact_matches": 3,
        },
        "selected_root_mapping_and_operands": {
            "selected_roots": 6,
            "selected_roots_by_domain": {"prose": 2, "reasoning": 2, "code": 2},
            "available_outcomes_by_domain": {"prose": [], "reasoning": [], "code": []},
            "selected_outcomes_by_domain": {"prose": [], "reasoning": [], "code": []},
            **{
                field: True
                for field in (
                    "token_ids_exact",
                    "absolute_d2t_exact",
                    "decoder_positions_exact",
                    "causal_visibility_exact",
                    "cache_lengths_exact",
                    "finite_target_features",
                    "finite_student_logits",
                    "finite_gradients",
                    "native_projected_kv_matches_stored_f16",
                    "torch_cache_uses_f16_storage_rounding",
                    "embedding_rows_exact",
                    "hard_sign_bits_exact",
                    "row_scales_exact",
                )
            },
        },
        "student_native_proposal_and_response_agreement": {
            "shared_roots": 3,
            "top_choice_disagreements": 0,
            "changed_proposal_margins": [],
            "near_tie_disagreements": 0,
            "high_margin_changed_proposals": 0,
            "q4_response_ids_exact": True,
        },
        "student_native_numeric_tolerance": {
            "max_state_relative_rms": 0.01,
            "max_logits_relative_rms": 0.02,
            "native_head_replay_top_two_margins": [0.1, 0.2],
        },
        "provider_round_label_and_teacher_contract": {
            **(
                provider_inventory
                or {
                    "eligible_rounds": 100,
                    "supported_labels": 99,
                    "exact_prefix_joins": 100,
                    "compact_teacher_attached": False,
                }
            ),
        },
    }
    checks = {}
    for name, result in results.items():
        evidence_path = Path(directory) / f"{name}.json"
        evidence_path.write_text(json.dumps(result))
        checks[name] = {
            "status": "pass",
            "result": result,
            "evidence": [
                {
                    "path": str(evidence_path.resolve()),
                    "sha256": provider_module.sha256(evidence_path),
                }
            ],
        }
    return checks


class FakeCapture:
    def __init__(self):
        original = provider_round()
        rows = []
        for i, row in enumerate(original.rows):
            rows.append(dict(row, target_logits_row=i))
        self.round = SimpleNamespace(
            anchor=original.anchor,
            rows=tuple(rows),
            prefix_token_ids=original.prefix_token_ids,
            raw_target_features=original.raw_target_features,
            feature_positions=original.feature_positions,
        )
        self.anchors = {("p", 0): original.anchor}
        self.rows = {("p", 0): tuple(rows)}

    def round_inputs(self, prompt_id, round_index):
        if (prompt_id, round_index) != ("p", 0):
            raise KeyError((prompt_id, round_index))
        return self.round


class ProviderFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.capture_dir = self.root / "capture"
        self.capture_dir.mkdir()
        self.prompts = self.capture_dir / "train_prompts.jsonl"
        self.prompts.write_text(
            json.dumps({"id": "p", "messages": [{"role": "user", "content": "x"}]}) + "\n"
        )
        self.offsets = self.capture_dir / "offsets.npy"
        np.save(self.offsets, np.array([0, 1, 1], dtype=np.int64))
        self.absolute = self.root / "d2t.npy"
        np.save(self.absolute, np.array([0, 2, 3], dtype=np.int64))
        self.target = self.root / "target.gguf"
        self.candidate = self.root / "candidate.gguf"
        self.base = self.root / "draft-f16.gguf"
        for path, data in (
            (self.target, b"target"),
            (self.candidate, b"candidate"),
            (self.base, b"draft"),
        ):
            path.write_bytes(data)
        self.target_hash = provider_module.sha256(self.target)
        self.candidate_hash = provider_module.sha256(self.candidate)
        self.base_hash = provider_module.sha256(self.base)
        self.cell = self.capture_dir / "source_cell_manifest.json"
        self.cell.write_text(
            json.dumps({"target_sha256": self.target_hash, "draft_sha256": self.candidate_hash})
        )
        self.capture_manifest = self.capture_dir / "manifest.json"
        self._write_capture(False)
        self.model_manifest = self.root / "models.json"
        models = {}
        for role, repo, revision in (
            ("target", provider_module.TARGET_REPO, provider_module.TARGET_REVISION),
            ("draft", provider_module.DRAFT_REPO, provider_module.DRAFT_REVISION),
        ):
            directory = self.root / role
            directory.mkdir()
            file = directory / "tiny.safetensors"
            file.write_bytes(role.encode())
            models[role] = {
                "repo": repo,
                "revision": revision,
                "directory": str(directory),
                "files": [
                    {
                        "path": file.name,
                        "bytes": file.stat().st_size,
                        "sha256": provider_module.sha256(file),
                    }
                ],
            }
        self.model_manifest.write_text(json.dumps({"models": models}))
        self.capture_loader_calls = 0
        self.model_loader_calls = 0

    def _write_capture(self, eligible):
        value = {
            "schema": "recurrent_binary_capture_v1",
            "split": "train",
            "training_eligible": eligible,
            "readiness": "full_body_qat_eligible" if eligible else "preparation_only",
            "unverified_gates": [] if eligible else ["native_trajectory"],
            "pinned_source_artifact_hashes_verified": True,
            "prompts_sha256": provider_module.sha256(self.prompts),
            "target_vocab_size": 4,
            "draft_vocab_size": 3,
            "max_depth": 2,
            "offsets": {"path": self.offsets.name, "sha256": provider_module.sha256(self.offsets)},
            "source_report_sha256": {"cell_manifest": provider_module.sha256(self.cell)},
        }
        self.capture_manifest.write_text(json.dumps(value))

    def _capture_loader(self, *args, **kwargs):
        self.capture_loader_calls += 1
        self.assertEqual(kwargs["expected_prompt_count"], 1)
        return FakeCapture()

    def _model_loader(self, paths):
        self.model_loader_calls += 1
        return dense_drafter(), nn.Linear(4, 4)

    def _prepare(self, output, *, teacher_dir=None):
        return prep.prepare(
            self.capture_manifest,
            self.prompts,
            self.absolute,
            self.target,
            self.candidate,
            self.base,
            self.base_hash,
            self.model_manifest,
            "cap",
            output,
            teacher_dir=teacher_dir,
        )

    def _make_teacher(self):
        logits = self.root / "logits.npy"
        np.save(logits, np.array([[0.0, 0.1, 1.0, -0.2], [0.2, -0.1, 1.0, 0.0]], dtype=np.float32))
        rows = self.root / "teacher-rows.jsonl"
        records = [
            {
                "id": "r0",
                "prompt_id": "p",
                "capture_id": "cap",
                "logits_row": 0,
                "prefix_token_ids": [0, 1],
                "next_target_id": 2,
            },
            {
                "id": "r1",
                "prompt_id": "p",
                "capture_id": "cap",
                "logits_row": 1,
                "prefix_token_ids": [0, 1, 1],
                "next_target_id": 2,
            },
        ]
        rows.write_text("".join(json.dumps(row) + "\n" for row in records))
        output = self.root / "teacher"
        compact(
            logits,
            "npy",
            rows,
            self.absolute,
            output,
            target_vocab=4,
            topk=2,
            shard_rows=1,
            target_gguf_sha256=self.target_hash,
            prompts_sha256=provider_module.sha256(self.prompts),
            capture_manifest_sha256=provider_module.sha256(self.capture_manifest),
        )
        return output

    def test_ineligible_bundle_fails_before_capture_or_model_loader(self):
        with (
            patch.object(prep, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(prep, "CANDIDATE_D_SHA256", self.candidate_hash),
            patch.object(provider_module, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(provider_module, "CANDIDATE_D_SHA256", self.candidate_hash),
        ):
            manifest = self.root / "train.json"
            result = self._prepare(manifest)
            self.assertFalse(result["training_eligible"])
            with self.assertRaisesRegex(ValueError, "not eligible"):
                provider_module.NativeCaptureProvider(
                    JointQATConfig(W1AxContract(4)),
                    manifest,
                    capture_loader=self._capture_loader,
                    model_loader=self._model_loader,
                )
            tampered = json.loads(manifest.read_text())
            tampered["training_eligible"] = True
            manifest.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, "remains training-ineligible"):
                provider_module.NativeCaptureProvider(
                    JointQATConfig(W1AxContract(4)),
                    manifest,
                    capture_loader=self._capture_loader,
                    model_loader=self._model_loader,
                )
            self.assertEqual(self.capture_loader_calls, 0)
            self.assertEqual(self.model_loader_calls, 0)

    def test_eligible_compact_provider_binds_shards_then_loads_doubles(self):
        self._write_capture(True)
        with (
            patch.object(prep, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(prep, "CANDIDATE_D_SHA256", self.candidate_hash),
            patch.object(provider_module, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(provider_module, "CANDIDATE_D_SHA256", self.candidate_hash),
        ):
            teacher = self._make_teacher()
            manifest = self.root / "train.json"
            self._prepare(manifest, teacher_dir=teacher)
            config = JointQATConfig(W1AxContract(4), objective="compact_probability")
            provider = provider_module.NativeCaptureProvider(
                config,
                manifest,
                capture_loader=self._capture_loader,
                model_loader=self._model_loader,
                adapter_factory=TinyStep,
            )
            self.assertEqual(self.model_loader_calls, 0)
            self.assertEqual(len(list(provider.rounds())), 1)
            linears, metrics = train_from_provider(provider, config, max_rounds=1)
            self.assertEqual(len(linears), 9)
            self.assertGreater(metrics[0]["loss"], 0)
            self.assertEqual(self.model_loader_calls, 1)
            self.assertEqual(metrics[0]["gradient_tensors"], 18)

    def test_teacher_from_other_capture_fails_before_model_loader(self):
        self._write_capture(True)
        with (
            patch.object(prep, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(prep, "CANDIDATE_D_SHA256", self.candidate_hash),
            patch.object(provider_module, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(provider_module, "CANDIDATE_D_SHA256", self.candidate_hash),
        ):
            teacher = self._make_teacher()
            teacher_manifest = teacher / "manifest.json"
            content = json.loads(teacher_manifest.read_text())
            content["capture_manifest_sha256"] = "0" * 64
            teacher_manifest.write_text(json.dumps(content))
            manifest = self.root / "wrong-teacher.json"
            self._prepare(manifest, teacher_dir=teacher)
            with self.assertRaisesRegex(ValueError, "different native capture"):
                provider_module.NativeCaptureProvider(
                    JointQATConfig(W1AxContract(4), objective="compact_probability"),
                    manifest,
                    capture_loader=self._capture_loader,
                    model_loader=self._model_loader,
                )
            self.assertEqual(self.model_loader_calls, 0)

    def test_calibration_readiness_contract_requires_all_hashed_passing_checks(self):
        unresolved = ["native_trajectory"]
        digest = "a" * 64
        report = {
            "schema": CALIBRATION_READINESS_SCHEMA,
            "scope": CALIBRATION_ONLY_SCOPE,
            "training_eligible": False,
            "full_body_qat_eligible": False,
            "capture_manifest_sha256": digest,
            "unresolved_full_body_gates": unresolved,
            "relative_rms_definition": "rms_delta_over_max_rms_native_1e-8",
            "budget": {"steps": 100, "rounds": 100},
            "objective": "hard_ce",
            "contract": {"activation_bits": 16, "scale_layout": "row"},
            "inputs": {
                **{
                    name: digest
                    for name in (
                        "capture_manifest",
                        "prompts",
                        "absolute_d2t",
                        "target_gguf",
                        "candidate_d_gguf",
                        "model_snapshot_manifest",
                        "base_draft_gguf",
                        "native_trace",
                        "torch_numeric_report",
                    )
                },
                **CALIBRATION_PINNED_INPUT_SHA256,
            },
            "checks": {
                "pinned_inputs_and_response_ancestry": {
                    "status": "pass",
                    "evidence": [{"path": "/tmp/ancestry.json", "sha256": digest}],
                    "result": {
                        "prompt_hash_match": True,
                        "bundle_audit_hash_match": True,
                        "model_hashes_match": True,
                        "response_requests": 3,
                        "response_exact_matches": 3,
                    },
                },
                "selected_root_mapping_and_operands": {
                    "status": "pass",
                    "evidence": [{"path": "/tmp/roots.json", "sha256": digest}],
                    "result": {
                        "selected_roots": 6,
                        "selected_roots_by_domain": {"prose": 2, "reasoning": 2, "code": 2},
                        "available_outcomes_by_domain": {
                            "prose": ["accepted_continuation", "verifier_rejection"],
                            "reasoning": ["accepted_continuation", "verifier_rejection"],
                            "code": ["accepted_continuation", "verifier_rejection"],
                        },
                        "selected_outcomes_by_domain": {
                            "prose": ["accepted_continuation", "verifier_rejection"],
                            "reasoning": ["accepted_continuation", "verifier_rejection"],
                            "code": ["accepted_continuation", "verifier_rejection"],
                        },
                        **{
                            field: True
                            for field in (
                                "token_ids_exact",
                                "absolute_d2t_exact",
                                "decoder_positions_exact",
                                "causal_visibility_exact",
                                "cache_lengths_exact",
                                "finite_target_features",
                                "finite_student_logits",
                                "finite_gradients",
                                "native_projected_kv_matches_stored_f16",
                                "torch_cache_uses_f16_storage_rounding",
                                "embedding_rows_exact",
                                "hard_sign_bits_exact",
                                "row_scales_exact",
                            )
                        },
                    },
                },
                "student_native_proposal_and_response_agreement": {
                    "status": "pass",
                    "evidence": [{"path": "/tmp/proposals.json", "sha256": digest}],
                    "result": {
                        "shared_roots": 3,
                        "top_choice_disagreements": 0,
                        "changed_proposal_margins": [],
                        "near_tie_disagreements": 0,
                        "high_margin_changed_proposals": 0,
                        "q4_response_ids_exact": True,
                    },
                },
                "student_native_numeric_tolerance": {
                    "status": "pass",
                    "evidence": [{"path": "/tmp/numeric.json", "sha256": digest}],
                    "result": {
                        "max_state_relative_rms": 0.01,
                        "max_logits_relative_rms": 0.02,
                        "native_head_replay_top_two_margins": [0.1, 0.2],
                    },
                },
                "provider_round_label_and_teacher_contract": {
                    "status": "pass",
                    "evidence": [{"path": "/tmp/provider.json", "sha256": digest}],
                    "result": {
                        "eligible_rounds": 100,
                        "supported_labels": 99,
                        "exact_prefix_joins": 100,
                        "compact_teacher_attached": False,
                    },
                },
            },
        }
        validate_calibration_readiness_report(
            report,
            capture_manifest_sha256=digest,
            unresolved_full_body_gates=unresolved,
        )
        for mutate, message in (
            (
                lambda value: value["checks"].pop(next(iter(CALIBRATION_CHECKS))),
                "missing or unexpected checks",
            ),
            (
                lambda value: value["checks"]["pinned_inputs_and_response_ancestry"].__setitem__(
                    "status", "fail"
                ),
                "failed",
            ),
            (lambda value: value["budget"].__setitem__("steps", 99), "budget"),
            (lambda value: value.__setitem__("objective", "compact_probability"), "objective"),
            (lambda value: value["contract"].__setitem__("activation_bits", 8), "width"),
            (
                lambda value: value["checks"]["selected_root_mapping_and_operands"]["result"][
                    "available_outcomes_by_domain"
                ].__setitem__("prose", ["accepted", "rejected"]),
                "outcome labels for prose",
            ),
            (lambda value: value.__setitem__("full_body_qat_eligible", True), "scope"),
            (
                lambda value: value["checks"]["student_native_numeric_tolerance"][
                    "result"
                ].__setitem__("max_state_relative_rms", 0.11),
                "0.10",
            ),
            (
                lambda value: value["checks"]["student_native_proposal_and_response_agreement"][
                    "result"
                ].update(
                    top_choice_disagreements=1,
                    changed_proposal_margins=[0.021],
                    near_tie_disagreements=0,
                    high_margin_changed_proposals=1,
                ),
                "proposal measurements",
            ),
        ):
            changed = json.loads(json.dumps(report))
            mutate(changed)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                validate_calibration_readiness_report(
                    changed,
                    capture_manifest_sha256=digest,
                    unresolved_full_body_gates=unresolved,
                )

    def test_calibration_provider_keeps_full_body_bundle_ineligible(self):
        self._write_capture(False)
        with (
            patch.object(prep, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(prep, "CANDIDATE_D_SHA256", self.candidate_hash),
            patch.object(provider_module, "TARGET_GGUF_SHA256", self.target_hash),
            patch.object(provider_module, "CANDIDATE_D_SHA256", self.candidate_hash),
            patch.object(
                provider_module,
                "SHARD0000_CAPTURE_SHA256",
                provider_module.sha256(self.capture_manifest),
            ),
            patch.object(
                provider_module, "SHARD0000_PROMPTS_SHA256", provider_module.sha256(self.prompts)
            ),
        ):
            manifest_path = self.root / "calibration-provider.json"
            prepared = self._prepare(manifest_path)
            self.assertFalse(prepared["training_eligible"])
            source_hashes = prepared["sha256"]
            capture_data = json.loads(self.capture_manifest.read_text())
            digest = "b" * 64
            fake_capture = FakeCapture()
            fake_capture.anchors = {("p", index): fake_capture.round.anchor for index in range(100)}
            fake_capture.round_inputs = lambda prompt_id, round_index: fake_capture.round
            one_round_audit = validate_recurrent_trace(
                fake_capture.round.rows,
                [fake_capture.round.anchor],
                offsets=np.array([0, 1, 1], dtype=np.int64),
                target_vocab_size=4,
                draft_vocab_size=3,
                max_depth=2,
                allowed_prompt_ids={"p"},
                split="train",
            )
            one_round_supported = sum(one_round_audit.ce_mask)
            provider_inventory = {
                "eligible_rounds": 100 if one_round_supported else 0,
                "supported_labels": one_round_supported * 100,
                "exact_prefix_joins": len(fake_capture.round.feature_positions) * 100,
                "compact_teacher_attached": False,
            }
            report = {
                "schema": CALIBRATION_READINESS_SCHEMA,
                "scope": CALIBRATION_ONLY_SCOPE,
                "training_eligible": False,
                "full_body_qat_eligible": False,
                "capture_manifest_sha256": source_hashes["capture_manifest"],
                "unresolved_full_body_gates": capture_data["unverified_gates"],
                "relative_rms_definition": "rms_delta_over_max_rms_native_1e-8",
                "budget": {"steps": 100, "rounds": 100},
                "objective": "hard_ce",
                "contract": {"activation_bits": 16, "scale_layout": "row"},
                "inputs": {
                    **{
                        name: source_hashes[name]
                        for name in (
                            "capture_manifest",
                            "prompts",
                            "absolute_d2t",
                            "target_gguf",
                            "candidate_d_gguf",
                            "base_draft_gguf",
                            "model_snapshot_manifest",
                        )
                    },
                    **CALIBRATION_PINNED_INPUT_SHA256,
                    **{
                        name: digest
                        for name in (
                            "native_trace",
                            "torch_numeric_report",
                        )
                    },
                },
                "checks": passing_checks(self.root, digest, provider_inventory),
            }
            report_path = self.root / "calibration-readiness.json"
            report_path.write_text(json.dumps(report))
            spec = json.loads(manifest_path.read_text())
            spec["calibration_readiness"] = {
                "path": str(report_path.resolve()),
                "sha256": provider_module.sha256(report_path),
            }
            manifest_path.write_text(json.dumps(spec))

            def load_100_rounds(*args, **kwargs):
                self.capture_loader_calls += 1
                return fake_capture

            provider = provider_module.NativeCaptureProvider(
                JointQATConfig(W1AxContract(16)),
                manifest_path,
                capture_loader=load_100_rounds,
                model_loader=self._model_loader,
            )
            self.assertTrue(provider.training_eligible)
            self.assertFalse(provider.full_body_qat_eligible)
            self.assertEqual(provider.readiness_scope, CALIBRATION_ONLY_SCOPE)
            self.assertEqual(provider.total_rounds, 100)
            self.assertEqual(provider.source_metadata["full_body_qat_eligible"], False)
            self.assertEqual(len(list(provider.rounds())), 100)
            provider.validate_training_budget(JointQATConfig(W1AxContract(16)), 100)
            with self.assertRaisesRegex(ValueError, "steps 100"):
                provider.validate_training_budget(JointQATConfig(W1AxContract(16)), 99)
            with self.assertRaisesRegex(ValueError, "steps 100"):
                train_from_provider(provider, JointQATConfig(W1AxContract(16)), max_rounds=99)
            with self.assertRaisesRegex(ValueError, "max_rounds"):
                train_from_provider(provider, JointQATConfig(W1AxContract(16)), max_rounds=None)
            with self.assertRaisesRegex(ValueError, "all-rounds"):
                provider.validate_training_budget(
                    JointQATConfig(W1AxContract(16)), 100, all_rounds=True
                )
            fake_module = SimpleNamespace(factory=lambda config, path: provider)
            with (
                patch.object(train_joint, "importlib") as importlib_mock,
                patch.object(
                    sys,
                    "argv",
                    [
                        "train_joint_w1ax.py",
                        "--activation-bits",
                        "16",
                        "--steps",
                        "100",
                        "--provider",
                        "fake:factory",
                        "--provider-manifest",
                        str(manifest_path),
                        "--all-rounds",
                    ],
                ),
                self.assertRaisesRegex(ValueError, "all-rounds"),
            ):
                importlib_mock.import_module.return_value = fake_module
                train_joint.main()
            with (
                patch.object(train_joint, "importlib") as importlib_mock,
                patch.object(
                    sys,
                    "argv",
                    [
                        "train_joint_w1ax.py",
                        "--activation-bits",
                        "16",
                        "--steps",
                        "99",
                        "--provider",
                        "fake:factory",
                        "--provider-manifest",
                        str(manifest_path),
                    ],
                ),
                self.assertRaisesRegex(ValueError, "steps 100"),
            ):
                importlib_mock.import_module.return_value = fake_module
                train_joint.main()
            self.assertEqual(self.model_loader_calls, 0)


if __name__ == "__main__":
    unittest.main()

"""CPU-only collector fixtures. They never produce actual native/GPU evidence."""

import contextlib
import copy
import importlib.util
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "native_collector", SCRIPTS / "collect_qat_native_evidence.py"
)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)
common = collector.common


def f32(*values):
    return struct.pack(f"<{len(values)}f", *values).hex()


def context():
    return {
        "source_sha256": "a" * 64,
        "training_runtime": {"device_type": "cuda", "fixture_only": True},
        "recipe": {
            "activation_quantization": "learned",
            "fusion_correction": {"enabled": True, "rank": 1, "output_bias": False},
            "affine_weights": {"enabled": True, "coverage": "all"},
        },
        "native_commit": "c" * 40,
        "backend": "cuda",
        "hardware": {
            "device_type": "cuda",
            "name": "fixture metadata only",
            "compute_capability": [12, 0],
            "total_memory_bytes": 16 * 1024**3,
        },
    }


def operator_fixture():
    """Raw-byte interface simulation, explicitly not executed GPU evidence."""
    projections, packets, encoders = [], [], []
    boundaries = {
        "fc": "fc",
        "blk.0.attn_q": "qkv",
        "blk.0.attn_k": "qkv",
        "blk.0.attn_v": "qkv",
        "blk.0.attn_output": "attn_output",
        "blk.0.ffn_gate": "gate_up",
        "blk.0.ffn_up": "gate_up",
        "blk.0.ffn_down": "down",
        "output": "head",
    }
    for bits in (1, 4, 8):
        for base in sorted(common.AFFINE_BASES):
            index = len(packets)
            scalar = {
                "learned": True,
                "threshold_delta": 0.75 if bits == 1 else 0.0,
                "clip_ratio": 0.625 if bits != 1 else 1.0,
            }
            packets.append(
                {
                    "bits": bits,
                    "k": 3,
                    "n": 1,
                    "input_f32_hex": f32(1, 2, 3),
                    **scalar,
                    "expected_packed_hex": "0100000002000000",
                    "native_packed_hex": "0100000002000000",
                    "layout": {"total_words": 2},
                }
            )
            projections.append(
                {
                    "base": base,
                    "boundary": boundaries[base],
                    "bits": bits,
                    "k": 3,
                    "n": 1,
                    "pack_case_index": index,
                    **scalar,
                    "input_f32_hex": f32(1, 2, 3),
                    "alpha_f32_hex": f32(0, 0.125, 0.25),
                    "midpoint_f32_hex": f32(0.25, 0, -0.125),
                    "expected_codes_f32_hex": f32(1, 2, 3),
                    "native_codes_f32_hex": f32(1, 2, 3),
                    "expected_scale_f32_hex": f32(1),
                    "native_scale_f32_hex": f32(1),
                    "expected_sum_f32_hex": f32(6),
                    "native_sum_f32_hex": f32(6),
                    "expected_output_f32_hex": f32(1.5, 0.5, -0.75),
                    "native_output_f32_hex": f32(1.5, 0.5, -0.75),
                    "native_baseline_output_f32_hex": f32(0, 0.5, 0.25),
                    "native_zero_midpoint_output_f32_hex": f32(0, 0.5, 0.25),
                }
            )
        for mode in ("default", "correction_zero", "correction_rank1"):
            raw = f32(1, 2, 3)
            value = 1.5 if mode == "correction_rank1" else 1.0
            encoders.append(
                {
                    "bits": bits,
                    "mode": mode,
                    "correction_rank": 1,
                    "correction_bias": False,
                    "raw_input_f32_hex": raw,
                    "expected_output_f32_hex": f32(value, value),
                    "native_output_f32_hex": f32(value, value),
                    "correction_expected_f32": 0.5 if mode == "correction_rank1" else 0.0,
                    "nodes": [
                        {
                            "name": "fc_correction_latent",
                            "op": "MUL_MAT",
                            "device": "fixture CUDA0",
                            "sources": [{"index": 1, "type": "f32", "raw_hex": raw}],
                        },
                        {
                            "name": "fc_correction_delta",
                            "op": "MUL_MAT",
                            "device": "fixture CUDA0",
                            "output_f32_hex": f32(0.5, 0.5),
                        },
                    ],
                }
            )
    return {
        "schema_version": 1,
        "input_scope": "synthetic_operator",
        "status": "passed",
        "requested_backend": "CUDA",
        "byte_order": "little",
        "backend": {
            "registration": "CUDA",
            "device": "fixture CUDA0",
            "hardware": "fixture metadata only",
        },
        "runtime": {"build_commit": "c" * 9},
        "pack_cases": packets,
        "projection_cases": projections,
        "encoder_cases": encoders,
        "loader_cases": [
            {"expected_valid": False, "native_loaded": False, "execution_backend": "CPU"}
        ],
        "counters": {
            "pack_cases": len(packets),
            "projection_cases": len(projections),
            "graph_cases": len(encoders),
            "loader_cases": 1,
            "arithmetic_nodes": sum(len(c["nodes"]) for c in encoders),
        },
    }


class CollectorTests(unittest.TestCase):
    def test_default_plan_imports_no_runtime_or_provider_and_executes_no_command(self):
        with (
            patch.object(
                collector, "runtime_api", side_effect=AssertionError("model/backend import")
            ),
            patch.object(
                collector.subprocess, "Popen", side_effect=AssertionError("native command")
            ),
            patch.object(
                collector,
                "require_unpaused",
                side_effect=AssertionError("should not inspect active host"),
            ),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(collector.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "plan_only")
        self.assertFalse(result["cuda_discovered"])
        self.assertFalse(result["native_executed"])

    def test_human_pause_blocks_before_any_backend_or_model_work(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control.json"
            control.write_text(json.dumps({"rtx5080": {"pause_requested": True}}))
            with self.assertRaisesRegex(ValueError, "human pause"):
                collector.require_unpaused("rtx5080", control)
            control.write_text(json.dumps({"rtx5080": {"pause_requested": False}}))
            collector.require_unpaused("rtx5080", control)
            with self.assertRaisesRegex(ValueError, "missing"):
                collector.require_unpaused("rtx5080", Path(directory) / "missing")
            args = [
                "--collect",
                "--allow-cuda",
                "--bootstrap-context",
                str(Path(directory) / "not-read.json"),
                "--output",
                str(Path(directory) / "new-run"),
            ]
            with (
                patch.object(collector, "require_unpaused", side_effect=ValueError("human pause")),
                patch.object(
                    collector, "runtime_api", side_effect=AssertionError("CUDA discovery")
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(collector.main(args), 1)
            self.assertTrue((Path(directory) / "new-run/failure.json").exists())

    def test_raw_native_report_refuses_cpu_fallback_revision_hardware_and_incomplete_counts(self):
        report = operator_fixture()
        collector.validate_operator_report(report, context())
        for section, key, value, reason in (
            (None, "requested_backend", "CPU", "actual completed CUDA"),
            ("backend", "registration", "CPU", "actual completed CUDA"),
            ("backend", "hardware", "other hardware", "hardware differs"),
            ("runtime", "build_commit", "badcommit", "revision differs"),
            ("counters", "pack_cases", 999, "counters differ"),
        ):
            bad = copy.deepcopy(report)
            (bad if section is None else bad[section])[key] = value
            with self.assertRaisesRegex(ValueError, reason):
                collector.validate_operator_report(bad, context())
        report["pack_cases"][0]["native_packed_hex"] = "0200000002000000"
        with self.assertRaisesRegex(ValueError, "packed bytes differ"):
            collector.validate_operator_report(report, context())

    def test_actual_bytes_derive_operator_cases_without_invented_train_ancestry(self):
        report = operator_fixture()
        cases = collector.operator_cases(report, 4, context()["recipe"])
        self.assertEqual(len(cases["learned_quantizers"]), 9)
        self.assertEqual(len(cases["affine_weights"]), 27)
        self.assertEqual(len(cases["fusion_correction"]), 1)
        self.assertEqual({c["boundary"] for c in cases["learned_quantizers"]}, common.BOUNDARIES)
        for values in cases.values():
            for case in values:
                self.assertEqual(case["input_scope"], "synthetic_operator")
                self.assertFalse(set(case) & {"prompt_id", "capture_id", "round_index"})
        fusion = cases["fusion_correction"][0]
        self.assertGreater(fusion["native_correction_delta_norm"], 0)
        self.assertEqual(fusion["raw_input_sha256"], fusion["native_raw_input_sha256"])

    def test_affine_join_mu_identity_and_raw_fusion_failures_are_measured(self):
        for mutation, reason in (
            (lambda r: r["projection_cases"][0].update(pack_case_index=999), "join.*pack"),
            (lambda r: r["projection_cases"][0].update(native_sum_f32_hex=f32(7)), "sum differs"),
            (
                lambda r: r["projection_cases"][0].update(
                    native_zero_midpoint_output_f32_hex=f32(1, 0.5, 0.25)
                ),
                "zero-mu execution",
            ),
            (
                lambda r: r["projection_cases"][0].update(native_output_f32_hex=f32(0, 0.5, -0.75)),
                "RMS.*numeric gate",
            ),
            (
                lambda r: r["encoder_cases"][2]["nodes"][0]["sources"][0].update(
                    raw_hex=f32(3, 2, 1)
                ),
                "different raw FC",
            ),
            (
                lambda r: r["encoder_cases"][2]["nodes"][1].update(device="CPU"),
                "actual requested backend",
            ),
        ):
            with self.subTest(reason=reason):
                report = operator_fixture()
                mutation(report)
                with self.assertRaisesRegex(ValueError, reason):
                    collector.operator_cases(report, 1, context()["recipe"])

    def test_synthetic_scope_is_accepted_only_for_operators_and_never_fake_train_ids(self):
        bound = context()
        deployments = {"A4": "4" * 64}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            decision = root / "decisions.json"
            common.write_new(
                decision,
                {
                    "schema": common.MEASUREMENT_SCHEMA,
                    "kind": "native_decisions",
                    "split": "train",
                    "activation_bits": 4,
                    "deployment_state_sha256": deployments["A4"],
                    **bound,
                    "cases": [
                        {
                            "prompt_id": "actual-train",
                            "capture_id": "actual-capture",
                            "round_index": 0,
                            "torch_choice": 2,
                            "native_choice": 2,
                            "native_margin": 0.1,
                            "state_relative_rms": 0.01,
                            "logit_relative_rms": 0.01,
                        }
                    ],
                },
            )
            evidence = collector.collect_evidence(
                bound,
                deployments,
                {"A4": collector.file_record(decision)},
                operator_fixture(),
                root,
                {"actual-train"},
                fixture_only=True,
            )
            self.assertTrue(evidence["fixture_only"])
            # Validator fixtures are explicitly forbidden as CUDA admission.
            with self.assertRaisesRegex(ValueError, "real zero-update"):
                common.validate_native_evidence(
                    evidence, root, bound, deployments, {"actual-train"}, expected_lanes=("A4",)
                )
            # Test only scope rules by calling the validator in memory; no
            # nonfixture file or receipt is emitted by this fixture.
            candidate = copy.deepcopy(evidence)
            candidate["fixture_only"] = False
            self.assertTrue(
                common.validate_native_evidence(
                    candidate, root, bound, deployments, {"actual-train"}, expected_lanes=("A4",)
                )["affine_weights"]["passed"]
            )
            record = candidate["lanes"]["A4"]["learned_quantizers"]
            path = Path(record["path"])
            artifact = common.read_json(path)
            artifact["cases"][0]["prompt_id"] = "invented"
            path.write_text(json.dumps(artifact))
            record["sha256"] = common.sha256(path)
            with self.assertRaisesRegex(ValueError, "fabricate train ancestry"):
                common.validate_native_evidence(
                    candidate, root, bound, deployments, {"actual-train"}, expected_lanes=("A4",)
                )

    def test_file_identity_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file.json"
            path.write_text("{}")
            record = collector.file_record(path)
            self.assertEqual(collector.checked_path(record), path.resolve())
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "identity differs"):
                collector.checked_path(record)
            with self.assertRaises(FileExistsError):
                common.write_new(path, {})

    def test_old_native_runtime_cannot_be_relabelled_as_selected_published_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "runtime.json"
            manifest.write_text(
                json.dumps({"native_commit": "c" * 40, "binary": {"sha256": "b" * 64}})
            )
            sources = {
                "native_runtime": {"immutable_manifest": collector.file_record(manifest)},
                "sha256": {"binary": "b" * 64},
            }
            collector.verify_native_revision(sources, "c" * 40)
            with self.assertRaisesRegex(ValueError, "selected published commit"):
                collector.verify_native_revision(sources, "d" * 40)
            sources["sha256"]["binary"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "server binary/build manifest"):
                collector.verify_native_revision(sources, "c" * 40)

    def test_bounded_cpu_subprocess_and_failure_do_not_claim_gpu_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cpu-only.log"
            result = collector.run_bounded(
                [sys.executable, "-c", "print('CPU fixture only')"], log, timeout_seconds=2
            )
            self.assertEqual(result["exit_code"], 0)
            self.assertIn("CPU fixture only", log.read_text())
            with self.assertRaises(FileExistsError):
                collector.run_bounded([sys.executable, "-c", "pass"], log, timeout_seconds=2)
            fake = SimpleNamespace(
                pid=12345, returncode=None, poll=lambda: None, wait=lambda timeout=None: None
            )
            with (
                patch.object(collector.subprocess, "Popen", return_value=fake),
                patch.object(collector.os, "killpg") as kill,
                self.assertRaisesRegex(RuntimeError, "time/log cap"),
            ):
                collector.run_bounded(
                    ["fixture-not-executed"], Path(directory) / "failed.log", timeout_seconds=0
                )
            self.assertTrue(kill.called)


if __name__ == "__main__":
    unittest.main()

"""Synthetic external producer/device fixtures cannot grant production admission."""

import argparse
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_block_capture_portability as gate  # noqa: E402
import test_block_data as fixtures  # noqa: E402

from w1a1_eagle.block_data import BlockDataset, file_sha256  # noqa: E402


class PortabilityTests(unittest.TestCase):
    import_ = fixtures.NativeRawImportTests.import_

    def setUp(self):
        fixtures.NativeRawImportTests.setUp(self)
        self.binary = self.root / "fake-native-binary"
        self.target = self.root / "fake-target"
        self.binary.write_bytes(b"explicit synthetic producer fixture")
        self.target.write_bytes(b"explicit synthetic target fixture")
        for chain in self.plan["chains"]:
            record = chain["native_receipt"]
            path = Path(record["path"])
            receipt = json.loads(path.read_text())
            receipt["target_sha256"] = file_sha256(self.target)
            receipt["producer_binary_sha256"] = file_sha256(self.binary)
            path.write_text(json.dumps(receipt))
            record["sha256"] = file_sha256(path)
        self.plan_path.write_text(json.dumps(self.plan))
        self.manifest = self.import_()
        self.dataset = BlockDataset(self.manifest, expected_sha256=file_sha256(self.manifest))
        # Import already produced the one mandatory initial admission.
        self.admission = self.root / "materialized/completed-admission.json"
        self.args = argparse.Namespace(
            manifest=self.manifest,
            manifest_sha256=file_sha256(self.manifest),
            admission=self.admission,
            admission_sha256=file_sha256(self.admission),
            binary=self.binary,
            binary_sha256=file_sha256(self.binary),
            target=self.target,
            producer_source_revision="a" * 40,
            output_root=self.root / "fresh",
            output=self.root / "result.json",
            expected_compute_capability=[7, 5],
            max_tokens=12,
            max_cases=3,
            gpu_layers=999,
            timeout_seconds=3,
            feature_atol=0.002,
            feature_rtol=0.002,
            logit_atol=0.02,
            logit_rtol=0.002,
        )
        self.device = {
            "name": "synthetic CUDA fixture",
            "compute_capability": [7, 5],
            "uuid": "synthetic",
        }
        self.instances = []

    def teacher(self, *, mutation=None, stop=False, bad_cleanup=False, cpu=False):
        owner = self

        class Teacher:
            def __init__(self, *args, **kwargs):
                self.closed = False
                owner.instances.append(self)

            def capture_prefix(
                self, tokens, taps, *, logits_mode, chain_ancestry, decode_history=None
            ):
                if stop:
                    raise InterruptedError("synthetic STOP")
                cid = chain_ancestry["prompt_id"]
                _, fresh_features, fresh_logits = owner.dataset.copy_golden_prefix(cid, len(tokens))
                if mutation:
                    mutation(fresh_features, fresh_logits)
                root = owner.root / f"fresh-{cid}"
                root.mkdir()
                files = {}
                for name, array in (("features", fresh_features), ("logits", fresh_logits)):
                    path = root / (name + ".f32")
                    array.tofile(path)
                    files[name] = {
                        "path": str(path),
                        "sha256": file_sha256(path),
                        "shape": list(array.shape),
                        "dtype": "float32",
                    }
                return {
                    "schema": "block_native_teacher_request_v1",
                    "complete": True,
                    "optimizer_updates": 0,
                    "decode_history": decode_history,
                    "target_sha256": file_sha256(owner.target),
                    "target_precision": "F16",
                    "kv_type": "F16",
                    "producer_binary_sha256": file_sha256(owner.binary),
                    "producer_source_revision": "a" * 40,
                    "teacher_context_reset_between_requests": True,
                    "prefix_contract": "teacher_forced_exact_caller_token_ids",
                    "prefix_freshness": "caller_current_student_prefix",
                    "tap_ids": list(taps),
                    "gpu_layers": 999,
                    "hardware": [owner.device["name"]],
                    "executed_result_buffers": ["CPU" if cpu else "CUDA0"],
                    "target_storage_buffers": {"CUDA0": {"tensor_count": 42}},
                    "tokens": tokens,
                    "chain_ancestry": chain_ancestry,
                    "files": files,
                }

            @staticmethod
            def array(receipt, name):
                record = receipt["files"][name]
                return np.memmap(
                    record["path"], dtype=np.float32, mode="r", shape=tuple(record["shape"])
                )

            def close(self):
                self.closed = not bad_cleanup
                if bad_cleanup:
                    raise RuntimeError("synthetic cleanup failure")

        return Teacher

    def run_(self, **kwargs):
        return gate.run_gate(
            self.args, teacher_factory=self.teacher(**kwargs), device_query=lambda: self.device
        )

    def test_synthetic_numeric_success_cannot_grant_production(self):
        report = self.run_()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["artifact_kind"], "synthetic_fixture")
        self.assertEqual(len(report["numeric_checks"]), 3)
        self.assertTrue(self.instances[0].closed)

    def test_numeric_failure_preserves_and_closes(self):
        report = self.run_(mutation=lambda features, logits: features.__setitem__((0, 0, 0), 1000))
        self.assertEqual(report["status"], "FAIL")
        self.assertGreater(report["numeric_checks"][0]["features"]["out_of_tolerance_values"], 0)
        self.assertTrue(self.instances[0].closed)
        self.assertTrue(self.args.output.exists())

    def test_target_argmax_change_fails_even_with_tolerance(self):
        self.args.logit_atol = 1e9
        report = self.run_(mutation=lambda features, logits: logits.__setitem__((0, 0), 10000))
        self.assertEqual(report["status"], "FAIL")
        self.assertTrue(report["numeric_checks"][0]["decision_changed"])

    def test_stop_cleans_producer(self):
        report = self.run_(stop=True)
        self.assertEqual(report["failure"]["type"], "InterruptedError")
        self.assertTrue(self.instances[0].closed)

    def test_cleanup_failure_rejects_numeric_pass(self):
        report = self.run_(bad_cleanup=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("cleanup", report["failure"]["message"])

    def test_cpu_actual_execution_cannot_grant_cuda(self):
        report = self.run_(cpu=True)
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("CUDA", report["failure"]["message"])

    def test_other_device_compute_capability_refuses_before_teacher(self):
        self.args.expected_compute_capability = [12, 0]
        with self.assertRaisesRegex(ValueError, "compute capability"):
            self.run_()
        self.assertEqual(self.instances, [])

    def test_numeric_nan_and_shape_refuse(self):
        gold = np.ones((2, 4), dtype=np.float32)
        current = gold.copy()
        current[0, 0] = np.nan
        self.assertEqual(gate.compare_matrix(gold, current, atol=0.1, rtol=0.1)["status"], "FAIL")
        with self.assertRaises(ValueError):
            gate.compare_matrix(gold, current[:, :2], atol=0.1, rtol=0.1)

    def eagle_goldens(self):
        cases = []
        for cid, chain in self.dataset.chains.items():
            receipt = json.loads(Path(chain["native_receipt"]["path"]).read_text())
            tokens, features, logits = self.dataset.copy_golden_prefix(cid, 3)
            files = {}
            for name, array in (
                ("features", np.asarray(features[:3, :3])),
                ("logits", logits),
            ):
                path = self.root / f"eagle-{cid}-{name}.f32"
                np.ascontiguousarray(array).tofile(path)
                files[name] = {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "shape": list(array.shape),
                    "dtype": "float32",
                }
            receipt.update(
                decode_history=[
                    {
                        "offset": 0,
                        "count": 3,
                        "phase": "prefill",
                        "kv_reused_from_same_chain": False,
                    }
                ],
                tokens=list(tokens[:3]),
                tap_ids=list(gate.EAGLE_TAPS),
                features_shape=[3, 3, 2],
                logits_shape=[1, 32],
                logits_mode="last",
                files=files,
                executed_result_buffers=["CUDA0"],
                hardware=[self.device["name"]],
            )
            receipt_path = self.root / f"eagle-{cid}-receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            cases.append(
                {
                    "chain_id": cid,
                    "native_receipt": {
                        "path": str(receipt_path),
                        "sha256": file_sha256(receipt_path),
                    },
                }
            )
        manifest = {
            "schema": "nine_model_train_capture_goldens_v1",
            "family": "eagle",
            "tap_ids": list(gate.EAGLE_TAPS),
            "vocab_size": 32,
            "target_width": 2,
            "target_sha256": file_sha256(self.target),
            "train_inventory": self.plan["train_inventory"],
            "cases": cases,
        }
        path = self.root / "eagle-goldens.json"
        path.write_text(json.dumps(manifest))
        return path

    def test_new_eagle_three_tap_goldens_exact_api_and_cleanup(self):
        path = self.eagle_goldens()
        self.dataset = gate.NativeCaptureGoldens(path, expected_sha256=file_sha256(path))
        self.args.manifest, self.args.manifest_sha256 = path, file_sha256(path)
        self.args.admission = self.args.admission_sha256 = None
        report = gate.run_gate(
            self.args,
            teacher_factory=self.teacher(),
            device_query=lambda: self.device,
            dataset_factory=gate.NativeCaptureGoldens,
            expected_family="eagle",
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["family"], "eagle")
        self.assertEqual(report["artifact_kind"], "synthetic_fixture")
        self.assertEqual(report["numeric_checks"][0]["features"]["shape"], [3, 3, 2])

    def test_speculative_tree_capture_cannot_be_relabelled_golden(self):
        path = self.eagle_goldens()
        manifest = json.loads(path.read_text())
        record = manifest["cases"][0]["native_receipt"]
        receipt_path = Path(record["path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["prefix_contract"] = "speculative_tree_captured_prefix"
        receipt_path.write_text(json.dumps(receipt))
        record["sha256"] = file_sha256(receipt_path)
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "provenance"):
            gate.NativeCaptureGoldens(path, expected_sha256=file_sha256(path))

    def test_generated_metadata_and_bad_freshness_cannot_be_relabelled_golden(self):
        path = self.eagle_goldens()
        manifest = json.loads(path.read_text())
        record = manifest["cases"][0]["native_receipt"]
        receipt_path = Path(record["path"])
        original = json.loads(receipt_path.read_text())
        # The valid fixture mirrors NativeTeacher.capture_prefix, with no generation
        # payload and caller_current_student_prefix freshness.
        gate.NativeCaptureGoldens(path, expected_sha256=file_sha256(path))
        for field, value in (
            ("prefix_freshness", "native_generated_chain"),
            ("prefix_freshness", None),
            ("generation", {}),
            ("prompt", {}),
            ("prompt_source_sha256", "e" * 64),
        ):
            receipt_path.write_text(json.dumps(original | {field: value}))
            record["sha256"] = file_sha256(receipt_path)
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "provenance"):
                gate.NativeCaptureGoldens(path, expected_sha256=file_sha256(path))

    def test_runtime_replay_rejects_generated_metadata_and_bad_freshness(self):
        receipt = dict(
            next(iter(self.dataset.native_receipts.values())),
            gpu_layers=999,
            executed_result_buffers=["CUDA0"],
            target_storage_buffers={"CUDA0": 1},
            hardware=[self.device["name"]],
        )
        kwargs = {"binary_sha256": file_sha256(self.binary), "source_revision": "a" * 40}
        gate.check_producer(receipt, self.dataset, self.device, **kwargs)
        for field, value in (
            ("prefix_freshness", "native_generated_chain"),
            ("generation", {}),
            ("prompt", {}),
            ("prompt_source_sha256", "e" * 64),
        ):
            with self.assertRaisesRegex(ValueError, "producer lacks"):
                gate.check_producer(receipt | {field: value}, self.dataset, self.device, **kwargs)

    def test_new_golden_storage_budget_refuses(self):
        path = self.eagle_goldens()
        with self.assertRaisesRegex(MemoryError, "storage"):
            gate.NativeCaptureGoldens(path, expected_sha256=file_sha256(path), max_capture_bytes=10)

    def test_eagle_three_tap_pin_cannot_accept_five_tap(self):
        path = self.eagle_goldens()
        manifest = json.loads(path.read_text())
        manifest["tap_ids"] = [2, 10, 18, 26, 34]
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "taps"):
            gate.NativeCaptureGoldens(path, expected_sha256=file_sha256(path))

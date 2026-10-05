"""Independent negative tests for prelaunch readiness versus campaign outcomes."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path(os.environ.get("NINE_MODEL_CODE_ROOT", ROOT)).resolve()
CPU_CAPTURE_PATH = Path(
    os.environ.get(
        "NINE_MODEL_CPU_CAPTURE_SCRIPT", CODE_ROOT / "scripts/capture_nine_model_train_data.py"
    )
).resolve()
ACTUAL_CPU_CAPTURE = Path(
    os.environ.get(
        "NINE_MODEL_ACTUAL_CPU_CAPTURE",
        "/Users/pippo/github/binary-eagle-decoding/results/nine-model-qat-preparation/"
        "development-cpu-pilot-capture-20261004-01",
    )
).resolve()
ACTUAL_CPU_PLAN = Path(
    os.environ.get(
        "NINE_MODEL_ACTUAL_CPU_PLAN",
        "/Users/pippo/github/binary-eagle-decoding/results/nine-model-qat-preparation/"
        "development-cpu-pilot-plan-20261004-01",
    )
).resolve()
spec = importlib.util.spec_from_file_location(
    "qa_prelaunch_bundle_builder", CODE_ROOT / "scripts/prepare_nine_model_bundle.py"
)
builder = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(builder)
from check_block_capture_portability import NativeCaptureGoldens  # noqa: E402

cpu_spec = importlib.util.spec_from_file_location("qa_cpu_capture", CPU_CAPTURE_PATH)
cpu_capture = importlib.util.module_from_spec(cpu_spec)
assert cpu_spec.loader is not None
cpu_spec.loader.exec_module(cpu_capture)

from w1a1_eagle.nine_model_admission import backward_summary, portability_summary  # noqa: E402


def _profile(status: str, scope: str) -> dict:
    required = ("source", "initial_export") if status == "q4" else builder.PORTABLE
    return {
        "status": "PENDING",
        "prelaunch_status": "PASS",
        "prelaunch_requirements": {
            name: {"status": "PASS", "evidence": [{"scope": scope, "artifact_kind": "production"}]}
            for name in required
        },
    }


class PrelaunchLedgerContracts(unittest.TestCase):
    def test_current_ledger_truthfully_keeps_every_prelaunch_and_campaign_gate_pending(self):
        ledger = json.loads(
            (ROOT / "experiments/nine-model-qat-preparation/qa-ledger.json").read_text()
        )
        self.assertTrue(
            all(record["status"] == "PENDING" for record in ledger["profiles"].values())
        )
        self.assertTrue(
            all(
                record.get("prelaunch_status") == "PENDING"
                for record in ledger["profiles"].values()
            )
        )
        pending = builder.ledger_pending(ledger)
        for profile, record in ledger["profiles"].items():
            self.assertIn(profile + ": prelaunch production dependencies PENDING", pending)
            if not profile.endswith("q4"):
                self.assertIn(profile + ": data PENDING", pending)
                self.assertIn(profile + ": calibration_initialization PENDING", pending)
                self.assertIn(profile + ": initial_export PENDING", pending)

    def test_aggregate_campaign_pass_cannot_replace_missing_prelaunch_contract(self):
        ledger = {"schema": "nine_model_qa_ledger_v1", "profiles": {}}
        for profile in builder.CELLS:
            ledger["profiles"][profile] = {
                "status": "PASS",
                "requirements": {
                    name: {"status": "PASS", "evidence": [{"scope": "actual production"}]}
                    for name in builder.PORTABLE
                },
            }
        pending = builder.ledger_pending(ledger)
        for profile in builder.CELLS:
            self.assertIn(profile + ": prelaunch production dependencies PENDING", pending)

    def test_campaign_outcomes_can_remain_pending_after_prelaunch_gates_pass(self):
        ledger = {
            "schema": "nine_model_qa_ledger_v1",
            "fresh_sm120": {"status": "PENDING"},
            "profiles": {
                profile: _profile(
                    "q4" if profile.endswith("q4") else "candidate",
                    "actual production model and data",
                )
                for profile in builder.CELLS
            },
        }
        self.assertEqual(builder.ledger_pending(ledger), [])
        self.assertTrue(
            all(record["status"] == "PENDING" for record in ledger["profiles"].values())
        )
        self.assertEqual(ledger["fresh_sm120"]["status"], "PENDING")

    def test_development_synthetic_receipts_cannot_satisfy_actual_prelaunch_scopes(self):
        ledger = {
            "schema": "nine_model_qa_ledger_v1",
            "profiles": {
                profile: _profile(
                    "q4" if profile.endswith("q4") else "candidate",
                    "actual_source_model_cpu_synthetic_inputs",
                )
                for profile in builder.CELLS
            },
        }
        pending = builder.ledger_pending(ledger)
        self.assertIn("dspark_a8: data lacks explicit actual production evidence scope", pending)
        self.assertIn(
            "dflash_a1: calibration_initialization lacks explicit actual production evidence scope",
            pending,
        )
        self.assertIn(
            "eagle_a8: hard_forward lacks explicit actual production evidence scope", pending
        )
        self.assertIn(
            "dflash_q4: initial_export lacks explicit actual production evidence scope", pending
        )

    def test_cpu_development_goldens_and_receipts_cannot_grant_sm120_gates(self):
        cpu_golden = {
            "schema": "nine_model_cpu_development_goldens_v1",
            "family": "dspark",
            "tap_ids": [2, 10, 18, 26, 34],
            "vocab_size": 32,
            "target_width": 32,
            "target_sha256": "a" * 64,
            "train_inventory": {"path": "missing-inventory.json", "sha256": "b" * 64},
            "cases": [],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cpu-goldens.json"
            path.write_text(json.dumps(cpu_golden))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "unsupported bounded native TRAIN golden"):
                NativeCaptureGoldens(path, expected_sha256=digest)

        cpu_portability = {
            "schema": "nine_model_capture_portability_v1",
            "status": "PASS",
            "artifact_kind": "development_CPU",
            "family": "dspark",
            "target_sha256": "a" * 64,
            "compute_capability": None,
            "optimizer_updates": 0,
            "producer_closed": True,
            "producer_returncode": 0,
            "numeric_checks": [],
        }
        with self.assertRaises(ValueError):
            portability_summary(cpu_portability, "dspark", "a" * 64)

        cpu_backward = {
            "schema": "nine_model_model_smoke_v1",
            "status": "PASS",
            "artifact_kind": "development_CPU",
            "bundle_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "source": {"actual_source": "e" * 64, "bundle_sha256": "c" * 64},
            "optimizer_updates": 0,
            "hardware": {"compute_capability": None, "gpu_uuid": None},
        }
        candidate = {
            "family": "dspark",
            "final_bits": 8,
            "config": {"sha256": "d" * 64},
            "source_bindings": {"actual_source": "e" * 64},
        }
        with self.assertRaises(ValueError):
            backward_summary(cpu_backward, candidate, "c" * 64, gpu_uuid=None)

    @staticmethod
    def _cpu_build_proof(root: Path, extra_library: str | None = None) -> tuple[dict, dict]:
        binary = root / "test-block-binary"
        binary.write_bytes(b"pinned CPU executable fixture")
        cache = root / "CMakeCache.txt"
        cache.write_text(
            "GGML_CPU:BOOL=ON\n"
            + "".join(f"{name}:BOOL=OFF\n" for name in cpu_capture.CPU_OFF_OPTIONS)
        )
        libraries = []
        for name in (
            "libllama.0.dylib",
            "libggml.0.dylib",
            "libggml-base.0.dylib",
            "libggml-cpu.0.dylib",
        ):
            path = root / name
            path.write_bytes((name + " fixture").encode())
            libraries.append({"path": str(path), "sha256": cpu_capture.file_sha256(path)})
        links = root / "otool-L.txt"
        link_text = "\n".join("@rpath/" + Path(item["path"]).name for item in libraries)
        if extra_library is not None:
            link_text += "\n@rpath/" + extra_library
        links.write_text(link_text)
        proof = {
            "schema": "nine_model_cpu_teacher_build_v1",
            "binary_sha256": cpu_capture.file_sha256(binary),
            "source_revision": "a" * 40,
            "cmake_cache": {"path": str(cache), "sha256": cpu_capture.file_sha256(cache)},
            "dylibs": libraries,
            "otool_links": {"path": str(links), "sha256": cpu_capture.file_sha256(links)},
            "library_directory": str(root),
        }
        proof_path = root / "cpu-build.json"
        proof_path.write_text(json.dumps(proof))
        record = {"path": str(proof_path), "sha256": cpu_capture.file_sha256(proof_path)}
        native = {
            "binary": {"path": str(binary), "sha256": proof["binary_sha256"]},
            "source_revision": proof["source_revision"],
        }
        return record, native

    def test_cpu_build_and_loaded_runtime_refuse_any_gpu_backend_dylib(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            record, native = self._cpu_build_proof(root, "libggml-hip.0.dylib")
            with self.assertRaises(ValueError):
                cpu_capture.validate_cpu_build(record, native, root)

            record, native = self._cpu_build_proof(root)
            build = cpu_capture.validate_cpu_build(record, native, root)
            log = root / "dyld.log"
            pid = 42
            lines = [f"dyld[{pid}]: loaded: {item['path']}" for item in build["dylibs"]]
            lines.append(f"dyld[{pid}]: loaded: {root}/libggml-hip.0.dylib")
            log.write_text("\n".join(lines))
            teacher = type(
                "Teacher",
                (),
                {
                    "log": type("Log", (), {"name": str(log)})(),
                    "process": type("Proc", (), {"pid": pid})(),
                },
            )()
            with self.assertRaises(ValueError):
                cpu_capture.cpu_loaded_library_proof(teacher, build)

    def test_preserved_cpu_attempt_receipt_requires_one_exact_registered_device(self):
        receipt_path = ACTUAL_CPU_CAPTURE / "native/4b6db6afd87b43b39693b957cf56cdea/receipt.json"
        plan_path = ACTUAL_CPU_PLAN / "plan.json"
        if not receipt_path.is_file() or not plan_path.is_file():
            self.skipTest("preserved development_CPU attempt 01 is unavailable")
        receipt = json.loads(receipt_path.read_text())
        plan = json.loads(plan_path.read_text())
        self.assertEqual(receipt["hardware"], ["Apple M3 Max"])
        self.assertEqual(receipt["gpu_layers"], 0)
        self.assertEqual(receipt["executed_result_buffers"], ["CPU"])
        self.assertEqual(set(receipt["target_storage_buffers"]), {"CPU_Mapped"})
        # Attempt 01 predates the current prefix-contract label. Adapt only this
        # unrelated label so this test reaches the CPU device-list guard.
        receipt["prefix_contract"] = "teacher_forced_exact_caller_token_ids"
        with tempfile.TemporaryDirectory() as folder:
            for name, descriptor in receipt["files"].items():
                path = Path(folder) / f"{name}.f32"
                with path.open("wb") as stream:
                    stream.truncate(math.prod(descriptor["shape"]) * 4)
                descriptor["path"] = str(path)
                descriptor["sha256"] = cpu_capture.file_sha256(path)
            device = {"name": "Apple M3 Max", "backend": "CPU", "compute_capability": None}
            cpu_capture.validate_replay(
                receipt,
                receipt["tokens"],
                receipt["tap_ids"],
                plan["native"],
                device,
                execution_profile="development_CPU",
            )
            for bad_hardware in (["Apple M3 Max", "Apple M3 Max"], ["Apple M3 Max", "CPU"]):
                mixed = json.loads(json.dumps(receipt))
                mixed["hardware"] = bad_hardware
                with self.assertRaises(ValueError):
                    cpu_capture.validate_replay(
                        mixed,
                        mixed["tokens"],
                        mixed["tap_ids"],
                        plan["native"],
                        device,
                        execution_profile="development_CPU",
                    )

    def test_attempt_01_dyld_closure_matches_four_pins_and_records_system_metal(self):
        plan_path = ACTUAL_CPU_PLAN / "plan.json"
        proof_path = ACTUAL_CPU_PLAN / "cpu-build.json"
        log_path = ACTUAL_CPU_CAPTURE / "native/producer-cb97282110964487bfa440391b9f5837.log"
        if not all(path.is_file() for path in (plan_path, proof_path, log_path)):
            self.skipTest("preserved development_CPU attempt 01 runtime proof is unavailable")
        plan = json.loads(plan_path.read_text())
        proof_record = {"path": str(proof_path), "sha256": cpu_capture.file_sha256(proof_path)}
        build = cpu_capture.validate_cpu_build(proof_record, plan["native"], ACTUAL_CPU_PLAN)
        teacher = types.SimpleNamespace(
            log=types.SimpleNamespace(name=str(log_path)),
            process=types.SimpleNamespace(pid=96621),
        )
        runtime = cpu_capture.cpu_loaded_library_proof(teacher, build)
        self.assertTrue(runtime["actual_dyld_paths_checked"])
        self.assertFalse(runtime["actual_linux_mapping_checked"])
        self.assertEqual(
            {Path(item["path"]).name.split(".")[0] for item in runtime["libraries"]},
            {"libllama", "libggml", "libggml-cpu", "libggml-base"},
        )
        self.assertTrue(runtime["observed_system_metal_framework"])
        with self.assertRaises(ValueError):
            cpu_capture.require_cpu_project_libraries("/private/build/libggml-metal.dylib")
        with tempfile.TemporaryDirectory() as folder:
            hostile_log = Path(folder) / "qa-extra-ggml-dyld.log"
            hostile_log.write_text(
                log_path.read_text() + "\ndyld[96621]: /private/build/libggml-metal.0.dylib\n"
            )
            hostile = types.SimpleNamespace(
                log=types.SimpleNamespace(name=str(hostile_log)),
                process=types.SimpleNamespace(pid=96621),
            )
            with self.assertRaises(ValueError):
                cpu_capture.cpu_loaded_library_proof(hostile, build)


if __name__ == "__main__":
    unittest.main()

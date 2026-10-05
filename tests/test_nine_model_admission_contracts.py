"""Independent source, precision-stage, and release checks for SM120 admission."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from w1a1_eagle.nine_model_admission import (
    Admission,
    backward_summary,
    native_summary,
    validate_plan,
)
from w1a1_eagle.nine_model_pipeline import CANDIDATES, Files, atomic_json, sha256


def _locator(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _plan(root: Path) -> dict:
    python_invocation = str(Path(sys.executable).resolve())
    source_dir = root / "source"
    source_dir.mkdir()
    relative_sources = {
        "scripts/train_nine_model_qat.py": "trainer",
        "scripts/check_eagle_binary_native.py": "eagle_native",
        "scripts/check_block_binary_native.py": "block_native",
        "scripts/check_eagle_capture_portability.py": "eagle_portability",
        "scripts/check_block_capture_portability.py": "block_portability",
    }
    source = {}
    paths = {}
    for relative, name in relative_sources.items():
        path = source_dir / Path(relative).name
        path.write_text("pinned " + name)
        paths[name] = path
        source[relative] = _locator(path)

    target = root / "target.gguf"
    target.write_bytes(b"target fixture")
    backend = root / "backend"
    backend.write_bytes(b"backend fixture")
    candidates = {}
    for name in CANDIDATES:
        family, precision = name.split("_")
        config = root / (name + ".json")
        config.write_text(json.dumps({"precision_stage": "direct"}))
        model = root / (name + ".gguf")
        model.write_bytes((name + " model").encode())
        export = root / (name + ".audit.json")
        projection_count = 9 if family == "eagle" else 15
        export.write_text(
            json.dumps(
                {"projections": {f"projection_{index}": {} for index in range(projection_count)}}
            )
        )
        native = paths["eagle_native" if family == "eagle" else "block_native"]
        candidates[name] = {
            "family": family,
            "final_bits": int(precision[1:]),
            "precision_stage": "direct",
            "config": _locator(config),
            "model": _locator(model),
            "export": _locator(export),
            "source_bindings": {"model_sha256": _locator(model)["sha256"]},
            "native": {
                "producer": _locator(native),
                "argv": [python_invocation, str(native), "--require-cuda"],
                "wall_seconds": 20,
            },
            "backward": {
                "producer": _locator(paths["trainer"]),
                "argv": [
                    python_invocation,
                    str(paths["trainer"]),
                    "--smoke-zero-updates",
                ],
                "wall_seconds": 20,
            },
        }
    return {
        "schema": "nine_model_sm120_plan_v1",
        "artifact_kind": "fixture",
        "source": source,
        "training_source_files": {
            "scripts/train_nine_model_qat.py": source["scripts/train_nine_model_qat.py"]["sha256"]
        },
        "python": _locator(Path(sys.executable)),
        "python_invocation": python_invocation,
        "backend_binary": _locator(backend),
        "target": _locator(target),
        "candidates": candidates,
        "portability": {
            family: {
                "producer": _locator(
                    paths["eagle_portability" if family == "eagle" else "block_portability"]
                ),
                "argv": [
                    python_invocation,
                    str(paths["eagle_portability" if family == "eagle" else "block_portability"]),
                ],
                "wall_seconds": 20,
            }
            for family in ("eagle", "dspark", "dflash")
        },
        "resource_policy": {
            "host_floor_bytes": 1,
            "gpu_floor_bytes": 1,
            "host_return_tolerance_bytes": 0,
            "gpu_return_tolerance_bytes": 0,
        },
    }


class AdmissionPlanSourceJoinTests(unittest.TestCase):
    def test_valid_plan_joins_each_producer_to_current_source_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            plan = _plan(root)
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            checked, _ = validate_plan(path, fixture=True)
            self.assertEqual(checked["artifact_kind"], "fixture")

    def test_producer_locator_cannot_escape_pinned_source_inventory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            plan = _plan(root)
            decoy = root / "decoy" / "train_nine_model_qat.py"
            decoy.parent.mkdir()
            decoy.write_text("different producer")
            candidate = plan["candidates"]["eagle_a1"]
            candidate["backward"]["producer"] = _locator(decoy)
            candidate["backward"]["argv"][1] = str(decoy)
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                validate_plan(path, fixture=True)

    def test_argv_must_execute_the_pinned_script_instead_of_mentioning_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            plan = _plan(root)
            candidate = plan["candidates"]["eagle_a1"]
            producer = candidate["backward"]["producer"]["path"]
            candidate["backward"]["argv"] = [
                sys.executable,
                "-c",
                "pass",
                producer,
                "--smoke-zero-updates",
            ]
            path = root / "plan.json"
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                validate_plan(path, fixture=True)


class WarmStageMemoryTests(unittest.TestCase):
    @staticmethod
    def _memory_stage(peak: int = 32) -> dict:
        return {
            "status": "PASS",
            "reserved_moment_bytes": 16,
            "optimizer_updates": 0,
            "optimizer_moment_tensors": 0,
            "resources": {"cuda_peak_reserved_bytes": peak},
        }

    def _record(self) -> tuple[dict, dict]:
        candidate = {
            "family": "eagle",
            "final_bits": 1,
            "precision_stage": "a8_to_a1",
            "config": {"sha256": "a" * 64},
            "source_bindings": {"source": "b" * 64},
        }
        record = {
            "schema": "nine_model_model_smoke_v1",
            "status": "PASS",
            "artifact_kind": "fixture",
            "bundle_sha256": "c" * 64,
            "config_sha256": "a" * 64,
            "source": {"source": "b" * 64},
            "optimizer_updates": 0,
            "hardware": {"compute_capability": [12, 0], "gpu_uuid": "fixture"},
            "checks": {"model": "PASS", "backward": "PASS", "memory": "PASS"},
            "smoke_contract": {
                "hard_forward": True,
                "optimizer_updates": 0,
                "optimizer_moment_tensors": 0,
                "activation_bits_exercised": [8, 1],
            },
            "smoke": [
                {"activation_bits": bits, "execution": {"context_cache_calls": 2}}
                for bits in (8, 1)
            ],
            "training_memory": {"A8": self._memory_stage(), "A1": self._memory_stage()},
        }
        return candidate, record

    def test_warm_profile_requires_reserved_peak_for_both_a8_and_a1(self):
        candidate, record = self._record()
        backward_summary(record, candidate, "c" * 64, fixture=True, gpu_uuid="fixture")
        for broken in (
            {"A8": record["training_memory"]["A8"]},
            {"A8": self._memory_stage(15), "A1": record["training_memory"]["A1"]},
        ):
            with self.subTest(stages=sorted(broken)), self.assertRaises(ValueError):
                record["training_memory"] = broken
                backward_summary(record, candidate, "c" * 64, fixture=True, gpu_uuid="fixture")

    def test_warm_eagle_requires_execution_receipts_for_both_precision_stages(self):
        candidate, record = self._record()
        record["smoke"] = record["smoke"][:1]
        with self.assertRaises(ValueError):
            backward_summary(record, candidate, "c" * 64, fixture=True, gpu_uuid="fixture")


class EagleNativeDispatchReceiptTests(unittest.TestCase):
    def _candidate_and_receipt(self, root: Path) -> tuple[dict, dict, list[str]]:
        packed_names = [f"projection_{index}.w1a1_packed" for index in range(9)]
        audit = root / "export-audit.json"
        audit.write_text(
            json.dumps(
                {"projections": {name.removesuffix(".w1a1_packed"): {} for name in packed_names}}
            )
        )
        model = root / "candidate.gguf"
        model.write_bytes(b"model")
        candidate = {
            "family": "eagle",
            "final_bits": 8,
            "model": _locator(model),
            "export": _locator(audit),
        }
        receipt = {
            "schema": "eagle_native_graph_smoke_v1",
            "status": "PASS",
            "model_sha256": candidate["model"]["sha256"],
            "activation_bits": 8,
            "optimizer_updates": 0,
            "cuda_dispatch_observed": True,
            "selected_projection_count": 9,
            "observed_dispatch": {
                "schema": "nine_model_observed_w1_dispatch_v1",
                "status": "PASS",
                "activation_bits": 8,
                "packed_names": packed_names,
                "selected_projection_count": 9,
                "actual_cuda_operations": True,
            },
        }
        return candidate, receipt, packed_names

    def test_all_packed_projections_must_join_export_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            candidate, receipt, _ = self._candidate_and_receipt(root)
            native_summary(receipt, candidate)
            for mutate in (
                lambda record: record.pop("observed_dispatch"),
                lambda record: record["observed_dispatch"].update(
                    packed_names=[f"wrong_{index}" for index in range(9)]
                ),
            ):
                broken = json.loads(json.dumps(receipt))
                mutate(broken)
                with self.assertRaises(ValueError):
                    native_summary(broken, candidate)


class AdmissionReleaseTests(unittest.TestCase):
    class Resources:
        def __init__(self):
            self.release_checks = 0

        def snapshot(self):
            return {
                "compute_capability": [12, 0],
                "gpu_uuid": "fixture",
                "host_available_bytes": 100,
                "gpu_free_bytes": 100,
                "dxg_holders": [],
            }

        def require_released(self, *_):
            self.release_checks += 1
            raise ValueError("surviving owned GPU PID")

    class SafeResources(Resources):
        def require_released(self, *_):
            self.release_checks += 1
            return {"owned_process_groups_absent": True, "owned_cuda_pids_absent": True}

    class Runner:
        def __init__(self, plan):
            self.plan = plan
            self.process_groups = []
            self.process_identities = []
            self.events = []

        @staticmethod
        def _kernel_log():
            lines = [
                f"W1A1_MUL_MAT(k={width},strided=0,bits={bits},n_tokens=7) [CUDA0]: OK"
                for bits in (1, 8)
                for width in (2560, 4096, 7680, 9728, 12800)
            ]
            return "CUDA0\n" + "\n".join(lines) + "\n10/10 tests passed\n"

        def run(self, argv, *, directory, **_):
            self.events.append(str(directory))
            directory.mkdir(parents=True, exist_ok=True)
            if directory.name == "kernel":
                (directory / "stdout.log").write_text(self._kernel_log())
                return
            family = directory.parent.name.split("_")[0]
            if directory.name == "portability":
                record = {
                    "schema": "nine_model_capture_portability_v1",
                    "status": "PASS",
                    "artifact_kind": "fixture",
                    "family": family,
                    "target_sha256": self.plan["target"]["sha256"],
                    "compute_capability": [12, 0],
                    "optimizer_updates": 0,
                    "producer_closed": True,
                    "producer_returncode": 0,
                    "numeric_checks": [
                        {
                            "domain": domain,
                            "decision_changed": False,
                            "saved_target_argmax": 3,
                            "fresh_target_argmax": 3,
                            "features": {"status": "PASS", "max_absolute_error": 0.0},
                            "full_vocab_logits": {
                                "status": "PASS",
                                "max_absolute_error": 0.0,
                            },
                        }
                        for domain in ("prose", "code", "reasoning")
                    ],
                }
            else:
                candidate = self.plan["candidates"][directory.parent.name]
                if directory.name == "native":
                    record = {
                        "model_sha256": candidate["model"]["sha256"],
                        "activation_bits": candidate["final_bits"],
                        "optimizer_updates": 0,
                    }
                    if family == "eagle":
                        audit = json.loads(Path(candidate["export"]["path"]).read_text())
                        packed_names = sorted(
                            base + ".w1a1_packed" for base in audit["projections"]
                        )
                        record.update(
                            schema="eagle_native_graph_smoke_v1",
                            status="PASS",
                            cuda_dispatch_observed=True,
                            selected_projection_count=9,
                            observed_dispatch={
                                "schema": "nine_model_observed_w1_dispatch_v1",
                                "status": "PASS",
                                "activation_bits": candidate["final_bits"],
                                "packed_names": packed_names,
                                "selected_projection_count": 9,
                                "actual_cuda_operations": True,
                            },
                        )
                    else:
                        record.update(
                            schema="block_binary_native_admission_v1",
                            passed=True,
                            cuda_required=True,
                            source_export_sha256=candidate["export"]["sha256"],
                            selected_dense_fallback=False,
                            profile="ffn15",
                            nodes=[
                                {
                                    "packed": f"projection_{index}.w1a1_packed",
                                    "packed_type": "i32",
                                    "activation_bits": candidate["final_bits"],
                                    "output_buffer": "CUDA0",
                                }
                                for index in range(15)
                            ],
                        )
                else:
                    source = dict(candidate["source_bindings"])
                    if family != "eagle":
                        source["bundle_sha256"] = "c" * 64
                    bits = candidate["final_bits"]
                    record = {
                        "schema": "nine_model_model_smoke_v1",
                        "status": "PASS",
                        "artifact_kind": "fixture",
                        "bundle_sha256": "c" * 64,
                        "config_sha256": candidate["config"]["sha256"],
                        "source": source,
                        "optimizer_updates": 0,
                        "hardware": {"compute_capability": [12, 0], "gpu_uuid": "fixture"},
                        "checks": {"model": "PASS", "backward": "PASS", "memory": "PASS"},
                        "smoke_contract": {
                            "hard_forward": True,
                            "optimizer_updates": 0,
                            "optimizer_moment_tensors": 0,
                            "activation_bits_exercised": [bits],
                        },
                        "smoke": [
                            {"activation_bits": bits, "execution": {"context_cache_calls": 2}}
                        ],
                        "training_memory": {
                            "status": "PASS",
                            "reserved_moment_bytes": 16,
                            "optimizer_updates": 0,
                            "optimizer_moment_tensors": 0,
                            "resources": {"cuda_peak_reserved_bytes": 32},
                        },
                    }
            atomic_json(directory / "receipt.json", record)

    def test_admission_refuses_if_owned_context_release_cannot_be_proven(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            plan = _plan(root)
            resources = self.Resources()
            output = root / "admission.json"
            with self.assertRaisesRegex(ValueError, "surviving owned GPU PID"):
                Admission(
                    plan,
                    Files(),
                    self.Runner(plan),
                    resources,
                    "c" * 64,
                    root / "run",
                    fixture=True,
                ).execute(output)
            self.assertGreater(resources.release_checks, 0)
            self.assertFalse(output.exists())
            self.assertEqual(json.loads((root / "run/state.json").read_text())["status"], "failed")

    def test_preexisting_stop_prevents_admission_producers_from_starting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            plan = _plan(root)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "STOP").write_text("operator stop")
            runner = self.Runner(plan)
            resources = self.SafeResources()
            output = root / "admission.json"
            with self.assertRaises(InterruptedError):
                Admission(
                    plan,
                    Files(),
                    runner,
                    resources,
                    "c" * 64,
                    run_dir,
                    fixture=True,
                ).execute(output)
            self.assertEqual(runner.events, [])
            self.assertFalse(output.exists())
            self.assertEqual(json.loads((run_dir / "state.json").read_text())["status"], "failed")


if __name__ == "__main__":
    unittest.main()

"""CPU fixture evidence must never be consumable as a production admission."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from w1a1_eagle.nine_model_admission import (
    Admission,
    backward_summary,
    kernel_summary,
    native_summary,
    portability_summary,
    validate_plan,
)
from w1a1_eagle.nine_model_pipeline import CANDIDATES, Files, atomic_json, sha256


def kernel_log():
    return (
        "CUDA0\n"
        + "\n".join(
            f"W1A1_MUL_MAT(k={width},strided=0,bits={bits},n_tokens=7) [CUDA0]: OK"
            for bits in (1, 8)
            for width in (2560, 4096, 7680, 9728, 12800)
        )
        + "\n10/10 tests passed\n1/1 backends passed\n"
    )


class Resources:
    def __init__(self):
        self.fail = False
        self.release_calls = 0

    def require_released(self, groups, identities):
        self.release_calls += 1
        return {"fixture_owned_process_groups_absent": True}

    def snapshot(self):
        return {
            "compute_capability": [12, 0],
            "gpu_uuid": "fixture-uuid",
            "host_available_bytes": 0 if self.fail else 100,
            "gpu_free_bytes": 100,
            "dxg_holders": [],
        }


class Runner:
    def __init__(self, plan, *, fail_native=False, update=False):
        self.plan, self.fail_native, self.update = plan, fail_native, update
        self.events = []
        self.process_groups = []
        self.process_identities = []

    def run(self, argv, *, directory, **_):
        self.events.append(str(directory))
        self.process_groups.append(len(self.events))
        directory.mkdir(parents=True, exist_ok=True)
        if directory.name == "kernel":
            (directory / "stdout.log").write_text(kernel_log())
            return
        family = directory.parent.name.split("_")[0]
        if directory.name == "portability":
            record = {
                "schema": "nine_model_capture_portability_v1",
                "status": "PASS",
                "artifact_kind": "fixture",
                "target_sha256": self.plan["target"]["sha256"],
                "compute_capability": [12, 0],
                "optimizer_updates": 0,
                "family": family,
                "producer_closed": True,
                "producer_returncode": 0,
                "numeric_checks": [
                    {
                        "domain": domain,
                        "decision_changed": False,
                        "saved_target_argmax": 4,
                        "fresh_target_argmax": 4,
                        "features": {"status": "PASS", "max_absolute_error": 0.0},
                        "full_vocab_logits": {"status": "PASS", "max_absolute_error": 0.0},
                    }
                    for domain in ("prose", "code", "reasoning")
                ],
            }
        else:
            c = self.plan["candidates"][directory.parent.name]
            if directory.name == "native":
                record = {
                    "model_sha256": c["model"]["sha256"],
                    "activation_bits": c["final_bits"],
                    "optimizer_updates": 0,
                }
                if family == "eagle":
                    audit = json.loads(Path(c["export"]["path"]).read_text())
                    record.update(
                        schema="eagle_native_graph_smoke_v1",
                        status="PASS",
                        cuda_dispatch_observed=True,
                        selected_projection_count=9,
                        observed_dispatch={
                            "schema": "nine_model_observed_w1_dispatch_v1",
                            "status": "PASS",
                            "actual_cuda_operations": True,
                            "activation_bits": c["final_bits"],
                            "selected_projection_count": 9,
                            "packed_names": sorted(
                                base + ".w1a1_packed" for base in audit["projections"]
                            ),
                        },
                    )
                else:
                    record.update(
                        schema="block_binary_native_admission_v1",
                        passed=True,
                        cuda_required=True,
                        source_export_sha256=c["export"]["sha256"],
                        selected_dense_fallback=self.fail_native,
                        profile="ffn15",
                        nodes=[
                            {
                                "packed": f"blk.{i}.w1a1_packed",
                                "packed_type": "i32",
                                "activation_bits": c["final_bits"],
                                "output_buffer": "CUDA0",
                            }
                            for i in range(15)
                        ],
                    )
            else:
                record = {
                    "schema": "nine_model_model_smoke_v1",
                    "status": "PASS",
                    "artifact_kind": "fixture",
                    "bundle_sha256": "a" * 64,
                    "config_sha256": c["config"]["sha256"],
                    "source": (
                        dict(c["source_bindings"], bundle_sha256="a" * 64)
                        if family != "eagle"
                        else c["source_bindings"]
                    ),
                    "optimizer_updates": int(self.update),
                    "hardware": {"compute_capability": [12, 0], "gpu_uuid": "fixture-uuid"},
                    "checks": {"model": "PASS", "backward": "PASS", "memory": "PASS"},
                    "smoke_contract": {
                        "optimizer_updates": int(self.update),
                        "optimizer_moment_tensors": 0,
                        "hard_forward": True,
                        "activation_bits_exercised": [c["final_bits"]],
                    },
                    "smoke": {
                        f"A{c['final_bits']}": {
                            "execution": {"context_cache_calls": 2, "effective_batched": True}
                        }
                    },
                    "training_memory": {
                        "status": "PASS",
                        "reserved_moment_bytes": 10,
                        "optimizer_moments_attached": 0,
                        "optimizer_updates": 0,
                        "resources": {"cuda_peak_reserved_bytes": 30},
                    },
                }
        atomic_json(directory / "receipt.json", record)


class AdmissionTests(unittest.TestCase):
    def fixture(self, root):
        artifact = root / "input"
        artifact.write_text("fixture data")
        locator = {"path": str(artifact), "sha256": sha256(artifact)}
        candidate = {
            "config": locator,
            "model": locator,
            "export": locator,
            "source_bindings": {"synthetic": True},
            "native": {"producer": locator, "argv": ["fixture"], "wall_seconds": 1},
            "backward": {"producer": locator, "argv": ["fixture"], "wall_seconds": 1},
        }
        plan = {
            "source": {"fixture": locator},
            "training_source_files": {"fixture": locator["sha256"]},
            "backend_binary": locator,
            "target": locator,
            "resource_policy": {
                "host_floor_bytes": 90,
                "gpu_floor_bytes": 90,
                "host_return_tolerance_bytes": 0,
                "gpu_return_tolerance_bytes": 0,
            },
            "candidates": {
                name: dict(
                    copy.deepcopy(candidate), family=name.split("_")[0], final_bits=int(name[-1])
                )
                for name in CANDIDATES
            },
            "portability": {
                f: {"producer": locator, "argv": ["fixture"], "wall_seconds": 1}
                for f in ("eagle", "dspark", "dflash")
            },
        }
        for family in ("eagle", "dspark", "dflash"):
            bases = (
                [
                    "fc",
                    "output",
                    *(
                        "blk.0." + name
                        for name in (
                            "attn_q",
                            "attn_k",
                            "attn_v",
                            "attn_output",
                            "ffn_gate",
                            "ffn_up",
                            "ffn_down",
                        )
                    ),
                ]
                if family == "eagle"
                else [f"blk.{i}" for i in range(15)]
            )
            audit = root / (family + "-export.json")
            atomic_json(audit, {"projections": {name: {"shape": [32, 32]} for name in bases}})
            for name, c in plan["candidates"].items():
                if name.startswith(family + "_"):
                    c["export"] = {"path": str(audit), "sha256": sha256(audit)}
        return plan

    def test_full_six_candidate_chain_emits_only_fixture_admissions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = self.fixture(root)
            runner = Runner(plan)
            result = Admission(
                plan, Files(), runner, Resources(), "a" * 64, root / "run", fixture=True
            ).execute(root / "result.json")
            self.assertEqual(set(result["candidate_admissions"]), set(CANDIDATES))
            self.assertEqual(len(runner.events), 16)
            for locator in result["candidate_admissions"].values():
                self.assertEqual(sha256(locator["path"]), locator["sha256"])
                self.assertEqual(
                    json.loads(Path(locator["path"]).read_text())["artifact_kind"], "fixture"
                )

    def test_native_fallback_and_optimizer_update_preserve_failure_without_pass(self):
        for flags in ({"fail_native": True}, {"update": True}):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temp:
                root = Path(temp).resolve()
                plan = self.fixture(root)
                with self.assertRaises(ValueError):
                    Admission(
                        plan,
                        Files(),
                        Runner(plan, **flags),
                        Resources(),
                        "a" * 64,
                        root / "run",
                        fixture=True,
                    ).execute(root / "result.json")
                self.assertFalse((root / "result.json").exists())
                self.assertEqual(
                    json.loads((root / "run/state.json").read_text())["status"], "failed"
                )
                self.assertTrue(list((root / "run").rglob("receipt.json")))

    def test_real_plan_refuses_fixture_provenance_before_gpu_access(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(
                json.dumps({"schema": "nine_model_sm120_plan_v1", "artifact_kind": "fixture"})
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                validate_plan(path)

    def test_zero_cases_cpu_missing_width_and_wrong_precision_refuse(self):
        self.assertEqual(kernel_summary(kernel_log())["cases"], 10)
        for bad in (
            "CUDA0\n0/0 tests passed",
            kernel_log().replace("CUDA0", "CPU"),
            kernel_log().replace("k=9728", "k=129"),
            kernel_log().replace("bits=8", "bits=4"),
        ):
            with self.subTest(output=bad[:40]), self.assertRaises(ValueError):
                kernel_summary(bad)

    def test_paused_cli_refuses_before_plan_or_gpu_access(self):
        from admit_nine_model_sm120 import main

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            control = root / "control.json"
            atomic_json(control, {"rtx5080": {"pause_requested": True}})
            with (
                patch("admit_nine_model_sm120.validate_plan") as plan,
                patch("admit_nine_model_sm120.LinuxResources") as device,
            ):
                with self.assertRaisesRegex(InterruptedError, "paused"):
                    main(
                        [
                            "--plan",
                            str(root / "missing"),
                            "--plan-sha256",
                            "b" * 64,
                            "--bundle-sha256",
                            "a" * 64,
                            "--run-dir",
                            str(root / "run"),
                            "--receipt",
                            str(root / "receipt"),
                            "--gpu-uuid",
                            "unqueried",
                            "--gpu-control",
                            str(control),
                        ]
                    )
                plan.assert_not_called()
                device.assert_not_called()

    def test_backward_gpu_moment_memory_and_warm_stage_contracts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = self.fixture(root)
            candidate = plan["candidates"]["dspark_a1"]
            directory = root / "dspark_a1/backward"
            Runner(plan).run([], directory=directory)
            record = json.loads((directory / "receipt.json").read_text())
            arguments = {"fixture": True, "gpu_uuid": "fixture-uuid"}
            backward_summary(record, candidate, "a" * 64, **arguments)
            for mutate in (
                lambda r: r["hardware"].update(gpu_uuid="another-device"),
                lambda r: r["training_memory"].update(reserved_moment_bytes=0),
                lambda r: r["training_memory"].update(optimizer_moments_attached=2),
                lambda r: r["source"].update(bundle_sha256="b" * 64),
            ):
                broken = copy.deepcopy(record)
                mutate(broken)
                with self.assertRaises(ValueError):
                    backward_summary(broken, candidate, "a" * 64, **arguments)
            warm = dict(candidate, precision_stage="a8_to_a1")
            with self.assertRaisesRegex(ValueError, "stages"):
                backward_summary(record, warm, "a" * 64, **arguments)
            record["smoke_contract"]["activation_bits_exercised"] = [8, 1]
            memory = record["training_memory"]
            record["training_memory"] = {"A8": copy.deepcopy(memory), "A1": memory}
            backward_summary(record, warm, "a" * 64, **arguments)

    def test_portability_label_change_and_cpu_host_buffer_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = self.fixture(root)
            runner = Runner(plan)
            directory = root / "dspark/portability"
            runner.run([], directory=directory)
            record = json.loads((directory / "receipt.json").read_text())
            portability_summary(record, "dspark", plan["target"]["sha256"], fixture=True)
            record["numeric_checks"][0]["decision_changed"] = True
            with self.assertRaises(ValueError):
                portability_summary(record, "dspark", plan["target"]["sha256"], fixture=True)
            directory = root / "dspark_a8/native"
            runner.run([], directory=directory)
            record = json.loads((directory / "receipt.json").read_text())
            record["nodes"][0]["output_buffer"] = "CUDA_Host"
            with self.assertRaises(ValueError):
                native_summary(record, plan["candidates"]["dspark_a8"])

    def test_source_join_exact_python_execution_and_phase_validation(self):
        from w1a1_eagle.nine_model_admission import NATIVE, PORTABILITY, TRAINER

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = self.fixture(root)
            (root / "input").write_text(json.dumps({"precision_stage": "direct"}))
            for c in plan["candidates"].values():
                for key in ("config", "model"):
                    c[key]["sha256"] = sha256(root / "input")
            for record in (plan["source"]["fixture"], plan["backend_binary"], plan["target"]):
                record["sha256"] = sha256(root / "input")
            plan["training_source_files"]["fixture"] = sha256(root / "input")
            python = Path(sys.executable).resolve()
            plan.update(
                schema="nine_model_sm120_plan_v1",
                artifact_kind="fixture",
                python={"path": str(python), "sha256": sha256(python)},
                python_invocation=sys.executable,
            )
            for name in {TRAINER, *NATIVE.values(), *PORTABILITY.values()}:
                path = root / name
                path.write_text("# fixture producer\n")
                plan["source"]["scripts/" + name] = {"path": str(path), "sha256": sha256(path)}
            for c in plan["candidates"].values():
                for role, script in (("native", NATIVE[c["family"]]), ("backward", TRAINER)):
                    producer = plan["source"]["scripts/" + script]
                    c[role]["producer"] = producer
                    c[role]["argv"] = [sys.executable, producer["path"], "--smoke-zero-updates"]
            for family, command in plan["portability"].items():
                producer = plan["source"]["scripts/" + PORTABILITY[family]]
                command.update(producer=producer, argv=[sys.executable, producer["path"]])
            path = root / "plan.json"
            atomic_json(path, plan)
            validate_plan(path, fixture=True)
            for mutation in ("unused_script", "decoy", "stage"):
                invalid = copy.deepcopy(plan)
                command = invalid["candidates"]["dspark_a1"]["backward"]
                if mutation == "unused_script":
                    command["argv"] = [
                        sys.executable,
                        "-c",
                        "pass",
                        command["producer"]["path"],
                        "--smoke-zero-updates",
                    ]
                elif mutation == "decoy":
                    decoy = root / "elsewhere" / TRAINER
                    decoy.parent.mkdir()
                    decoy.write_text("# decoy\n")
                    command["producer"] = {"path": str(decoy), "sha256": sha256(decoy)}
                    command["argv"][1] = str(decoy)
                else:
                    invalid["candidates"]["dspark_a1"]["precision_stage"] = "a8_to_a1"
                atomic_json(path, invalid)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    validate_plan(path, fixture=True)

    def test_owned_context_release_failure_cannot_publish_admission(self):
        class NotReleased(Resources):
            def require_released(self, groups, identities):
                raise ValueError("owned CUDA context survives")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "survives"):
                Admission(
                    plan, Files(), Runner(plan), NotReleased(), "a" * 64, root / "run", fixture=True
                ).execute(root / "result.json")
            self.assertFalse((root / "result.json").exists())
            state = json.loads((root / "run/state.json").read_text())
            self.assertEqual(state["cleanup_failure"]["reason"], "owned CUDA context survives")

    def test_preexisting_stop_refuses_before_resources_or_producers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = self.fixture(root)
            run = root / "run"
            run.mkdir()
            (run / "STOP").touch()
            runner, resources = Runner(plan), Resources()
            with patch.object(resources, "snapshot") as snapshot:
                with self.assertRaisesRegex(InterruptedError, "STOP"):
                    Admission(
                        plan, Files(), runner, resources, "a" * 64, run, fixture=True
                    ).execute(root / "result.json")
                snapshot.assert_not_called()
                self.assertEqual(resources.release_calls, 0)
            self.assertFalse(runner.events)
            self.assertFalse((root / "result.json").exists())


if __name__ == "__main__":
    unittest.main()

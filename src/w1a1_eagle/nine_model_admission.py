"""Fresh device admission from concrete native and zero-update producers.

This module does no corpus audit or training. Fixtures emit fixture receipts,
which cannot admit the production trainer. Raw failure evidence is retained.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .nine_model_pipeline import (
    CANDIDATES,
    GATES,
    Files,
    atomic_json,
    require,
    resource_gate,
    sha256,
)

TRAINER = "train_nine_model_qat.py"
NATIVE = {
    "eagle": "check_eagle_binary_native.py",
    "dspark": "check_block_binary_native.py",
    "dflash": "check_block_binary_native.py",
}
PORTABILITY = {
    "eagle": "check_eagle_capture_portability.py",
    "dspark": "check_block_capture_portability.py",
    "dflash": "check_block_capture_portability.py",
}


def validate_plan(path, *, fixture=False):
    plan = json.loads(Path(path).read_text())
    kind = "fixture" if fixture else "production"
    require(
        plan.get("schema") == "nine_model_sm120_plan_v1" and plan.get("artifact_kind") == kind,
        "admission plan provenance differs",
    )
    require(set(plan.get("candidates", {})) == set(CANDIDATES), "six candidate admissions required")
    require(
        set(plan.get("portability", {})) == {"eagle", "dspark", "dflash"},
        "three family capture-portability checks required",
    )
    files = Files()
    require(plan.get("source"), "admission source inventory absent")
    for record in plan["source"].values():
        files.check(record)
    files.check(plan["backend_binary"])
    files.check(plan["target"])
    interpreter = files.check(plan["python"])
    invocation = Path(plan.get("python_invocation", ""))
    require(
        invocation.is_absolute() and invocation.resolve() == interpreter,
        "declared Python invocation does not resolve to pinned interpreter",
    )
    require(plan.get("training_source_files"), "training implementation identity absent")
    for name, digest in plan["training_source_files"].items():
        require(
            name in plan["source"] and plan["source"][name]["sha256"] == digest,
            "training source identity does not join current inventory",
        )
    for name, candidate in plan["candidates"].items():
        family, precision = name.split("_")
        require(
            candidate.get("family") == family
            and type(candidate.get("final_bits")) is int
            and candidate.get("final_bits") == int(precision[1:]),
            "candidate precision differs",
        )
        for key in ("config", "model", "export"):
            files.check(candidate[key])
        config = json.loads(Path(candidate["config"]["path"]).read_text())
        require(
            candidate.get("precision_stage", "direct") == config.get("precision_stage", "direct"),
            "admission precision stage differs from pinned training config",
        )
        require(
            isinstance(candidate.get("source_bindings"), dict) and candidate["source_bindings"],
            "actual training source bindings absent",
        )
        require(
            "bundle_sha256" not in candidate["source_bindings"],
            "admission plan cannot contain a circular bundle hash",
        )
        for role, script in (("native", NATIVE[family]), ("backward", TRAINER)):
            validate_command(candidate[role], files, script, plan)
    for family, spec in plan["portability"].items():
        validate_command(spec, files, PORTABILITY[family], plan)
    return plan, files


def validate_command(spec, files, script, plan):
    path = files.check(spec["producer"])
    require(path.name == script, f"concrete producer required: {script}")
    require(
        spec["producer"] == plan["source"].get("scripts/" + script),
        "producer does not join current source inventory",
    )
    require(
        isinstance(spec.get("argv"), list)
        and spec["argv"]
        and all(isinstance(x, str) and x for x in spec["argv"])
        and len(spec["argv"]) >= 2
        and spec["argv"][0] == plan["python_invocation"]
        and spec["argv"][1] == str(path),
        "producer command not bound",
    )
    require(
        type(spec.get("wall_seconds")) in (int, float) and 0 < spec["wall_seconds"] <= 1800,
        "bounded admission wall cap required",
    )
    if script == TRAINER:
        require(
            "--smoke-zero-updates" in spec["argv"] and "--resume" not in spec["argv"],
            "backward admission must prohibit optimizer progress",
        )


def kernel_summary(output):
    text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    totals = re.findall(r"(\d+)/(\d+) tests passed", text)
    require(
        len(totals) == 1 and int(totals[0][1]) > 0 and totals[0][0] == totals[0][1],
        "native kernel correctness failed or ran zero cases",
    )
    require("CUDA0" in text and "FAIL" not in text, "actual CUDA backend proof absent")
    for bits in (1, 8):
        require(
            re.search(rf"W1A1_MUL_MAT\([^\n]*\bbits={bits}\b[^\n]*\bOK\b", text),
            f"executed native A{bits} cases absent",
        )
        for width in (2560, 4096, 7680, 9728, 12800):
            require(
                re.search(
                    rf"W1A1_MUL_MAT\([^\n]*\bk={width}\b[^\n]*\bbits={bits}\b[^\n]*\bOK\b", text
                ),
                f"executed native A{bits} reduction K{width} absent",
            )
    return {"backend": "CUDA0", "cases": int(totals[0][1]), "activation_bits": [1, 8]}


def native_summary(record, candidate):
    family = candidate["family"]
    require(record.get("optimizer_updates") == 0, "native probe updated optimizer")
    require(
        record.get("model_sha256") == candidate["model"]["sha256"], "native probe model differs"
    )
    if family == "eagle":
        require(
            record.get("schema") == "eagle_native_graph_smoke_v1"
            and record.get("status") == "PASS"
            and record.get("cuda_dispatch_observed") is True
            and record.get("selected_projection_count") == 9,
            "EAGLE native gate incomplete",
        )
    else:
        require(
            record.get("schema") == "block_binary_native_admission_v1"
            and record.get("passed") is True
            and record.get("cuda_required") is True
            and record.get("source_export_sha256") == candidate["export"]["sha256"]
            and record.get("selected_dense_fallback") is False,
            "block native source/export/CUDA gate incomplete",
        )
        nodes = record.get("nodes", [])
        expected = 16 if record.get("profile") == "ffn15_fusion" else 15
        require(
            len(nodes) == expected and len({n.get("packed") for n in nodes}) == expected,
            "selected native node coverage incomplete",
        )
        require(
            all(
                n.get("packed_type") == "i32"
                and n.get("activation_bits") == candidate["final_bits"]
                and re.fullmatch(r"CUDA\d+", n.get("output_buffer", "")) is not None
                for n in nodes
            ),
            "native dense/CPU/precision fallback",
        )
    require(record.get("activation_bits") == candidate["final_bits"], "native final bits differ")


def backward_summary(record, candidate, bundle_hash, *, fixture=False, gpu_uuid=None):
    kind = "fixture" if fixture else "production"
    expected_source = dict(candidate["source_bindings"])
    if candidate["family"] != "eagle":
        expected_source["bundle_sha256"] = bundle_hash
    require(
        record.get("schema") == "nine_model_model_smoke_v1"
        and record.get("status") == "PASS"
        and record.get("artifact_kind") == kind
        and record.get("bundle_sha256") == bundle_hash
        and record.get("config_sha256") == candidate["config"]["sha256"]
        and record.get("source") == expected_source
        and record.get("optimizer_updates") == 0
        and record.get("hardware", {}).get("compute_capability") == [12, 0],
        "actual source/config/device-bound backward receipt differs",
    )
    require(
        record.get("hardware", {}).get("gpu_uuid") == gpu_uuid,
        "actual model/backward GPU UUID differs",
    )
    require(
        record.get("checks") == {"model": "PASS", "backward": "PASS", "memory": "PASS"},
        "actual model/backward/memory gate incomplete",
    )
    smoke = record.get("smoke_contract", {})
    expected_bits = (
        [8, 1] if candidate.get("precision_stage") == "a8_to_a1" else [candidate["final_bits"]]
    )
    require(
        smoke.get("optimizer_updates") == 0
        and smoke.get("optimizer_moment_tensors") == 0
        and smoke.get("hard_forward") is True,
        "zero-update hard-forward gate incomplete",
    )
    require(
        smoke.get("activation_bits_exercised") == expected_bits,
        "all selected precision stages must execute backward",
    )
    memory = record.get("training_memory", {})
    if "stages" in memory:
        stages = memory["stages"]
    elif "A8" in memory:
        stages = {str(int(name[1:])): value for name, value in memory.items()}
    else:
        stages = {str(candidate["final_bits"]): memory}
    require(set(stages) == {str(b) for b in expected_bits}, "training memory stages incomplete")
    for value in stages.values():
        require(
            value.get("status") == "PASS"
            and type(value.get("reserved_moment_bytes")) is int
            and value["reserved_moment_bytes"] > 0
            and value.get("optimizer_updates", 0) == 0
            and value.get("optimizer_moment_tensors", value.get("optimizer_moments_attached")) == 0,
            "actual forward/backward with FP32 moment reservations incomplete",
        )
        metrics = value.get("resources", value.get("resources_gradients_resident", {}))
        require(
            type(metrics.get("cuda_peak_reserved_bytes")) is int
            and metrics["cuda_peak_reserved_bytes"] >= value["reserved_moment_bytes"],
            "actual reserved training-memory peak missing",
        )
    if candidate["family"] == "eagle":
        details = record["smoke"]
        paths = (
            {str(item["activation_bits"]): item["execution"] for item in details}
            if isinstance(details, list)
            else {str(bits): details[f"A{bits}"]["execution"] for bits in expected_bits}
        )
        return paths
    return {}


def portability_summary(record, family, target_sha, *, fixture=False):
    require(
        record.get("schema") == "nine_model_capture_portability_v1"
        and record.get("status") == "PASS"
        and record.get("artifact_kind") == ("fixture" if fixture else "production")
        and record.get("family") == family
        and record.get("target_sha256") == target_sha
        and record.get("compute_capability") == [12, 0]
        and record.get("optimizer_updates") == 0
        and record.get("producer_closed") is True
        and record.get("producer_returncode") == 0,
        "native TRAIN capture portability/producer-release gate incomplete",
    )
    cases = record.get("numeric_checks", [])
    require(
        len(cases) == 3
        and {case.get("domain") for case in cases} == {"prose", "code", "reasoning"},
        "bounded three-domain capture checks required",
    )
    for case in cases:
        require(
            case.get("decision_changed") is False
            and case.get("saved_target_argmax") == case.get("fresh_target_argmax"),
            "native teacher target-label change",
        )
        for field in ("features", "full_vocab_logits"):
            metric = case.get(field, {})
            require(metric.get("status") == "PASS", "native teacher numeric gate incomplete")
            require(
                all(
                    not isinstance(value, bool) and math.isfinite(value)
                    for value in metric.values()
                    if isinstance(value, (float, int)) and not isinstance(value, bool)
                ),
                "nonfinite native teacher diagnostic",
            )


class Admission:
    def __init__(self, plan, files, runner, resources, bundle_hash, run_dir, *, fixture=False):
        self.plan, self.files, self.runner, self.resources = plan, files, runner, resources
        self.bundle_hash, self.run = bundle_hash, Path(run_dir).resolve()
        self.fixture = fixture

    def release(self, baseline):
        report = self.resources.require_released(
            self.runner.process_groups, self.runner.process_identities
        )
        resource_gate(self.resources.snapshot(), baseline, self.plan["resource_policy"])
        atomic_json(
            self.run / ("release-" + str(len(self.runner.process_groups)) + ".json"), report
        )
        return report

    def command(self, name, spec, **values):
        self.files.check(spec["producer"])
        directory = self.run / name
        receipt = directory / "receipt.json"
        require(not receipt.exists(), "fresh admission cannot reuse old producer receipts")
        replacements = {
            "receipt": str(receipt),
            "run_dir": str(directory),
            "bundle_sha256": self.bundle_hash,
            **values,
        }
        argv = [arg.format_map(replacements) for arg in spec["argv"]]
        self.runner.run(
            argv,
            directory=directory,
            stop_path=self.run / "STOP",
            wall_seconds=spec["wall_seconds"],
        )
        require(receipt.is_file(), f"actual producer receipt missing: {name}")
        return json.loads(receipt.read_text()), {"path": str(receipt), "sha256": sha256(receipt)}

    def execute(self, output):
        require(not Path(output).exists(), "preserve previous admission output")
        self.run.mkdir(parents=True, exist_ok=True)
        kind = "fixture" if self.fixture else "production"
        state = {
            "schema": "nine_model_sm120_state_v1",
            "artifact_kind": kind,
            "bundle_sha256": self.bundle_hash,
            "optimizer_updates": 0,
        }
        try:
            baseline = self.resources.snapshot()
            require(baseline.get("compute_capability") == [12, 0], "fresh actual SM120 required")
            resource_gate(baseline, baseline, self.plan["resource_policy"])
            backend = self.files.check(self.plan["backend_binary"])
            kernel_dir = self.run / "kernel"
            self.runner.run(
                [str(backend), "test", "-b", "CUDA0", "-o", "W1A1_MUL_MAT"],
                directory=kernel_dir,
                stop_path=self.run / "STOP",
                wall_seconds=300,
            )
            kernel_log = kernel_dir / "stdout.log"
            kernel = kernel_summary(kernel_log.read_text())
            kernel_evidence = {"path": str(kernel_log), "sha256": sha256(kernel_log)}
            self.release(baseline)
            portable = {}
            for family, spec in self.plan["portability"].items():
                record, evidence = self.command(family + "/portability", spec)
                portability_summary(
                    record, family, self.plan["target"]["sha256"], fixture=self.fixture
                )
                portable[family] = evidence
                self.release(baseline)
            admitted = {}
            for name, candidate in self.plan["candidates"].items():
                native, native_evidence = self.command(name + "/native", candidate["native"])
                native_summary(native, candidate)
                self.release(baseline)
                smoke, backward_evidence = self.command(name + "/backward", candidate["backward"])
                paths = backward_summary(
                    smoke,
                    candidate,
                    self.bundle_hash,
                    fixture=self.fixture,
                    gpu_uuid=baseline["gpu_uuid"],
                )
                self.release(baseline)
                record = {
                    "schema": "nine_model_training_admission_v1",
                    "status": "PASS",
                    "artifact_kind": kind,
                    "bundle_sha256": self.bundle_hash,
                    "candidate": name,
                    "config_sha256": candidate["config"]["sha256"],
                    "source": candidate["source_bindings"],
                    "source_inventory": self.plan["source"],
                    "source_files": self.plan["training_source_files"],
                    "executed_paths": paths,
                    "gpu_uuid": baseline["gpu_uuid"],
                    "compute_capability": [12, 0],
                    "optimizer_updates": 0,
                    "checks": dict.fromkeys(
                        ["source", "resource", *sorted(GATES - {"resources"})], "PASS"
                    ),
                    "evidence": {
                        "kernel": kernel_evidence,
                        "native": native_evidence,
                        "backward": backward_evidence,
                        "capture_portability": portable[candidate["family"]],
                    },
                }
                path = self.run / name / "training-admission.json"
                atomic_json(path, record)
                admitted[name] = {"path": str(path), "sha256": sha256(path)}
            for record in self.plan["source"].values():
                self.files.check(record)
            result = {
                "schema": "nine_model_stage_receipt_v1",
                "stage": "admission",
                "status": "PASS",
                "artifact_kind": kind,
                "bundle_sha256": self.bundle_hash,
                "gates": dict.fromkeys(sorted(GATES), "PASS"),
                "compute_capability": [12, 0],
                "gpu_uuid": baseline["gpu_uuid"],
                "optimizer_updates": 0,
                "candidate_admissions": admitted,
                "kernel": kernel,
                "resource_baseline": baseline,
                "resource_return": self.resources.snapshot(),
            }
            resource_gate(result["resource_return"], baseline, self.plan["resource_policy"])
            atomic_json(output, result)
            state.update(
                status="complete",
                receipt={"path": str(Path(output).resolve()), "sha256": sha256(output)},
            )
            return result
        except BaseException as error:
            state.update(
                status="failed", failure={"type": type(error).__name__, "reason": str(error)}
            )
            try:
                state["owned_release"] = self.resources.require_released(
                    self.runner.process_groups, self.runner.process_identities
                )
            except BaseException as cleanup:
                state["cleanup_failure"] = {"type": type(cleanup).__name__, "reason": str(cleanup)}
            raise
        finally:
            atomic_json(self.run / "state.json", state)

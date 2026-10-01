#!/usr/bin/env python3
"""Prepare checkpoint bundles and collect actual native QAT measurements.

Default: print a CPU-only plan, importing no model/provider/backend. The three
explicit actions are --bootstrap (CPU checkpoint initialization and a CUDA
context), --run-native (bounded serial commands), and --collect (verify existing
measured reports against an eligible current provider). Every action requires
--allow-cuda and an externally resumed, ownership-cleared GPU. No optimizer
step is performed. A bootstrap never grants data eligibility or readiness.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import inspect
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_qat_optimization_readiness as common  # noqa: E402


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": common.sha256(path)}


def checked_path(record):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("file record needs exact path/sha256")
    common.checked_hex(record["sha256"], 64, "file SHA")
    path = Path(record["path"])
    if not path.is_absolute() or common.sha256(path) != record["sha256"]:
        raise ValueError("native input file identity differs")
    return path


def require_unpaused(host, control_path=None):
    path = (
        Path(control_path)
        if control_path
        else Path.home() / ".config/binary-eagle-decoding/gpu-control.json"
    )
    if not path.is_file():
        raise ValueError(
            "GPU control state missing; external human resume/ownership clearance required"
        )
    control = common.read_json(path)
    if control.get(host, {}).get("pause_requested") is not False:
        raise ValueError("GPU remains paused; --allow-cuda cannot override the human pause")


def verify_native_revision(sources, expected_commit):
    """A frozen older server cannot be relabelled as the selected new runtime."""
    record = sources.get("native_runtime", {}).get("immutable_manifest")
    if record is None:
        raise ValueError("native runtime needs a hashed immutable published build manifest")
    manifest = common.read_json(checked_path(record))
    commits, strings = set(), set()

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {
                    "commit",
                    "source_commit",
                    "git_commit",
                    "native_commit",
                    "llama_cpp_commit",
                    "llama_commit",
                    "revision",
                } and isinstance(item, str):
                    commits.add(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            strings.add(value)

    visit(manifest)
    if expected_commit not in commits or sources.get("sha256", {}).get("binary") not in strings:
        raise ValueError(
            "native server binary/build manifest does not bind the selected published commit"
        )


def _hex(value, name):
    if not isinstance(value, str) or len(value) % 2 or len(value) > 16 * 1024**2:
        raise ValueError(name + " is not bounded raw hexadecimal data")
    try:
        return bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(name + " contains invalid raw bytes") from error


def floats(value, name):
    raw = _hex(value, name)
    if not raw or len(raw) % 4:
        raise ValueError(name + " needs nonempty little-endian F32 bytes")
    values = struct.unpack(f"<{len(raw) // 4}f", raw)
    if not all(math.isfinite(x) for x in values):
        raise ValueError(name + " contains nonfinite measured F32 values")
    return values


def hash_hex(value, name):
    import hashlib

    return hashlib.sha256(_hex(value, name)).hexdigest()


def rms(expected, actual):
    if len(expected) != len(actual) or not expected:
        raise ValueError("measured output lengths differ")
    error = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(expected, actual)) / len(expected))
    norm = math.sqrt(math.fsum(a * a for a in expected) / len(expected))
    return 0.0 if error == 0 else error / max(norm, 1e-30)


def norm(values):
    return math.sqrt(math.fsum(x * x for x in values))


def output_sign_sha(value):
    import hashlib

    raw = _hex(value, "output sign source")
    if len(raw) % 4:
        raise ValueError("output signs need F32 raw bytes")
    codes = [
        int(bool(v & 0x80000000) and bool(v & 0x7FFFFFFF)) for (v,) in struct.iter_unpack("<I", raw)
    ]
    return hashlib.sha256(bytes(codes)).hexdigest()


def load_operator_report(path):
    path = Path(path)
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError("native unit report exceeds 32 MiB")
    # Same duplicate/nonfinite parser; current reports are below its 4 MiB cap.
    return common.read_json(path)


def validate_operator_report(report, binding):
    if (
        report.get("schema_version") != 1
        or report.get("input_scope") != "synthetic_operator"
        or report.get("status") != "passed"
        or report.get("requested_backend") != "CUDA"
        or report.get("backend", {}).get("registration") != "CUDA"
    ):
        raise ValueError(
            "actual completed CUDA operator measurements required; CPU/fallback cannot qualify"
        )
    if report.get("backend", {}).get("hardware") != binding["hardware"]["name"]:
        raise ValueError("native operator hardware differs from the current CUDA context")
    commit = report.get("runtime", {}).get("build_commit")
    if (
        not isinstance(commit, str)
        or not 7 <= len(commit) <= 40
        or not binding["native_commit"].startswith(commit)
    ):
        raise ValueError(
            "native unit build revision differs from the full externally verified published commit"
        )
    if report.get("byte_order", "little") != "little":
        raise ValueError("native unit bytes must declare little endian")
    counts = {
        "pack_cases": "pack_cases",
        "graph_cases": "encoder_cases",
        "loader_cases": "loader_cases",
        "projection_cases": "projection_cases",
    }
    for counter, field in counts.items():
        values = report.get(field)
        if (
            not isinstance(values, list)
            or not values
            or len(values) > 10000
            or type(report.get("counters", {}).get(counter)) is not int
            or report["counters"][counter] != len(values)
        ):
            raise ValueError("native unit completion counters differ from measured cases")
    if report["counters"].get("arithmetic_nodes") != sum(
        len(c.get("nodes", [])) for c in report["encoder_cases"]
    ):
        raise ValueError("native graph node coverage is incomplete")
    for case in report["pack_cases"]:
        size = case.get("layout", {}).get("total_words")
        if (
            type(size) is not int
            or size < 1
            or len(_hex(case.get("expected_packed_hex"), "expected pack")) != size * 4
        ):
            raise ValueError("native packed buffer length/layout differs")
        if _hex(case.get("expected_packed_hex"), "expected pack") != _hex(
            case.get("native_packed_hex"), "native pack"
        ):
            raise ValueError("native quantizer packed bytes differ")
    for case in report["loader_cases"]:
        if (
            type(case.get("expected_valid")) is not bool
            or case.get("native_loaded") is not case["expected_valid"]
        ):
            raise ValueError("native loader measured acceptance/rejection differs")


def operator_cases(report, bits, recipe):
    """Convert raw measured bytes; no summary flags become readiness gates."""
    projections = [c for c in report["projection_cases"] if c.get("bits") == bits]
    if (
        len(projections) != len(common.AFFINE_BASES)
        or {c.get("base") for c in projections} != common.AFFINE_BASES
    ):
        raise ValueError("actual native projection fixtures must cover all nine bases")
    learned, affine = [], []
    coverage = (recipe.get("affine_weights") or {}).get("coverage", "fusion")
    selected_bases = common.AFFINE_BASES if coverage == "all" else {"fc"}
    for projection in projections:
        index = projection.get("pack_case_index")
        if type(index) is not int or not 0 <= index < len(report["pack_cases"]):
            raise ValueError("native projection does not join an actual shared pack")
        packet = report["pack_cases"][index]
        for key in ("bits", "k", "n", "input_f32_hex", "learned", "threshold_delta", "clip_ratio"):
            if projection.get(key) != packet.get(key):
                raise ValueError("native projection/shared-pack ancestry differs: " + key)
        for pair in ("codes", "scale", "sum"):
            if _hex(projection["expected_" + pair + "_f32_hex"], pair) != _hex(
                projection["native_" + pair + "_f32_hex"], pair
            ):
                raise ValueError("native projection quantized " + pair + " differs")
        expected = floats(projection["expected_output_f32_hex"], "expected output")
        actual = floats(projection["native_output_f32_hex"], "native output")
        relative_rms = rms(expected, actual)
        common.finite_number(relative_rms, "synthetic native output RMS", maximum=1e-4)
        ancestry = {
            "input_scope": "synthetic_operator",
            "operator_projection_base": projection["base"],
            "pack_case_index": index,
            "input_f32_sha256": hash_hex(projection["input_f32_hex"], "input"),
        }
        if bits in (1, 4, 8):
            if (
                packet.get("learned") is not True
                or (bits == 1 and packet.get("threshold_delta") == 0)
                or (bits in (4, 8) and packet.get("clip_ratio") == 1)
            ):
                raise ValueError(
                    "learned operator proof must execute nondefault native scalar values"
                )
            learned.append(
                {
                    **ancestry,
                    "boundary": projection["boundary"],
                    "packed_expected_sha256": hash_hex(
                        packet["expected_packed_hex"], "expected pack"
                    ),
                    "packed_native_sha256": hash_hex(packet["native_packed_hex"], "native pack"),
                    "native_output_relative_rms": relative_rms,
                    "tested_threshold_delta": packet["threshold_delta"],
                    "tested_clip_ratio": packet["clip_ratio"],
                }
            )
        if projection["base"] not in selected_bases:
            continue
        alpha = floats(projection["alpha_f32_hex"], "alpha")
        midpoint = floats(projection["midpoint_f32_hex"], "midpoint")
        if len(alpha) != len(midpoint) or len(expected) != len(alpha) * projection["n"]:
            raise ValueError("affine row/tokens measurement dimensions differ")
        row = next(
            (i for i, (a, mu) in enumerate(zip(alpha, midpoint)) if a == 0 and mu != 0), None
        )
        if row is None:
            raise ValueError("native affine fixture never executes alpha0/nonzero mu")
        expected_row = expected[row :: len(alpha)]
        native_row = actual[row :: len(alpha)]
        if norm(expected_row) <= 0 or norm(native_row) <= 0:
            raise ValueError("native affine alpha0/mu probe has no positive measured output")
        baseline = floats(projection["native_baseline_output_f32_hex"], "native baseline")
        zero_mu = floats(projection["native_zero_midpoint_output_f32_hex"], "native zero midpoint")
        if len(baseline) != len(zero_mu) or any(a != b for a, b in zip(baseline, zero_mu)):
            raise ValueError("native zero-mu execution differs from actual nonaffine control")
        base_case = {
            **ancestry,
            "projection_base": projection["base"],
            "expected_code_sum_sha256": hash_hex(
                projection["expected_sum_f32_hex"], "expected sum"
            ),
            "native_code_sum_sha256": hash_hex(projection["native_sum_f32_hex"], "native sum"),
            "expected_output_sign_sha256": output_sign_sha(projection["expected_output_f32_hex"]),
            "native_output_sign_sha256": output_sign_sha(projection["native_output_f32_hex"]),
            "expected_tail_sha256": hash_hex(
                packet["expected_packed_hex"], "expected packet tails"
            ),
            "native_tail_sha256": hash_hex(packet["native_packed_hex"], "native packet tails"),
            "tail_hash_scope": "entire_packet_including_padded_codes_and_planes",
            "native_output_relative_rms": relative_rms,
        }
        affine.extend(
            [
                {**base_case, "fixture": "mu_zero_identity", "mu": 0.0, "identity_max_abs": 0.0},
                {
                    **base_case,
                    "fixture": "alpha_zero_nonzero_mu",
                    "alpha": alpha[row],
                    "mu": midpoint[row],
                    "expected_output_norm": norm(expected_row),
                    "native_output_norm": norm(native_row),
                },
                {**base_case, "fixture": "quantized_code_sum"},
            ]
        )
    correction = recipe.get("fusion_correction") or {}
    rank, bias = correction.get("rank", 1), correction.get("output_bias", False)
    encoders = [c for c in report["encoder_cases"] if c.get("bits") == bits]
    defaults = [c for c in encoders if c.get("mode") == "default"]
    zero_mode = "correction_zero_rank4" if rank == 4 else "correction_zero"
    active_mode = (
        ("correction_rank4_bias" if rank == 4 else "correction_bias")
        if bias
        else f"correction_rank{rank}"
    )
    nonzero = [c for c in encoders if c.get("mode") == active_mode]
    zeros = [c for c in encoders if c.get("mode") == zero_mode]
    fusion = []
    for case in nonzero:
        zero = next((c for c in zeros if c["raw_input_f32_hex"] == case["raw_input_f32_hex"]), None)
        baseline = next(
            (c for c in defaults if c["raw_input_f32_hex"] == case["raw_input_f32_hex"]), None
        )
        if (
            zero is None
            or baseline is None
            or case.get("correction_rank") != rank
            or case.get("correction_bias") is not bias
        ):
            raise ValueError("actual fusion rank/bias and zero/default controls are missing")
        nodes = {n["name"]: n for n in case["nodes"]}
        latent, delta = nodes.get("fc_correction_latent"), nodes.get("fc_correction_delta")
        if any(
            n is None or n.get("op") != "MUL_MAT" or n.get("device") != report["backend"]["device"]
            for n in (latent, delta)
        ):
            raise ValueError(
                "raw fusion correction did not execute on the actual requested backend"
            )
        source = next((s for s in latent["sources"] if s["index"] == 1), None)
        if (
            source is None
            or source.get("type") != "f32"
            or source.get("raw_hex") != case["raw_input_f32_hex"]
        ):
            raise ValueError("native fusion used a different raw FC input")
        zero_values = floats(zero["native_output_f32_hex"], "zero fusion")
        baseline_values = floats(baseline["native_output_f32_hex"], "baseline fusion")
        if len(zero_values) != len(baseline_values):
            raise ValueError("zero fusion/default shapes differ")
        identity = max(abs(a - b) for a, b in zip(zero_values, baseline_values))
        expected_values = floats(case["expected_output_f32_hex"], "expected fusion")
        actual_values = floats(case["native_output_f32_hex"], "native fusion")
        native_delta = norm(floats(delta["output_f32_hex"], "actual native correction delta"))
        expected_delta = abs(case["correction_expected_f32"]) * math.sqrt(len(expected_values))
        if identity != 0 or native_delta <= 0 or expected_delta <= 0:
            raise ValueError("actual fusion zero identity/nonzero execution failed")
        fusion.append(
            {
                "input_scope": "synthetic_operator",
                "mode": active_mode,
                "rank": rank,
                "output_bias": bias,
                "raw_input_sha256": hash_hex(case["raw_input_f32_hex"], "raw FC input"),
                "native_raw_input_sha256": hash_hex(source["raw_hex"], "actual raw FC input"),
                "zero_identity_max_abs": identity,
                "expected_correction_delta_norm": expected_delta,
                "native_correction_delta_norm": native_delta,
                "nonzero_forward_relative_rms": rms(expected_values, actual_values),
            }
        )
    return {"learned_quantizers": learned, "affine_weights": affine, "fusion_correction": fusion}


def collect_evidence(
    binding, deployments, decisions, operator_report, output, allowed_prompts, *, fixture_only=False
):
    validate_operator_report(operator_report, binding)
    evidence = {
        "schema": common.NATIVE_SCHEMA,
        "split": "train",
        "fixture_only": fixture_only,
        "optimizer_updates": 0,
        **binding,
        "lanes": {},
    }
    for lane, deployment in deployments.items():
        decision_path = checked_path(decisions[lane])
        measured = common.read_json(decision_path)
        expected = {
            "schema": common.MEASUREMENT_SCHEMA,
            "kind": "native_decisions",
            "split": "train",
            "activation_bits": int(lane[1:]),
            "deployment_state_sha256": deployment,
            **binding,
        }
        if any(measured.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "decision artifact differs from actual source/recipe/deployment binding"
            )
        lane_data = {
            "deployment_state_sha256": deployment,
            "native_decisions": file_record(decision_path),
        }
        cases = operator_cases(operator_report, int(lane[1:]), binding["recipe"])
        options = binding["recipe"]
        required = {
            "learned_quantizers": options.get("activation_quantization") == "learned",
            "fusion_correction": (options.get("fusion_correction") or {}).get("enabled") is True,
            "affine_weights": (options.get("affine_weights") or {}).get("enabled") is True,
        }
        for kind, enabled in required.items():
            if not enabled:
                continue
            if not cases[kind]:
                raise ValueError("actual native operator measurements missing for " + kind)
            artifact = {
                **expected,
                "kind": kind,
                "split": "synthetic_operator",
                "input_scope": "synthetic_operator",
                "cases": cases[kind],
            }
            if kind == "affine_weights":
                artifact["projection_bases"] = sorted({c["projection_base"] for c in cases[kind]})
            path = output / (lane + "-" + kind + ".json")
            common.write_new(path, artifact)
            lane_data[kind] = file_record(path)
        evidence["lanes"][lane] = lane_data
    if not fixture_only:
        common.validate_native_evidence(
            evidence,
            output,
            binding,
            deployments,
            allowed_prompts,
            expected_lanes=tuple(deployments),
        )
    common.write_new(output / "native-evidence.json", evidence)
    return evidence


def runtime_api(curriculum=False):
    api = common.runtime_api()
    import w1ax_capture_provider as pins
    import w1ax_continuous_stages as stages
    from evaluate_pytorch_w1a1 import verify_model_snapshot

    from w1a1_eagle.continuous_resources import linux_host_memory, require_host_memory
    from w1a1_eagle.official_loader import load_official_eagle3
    from w1a1_eagle.recurrent_qat import W1AxContract, install_joint_linears, save_joint_checkpoint

    api.install_joint_linears, api.save_joint_checkpoint, api.W1AxContract = (
        install_joint_linears,
        save_joint_checkpoint,
        W1AxContract,
    )
    api.load_official_eagle3, api.verify_model_snapshot, api.pins = (
        load_official_eagle3,
        verify_model_snapshot,
        pins,
    )
    api.stages, api.linux_host_memory, api.require_host_memory = (
        stages,
        linux_host_memory,
        require_host_memory,
    )
    if curriculum:
        import check_curriculum_qat_readiness as curriculum_tool

        api.curriculum_tool = curriculum_tool
    return api


def recipe_context(args, api):
    path = args.curriculum_config or args.config
    spec = common.read_json(path)
    if args.curriculum_config:
        import train_qat_curriculum as training

        from w1a1_eagle.qat_curriculum_runner import curriculum_runtime
        from w1a1_eagle.qat_readiness import curriculum_readiness_config

        _, curriculum, qat, runner = training.load_config(path)
        bits = tuple(stage.activation_bits for stage in curriculum.stages)
        config = qat
        recipe = api.recipe_identity(curriculum_readiness_config(qat, runner, curriculum))
    else:
        common.validate_config_spec(spec)
        config = api.ContinuousConfig(**spec["training"])
        bits = (8, 1)
        recipe = api.recipe_identity(config)
    hardware = common.cuda_environment(api, config, spec)
    runtime = (
        curriculum_runtime(config.device)
        if args.curriculum_config
        else api.training_runtime_identity(config.device)
    )
    return spec, config, bits, recipe, runtime, hardware


def bootstrap(args, output):
    api = runtime_api(bool(args.curriculum_config))
    spec, config, bits, recipe, runtime, hardware = recipe_context(args, api)
    sources = common.read_json(args.sources)
    api.stages.require_unsealed_prompts(args.prompts, sources=sources)
    api.stages.verify_sources(sources)
    verify_native_revision(sources, spec["native"]["expected_commit"])
    snapshot = common.read_json(sources["model_snapshot_manifest"])
    paths = {}
    for role, repo, revision in (
        ("target", api.pins.TARGET_REPO, api.pins.TARGET_REVISION),
        ("draft", api.pins.DRAFT_REPO, api.pins.DRAFT_REVISION),
    ):
        entry = snapshot["models"][role]
        if entry.get("repo") != repo or entry.get("revision") != revision:
            raise ValueError("model snapshot differs from the pinned source")
        paths[role] = Path(entry["directory"])
        api.verify_model_snapshot(paths[role], entry)
    api.require_host_memory(
        api.linux_host_memory(),
        floor_bytes=2 * 1024**3,
        additional_bytes=12 * 1024**3,
        stage="native evidence CPU checkpoint bootstrap",
    )
    model = api.load_official_eagle3(
        paths["target"],
        paths["draft"],
        angelslim_revision=api.pins.ANGELSLIM_REVISION,
        total_token=60,
        depth=5,
        top_k=10,
        threshold=1.0,
        target_load_kwargs={"dtype": api.torch.float16, "device_map": "cpu"},
    )
    model.eval()
    bundles, deployments = {}, {}
    for precision in bits:
        drafter = copy.deepcopy(model.eagle_layer)
        qat = (
            config.qat(precision)
            if not args.curriculum_config
            else dataclasses.replace(config, contract=api.W1AxContract(precision))
        )
        qat = dataclasses.replace(qat, device="cpu", allow_accelerator=False)
        linears = api.install_joint_linears(drafter, model.base_model, qat)
        if {
            path: [m.out_features, m.in_features] for path, m in linears.items()
        } != common.PINNED_SHAPES:
            raise ValueError("bootstrap requires the nine pinned full-model shapes")
        lane = f"A{precision}"
        directory = output / lane
        directory.mkdir()
        api.save_joint_checkpoint(
            linears,
            qat,
            sources["sha256"]["base_draft_gguf"],
            directory / "joint.npz",
            directory / "joint.json",
        )
        bundles[lane] = {
            "checkpoint": file_record(directory / "joint.npz"),
            "checkpoint_manifest": file_record(directory / "joint.json"),
        }
        deployments[lane] = api.deployment_state_sha256(linears)
        del linears, drafter
        gc.collect()
    context = {
        "schema": "qat_native_bootstrap_v1",
        "fixture_only": False,
        "optimizer_updates": 0,
        "data_eligibility_granted": False,
        "native_evidence_produced": False,
        "recipe": recipe,
        "training_runtime_base": runtime,
        "hardware": hardware,
        "native_commit": spec["native"]["expected_commit"],
        "backend": "cuda",
        "config": file_record(args.curriculum_config or args.config),
        "curriculum": bool(args.curriculum_config),
        "sources": file_record(args.sources),
        "prompts": file_record(args.prompts),
        "unit_binary": file_record(args.unit_binary),
        "bundles": bundles,
        "deployment_state_sha256": deployments,
    }
    common.write_new(output / "bootstrap.json", context)
    context_record = file_record(output / "bootstrap.json")
    commands = []
    for lane in bundles:
        commands.append(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--execute-decision",
                lane,
                "--allow-cuda",
                "--bootstrap-context",
                context_record["path"],
                "--output",
                str(output / ("gate-" + lane)),
                "--gpu-host",
                args.gpu_host,
            ]
        )
    commands.append(
        [
            str(Path(args.unit_binary).resolve()),
            "--backend",
            "CUDA",
            "--json-report",
            str(output / "native-operator.json"),
        ]
    )
    common.write_new(
        output / "commands.json",
        {
            "schema": "qat_native_commands_v1",
            "bootstrap": context_record,
            "commands": commands,
            "timeout_seconds": args.timeout_seconds,
            "log_cap_bytes": 8 * 1024**2,
            "execution_status": "plan_only",
            "external_prerequisite": "human resume and GPU ownership clearance",
        },
    )
    return context


def execute_decision(args, output):
    context = common.read_json(args.bootstrap_context)
    if (
        context.get("schema") != "qat_native_bootstrap_v1"
        or args.execute_decision not in context["bundles"]
    ):
        raise ValueError("invalid native bootstrap/lane")
    sources = common.read_json(checked_path(context["sources"]))
    verify_native_revision(sources, context["native_commit"])
    prompts = checked_path(context["prompts"])
    for record in context["bundles"][args.execute_decision].values():
        checked_path(record)
    sys.path.insert(0, str(ROOT / "src"))
    import check_continuous_w1ax_readiness as producer

    gate = producer.run_gate(
        sources,
        prompts,
        output,
        int(args.execute_decision[1:]),
        expected_prompt_sha256=context["prompts"]["sha256"],
        checkpoint_bundle=context["bundles"][args.execute_decision],
    )
    return {
        "schema": "qat_native_decision_run_v1",
        "gate": file_record(gate),
        "optimizer_updates": 0,
    }


def run_bounded(command, log_path, *, timeout_seconds, log_cap_bytes=8 * 1024**2):
    start = time.monotonic()
    with log_path.open("xb") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            while process.poll() is None:
                if (
                    time.monotonic() - start > timeout_seconds
                    or log_path.stat().st_size > log_cap_bytes
                ):
                    raise RuntimeError("native command exceeded time/log cap")
                try:
                    process.wait(timeout=0.25)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode != 0:
                raise RuntimeError(f"native command failed with exit code {process.returncode}")
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise
    return {
        "command": command,
        "elapsed_seconds": time.monotonic() - start,
        "log": file_record(log_path),
        "exit_code": process.returncode,
    }


def run_native(args, output):
    recipe = common.read_json(Path(args.bootstrap_context).parent / "commands.json")
    if (
        recipe.get("schema") != "qat_native_commands_v1"
        or checked_path(recipe["bootstrap"]) != Path(args.bootstrap_context).resolve()
    ):
        raise ValueError("native command recipe/bootstrap identity differs")
    context = common.read_json(args.bootstrap_context)
    checked_path(context["unit_binary"])
    reports = []
    for index, command in enumerate(recipe["commands"]):
        require_unpaused(args.gpu_host)
        reports.append(
            run_bounded(
                command, output / f"command-{index}.log", timeout_seconds=args.timeout_seconds
            )
        )
        common.write_new(output / f"command-{index}.json", reports[-1])
    return {"schema": "qat_native_serial_execution_v1", "commands": reports, "optimizer_updates": 0}


def collect(args, output):
    context = common.read_json(args.bootstrap_context)
    config_path = checked_path(context["config"])
    args.curriculum_config = config_path if context["curriculum"] else None
    args.config = None if context["curriculum"] else config_path
    api = runtime_api(context["curriculum"])
    spec, config, bits, recipe, runtime, hardware = recipe_context(args, api)
    if (
        recipe != context["recipe"]
        or hardware != context["hardware"]
        or spec["native"]["expected_commit"] != context["native_commit"]
        or runtime != context["training_runtime_base"]
    ):
        raise ValueError("collection changes bootstrap recipe/runtime/hardware/revision")
    if context["curriculum"]:
        import train_qat_curriculum as training

        factory = training.provider_factory(spec["provider"])
        providers = [
            factory(dataclasses.replace(config, contract=api.W1AxContract(b))) for b in bits
        ]
        if any(p.source_metadata != providers[0].source_metadata for p in providers):
            raise ValueError("curriculum stage source identities differ")
        provider = providers[0]
        runtime["provider_implementation_sha256"] = common.sha256(inspect.getfile(type(provider)))
    else:
        provider = common.make_provider(
            args.provider or spec["provider"]["factory"],
            args.provider_manifest or spec["provider"]["manifest"],
            config,
        )
    binding = {
        "source_sha256": common.digest(provider.source_metadata),
        "training_runtime": runtime,
        "recipe": recipe,
        "native_commit": context["native_commit"],
        "backend": "cuda",
        "hardware": hardware,
    }
    sources = common.read_json(checked_path(context["sources"]))
    verify_native_revision(sources, context["native_commit"])
    common_hashes = {
        key: sources["sha256"][key]
        for key in (
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
        )
    }
    provider_hashes = provider.source_metadata.get(
        "common_source_sha256", getattr(provider, "hashes", {})
    )
    if any(provider_hashes.get(k) != v for k, v in common_hashes.items()):
        raise ValueError("provider frozen-model/map ancestry differs from native captures")
    import check_continuous_w1ax_readiness as producer

    records = {}
    for lane, deployment in context["deployment_state_sha256"].items():
        path = Path(args.bootstrap_context).parent / ("gate-" + lane) / "gate.json"
        gate = common.read_json(path)
        producer.validate_gate_report(gate, int(lane[1:]), common_hashes)
        if gate.get("deployment_state_sha256") != deployment or any(
            gate["evidence"][k] != v for k, v in context["bundles"][lane].items()
        ):
            raise ValueError("actual decision report belongs to a different checkpoint deployment")
        path = output / (lane + "-native-decisions.json")
        common.write_new(
            path,
            {
                "schema": common.MEASUREMENT_SCHEMA,
                "kind": "native_decisions",
                "split": "train",
                "activation_bits": int(lane[1:]),
                "deployment_state_sha256": deployment,
                **binding,
                "cases": [
                    {
                        "prompt_id": r["prompt_id"],
                        "round_index": r["round_index"],
                        "capture_id": r["capture_id"],
                        "state_relative_rms": r["state_relative_rms"],
                        "logit_relative_rms": r["logit_relative_rms"],
                        "torch_choice": r["torch_choice"],
                        "native_choice": r["native_choice"],
                        "native_margin": r["decision_margin"],
                    }
                    for r in gate["roots"]
                ],
                "source_gate": file_record(
                    Path(args.bootstrap_context).parent / ("gate-" + lane) / "gate.json"
                ),
            },
        )
        records[lane] = file_record(path)
    unit_path = args.unit_report or Path(args.bootstrap_context).parent / "native-operator.json"
    report = load_operator_report(unit_path)
    binary_path = checked_path(context["unit_binary"])
    if (
        not isinstance(report.get("command"), list)
        or not report["command"]
        or Path(report["command"][0]).resolve() != binary_path
    ):
        raise ValueError("native operator JSON belongs to a different executable")
    common.write_new(
        output / "collection-context.json",
        {
            **binding,
            "bootstrap": file_record(args.bootstrap_context),
            "native_operator": file_record(unit_path),
            "unit_binary": context["unit_binary"],
            "optimizer_updates": 0,
        },
    )
    return collect_evidence(
        binding,
        context["deployment_state_sha256"],
        records,
        report,
        output,
        provider.allowed_prompt_ids,
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    action = result.add_mutually_exclusive_group()
    for name in ("bootstrap", "run-native", "collect"):
        action.add_argument("--" + name, action="store_true")
    action.add_argument("--execute-decision", help=argparse.SUPPRESS)
    result.add_argument("--allow-cuda", action="store_true")
    configs = result.add_mutually_exclusive_group()
    configs.add_argument("--config", type=Path)
    configs.add_argument("--curriculum-config", type=Path)
    for name in (
        "sources",
        "prompts",
        "unit-binary",
        "bootstrap-context",
        "unit-report",
        "provider-manifest",
        "output",
    ):
        result.add_argument("--" + name, type=Path)
    result.add_argument("--provider")
    result.add_argument("--gpu-host", choices=("rtx5080", "rtx2080ti"), default="rtx5080")
    result.add_argument("--timeout-seconds", type=int, default=900)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    active = args.bootstrap or args.run_native or args.collect or args.execute_decision
    if not active:
        print(
            json.dumps(
                {
                    "schema": "qat_native_evidence_plan_v1",
                    "status": "plan_only",
                    "model_loaded": False,
                    "cuda_discovered": False,
                    "native_executed": False,
                    "optimizer_updates": 0,
                    "phases": [
                        "bootstrap same-config CPU checkpoint bundles",
                        "run bounded serial native gates/operator JSON",
                        "collect verified measurements with actual current eligible provider",
                    ],
                    "prerequisites": [
                        "both GPUs remain paused until human resume",
                        "external GPU ownership/idle clearance",
                        "published source/runtime binary and fresh recipe/data proofs",
                    ],
                    "operator_scope": "synthetic_operator; train ancestry is excluded",
                },
                indent=2,
            )
        )
        return 0
    if not args.allow_cuda or args.output is None or not 1 <= args.timeout_seconds <= 3600:
        cli.error(
            "explicit action requires --allow-cuda, unique --output and timeout in 1..3600 seconds"
        )
    if args.bootstrap and (
        not (args.config or args.curriculum_config)
        or any(getattr(args, n) is None for n in ("sources", "prompts", "unit_binary"))
    ):
        cli.error(
            "bootstrap requires config, sources, unsealed prompts and actual native unit binary"
        )
    if not args.bootstrap and args.bootstrap_context is None:
        cli.error("native execution/collection requires an exact bootstrap context")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        require_unpaused(args.gpu_host)
        if args.bootstrap:
            result = bootstrap(args, output)
        elif args.run_native:
            result = run_native(args, output)
        elif args.execute_decision:
            result = execute_decision(args, output)
        else:
            result = collect(args, output)
        common.write_new(output / "result.json", result)
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except Exception as error:
        result = {
            "schema": "qat_native_evidence_attempt_v1",
            "status": "failed",
            "native_evidence_produced": False,
            "optimizer_updates": 0,
            "error_type": type(error).__name__,
            "reason": str(error),
        }
        common.write_new(output / "failure.json", result)
        print(json.dumps(result), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

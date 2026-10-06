#!/usr/bin/env python3
"""Bounded instrumented reproduction of the run02 EAGLE greedy mismatch.

Inspection is the default. Execution needs the paused adapter's authenticated
inputs, reused trained export, fresh lease, shared lock and remote_job supervision.
This diagnostic never grants numeric tolerance or changes the failed strict gate.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_bootstrap.add_argument("--helper-checkout", type=Path, required="--help" not in sys.argv)
_helper, _ = _bootstrap.parse_known_args()
HELPER_ROOT = (_helper.helper_checkout or ROOT).resolve()
sys.path[:0] = [str(HELPER_ROOT / "scripts"), str(HELPER_ROOT / "src")]
# Load the later adapter explicitly; its endpoint/helpers resolve in the frozen
# checkout first. Never select a later endpoint implementation by accident.
_spec = importlib.util.spec_from_file_location(
    "paused_eagle_diagnostic_adapter", ROOT / "scripts/evaluate_paused_eagle_checkpoint.py"
)
paused = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paused)
endpoint = paused.endpoint
require = paused.require
atomic_json = paused.atomic_json
import benchmark_native_eagle as benchmark  # noqa: E402
from w1a1_eagle.nine_model_pipeline import validate_cuda_dispatch  # noqa: E402

NATIVE_COMMIT = "cc9cab3c64f61580cf63e5ef050b075b11cd1fb9"
HELPER_COMMIT = "c2544aa7928b0d0c454099a56ae912262c6b0ab5"
CASE = "gsm8k:train-006474"
POSITION = 98
POSITIONS = (96, 97, 98, 99, 100)
ARMS = ("eagle_a8", "eagle_q4", "target_only")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def native_source(root):
    """Prove exact committed sources, including the target sampler default."""
    root = Path(root).resolve()
    require(git(root, "rev-parse", "HEAD").decode().strip() == NATIVE_COMMIT,
            "frozen cc9 native checkout required")
    require(not git(root, "status", "--porcelain", "--untracked-files=no"),
            "tracked native source changed")
    records = {}
    for name in ("common/common.h", "common/arg.cpp", "tools/server/server-schema.cpp",
                 "tools/server/server-context.cpp"):
        path = root / name
        require(not path.is_symlink() and path.read_bytes() == git(root, "show", f"{NATIVE_COMMIT}:{name}"),
                "committed native source bytes differ: " + name)
        records[name] = endpoint.pin(path)
    header = (root / "common/common.h").read_text()
    sampling = header.split("struct common_params_sampling", 1)[1].split("\n};", 1)[0]
    require(re.search(r"\bbool\s+backend_sampling\s*=\s*false\s*;", sampling),
            "target backend sampling default is not false")
    schema = (root / "tools/server/server-schema.cpp").read_text()
    require('field_bool("backend_sampling", params.sampling.backend_sampling)' in schema,
            "request sampler default source differs")
    server = (root / "tools/server/server-context.cpp").read_text()
    require(all(s in server for s in ("W1AX_VERIFY_TRACE_JSONL", "W1AX_VERIFY_TRACE_POSITIONS",
                                    "W1AX_ROUND_TRACE_JSONL", "use_backend_sampling &= !need_pre_sample_logits")),
            "raw trace/sampler control source absent")
    return {"checkout": str(root), "commit": NATIVE_COMMIT, "source": records,
            "target_backend_sampling_default": False,
            "scope": "instrumented raw logits; target backend sampling remains disabled"}


def request_sequence(prompts):
    matches = [i for i, row in enumerate(prompts) if row["id"] == CASE]
    require(len(matches) == 1 and len(prompts) == 24, "exact failing case/original24 required")
    sequence = [(True, i, prompts[i]) for i in range(2)]
    sequence += [(False, i, row) for i, row in enumerate(prompts[:matches[0] + 1])]
    require(len(sequence) <= 26, "diagnostic exceeds26 requests per arm")
    return sequence


def config(protocol):
    return {"evaluation": {"max_output_tokens": protocol["max_output_tokens"],
                           "temperature": 0.0, "seed": protocol["seed"], "enable_thinking": False}}


def sampler_disabled(command, body):
    require(not any(arg.split("=", 1)[0] in {"-bs", "--backend-sampling"} for arg in command),
            "target backend sampling flag enabled")
    require("backend_sampling" not in body and "logprobs" not in body,
            "request changes frozen sampler mode")


def response_join(response, ids):
    verbose = response.get("__verbose", {})
    require(benchmark.generated_token_ids(response) == ids, "response/output raw ID join failed")
    require(verbose.get("generation_settings", {}).get("backend_sampling") is False,
            "runtime target backend sampling is not disabled")
    require(verbose.get("prompt") is not None, "actual rendered prompt absent")
    return verbose["prompt"]


def raw_point(trace, task, position, ids):
    points = [r for r in trace if r.get("task_id") == task and r.get("generated_position") == position
              and r.get("sampled") is True and r.get("emitted") is True and r.get("replay") is False]
    require(len(points) == 1, "one sampled/emitted/nonreplay point required")
    row = points[0]
    require(row.get("schema") == "w1ax_verify_logits_v1" and row.get("has_logits") is True
            and row.get("nan_logit_count") == 0
            and row.get("sampled_token_id") == row.get("emitted_token_id") == ids[position],
            "raw task/output/NaN join failed")
    top = row.get("raw_top5", [])
    require(len(top) == 5 and len({t.get("token_id") for t in top}) == 5
            and all(type(t.get("token_id")) is int and type(t.get("logit")) in (int, float)
                    and math.isfinite(t["logit"]) for t in top), "raw top5 missing/nonfinite")
    require(top[0]["token_id"] == ids[position]
            and all(top[i]["logit"] >= top[i + 1]["logit"] for i in range(4)),
            "raw argmax/output join failed")
    return row


def causal(row, rounds, ids):
    """Validate reached proposal prefixes; no threshold copied from numeric_gate."""
    if row["mode"] == "target_only":
        require(row["row"] == 0, "target-only causal row0 required")
        return {"mode": "target_only", "generated_prefix_ids": ids[:row["generated_position"]]}
    require(row["mode"] == "speculative_verify", "unknown verifier mode")
    actual = sorted([r for r in rounds if r["task_id"] == row["task_id"]
                     and r["status"] in ("complete", "no_proposal") and not r.get("replay")],
                    key=lambda r: r["round_index"])
    require(len({r["round_index"] for r in actual}) == len(actual), "duplicate causal rounds")
    emitted = [t for r in actual for t in r["emitted_token_ids"]]
    offset = 0 if emitted == ids else 1 if emitted == ids[1:] else None
    require(offset is not None, "round stream differs from complete output IDs")
    cursor, matches = offset, []
    for r in actual:
        if cursor + row["row"] == row["generated_position"] and row["row"] < len(r["emitted_token_ids"]):
            matches.append((cursor, r))
        cursor += len(r["emitted_token_ids"])
    require(len(matches) == 1, "ambiguous causal retained-prefix round")
    base, r = matches[0]
    j = row["row"]
    require(row.get("base_generated_position") == base and r["round_index"] + 1 == row["round_index"]
            and r["n_accepted"] >= j and r["proposed_token_ids"][:j]
            == ids[base:base + j] == row.get("verifier_prefix_draft_ids")
            and r["verified_token_ids"][j] == r["emitted_token_ids"][j] == ids[base + j],
            "unreached/rejected proposal prefix")
    return {"task_id": row["task_id"], "round_index": r["round_index"], "row": j,
            "n_accepted": r["n_accepted"], "base_generated_position": base,
            "prior_draft_prefix": row["verifier_prefix_draft_ids"],
            "generated_prefix_ids": ids[:row["generated_position"]]}


def reference(args, plan, files, checkpoint, sequence, protocol):
    progress_pin = paused.checked(files, args.reference_progress, args.reference_progress_sha256)
    ancestry_pin = paused.checked(files, args.reference_ancestry, args.reference_ancestry_sha256)
    ancestry = paused.read(ancestry_pin["path"])
    require(ancestry["plan"] == endpoint.pin(args.plan)
            and ancestry["checkpoint"]["records"] == checkpoint["records"]
            and ancestry["checkpoint"]["existing_export"] == checkpoint["existing_export"],
            "run02 plan/paused publication ancestry differs")
    progress = paused.read(progress_pin["path"])
    rows = progress["clean_records"]
    diagnostic = progress["diagnostic_records"]
    require(len(rows) == 360 and len(diagnostic) == 72, "complete360clean/72diagnostic run02 required")
    lookup, pins = {}, [progress_pin, ancestry_pin]
    for arm in ARMS:
        for warmup, ordinal, prompt in sequence:
            if warmup:
                continue
            selected = [r for r in rows if r["cell"] == arm and r["prompt_id"] == prompt["id"]]
            require(len(selected) == 5 and {r["repetition"] for r in selected} == set(range(5)),
                    "run02 five clean repetitions absent")
            selected += [r for r in diagnostic if r["cell"] == arm and r["prompt_id"] == prompt["id"]]
            require(len(selected) == 6, "run02 diagnostic record absent")
            for row in selected:
                measurement_path = files.check(row["raw_result"])
                measurement = paused.read(measurement_path)
                require(measurement["generated_token_ids"] == row["generated_token_ids"],
                        "run02 raw output pin differs")
                request_path = measurement_path.parent / "request.json"
                require(paused.read(request_path) == endpoint.request_body(config(protocol), prompt),
                        "run02 request differs from original request")
                sampler_disabled([], paused.read(request_path))
                response_path = measurement_path.parent / "response.json"
                rendered = response_join(paused.read(response_path), measurement["generated_token_ids"])
                pins += [row["raw_result"], endpoint.pin(request_path), endpoint.pin(response_path)]
                key = (arm, ordinal)
                observed = {k: measurement.get(k) for k in
                            ("generated_token_ids", "finish_reason", "completion_sha256", "completion_tokens")}
                observed["actual_rendered_prompt"] = rendered
                require(key not in lookup or lookup[key] == observed,
                        "run02 complete output not stable in all five reps/diagnostic")
                lookup[key] = observed
    index = sequence[-1][1]
    ids = {arm: lookup[arm, index]["generated_token_ids"] for arm in ARMS}
    require(ids["eagle_a8"] == ids["eagle_q4"] and all(len(v) > POSITION for v in ids.values())
            and ids["eagle_a8"][:POSITION] == ids["target_only"][:POSITION]
            and ids["eagle_a8"][POSITION] != ids["target_only"][POSITION],
            "run02 known first divergence98 not reproduced in reference")
    return lookup, pins


def run_arm(plan, protocol, model, arm, port, sequence, stage, authorization, released, remaining, *, audit=None):
    """One synchronous client and fresh owned process; cleanup even on source loss."""
    authorization()
    released()
    require(endpoint.available_port("127.0.0.1", port), "fresh diagnostic port occupied")
    stage.mkdir()
    verify, rounds = stage / "verify.jsonl", stage / "rounds.jsonl"
    command = endpoint.native_command(plan, protocol, arm, model, port, diagnostic=True)
    sampler_disabled(command, {})
    environment = endpoint.native_environment("target_only" if arm == "target_only" else "eagle",
                                              8 if arm == "eagle_a8" else None, plan.get("environment"))
    require("LLAMA_ARG_BACKEND_SAMPLING" not in environment,
            "inherited target backend sampling override forbidden")
    environment.update(CUDA_VISIBLE_DEVICES=plan["gpu_uuid"], W1AX_VERIFY_TRACE_JSONL=str(verify),
                       W1AX_VERIFY_TRACE_POSITIONS=",".join(map(str, POSITIONS)),
                       W1AX_ROUND_TRACE_JSONL=str(rounds))
    if arm == "eagle_a8":
        environment["GGML_W1AX_ADMISSION_TRACE"] = "1"
    atomic_json(stage / "launch.json", {"command": command, "environment": environment,
                                        "scope": "instrumented diagnostic; no timing claim"})
    proc, result = None, []
    with (stage / "server.log").open("wb") as log:
        try:
            mask = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
            prior = signal.pthread_sigmask(signal.SIG_BLOCK, mask)
            try:
                proc = subprocess.Popen(command, cwd=HELPER_ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                        preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, prior))
                identity = paused.process_identity(proc.pid)
                atomic_json(stage / "process.json", {"pid": proc.pid, "pgid": proc.pid,
                                                      "kernel_identity": identity, "argv": command})
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, prior)
            endpoint.wait_ready(proc, f"http://127.0.0.1:{port}", min(protocol["startup_wall_seconds"], remaining()))
            for warmup, ordinal, prompt in sequence:
                authorization()
                require(proc.poll() is None, "owned diagnostic server exited")
                before, round_before = len(endpoint.rows(verify)), len(endpoint.rows(rounds))
                body = endpoint.request_body(config(protocol), prompt)
                sampler_disabled(command, body)
                destination = stage / (f"warmup-{ordinal:04d}" if warmup else f"prompt-{ordinal:04d}")
                measurement = endpoint.execute_request(f"http://127.0.0.1:{port}", body,
                                                       min(180, remaining()), destination)
                ids = measurement.get("generated_token_ids")
                require(isinstance(ids, list) and ids and all(type(t) is int for t in ids),
                        "complete native output IDs absent")
                trace = endpoint.rows(verify)[before:]
                round_rows = endpoint.rows(rounds)[round_before:]
                task_ids = {r["task_id"] for r in trace}
                require(len(task_ids) <= 1, "concurrent diagnostic tasks in one request window")
                response = paused.read(destination / "response.json")
                rendered = response_join(response, ids)
                joined = []
                if prompt["id"] == CASE:
                    require(len(task_ids) == 1 and len(ids) > max(POSITIONS), "failing request trace/window absent")
                    task = next(iter(task_ids))
                    for position in POSITIONS:
                        point = raw_point(trace, task, position, ids)
                        require((point["mode"] == "target_only") == (arm == "target_only"),
                                "trace arm/mode differs")
                        joined.append({"position": position, "raw": point,
                                       "causal": causal(point, round_rows, ids)})
                row = {"warmup": warmup, "ordinal": ordinal, "prompt_id": prompt["id"],
                       "measurement": measurement, "raw_result": endpoint.pin(destination / "measurement.json"),
                       "request": endpoint.pin(destination / "request.json"),
                       "response": endpoint.pin(destination / "response.json"),
                       "actual_rendered_prompt": rendered, "task_ids": sorted(task_ids), "points": joined}
                result.append(row)
                atomic_json(stage / "progress.json", {"records": result})
        finally:
            if proc is not None:
                endpoint.stop_owned_server(proc, grace_s=15)
    resources = released()
    require(endpoint.available_port("127.0.0.1", port), "owned server port remains occupied")
    atomic_json(stage / "resource-return.json", resources)
    if arm == "eagle_a8":
        require(audit is not None, "authenticated candidate audit required")
        actual = validate_cuda_dispatch((stage / "server.log").read_text(errors="replace"), paused.read(audit["path"]), activation_bits=8)
        require(actual["selected_projection_count"] == 9, "actual all-nine CUDA dispatch absent")
        atomic_json(stage / "actual-dispatch.json", actual)
    return {"records": result, "verify": endpoint.pin(verify),
            "rounds": endpoint.pin(rounds) if rounds.exists() else None}


def summarize(arms, original):
    cases = {}
    for arm in ARMS:
        for row in arms[arm]["records"]:
            if row["warmup"]:
                continue
            observed = {k: row["measurement"].get(k) for k in
                        ("generated_token_ids", "finish_reason", "completion_sha256", "completion_tokens")}
            observed["actual_rendered_prompt"] = row["actual_rendered_prompt"]
            require(observed == original[arm, row["ordinal"]], "diagnostic differs from run02 complete output")
            if row["prompt_id"] == CASE:
                cases[arm] = row
    require(set(cases) == set(ARMS), "three diagnostic cases required")
    ids = {arm: row["measurement"]["generated_token_ids"] for arm, row in cases.items()}
    require(ids["eagle_a8"] == ids["eagle_q4"] and ids["eagle_a8"][:POSITION] == ids["target_only"][:POSITION]
            and ids["eagle_a8"][POSITION] != ids["target_only"][POSITION], "diagnostic first98 prefix differs")
    require(all(row["actual_rendered_prompt"] == cases["target_only"]["actual_rendered_prompt"]
                for row in cases.values()), "actual rendered prompt prefix differs across arms")
    pair = [ids["eagle_a8"][POSITION], ids["target_only"][POSITION]]
    margins = {}
    for arm, row in cases.items():
        point = next(p["raw"] for p in row["points"] if p["position"] == POSITION)
        scores = {t["token_id"]: t["logit"] for t in point["raw_top5"]}
        margins[arm] = {"raw_argmax_id": point["raw_top5"][0]["token_id"],
                        "competing_ids": pair,
                        "competing_logits": [scores.get(t) for t in pair],
                        "signed_a8_minus_target_margin": scores[pair[0]] - scores[pair[1]]
                        if all(t in scores for t in pair) else None,
                        "both_competing_ids_in_raw_top5": all(t in scores for t in pair)}
    return {"schema": "eagle_verifier_raw_diagnostic_v1", "status": "DIAGNOSTIC_COLLECTED",
            "strict_target_only_parity": "FAILED", "original_failed_gate_preserved": True,
            "scope": "instrumented diagnostic; no harmlessness, numeric threshold or throughput claim",
            "case": CASE, "first_difference": POSITION, "positions": POSITIONS,
            "original_complete_outputs_reproduced": True,
            "actual_rendered_prompt": cases["target_only"]["actual_rendered_prompt"],
            "shared_first98_generated_ids": ids["target_only"][:POSITION],
            "complete_generated_ids": ids, "margins_at98": margins, "arms": arms,
            "later_positions_scope": "each arm's own reached trajectory after divergence"}


def inspect(args):
    require(args.comparison == "trained", "diagnostic supports the paused trained snapshot only")
    require(Path(args.helper_checkout).resolve() == HELPER_ROOT,
            "helper checkout differs from preimport selection")
    require(git(HELPER_ROOT, "rev-parse", "HEAD").decode().strip() == HELPER_COMMIT,
            "frozen c254 helper checkout required")
    plan, files, policy, protocol, checkpoint, initial, budget = paused.inspect(args)
    require("existing_export" in checkpoint, "authenticated existing trained export required; no new serialization")
    source = native_source(args.native_checkout)
    _, prompts = paused.development_authority(plan, files)
    sequence = request_sequence(prompts)
    original, reference_pins = reference(args, plan, files, checkpoint, sequence, protocol)
    return plan, files, protocol, checkpoint, initial, budget, source, sequence, original, reference_pins


def run(args):
    plan, files, protocol, checkpoint, initial, budget, source, sequence, original, reference_pins = inspect(args)
    if not args.start:
        return {"status": "INSPECTED_NOT_EXECUTED", "strict_target_only_parity": "FAILED",
                "campaign_complete": False, "requests_per_arm": len(sequence), "arms": ARMS,
                "native_source": source, "reference": reference_pins[:2], "gpu_queried": False}
    require(args.availability and args.supervisor_state, "fresh lease/remote_job required")
    require(0 < args.wall_seconds <= 600, "diagnostic wall cap must be<=600s")
    paused.require_unpaused(args.control_path)
    owner, supervisor = paused.supervised(args)
    lease = paused.require_available(args.availability, args.plan_sha256)
    require(lease["gpu_uuid"] == plan["gpu_uuid"]
            and owner not in (plan["controller_identity"], plan["supervisor_identity"]),
            "fresh GPU/owner binding differs")
    output = Path(args.output_root).resolve()
    require(not output.exists() and not output.is_relative_to(Path(plan["training_run_dir"])),
            "fresh output outside original training run required")
    lock = Path.home() / ".config/binary-eagle-decoding/rtx5080-campaign.lock"
    with paused.owner_lock(lock, owner), paused.stop_signals():
        output.mkdir(parents=True)
        observer = paused.LinuxResources(plan["gpu_uuid"])
        groups, identities = endpoint.owned_training_processes(plan)
        identities += [plan["controller_identity"], plan["supervisor_identity"]]
        baseline = None
        pins = [*plan["source"].values(), *plan["runtime"].values(), plan["target"], plan["control"]["model"],
                *checkpoint["records"].values(), *initial["publication"]["records"].values(),
                initial["model"], initial["audit"], checkpoint["existing_export"]["model"],
                checkpoint["existing_export"]["audit"], plan["prompts"], plan["protocol"],
                plan["development_admission"], *checkpoint["development_prompt_authority"]["evidence"],
                *reference_pins, *source["source"].values(), endpoint.pin(args.plan),
                endpoint.pin(Path(__file__)), endpoint.pin(ROOT / "scripts/evaluate_paused_eagle_checkpoint.py")]

        def authorization():
            paused.require_unpaused(args.control_path)
            require(not (output / "STOP").exists(), "diagnostic STOP requested")
            require(paused.read(args.availability) == lease and paused.read(lock) == owner,
                    "fresh lease/shared lock changed")
            require(paused.identity_active(owner) and paused.identity_active(supervisor), "fresh owner identity changed")
            state = paused.read(args.supervisor_state)
            require(state.get("pid") == owner["pid"] and state.get("supervisor_pid") == supervisor["pid"]
                    and state.get("status") == "running", "remote_job supervision changed")
            require(not any(paused.identity_active(i) for i in identities), "retired training/producer active")
            for record in pins:
                files.check(record)

        def released():
            owned = [paused.read(p) for p in output.glob("*/process.json")]
            for p in owned:
                require(p["pid"] == p["pgid"] == p["kernel_identity"]["pid"], "owned group identity differs")
            proof = observer.require_released(groups + [p["pgid"] for p in owned],
                                              identities + [p["kernel_identity"] for p in owned])
            current = observer.snapshot()
            require(proof.get("other_context_pids") == [] and current.get("dxg_holders") == [],
                    "foreign CUDA/DXG context forbids diagnostic")
            paused.resource_gate(current, current if baseline is None else baseline, plan["resource_policy"])
            return current

        try:
            authorization()
            baseline = released()
            atomic_json(output / "input-ancestry.json", {"plan": endpoint.pin(args.plan), "checkpoint": checkpoint,
                        "initial": initial, "budget": budget, "native_source": source, "reference": reference_pins[:2],
                        "fresh_owner": owner, "fresh_supervisor": supervisor, "availability": endpoint.pin(args.availability),
                        "inputs": pins, "wall_seconds": args.wall_seconds, "requests_per_arm": len(sequence),
                        "strict_target_only_parity": "FAILED", "original_training_budget_complete": False})
            started = time.monotonic()

            def remaining():
                left = args.wall_seconds - (time.monotonic() - started)
                require(left > 0, "diagnostic600s wall cap exhausted")
                return left

            models = {"eagle_a8": checkpoint["existing_export"]["model"], "eagle_q4": plan["control"]["model"]}
            arms = {}
            with paused.wall_cap(args.wall_seconds):
                for i, arm in enumerate(ARMS):
                    port = protocol["port"] + i
                    require(0 < port <= 65535, "three distinct valid ports required")
                    arms[arm] = run_arm(plan, protocol, models.get(arm), arm, port, sequence, output / arm,
                                        authorization, released, remaining,
                                        audit=checkpoint["existing_export"]["audit"] if arm == "eagle_a8" else None)
                authorization()
                summary = summarize(arms, original)
            summary.update(campaign_complete=False, owned_release={"status": "PASS", "resources": released()},
                           input_ancestry=endpoint.pin(output / "input-ancestry.json"))
            atomic_json(output / "summary.json", summary)
            return summary
        except BaseException as error:
            try:
                release = {"status": "PASS", "resources": released()}
            except BaseException as cleanup:
                release = {"status": "FAILED", "reason": str(cleanup)}
            atomic_json(output / "failure.json", {"status": "FAILED", "strict_target_only_parity": "FAILED",
                        "campaign_complete": False, "error_type": type(error).__name__, "error": str(error),
                        "owned_release": release})
            raise


def parser():
    cli = paused.parser()
    cli.description = __doc__
    cli.set_defaults(comparison="trained")
    cli.add_argument("--native-checkout", type=Path, required=True)
    for name in ("reference-progress", "reference-ancestry"):
        cli.add_argument("--" + name, type=Path, required=True)
        cli.add_argument("--" + name + "-sha256", required=True)
    cli.add_argument("--wall-seconds", type=float, default=600)
    return cli


def main():
    import json
    print(json.dumps(run(parser().parse_args()), sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plan or measure paired full-model QAT readiness with zero optimizer updates.

Default invocation is a CPU-only plan: no provider/model/native execution and
no accelerator discovery. GPU ownership clearance is an external prerequisite.
The explicit CUDA action accepts actual independently produced native artifacts;
it does not manufacture native evidence or run training updates.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "qat_optimization_readiness_v1"
NATIVE_SCHEMA = "qat_native_optimization_evidence_v1"
MEASUREMENT_SCHEMA = "qat_native_measurements_v1"
PINNED_SHAPES = {
    "fc": [2560, 7680],
    "midlayer.self_attn.q_proj": [4096, 5120],
    "midlayer.self_attn.k_proj": [1024, 5120],
    "midlayer.self_attn.v_proj": [1024, 5120],
    "midlayer.self_attn.o_proj": [2560, 4096],
    "midlayer.mlp.gate_proj": [9728, 2560],
    "midlayer.mlp.up_proj": [9728, 2560],
    "midlayer.mlp.down_proj": [2560, 9728],
    "lm_head": [32000, 2560],
}
BOUNDARIES = {"fc", "qkv", "attn_output", "gate_up", "down", "head"}
BINDINGS = ("source_sha256", "training_runtime", "recipe", "native_commit", "backend", "hardware")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _pairs(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate JSON field: " + name)
        result[name] = value
    return result


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("readiness JSON exceeds 4 MiB")
    def invalid(value):
        raise ValueError("nonfinite JSON value: " + value)
    value = json.loads(path.read_text(), object_pairs_hook=_pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("readiness JSON must be an object")
    canonical(value)  # Also rejects overflowed JSON numbers such as 1e999.
    return value


def write_new(path, value):
    """Never replace a prior report, including after a failed attempt."""
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def checked_artifact(record, base):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("native artifact requires exact path/sha256 record")
    if not isinstance(record["path"], str) or not record["path"]:
        raise ValueError("native artifact path is missing")
    checked_hex(record["sha256"], 64, "artifact SHA")
    path = Path(record["path"])
    if not path.is_absolute():
        path = Path(base) / path
    if sha256(path) != record["sha256"]:
        raise ValueError("native artifact hash mismatch: " + str(path))
    return read_json(path)


def checked_hex(value, length, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None:
        raise ValueError(name + " must be a full lowercase hexadecimal identity")
    return value


def finite_number(value, name, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + " violates the practical numeric gate")
    return value


def deterministic_state_sha256(linears, *, chunk_elements=262144):
    """Device-independent exact live checkpoint state fingerprint.

    Sorted projection/state keys, canonical metadata and raw contiguous tensor
    bytes, streamed through bounded host copies. Frozen scales/biases and
    attached recipe state are included. This intentionally hashes masters, not
    just exported signs. This fingerprint verifies unchanged training masters
    before/after readiness. Native matching separately uses the shared
    qat_state.deployment_state_sha256 effective-state fingerprint.
    No optimizer update or accelerator query occurs here.
    """
    import torch
    if type(chunk_elements) is not int or chunk_elements < 1:
        raise ValueError("state-hash chunk size must be positive")
    result = hashlib.sha256()
    for path in sorted(linears):
        state = linears[path].state_dict()
        for key in sorted(state):
            value = state[key]
            name = path + "." + key
            if isinstance(value, torch.Tensor):
                header = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
                result.update(canonical(header) + b"\n")
                flat = value.detach().reshape(-1)
                for start in range(0, flat.numel(), chunk_elements):
                    part = flat[start : start + chunk_elements].to("cpu").contiguous()
                    result.update(part.view(torch.uint8).numpy().tobytes())
            else:
                result.update(canonical({"name": name, "value": value}) + b"\n")
            result.update(b"\n")
    return result.hexdigest()


def validate_config_spec(spec):
    if spec.get("schema") != "continuous_w1ax_experiment_v1":
        raise ValueError("expected continuous_w1ax_experiment_v1 config")
    shapes = spec.get("model_shapes")
    if shapes != list(PINNED_SHAPES.values()) or any(
        type(value) is not int for shape in (shapes or []) for value in shape
    ):
        raise ValueError("model_shapes must describe all nine pinned full-model projections")
    training = spec.get("training")
    if not isinstance(training, dict) or training.get("device") != "cuda:0":
        raise ValueError("readiness requires the explicit cuda:0 full-model recipe")
    expected = spec.get("native", {}).get("expected_commit")
    checked_hex(expected, 40, "native.expected_commit")
    hardware = spec.get("hardware")
    if (not isinstance(hardware, dict) or not isinstance(hardware.get("device_name"), str)
            or not hardware["device_name"] or not isinstance(hardware.get("compute_capability"), list)
            or len(hardware["compute_capability"]) != 2
            or any(type(v) is not int or v < 0 for v in hardware["compute_capability"])):
        raise ValueError("config needs explicit expected hardware name/compute capability")
    return spec


def plan(args):
    if args.config is not None:
        validate_config_spec(read_json(args.config))
    return {
        "schema": "qat_optimization_readiness_plan_v1", "evidence_device_type": "cpu",
        "status": "plan_only", "passed": False, "receipt_emitted": False,
        "optimizer_updates": 0, "provider_imported": False, "model_loaded": False,
        "cuda_discovered": False, "native_executed": False,
        "required_inputs": ["--allow-cuda", "--provider MODULE:FACTORY", "--provider-manifest",
                            "--config", "--native-evidence", "--output NEW_DIRECTORY"],
        "external_prerequisites": ["GPU ownership clearance and idle proof by orchestrator",
                                   "root-verified published native.expected_commit",
                                   "independent current-checkpoint native measurement artifacts"],
        "planned_cases": {"groups": [1, 2, 4], "paired_lanes": ["A8", "A1"],
                          "warmups": 1, "timing_repeats": 5, "optimizer_updates": 0,
                          "timing_order": "rotating variants and alternating lanes"},
        "model_shapes": PINNED_SHAPES,
    }


def runtime_api():
    """Import model APIs only for the explicit CUDA action."""
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    from w1a1_eagle.continuous_qat import (
        ContinuousConfig, ContinuousTrainer, ObservedAdapter, build_lanes, later_gradient,
    )
    from w1a1_eagle.continuous_runtime import training_runtime_identity
    from w1a1_eagle.qat_curriculum import depth_weighted_supported_ce
    from w1a1_eagle.qat_readiness import recipe_identity, validate_optimization_readiness
    from w1a1_eagle.qat_state import deployment_state_sha256
    from w1a1_eagle.recurrent_provider import audit_provider_round, forward_torch_round
    from w1a1_eagle.recurrent_qat import joint_parameter_families, shared_round_hard_signs
    return SimpleNamespace(**locals())


def cuda_environment(api, config, spec):
    torch = api.torch
    if torch.device(config.device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("explicit CUDA execution requires an available CUDA runtime")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    props = torch.cuda.get_device_properties(config.device)
    hardware = {"device_type": "cuda", "name": torch.cuda.get_device_name(config.device),
                "compute_capability": [int(props.major), int(props.minor)],
                "total_memory_bytes": int(props.total_memory)}
    if (hardware["name"] != spec["hardware"]["device_name"]
            or hardware["compute_capability"] != spec["hardware"]["compute_capability"]):
        raise ValueError("actual CUDA hardware differs from the configured native gate target")
    return hardware


def make_provider(factory, manifest, config):
    name, separator, entry = factory.partition(":")
    if not separator or not name or not entry or ":" in entry:
        raise ValueError("provider requires MODULE:FACTORY")
    creator = getattr(importlib.import_module(name), entry)
    providers = [creator(config.qat(bits), Path(manifest)) for bits in (8, 1)]
    for provider in providers:
        if (provider.training_eligible is not True or provider.split != "train"
                or getattr(provider, "data_split", "train") != "train"
                or getattr(provider, "full_body_qat_eligible", False) is not True
                or getattr(provider, "readiness_scope", None) == "row_a16_hard_ce_100_steps"
                or not provider.allowed_prompt_ids
                or provider.source_metadata.get("split") != "train"):
            raise ValueError("readiness needs an eligible full-body train provider; calibration/finals excluded")
    if providers[0].source_metadata != providers[1].source_metadata:
        raise ValueError("paired providers differ in immutable source identity")
    source = providers[0].source_metadata
    manifest_hash = sha256(manifest)
    candidates = {source.get("execution_manifest_sha256"), source.get("provider_manifest_sha256")}
    if manifest_hash not in candidates:
        raise ValueError("provider does not bind the requested manifest bytes")
    return providers[0]


def validate_model_shapes(lanes):
    import torch
    for lane in lanes:
        if set(lane.linears) != set(PINNED_SHAPES):
            raise ValueError("full-model projection inventory differs")
        for path, expected in PINNED_SHAPES.items():
            module = lane.linears[path]
            if [module.out_features, module.in_features] != expected:
                raise ValueError("tiny fixture or wrong full-model projection shape: " + path)
            if (not isinstance(module.latent_sign, torch.nn.Parameter)
                    or module.latent_sign.dtype != torch.float32
                    or not isinstance(module.scale_offset, torch.nn.Parameter)
                    or module.scale_offset.dtype != torch.float32
                    or module.latent_sign.device.type != "cuda"
                    or list(module.latent_sign.shape) != expected
                    or module.scale_offset.device.type != "cuda"
                    or list(module.scale_offset.shape) != [expected[0]]):
                raise ValueError("full-model trainable masters must reside on actual CUDA")


@contextlib.contextmanager
def zero_updates(lanes):
    originals = []
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("optimizer.step is forbidden in readiness preparation")
    try:
        for lane in lanes:
            originals.append((lane.optimizer, lane.optimizer.step))
            lane.optimizer.step = forbidden
        yield
    finally:
        for optimizer, original in originals:
            optimizer.step = original


def require_zero_progress(lanes):
    for lane in lanes:
        for state in lane.optimizer.state.values():
            step = state.get("step", 0)
            if hasattr(step, "item"):
                step = step.item()
            if isinstance(step, bool) or step != 0:
                raise ValueError("readiness found optimizer progress")


class MemoryAdmissionError(RuntimeError):
    pass


def memory_snapshot(api, config):
    torch = api.torch
    free, total = torch.cuda.mem_get_info(config.device)
    result = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(config.device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(config.device)),
        "allocated_bytes": int(torch.cuda.memory_allocated(config.device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(config.device)),
        "free_bytes": int(free), "total_bytes": int(total),
    }
    if (result["peak_reserved_bytes"] > config.max_cuda_reserved_bytes
            or result["free_bytes"] < config.min_cuda_free_bytes):
        error = MemoryAdmissionError("CUDA allocation ceiling/free-memory floor exceeded")
        error.measurement = result
        raise error
    return result


def selected_rounds(provider, api, config, *, maximum=4, scan_limit=256):
    rounds = []
    identities = set()
    for scanned, batch in enumerate(provider.rounds()):
        if scanned >= scan_limit:
            break
        audit = api.audit_provider_round(batch, provider)
        if (len(batch.prefix_token_ids) > config.max_prefix_tokens or not any(audit.ce_mask[1:])):
            continue
        identity = (batch.capture_id, batch.anchor.prompt_id, batch.anchor.round_index)
        if identity in identities:
            continue
        identities.add(identity)
        rounds.append((batch, audit))
        if len(rounds) == maximum:
            break
    if not rounds:
        raise ValueError("no bounded train round with supported later supervision")
    return rounds


def measure_lane(api, trainer, lane, rounds, flags):
    """One same-snapshot group backward; never an optimizer update."""
    torch, config = api.torch, trainer.config
    lane.optimizer.zero_grad(set_to_none=True)
    batches = [(dataclasses.replace(b, raw_target_features=b.raw_target_features.to(config.device)), a)
               for b, a in rounds]
    resources = trainer.resources()  # Host/disk and current whole-device floors.
    torch.cuda.reset_peak_memory_stats(config.device)
    before = memory_snapshot(api, config)
    torch.cuda.synchronize(config.device)
    start = time.perf_counter()
    losses, outputs, diagnostics = [], [], []
    with api.shared_round_hard_signs(lane.linears):
        for batch, audit in batches:
            observer = api.ObservedAdapter(lane.adapter)
            logits = api.forward_torch_round(batch, observer, trainer.provider.draft_vocab_size,
                optimize_cache=flags[0], optimize_head=flags[1], context_chunk_size=config.context_chunk_size)
            diagnostics.append(api.later_gradient(logits, audit, observer))
            losses.append(api.depth_weighted_supported_ce(logits, audit,
                [r["depth"] for r in batch.rows], decay=config.depth_loss_decay))
            outputs.append(logits)
        torch.cuda.synchronize(config.device)
        forward_seconds = time.perf_counter() - start
        loss = torch.stack(losses).mean()  # Declared equal-round weighting for graph probe.
        backward_start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize(config.device)
        backward_seconds = time.perf_counter() - backward_start
    families = api.joint_parameter_families(lane.linears)
    for family, parameters in families.items():
        for parameter in parameters:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError("missing/nonfinite " + family + " gradient")
            if family in {"sign", "scale"} and not bool((parameter.grad != 0).any()):
                raise ValueError("all-nine binary gradient reachability failed")
    if not bool(torch.isfinite(loss)):
        raise ValueError("nonfinite readiness loss")
    for row in diagnostics:
        for name in ("later_state_gradient_norm", "later_k_gradient_norm", "later_v_gradient_norm"):
            if finite_number(row.get(name), name) <= 0:
                raise ValueError("missing attached later-state/K/V gradient")
    after = memory_snapshot(api, config)
    cpu_outputs = [x.detach().cpu() for x in outputs]
    lane.optimizer.zero_grad(set_to_none=True)
    require_zero_progress(trainer.lanes)
    return ({"lane": lane.name, "group_size": len(rounds), "body_batching": False,
        "optimizer_updates": 0, "weighting": "equal_round_mean", "cache_optimized": flags[0],
        "head_optimized": flags[1], "forward_calls": len(rounds), "backward_calls": 1,
        "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
        "forward_backward_seconds": forward_seconds + backward_seconds,
        "loss": float(loss.detach()), "later_gradients": diagnostics,
        "memory_before": before, "memory_after": after, "resources_before": resources,
        "rounds": [{"capture_id": b.capture_id, "prompt_id": b.anchor.prompt_id,
                    "round_index": b.anchor.round_index, "parent_position": b.rows[0]["parent_position"],
                    "prefix_rows": b.raw_target_features.shape[0] - 1,
                    "valid_depth": sum(a.valid_mask), "supervised_tokens": sum(a.ce_mask)}
                   for b, a in rounds]}, cpu_outputs)


def validate_native_evidence(evidence, base, binding, state_hashes, allowed_prompts):
    """Derive flags from measured independently hashed artifacts, not claims."""
    if (evidence.get("schema") != NATIVE_SCHEMA or evidence.get("split") != "train"
            or evidence.get("fixture_only") is not False
            or type(evidence.get("optimizer_updates")) is not int
            or evidence["optimizer_updates"] != 0):
        raise ValueError("native evidence must be real zero-update train measurements")
    for name in BINDINGS:
        if evidence.get(name) != binding[name]:
            raise ValueError("native evidence binding differs: " + name)
    if set(evidence.get("lanes", {})) != {"A8", "A1"}:
        raise ValueError("native evidence needs both independent lanes")
    decisions, learned, corrections = [], [], []
    artifact_hashes = {"learned_quantizers": [], "fusion_correction": []}
    learned_required = binding["recipe"].get("activation_quantization") == "learned"
    fusion_required = (binding["recipe"].get("fusion_correction") or {}).get("enabled") is True
    for lane_name, lane in evidence["lanes"].items():
        if lane.get("deployment_state_sha256") != state_hashes[lane_name]:
            raise ValueError("native evidence checkpoint differs from current live " + lane_name)
        for kind, required in (("native_decisions", True), ("learned_quantizers", learned_required),
                               ("fusion_correction", fusion_required)):
            if kind not in lane:
                if required:
                    raise ValueError("native evidence is missing actual " + kind + " measurements")
                continue
            measurement = checked_artifact(lane[kind], base)
            if (measurement.get("schema") != MEASUREMENT_SCHEMA or measurement.get("kind") != kind
                    or measurement.get("split") != "train"
                    or measurement.get("activation_bits") != int(lane_name[1:])
                    or measurement.get("deployment_state_sha256") != state_hashes[lane_name]):
                raise ValueError("native measurement schema/lane/checkpoint differs")
            for name in BINDINGS:
                if measurement.get(name) != binding[name]:
                    raise ValueError("native artifact binding differs: " + name)
            cases = measurement.get("cases")
            if not isinstance(cases, list) or not cases:
                raise ValueError("native artifact needs actual measured cases")
            boundaries = set()
            for case in cases:
                if (not isinstance(case, dict) or case.get("prompt_id") not in allowed_prompts
                        or type(case.get("round_index")) is not int or case["round_index"] < 0
                        or not isinstance(case.get("capture_id"), str) or not case["capture_id"]):
                    raise ValueError("native case is outside the eligible train source")
                if kind == "native_decisions":
                    finite_number(case.get("state_relative_rms"), "native state RMS", maximum=.10)
                    finite_number(case.get("logit_relative_rms"), "native logit RMS", maximum=.10)
                    margin = finite_number(case.get("native_margin"), "native choice margin")
                    if any(type(case.get(k)) is not int or case[k] < 0 for k in ("torch_choice", "native_choice")):
                        raise ValueError("native decision choices must be actual nonnegative IDs")
                    if case["torch_choice"] != case["native_choice"] and margin > .02:
                        raise ValueError("material native decision changed")
                    decisions.append(case)
                elif kind == "learned_quantizers":
                    for k in ("packed_expected_sha256", "packed_native_sha256"):
                        checked_hex(case.get(k), 64, k)
                    if case["packed_expected_sha256"] != case["packed_native_sha256"]:
                        raise ValueError("native learned quantizer packed codes differ")
                    finite_number(case.get("native_output_relative_rms"), "learned output RMS", maximum=.10)
                    boundaries.add(case.get("boundary"))
                    learned.append(case)
                else:
                    for k in ("raw_input_sha256", "native_raw_input_sha256"):
                        checked_hex(case.get(k), 64, k)
                    if (case["raw_input_sha256"] != case["native_raw_input_sha256"]
                            or finite_number(case.get("zero_identity_max_abs"), "zero correction identity") != 0):
                        raise ValueError("fusion raw-input ancestry or zero identity failed")
                    if (finite_number(case.get("expected_correction_delta_norm"), "expected correction delta") <= 0
                            or finite_number(case.get("native_correction_delta_norm"), "native correction delta") <= 0):
                        raise ValueError("nonzero correction probe must execute a positive native delta")
                    finite_number(case.get("nonzero_forward_relative_rms"), "nonzero fusion RMS", maximum=.10)
                    corrections.append(case)
            if kind == "learned_quantizers" and boundaries != BOUNDARIES:
                raise ValueError("native learned measurements must cover every shared boundary")
            if kind in artifact_hashes:
                artifact_hashes[kind].append(lane[kind]["sha256"])
    changed = [c for c in decisions if c["torch_choice"] != c["native_choice"]]
    gates = {"native_decisions": {"passed": True, "executed": True, "cases": len(decisions),
        "changed_choice_count": len(changed), "material_choice_changes": 0,
        "max_changed_choice_margin": max([c["native_margin"] for c in changed], default=0.0),
        "max_state_relative_rms": max(c["state_relative_rms"] for c in decisions),
        "max_logit_relative_rms": max(c["logit_relative_rms"] for c in decisions)}}
    if learned_required:
        gates["learned_quantizers"] = {"passed": True, "executed": True, "exact_pack": True,
            "cases": len(learned), "artifact_sha256": digest(artifact_hashes["learned_quantizers"])}
    if fusion_required:
        gates["fusion_correction"] = {"passed": True, "raw_fc_executed": True,
            "zero_identity_passed": True, "nonzero_forward_passed": True, "raw_input_ancestry_passed": True,
            "cases": len(corrections), "artifact_sha256": digest(artifact_hashes["fusion_correction"])}
    return gates


def _aggregate_memory(records):
    snapshots = [row[k] for row in records for k in ("memory_before", "memory_after")]
    return {"passed": True, "scope": "admitted_b1_paired_residency",
        "peak_allocated_bytes": max(s["peak_allocated_bytes"] for s in snapshots),
        "peak_reserved_bytes": max(s["peak_reserved_bytes"] for s in snapshots),
        "min_free_bytes": min(s["free_bytes"] for s in snapshots)}


def preflight_native(evidence, spec, base):
    """Reject missing/malformed native measurements before model/CUDA work."""
    if (evidence.get("schema") != NATIVE_SCHEMA
            or evidence.get("native_commit") != spec["native"]["expected_commit"]
            or evidence.get("backend") != "cuda" or evidence.get("split") != "train"
            or evidence.get("fixture_only") is not False
            or type(evidence.get("optimizer_updates")) is not int or evidence["optimizer_updates"] != 0
            or set(evidence.get("lanes", {})) != {"A8", "A1"}):
        raise ValueError("independent native evidence/expected published revision is missing or mismatched")
    required = ["native_decisions"]
    training = spec["training"]
    if training.get("activation_quantization") == "learned":
        required.append("learned_quantizers")
    if (training.get("fusion_correction") or {}).get("enabled") is True:
        required.append("fusion_correction")
    for lane in evidence["lanes"].values():
        checked_hex(lane.get("deployment_state_sha256"), 64, "native deployment state SHA")
        for kind in required:
            if kind not in lane:
                raise ValueError("native evidence is missing actual " + kind + " measurements")
            artifact = checked_artifact(lane[kind], base)
            if (artifact.get("schema") != MEASUREMENT_SCHEMA or artifact.get("kind") != kind
                    or not isinstance(artifact.get("cases"), list) or not artifact["cases"]):
                raise ValueError("actual native measurement artifact is missing or malformed")


def publish_receipt(api, config, binding, receipt, output):
    """Validate a candidate before publishing an admission locator."""
    candidate = output / "readiness-candidate.json"
    write_new(candidate, receipt)
    def validate(path):
        locator = {"path": str(path.resolve()), "sha256": sha256(path)}
        validation_config = dataclasses.replace(config, optimization_readiness=locator)
        api.validate_optimization_readiness(validation_config,
            source_sha256=binding["source_sha256"], runtime_identity=binding["training_runtime"],
            native_commit=binding["native_commit"], backend="cuda", hardware=binding["hardware"],
            device=config.device, max_cuda_reserved_bytes=config.max_cuda_reserved_bytes,
            min_cuda_free_bytes=config.min_cuda_free_bytes)
        return locator
    validate(candidate)
    receipt_path = output / "readiness.json"
    write_new(receipt_path, receipt)
    locator = validate(receipt_path)
    write_new(output / "receipt-locator.json", locator)
    return locator


def run_cuda(args, output):
    """Only called after explicit allow-cuda and all required CLI inputs."""
    spec = validate_config_spec(read_json(args.config))
    evidence = read_json(args.native_evidence)
    preflight_native(evidence, spec, Path(args.native_evidence).parent)
    api = runtime_api()
    config = api.ContinuousConfig(**spec["training"])
    hardware = cuda_environment(api, config, spec)
    runtime = api.training_runtime_identity(config.device)
    provider = make_provider(args.provider, args.provider_manifest, config)
    binding = {"source_sha256": digest(provider.source_metadata), "training_runtime": runtime,
               "recipe": api.recipe_identity(config), "native_commit": spec["native"]["expected_commit"],
               "backend": "cuda", "hardware": hardware}
    for name in BINDINGS:
        if evidence.get(name) != binding[name]:
            raise ValueError("native evidence differs before model load: " + name)
    write_new(output / "context.json", {**binding, "config_sha256": sha256(args.config),
        "provider_manifest_sha256": sha256(args.provider_manifest), "native_evidence_sha256": sha256(args.native_evidence),
        "cuda_version": api.torch.version.cuda, "tool_sha256": sha256(__file__), "optimizer_updates": 0})
    rounds = selected_rounds(provider, api, config)
    lanes = api.build_lanes(provider, config, output)
    validate_model_shapes(lanes)
    states = {lane.name: deterministic_state_sha256(lane.linears) for lane in lanes}
    deployment_states = {lane.name: api.deployment_state_sha256(lane.linears) for lane in lanes}
    native_gates = validate_native_evidence(evidence, Path(args.native_evidence).parent,
                                           binding, deployment_states, provider.allowed_prompt_ids)
    trainer = api.ContinuousTrainer(provider, lanes, config, output)
    records, skipped, smoke_rows = [], [], []
    with zero_updates(lanes):
        require_zero_progress(lanes)
        api.torch.cuda.reset_peak_memory_stats(config.device)
        api.torch.cuda.synchronize(config.device)
        smoke = trainer.smoke(rounds[0][0])
        api.torch.cuda.synchronize(config.device)
        smoke_memory = memory_snapshot(api, config)
        smoke_rows.append({"memory_before": smoke_memory, "memory_after": smoke_memory})
        variants = [("reference", (False, False)), ("cache", (True, False)),
                    ("configured", (config.optimize_cache, config.optimize_head))]
        blocked = False
        for group in (1, 2, 4):
            if blocked or len(rounds) < group:
                skipped.append({"group_size": group, "status": "skipped", "fit_claimed": False,
                    "reason": "earlier memory admission failed" if blocked else "not enough distinct audited train rounds"})
                continue
            # Warm each case once; rotating starts balance five measured repeats.
            try:
                for _, flags in variants:
                    for lane in lanes:
                        measure_lane(api, trainer, lane, rounds[:group], flags)
                for repeat in range(5):
                    ordered = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
                    lane_order = lanes if repeat % 2 == 0 else list(reversed(lanes))
                    for variant, flags in ordered:
                        for lane in lane_order:
                            row, _ = measure_lane(api, trainer, lane, rounds[:group], flags)
                            row.update(variant=variant, repeat=repeat, measured_order=len(records))
                            records.append(row)
                            with (output / "measurements.jsonl").open("a") as stream:
                                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                                stream.flush()
            except (MemoryAdmissionError, api.torch.cuda.OutOfMemoryError) as error:
                if group == 1:
                    raise
                blocked = True
                failed_memory = getattr(error, "measurement", None)
                if failed_memory is None:
                    try:
                        failed_memory = memory_snapshot(api, config)
                    except MemoryAdmissionError as diagnostic:
                        failed_memory = diagnostic.measurement
                skipped.append({"group_size": group, "status": "skipped", "fit_claimed": False,
                    "reason": str(error), "memory": failed_memory})
                for lane in lanes:
                    lane.optimizer.zero_grad(set_to_none=True)
                gc.collect()
                api.torch.cuda.empty_cache()
                api.torch.cuda.reset_peak_memory_stats(config.device)
                # Failed grouped probes never become proof of admitted fit.
                trainer.resources()
        require_zero_progress(lanes)
        if states != {lane.name: deterministic_state_sha256(lane.linears) for lane in lanes}:
            raise ValueError("readiness mutated current checkpoint state")
    b1 = [r for r in records if r["group_size"] == 1]
    if len(b1) != 5 * len(variants) * len(lanes):
        raise ValueError("incomplete five-repeat full-model B1 evidence")
    gradients = [g for row in b1 for g in row["later_gradients"]]
    gates = {**native_gates,
        "full_model": {"passed": True, "model_scope": "full_model", "finite_gradients": True,
            "forward_calls": sum(row["forward_calls"] for row in b1),
            "backward_calls": sum(row["backward_calls"] for row in b1),
            "later_state_gradient_norm": min(g["later_state_gradient_norm"] for g in gradients),
            "later_key_gradient_norm": min(g["later_k_gradient_norm"] for g in gradients),
            "later_value_gradient_norm": min(g["later_v_gradient_norm"] for g in gradients)},
        "memory": _aggregate_memory(b1 + smoke_rows)}
    receipt = {"schema": SCHEMA, "evidence_device_type": "cuda", "fixture_only": False,
        **binding, "gates": gates, "optimizer_updates": 0, "state_sha256": states,
        "deployment_state_sha256": deployment_states,
        "model_shapes": PINNED_SHAPES, "native_evidence_sha256": sha256(args.native_evidence),
        "timing_repeats": 5, "grouped_probe_skips": skipped,
        "grouped_probe_recipe": "independent graphs at one snapshot; no body batching or optimizer updates",
        "gpu_ownership": "externally cleared by orchestrator; no ownership action performed by this tool"}
    publish_receipt(api, config, binding, receipt, output)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--allow-cuda", action="store_true")
    result.add_argument("--provider")
    result.add_argument("--provider-manifest", type=Path)
    result.add_argument("--config", type=Path)
    result.add_argument("--native-evidence", type=Path)
    result.add_argument("--output", type=Path)
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if args.allow_cuda and any(getattr(args, name) is None for name in
                              ("provider", "provider_manifest", "config", "native_evidence", "output")):
        cli.error("--allow-cuda requires provider, manifest, config, native evidence and unique output")
    output = args.output
    if output is not None:
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    try:
        result = run_cuda(args, output) if args.allow_cuda else plan(args)
        if output is not None and not args.allow_cuda:
            write_new(output / "plan.json", result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        failure = {"schema": "qat_optimization_readiness_attempt_v1", "status": "failed",
            "passed": False, "error_type": type(error).__name__, "reason": str(error), "optimizer_updates": 0}
        if output is not None:
            write_new(output / "failure.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

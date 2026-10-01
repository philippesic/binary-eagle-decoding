#!/usr/bin/env python3
"""Plan or measure all curriculum precisions, with zero optimizer updates.

The default CPU plan imports no provider/model/Torch and discovers no device.
--allow-cuda requires externally cleared GPU ownership and independently
collected current-source native evidence for every requested precision. A4
evidence does not grant A4 training permission: every stage factory must already
authorize the exact audited train source. Grouped probes retain independent
graphs at one snapshot; they never change the training optimizer cadence.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_qat_optimization_readiness as common  # noqa: E402


def validate_spec(spec, *, native_required=False):
    if spec.get("schema") != "qat_curriculum_experiment_v1":
        raise ValueError("expected qat_curriculum_experiment_v1 config")
    stages = spec.get("curriculum", {}).get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("explicit curriculum stages required")
    bits = []
    for stage in stages:
        if not isinstance(stage, dict) or type(stage.get("activation_bits")) is not int:
            raise ValueError("curriculum activation bits must be integers")
        bits.append(stage["activation_bits"])
        if common.finite_number(stage.get("gpu_seconds"), "stage GPU budget") <= 0:
            raise ValueError("positive stage GPU budget required")
        if type(stage.get("max_updates")) is not int or stage["max_updates"] < 1:
            raise ValueError("positive integer stage update budget required")
    if tuple(bits) not in ((1,), (8, 1), (8, 4, 1)):
        raise ValueError("supported schedules are A1, A8/A1, or A8/A4/A1")
    qat = spec.get("qat", {})
    if (
        qat.get("device") != "cuda:0"
        or qat.get("allow_accelerator") is not True
        or qat.get("contract", {}).get("activation_bits") != bits[0]
    ):
        raise ValueError("explicit CUDA recipe with matching initial precision required")
    hardware = spec.get("hardware", {})
    if (
        set(hardware) != {"device_name", "compute_capability"}
        or not isinstance(hardware["device_name"], str)
        or not hardware["device_name"]
        or not isinstance(hardware["compute_capability"], list)
        or len(hardware["compute_capability"]) != 2
        or any(type(v) is not int or v < 0 for v in hardware["compute_capability"])
    ):
        raise ValueError("explicit expected CUDA hardware required")
    if native_required or "native" in spec:
        common.checked_hex(spec.get("native", {}).get("expected_commit"), 40, "native commit")
    return tuple(bits)


def plan(args):
    bits = validate_spec(common.read_json(args.config)) if args.config else (8, 4, 1)
    return {
        "schema": "qat_curriculum_optimization_readiness_plan_v1",
        "evidence_device_type": "cpu",
        "status": "plan_only",
        "passed": False,
        "receipt_emitted": False,
        "optimizer_updates": 0,
        "provider_imported": False,
        "model_loaded": False,
        "cuda_discovered": False,
        "native_executed": False,
        "activation_bits": list(bits),
        "model_shapes": common.PINNED_SHAPES,
        "planned_cases": {"groups": list(args.groups), "warmups": 1, "timing_repeats": 5},
        "required_inputs": [
            "--allow-cuda",
            "--config",
            "--native-evidence",
            "--output NEW_DIRECTORY",
        ],
        "external_prerequisites": [
            "orchestrator GPU ownership clearance and idle proof",
            "published freshly compiled native commit matching this checkout",
            "fresh factory training permission and identical source for every precision",
            "source-bound train native decisions and synthetic-operator native measurements",
        ],
    }


def runtime_api():
    """Heavy imports occur only after explicit CUDA authorization."""
    api = common.runtime_api()
    import train_qat_curriculum as training

    from w1a1_eagle.continuous_qat import restore_rng
    from w1a1_eagle.qat_curriculum_runner import CurriculumRunner, _cpu_tree, curriculum_runtime
    from w1a1_eagle.qat_readiness import curriculum_readiness_config, native_checkout_commit
    from w1a1_eagle.recurrent_qat import W1AxContract

    api.load_config = training.load_config
    api.provider_factory = training.provider_factory
    api.validate_readiness = training.validate_readiness
    api.CurriculumRunner = CurriculumRunner
    api.curriculum_runtime = curriculum_runtime
    api.curriculum_readiness_config = curriculum_readiness_config
    api.native_checkout_commit = native_checkout_commit
    api.W1AxContract = W1AxContract
    api.cpu_tree = _cpu_tree
    api.restore_rng = restore_rng
    return api


def runtime_binding(api, provider, qat, runner_config, curriculum, native_commit, hardware):
    """Same source/runtime/provider implementation identity as CurriculumRunner."""
    runtime = api.curriculum_runtime(qat.device)
    runtime["provider_implementation_sha256"] = common.sha256(inspect.getfile(type(provider)))
    config = api.curriculum_readiness_config(qat, runner_config, curriculum)
    return {
        "source_sha256": common.digest(provider.source_metadata),
        "training_runtime": runtime,
        "recipe": api.recipe_identity(config),
        "native_commit": native_commit,
        "backend": "cuda",
        "hardware": hardware,
    }, config


@contextlib.contextmanager
def forbid_optimizer_updates(torch):
    """Guard classes so freshly rebound stage optimizers are also covered."""
    originals = []

    def forbidden(*_args, **_kwargs):
        raise RuntimeError("optimizer.step is forbidden in curriculum readiness")

    try:
        for optimizer_type in (torch.optim.AdamW, torch.optim.SGD):
            originals.append((optimizer_type, optimizer_type.step))
            optimizer_type.step = forbidden
        yield
    finally:
        for optimizer_type, original in originals:
            optimizer_type.step = original


@contextlib.contextmanager
def resident_optimizer_buffers(api, optimizer):
    """Measure resident F32 state without creating optimizer progress/moments."""
    torch = api.torch
    if isinstance(optimizer, torch.optim.AdamW):
        count = 2
    elif isinstance(optimizer, torch.optim.SGD):
        count = 1  # Conservative even when configured momentum is zero.
    else:
        raise ValueError("unmeasured optimizer state layout cannot grant readiness")
    buffers = [
        torch.zeros_like(p)
        for group in optimizer.param_groups
        for p in group["params"]
        for _ in range(count)
    ]
    try:
        yield {
            "buffers_per_parameter": count,
            "resident_bytes": sum(t.numel() * t.element_size() for t in buffers),
        }
    finally:
        buffers.clear()


def current_lane(runner):
    return SimpleNamespace(
        name=f"A{runner.qat.contract.activation_bits}",
        linears=runner.linears,
        optimizer=runner.optimizer,
        adapter=runner.adapter,
        config=runner.qat,
    )


def measurement_trainer(runner, lane):
    # Paired helper executes the same configured cache/head computation and
    # depth-weighted surrogate; only its resource wrapper is adapted here.
    config = SimpleNamespace(
        device=runner.device,
        depth_loss_decay=runner.qat.depth_loss_decay,
        context_chunk_size=runner.qat.context_chunk_size,
        max_cuda_reserved_bytes=runner.config.max_cuda_reserved_bytes,
        min_cuda_free_bytes=runner.config.min_cuda_free_bytes,
    )
    return SimpleNamespace(
        config=config, provider=runner.provider, lanes=[lane], resources=runner.resources
    )


def full_model_gate(records):
    gradients = [g for record in records for g in record["later_gradients"]]
    if len(records) != 5 or not gradients:
        raise ValueError("every precision requires five measured full-model B1 repeats")
    for g in gradients:
        for field in (
            "later_state_gradient_norm",
            "later_k_gradient_norm",
            "later_v_gradient_norm",
        ):
            if common.finite_number(g.get(field), field) <= 0:
                raise ValueError("attached later-state/K/V gradient required")
    return {
        "passed": True,
        "model_scope": "full_model",
        "finite_gradients": True,
        "forward_calls": sum(r["forward_calls"] for r in records),
        "backward_calls": sum(r["backward_calls"] for r in records),
        "later_state_gradient_norm": min(g["later_state_gradient_norm"] for g in gradients),
        "later_key_gradient_norm": min(g["later_k_gradient_norm"] for g in gradients),
        "later_value_gradient_norm": min(g["later_v_gradient_norm"] for g in gradients),
    }


def measure_stages(api, runner, curriculum, rounds, evidence, base, binding, output, groups):
    """Measure fresh stage recipes, preserving the complete initial checkpoint."""
    original = runner._model_payload()
    initial_rng = api.cpu_tree(runner.rng)
    initial_hash = common.deterministic_state_sha256(runner.linears)
    records, stage_gates, skipped, deployment = [], [], [], {}
    try:
        for index, stage in enumerate(curriculum.stages):
            runner._bind_phase(index)
            lane = current_lane(runner)
            common.validate_model_shapes([lane])
            stage_hash = common.deterministic_state_sha256(runner.linears)
            deployment[lane.name] = api.deployment_state_sha256(runner.linears)
            native = common.validate_native_evidence(
                {**evidence, "lanes": {lane.name: evidence["lanes"][lane.name]}},
                base,
                binding,
                {lane.name: deployment[lane.name]},
                runner.provider.allowed_prompt_ids,
                expected_lanes=(lane.name,),
            )
            trainer = measurement_trainer(runner, lane)
            flags = (runner.qat.optimize_cache, runner.qat.optimize_head)
            measured_b1 = []
            with resident_optimizer_buffers(api, runner.optimizer) as state_buffers:
                common.require_zero_progress([lane])
                for group in groups:
                    if len(rounds) < group:
                        skipped.append(
                            {
                                "activation_bits": stage.activation_bits,
                                "group_size": group,
                                "fit_claimed": False,
                                "reason": "not enough distinct bounded audited roots",
                            }
                        )
                        continue
                    try:
                        common.measure_lane(api, trainer, lane, rounds[:group], flags)
                        for repeat in range(5):
                            row, _ = common.measure_lane(api, trainer, lane, rounds[:group], flags)
                            row.update(
                                repeat=repeat,
                                activation_bits=stage.activation_bits,
                                resident_optimizer_buffers=state_buffers,
                                measured_order=len(records),
                            )
                            records.append(row)
                            if group == 1:
                                measured_b1.append(row)
                            with (output / "measurements.jsonl").open("a") as stream:
                                stream.write(
                                    json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
                                )
                                stream.flush()
                    except (common.MemoryAdmissionError, api.torch.cuda.OutOfMemoryError) as error:
                        if group == 1:
                            raise
                        skipped.append(
                            {
                                "activation_bits": stage.activation_bits,
                                "group_size": group,
                                "fit_claimed": False,
                                "reason": str(error),
                                "memory": getattr(error, "measurement", None),
                            }
                        )
                        runner.optimizer.zero_grad(set_to_none=True)
                        gc.collect()
                        api.torch.cuda.empty_cache()
                        api.torch.cuda.reset_peak_memory_stats(runner.device)
                        runner.resources()
                        break  # Larger grouped probes inherit no fit claim.
                common.require_zero_progress([lane])
                if common.deterministic_state_sha256(runner.linears) != stage_hash:
                    raise ValueError("readiness changed a stage checkpoint master/bank/correction")
            stage_gates.append(
                {
                    "activation_bits": stage.activation_bits,
                    "full_model": full_model_gate(measured_b1),
                    "native_gates": native,
                    "memory": common._aggregate_memory(measured_b1),
                }
            )
            runner.optimizer.zero_grad(set_to_none=True)
        native = common.validate_native_evidence(
            evidence,
            base,
            binding,
            deployment,
            runner.provider.allowed_prompt_ids,
            expected_lanes=tuple(deployment),
        )
    finally:
        runner._bind_phase(0)
        runner._load_model(original)
        runner.rng = initial_rng
        api.restore_rng(initial_rng, runner.device)
    if common.deterministic_state_sha256(runner.linears) != initial_hash:
        raise ValueError("readiness did not restore the complete initial model state")
    if runner.state.global_updates != 0:
        raise ValueError("curriculum optimizer progress is forbidden")
    return records, stage_gates, skipped, deployment, native, initial_hash


def publish_receipt(api, config, runner_config, binding, receipt, output):
    def validate(path):
        locator = {"path": str(path.resolve()), "sha256": common.sha256(path)}
        api.validate_optimization_readiness(
            {**config, "optimization_readiness": locator},
            source_sha256=binding["source_sha256"],
            runtime_identity=binding["training_runtime"],
            native_commit=binding["native_commit"],
            backend="cuda",
            hardware=binding["hardware"],
            device=config["device"],
            max_cuda_reserved_bytes=runner_config.max_cuda_reserved_bytes,
            min_cuda_free_bytes=runner_config.min_cuda_free_bytes,
        )
        return locator

    candidate = output / "readiness-candidate.json"
    common.write_new(candidate, receipt)
    validate(candidate)
    path = output / "readiness.json"
    common.write_new(path, receipt)
    locator = validate(path)
    common.write_new(output / "receipt-locator.json", locator)
    return locator


def run_cuda(args, output):
    spec = common.read_json(args.config)
    bits = validate_spec(spec, native_required=True)
    names = tuple(f"A{b}" for b in bits)
    evidence = common.read_json(args.native_evidence)
    evidence_spec = {"native": spec["native"], "training": spec["qat"]}
    base = Path(args.native_evidence).parent
    common.preflight_native(evidence, evidence_spec, base, expected_lanes=names)
    api = runtime_api()
    _, curriculum, qat, runner_config = api.load_config(args.config)
    factory = api.provider_factory(spec["provider"])
    # Validate every precision factory before any full model is loaded.
    validated = api.validate_readiness(factory, curriculum, qat)
    provider = factory(qat)
    if provider.source_metadata != validated["source"] or not provider.allowed_prompt_ids:
        raise ValueError("fresh provider train ancestry/allowed prompt inventory differs")
    hardware = common.cuda_environment(api, qat, {"hardware": spec["hardware"]})
    native_commit = api.native_checkout_commit()
    if native_commit != spec["native"]["expected_commit"]:
        raise ValueError("native compiled evidence must match this actual native checkout")
    binding, config = runtime_binding(
        api, provider, qat, runner_config, curriculum, native_commit, hardware
    )
    for name in common.BINDINGS:
        if evidence.get(name) != binding[name]:
            raise ValueError("native curriculum evidence differs before model load: " + name)
    common.write_new(
        output / "context.json",
        {
            **binding,
            "config_sha256": common.sha256(args.config),
            "native_evidence_sha256": common.sha256(args.native_evidence),
            "tool_sha256": common.sha256(__file__),
            "optimizer_updates": 0,
        },
    )
    rounds = common.selected_rounds(provider, api, runner_config, maximum=4)
    rounds.sort(key=lambda row: (len(row[0].prefix_token_ids), len(row[0].rows)), reverse=True)
    with forbid_optimizer_updates(api.torch):
        runner = api.CurriculumRunner(
            provider,
            curriculum,
            qat,
            output / "preparation",
            config=runner_config,
            start=True,
            allow_cuda=True,
            source_revalidator=factory,
        )
        if runner.runtime != binding["training_runtime"] or runner.source != validated["source"]:
            raise ValueError("curriculum source/runtime changed during model construction")
        common.validate_model_shapes([current_lane(runner)])
        before = common.deterministic_state_sha256(runner.linears)
        api.torch.cuda.reset_peak_memory_stats(qat.device)
        report = runner.prepare(rounds[0][0])
        if [row["activation_bits"] for row in report] != list(bits):
            raise ValueError("all requested stage smokes required")
        if any(
            row.get("all_nine_binary_gradients_passed") is not True
            or row.get("later_state_kv_passed") is not True
            or row.get("optimizer_updates") != 0
            for row in report
        ):
            raise ValueError("strict all-nine/later-gradient smoke did not pass")
        smoke_memory = common.memory_snapshot(
            api,
            SimpleNamespace(
                device=qat.device,
                max_cuda_reserved_bytes=runner_config.max_cuda_reserved_bytes,
                min_cuda_free_bytes=runner_config.min_cuda_free_bytes,
            ),
        )
        if before != common.deterministic_state_sha256(runner.linears):
            raise ValueError("prepare changed the complete initial checkpoint")
        records, stages, skipped, deployment, native_gates, state_sha = measure_stages(
            api, runner, curriculum, rounds, evidence, base, binding, output, args.groups
        )
    b1 = [row for row in records if row["group_size"] == 1]
    full = {
        "passed": True,
        "model_scope": "full_model",
        "finite_gradients": True,
        "forward_calls": sum(s["full_model"]["forward_calls"] for s in stages),
        "backward_calls": sum(s["full_model"]["backward_calls"] for s in stages),
    }
    for field in (
        "later_state_gradient_norm",
        "later_key_gradient_norm",
        "later_value_gradient_norm",
    ):
        full[field] = min(s["full_model"][field] for s in stages)
    receipt = {
        "schema": common.SCHEMA,
        "evidence_device_type": "cuda",
        "fixture_only": False,
        **binding,
        "optimizer_updates": 0,
        "state_sha256": state_sha,
        "deployment_state_sha256": deployment,
        "model_shapes": common.PINNED_SHAPES,
        "timing_repeats": 5,
        "grouped_probe_skips": skipped,
        "native_evidence_sha256": common.sha256(args.native_evidence),
        "gates": {
            **native_gates,
            "full_model": full,
            "memory": common._aggregate_memory(
                b1 + [{"memory_before": smoke_memory, "memory_after": smoke_memory}]
            ),
            "curriculum_stages": {"passed": True, "activation_bits": list(bits), "stages": stages},
        },
    }
    publish_receipt(api, config, runner_config, binding, receipt, output)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--curriculum-config", dest="config", type=Path)
    parser.add_argument("--native-evidence", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-cuda", action="store_true")
    parser.add_argument("--groups", type=int, nargs="+", default=[1], choices=(1, 2, 4))
    args = parser.parse_args(argv)
    if args.groups != sorted(set(args.groups)) or args.groups[0] != 1:
        parser.error("groups must be unique increasing 1, optional 2 and/or 4")
    if args.allow_cuda and any(
        getattr(args, name) is None for name in ("config", "native_evidence", "output")
    ):
        parser.error("--allow-cuda requires config, native evidence and new output")
    output = args.output.resolve() if args.output else None
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    try:
        result = run_cuda(args, output) if args.allow_cuda else plan(args)
        if output is not None and not args.allow_cuda:
            common.write_new(output / "plan.json", result)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as error:
        failure = {
            "schema": "qat_curriculum_optimization_readiness_attempt_v1",
            "status": "failed",
            "passed": False,
            "optimizer_updates": 0,
            "error_type": type(error).__name__,
            "reason": str(error),
        }
        if output is not None:
            common.write_new(output / "failure.json", failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

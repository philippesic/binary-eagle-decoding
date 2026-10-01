"""USER-started bounded native/CUDA gates for separate row W1A8 and W1A1.

No model, accelerator query or native process is touched on import. This is a
training-deployment gate, not a throughput or held-out accuracy claim. Numeric
checks are restricted to exact captured roots and actionable decision margins.
"""

from __future__ import annotations

import ctypes
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from w1ax_continuous_stages import (
    build_native_labels,
    checked_record,
    file_record,
    host_admission,
    load_native_labels,
    native_capture,
    sha256,
    write_json,
)

SCHEMA = "w1ax_continuous_precision_gate_v1"
RECIPE_SCHEMA = "w1ax_continuous_precision_gate_v2"
CHECKS = (
    "export_serialization_and_frozen_operands",
    "native_stored_f16_cache_and_masks",
    "exact_prefix_states_and_logits",
    "no_high_margin_changed_native_decisions",
    "all_nine_finite_gradients",
    "later_loss_reaches_earlier_student_state_and_cache",
    "torch_f16_cache_and_positions",
)


def _host_memory_snapshot(proc_root: Path = Path("/proc")) -> dict:
    from w1a1_eagle.continuous_resources import _kib_field, linux_host_memory

    try:
        stats = linux_host_memory(proc_root)
        stats["process_rss_anon_bytes"] = _kib_field(
            (proc_root / "self/status").read_text(), "RssAnon"
        )
        return {"status": "recorded", **stats}
    except (OSError, RuntimeError) as error:
        return {"status": "unavailable", "reason": str(error)}


def _trim_host_allocator() -> dict:
    """Optionally ask Linux/glibc to return free CPU allocator pages."""
    if sys.platform != "linux":
        return {"status": "unavailable", "reason": "requires Linux/glibc"}
    try:
        libc = ctypes.CDLL(None)
        version = libc.gnu_get_libc_version
        version.argtypes = []
        version.restype = ctypes.c_char_p
        glibc_version = version().decode("ascii")
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
    except (OSError, AttributeError) as error:
        return {"status": "unavailable", "reason": str(error)}
    try:
        return {
            "status": "called",
            "glibc_version": glibc_version,
            "return_code": int(trim(0)),
        }
    except (OSError, ValueError, ctypes.ArgumentError) as error:
        return {"status": "failed", "glibc_version": glibc_version, "reason": str(error)}


def _reclaim_and_admit_host(output: Path, stage: str, diagnostic_filename: str) -> dict:
    """Persist before/after CPU retention evidence, then apply the same gate."""
    before = _host_memory_snapshot()
    collected = gc.collect()
    trim = _trim_host_allocator()
    after = _host_memory_snapshot()
    write_json(
        output / diagnostic_filename,
        {
            "schema": "w1ax_gate_host_memory_diagnostic_v1",
            "admission_stage": stage,
            "before": before,
            "gc_collected_objects": collected,
            "malloc_trim": trim,
            "after": after,
        },
    )
    return host_admission(stage, 12 * 1024**3)


def validate_gate_report(report: dict, bits: int, common: dict) -> None:
    modern = report.get("schema") == RECIPE_SCHEMA
    mismatches = [
        name
        for name, expected in {
            "schema": RECIPE_SCHEMA if modern else SCHEMA,
            "activation_bits": bits,
            "execution_device": "cuda:0",
            "scale_layout": "row",
            "objective": "hard_ce",
            "optimizer_steps": 0,
            "common_source_sha256": common,
            "limits": {"relative_rms": 0.10, "decision_margin": 0.02},
        }.items()
        if report.get(name) != expected
    ]
    if bits not in ({1, 4, 8} if modern else {1, 8}):
        mismatches.append("requested precision requires an independent versioned gate")
    if len(report.get("roots", [])) < 6:
        mismatches.append("roots require at least six entries")
    checks = report.get("checks", {})
    if not isinstance(checks, dict):
        mismatches.append("checks must be an object")
        failed_checks = []
    else:
        if set(checks) != set(CHECKS):
            mismatches.append(
                f"checks inventory missing={sorted(set(CHECKS) - set(checks))} "
                f"unexpected={sorted(set(checks) - set(CHECKS))}"
            )
        failed_checks = sorted(name for name, value in checks.items() if value is not True)
    if mismatches or failed_checks:
        raise ValueError(
            "independent A8/A1 numeric/cache/export/backward gate failed: "
            f"metadata mismatches={mismatches}; failed checks={failed_checks}"
        )
    for name, record in report.get("evidence", {}).items():
        checked_record(record)
    if set(report.get("evidence", {})) != {
        "export_audit",
        "native_cache_audit",
        "native_capture_manifest",
        "checkpoint",
        "checkpoint_manifest",
        "export",
        "native_cell",
    }:
        raise ValueError("precision gate evidence inventory differs")
    evidence = {
        name: json.loads(checked_record(record).read_text())
        for name, record in report["evidence"].items()
        if name
        in {
            "export_audit",
            "native_cache_audit",
            "native_capture_manifest",
            "checkpoint_manifest",
            "native_cell",
        }
    }
    checkpoint = evidence["checkpoint_manifest"]
    native = evidence["native_cell"]
    exported = evidence["export_audit"]
    captured = evidence["native_capture_manifest"]
    cache = evidence["native_cache_audit"]
    if modern:
        from w1ax_capture_provider import validate_actor_export

        manifest = validate_actor_export(
            {
                name: report["evidence"][name]
                for name in ("checkpoint", "checkpoint_manifest", "export", "export_audit")
            },
            activation_bits=bits,
            base_hash=common["base_draft_gguf"],
        )
        expected_recipe = checkpoint_recipe(manifest)
        if report.get("recipe") != expected_recipe:
            raise ValueError("precision gate recipe differs from independently audited actor")
        from w1a1_eagle.trajectory_refresh import checked_hash

        checked_hash(report.get("deployment_state_sha256"))
        expected_counts = {
            "sign": 9,
            "scale": 9,
            "activation": 6 if manifest.get("activation_quantizers") is not None else 0,
            "fusion": (2 + int(manifest["fusion_correction"]["bias_name"] is not None))
            if manifest.get("fusion_correction") is not None
            else 0,
        }
        affine = manifest.get("affine_weights")
        reported_counts = dict(report.get("trainable_parameter_counts", {}))
        if affine is not None:
            expected_counts["midpoint"] = len(affine["tensors"])
        elif reported_counts.get("midpoint") == 0:
            reported_counts.pop("midpoint")
        if reported_counts != expected_counts:
            raise ValueError("precision gate trainable ownership differs from recipe")
        expected_gradient_tensors = sum(expected_counts.values())
    else:
        expected_gradient_tensors = 18
        if checkpoint.get("schema_version") not in (None, 2) or any(
            checkpoint.get(name) is not None
            for name in ("activation_quantizers", "fusion_correction", "affine_weights")
        ):
            raise ValueError("legacy gate cannot grant learned/correction/affine readiness")
    if (
        checkpoint.get("activation_bits") != bits
        or checkpoint.get("scale_layout") != "row"
        or checkpoint.get("objective") != "hard_ce"
        or (
            not modern
            and checkpoint.get("activation_rule")
            != "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive"
        )
        or checkpoint.get("weight_rule") != "hard_sign_zero_positive_clipped_identity_ste"
        or checkpoint.get("base_gguf_sha256") != common["base_draft_gguf"]
        or checkpoint.get("checkpoint_sha256") != report["evidence"]["checkpoint"]["sha256"]
        or exported.get("activation_bits") != bits
        or exported.get("serialization_audit_passed") is not True
        or exported.get("output", {}).get("sha256") != report["evidence"]["export"]["sha256"]
        or native.get("env", {}).get("GGML_W1AX_ACT_BITS") != str(bits)
        or native.get("target_sha256") != common["target_gguf"]
        or native.get("draft_sha256") != report["evidence"]["export"]["sha256"]
        or native.get("binary_sha256") != report.get("native_binary_sha256")
        or captured.get("activation_bits") != bits
        or captured.get("target_sha256") != common["target_gguf"]
        or captured.get("draft_sha256") != report["evidence"]["export"]["sha256"]
        or cache.get("schema") != "recurrent_stored_draft_cache_audit_v2"
        or cache.get("status") != "stored_f16_rows_and_exact_prefix_masks_compared"
        or cache.get("execution_device") != "cuda"
        or type(cache.get("key_elements")) is not int
        or cache["key_elements"] < 1
        or cache.get("key_elements") != cache.get("key_equal_elements")
        or cache.get("value_elements") != cache.get("value_equal_elements")
    ):
        raise ValueError("precision evidence differs from runtime/checkpoint/export contracts")
    native_directory = checked_record(report["evidence"]["native_cell"]).parent
    if modern:
        log = checked_record(
            {
                "path": str(native_directory / "server.log"),
                "sha256": native.get("files", {}).get("server.log", {}).get("sha256"),
            }
        )
        markers = {
            1: "CUDA packed W1A1 XOR/POPCOUNT dispatch",
            4: "CUDA packed W1A4 BITSERIAL dispatch",
            8: "CUDA packed W1A8 INT8 dispatch",
        }
        if markers[bits] not in log.read_text(errors="replace"):
            raise ValueError(
                "precision gate lacks independently hashed packed native dispatch proof"
            )
    for field, filename in (
        ("cache_index", "heads.draft_cache.jsonl"),
        ("cache_rows", "heads.draft_cache.f16"),
        ("cache_masks", "heads.draft_cache.mask"),
        ("graph_index", "heads.draft_graph.jsonl"),
        ("graph_values", "heads.draft_graph.f32"),
    ):
        checked_record(
            {"path": str(native_directory / filename), "sha256": cache["source_sha256"][field]}
        )
    if report.get("domains") != ["code", "prose", "reasoning"]:
        raise ValueError("precision gate requires all three declared training domains")
    if any(
        sum(root["domain"] == domain for root in report["roots"]) < 2
        for domain in report["domains"]
    ):
        raise ValueError("precision gate requires two actual roots per domain")
    for root in report["roots"]:
        if (
            not all(
                np.isfinite(root[name])
                for name in ("state_relative_rms", "logit_relative_rms", "decision_margin", "loss")
            )
            or root["state_relative_rms"] > 0.10
            or root["logit_relative_rms"] > 0.10
            or root["finite_gradient_tensors"] != expected_gradient_tensors
            or not np.isfinite(root["loss"])
            or (not root["top_choice_matches"] and root["decision_margin"] > 0.02)
        ):
            raise ValueError("precision gate measured row violates numeric/backward limits")


def _publish_gate_report(report: dict, bits: int, common: dict, report_path: Path) -> None:
    """Preserve each diagnostic candidate; only validated reports are readiness."""
    attempt = str(time.time_ns())
    candidate_path = report_path.parent / f"gate-candidate-{attempt}.json"
    write_json(
        candidate_path,
        {
            **report,
            "diagnostic_attempt": attempt,
            "training_eligible": False,
            "readiness_evidence": False,
        },
    )
    try:
        validate_gate_report(report, bits, common)
    except Exception as error:
        checks = report.get("checks", {})
        write_json(
            report_path.parent / f"gate-failure-{attempt}.json",
            {
                "schema": "w1ax_continuous_gate_failure_v1",
                "diagnostic_attempt": attempt,
                "activation_bits": bits,
                "optimizer_steps": report.get("optimizer_steps"),
                "training_eligible": False,
                "readiness_evidence": False,
                "candidate": file_record(candidate_path),
                "validation_error_type": type(error).__name__,
                "validation_error": str(error),
                "failed_checks": sorted(name for name, value in checks.items() if value is not True)
                if isinstance(checks, dict)
                else [],
            },
        )
        raise
    write_json(report_path, report)


def checkpoint_recipe(manifest: dict) -> dict:
    result = {
        "schema_version": manifest["schema_version"],
        "activation_bits": manifest["activation_bits"],
        "activation_quantizers": manifest.get("activation_quantizers"),
        "fusion_correction": manifest.get("fusion_correction"),
    }
    if manifest.get("affine_weights") is not None:
        result["affine_weights"] = manifest["affine_weights"]
    return result


def checkpoint_joint_config(
    manifest_path, bits, base_hash, *, device="cpu", allow_accelerator=False
):
    """Construct declared deployment options before installing modules."""
    from export_recurrent_binary import check_manifest

    from w1a1_eagle.affine_binary import AffineBinaryConfig
    from w1a1_eagle.fusion_correction import FusionCorrectionConfig
    from w1a1_eagle.recurrent_qat import JointQATConfig, W1AxContract

    manifest = json.loads(Path(manifest_path).read_text())
    check_manifest(manifest, base_hash)
    if (
        manifest["schema_version"] not in (2, 3, 4, 5)
        or manifest["activation_bits"] != bits
        or manifest["objective"] != "hard_ce"
    ):
        raise ValueError("precision checkpoint options differ")
    correction = manifest.get("fusion_correction")
    config = (
        None
        if correction is None
        else FusionCorrectionConfig(
            enabled=True,
            rank=correction["rank"],
            output_bias=correction["bias_name"] is not None,
            bias_bound=correction["bias_bound"] if correction["bias_name"] is not None else 0.1,
        )
    )
    return JointQATConfig(
        W1AxContract(bits, "row"),
        device=device,
        allow_accelerator=allow_accelerator,
        activation_quantization="learned"
        if manifest.get("activation_quantizers") is not None
        else "fixed",
        fusion_correction=config,
        affine_weights=AffineBinaryConfig(
            enabled=True, coverage=manifest["affine_weights"]["coverage"]
        )
        if manifest.get("affine_weights") is not None
        else None,
    )


def _load_checkpoint(path, manifest_path, linears, bits, base_hash):
    """Load effective deployed state, validating all arrays before any mutation.

    This is a zero-update deployment replay, not a training optimizer resume:
    correction factors in the export checkpoint are effective F16 tensors.
    """
    from export_recurrent_binary import (
        check_manifest,
        load_affine_weights,
        load_checkpoint,
        load_fusion_correction,
    )

    from w1a1_eagle.learned_activation import BOUNDARY_PATHS, LearnedActivationBank

    path, manifest_path = Path(path), Path(manifest_path)
    manifest = json.loads(Path(manifest_path).read_text())
    expected = check_manifest(manifest, base_hash)
    if (
        manifest["schema_version"] not in (2, 3, 4, 5)
        or manifest["activation_bits"] != bits
        or manifest["objective"] != "hard_ce"
        or manifest["checkpoint_sha256"] != sha256(path)
    ):
        raise ValueError("continuous precision checkpoint contract differs")
    correction = manifest.get("fusion_correction")
    extras = (
        load_fusion_correction(Path(path), correction, expected["fc"][1])
        if correction is not None
        else {}
    )
    affine = manifest.get("affine_weights")
    midpoints = load_affine_weights(path, affine, expected) if affine is not None else {}
    load_checkpoint(Path(path), expected, row_scale=True, extra_names=set(extras) | set(midpoints))
    if set(linears) != {name.removesuffix(".weight") for name, _ in expected.values()}:
        raise ValueError("checkpoint linear inventory differs")
    quantizers = manifest.get("activation_quantizers")
    bank = LearnedActivationBank.from_attached(linears) if quantizers is not None else None
    if (
        quantizers is not None
        and bits != 1
        and any(item["clip_ratio"] < 2**-16 for item in quantizers["boundaries"].values())
    ):
        raise ValueError("deployed clip parameter is outside trainable quantizer contract")
    if quantizers is None and any(
        getattr(module, "activation_quantizer", None) is not None for module in linears.values()
    ):
        raise ValueError("checkpoint has undeclared activation quantizer")
    attached = getattr(linears["fc"], "fusion_correction", None)
    if (attached is None) != (correction is None) or (
        attached is not None and attached.native_payload()[0] != correction
    ):
        raise ValueError("checkpoint fusion correction differs from attached config")
    affine_modules = {
        name: module.affine_binary
        for name, module in linears.items()
        if getattr(module, "affine_binary", None) is not None
    }
    expected_affine = (
        {}
        if affine is None
        else {
            expected[base][0].removesuffix(".weight"): tensor
            for base, tensor in affine["tensors"].items()
        }
    )
    if set(affine_modules) != set(expected_affine) or any(
        not module.config.enabled or module.config.coverage != affine["coverage"]
        for module in affine_modules.values()
    ):
        raise ValueError("checkpoint affine midpoint coverage differs from attachment")
    with np.load(path, allow_pickle=False) as archive:
        for name, module in affine_modules.items():
            values = archive[expected_affine[name]]
            if values.shape != tuple(module.midpoint.shape) or (
                module.config.midpoint_bound is not None
                and (np.abs(values) > module.config.midpoint_bound).any()
            ):
                raise ValueError("checkpoint affine midpoint shape/bound differs from attachment")
    for name, shape in expected.values():
        module = linears[name.removesuffix(".weight")]
        if (
            tuple(module.latent_sign.shape) != shape
            or module.contract.activation_bits != bits
            or module.contract.scale_layout != "row"
            or getattr(module, "_round_hard_signs", None) is not None
        ):
            raise ValueError("checkpoint shape/precision/cache differs from installed module")
    with np.load(path, allow_pickle=False) as archive, torch.no_grad():
        for name, _ in expected.values():
            module = linears[name.removesuffix(".weight")]
            module.latent_sign.copy_(
                torch.from_numpy(archive[name + ".latent"]).to(module.latent_sign)
            )
            module.initial_scale.copy_(
                torch.from_numpy(archive[name + ".scale"]).to(module.initial_scale)
            )
            module.scale_offset.zero_()
        if bank is not None:
            for boundary in BOUNDARY_PATHS:
                item = quantizers["boundaries"][boundary]
                bank.quantizers[boundary].parameter.fill_(
                    item["threshold_delta"] if bits == 1 else item["clip_ratio"]
                )
        if attached is not None:
            attached.u.copy_(torch.from_numpy(extras[correction["u_name"]]).to(attached.u))
            attached.v.copy_(torch.from_numpy(extras[correction["v_name"]]).to(attached.v))
            if attached.output_bias is not None:
                attached.output_bias.copy_(
                    torch.from_numpy(extras[correction["bias_name"]]).to(attached.output_bias)
                )
        for name, module in affine_modules.items():
            # Export audit permutes Q/K into GGUF order; live modules retain
            # the original checkpoint row order from the actual NPZ array.
            module.midpoint.copy_(
                torch.from_numpy(archive[expected_affine[name]]).to(module.midpoint)
            )
    return checkpoint_recipe(manifest)


def _replay_head(packed, scales, k, state, bits, *, quantizer=None, midpoint=None):
    from w1a1_eagle.recurrent_qat import hard_activation

    if quantizer is not None:
        result = quantizer(state)
        activation, activation_scale = result.codes.float(), result.scale
    elif bits == 1:
        state_f32 = state.to(dtype=torch.float32)
        raw = state_f32.view(torch.int32)
        negative = ((raw & -2147483648) != 0) & ((raw & 2147483647) != 0)
        activation = torch.where(negative, -torch.ones_like(state_f32), torch.ones_like(state_f32))
        activation_scale = state_f32.abs().double().mean(dim=-1, keepdim=True).float()
    elif midpoint is not None:
        from w1a1_eagle.recurrent_qat import hard_activation_with_codes

        values, activation_scale, _, codes = hard_activation_with_codes(state, bits)
        activation = values if codes is None else codes.float()
        if codes is None:
            activation_scale = None
    else:
        activation = hard_activation(state, bits)[0]
    outputs = []
    for start in range(0, len(scales), 4096):
        words = np.ascontiguousarray(packed[start : start + 4096]).view(np.uint8)
        signs = np.unpackbits(words, axis=1, bitorder="little")[:, :k].astype(np.float32) * 2 - 1
        weight = torch.from_numpy(signs).to(state.device)
        scale = torch.from_numpy(scales[start : start + 4096]).to(state.device)
        if bits == 1 or quantizer is not None or midpoint is not None:
            # Unscaled +/-1 sums are exact integers at the native head widths.
            # Native rounds dot*weight_scale to F32 before activation_scale.
            output = torch.nn.functional.linear(activation, weight) * scale.to(dtype=torch.float32)
            if activation_scale is not None:
                output = output * activation_scale
            if midpoint is not None:
                mu = midpoint[start : start + 4096].to(device=state.device, dtype=torch.float32)
                delta = activation.sum(dim=-1, keepdim=True) * mu
                output = output + (delta if activation_scale is None else delta * activation_scale)
            outputs.append(output)
        else:
            outputs.append(torch.nn.functional.linear(activation, weight) * scale)
    return torch.cat(outputs)


def run_gate(
    sources: dict,
    prompts: Path,
    output: Path,
    bits: int,
    *,
    expected_prompt_sha256: str,
    checkpoint_bundle: dict | None = None,
    measurement_binding: dict | None = None,
) -> Path:
    """Execute capture then checker sequentially; no optimizer updates."""
    from audit_recurrent_draft_cache import audit as audit_cache
    from check_w1ax_pilot_trajectory import (
        _frozen_identity_check,
        _packed_head,
        check_cache_contract,
        relative_rms,
        top_two_margin,
    )
    from evaluate_pytorch_w1a1 import verify_model_snapshot
    from export_recurrent_binary import export_model
    from prepare_w1ax_checkpoint_zero import prepare as checkpoint_zero
    from w1ax_capture_provider import ANGELSLIM_REVISION
    from w1ax_continuous_stages import require_unsealed_prompts

    from w1a1_eagle.frozen_operands import FrozenOperands
    from w1a1_eagle.native_step import NativeStepAdapter, bind_frozen_norms
    from w1a1_eagle.official_loader import load_official_eagle3
    from w1a1_eagle.recurrent_loss import supported_prefix_ce
    from w1a1_eagle.recurrent_provider import (
        ProviderRound,
        audit_provider_round,
        forward_torch_round,
    )
    from w1a1_eagle.recurrent_qat import (
        install_joint_linears,
        joint_optimizer,
        joint_parameter_families,
        shared_round_hard_signs,
    )
    from w1a1_eagle.recurrent_rollout import rebuild_prefix_cache

    require_unsealed_prompts(prompts, sources=sources, expected_sha256=expected_prompt_sha256)
    if bits not in {1, 4, 8} or sha256(prompts) != expected_prompt_sha256:
        raise ValueError("gate arithmetic or frozen prompt hash differs")
    modern = checkpoint_bundle is not None or bits == 4
    if checkpoint_bundle is not None:
        if set(checkpoint_bundle) != {"checkpoint", "checkpoint_manifest"}:
            raise ValueError("gate checkpoint bundle inventory differs")
        checkpoint, checkpoint_manifest = (
            checked_record(checkpoint_bundle[name])
            for name in ("checkpoint", "checkpoint_manifest")
        )
    prompt_rows = [json.loads(line) for line in prompts.read_text().splitlines() if line.strip()]
    by_domain = {p["domain"]: p["id"] for p in prompt_rows}
    if len(prompt_rows) != 3 or set(by_domain) != {"prose", "reasoning", "code"}:
        raise ValueError("bounded gate needs one training prompt in each of prose/reasoning/code")
    report_path = output / "gate.json"
    common = {
        k: sources["sha256"][k]
        for k in (
            "target_gguf",
            "candidate_d_gguf",
            "base_draft_gguf",
            "absolute_d2t",
            "model_snapshot_manifest",
        )
    }
    if report_path.exists():
        saved_report = json.loads(report_path.read_text())
        validate_gate_report(saved_report, bits, common)
        if checkpoint_bundle is not None and any(
            saved_report["evidence"][name] != checkpoint_bundle[name] for name in checkpoint_bundle
        ):
            raise ValueError("cached gate belongs to different actor checkpoint")
        if modern and saved_report.get("schema") != RECIPE_SCHEMA:
            raise ValueError("new recipe cannot inherit a legacy gate")
        if measurement_binding is not None:
            artifact = output / "native-decisions.json"
            if not artifact.is_file():
                raise ValueError(
                    "cached gate lacks bound decision artifact; use a new explicit gate output"
                )
            measured = json.loads(artifact.read_text())
            if (
                measured.get("deployment_state_sha256")
                != saved_report.get("deployment_state_sha256")
                or measured.get("activation_bits") != bits
                or any(measured.get(key) != value for key, value in measurement_binding.items())
            ):
                raise ValueError(
                    "cached decision artifact differs from requested deployment/source/recipe"
                )
        return report_path
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoint-zero"
    if checkpoint_bundle is None and not checkpoint_dir.exists():
        _reclaim_and_admit_host(
            output,
            "bounded checkpoint-zero CPU initialization",
            "host-memory-checkpoint-zero-admission.json",
        )
        checkpoint_zero(
            Path(sources["model_snapshot_manifest"]),
            Path(sources["base_draft_gguf"]),
            common["base_draft_gguf"],
            bits,
            checkpoint_dir,
        )
    if checkpoint_bundle is None:
        checkpoint, checkpoint_manifest = (
            checkpoint_dir / "joint.npz",
            checkpoint_dir / "joint.json",
        )
    config = checkpoint_joint_config(
        checkpoint_manifest,
        bits,
        common["base_draft_gguf"],
        device="cuda:0",
        allow_accelerator=True,
    )
    exported = output / "drafter.gguf"
    export_audit = output / "export-audit.json"
    if not exported.exists():
        write_json(
            export_audit,
            export_model(
                Path(sources["base_draft_gguf"]), checkpoint, checkpoint_manifest, exported
            ),
        )
    if modern:
        from w1ax_capture_provider import validate_actor_export

        validate_actor_export(
            {
                "checkpoint": file_record(checkpoint),
                "checkpoint_manifest": file_record(checkpoint_manifest),
                "export": file_record(exported),
                "export_audit": file_record(export_audit),
            },
            activation_bits=bits,
            base_hash=common["base_draft_gguf"],
        )
    native = output / "native"
    if not (native / "d_d/manifest.json").exists():
        if native.exists():
            raise RuntimeError("partial native gate remains preserved; explicit recovery required")
        native_capture(
            sources,
            prompts,
            native,
            activation_bits=bits,
            draft=exported,
            cache_gate=True,
            tokens=32,
        )
    labels = output / "labels"
    if not labels.exists():
        build_native_labels(native, prompts, Path(sources["absolute_d2t"]), labels, split="train")
    capture = load_native_labels(
        labels / "manifest.json",
        expected_prompt_sha256=expected_prompt_sha256,
        expected_prompt_count=3,
    )
    cache_report = audit_cache(native / "d_d")
    if cache_report["execution_device"] != "cuda":
        raise ValueError("gate needs actual native CUDA cache execution")
    cache_audit = output / "cache-audit.json"
    write_json(cache_audit, cache_report)
    snapshot = json.loads(Path(sources["model_snapshot_manifest"]).read_text())
    for role in ("draft", "target"):
        verify_model_snapshot(Path(snapshot["models"][role]["directory"]), snapshot["models"][role])
    _reclaim_and_admit_host(
        output,
        "bounded numeric gate CPU target/draft load",
        "host-memory-numeric-admission.json",
    )
    model = load_official_eagle3(
        Path(snapshot["models"]["target"]["directory"]),
        Path(snapshot["models"]["draft"]["directory"]),
        angelslim_revision=ANGELSLIM_REVISION,
        total_token=60,
        depth=5,
        top_k=10,
        threshold=1.0,
        target_load_kwargs={"dtype": torch.float16, "device_map": "cpu"},
    )
    drafter, target = model.eagle_layer, model.base_model
    linears = install_joint_linears(drafter, target, config)
    drafter.to("cuda:0")
    recipe = _load_checkpoint(
        checkpoint, checkpoint_manifest, linears, bits, common["base_draft_gguf"]
    )
    families = joint_parameter_families(linears)
    parameter_counts = {name: len(parameters) for name, parameters in families.items()}
    optimizer = joint_optimizer(linears, config)
    owned = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if len(owned) != sum(parameter_counts.values()) or len({id(p) for p in owned}) != len(owned):
        raise ValueError("native gate optimizer ownership does not match declared recipe")
    deployment_state_sha256 = None
    if modern or measurement_binding is not None:
        from w1a1_eagle.qat_state import deployment_state_sha256 as deployment_digest

        deployment_state_sha256 = deployment_digest(linears)
    operands = FrozenOperands(
        Path(sources["target_gguf"]),
        Path(sources["candidate_d_gguf"]),
        target_sha256=common["target_gguf"],
        draft_sha256=common["candidate_d_gguf"],
        vocab_size=151936,
        hidden_size=drafter.config.hidden_size,
    )
    bind_frozen_norms(drafter, operands.norm_arrays)
    adapter = NativeStepAdapter(drafter)
    packed, scales, k = _packed_head(exported)
    native_heads = [
        json.loads(line) for line in (native / "d_d/heads.jsonl").read_text().splitlines()
    ]
    state_values = np.memmap(native / "d_d/heads.f32", mode="r", dtype="<f4").reshape(-1, k)
    tasks = json.loads((native / "task_prompt_ids.json").read_text())
    root_index = {
        (tasks[str(r["task_id"])], tuple(r["prefix_token_ids"])): r
        for r in native_heads
        if r["depth"] == 0
    }
    manifest = json.loads((labels / "manifest.json").read_text())
    f = manifest["files"]
    offsets = np.load(labels / f["offsets"]["path"], allow_pickle=False)
    provider = type(
        "GateProvider",
        (),
        {
            "d2t_offsets": offsets,
            "target_vocab_size": 151936,
            "draft_vocab_size": 32000,
            "max_depth": 5,
            "allowed_prompt_ids": set(by_domain.values()),
            "split": "train",
        },
    )()
    roots, all_tokens, later_paths = [], set(), []
    for domain, prompt_id in sorted(by_domain.items()):
        selected = [key for key in sorted(capture.anchors) if key[0] == prompt_id]
        # Include a later-label chain when available, then one later prefix.
        selected.sort(key=lambda key: (-sum(r["valid"] for r in capture.rows[key]), key[1]))
        selected = selected[:2]
        if len(selected) < 2:
            raise ValueError(f"native {domain} gate has fewer than two audited roots")
        for key in selected:
            raw = capture.round_inputs(*key)
            native_row = root_index[(prompt_id, tuple(raw.prefix_token_ids))]
            native_state = torch.from_numpy(
                np.array(state_values[native_row["state_row"]], copy=True)
            ).to("cuda:0")
            batch = ProviderRound(
                raw.anchor,
                raw.rows,
                raw.prefix_token_ids,
                raw.raw_target_features.to("cuda:0"),
                raw.feature_positions,
                sha256(labels / "manifest.json"),
            )
            audit = audit_provider_round(batch, provider)
            all_tokens.update(batch.prefix_token_ids)
            all_tokens.update(row["input_token_id"] for row in batch.rows if row["valid"])
            with torch.no_grad(), shared_round_hard_signs(linears):
                rebuilt = rebuild_prefix_cache(
                    batch.prefix_token_ids,
                    batch.raw_target_features,
                    batch.feature_positions,
                    parent_position=len(batch.anchor.prefix_token_ids) - 1,
                    encode_feature=adapter.encode_feature,
                    decode_context=adapter.decode_context,
                    new_cache=adapter.new_cache,
                )
                step = adapter.decode_step(
                    rebuilt.seed_token,
                    adapter.encode_feature(rebuilt.seed_raw_features),
                    rebuilt.decoder_position,
                    rebuilt.cache,
                )
                normalized = adapter._rms_norm(step.pre_norm, drafter.norm)
                replay = _replay_head(
                    packed,
                    scales,
                    k,
                    native_state,
                    bits,
                    quantizer=getattr(linears["lm_head"], "activation_quantizer", None),
                    midpoint=getattr(
                        getattr(linears["lm_head"], "affine_binary", None), "midpoint", None
                    ),
                )
                state_rms = relative_rms(normalized.cpu().numpy(), native_state.cpu().numpy())
                logit_rms = relative_rms(step.logits.cpu().numpy(), replay.cpu().numpy())
                absolute_id = int(
                    offsets[int(torch.argmax(step.logits))] + int(torch.argmax(step.logits))
                )
                matches = absolute_id == native_row["proposed_token_id"]
                margin = top_two_margin(replay)
            saved_context, saved_step = adapter.decode_context, adapter.decode_step
            positions, states, caches = [], [], []

            def checked_context(token, feature, position, cache):
                result = saved_context(token, feature, position, cache)
                check_cache_contract(result.cache, position + 1, torch.device("cuda:0"))
                return result

            def checked_step(token, feature, position, cache, **kwargs):
                result = saved_step(token, feature, position, cache, **kwargs)
                check_cache_contract(result.cache, position + 1, torch.device("cuda:0"))
                if kwargs.get("compute_logits", True):
                    positions.append(position)
                    result.pre_norm.retain_grad()
                    result.cache.key.retain_grad()
                    result.cache.value.retain_grad()
                    states.append(result.pre_norm)
                    caches.append(result.cache)
                return result

            adapter.decode_context, adapter.decode_step = checked_context, checked_step
            try:
                with shared_round_hard_signs(linears):
                    logits = forward_torch_round(batch, adapter, 32000)
            finally:
                adapter.decode_context, adapter.decode_step = saved_context, saved_step
            expected_positions = [r["input_position"] - 1 for r in raw.rows if r["valid"]]
            if positions != expected_positions or not torch.isfinite(logits).all():
                raise ValueError("Torch exact-prefix positions/nonfinite logits gate failed")
            loss = supported_prefix_ce(logits, audit)
            loss.backward(retain_graph=True)
            grads = [parameter.grad for parameter in owned]
            finite = sum(g is not None and bool(torch.isfinite(g).all()) for g in grads)
            base_grads = [p.grad for name in ("sign", "scale") for p in families[name]]
            if finite != len(owned) or (
                config.affine_weights is None
                and not all(bool(g.abs().sum() > 0) for g in base_grads)
            ):
                raise ValueError(
                    "joint hard CE requires finite declared-family and nonzero sign/scale gradients"
                )
            later = [i for i, ok in enumerate(audit.ce_mask) if ok and i > 0]
            if later:
                for state in states:
                    state.grad = None
                for cache in caches:
                    cache.key.grad = cache.value.grad = None
                last = later[-1]
                later_loss = torch.nn.functional.cross_entropy(
                    logits[last : last + 1],
                    torch.tensor([audit.draft_labels[last]], device=logits.device),
                )
                later_loss.backward()

                def positive_finite(grad):
                    return (
                        grad is not None
                        and bool(torch.isfinite(grad).all())
                        and bool(grad.abs().sum() > 0)
                    )

                path = positive_finite(states[0].grad) and (
                    positive_finite(
                        caches[0].key.grad[:, -1:, :] if caches[0].key.grad is not None else None
                    )
                    and positive_finite(
                        caches[0].value.grad[:, -1:, :]
                        if caches[0].value.grad is not None
                        else None
                    )
                )
                later_paths.append(path)
                if not path:
                    raise ValueError("later loss does not reach earlier student state/cache")
            roots.append(
                {
                    "domain": domain,
                    "prompt_id": prompt_id,
                    "round_index": key[1],
                    "prefix_token_ids": list(batch.prefix_token_ids),
                    "loss": float(loss.detach()),
                    "state_relative_rms": state_rms,
                    "logit_relative_rms": logit_rms,
                    "top_choice_matches": matches,
                    "decision_margin": margin,
                    "torch_choice": absolute_id,
                    "native_choice": native_row["proposed_token_id"],
                    "capture_id": sha256(labels / "manifest.json"),
                    "finite_gradient_tensors": finite,
                    "supported_rows": sum(audit.ce_mask),
                }
            )
            optimizer.zero_grad(set_to_none=True)
            states.clear()
            caches.clear()
            if later:
                del later_loss
            del logits, loss, batch
    _frozen_identity_check(
        operands, drafter, exported, Path(sources["candidate_d_gguf"]), offsets, all_tokens
    )
    export_result = json.loads(export_audit.read_text())
    checks = {name: True for name in CHECKS}
    checks["export_serialization_and_frozen_operands"] = export_result["serialization_audit_passed"]
    checks["exact_prefix_states_and_logits"] = all(
        r["state_relative_rms"] <= 0.10 and r["logit_relative_rms"] <= 0.10 for r in roots
    )
    checks["no_high_margin_changed_native_decisions"] = all(
        r["top_choice_matches"] or r["decision_margin"] <= 0.02 for r in roots
    )
    checks["later_loss_reaches_earlier_student_state_and_cache"] = bool(later_paths) and all(
        later_paths
    )
    report = {
        "schema": RECIPE_SCHEMA if modern else SCHEMA,
        "activation_bits": bits,
        "scale_layout": "row",
        "objective": "hard_ce",
        "optimizer_steps": 0,
        "execution_device": "cuda:0",
        "hardware": torch.cuda.get_device_name(0),
        "checks": checks,
        "native_binary_sha256": sources["sha256"]["binary"],
        "native_runtime": sources["native_runtime"],
        "domains": sorted(by_domain),
        "roots": roots,
        "common_source_sha256": common,
        "limits": {"relative_rms": 0.10, "decision_margin": 0.02},
        "evidence": {
            "export_audit": file_record(export_audit),
            "native_cache_audit": file_record(cache_audit),
            "native_capture_manifest": file_record(labels / "manifest.json"),
            "checkpoint": file_record(checkpoint),
            "checkpoint_manifest": file_record(checkpoint_manifest),
            "export": file_record(exported),
            "native_cell": file_record(native / "d_d/manifest.json"),
        },
        "limits_of_claim": "bounded train roots; no held-out acceptance or speed claim",
        "native_capture_budget": {
            "prompts": 3,
            "max_outputs_per_prompt": 32,
            "decoder_executions_conservative_upper": 3 * 32 * 6 + 3,
            "max_executions": 1024,
            "max_graph_bytes": 1024**3,
            "scope": "cache projection operands and stored writes/masks",
        },
    }
    if modern:
        report.update(
            recipe=recipe,
            deployment_state_sha256=deployment_state_sha256,
            trainable_parameter_counts=parameter_counts,
        )
    if measurement_binding is not None:
        required = {"source_sha256", "training_runtime", "recipe", "native_commit"}
        if set(measurement_binding) != required:
            raise ValueError("native decision measurement binding inventory differs")
        from w1a1_eagle.trajectory_refresh import checked_hash

        checked_hash(measurement_binding["source_sha256"])
        import re

        if not re.fullmatch("[0-9a-f]{40}", measurement_binding["native_commit"]):
            raise ValueError("native decision measurements require pinned native commit")
        props = torch.cuda.get_device_properties(0)
        write_json(
            output / "native-decisions.json",
            {
                "schema": "qat_native_measurements_v1",
                "kind": "native_decisions",
                "split": "train",
                "activation_bits": bits,
                "deployment_state_sha256": deployment_state_sha256,
                **measurement_binding,
                "backend": "cuda",
                "hardware": {
                    "device_type": "cuda",
                    "name": props.name,
                    "compute_capability": [props.major, props.minor],
                    "total_memory_bytes": props.total_memory,
                },
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
                    for r in roots
                ],
            },
        )
    _publish_gate_report(report, bits, common, report_path)
    del adapter, linears, drafter, target, model
    gc.collect()
    torch.cuda.empty_cache()
    return report_path

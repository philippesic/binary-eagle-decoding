"""USER-started bounded native/CUDA gates for separate row W1A8 and W1A1.

No model, accelerator query or native process is touched on import. This is a
training-deployment gate, not a throughput or held-out accuracy claim. Numeric
checks are restricted to exact captured roots and actionable decision margins.
"""

from __future__ import annotations

import gc
import json
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
CHECKS = (
    "export_serialization_and_frozen_operands",
    "native_stored_f16_cache_and_masks",
    "exact_prefix_states_and_logits",
    "no_high_margin_changed_native_decisions",
    "all_nine_finite_gradients",
    "later_loss_reaches_earlier_student_state_and_cache",
    "torch_f16_cache_and_positions",
)


def validate_gate_report(report: dict, bits: int, common: dict) -> None:
    if (
        report.get("schema") != SCHEMA
        or report.get("activation_bits") != bits
        or bits not in {1, 8}
        or report.get("execution_device") != "cuda:0"
        or report.get("scale_layout") != "row"
        or report.get("objective") != "hard_ce"
        or report.get("optimizer_steps") != 0
        or report.get("common_source_sha256") != common
        or set(report.get("checks", {})) != set(CHECKS)
        or not all(value is True for value in report["checks"].values())
        or report.get("limits") != {"relative_rms": 0.10, "decision_margin": 0.02}
        or len(report.get("roots", [])) < 6
    ):
        raise ValueError("independent A8/A1 numeric/cache/export/backward gate failed")
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
    if (
        checkpoint.get("activation_bits") != bits
        or checkpoint.get("scale_layout") != "row"
        or checkpoint.get("objective") != "hard_ce"
        or checkpoint.get("activation_rule")
        != "a16_f16_cast_a8a4_absmax_even_a1_f64_meanabs_sign_zero_positive"
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
            or root["finite_gradient_tensors"] != 18
            or not np.isfinite(root["loss"])
            or (not root["top_choice_matches"] and root["decision_margin"] > 0.02)
        ):
            raise ValueError("precision gate measured row violates numeric/backward limits")


def _load_checkpoint(path, manifest_path, linears, bits, base_hash):
    from w1a1_eagle.recurrent_training import CHECKPOINT_NAMES

    m = json.loads(Path(manifest_path).read_text())
    if (
        m.get("schema_version") != 2
        or m.get("activation_bits") != bits
        or m.get("scale_layout") != "row"
        or m.get("objective") != "hard_ce"
        or m.get("base_gguf_sha256") != base_hash
        or m.get("checkpoint_sha256") != sha256(path)
    ):
        raise ValueError("continuous precision checkpoint contract differs")
    with np.load(path, allow_pickle=False) as archive, torch.no_grad():
        if set(archive.files) != {
            n + s for n in CHECKPOINT_NAMES.values() for s in (".latent", ".scale")
        }:
            raise ValueError("checkpoint must own exactly all nine sign/scale pairs")
        for name in CHECKPOINT_NAMES.values():
            module = linears[name.removesuffix(".weight")]
            latent, scale = archive[name + ".latent"], archive[name + ".scale"]
            if (
                latent.dtype != np.float32
                or scale.dtype != np.float32
                or latent.shape != tuple(module.latent_sign.shape)
                or scale.shape != tuple(module.initial_scale.shape)
                or not np.isfinite(latent).all()
                or not np.isfinite(scale).all()
                or (scale < 0).any()
            ):
                raise ValueError("checkpoint sign/scale array invalid")
            module.latent_sign.copy_(torch.from_numpy(latent).to(module.latent_sign.device))
            module.scale_offset.copy_(
                torch.from_numpy(scale).to(module.scale_offset.device) - module.initial_scale
            )


def _replay_head(packed, scales, k, state, bits):
    from w1a1_eagle.recurrent_qat import hard_activation

    activation = hard_activation(state, bits)[0]
    outputs = []
    for start in range(0, len(scales), 4096):
        words = np.ascontiguousarray(packed[start : start + 4096]).view(np.uint8)
        signs = np.unpackbits(words, axis=1, bitorder="little")[:, :k].astype(np.float32) * 2 - 1
        weight = torch.from_numpy(signs).to(state.device)
        scale = torch.from_numpy(scales[start : start + 4096]).to(state.device)
        outputs.append(torch.nn.functional.linear(activation, weight) * scale)
    return torch.cat(outputs)


def run_gate(
    sources: dict, prompts: Path, output: Path, bits: int, *, expected_prompt_sha256: str
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
        JointQATConfig,
        W1AxContract,
        install_joint_linears,
        shared_round_hard_signs,
    )
    from w1a1_eagle.recurrent_rollout import rebuild_prefix_cache

    if bits not in {1, 8} or sha256(prompts) != expected_prompt_sha256:
        raise ValueError("gate arithmetic or frozen prompt hash differs")
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
        validate_gate_report(json.loads(report_path.read_text()), bits, common)
        return report_path
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoint-zero"
    if not checkpoint_dir.exists():
        host_admission("bounded checkpoint-zero CPU initialization", 12 * 1024**3)
        checkpoint_zero(
            Path(sources["model_snapshot_manifest"]),
            Path(sources["base_draft_gguf"]),
            common["base_draft_gguf"],
            bits,
            checkpoint_dir,
        )
    checkpoint, checkpoint_manifest = checkpoint_dir / "joint.npz", checkpoint_dir / "joint.json"
    exported = output / "drafter.gguf"
    export_audit = output / "export-audit.json"
    if not exported.exists():
        write_json(
            export_audit,
            export_model(
                Path(sources["base_draft_gguf"]), checkpoint, checkpoint_manifest, exported
            ),
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
    config = JointQATConfig(W1AxContract(bits, "row"), device="cuda:0", allow_accelerator=True)
    host_admission("bounded numeric gate CPU target/draft load", 12 * 1024**3)
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
    _load_checkpoint(checkpoint, checkpoint_manifest, linears, bits, common["base_draft_gguf"])
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
                replay = _replay_head(packed, scales, k, native_state, bits)
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
            grads = [
                g
                for layer in linears.values()
                for g in (layer.latent_sign.grad, layer.scale_offset.grad)
            ]
            finite = sum(g is not None and bool(torch.isfinite(g).all()) for g in grads)
            if finite != 18 or not all(bool(g.abs().sum() > 0) for g in grads):
                raise ValueError("joint hard CE requires 18 finite sign/scale gradients")
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
                    "finite_gradient_tensors": finite,
                    "supported_rows": sum(audit.ce_mask),
                }
            )
            for layer in linears.values():
                layer.latent_sign.grad = layer.scale_offset.grad = None
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
        "schema": SCHEMA,
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
    validate_gate_report(report, bits, common)
    write_json(report_path, report)
    del adapter, linears, drafter, target, model
    gc.collect()
    torch.cuda.empty_cache()
    return report_path

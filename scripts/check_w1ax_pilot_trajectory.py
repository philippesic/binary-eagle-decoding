#!/usr/bin/env python3
"""Bounded CUDA row-A16 replay against three frozen native prompt roots.

This is a numerical diagnostic only. It does not approve a provider, training
run, model, or deployment. It joins student depth-zero roots to the audited
candidate-D bundle by the complete accepted-prefix token sequence, rebuilds
that prefix with the current row-A16 checkpoint, and compares normalized
states and logits against native captured states and an independent packed
row-head replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_recurrent_binary_capture import load_audited_capture  # noqa: E402
from evaluate_pytorch_w1a1 import verify_model_snapshot  # noqa: E402
from export_binary_rescue import Model  # noqa: E402
from w1ax_capture_provider import (  # noqa: E402
    ANGELSLIM_REVISION,
    DRAFT_REPO,
    DRAFT_REVISION,
    TARGET_REPO,
    TARGET_REVISION,
)

from w1a1_eagle.frozen_operands import FrozenOperands  # noqa: E402
from w1a1_eagle.native_step import NativeStepAdapter, bind_frozen_norms  # noqa: E402
from w1a1_eagle.official_loader import load_official_eagle3  # noqa: E402
from w1a1_eagle.recurrent_qat import (  # noqa: E402
    JointQATConfig,
    W1AxContract,
    install_joint_linears,
)
from w1a1_eagle.recurrent_rollout import rebuild_prefix_cache  # noqa: E402

CHECKPOINT_NAMES = {
    "fc": "fc.weight",
    "output": "lm_head.weight",
    "blk.0.attn_q": "midlayer.self_attn.q_proj.weight",
    "blk.0.attn_k": "midlayer.self_attn.k_proj.weight",
    "blk.0.attn_v": "midlayer.self_attn.v_proj.weight",
    "blk.0.attn_output": "midlayer.self_attn.o_proj.weight",
    "blk.0.ffn_gate": "midlayer.mlp.gate_proj.weight",
    "blk.0.ffn_up": "midlayer.mlp.up_proj.weight",
    "blk.0.ffn_down": "midlayer.mlp.down_proj.weight",
}
DOMAIN_PREFIXES = {"prose": "dolly:", "reasoning": "gsm8k:", "code": "mbpp:"}
CHECKPOINT_ZERO_SHA256 = "5b371f79831c4c8da0ffc4bc5a0a4b9a6817a1fdf7c2bddfeaec6b001c4f6b78"
CHECKPOINT_ZERO_EXPORT_SHA256 = "fa9406b72fb6ef19e2eca0bc0891bc20099dc318283721af3f540e056ebd3f45"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_rms(actual: np.ndarray, reference: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if (
        actual.shape != reference.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("relative-RMS operands must be same-shape and finite")
    return float(
        np.sqrt(np.mean((actual - reference) ** 2)) / max(np.sqrt(np.mean(reference**2)), 1e-8)
    )


def top_two_margin(logits: torch.Tensor) -> float:
    if logits.ndim != 1 or logits.numel() < 2 or not torch.isfinite(logits).all().item():
        raise ValueError("head logits must be a finite vector with at least two entries")
    top = torch.topk(logits.float(), 2).values
    return float((top[0] - top[1]).item())


def join_roots(
    student_heads: list[dict],
    task_to_prompt: dict[int, str],
    candidate_capture,
    student_rounds: list[dict] | None = None,
) -> tuple[
    dict[str, list[tuple[dict, object, str]]],
    dict[str, list[dict]],
    int,
    dict[str, set[tuple[int, ...]]],
]:
    """Join first and later common depth-zero roots in output-position order."""
    candidate_roots = {}
    for key in sorted(candidate_capture.anchors):
        prompt_id, round_index = key
        round_data = candidate_capture.round_inputs(prompt_id, round_index)
        candidate_roots.setdefault((prompt_id, tuple(round_data.prefix_token_ids)), round_data)

    grouped: dict[str, list[dict]] = {}
    unknown_task_rows = 0
    for row in student_heads:
        if row.get("depth") != 0:
            continue
        task_id = row.get("task_id")
        if type(task_id) is not int or task_id not in task_to_prompt:
            unknown_task_rows += 1  # warmups and unrelated requests stay unmatched
            continue
        prefix = row.get("prefix_token_ids")
        if (
            not isinstance(prefix, list)
            or not prefix
            or any(type(token) is not int for token in prefix)
        ):
            raise ValueError("student root lacks a complete integer prefix")
        prompt_id = task_to_prompt[task_id]
        grouped.setdefault(prompt_id, []).append(row)

    outcome = {}
    for row in student_rounds or []:
        key = (row.get("task_id"), row.get("round_index"))
        proposed, accepted = row.get("n_proposed"), row.get("n_accepted")
        if (
            type(key[0]) is int
            and type(key[1]) is int
            and type(proposed) is int
            and type(accepted) is int
            and 0 <= accepted <= proposed
        ):
            outcome[key] = (
                "accepted"
                if accepted > 0
                else ("rejected" if proposed > accepted else "no_acceptance")
            )

    selected: dict[str, list[tuple[dict, object, str]]] = {}
    unmatched: dict[str, list[dict]] = {}
    all_joined: dict[str, set[tuple[int, ...]]] = {}
    for prompt_id, rows in grouped.items():
        rows.sort(key=lambda row: (row.get("parent_position", -1), row.get("round_index", -1)))
        matches = []
        missing = []
        seen_prefixes = set()
        for row in rows:
            prefix = tuple(row["prefix_token_ids"])
            if prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            round_data = candidate_roots.get((prompt_id, prefix))
            if round_data is not None:
                prior = outcome.get((row["task_id"], row.get("round_index", 0) - 1), "unknown")
                matches.append((row, round_data, prior))
            else:
                missing.append(
                    {
                        "round_index": row.get("round_index"),
                        "parent_position": row.get("parent_position"),
                        "prefix_token_ids": list(prefix),
                    }
                )
        all_joined[prompt_id] = {tuple(item[0]["prefix_token_ids"]) for item in matches}
        # Freeze the earliest common root, then prefer one root after an
        # accepted continuation and one after a rejection when both are present.
        chosen = matches[:1]
        later = matches[1:]
        accepted = next((item for item in later if item[2] == "accepted"), None)
        rejected = next((item for item in later if item[2] == "rejected"), None)
        if accepted is not None:
            chosen.append(accepted)
        if rejected is not None and rejected not in chosen:
            chosen.append(rejected)
        if len(chosen) == 1 and later:
            chosen.append(later[0])
        selected[prompt_id] = chosen
        unmatched[prompt_id] = missing
    return selected, unmatched, unknown_task_rows, all_joined


def _validate_three_domains(task_to_prompt: dict[int, str]) -> dict[str, str]:
    prompts = set(task_to_prompt.values())
    result = {}
    for domain, prefix in DOMAIN_PREFIXES.items():
        matches = sorted(prompt for prompt in prompts if prompt.startswith(prefix))
        if len(matches) != 1:
            raise ValueError(f"frozen sample must contain exactly one {domain} prompt")
        result[domain] = matches[0]
    if len(prompts) != 3:
        raise ValueError("frozen sample must contain exactly three distinct source prompts")
    return result


def _load_row_checkpoint(path: Path, manifest_path: Path, linears: dict, base_hash: str) -> None:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version") != 2
        or manifest.get("scale_layout") != "row"
        or manifest.get("activation_bits") != 16
        or manifest.get("objective") != "hard_ce"
        or manifest.get("base_gguf_sha256") != base_hash
        or manifest.get("checkpoint_sha256") != sha256(path)
    ):
        raise ValueError("checkpoint manifest is not the pinned row-A16 hard-CE checkpoint")
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            name + suffix for name in CHECKPOINT_NAMES.values() for suffix in (".latent", ".scale")
        }
        if set(archive.files) != expected:
            raise ValueError("row checkpoint must contain exactly nine latent/scale pairs")
        with torch.no_grad():
            for name in CHECKPOINT_NAMES.values():
                module = linears[name]
                latent = archive[name + ".latent"]
                scales = archive[name + ".scale"]
                if latent.dtype != np.float32 or latent.shape != tuple(module.latent_sign.shape):
                    raise ValueError(f"{name}: checkpoint latent shape or dtype mismatch")
                if scales.dtype != np.float32 or scales.shape != tuple(module.initial_scale.shape):
                    raise ValueError(f"{name}: checkpoint scale shape or dtype mismatch")
                if (
                    not np.isfinite(latent).all()
                    or not np.isfinite(scales).all()
                    or (scales < 0).any()
                ):
                    raise ValueError(
                        f"{name}: checkpoint arrays must be finite and scales nonnegative"
                    )
                module.latent_sign.copy_(torch.from_numpy(latent).to(module.latent_sign.device))
                module.scale_offset.copy_(
                    torch.from_numpy(scales).to(module.scale_offset.device) - module.initial_scale
                )


def _packed_head(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    import gguf

    reader = gguf.GGUFReader(path)
    tensors = {item.name: item for item in reader.tensors}
    packed, scales = tensors.get("output.w1a1_packed"), tensors.get("output.w1a1_scale")
    if packed is None or scales is None or "output.weight" in tensors:
        raise ValueError("row-A16 export must contain only its packed output head")
    field = reader.get_field("eagle3.w1a1.tensor.output_weight.logical_k")
    if field is None:
        raise ValueError("exported row head lacks its logical input width")
    k = int(field.contents())
    bits = np.asarray(packed.data, dtype="<i4").view("<u4").copy()
    scale_values = np.asarray(scales.data, dtype=np.float32).copy()
    if bits.ndim != 2 or scale_values.shape != (bits.shape[0],) or bits.shape[1] != (k + 31) // 32:
        raise ValueError("exported packed row head has inconsistent dimensions")
    return bits, scale_values, k


def packed_head_replay(
    packed: np.ndarray,
    scales: np.ndarray,
    logical_k: int,
    states: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Replay packed row-A16 head in output chunks using explicit unpacked signs."""
    if states.ndim != 1 or states.shape[0] != logical_k or not torch.isfinite(states).all().item():
        raise ValueError("native head state has invalid width or nonfinite values")
    activation = states.to(torch.float16).float()
    rows, words = packed.shape
    outputs = []
    chunk = 4096
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        byte_view = packed[start:end].view(np.uint8)
        signs = np.unpackbits(byte_view, axis=1, bitorder="little")[:, :logical_k]
        weight = torch.from_numpy(signs.astype(np.float32) * 2.0 - 1.0).to(device)
        row_scale = torch.from_numpy(scales[start:end]).to(device)
        outputs.append(torch.nn.functional.linear(activation, weight) * row_scale)
    result = torch.cat(outputs)
    if result.numel() != rows or not torch.isfinite(result).all().item():
        raise ValueError("packed row-A16 replay produced invalid logits")
    return result


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("this bounded trajectory checker requires CUDA")
    if args.report.exists():
        raise FileExistsError(args.report)
    started = time.monotonic()
    task_map_raw = json.loads(args.student_task_map.read_text())
    if not isinstance(task_map_raw, dict):
        raise ValueError("student task map must be a JSON object")
    task_to_prompt = {int(key): value for key, value in task_map_raw.items()}
    domains = _validate_three_domains(task_to_prompt)

    candidate_capture = load_audited_capture(
        args.candidate_manifest,
        args.candidate_prompts,
        args.candidate_prompts_sha256,
        expected_prompt_count=31,
    )
    with args.student_heads.open() as stream:
        student_heads = [json.loads(line) for line in stream if line.strip()]
    with args.student_rounds.open() as stream:
        student_rounds = [json.loads(line) for line in stream if line.strip()]
    selected, unmatched_student, unknown_task_rows, all_joined = join_roots(
        student_heads, task_to_prompt, candidate_capture, student_rounds
    )
    if sha256(args.checkpoint) != CHECKPOINT_ZERO_SHA256:
        raise ValueError("row checkpoint differs from the frozen checkpoint-zero artifact")
    if sha256(args.row_export_gguf) != CHECKPOINT_ZERO_EXPORT_SHA256:
        raise ValueError("row export differs from the frozen checkpoint-zero artifact")
    unmatched_candidate = {}
    for domain, prompt_id in domains.items():
        candidate_roots = []
        for key in sorted(candidate_capture.anchors):
            if key[0] != prompt_id:
                continue
            item = candidate_capture.round_inputs(*key)
            candidate_roots.append((key[1], list(item.prefix_token_ids)))
        common = all_joined.get(prompt_id, set())
        unmatched_candidate[domain] = [
            {"round_index": index, "prefix_token_ids": prefix}
            for index, prefix in candidate_roots
            if tuple(prefix) not in common
        ]

    snapshot = json.loads(args.model_snapshot_manifest.read_text())
    model_dirs = {}
    for role, repo, revision in (
        ("target", TARGET_REPO, TARGET_REVISION),
        ("draft", DRAFT_REPO, DRAFT_REVISION),
    ):
        entry = snapshot.get("models", {}).get(role, {})
        if entry.get("repo") != repo or entry.get("revision") != revision:
            raise ValueError(f"{role} model snapshot differs from the frozen revision")
        model_dirs[role] = Path(entry["directory"])
        verify_model_snapshot(model_dirs[role], entry)
    config = JointQATConfig(W1AxContract(16, "row"), device="cuda:0", allow_accelerator=True)
    model = load_official_eagle3(
        model_dirs["target"],
        model_dirs["draft"],
        angelslim_revision=ANGELSLIM_REVISION,
        total_token=60,
        depth=5,
        top_k=10,
        threshold=1.0,
        target_load_kwargs={"dtype": torch.float16, "device_map": "cpu"},
    )
    model.eval()
    drafter, target = model.eagle_layer, model.base_model
    drafter.to("cuda:0")
    linears = install_joint_linears(drafter, target, config)
    _load_row_checkpoint(
        args.checkpoint, args.checkpoint_manifest, linears, sha256(args.base_draft_gguf)
    )
    operands = FrozenOperands(
        args.target_gguf,
        args.candidate_d_gguf,
        vocab_size=drafter.config.vocab_size,
        hidden_size=drafter.config.hidden_size,
    )
    bind_frozen_norms(drafter, operands.norm_arrays)
    adapter = NativeStepAdapter(drafter)
    packed, head_scales, logical_k = _packed_head(args.row_export_gguf)
    mapping = np.asarray(Model(args.row_export_gguf).tensors["d2t"].data)
    if logical_k != drafter.config.hidden_size or len(mapping) != packed.shape[0]:
        raise ValueError("exported row head or d2t map differs from the pinned drafter")
    native_states = np.memmap(args.student_states, dtype="<f4", mode="r")
    if native_states.size != len(student_heads) * logical_k:
        raise ValueError("student normalized-state payload row count differs from heads.jsonl")
    native_states = native_states.reshape(-1, logical_k)

    per_domain = {}
    for domain, prompt_id in domains.items():
        roots = selected.get(prompt_id, [])
        rows = []
        if len(roots) < 2:
            per_domain[domain] = {
                "prompt_id": prompt_id,
                "status": "insufficient_shared_roots",
                "matched_roots": len(roots),
                "roots": [],
            }
            continue
        for student_row, captured, preceding_outcome in roots:
            state_row = student_row.get("state_row")
            prefix = captured.prefix_token_ids
            if type(state_row) is not int or not 0 <= state_row < len(native_states):
                raise ValueError("selected native state row is out of range")
            raw = captured.raw_target_features.to(device="cuda:0")
            with torch.no_grad():
                rebuilt = rebuild_prefix_cache(
                    prefix,
                    raw,
                    captured.feature_positions,
                    parent_position=len(captured.anchor.prefix_token_ids) - 1,
                    encode_feature=adapter.encode_feature,
                    decode_context=adapter.decode_context,
                    new_cache=adapter.new_cache,
                )
                feature = adapter.encode_feature(rebuilt.seed_raw_features.to("cuda:0"))
                step = adapter.decode_step(
                    rebuilt.seed_token, feature, rebuilt.decoder_position, rebuilt.cache
                )
                normalized = adapter._rms_norm(step.pre_norm, drafter.norm)
            native = torch.from_numpy(np.array(native_states[state_row], copy=True)).to("cuda:0")
            replay_logits = packed_head_replay(
                packed, head_scales, logical_k, native, torch.device("cuda:0")
            )
            if step.logits.shape != replay_logits.shape:
                raise ValueError("Torch and packed head logit shapes differ")
            torch_logits = step.logits.float()
            native_rms = relative_rms(normalized.detach().cpu().numpy(), native.cpu().numpy())
            logit_rms = relative_rms(
                torch_logits.detach().cpu().numpy(), replay_logits.cpu().numpy()
            )
            top_id = int(torch.argmax(torch_logits).item())
            mapped_id = int(mapping[top_id])
            proposed = student_row.get("proposed_token_id")
            delta = (normalized - native).abs()
            rows.append(
                {
                    "round_index": student_row.get("round_index"),
                    "parent_position": student_row.get("parent_position"),
                    "state_row": state_row,
                    "prefix_token_ids": list(prefix),
                    "preceding_student_round_outcome": preceding_outcome,
                    "state_relative_rms": native_rms,
                    "state_max_abs_error": float(delta.max().item()),
                    "logit_relative_rms": logit_rms,
                    "logit_max_abs_error": float((torch_logits - replay_logits).abs().max().item()),
                    "native_packed_head_top_two_margin": top_two_margin(replay_logits),
                    "torch_top_two_margin_raw_logit": top_two_margin(torch_logits),
                    "torch_top1_target_id": mapped_id,
                    "native_proposed_target_id": proposed,
                    "top1_target_id_matches_native": type(proposed) is int
                    and mapped_id == proposed,
                    "proposal_margin_over_0_02": top_two_margin(torch_logits) > 0.02,
                }
            )
        per_domain[domain] = {
            "prompt_id": prompt_id,
            "status": "measured",
            "matched_roots": len(roots),
            "roots": rows,
        }

    limits = {"relative_rms": 0.10, "changed_top_choice_block_margin": 0.02}
    all_rows = [row for item in per_domain.values() for row in item.get("roots", [])]
    report = {
        "schema": "w1ax_pilot_torch_cuda_trajectory_v1",
        "status": "measured_numeric_gate_only",
        "eligibility_decision": "not_evaluated_by_this_checker",
        "execution_device": "CUDA",
        "precision": "Torch CUDA F32 compute with row-A16 F16 activation boundary",
        "domains": per_domain,
        "unmatched_student_roots": {
            domain: unmatched_student.get(prompt_id, []) for domain, prompt_id in domains.items()
        },
        "unmatched_candidate_roots": unmatched_candidate,
        "ignored_unmapped_student_depth_zero_rows": unknown_task_rows,
        "limits": limits,
        "checks": {
            "all_three_domains_have_two_roots": all(
                item.get("status") == "measured" for item in per_domain.values()
            ),
            "all_state_relative_rms_at_most_0_10": all(
                row["state_relative_rms"] <= 0.10 for row in all_rows
            ),
            "all_logit_relative_rms_at_most_0_10": all(
                row["logit_relative_rms"] <= 0.10 for row in all_rows
            ),
            "proposal_disagreements": sum(
                not row["top1_target_id_matches_native"] for row in all_rows
            ),
            "changed_top_choice_with_margin_over_0_02": sum(
                not row["top1_target_id_matches_native"] and row["proposal_margin_over_0_02"]
                for row in all_rows
            ),
        },
        "unverified_by_this_checker": [
            "selected-root cache values, cache lengths, causal masks and decoder-position parity",
            "F16 K/V operand identity and finite training gradients",
            "near-tie verifier acceptance and matched Q4_0 response-ID condition",
            "source and response ancestry hashes beyond the audited candidate-D capture",
        ],
        "input_sha256": {
            "candidate_manifest": sha256(args.candidate_manifest),
            "candidate_prompts": sha256(args.candidate_prompts),
            "student_heads": sha256(args.student_heads),
            "student_rounds": sha256(args.student_rounds),
            "student_states": sha256(args.student_states),
            "student_task_map": sha256(args.student_task_map),
            "checkpoint": sha256(args.checkpoint),
            "checkpoint_manifest": sha256(args.checkpoint_manifest),
            "row_export_gguf": sha256(args.row_export_gguf),
            "base_draft_gguf": sha256(args.base_draft_gguf),
            "target_gguf": sha256(args.target_gguf),
            "candidate_d_gguf": sha256(args.candidate_d_gguf),
            "model_snapshot_manifest": sha256(args.model_snapshot_manifest),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-prompts", type=Path, required=True)
    parser.add_argument("--candidate-prompts-sha256", required=True)
    parser.add_argument("--student-heads", type=Path, required=True)
    parser.add_argument("--student-states", type=Path, required=True)
    parser.add_argument("--student-rounds", type=Path, required=True)
    parser.add_argument(
        "--student-task-map",
        type=Path,
        required=True,
        help="JSON object mapping native task_id to source prompt ID",
    )
    parser.add_argument("--model-snapshot-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--base-draft-gguf", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d-gguf", type=Path, required=True)
    parser.add_argument("--row-export-gguf", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {"status": report["status"], "checks": report["checks"], "report": str(args.report)}
        )
    )


if __name__ == "__main__":
    main()

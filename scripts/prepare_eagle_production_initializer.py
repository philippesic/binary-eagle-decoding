#!/usr/bin/env python3
"""Fit scale-only fusion from original admitted continuous TRAIN accepted prefixes.

This CPU helper consumes existing native captures; it never captures, loads a
model actor, modifies historical readiness, or grants current GPU admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
DOMAINS = ("prose", "code", "reasoning")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(path, expected):
    path = Path(path).resolve()
    before = path.stat()
    require(sha256(path) == expected, "source hash differs: " + str(path))
    require(path.stat() == before, "source changed during verification")
    return path


def select_prompts(records, fit_per_domain, validation_per_domain):
    """Deterministic opaque selection; groups/topics/content cannot cross splits."""
    require(
        all(type(n) is int and n > 0 for n in (fit_per_domain, validation_per_domain)),
        "positive fit and validation prompt quotas required",
    )
    counts, seen_ids, seen_groups, seen_topics, seen_contents = (
        Counter(),
        set(),
        set(),
        set(),
        set(),
    )
    chosen = []
    for row in records:
        domain = row.get("domain")
        if domain not in DOMAINS:
            continue
        require(
            isinstance(row.get("group"), str)
            and row["group"]
            and isinstance(row.get("content_sha256"), str)
            and len(row["content_sha256"]) == 64,
            "source TRAIN group/content identity absent",
        )
        if counts[domain] >= fit_per_domain + validation_per_domain:
            continue
        if (
            row["id"] in seen_ids
            or row["group"] in seen_groups
            or row.get("topic") in seen_topics
            or row["content_sha256"] in seen_contents
        ):
            continue
        split = "fit" if counts[domain] < fit_per_domain else "validation"
        chosen.append(dict(row, calibration_split=split))
        counts[domain] += 1
        seen_ids.add(row["id"])
        seen_groups.add(row["group"])
        seen_contents.add(row["content_sha256"])
        if row.get("topic") is not None:
            seen_topics.add(row["topic"])
    require(
        all(counts[domain] == fit_per_domain + validation_per_domain for domain in DOMAINS),
        "insufficient independent TRAIN prompts for all three domains",
    )
    return chosen


def source_prompt_records(auth, spec):
    """Join retained capture positions to the original frozen corpus index."""
    files, stages = auth["files"], spec["stages"]
    locator = stages["corpus_manifest"]
    corpus = files.read(locator["path"], locator["sha256"])
    train = [row for row in stages["captures"] if row["split"] == "train"]
    require(len(train) == len(auth["records"]), "retained TRAIN plan membership differs")
    result = []
    for ordinal, ((_, child, _), plan) in enumerate(zip(auth["records"], train)):
        require(plan["prompts_sha256"] == child["sha256"]["prompts"], "provider/plan differs")
        shard = next(
            row
            for row in corpus["files"]["train"]["shards"]
            if row["prompts_sha256"] == plan["source_prompts_sha256"]
        )
        require(shard["index_sha256"] == plan["source_index_sha256"], "TRAIN index differs")
        index = [
            json.loads(line)
            for line in files.text(
                Path(locator["path"]).parent / shard["index"], shard["index_sha256"]
            ).split("\n")
            if line.strip()
        ]
        prompts = [
            json.loads(line)
            for line in files.text(child["paths"]["prompts"], child["sha256"]["prompts"]).split(
                "\n"
            )
            if line.strip()
        ]
        require(len(prompts) == len(plan["source_positions"]), "retained prompt count differs")
        for prompt, position in zip(prompts, plan["source_positions"]):
            row = index[position]
            require(
                row["id"] == prompt["id"] and row["domain"] == prompt["domain"],
                "TRAIN index join differs",
            )
            content = digest(
                json.dumps(
                    prompt["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            )
            require(content == row["content_sha256"], "TRAIN prompt content differs from index")
            result.append(dict(row, shard_ordinal=ordinal))
    return result


def longest_eligible_round(child, prompt_id):
    """Pick the longest authentically supervised round before reading features.

    A terminal accepted prefix may have no next CE label. Structural corruption
    still raises immediately; only a valid all-false CE mask is skipped.
    """
    from w1a1_eagle.recurrent_trace import validate_recurrent_trace

    keys = [key for key in child.capture.anchors if key[0] == prompt_id]
    require(keys, "selected TRAIN prompt has no native accepted prefix")
    keys.sort(
        key=lambda key: (len(child.capture.anchors[key].prefix_token_ids), key[1]),
        reverse=True,
    )
    skipped = []
    for key in keys:
        audit = validate_recurrent_trace(
            child.capture.rows[key],
            [child.capture.anchors[key]],
            offsets=child.d2t_offsets,
            target_vocab_size=child.target_vocab_size,
            draft_vocab_size=child.draft_vocab_size,
            allowed_prompt_ids=child.allowed_prompt_ids,
            split=child.split,
            max_depth=child.max_depth,
        )
        if any(audit.ce_mask):
            return key, audit, skipped
        skipped.append(key[1])
    raise ValueError("selected native TRAIN prompt has no admitted-label round: " + prompt_id)


def collect_operands(auth, continuous, selected, rows_per_prompt):
    from train_prepared_continuous_w1ax import create_current_native_child

    from w1a1_eagle.recurrent_provider import ProviderRound, audit_provider_round

    require(
        type(rows_per_prompt) is int and rows_per_prompt > 0, "positive rows per prompt required"
    )
    raw, evidence, mask = [], [], []
    for ordinal in sorted({row["shard_ordinal"] for row in selected}):
        record, child_spec, ids = auth["records"][ordinal]
        path = auth["files"].check(record["provider_manifest"], record["provider_manifest_sha256"])
        child = create_current_native_child(continuous.qat(continuous.activation_bits[0]), path)
        require(
            child.training_eligible is True
            and child.full_body_qat_eligible is True
            and child.data_split == "train"
            and child.allowed_prompt_ids == set(ids)
            and child.capture_id == child_spec["capture_id"],
            "selected native child source/eligibility differs",
        )
        for prompt in (row for row in selected if row["shard_ordinal"] == ordinal):
            key, selection_audit, skipped_rounds = longest_eligible_round(child, prompt["id"])
            data = child.capture.round_inputs(*key)
            batch = ProviderRound(
                data.anchor,
                data.rows,
                data.prefix_token_ids,
                data.raw_target_features,
                data.feature_positions,
                child.capture_id,
            )
            audit = audit_provider_round(batch, child)
            require(
                any(audit.ce_mask) and audit == selection_audit,
                "selected native TRAIN round labels changed between metadata and feature load",
            )
            positions = tuple(data.feature_positions)
            require(len(positions) >= rows_per_prompt, "selected TRAIN prompt has too few raw rows")
            require(
                positions == tuple(range(len(data.anchor.prefix_token_ids))),
                "native cache prefix positions differ",
            )
            values = data.raw_target_features.detach().cpu().numpy()
            require(
                values.dtype == np.float32 and values.shape == (len(positions), 7680),
                "raw native taps differ",
            )
            for position in np.linspace(0, len(positions) - 1, rows_per_prompt, dtype=int):
                prefix = tuple(data.anchor.prefix_token_ids[: int(position) + 1])
                feature_row = child.capture.feature_lookup[(prompt["id"], prefix)]
                x = np.array(values[position], dtype="<f4", copy=True)
                require(np.isfinite(x).all(), "nonfinite native TRAIN input")
                raw.append(x)
                mask.append(prompt["calibration_split"] == "fit")
                evidence.append(
                    {
                        "prompt_id": prompt["id"],
                        "domain": prompt["domain"],
                        "calibration_split": prompt["calibration_split"],
                        "source_split": "train",
                        "shard_ordinal": ordinal,
                        "capture_id": child.capture_id,
                        "capture_manifest_sha256": child.hashes["capture_manifest"],
                        "round_index": key[1],
                        "feature_row": feature_row,
                        "position": int(position),
                        "prefix_token_ids": list(prefix),
                        "raw_input_sha256": digest(x.tobytes()),
                        "boundary": "native_target_block_inputs_concat_before_draft_fc",
                        "tap_ids": [2, 18, 33],
                        "accepted_prefix": True,
                        "round_ce_mask": list(audit.ce_mask),
                        "round_selection": "longest_native_round_with_admitted_ce_labels",
                        "skipped_unsupervised_longer_rounds": skipped_rounds,
                    }
                )
            del data, batch, values
        del child
    return np.stack(raw), np.asarray(mask, dtype=bool), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages-manifest", type=Path)
    parser.add_argument("--prepared-run-dir", type=Path, required=True)
    parser.add_argument("--prepared-ready-sha256", required=True)
    parser.add_argument("--source-weights", type=Path, required=True)
    parser.add_argument("--source-weights-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--activation-bits", type=int, choices=(8, 1), default=8)
    parser.add_argument("--fit-prompts-per-domain", type=int, default=32)
    parser.add_argument("--validation-prompts-per-domain", type=int, default=16)
    parser.add_argument("--rows-per-prompt", type=int, default=16)
    parser.add_argument("--max-fit-seconds", type=float, default=300)
    args = parser.parse_args()
    from dataclasses import replace

    import train_continuous_w1ax as api
    from prepare_fusion_binary_train_operands import read_bf16_fusion
    from prepared_continuous_provider import authenticate

    from w1a1_eagle.block_fusion import FusionFitConfig, diagnostics, fit_fusion, project, quantize

    output = args.output_dir.resolve()
    from prepare_continuous_w1ax_data import require_untracked_destination

    require_untracked_destination(output)
    require(not output.exists(), "initializer output already exists")
    spec, continuous = api.load_config(args.config.resolve())
    if args.stages_manifest:
        spec["stages"] = json.loads(args.stages_manifest.read_text())
    # Historical declaration authenticates the full corpus; selected arithmetic
    # changes only the current CPU scale fit and does not relabel teacher data.
    auth = authenticate(api, spec, output, args.prepared_run_dir, args.prepared_ready_sha256)
    continuous = replace(continuous, activation_bits=(args.activation_bits,))
    selected = select_prompts(
        source_prompt_records(auth, spec),
        args.fit_prompts_per_domain,
        args.validation_prompts_per_domain,
    )
    raw, mask, evidence = collect_operands(auth, continuous, selected, args.rows_per_prompt)
    weight_path = stable_hash(args.source_weights, args.source_weights_sha256)
    model_files = auth["records"][0][1]["paths"]["model_snapshot_manifest"]
    snapshot = auth["files"].read(
        model_files, auth["records"][0][1]["sha256"]["model_snapshot_manifest"]
    )
    require(
        any(
            row["sha256"] == args.source_weights_sha256 and row["path"] == weight_path.name
            for row in snapshot["models"]["draft"]["files"]
        ),
        "fusion weights not bound to original drafter snapshot",
    )
    require(
        weight_path.parent == Path(auth["records"][0][1]["paths"]["draft_model_dir"]).resolve(),
        "fusion weights escaped original drafter snapshot directory",
    )
    weight = read_bf16_fusion(weight_path)
    fit_config = FusionFitConfig(
        args.activation_bits,
        False,
        0,
        args.max_fit_seconds,
        reference_kind="eagle_fixed_reference_0.5",
    )
    candidate = fit_fusion(raw[mask], (raw[mask] @ weight.T).astype(np.float32), weight, fit_config)
    require(not candidate["report"]["events"], "scale-only fit changed signs")
    validation = {}
    for domain in DOMAINS:
        domain_mask = np.asarray([row["domain"] == domain for row in evidence]) & ~mask
        codes, beta = quantize(raw[domain_mask], args.activation_bits)
        validation[domain] = diagnostics(
            project(codes, beta, candidate["hard_signs"], candidate["scale"]),
            (raw[domain_mask] @ weight.T).astype(np.float32),
        )
    # No artifacts are published until inputs, fit and all-domain validation pass.
    output.mkdir(parents=True)
    np.save(output / "raw-input.npy", raw, allow_pickle=False)
    np.savez(
        output / "initializer.npz",
        **{"fc.latent": candidate["latent"], "fc.scale": candidate["scale"]},
    )
    init = candidate["latent_initialization"]
    locator = {
        "path": str(output / "initializer.npz"),
        "sha256": sha256(output / "initializer.npz"),
        "encoding": "policy_latents",
        "activation_bits": args.activation_bits,
        "latent_initialization": {
            key: init[key] for key in ("policy", "reference_kind", "reference_sha256")
        },
    }
    report = {
        "schema": "eagle_production_fusion_initializer_v1",
        "activation_bits": args.activation_bits,
        "fit_config": asdict(fit_config),
        "fit": candidate["report"],
        "validation": validation,
        "initializer": locator,
        "selected_prompts": selected,
        "fit_rows": int(mask.sum()),
        "validation_rows": int((~mask).sum()),
        "prepared_run_dir": str(args.prepared_run_dir.resolve()),
        "prepared_ready_sha256": args.prepared_ready_sha256,
        "authenticated_full_source": auth["source"],
        "source_sha256": auth["binding"]["source_sha256"],
        "source_weights": {"path": str(weight_path), "sha256": args.source_weights_sha256},
        "config": {"path": str(args.config.resolve()), "sha256": sha256(args.config)},
        "stages_manifest": None
        if not args.stages_manifest
        else {"path": str(args.stages_manifest.resolve()), "sha256": sha256(args.stages_manifest)},
        "raw_input_sha256": sha256(output / "raw-input.npy"),
        "helper_sha256": sha256(Path(__file__)),
        "fusion_module_sha256": sha256(ROOT / "src/w1a1_eagle/block_fusion.py"),
        "calibration_hardware": platform.platform() + " / CPU",
        "status": "calibrated initializer only; current actor/native/SM120 admission required",
        "validation_scope": (
            "prompt/group/topic/content-disjoint TRAIN holdout; "
            "not sealed evaluation or model quality"
        ),
    }
    (output / "row-evidence.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evidence)
    )
    report["row_evidence_sha256"] = sha256(output / "row-evidence.jsonl")
    for name, value in (("initializer.json", locator), ("report.json", report)):
        (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "initializer": locator,
                "fit_rows": report["fit_rows"],
                "validation_rows": report["validation_rows"],
            }
        )
    )


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    main()

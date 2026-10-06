#!/usr/bin/env python3
"""Prepare both unselected captured-prefix proposals; CPU files only, never execute.

Historical selectors and corpus receipts remain unchanged. Derived manifests map
only original TRAIN shard zero onto byte-preserving transported asset locators.
The actual producer validates local plans; remote plans need its fresh remote
CPU validation and separate runtime/build/library/resource admission.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capture_nine_model_train_data as capture

GIB = 1024**3
NATIVE_REVISION = "ecff6d4e74814c631801df2e74f562d4ed6bd0eb"
BINARY_SHA = "63eacbb8e600488c76122dfd29d2da168a8a371db7c244b2b10d90d1d21c478f"
TARGET_SHA = "05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6"
BUILD_SHA = "b65884e17e8b0f293bb6a92178059d5d6ff4b5055f17c35f2ce7f4e6df2e17f9"
TRANSFER_SHA = "9921b8815179df7902f865a8e24fc7ae480c637a3982b4cf379a211c5f2e9039"
DEPTH_SHA = "7e50d79353de6714011d0150d5e7d0ada117e0968a30dcab665444db860dbbd6"
BINARY_RELATIVE = "runs/build/native-indexed-ecff-sm120-20261005-v2/bin/llama-block-teacher"
STATUS = "PROPOSAL_ONLY_NOT_SELECTED"


def load_json(path):
    return json.loads(Path(path).read_text())


def require_file(path, digest):
    path = Path(path).resolve()
    if not path.is_file() or capture.file_sha256(path) != digest:
        raise ValueError(f"source bytes differ: {path}")
    return path


def relative_source(record, historical_root):
    """Never map an arbitrary historical locator outside the original checkout."""
    return Path(record["path"]).relative_to(Path(historical_root)).as_posix()


def validate_selection(option, depth, corpus_path):
    if option["source_files"] != depth["source_files"]:
        raise ValueError("historical option and pinned depth source files differ")
    selection = option["original_train_selection"]
    if selection != option["capture_plan_draft"]["selection"]:
        raise ValueError("historical draft and original selectors differ")
    typed = ("shard", "row", "prompt_id", "prompt_sha256", "domain", "split")
    if selection != [{k: s[k] for k in typed} for s in depth["selected"]]:
        raise ValueError("depth evidence and historical selectors differ")
    quota = depth["quota"]
    if quota not in (150, 250) or depth["generation"] != 128:
        raise ValueError("only the two original 128-token options are supported")
    corpus = load_json(corpus_path)
    shard = corpus["files"]["train"]["shards"][0]
    index_path = require_file(corpus_path.parent / shard["index"], shard["index_sha256"])
    rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    if any(s["shard"] != 0 for s in selection):
        raise ValueError("proposal selects a shard outside transported original shard zero")
    for domain in capture.DOMAINS:
        eligible = [i for i, r in enumerate(rows) if r["domain"] == domain and r["input_tokens"] <= 512]
        for role, expected in (
            ("calibration_fit", eligible[:32]),
            ("calibration_validation", eligible[32:48]),
            ("train", eligible[48:48 + quota]),
        ):
            actual = sorted(s["row"] for s in selection if s["domain"] == domain and s["split"] == role)
            if actual != expected or len(actual) != {"train": quota, "calibration_fit": 32, "calibration_validation": 16}[role]:
                raise ValueError("original ascending eligible role/domain selectors differ")
    return selection, shard


def transport_corpus(original_pin, shard, mappings):
    # A new selector-scoped manifest, never an edited historical receipt. Other
    # corpus shards and held-out files are deliberately not claimed transported.
    return {
        "schema": "nine_model_train_capture_transport_corpus_v1",
        "original_corpus": original_pin,
        "scope": "only original TRAIN shard ordinal zero; original row ordinals unchanged",
        "files": {"train": {"shards": [shard | {
            name: mappings[name]["path"] for name in ("prompts", "index")
        }]}},
    }


def envelope(chain_count, source_bytes):
    width, vocab, chain_rows, teacher_rows, peak_rows = 2560, 151936, 640, 126, 129
    features = chain_rows * 5 * width * 4
    retained_logits = teacher_rows * vocab * 4
    peak_logits = peak_rows * vocab * 4
    map_metadata = 7 * (64 + 16 * teacher_rows)
    token_copies = 2 * chain_count * (128 + chain_rows * 8)
    goldens = 3 * (chain_rows * 5 * width * 4 + vocab * 4) + 3 * (chain_rows * 3 * width * 4 + vocab * 4)
    metadata = chain_count * (262144 + map_metadata) + 6 * 262144 + 16 * 1024**2
    retained = chain_count * (features + retained_logits) + token_copies + goldens + metadata + source_bytes
    peak = retained + chain_count * (peak_logits - retained_logits)
    return {
        "feature_bytes_per_chain": features,
        "retained_logit_bytes_per_chain": retained_logits,
        "peak_logit_bytes_per_chain": peak_logits,
        "native_peak_request_bytes": features + peak_logits,
        "raw_retained_bytes": chain_count * (features + retained_logits),
        "both_family_token_npy_bytes": token_copies,
        "six_golden_raw_bytes": goldens,
        "receipt_manifest_log_metadata_bytes": metadata,
        "source_bytes_charged": source_bytes,
        "capture_retained_bytes": retained,
        "capture_envelope_bytes": peak,
        "capture_incremental_prelaunch_free_bytes": peak + 10 * GIB,
        "envelope_scope": "both family imports share raw F32 payloads; temporary peak charged to every chain; reserve is additional",
    }


def disk_gate(required, observed):
    return {
        "required_incremental_capture_free_bytes": required,
        "observed_free_bytes": observed,
        "status": "PENDING_FRESH_DISK" if observed is None else "INSUFFICIENT" if observed < required else "SNAPSHOT_SUFFICIENT_NOT_ADMITTED",
        "scope": "capture only, after existing protected assets; fresh inventory still required for subsequent training/export",
    }


def prepare(*, options_dir, depth_evidence, source_root, checkout_root, asset_root,
            output_root, build_provenance=None, verified_transfer=None, observed_free_disk_bytes=None):
    options_dir, source_root, output_root = map(Path, (options_dir, source_root, output_root))
    checkout_root, asset_root = Path(checkout_root), Path(asset_root)
    if not checkout_root.is_absolute() or not asset_root.is_absolute():
        raise ValueError("explicit absolute remote checkout and asset roots required")
    depth_path = require_file(depth_evidence, DEPTH_SHA)
    depths = load_json(depth_path)
    if depths["status"] != STATUS:
        raise ValueError("depth evidence is not an unselected proposal")
    provenance = None
    if build_provenance is not None:
        provenance = require_file(build_provenance, BUILD_SHA)
    transfer = None
    if verified_transfer is not None:
        transfer = require_file(verified_transfer, TRANSFER_SHA)
    if observed_free_disk_bytes is not None and (type(observed_free_disk_bytes) is not int or observed_free_disk_bytes < 0):
        raise ValueError("observed free disk must be typed nonnegative bytes")
    if output_root.exists():
        raise FileExistsError("refuse to overwrite proposal history")
    output_root.mkdir(parents=True)
    shutil.copyfile(depth_path, output_root / "historical-depth-evidence.json")
    if provenance:
        shutil.copyfile(provenance, output_root / "build-provenance.json")
    if transfer:
        shutil.copyfile(transfer, output_root / "verified-transfer.json")
    summaries = []
    for quota in (150, 250):
        original_option = options_dir / f"{quota}-train-per-domain-exact_soft.json"
        option = load_json(original_option)
        if option["selected_for_execution"] is not False or option["artifact_kind"] != "proposal_ONLY":
            raise ValueError("historical option was selected or not a proposal")
        depth = next(d for d in depths["options"] if d["quota"] == quota and d["generation"] == 128)
        historical_root = Path(option["source_modules"]["scripts/capture_block_qat_teacher.py"]["path"]).parents[1]
        files = {}
        for name, pin in option["source_files"].items():
            rel = relative_source(pin, historical_root)
            local = require_file(source_root / rel, pin["sha256"])
            files[name] = {"original": pin, "local": {"path": str(local), "sha256": pin["sha256"]},
                           "remote": {"path": str(asset_root / rel), "sha256": pin["sha256"]}}
            if transfer:
                actual = next((r for r in load_json(transfer)["files"] if r["path"] == rel), None)
                if actual is None or actual["actual_path"] != str(asset_root / rel) or actual["sha256"] != pin["sha256"]:
                    raise ValueError("verified transport does not bind proposed original source locator")
        corpus_path = Path(files["corpus-manifest.json"]["local"]["path"])
        selection, shard = validate_selection(option, depth, corpus_path)
        # Both explicit pins and original corpus-selected shard pins must agree.
        for name, file_name in (("prompts", "train-00000.jsonl"), ("index", "train-00000.index.jsonl")):
            if shard[name + "_sha256"] != files[file_name]["local"]["sha256"]:
                raise ValueError("corpus and explicit transported shard hash differ")
        directory = output_root / f"captured-prefix-{3 * quota}"
        directory.mkdir()
        shutil.copyfile(original_option, directory / "historical-option.json")
        native = copy.deepcopy(option["capture_plan_draft"]["native"])
        client_relative = relative_source(native["client_source"], historical_root)
        # Prefer the durable original/current source checkout when it contains
        # the pinned client. Transport-only roots have no scripts, so use this
        # frozen helper checkout in that case. Both choices authenticate bytes.
        client_locator = source_root / client_relative
        if not client_locator.is_file():
            client_locator = ROOT / client_relative
        local_client = require_file(client_locator, native["client_source"]["sha256"])
        if native["source_revision"] != NATIVE_REVISION or native["target"]["sha256"] != TARGET_SHA:
            raise ValueError("historical native/target pin differs from actual indexed build")
        target_pins = load_json(files["target-source-pins.json"]["local"]["path"])
        if any(native[k] != target_pins[k] for k in ("tokenizer_metadata_sha256", "chat_template_sha256")):
            raise ValueError("historical tokenizer/template differs from actual target metadata pins")
        native["binary"] = {"path": str(checkout_root / BINARY_RELATIVE), "sha256": BINARY_SHA}
        native["target"]["path"] = str(checkout_root / "models/gguf/Qwen3-4B-f16.gguf")
        native["client_source"]["path"] = str(checkout_root / client_relative)
        runtime = {
            "schema": "nine_model_train_capture_runtime_v1", "binary_sha256": BINARY_SHA,
            "target_sha256": TARGET_SHA, "native_source_revision": NATIVE_REVISION,
            "teacher_client_sha256": native["client_source"]["sha256"],
            "tokenizer_metadata_sha256": native["tokenizer_metadata_sha256"],
            "chat_template_sha256": native["chat_template_sha256"],
        }
        capture.write_json(directory / "runtime.json", runtime)
        corpus_pins = {}
        for location in ("local", "remote"):
            mappings = {name: files[file_name][location] for name, file_name in (("prompts", "train-00000.jsonl"), ("index", "train-00000.index.jsonl"))}
            corpus = transport_corpus(files["corpus-manifest.json"][location], shard, mappings)
            pin = capture.write_json(directory / f"{location}-corpus.json", corpus)
            corpus_pins[location] = {"path": f"{location}-corpus.json", "sha256": pin["sha256"]}
        # Charge the larger derived manifest plus runtime/current client and all
        # four immutable historical source files. Producer's smaller own bound
        # is checked below rather than substituted for full importer overhead.
        source_bytes = sum(Path(f["local"]["path"]).stat().st_size for f in files.values())
        source_bytes += max((directory / f"{k}-corpus.json").stat().st_size for k in ("local", "remote"))
        source_bytes += (directory / "runtime.json").stat().st_size + local_client.stat().st_size
        costs = envelope(len(selection), source_bytes)
        if costs["raw_retained_bytes"] != depth["raw_retained"]:
            raise ValueError("current geometry does not reproduce original exposure evidence")
        caps = {
            "max_requests": len(selection) + 6, "max_tokens_per_chain": 640, "max_prompt_tokens": 512,
            "max_request_bytes": costs["native_peak_request_bytes"] + 262144,
            "max_shard_bytes": costs["native_peak_request_bytes"] + 262144,
            "max_total_bytes": costs["capture_envelope_bytes"], "max_source_bytes": 8 * 1024**2,
            "max_source_row_bytes": 65536, "max_host_rss_bytes": 12 * GIB,
            "min_host_available_bytes": 4 * GIB, "min_free_disk_bytes": 10 * GIB,
            "request_timeout_seconds": 120, "total_timeout_seconds": (len(selection) + 6) * 12,
            "max_eagle_golden_tokens": 640,
        }
        runtime_pin = {"path": "runtime.json", "sha256": capture.file_sha256(directory / "runtime.json")}
        plans = {}
        for location in ("local", "remote"):
            binding = copy.deepcopy(native)
            if location == "local":
                binding["client_source"]["path"] = str(local_client)
            plan = {"schema": "nine_model_train_capture_plan_v1", "corpus": corpus_pins[location],
                    "runtime": runtime_pin, "native": binding, "vocab_size": 151936,
                    "target_width": 2560, "mask_token_id": 151669, "input_mode": "native_chat",
                    "objective": "exact_soft", "teacher_logits_layout": "indexed",
                    "generation": {"max_new_tokens": 128, "algorithm": "greedy"},
                    "caps": caps, "selection": selection}
            plans[location] = capture.write_json(directory / f"{location}-plan.json", plan)
        _, records, producer_cost = capture.prepare_plan(plans["local"]["path"], plans["local"]["sha256"])
        if len(records) != len(selection) or producer_cost["max_block_retained_teacher_rows"] != 126:
            raise ValueError("actual producer geometry differs")
        capture.write_json(directory / "producer-local-cost.json", producer_cost)
        summary = {
            "schema": "block_production_capture_proposal_v1", "status": STATUS,
            "selected_for_execution": False, "production_data_status": "PENDING_NATIVE_CAPTURE",
            "conditioning": "captured_prefix", "scientific_selection": "PENDING_HUMAN",
            "quota_per_domain": quota, "train_prompts": 3 * quota, "all_role_prompts": len(selection),
            "role_counts": producer_cost["counts"], "max_train_blocks_per_pass": 3 * quota * 18,
            "max_train_labels_per_pass": 3 * quota * 126, "geometry": {
                "max_prompt_tokens": 512, "max_new_tokens": 128, "max_chain_rows": 640,
                "feature_shape": [640, 5, 2560], "feature_dtype": "F32",
                "max_teacher_shape": [126, 151936], "teacher_dtype": "F32",
                "peak_teacher_rows": 129, "tap_ids": list(capture.TAPS), "full_context_retained": True,
                "all_potential_teacher_rows_per_anchor": True,
            }, "plans": {k: {"path": f"{k}-plan.json", "sha256": v["sha256"]} for k, v in plans.items()},
            "source_ancestry": files, "historical_option_sha256": capture.file_sha256(original_option),
            "historical_depth_sha256": DEPTH_SHA, "costs": costs, "caps": caps,
            "producer_bound_bytes": producer_cost["max_capture_bytes"],
            "disk_gate": disk_gate(costs["capture_incremental_prelaunch_free_bytes"], observed_free_disk_bytes),
            "historical_gross_workspace_allowances": {k: depth[k] for k in ("protected", "prelaunch", "illustrative_extra", "illustrative_allocation")},
            "time_scope": "unmeasured operational wall/request caps; not ETA; wall checks occur at operation boundaries, so in-flight import/hash work can overshoot; a wall expiry may precede completion",
            "memory_scope": "12 GiB owner+native RSS cap and 4 GiB MemAvailable floor at operation boundaries; no continuous production monitor or hard cgroup guarantee; importer/FD and GPU/KV/target peaks require fresh admission",
            "consumer_scope": "DSpark captured-prefix full_probability_l1; DFlash hard CE with separate family manifests/admission",
            "runtime_build_provenance": {"sha256": BUILD_SHA, "local_copy": "../build-provenance.json" if provenance else None},
            "runtime_library_and_build_files": load_json(provenance)["files"] if provenance else None,
            "verified_asset_transport": {"sha256": TRANSFER_SHA, "local_copy": "../verified-transfer.json" if transfer else None},
            "runtime_library_scope": "strict capture runtime schema omits libraries; separate actual build library pins and postlaunch NativeTeacher mapped-library binding required",
            "source_sha256": {p: capture.file_sha256(ROOT / p) for p in (
                "scripts/prepare_block_production_capture_plans.py", "scripts/capture_nine_model_train_data.py",
                "scripts/capture_block_qat_teacher.py", "src/w1a1_eagle/block_data.py")},
            "missing_inputs": ["human exposure/conditioning choice", "fresh remote CPU plan/hash/path validation",
                "current GPU release/lease/device resources and target/client/library hashes",
                "fresh incremental disk inventory; future retention/export budget; importer RSS/FD limits", "actual native token lengths/EOG/realized role coverage",
                "genuine capture/portability/numeric/decision and both family admissions", "production fusion fits and initialized actors/exports/fresh SM120 model admission"],
        }
        capture.write_json(directory / "proposal.json", summary)
        summaries.append(summary)
    capture.write_json(output_root / "proposals.json", {"status": STATUS, "selected_for_execution": False, "proposals": summaries})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options-dir", type=Path, required=True)
    parser.add_argument("--depth-evidence", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True, help="root containing preserved original relative source paths")
    parser.add_argument("--checkout-root", type=Path, required=True, help="actual remote source checkout/root")
    parser.add_argument("--asset-root", type=Path, required=True, help="actual verified remote transport root")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-provenance", type=Path)
    parser.add_argument("--verified-transfer", type=Path)
    parser.add_argument("--observed-free-disk-bytes", type=int)
    args = parser.parse_args()
    summaries = prepare(**vars(args))
    print(json.dumps({"status": STATUS, "selected_for_execution": False,
                      "options": [{"train_prompts": s["train_prompts"], "disk_gate": s["disk_gate"]} for s in summaries]}))


if __name__ == "__main__":
    main()

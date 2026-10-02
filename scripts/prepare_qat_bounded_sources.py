#!/usr/bin/env python3
"""Join pinned CPU metadata into collector sources; never grant admission.

Source registration schema: qat_bounded_source_records_v1, parent_commit and
native_commit (full commits), six absolute fields binary/target_gguf/
candidate_d_gguf/base_draft_gguf/absolute_d2t/model_snapshot_manifest, sha256
with exactly those six keys, and corpus_manifest={path,sha256}. The owner must
authenticate these records before supplying them. This tool opens only the
four explicitly pinned metadata/prompt inputs and the runtime's pinned JSON
manifest. It does not discover files or read model/map/corpus bytes. Downstream
verify_sources and verify_native_revision must validate actual bytes before
execution. A pin joins identities; it does not establish owner authentication.

Example (all paths absolute; SHAs and commits supplied by the owner):
  python scripts/prepare_qat_bounded_sources.py \\
    --native-runtime /run/native-runtime.json --native-runtime-sha256 SHA \\
    --source-records /run/owner-sources.json --source-records-sha256 SHA \\
    --selection /run/selection.json --selection-sha256 SHA \\
    --prompts /run/prompts.train.jsonl --prompts-sha256 SHA \\
    --expected-parent-commit COMMIT --expected-native-commit COMMIT \\
    --output /run/new-bounded-sources

Remote materialization may relocate prompt/corpus paths. Their hashes must
match the original selection; the entire original ancestry remains in output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path

REQUIRED = (
    "binary",
    "target_gguf",
    "candidate_d_gguf",
    "base_draft_gguf",
    "absolute_d2t",
    "model_snapshot_manifest",
)
MAX_METADATA_BYTES = 4 * 1024**2
DOMAINS = ["prose", "code", "reasoning"]


def hex_value(value, length, name):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value):
        raise ValueError(name + " must be a full lowercase hex identity")
    return value


def absolute_path(value):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("record path must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or str(path) != value or ".." in path.parts:
        raise ValueError("record path must be a normalized absolute path")
    return value


def record(value):
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError("expected an exact path/sha256 record")
    absolute_path(value["path"])
    hex_value(value["sha256"], 64, "SHA256")
    return copy.deepcopy(value)


def pinned_bytes(value):
    value = record(value)
    with Path(value["path"]).open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise ValueError("metadata/prompt input exceeds bounded size")
    if hashlib.sha256(raw).hexdigest() != value["sha256"]:
        raise ValueError("metadata/prompt SHA256 mismatch")
    return raw


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def pinned_json(value):
    result = parse_json(pinned_bytes(value))
    if not isinstance(result, dict):
        raise ValueError("metadata must be a JSON object")
    return result


def reject_admission(value):
    for name in ("training_eligible", "provider_admission", "readiness_granted"):
        if name in value and value[name] is not False:
            raise ValueError("metadata cannot carry admission: " + name)


def validate_selection(selection, prompts_record, raw, corpus):
    if selection.get("schema") != "qat_bounded_train_prompt_selection_v1":
        raise ValueError("unsupported prompt selection schema")
    if (
        selection.get("training_eligible") is not False
        or selection.get("provider_admission") is not False
        or selection.get("sealed_or_reserve_payload_read") is not False
        or selection.get("dev_payload_read") is not False
    ):
        raise ValueError("selection must preserve unsealed TRAIN-only non-admission")
    original_corpus = record(selection["corpus_manifest"])
    record(selection["packet"])
    if corpus["sha256"] != original_corpus["sha256"]:
        raise ValueError("registered corpus differs from selection ancestry")
    selected = selection["selection"]
    original_prompt = record({k: selected[k] for k in ("path", "sha256")})
    if original_prompt["sha256"] != prompts_record["sha256"]:
        raise ValueError("prompt source differs from selection")
    if type(selected.get("count")) is not int or selected["count"] != 3:
        raise ValueError("selection requires exactly three prompts")
    if selected.get("roles") != ["train"]:
        raise ValueError("selection role must be exactly TRAIN")
    for path in (original_prompt["path"], prompts_record["path"]):
        if any(word in path.lower() for word in ("sealed", "final", "reserve")):
            raise ValueError("reserved prompt path is prohibited")
    shard = selection["source_train_shard"]
    for key in ("prompts", "index"):
        if not isinstance(shard[key], str) or not re.fullmatch(r"train-[\w.-]+", shard[key]):
            raise ValueError("ancestry must name an original TRAIN shard")
        hex_value(shard[key + "_sha256"], 64, "TRAIN shard SHA256")
    if type(shard["prompts_count"]) is not int or shard["prompts_count"] < 3:
        raise ValueError("invalid original TRAIN shard count")
    rows = selected["rows"]
    lines = raw.splitlines(keepends=True)
    if not isinstance(rows, list) or len(rows) != 3 or len(lines) != 3:
        raise ValueError("exact three ordered prompt rows required")
    ids, ordinals = [], []
    for domain, row, line in zip(DOMAINS, rows, lines, strict=True):
        prompt = parse_json(line)
        if not isinstance(prompt, dict) or prompt.get("domain") != domain:
            raise ValueError("prompt domain/order differs from frozen selection")
        if row.get("domain") != domain or prompt.get("id") != row.get("id"):
            raise ValueError("prompt ordered IDs/domain differ from selection")
        if any(
            item.get(key, "train") != "train" for item in (prompt, row) for key in ("role", "split")
        ):
            raise ValueError("prompt row role differs from TRAIN")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("prompt ID is missing")
        if row["id"] != row.get("source_id", "") + ":" + row.get("source_row_id", ""):
            raise ValueError("prompt ID differs from original source join")
        if any(word in row["id"].lower() for word in ("sealed", "final", "reserve")):
            raise ValueError("reserved prompt ID is prohibited")
        ordinal = row["source_row_ordinal"]
        if type(ordinal) is not int or not 0 <= ordinal < shard["prompts_count"]:
            raise ValueError("source row ordinal outside original TRAIN shard")
        messages = prompt.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("prompt messages missing")
        canonical = json.dumps(
            messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        if hashlib.sha256(canonical).hexdigest() != row["content_sha256"]:
            raise ValueError("prompt content differs from source row")
        if hashlib.sha256(line).hexdigest() != row["raw_prompt_line_sha256"]:
            raise ValueError("raw prompt line differs from original TRAIN selection")
        ids.append(row["id"])
        ordinals.append(ordinal)
    if len(set(ids)) != 3 or ordinals != sorted(set(ordinals)):
        raise ValueError("duplicate or reordered original TRAIN source rows")
    return ids


def build_sources(
    native_runtime,
    source_records,
    selection_record,
    prompts_record,
    expected_parent,
    expected_native,
):
    """Read pinned metadata only; model/map/corpus records are intentionally opaque."""
    hex_value(expected_parent, 40, "parent commit")
    hex_value(expected_native, 40, "native commit")
    runtime = pinned_json(native_runtime)
    registration = pinned_json(source_records)
    selection = pinned_json(selection_record)
    prompts_record = record(prompts_record)
    raw = pinned_bytes(prompts_record)
    if registration.get("schema") != "qat_bounded_source_records_v1":
        raise ValueError("unsupported owner source registration schema")
    allowed = set(REQUIRED) | {
        "schema",
        "parent_commit",
        "native_commit",
        "sha256",
        "corpus_manifest",
        "training_eligible",
        "provider_admission",
        "readiness_granted",
    }
    if set(registration) - allowed:
        raise ValueError("unrecognized source registration fields/pins")
    reject_admission(registration)
    for key, expected in (("parent_commit", expected_parent), ("native_commit", expected_native)):
        if registration.get(key) != expected:
            raise ValueError("source registration differs from pinned " + key)
    if set(registration.get("sha256", {})) != set(REQUIRED):
        raise ValueError("source registration needs exactly six required hashes")
    records = {
        key: record({"path": registration[key], "sha256": registration["sha256"][key]})
        for key in REQUIRED
    }
    corpus = record(registration["corpus_manifest"])
    if runtime.get("schema") != "qat_current_native_runtime_v1":
        raise ValueError("fresh current-native runtime inventory required")
    reject_admission(runtime)
    manifest_record = record(runtime["immutable_manifest"])
    manifest = pinned_json(manifest_record)
    if manifest.get("schema") != "qat_current_native_server_build_v1":
        raise ValueError("current server build manifest required")
    if (
        manifest.get("parent_commit") != expected_parent
        or manifest.get("native_commit") != expected_native
        or manifest.get("training_eligible") is not False
        or manifest.get("readiness_granted") is not False
    ):
        raise ValueError("runtime manifest differs from full pinned source/non-admission")
    commit = manifest.get("build_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError("runtime build commit is incomplete")
    if not expected_native.startswith(commit):
        raise ValueError("stale runtime build commit")
    if record(manifest["binary"]) != records["binary"]:
        raise ValueError("registered server differs from runtime server pin")
    directory = str(Path(records["binary"]["path"]).parent)
    if runtime.get("directory") != directory or runtime.get("ld_library_path") != directory:
        raise ValueError("runtime loader directory differs from server inventory")
    if runtime.get("libraries") != manifest.get("libraries") or not runtime.get("libraries"):
        raise ValueError("runtime library inventory is incomplete or differs from manifest")
    libraries = [record(item) for item in runtime["libraries"]]
    if len({item["path"] for item in libraries}) != len(libraries):
        raise ValueError("duplicate runtime library path")
    names = [Path(item["path"]).name for item in libraries]
    if not any(n.startswith("libllama.so") for n in names) or not any(
        n.startswith("libggml-cuda.so") for n in names
    ):
        raise ValueError("complete llama/CUDA runtime libraries required")
    if any(str(Path(item["path"]).parent) != directory for item in libraries):
        raise ValueError("project library outside inventoried runtime directory")
    pins = {}
    for item in [
        *records.values(),
        corpus,
        manifest_record,
        *libraries,
        record(native_runtime),
        record(source_records),
        record(selection_record),
        prompts_record,
        *map(record, manifest.get("allowed_runtime_dependency_artifacts", [])),
    ]:
        old = pins.setdefault(item["path"], item["sha256"])
        if old != item["sha256"]:
            raise ValueError("conflicting source pins for one path")
    ids = validate_selection(selection, prompts_record, raw, corpus)
    return {
        "schema": "qat_bounded_collector_sources_v1",
        "parent_commit": expected_parent,
        "native_commit": expected_native,
        **{key: value["path"] for key, value in records.items()},
        "sha256": {key: value["sha256"] for key, value in records.items()},
        "corpus_manifest": corpus,
        "native_runtime": copy.deepcopy(runtime),
        "source_registration": record(source_records),
        "native_runtime_inventory": record(native_runtime),
        "prompt_selection": {
            "manifest": record(selection_record),
            "prompts": prompts_record,
            "count": 3,
            "roles": ["train"],
            "ordered_ids": ids,
            "original_ancestry": copy.deepcopy(selection),
        },
        "scope": "CPU metadata join only; downstream actual byte validation required",
        "actual_source_bytes_validated": False,
        "training_eligible": False,
        "provider_admission": False,
        "readiness_granted": False,
        "hardware_measured": False,
        "optimizer_updates": 0,
    }


def write_sources(sources, output):
    """Publish only to a new output directory; never replace original evidence."""
    output = Path(absolute_path(str(output)))

    # Even a registered but absent payload path must never become our output.
    def protect(value):
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                if Path(value["path"]).is_relative_to(output):
                    raise ValueError("output overlaps an original input/source record")
            for item in value.values():
                protect(item)
        elif isinstance(value, list):
            for item in value:
                protect(item)

    protect(sources)
    if any(Path(sources[key]).is_relative_to(output) for key in REQUIRED):
        raise ValueError("output overlaps an original input/source record")
    # Serialize before creating anything, so invalid JSON cannot leave partial output.
    raw = (json.dumps(sources, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "sources.json"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    for name in ("native-runtime", "source-records", "selection", "prompts"):
        parser.add_argument("--" + name, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--expected-parent-commit", required=True)
    parser.add_argument("--expected-native-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be a new immutable directory")
    inputs = [
        {"path": getattr(args, name), "sha256": getattr(args, name + "_sha256")}
        for name in ("native_runtime", "source_records", "selection", "prompts")
    ]
    sources = build_sources(*inputs, args.expected_parent_commit, args.expected_native_commit)
    result = write_sources(sources, args.output)
    print(
        json.dumps(
            {
                "sources": result,
                "training_eligible": False,
                "provider_admission": False,
                "readiness_granted": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

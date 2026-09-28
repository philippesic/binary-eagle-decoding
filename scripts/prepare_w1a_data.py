#!/usr/bin/env python3
"""Prepare frozen, source-pinned W1A prompt tiers without running a model.

Input is a JSON source catalog with ``sources`` entries containing id, path,
sha256, revision, license, domain and optional field names. Source JSONL rows
must have a stable id and messages (chat role/content objects). Source files and
all output prompts belong under ignored data/, outside the tracked repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

DOMAINS = ("prose", "code", "reasoning")
ROLES = {"system", "user", "assistant", "tool"}
WORD = re.compile(r"\w+", re.UNICODE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def normalized(messages: list[dict]) -> str:
    return " ".join(WORD.findall(" ".join(m["content"] for m in messages).casefold()))


def shingles(text: str) -> set[str]:
    words = text.split()
    if len(words) < 5:
        return {" ".join(words)}
    return {" ".join(words[i : i + 5]) for i in range(len(words) - 4)}


def simhash(items: set[str]) -> int:
    scores = [0] * 64
    for item in items:
        bits = int.from_bytes(hashlib.blake2b(item.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            scores[bit] += 1 if bits & (1 << bit) else -1
    return sum((1 << bit) for bit, score in enumerate(scores) if score > 0)


def validate_messages(messages: object) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list")
    out = []
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ROLES
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("invalid role/content in messages")
        out.append({"role": message["role"], "content": message["content"]})
    if not any(m["role"] == "user" for m in out):
        raise ValueError("messages lack a user turn")
    return out


def load_catalog(
    path: Path, min_chars: int, max_chars: int
) -> tuple[list[dict], list[dict], Counter]:
    catalog = json.loads(path.read_text())
    if catalog.get("schema") != "w1a_source_catalog_v1" or not isinstance(
        catalog.get("sources"), list
    ):
        raise ValueError("catalog schema must be w1a_source_catalog_v1")
    rows, sources, rejected = [], [], Counter()
    seen_source_ids, seen_row_ids = set(), set()
    for source in catalog["sources"]:
        if not isinstance(source, dict):
            raise ValueError("source must be an object")
        sid = source.get("id")
        if not isinstance(sid, str) or not sid or sid in seen_source_ids:
            raise ValueError("source id missing or repeated")
        seen_source_ids.add(sid)
        domain = source.get("domain")
        if domain not in DOMAINS:
            raise ValueError(f"{sid}: domain must be prose, code or reasoning")
        for key in ("uri", "revision", "license", "sha256", "path"):
            if not isinstance(source.get(key), str) or not source[key]:
                raise ValueError(f"{sid}: missing {key}")
        source_path = (path.parent / source["path"]).resolve()
        if not source_path.is_file() or sha256(source_path) != source["sha256"]:
            raise ValueError(f"{sid}: source file hash mismatch")
        fields = source.get("fields", {})
        if not isinstance(fields, dict):
            raise ValueError(f"{sid}: fields must be an object")
        field = {
            name: fields.get(name, name) for name in ("id", "messages", "group_id", "topic_id")
        }
        if any(not isinstance(value, str) or not value for value in field.values()):
            raise ValueError(f"{sid}: invalid field mapping")
        count = 0
        with source_path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if not isinstance(raw, dict):
                    raise ValueError(f"{sid}:{line_number}: row must be an object")
                rid = raw.get(field["id"])
                if not isinstance(rid, str) or not rid or (sid, rid) in seen_row_ids:
                    raise ValueError(f"{sid}:{line_number}: row id missing or repeated")
                seen_row_ids.add((sid, rid))
                messages = validate_messages(raw.get(field["messages"]))
                text = normalized(messages)
                if not min_chars <= len(text) <= max_chars:
                    rejected["length"] += 1
                    continue
                group = raw.get(field["group_id"], rid)
                topic = raw.get(field["topic_id"])
                if not isinstance(group, str) or not group:
                    raise ValueError(f"{sid}:{line_number}: invalid group id")
                if topic is not None and (not isinstance(topic, str) or not topic):
                    raise ValueError(f"{sid}:{line_number}: invalid topic id")
                key = f"{sid}:{rid}"
                rows.append(
                    {
                        "id": key,
                        "source_id": sid,
                        "source_row_id": rid,
                        "domain": domain,
                        "messages": messages,
                        "group": f"group:{sid}:{group}",
                        "topic": f"topic:{sid}:{topic}" if topic else None,
                        "content_sha256": hashlib.sha256(canonical(messages)).hexdigest(),
                        "text": text,
                    }
                )
                count += 1
        source_record = {
            key: source[key] for key in ("id", "uri", "domain", "revision", "license", "sha256")
        } | {"rows_after_length_filter": count}
        for key in ("upstream_sha256", "transform", "transform_sha256"):
            if key in source:
                if not isinstance(source[key], str) or not source[key]:
                    raise ValueError(f"{sid}: invalid {key}")
                source_record[key] = source[key]
        sources.append(source_record)
    return rows, sources, rejected


def deduplicate(rows: list[dict], threshold: float) -> tuple[list[dict], Counter]:
    """Exact hash plus deterministic SimHash candidates, verified by shingle Jaccard."""
    kept, exact, bands, shingles_by_id, dropped = [], {}, defaultdict(set), [], Counter()
    for row in rows:
        text = row["text"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in exact:
            dropped["exact"] += 1
            continue
        grams = shingles(text)
        fingerprint = simhash(grams)
        candidates = set()
        for band in range(4):
            candidates.update(bands[(band, (fingerprint >> (16 * band)) & 0xFFFF)])
        near = False
        for candidate in sorted(candidates):
            other = shingles_by_id[candidate]
            if len(grams & other) / len(grams | other) >= threshold:
                near = True
                break
        if near:
            dropped["near"] += 1
            continue
        index = len(kept)
        kept.append(row)
        shingles_by_id.append(grams)
        exact[digest] = index
        for band in range(4):
            bands[(band, (fingerprint >> (16 * band)) & 0xFFFF)].add(index)
    return kept, dropped


def group_rows(rows: list[dict]) -> list[list[dict]]:
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    lookup = {}
    for i, row in enumerate(rows):
        for key in (row["group"], row["topic"]):
            if key is None:
                continue
            if key in lookup:
                parent[find(i)] = find(lookup[key])
            else:
                lookup[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def quotas(total: int) -> dict[str, int]:
    base, extra = divmod(total, len(DOMAINS))
    return {domain: base + (i < extra) for i, domain in enumerate(DOMAINS)}


def select(
    groups: list[list[dict]], want: dict[str, int], seed: int, name: str
) -> tuple[list[list[dict]], list[list[dict]]]:
    selected, rest, counts = [], [], Counter()
    ordered = sorted(groups, key=lambda g: stable_key(seed, f"{name}:{min(r['id'] for r in g)}"))
    for group in ordered:
        group_counts = Counter(row["domain"] for row in group)
        if any(counts[d] < want[d] for d in DOMAINS if group_counts[d]):
            selected.append(group)
            counts.update(group_counts)
        else:
            rest.append(group)
    missing = {d: want[d] - counts[d] for d in DOMAINS if counts[d] < want[d]}
    if missing:
        raise ValueError(f"insufficient eligible grouped prompts for {name}: {missing}")
    return selected, rest


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")


def prepare(
    catalog: Path,
    output: Path,
    train_small: int,
    train_large: int,
    dev: int,
    final: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    near_threshold: float,
) -> dict:
    if output.exists():
        raise ValueError("output must be a new directory; frozen sets are immutable")
    if not 0 < train_small <= train_large or min(dev, final) <= 0:
        raise ValueError("invalid tier sizes")
    if not 0 < near_threshold <= 1 or min_chars < 0 or max_chars < min_chars:
        raise ValueError("invalid filters")
    rows, sources, rejected = load_catalog(catalog, min_chars, max_chars)
    rows, duplicates = deduplicate(rows, near_threshold)
    groups = group_rows(rows)
    final_groups, groups = select(groups, quotas(final), seed, "final")
    dev_groups, groups = select(groups, quotas(dev), seed, "dev")
    large_groups, _ = select(groups, quotas(train_large), seed, "train_large")
    small_groups, _ = select(large_groups, quotas(train_small), seed, "train_small")
    selected = {
        "train_small": small_groups,
        "train_large": large_groups,
        "dev": dev_groups,
        "final": final_groups,
    }
    output.mkdir(parents=True)
    files = {}
    for name, split_groups in selected.items():
        split_rows = [row for group in split_groups for row in group]
        split_rows.sort(key=lambda row: stable_key(seed, f"order:{row['id']}"))
        prompts = [
            {"id": row["id"], "domain": row["domain"], "messages": row["messages"]}
            for row in split_rows
        ]
        records = [
            {
                "id": row["id"],
                "source_id": row["source_id"],
                "source_row_id": row["source_row_id"],
                "domain": row["domain"],
                "group": row["group"],
                "topic": row["topic"],
                "content_sha256": row["content_sha256"],
                "characters": len(row["text"]),
                "word_count": len(row["text"].split()),
            }
            for row in split_rows
        ]
        prompt_path, index_path = output / f"{name}.jsonl", output / f"{name}.index.jsonl"
        write_jsonl(prompt_path, prompts)
        write_jsonl(index_path, records)
        files[name] = {
            "prompts": prompt_path.name,
            "prompts_sha256": sha256(prompt_path),
            "index": index_path.name,
            "index_sha256": sha256(index_path),
            "prompts_count": len(split_rows),
            "groups_count": len(split_groups),
            "domains": dict(sorted(Counter(row["domain"] for row in split_rows).items())),
            "unique_sources": sorted({row["source_id"] for row in split_rows}),
            "words_total": sum(len(row["text"].split()) for row in split_rows),
            "characters_total": sum(len(row["text"]) for row in split_rows),
        }
    report = {
        "schema": "w1a_data_manifest_v1",
        "seed": seed,
        "catalog_sha256": sha256(catalog),
        "sources": sources,
        "selection": "grouped_source_topic_then_seeded_domain_quota_v1",
        "near_dedup": {
            "method": "simhash_4x16_candidate_shingle_jaccard",
            "threshold": near_threshold,
            "dropped": dict(duplicates),
        },
        "filter": {"min_chars": min_chars, "max_chars": max_chars, "rejected": dict(rejected)},
        "available_unique_prompts": len(rows),
        "files": files,
        "token_counts": "pending_pinned_target_tokenizer; word counts are not token counts",
        "final_status": "frozen_output_unreviewed; seal_after_manifest_hash_review",
    }
    (output / "manifest.json").write_bytes(canonical(report) + b"\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-small", type=int, default=2000)
    parser.add_argument("--train-large", type=int, default=10000)
    parser.add_argument("--dev", type=int, default=192)
    parser.add_argument("--final", type=int, default=192)
    parser.add_argument("--seed", type=int, default=1429)
    parser.add_argument("--min-chars", type=int, default=64)
    parser.add_argument("--max-chars", type=int, default=24000)
    parser.add_argument("--near-threshold", type=float, default=0.88)
    args = parser.parse_args()
    result = prepare(
        args.catalog,
        args.output,
        args.train_small,
        args.train_large,
        args.dev,
        args.final,
        args.seed,
        args.min_chars,
        args.max_chars,
        args.near_threshold,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output / "manifest.json"),
                "sha256": sha256(args.output / "manifest.json"),
                "counts": {key: value["prompts_count"] for key, value in result["files"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()

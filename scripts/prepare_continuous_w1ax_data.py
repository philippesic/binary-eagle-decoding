#!/usr/bin/env python3
"""CPU-only pinned public prompt corpus, with immutable grouped splits.

No model or tensor library is imported. Sealed text is written once, never reopened.
Token counts measure frozen Qwen chat-formatted INPUTS, not supervised predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from prepare_w1a_data import canonical, group_rows, normalized, quotas, select, sha256, stable_key

ROOT = Path(__file__).resolve().parents[1]


def require_untracked_destination(path: Path):
    resolved = path.resolve()
    for parent in [resolved, *resolved.parents]:
        if (parent / ".git").exists():
            relative = str(resolved.relative_to(parent))
            checked = subprocess.run(
                ["git", "-C", str(parent), "check-ignore", "--no-index", "-q", "--", relative],
                capture_output=True,
                check=False,
            )
            if checked.returncode != 0:
                raise ValueError("dataset destination must be ignored by Git (use data/)")
            break


@contextmanager
def staged_output(output: Path):
    require_untracked_destination(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".corpus-", dir=output.parent))
    try:
        yield temporary
        if output.exists():
            raise ValueError("frozen output appeared while preparing")
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def fetch(spec: dict, directory: Path, offline: bool) -> Path:
    require_untracked_destination(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec['id']}.jsonl"
    if not path.exists():
        if offline:
            raise ValueError(f"missing pinned raw source {spec['id']}")
        fd, temporary = tempfile.mkstemp(dir=directory, prefix=".download-")
        os.close(fd)
        try:
            with (
                urllib.request.urlopen(spec["uri"], timeout=120) as source,
                open(temporary, "wb") as out,
            ):
                while block := source.read(1024 * 1024):
                    out.write(block)
            if sha256(Path(temporary)) != spec["sha256"]:
                raise ValueError("download SHA256 mismatch")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if sha256(path) != spec["sha256"]:
        raise ValueError(f"pinned raw source hash mismatch: {spec['id']}")
    return path


def transform(spec: dict, path: Path, limit: int):
    """Stream upstream rows. Responses/answers/seeds never become teacher labels."""
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if limit and number > limit:
                break
            raw = json.loads(line)
            sid = spec["id"]
            if sid == "dolly":
                context = raw["context"].strip()
                text = raw["instruction"].strip() + ("\n\nContext:\n" + context if context else "")
                rid = f"line-{number:06d}"
                topic = (
                    "context:"
                    + hashlib.sha256(normalized([{"content": context}]).encode()).hexdigest()
                    if context
                    else None
                )
                category = raw["category"]
            elif sid == "gsm8k":
                text, rid, topic, category = (
                    raw["question"].strip(),
                    f"train-{number:06d}",
                    None,
                    "arithmetic_word_problem",
                )
            elif sid == "magicoder":
                text, rid, category = (
                    raw["problem"].strip(),
                    f"line-{number:06d}-index-{raw['index']}",
                    raw["lang"],
                )
                topic = "oss-seed:" + hashlib.sha256(raw["seed"].strip().encode()).hexdigest()
            else:
                raise ValueError(f"unsupported source adapter: {sid}")
            messages = [{"role": "user", "content": text}]
            yield {
                "id": f"{sid}:{rid}",
                "source_id": sid,
                "source_row_id": rid,
                "domain": spec["domain"],
                "messages": messages,
                "category": category,
                "group": f"group:{sid}:{rid}",
                "topic": f"topic:{sid}:{topic}" if topic else None,
                "legacy_topic": (
                    "topic:dolly:context-"
                    + hashlib.sha256(context.casefold().encode()).hexdigest()[:16]
                )
                if sid == "dolly" and context
                else None,
                "text": normalized(messages),
                "content_sha256": hashlib.sha256(canonical(messages)).hexdigest(),
            }


def opaque_exclusions(paths: list[Path]):
    ids, hashes, groups, topics, records = set(), set(), set(), set(), []
    for path in paths:
        # Indexes have IDs and hashes only. Refuse prompt files, including sealed text.
        if not path.name.endswith(".index.jsonl"):
            raise ValueError("exclusion must be an opaque .index.jsonl file")
        count = 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if "messages" in row or "text" in row:
                    raise ValueError("exclusion index contains forbidden prompt text")
                ids.add(row["id"])
                hashes.add(row["content_sha256"])
                groups.add(row.get("group"))
                topics.add(row.get("topic"))
                count += 1
        records.append({"file": path.name, "sha256": sha256(path), "rows": count})
    return ids, hashes, groups - {None}, topics - {None}, records


class DuplicateFilter:
    """Exact normalized equality plus LSH candidates verified by 5-gram Jaccard.

    Approximate candidates: 4 bands x 4 minhashes, XOR permutations of stable
    64-bit BLAKE2b shingles. No exhaustive/semantic near-dedup claim.
    """

    def __init__(self, threshold=0.88):
        self.exact = set()
        self.bands = defaultdict(set)
        self.texts = []
        self.threshold = threshold
        self.dropped = Counter()

    @staticmethod
    def grams(text):
        words = text.split()
        return {" ".join(words[i : i + 5]) for i in range(max(1, len(words) - 4))}

    @staticmethod
    def signature(grams):
        import numpy as np

        bits = np.array(
            [
                int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little")
                for g in grams
            ],
            dtype=np.uint64,
        )
        salts = np.array(
            [
                int.from_bytes(hashlib.blake2b(str(i).encode(), digest_size=8).digest(), "little")
                for i in range(16)
            ],
            dtype=np.uint64,
        )
        signature = np.min(bits[:, None] ^ salts, axis=0).tolist()
        return [tuple(signature[i : i + 4]) for i in range(0, 16, 4)]

    def add(self, text):
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in self.exact:
            self.dropped["exact_normalized"] += 1
            return False
        grams = self.grams(text)
        signature = self.signature(grams)
        candidates = set()
        for i, band in enumerate(signature):
            candidates.update(self.bands[(i, band)])
        for candidate in candidates:
            other = self.grams(self.texts[candidate])
            if len(grams & other) / len(grams | other) >= self.threshold:
                self.dropped["near_jaccard"] += 1
                return False
        index = len(self.texts)
        self.texts.append(text)
        self.exact.add(digest)
        for i, band in enumerate(signature):
            self.bands[(i, band)].add(index)
        return True


class CpuTokenizer:
    def __init__(self, directory: Path, spec: dict):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        for filename, digest in spec["files"].items():
            if sha256(directory / filename) != digest:
                raise ValueError(f"frozen tokenizer hash mismatch: {filename}")
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        config = json.loads((directory / "tokenizer_config.json").read_text())
        self.template = ImmutableSandboxedEnvironment().from_string(config["chat_template"])

    def count(self, messages):
        rendered = self.template.render(
            messages=messages, add_generation_prompt=True, enable_thinking=False, tools=None
        )
        return len(self.tokenizer.encode(rendered, add_special_tokens=False).ids)


def coverage(rows):
    lengths = sorted(r["input_tokens"] for r in rows)
    bins = Counter()
    for length in lengths:
        bins[
            "<128"
            if length < 128
            else "128-511"
            if length < 512
            else "512-2047"
            if length < 2048
            else ">=2048"
        ] += 1
    return {
        "prompts_count": len(rows),
        "unique_input_tokens_total": sum(lengths),
        "domains": dict(sorted(Counter(r["domain"] for r in rows).items())),
        "categories": dict(
            sorted(Counter(f"{r['source_id']}:{r['category']}" for r in rows).items())
        ),
        "token_length": {
            "min": lengths[0],
            "p50": lengths[len(lengths) // 2],
            "p95": lengths[int(len(lengths) * 0.95)],
            "max": lengths[-1],
            "bins": dict(bins),
        },
        "groups_count": len(group_rows(rows)),
    }


def write_split(output, name, rows, shard_rows, seed):
    rows = sorted(rows, key=lambda r: stable_key(seed, "order:" + r["id"]))
    shards = []
    for offset in range(0, len(rows), shard_rows):
        part = rows[offset : offset + shard_rows]
        prefix = f"{name}-{offset // shard_rows:05d}"
        prompt_path, index_path = output / (prefix + ".jsonl"), output / (prefix + ".index.jsonl")
        # Hash as we WRITE; no sealed text is opened again for hashing or inspection.
        prompt_digest, index_digest = hashlib.sha256(), hashlib.sha256()
        with prompt_path.open("xb") as prompts, index_path.open("xb") as index:
            for row in part:
                content = canonical({k: row[k] for k in ("id", "domain", "messages")}) + b"\n"
                prompts.write(content)
                prompt_digest.update(content)
                content = (
                    canonical(
                        {
                            k: row[k]
                            for k in (
                                "id",
                                "domain",
                                "source_id",
                                "source_row_id",
                                "group",
                                "topic",
                                "content_sha256",
                                "input_tokens",
                                "category",
                            )
                        }
                    )
                    + b"\n"
                )
                index.write(content)
                index_digest.update(content)
        shards.append(
            {
                "prompts": prompt_path.name,
                "prompts_sha256": prompt_digest.hexdigest(),
                "index": index_path.name,
                "index_sha256": index_digest.hexdigest(),
                "prompts_count": len(part),
            }
        )
    return coverage(rows) | {"shards": shards}


def prepare(
    lock,
    raw_dir,
    tokenizer_dir,
    output,
    exclusions,
    train=10000,
    dev=1002,
    sealed_test=1002,
    seed=29092026,
    code_limit=20000,
    shard_rows=1000,
    max_tokens=1792,
):
    if output.exists():
        raise ValueError("frozen output already exists; choose a new directory")
    if min(train, dev, sealed_test, shard_rows, max_tokens) <= 0 or code_limit < 0:
        raise ValueError("counts and token limits must be positive")
    spec = json.loads(lock.read_text())
    if spec.get("schema") != "continuous_w1ax_sources_v1":
        raise ValueError("source lock schema mismatch")
    tokenizer = CpuTokenizer(tokenizer_dir, spec["tokenizer"])
    banned_ids, banned_hashes, banned_groups, banned_topics, exclusion_records = opaque_exclusions(
        exclusions
    )
    rows, rejected, source_counts, seen_ids = [], Counter(), {}, set()
    duplicate = DuplicateFilter()
    for source in spec["sources"]:
        raw = raw_dir / f"{source['id']}.jsonl"
        if sha256(raw) != source["sha256"]:
            raise ValueError("raw source SHA256 mismatch")
        count = 0
        for row in transform(source, raw, code_limit if source["id"] == "magicoder" else 0):
            count += 1
            if row["id"] in seen_ids:
                raise ValueError("source row ID collision")
            seen_ids.add(row["id"])
            if (
                row["id"] in banned_ids
                or row["content_sha256"] in banned_hashes
                or row["group"] in banned_groups
                or row["topic"] in banned_topics
                or row["legacy_topic"] in banned_topics
            ):
                rejected["opaque_legacy_exclusion"] += 1
                continue
            if not 32 <= len(row["text"]) <= 48000:
                rejected["characters"] += 1
                continue
            row["input_tokens"] = tokenizer.count(row["messages"])
            if row["input_tokens"] > max_tokens:
                rejected["tokens"] += 1
                continue
            if duplicate.add(row["text"]):
                rows.append(row)
        source_counts[source["id"]] = count
    groups = group_rows(rows)
    finals, groups = select(groups, quotas(sealed_test), seed, "continuous_sealed_test")
    devs, groups = select(groups, quotas(dev), seed, "continuous_dev")
    trains, reserve = select(groups, quotas(train), seed, "continuous_train")
    selected = {"train": trains, "dev": devs, "sealed_test": finals, "train_reserve": reserve}
    identities = {name: {r["id"] for g in gs for r in g} for name, gs in selected.items()}
    for i, name in enumerate(selected):
        for other in list(selected)[i + 1 :]:
            if identities[name] & identities[other]:
                raise ValueError("split identity leakage")
    with staged_output(output) as staging:
        files = {
            name: write_split(staging, name, [r for g in gs for r in g], shard_rows, seed)
            for name, gs in selected.items()
            if gs
        }
        report = {
            "schema": "continuous_w1ax_prompt_manifest_v1",
            "seed": seed,
            "source_lock_sha256": sha256(lock),
            "preparation_script_sha256": sha256(Path(__file__)),
            "sources": spec["sources"],
            "source_rows_considered": source_counts,
            "tokenizer": spec["tokenizer"]
            | {
                "chat": "pinned_template;add_generation_prompt=true;enable_thinking=false",
                "implementation": "tokenizers+jinja2 sandbox (CPU only)",
                "versions": {
                    n: importlib.metadata.version(n) for n in ("tokenizers", "jinja2", "numpy")
                },
            },
            "filter": {
                "min_normalized_characters": 32,
                "max_normalized_characters": 48000,
                "max_input_tokens": max_tokens,
                "code_source_row_limit": code_limit,
                "rejected": dict(rejected),
            },
            "deduplication": {
                "method": (
                    "exact normalized words;minhash XOR LSH 4x4 candidate 5gram Jaccard>=0.88"
                ),
                "dropped": dict(duplicate.dropped),
                "scope": (
                    "all considered sources before grouped split;"
                    "approximate near candidates;no semantic proof"
                ),
            },
            "opaque_exclusions": exclusion_records,
            "leakage": {
                "group_identity_and_exact_normalized_cross_split": "passed",
                "legacy": (
                    "opaque listed source IDs/groups/topics and exact message hashes "
                    "excluded;no legacy semantic/near overlap claim"
                ),
            },
            "files": files,
            "sealed_test": (
                "frozen at creation;never reopened by preparation;"
                "not eligible for training/selection"
            ),
            "training_readiness": (
                "PROMPTS_ONLY;"
                " native exact-prefix teacher capture/audit and A8/A1 readiness "
                "required before optimization"
            ),
            "coverage_definition": (
                "unique input tokens count once per selected prompt;"
                "not unique vocabulary tokens or supervised predictions;"
                "repeated presentations must be tracked separately"
            ),
            "expansion": (
                "raw source streaming;code_limit=0 considers all pinned code rows;"
                "new immutable output required;"
                "reserve may be explicitly promoted after holdout-preserving exclusion"
            ),
        }
        (staging / "manifest.json").write_bytes(canonical(report) + b"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "configs/continuous_w1ax_sources.json")
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--exclude-index", type=Path, action="append", default=[])
    parser.add_argument("--train", type=int, default=10000)
    parser.add_argument("--dev", type=int, default=1002)
    parser.add_argument("--sealed-test", type=int, default=1002)
    parser.add_argument("--seed", type=int, default=29092026)
    parser.add_argument("--code-limit", type=int, default=20000)
    parser.add_argument("--shard-rows", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=1792)
    args = parser.parse_args()
    for source in json.loads(args.lock.read_text())["sources"]:
        fetch(source, args.raw_dir, args.offline)
    if args.fetch_only:
        print(json.dumps({"raw_dir": str(args.raw_dir), "status": "source hashes verified"}))
        return
    if args.output is None or args.tokenizer_dir is None:
        parser.error("--output and --tokenizer-dir required for preparation")
    result = prepare(
        args.lock,
        args.raw_dir,
        args.tokenizer_dir,
        args.output,
        args.exclude_index,
        args.train,
        args.dev,
        args.sealed_test,
        args.seed,
        args.code_limit,
        args.shard_rows,
        args.max_tokens,
    )
    print(
        json.dumps(
            {
                "manifest": str(args.output / "manifest.json"),
                "sha256": sha256(args.output / "manifest.json"),
                "counts": {k: v["prompts_count"] for k, v in result["files"].items()},
            }
        )
    )


if __name__ == "__main__":
    main()

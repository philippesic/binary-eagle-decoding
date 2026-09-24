"""Freeze the bounded head-QAT pilot's disjoint 96/24 prompt manifests.

Run once from the repository root::

    python scripts/generate_qat_head_prompts.py

The generated ``data/qat-head-pilot`` files are ignored by Git. Re-running with
the same source and held-out manifest produces byte-identical files.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260924

PROSE_TOPICS = (
    "how a violin bow makes a string vibrate",
    "why bread dough rises",
    "how a lighthouse lens focuses light",
    "how bees communicate a food source",
    "why a river forms a delta",
    "how a ceramic glaze changes in a kiln",
    "why ocean tides vary through the month",
    "how a camera aperture affects a photograph",
    "why a ship floats",
    "how a solar eclipse occurs",
    "why leaves change color",
    "how a bicycle derailleur shifts gears",
    "why a rainbow has ordered colors",
    "how a public library lends digital books",
    "how an orchestra follows a conductor",
    "why a cast iron pan holds heat",
    "how a sailboat moves across the wind",
    "how a seed disperses by wind",
    "why a bridge needs expansion joints",
    "how a telescope mirror gathers light",
    "why a snowflake grows branches",
    "how a beaver dam changes a stream",
    "why a compass needle points north",
    "how a pottery wheel centers clay",
    "why a guitar string changes pitch",
    "how a bat navigates in darkness",
    "why the moon has phases",
    "how a radio receiver selects a station",
    "why a glacier moves",
    "how a museum conservator protects paper",
    "why fermentation preserves cabbage",
    "how a wind turbine converts motion to electricity",
    "why an airplane wing produces lift",
    "how coral polyps build a reef",
    "why a thermos slows heat transfer",
    "how an ant colony allocates foraging work",
    "why a wooden door swells in humidity",
    "how a music box plays a melody",
    "why a hot air balloon rises",
    "how a lock and key interact",
)

CODE_TASKS = (
    "Parse a CSV line that may contain quoted commas in Python.",
    "Implement binary search for the first value greater than a target in Python.",
    "Find the longest common prefix of a list of strings in Python.",
    "Write a Python iterator that yields fixed-size chunks from a sequence.",
    "Validate balanced parentheses, brackets, and braces in Python.",
    "Implement a bounded least-recently-used cache in Python.",
    "Compute a histogram of Unicode characters in Python.",
    "Transpose a rectangular matrix in Python and reject ragged rows.",
    "Implement breadth-first traversal of a graph in Python.",
    "Find the first nonrepeating character in a string in Python.",
    "Rotate a square matrix clockwise in Python.",
    "Implement a min-heap push operation in Python.",
    "Deduplicate a sequence while preserving order in Python.",
    "Parse an ISO date and reject invalid calendar days in Python.",
    "Compute a rolling average over a fixed window in Python.",
    "Implement edit distance for two short strings in Python.",
    "Find connected components in an undirected graph in Python.",
    "Convert an adjacency list to an adjacency matrix in Python.",
    "Write a Python context manager that times a code block.",
    "Validate a simple IPv4 dotted-quad address in Python.",
    "Implement a trie insert and prefix lookup in Python.",
    "Find the maximum sum contiguous subarray in Python.",
    "Implement stable insertion sort in Python.",
    "Decode run-length-encoded text in Python with input validation.",
    "Write a JavaScript debounce helper that preserves the last arguments.",
    "Write a JavaScript promise pool with a concurrency limit.",
    "Flatten a nested JavaScript array without changing element order.",
    "Validate a JavaScript object against a set of required fields.",
    "Implement a JavaScript binary search over a sorted numeric array.",
    "Build a JavaScript event emitter with unsubscribe support.",
    "Write a SQL query to count active subscriptions by plan.",
    "Write a SQL query that finds products never purchased.",
    "Write a SQL query to compute a seven-day moving total of daily sales.",
    "Write a SQL query to rank exam scores within each class.",
    "Write a SQL query that detects duplicate email addresses.",
    "Explain how to index a table for a filter on city and a sort on date.",
    "Write a Python function to normalize whitespace outside quoted text.",
    "Implement topological sort with cycle rejection in Python.",
    "Write a Python function to compute a file's SHA256 by streaming chunks.",
    "Implement an interval overlap check for half-open intervals in Python.",
)


def reasoning_questions() -> list[str]:
    """Five unrelated arithmetic families, eight parameterizations each."""
    questions = []
    for index in range(8):
        n = 4 + index
        price = 7 + index
        discount = 2 + index
        questions.append(
            f"A shop sells {n} notebooks at ${price} each and applies a ${discount} "
            "discount to the total. What is the final price? Show the arithmetic."
        )
        red, blue = 3 + index, 5 + index
        questions.append(
            f"A bag contains {red} red beads and {blue} blue beads. Two beads are "
            "drawn without replacement. What is the probability both are blue? "
            "Show the calculation."
        )
        start, gap = 6 + index, 3 + index
        questions.append(
            f"A sequence starts at {start} and increases by {gap} each step. "
            "What is its 12th term? Explain the indexing."
        )
        width, length = 5 + index, 9 + index
        questions.append(
            f"A rectangular garden is {width} meters wide and {length} meters "
            "long. A one-meter border is added outside every edge. How much "
            "larger is the total area? Show the dimensions used."
        )
        workers, hours = 3 + index, 4 + index
        questions.append(
            f"If {workers} workers each pack 6 parcels per hour for {hours} "
            "hours, how many parcels are packed? Explain each factor."
        )
    return questions


def canonical_content(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(messages: list[dict]) -> str:
    return hashlib.sha256(canonical_content(messages).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_disjoint(prompts: list[dict], heldout: list[dict]) -> None:
    ids = [row["id"] for row in prompts]
    hashes = [content_hash(row["messages"]) for row in prompts]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("generated prompt IDs or content repeat")
    heldout_ids = {row["id"] for row in heldout}
    heldout_hashes = {content_hash(row["messages"]) for row in heldout}
    if heldout_ids.intersection(ids) or heldout_hashes.intersection(hashes):
        raise ValueError("QAT prompts overlap held-out acceptance prompts")
    if any(row.get("content_sha256") != content_hash(row["messages"]) for row in prompts):
        raise ValueError("generated prompt content hash mismatch")


def build_prompts() -> tuple[list[dict], list[dict]]:
    pools = {
        "prose": [
            f"Explain {topic} to a curious adult. "
            "Include a concrete example and a common misunderstanding."
            for topic in PROSE_TOPICS
        ],
        "code": [f"{task} Explain one edge case and the time complexity." for task in CODE_TASKS],
        "reasoning": reasoning_questions(),
    }
    if any(len(items) != 40 for items in pools.values()):
        raise RuntimeError("each prompt category must contain exactly 40 items")
    rng = random.Random(SEED)
    by_split = {"train": {}, "validation": {}}
    for category, items in pools.items():
        selected = items.copy()
        rng.shuffle(selected)
        by_split["train"][category] = selected[:32]
        by_split["validation"][category] = selected[32:]
    output = {}
    for split, per_category in by_split.items():
        rows = []
        for index in range(len(per_category["prose"])):
            for category in ("prose", "code", "reasoning"):
                messages = [{"role": "user", "content": per_category[category][index]}]
                rows.append(
                    {
                        "id": f"qat-{split}-{category}-{index + 1:03}",
                        "category": category,
                        "messages": messages,
                        "content_sha256": content_hash(messages),
                    }
                )
        output[split] = rows
    return output["train"], output["validation"]


def generate(output_dir: Path, heldout_path: Path) -> dict:
    heldout = read_jsonl(heldout_path)
    train, validation = build_prompts()
    validate_disjoint(train + validation, heldout)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {"train": output_dir / "train.jsonl", "validation": output_dir / "validation.jsonl"}
    for split, rows in (("train", train), ("validation", validation)):
        paths[split].write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
            )
        )
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "heldout_sha256": sha256_file(heldout_path),
        "generator_sha256": sha256_file(Path(__file__)),
        "splits": {
            split: {
                "path": path.name,
                "sha256": sha256_file(path),
                "prompts": len(rows),
                "categories": {
                    category: sum(row["category"] == category for row in rows)
                    for category in ("prose", "code", "reasoning")
                },
            }
            for split, path, rows in (
                ("train", paths["train"], train),
                ("validation", paths["validation"], validation),
            )
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/qat-head-pilot")
    parser.add_argument("--heldout", type=Path, default=ROOT / "configs/acceptance_prompts.jsonl")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.heldout), indent=2))


if __name__ == "__main__":
    main()

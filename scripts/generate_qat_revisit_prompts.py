# ruff: noqa: E501 -- keep frozen prompt literals intact for reproducible hashes
"""Freeze fresh prompt-level train/development/final manifests for QAT revisit.

Generated files default to the ignored ``data/qat-revisit`` directory. The
corpus is fixed in this source file; generation is deterministic and performs
overlap audits against the historical acceptance prompts and any first-pilot
manifests that exist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260924
SPLITS = ("train", "development", "final")

# Each family stays entirely in one split. The prompts within a family share a
# topic and response-template style, while families are distinct across splits.
CORPUS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "prose": (
        (
            "urban-waterways",
            (
                "Explain how a city canal can support transport and wildlife. Give one example of a tradeoff.",
                "Describe how restoring a buried urban stream could change a neighborhood over time.",
                "Write a short scene beside a canal at dawn, using sound and movement to establish place.",
                "Compare two ways a city can reduce litter entering its waterways.",
            ),
        ),
        (
            "night-sky-observation",
            (
                "Explain why a planet can appear to move backward against the stars. Use a simple analogy.",
                "Write a brief account of an amateur astronomer noticing a change in the night sky.",
                "Give a beginner a practical plan for observing the Moon over one month.",
                "Describe one common misconception about why stars twinkle and correct it.",
            ),
        ),
        (
            "food-preservation",
            (
                "Explain how drying can preserve fruit and name a condition that affects the result.",
                "Compare fermentation and refrigeration as ways to preserve vegetables.",
                "Write a short memory of learning a family recipe for preserving food.",
            ),
        ),
        (
            "community-libraries",
            (
                "Describe how a small library might help residents who have limited internet access.",
                "Compare lending physical books and lending e-books from a public library.",
                "Suggest a welcoming event that could introduce teenagers to a local library.",
            ),
        ),
        (
            "mountain-weather",
            (
                "Explain why weather can change quickly on a mountain. Include a safety example.",
                "Describe a hiker's decision to turn back when clouds gather near a ridge.",
                "Give a clear comparison between a valley breeze and a mountain breeze.",
            ),
        ),
        (
            "traditional-crafts",
            (
                "Explain how a potter uses controlled drying to reduce cracks in a clay vessel.",
                "Write a short scene in which a craftsperson repairs a damaged wooden chair.",
                "Compare hand weaving and machine weaving in terms of time and variation.",
            ),
        ),
        (
            "urban-trees",
            (
                "Explain two ways street trees can affect summer temperatures for pedestrians.",
                "Describe a disagreement about where to plant trees on a narrow street.",
                "Give a practical example of how tree roots and sidewalks can be planned together.",
            ),
        ),
        (
            "sound-and-music",
            (
                "Explain how changing the length of a vibrating string changes its pitch.",
                "Describe how a conductor can help an orchestra recover after losing its tempo.",
                "Compare the sound of a small room and a large hall for a string quartet.",
            ),
        ),
        (
            "coastal-erosion",
            (
                "Explain how waves can move sand along a coast and give a visible sign of this.",
                "Compare a seawall and a restored dune as responses to coastal erosion.",
                "Write a brief field note about a beach changing after a winter storm.",
            ),
        ),
        (
            "transport-access",
            (
                "Describe how a bus route change could affect people who work late shifts.",
                "Compare cycling and walking for a short trip through a busy town.",
                "Suggest a way to make a train station easier to use for travelers with luggage.",
            ),
        ),
        (
            "household-energy",
            (
                "Explain how insulation slows heat transfer through a home's walls.",
                "Compare two household habits that can reduce electricity use in summer.",
            ),
        ),
        (
            "pollinators",
            (
                "Explain how a garden can provide food for pollinators throughout the season.",
                "Describe how a bee communicates the location of a distant food source.",
                "Give an example of how a farm could support pollinators without losing its crop.",
            ),
        ),
        (
            "maps-and-navigation",
            (
                "Explain why a flat map distorts some properties of the Earth. Give an example.",
                "Write a short scene about a traveler choosing between an old map and a new sign.",
                "Describe how landmarks can help someone navigate when a phone battery dies.",
            ),
        ),
        (
            "water-in-landscapes",
            (
                "Explain how a wetland can reduce flooding downstream after heavy rain.",
                "Compare a pond and a reservoir as parts of a local water system.",
                "Describe a stream before and after beavers build a dam nearby.",
            ),
        ),
        (
            "museum-care",
            (
                "Explain why museums control light and humidity around fragile paper objects.",
                "Describe a conservator deciding how to display a faded travel journal.",
                "Suggest a label that helps visitors understand a repaired ceramic artifact.",
            ),
        ),
        (
            "weather-instruments",
            (
                "Explain how a barometer can provide clues about changing weather.",
                "Compare a rain gauge and a weather radar for measuring precipitation.",
            ),
        ),
    ),
    "code": (
        (
            "streaming-text",
            (
                "Write a Python generator that reads a large text file line by line and skips blank lines. State its memory use.",
                "Implement a Python function that counts words from an iterable without storing every word.",
                "Write a JavaScript async generator that yields paginated API records until no next-page token remains.",
                "Show how to process a newline-delimited JSON file and report malformed rows with line numbers.",
            ),
        ),
        (
            "graph-paths",
            (
                "Implement breadth-first search in Python to return the shortest unweighted path between two nodes.",
                "Write a JavaScript function that detects whether a directed graph contains a cycle.",
                "Give pseudocode for Dijkstra's algorithm and explain how stale priority-queue entries are handled.",
                "Implement topological ordering in Python and return a useful error when a cycle exists.",
            ),
        ),
        (
            "interval-scheduling",
            (
                "Write a Python function that merges overlapping half-open intervals and preserves sorted order.",
                "Implement a JavaScript check for whether a meeting can be inserted without overlap.",
                "Show an algorithm that selects the maximum number of nonoverlapping activities.",
            ),
        ),
        (
            "database-queries",
            (
                "Write SQL that returns each account's latest payment, keeping accounts that have no payments.",
                "Write SQL to find products that have never appeared in an order line.",
                "Show a SQL query for a seven-day rolling sum of daily event counts.",
            ),
        ),
        (
            "data-validation",
            (
                "Implement a Python validator for an IPv4 address that rejects out-of-range octets.",
                "Write JavaScript code that validates required fields and reports all missing field names.",
                "Parse an ISO date in Python and reject impossible dates such as April 31.",
            ),
        ),
        (
            "caches",
            (
                "Implement an LRU cache in Python with get and put operations and explain their expected complexity.",
                "Write a bounded JavaScript memoization helper with a maximum number of stored keys.",
                "Describe how to add expiration timestamps to a cache without scanning every entry on each read.",
            ),
        ),
        (
            "sequence-processing",
            (
                "Write a Python function that yields fixed-size chunks and includes a shorter final chunk.",
                "Implement stable deduplication of an iterable while preserving its first-seen order.",
                "Write JavaScript code to rotate an array left by k positions, including large and negative k.",
            ),
        ),
        (
            "string-algorithms",
            (
                "Implement a Unicode-aware character frequency counter in Python.",
                "Write a function that finds the longest common prefix in a list of strings.",
                "Implement a simple run-length decoder that rejects missing counts and malformed input.",
            ),
        ),
        (
            "concurrency",
            (
                "Write a JavaScript promise pool that runs at most n tasks at once and preserves result order.",
                "Explain how a Python thread-safe queue can coordinate producers and consumers.",
                "Implement a Python worker that retries a task twice and records the final exception.",
            ),
        ),
        (
            "trees",
            (
                "Implement iterative inorder traversal of a binary tree in Python.",
                "Write a function that verifies whether a binary search tree satisfies strict ordering bounds.",
                "Describe an efficient way to serialize a tree while preserving missing children.",
            ),
        ),
        (
            "numeric-arrays",
            (
                "Implement a rolling mean over a fixed-width numeric window without recomputing each sum.",
                "Write Python code that transposes a rectangular matrix and rejects ragged rows.",
                "Find the maximum-sum contiguous subarray and return both its sum and index range.",
            ),
        ),
        (
            "resource-management",
            (
                "Write a Python context manager that measures elapsed time even if the block raises an exception.",
                "Show how to close a file reliably when parsing can fail halfway through.",
                "Implement a JavaScript helper that aborts a fetch request after a timeout.",
            ),
        ),
        (
            "search-and-order",
            (
                "Implement binary search for the first element greater than a target in a sorted Python list.",
                "Write stable insertion sort and explain when it can outperform a comparison sort.",
            ),
        ),
        (
            "text-normalization",
            (
                "Write a Python function that normalizes whitespace outside quoted strings.",
                "Implement case-insensitive comparison that preserves the original spelling for display.",
            ),
        ),
        (
            "event-systems",
            (
                "Build a JavaScript event emitter with an unsubscribe function for each listener.",
                "Implement a Python observer registry that avoids calling a listener twice.",
                "Show how to coalesce repeated UI events while retaining the most recent arguments.",
                "Explain how to remove one-shot listeners safely while events are being dispatched.",
            ),
        ),
        (
            "file-integrity",
            (
                "Write a Python function that computes a file's SHA256 by reading fixed-size chunks.",
                "Implement a check that compares a file size and digest with a JSON manifest.",
            ),
        ),
    ),
    "reasoning": (
        (
            "rate-and-work",
            (
                "A pump adds 9 liters per minute while a drain removes 2 liters per minute. Starting empty, how long to reach 84 liters? Show the rate calculation.",
                "Four workers each pack 7 boxes per hour for 5 hours. How many boxes do they pack? Explain each factor.",
                "A cyclist travels 36 km in 90 minutes. What is the average speed in km/h? Show the unit conversion.",
                "Two printers produce 18 pages per minute and 12 pages per minute. How long do they need together for 900 pages? State the assumption.",
            ),
        ),
        (
            "fractions-and-ratios",
            (
                "A jar has 5 green and 7 yellow beads. Two are drawn without replacement. What is the probability both are green? Show the fractions.",
                "A drink uses 3 parts juice for every 5 parts water. How much juice is needed for 32 cups total? Explain the ratio.",
                "A recipe for 6 people uses 450 grams of rice. Scale it to 10 people and show the multiplier.",
                "A class has 12 students in one group and 18 in another. What fraction of the class is in the smaller group? Simplify it.",
            ),
        ),
        (
            "sequences",
            (
                "A sequence starts at 11 and adds 4 each time. What is its 15th term? Explain how you count steps.",
                "The sequence 2, 6, 18, 54 continues by the same rule. Give the next two terms and name the rule.",
                "A row of tiles uses 3 tiles in row one and adds 2 tiles per row. How many tiles are in row 20? Show the formula.",
            ),
        ),
        (
            "geometry-and-area",
            (
                "A rectangular garden is 8 meters by 13 meters. A one-meter border surrounds it outside. How much larger is the total area? Show dimensions.",
                "A square has perimeter 52 cm. What is its area? Show how you find the side length.",
                "A right triangle has legs of 6 and 8 units. Find its hypotenuse and explain the theorem used.",
            ),
        ),
        (
            "logic-and-constraints",
            (
                "Three boxes are labeled pears, plums, and mixed, and every label is wrong. One fruit may be drawn from one box. Which box do you choose and why?",
                "A, B, and C each make one statement; exactly one statement is true. A says the key is in B, B says it is not in B, and C says it is in A. Where can the key be? Explain.",
                "Four tasks must be scheduled in order. P precedes Q, R follows Q, and S is before P. Give a valid order and explain the constraints.",
            ),
        ),
        (
            "probability-and-counting",
            (
                "A fair six-sided die is rolled twice. What is the probability the sum is 7? Count the outcomes.",
                "How many different two-letter codes can be made from A, B, C, and D if letters cannot repeat? Explain the counting.",
                "A bag has 4 red and 6 blue tokens. One token is drawn, replaced, then another is drawn. What is the chance both are blue?",
            ),
        ),
        (
            "averages-and-mixtures",
            (
                "A route has two equal 60 km legs driven at 40 km/h and 80 km/h. Find the average speed for the full route.",
                "Five quiz scores average 14. Four scores are 10, 12, 15, and 18. What is the fifth score? Show the total.",
                "Mix 2 liters of a 10% solution with 3 liters of a 30% solution. What percentage of the mixture is active ingredient?",
            ),
        ),
        (
            "spatial-reasoning",
            (
                "A cube is painted on every outside face and cut into 27 equal small cubes. How many small cubes have exactly two painted faces? Explain their positions.",
                "A person walks 3 blocks north, 4 east, then 3 south. How far and in what direction are they from the start?",
                "A paper square is folded in half vertically and then horizontally. One hole is punched away from every fold. How many holes appear when unfolded?",
            ),
        ),
        (
            "money-and-discounts",
            (
                "A shop sells 7 notebooks at $8 each and takes $6 off the total. What is the final price? Show the arithmetic.",
                "A $120 item is discounted by 15%, then taxed by 10%. What is the final price? State the order of operations.",
                "A person saves $18 each week toward a $245 purchase and already has $47. How many full weeks are needed?",
            ),
        ),
        (
            "calendar-and-cycles",
            (
                "A project begins on a Tuesday and lasts 45 days, counting the start date as day one. What weekday is the final day? Show the remainder.",
                "Two rotating signs flash every 8 and 12 seconds. If they flash together now, when do they next flash together?",
                "A meeting repeats every 9 days. If one occurs on the 4th, on what date is the next one in a 31-day month?",
            ),
        ),
        (
            "comparative-evidence",
            (
                "A town's recycling rose after a campaign, but collection rules also changed. What can and cannot be concluded about the campaign?",
                "Two routes have equal distance; one is faster in light traffic but slower in heavy traffic. What additional data would help choose?",
                "A survey reports that 70% of 20 volunteers prefer option A. Explain why this may not represent all residents.",
            ),
        ),
        (
            "deductive-classification",
            (
                "All glims are blue. No blue objects are fragile. A particular glim is described. What follows about its color and fragility?",
                "Every sealed envelope contains a card; some cards are red. Can you conclude that some envelopes contain red cards? Explain.",
                "If a device is charging, its indicator is lit. The indicator is not lit. What can be concluded under this rule?",
            ),
        ),
        (
            "optimization",
            (
                "You have 24 meters of fencing for three sides of a rectangular pen against a wall. What dimensions maximize its area? Show the reasoning.",
                "A delivery van can carry at most 30 crates. Crate A earns $8 and weighs 2 units; crate B earns $11 and weighs 3 units. Which whole-crate mix earns most?",
            ),
        ),
        (
            "invariants-and-puzzles",
            (
                "A board has two opposite corners removed. Can the remaining squares be covered with dominoes that each cover two adjacent squares? Explain with an invariant.",
                "A jug holds 5 liters and another holds 3 liters. Using only these jugs and a tap, how can you measure exactly 4 liters?",
            ),
        ),
        (
            "proportional-change",
            (
                "A map uses 1 cm for 5 km. Two towns are 7.4 cm apart on the map. What is their real distance?",
                "A machine makes 240 parts in 6 hours at a constant rate. How many hours for 350 parts?",
            ),
        ),
        (
            "error-checking",
            (
                "A student says 3/4 + 2/5 = 5/9. Identify the mistake and calculate the correct sum.",
                "Someone claims that traveling 30 km at 30 km/h and 30 km at 60 km/h gives an average of 45 km/h. Check the claim.",
            ),
        ),
        (
            "multi-step-planning",
            (
                "A hiker has 18 km of daylight travel, walks 4 km/h, and must rest 30 minutes after every 2 hours of walking. Can they finish before 5 hours? Show the timeline.",
                "A group needs 73 seats. Each bus has 28 seats and costs $190; a van has 8 seats and costs $70. Find the cheapest whole-vehicle plan.",
            ),
        ),
    ),
}


def canonical_content(messages: list[dict]) -> str:
    return json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(messages: list[dict]) -> str:
    return hashlib.sha256(canonical_content(messages).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def family_split(index: int) -> str:
    if index < 10:
        return "train"
    if index < 13:
        return "development"
    return "final"


def build_splits() -> dict[str, list[dict]]:
    splits = {name: [] for name in SPLITS}
    family_ids: dict[str, str] = {}
    for category, families in CORPUS.items():
        expected_families = 17 if category == "reasoning" else 16
        if len(families) != expected_families:
            raise ValueError(f"{category} must define {expected_families} independent families")
        for family_index, (topic, prompts) in enumerate(families):
            split = family_split(family_index)
            family_id = f"{category}:{topic}"
            if family_id in family_ids:
                raise ValueError(f"duplicate topic family: {family_id}")
            family_ids[family_id] = split
            template_family = f"{category}:{topic}:template"
            for prompt_index, prompt in enumerate(prompts, start=1):
                messages = [{"role": "user", "content": prompt}]
                splits[split].append(
                    {
                        "id": f"qat-revisit-{split}-{category}-{topic}-{prompt_index:02d}",
                        "category": category,
                        "topic_family": family_id,
                        "template_family": template_family,
                        "messages": messages,
                        "content_sha256": content_hash(messages),
                    }
                )
    for split, rows in splits.items():
        for category in CORPUS:
            count = sum(row["category"] == category for row in rows)
            expected = 32 if split == "train" else 8
            if count != expected:
                raise ValueError(f"{split}/{category}: expected {expected}, got {count}")
    return splits


def _audit_against(rows: Iterable[dict], audit_rows: Iterable[dict], label: str) -> None:
    rows = list(rows)
    audit_rows = list(audit_rows)
    ids = {row["id"] for row in rows}
    hashes = {content_hash(row["messages"]) for row in rows}
    other_ids = {row["id"] for row in audit_rows}
    other_hashes = {content_hash(row["messages"]) for row in audit_rows}
    if ids & other_ids:
        raise ValueError(f"prompt ID overlap with {label}: {sorted(ids & other_ids)[:3]}")
    if hashes & other_hashes:
        raise ValueError(f"prompt content overlap with {label}")


def audit_splits(
    splits: dict[str, list[dict]], acceptance_path: Path, pilot_dir: Path | None = None
) -> list[dict]:
    rows = [row for subset in splits.values() for row in subset]
    ids = [row["id"] for row in rows]
    hashes = [content_hash(row["messages"]) for row in rows]
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError("new manifest prompt IDs or content repeat")
    if any(row.get("content_sha256") != content_hash(row["messages"]) for row in rows):
        raise ValueError("new manifest prompt content hash mismatch")
    topic_splits: dict[str, set[str]] = {}
    template_splits: dict[str, set[str]] = {}
    for split, subset in splits.items():
        for row in subset:
            topic_splits.setdefault(row["topic_family"], set()).add(split)
            template_splits.setdefault(row["template_family"], set()).add(split)
    if any(len(used) != 1 for used in (*topic_splits.values(), *template_splits.values())):
        raise ValueError("topic or template family crosses manifest splits")

    audit_sources = [
        {"name": "acceptance", "path": acceptance_path, "rows": read_jsonl(acceptance_path)}
    ]
    if pilot_dir and pilot_dir.exists():
        for path in sorted(pilot_dir.glob("*.jsonl")):
            audit_sources.append(
                {"name": f"first-pilot:{path.name}", "path": path, "rows": read_jsonl(path)}
            )
    audits = []
    for source in audit_sources:
        _audit_against(rows, source["rows"], source["name"])
        audits.append(
            {
                "name": source["name"],
                "path": display_path(source["path"]),
                "sha256": sha256_file(source["path"]),
                "prompts": len(source["rows"]),
            }
        )
    return audits


def generate(output_dir: Path, acceptance_path: Path, pilot_dir: Path | None = None) -> dict:
    splits = build_splits()
    audits = audit_splits(splits, acceptance_path, pilot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_info = {}
    for split, rows in splits.items():
        path = output_dir / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
        split_info[split] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "prompts": len(rows),
            "categories": {
                category: sum(row["category"] == category for row in rows) for category in CORPUS
            },
        }
    manifest = {
        "schema_version": 1,
        "seed": SEED,
        "split_policy": "whole topic_family and template_family; 32/8/8 per category",
        "generator_sha256": sha256_file(Path(__file__)),
        "audits": audits,
        "splits": split_info,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/qat-revisit")
    parser.add_argument(
        "--acceptance", type=Path, default=ROOT / "configs/acceptance_prompts.jsonl"
    )
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / "data/qat-head-pilot")
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir, args.acceptance, args.pilot_dir), indent=2))


if __name__ == "__main__":
    main()

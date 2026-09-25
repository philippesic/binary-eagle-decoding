"""Create the frozen, separate context-length diagnostic for W1Ax.

These prompts are intentionally synthetic. They test prefill and decode cost at
different context lengths, not draft quality, and do not use QAT-final data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"short": 3, "medium": 9, "long": 20}

PASSAGES = {
    "prose": (
        "The harbor team kept a paper log beside each tide gauge. A clear morning reading could still be followed by a difficult afternoon, because wind and rainfall changed the water level independently of the predicted tide. The observer wrote down the instrument time and the condition of the dock.",
        "Each evening a second observer compared the gauge log with photographs of the shoreline. The photographs made unusual readings easier to check, especially when floating debris pressed against a sensor. A missing photograph was marked as missing rather than silently replaced with an estimate.",
        "At the end of the week the team discussed which observations could guide repairs. They separated measurements from interpretation, recorded uncertainty about faulty sensors, and kept a copy of the original notes. This let a later reader understand why one proposed repair had been delayed.",
    ),
    "code": (
        "A small event service receives numbered records from several producers. It stores each record once by its stable identifier, then writes an acknowledgment after the storage transaction commits. A retry can repeat an identifier, so the insert must be idempotent and preserve the first accepted payload.",
        "A reader requests records by cursor and page size. The cursor refers to the last returned key, not an offset into a changing array. The service orders keys consistently, limits each page, and returns an explicit next cursor only when more keys remain.",
        "The operators test failure between insertion and acknowledgment, an empty page, and a duplicate retry. They record elapsed time and error counts separately from correctness. A compact log contains the request identifier, cursor, result count, and a status code for diagnosis.",
    ),
    "reasoning": (
        "A field team visits three monitoring sites along a river. Travel from the depot to the upstream site takes twenty minutes, and the journey between adjacent sites takes fifteen minutes. Each visit requires a ten minute measurement, while changing equipment at the middle site adds five minutes.",
        "The crew has one portable battery. It starts with one hundred units, loses eight units per journey, and uses twelve units during each measurement. Charging is available only at the depot, so the route must reserve enough energy for the return journey as well as every visit.",
        "A supervisor compares routes by total time and remaining battery. The written plan states the site order, accounts for every travel and measurement step, and checks the return leg explicitly. If two routes tie on time, the supervisor prefers the route with more battery left.",
    ),
}

QUESTIONS = {
    "prose": "Summarize how the team checked unusual readings and kept uncertainty visible. Use two sentences.",
    "code": "Describe the retry-safe insertion and cursor pagination rules in a concise numbered list.",
    "reasoning": "Explain which travel, measurement, and return costs a valid route calculation must include. Keep the answer concise.",
}


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category, passages in PASSAGES.items():
        for length_bin, count in COUNTS.items():
            body = "\n".join(f"Note {i + 1}: {passages[i % len(passages)]}" for i in range(count))
            content = f"Read the notes below.\n\n{body}\n\n{QUESTIONS[category]}"
            rows.append({
                "id": f"context-{category}-{length_bin}",
                "category": category,
                "context_bin": length_bin,
                "messages": [{"role": "user", "content": content}],
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data/w1ax-context/prompts.jsonl")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in build_rows())
    args.output.write_text(content)
    print(json.dumps({"path": str(args.output), "prompts": 9,
                      "sha256": hashlib.sha256(content.encode()).hexdigest()}))


if __name__ == "__main__":
    main()

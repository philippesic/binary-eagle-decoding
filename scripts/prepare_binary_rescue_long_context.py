"""Freeze the small longer-context diagnostic using development prompts only."""

import argparse
import hashlib
import json
from pathlib import Path

DEV_SHA256 = "a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885"
NOTE = "Reference entry: the archive stores dated notes in numbered folders.\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.development.read_bytes()
    if hashlib.sha256(raw).hexdigest() != DEV_SHA256:
        raise ValueError("expected frozen development manifest")
    rows = [json.loads(line) for line in raw.splitlines() if line]
    chosen = [row for category in ("prose", "code", "reasoning")
              for row in [r for r in rows if r["category"] == category][:2]]
    if len(chosen) != 6:
        raise ValueError("expected two prompts per category")
    output = []
    for row in chosen:
        prefix = "Background reference notes follow. Answer the request after the notes.\n"
        prefix += NOTE * 72 + "\nRequest:\n"
        messages = [dict(message) for message in row["messages"]]
        messages[0]["content"] = prefix + messages[0]["content"]
        output.append({"id": row["id"] + "-long-context", "category": row["category"],
                       "source_id": row["id"], "messages": messages})
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                      for row in output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(payload)
    print(json.dumps({"sha256": hashlib.sha256(payload.encode()).hexdigest(),
                      "prompts": 6, "context_note_repetitions": 72,
                      "output_cap": 256, "draft_length": 5, "p_min": 0,
                      "input_tokens": "measure actual native chat-template token count"}))


if __name__ == "__main__":
    main()

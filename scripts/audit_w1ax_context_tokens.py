"""Audit frozen W1Ax context diagnostic lengths with the target chat template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

BOUNDS = {"short": (1, 256), "medium": (257, 768), "long": (769, 1536)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    rows = []
    for line in args.prompts.read_text().splitlines():
        prompt = json.loads(line)
        token_ids = tokenizer.apply_chat_template(
            prompt["messages"], tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        )
        count = len(token_ids)
        lower, upper = BOUNDS[prompt["context_bin"]]
        rows.append({"id": prompt["id"], "category": prompt["category"],
                     "context_bin": prompt["context_bin"], "tokenized_prompt_length": count,
                     "within_bin": lower <= count <= upper,
                     "fits_2048_context_with_128_output": count + 128 <= 2048})
    report = {
        "prompts_sha256": sha256(args.prompts),
        "tokenizer_json_sha256": sha256(args.tokenizer / "tokenizer.json"),
        "tokenizer_config_sha256": sha256(args.tokenizer / "tokenizer_config.json"),
        "thinking_enabled": False,
        "add_generation_prompt": True,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not rows or any(not row["within_bin"] or not row["fits_2048_context_with_128_output"] for row in rows):
        raise SystemExit("context diagnostic token-length audit failed")


if __name__ == "__main__":
    main()

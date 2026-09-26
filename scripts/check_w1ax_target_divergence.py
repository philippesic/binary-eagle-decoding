#!/usr/bin/env python3
"""Replay one historical target-only versus ordinary-EAGLE token divergence.

Run under scripts/remote_job.py. This is a greedy correctness diagnostic, not
a timing comparison. API completion logprobs are retained as such and are not
treated as raw verifier-row logits.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "native_benchmark_w1ax.toml"
DEFAULT_PROMPT_ID = "reasoning-02"
EXPECTED_HISTORICAL_MISMATCH = {
    "prompt_id": "reasoning-02",
    "zero_based_position": 109,
    "target_only_token_id": 12,
    "ordinary_eagle_token_id": 29208,
}


def load_base_runner() -> Any:
    path = ROOT / "scripts" / "benchmark_native_eagle.py"
    spec = importlib.util.spec_from_file_location("benchmark_native_eagle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load base runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_runner()


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inspect_verifier_trace(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "unavailable",
            "path": str(path),
            "expected_schema": "w1ax_verify_logits_v1",
            "reason": "server did not create the requested trace file",
        }
    raw = path.read_bytes()
    nonempty_lines = [line for line in raw.splitlines() if line.strip()]
    parsed_rows = []
    malformed_lines = 0
    for line in nonempty_lines:
        try:
            parsed_rows.append(json.loads(line))
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed_lines += 1
    observed_schemas = sorted({
        row.get("schema") for row in parsed_rows
        if isinstance(row, dict) and isinstance(row.get("schema"), str)
    })
    return {
        "status": "available" if nonempty_lines else "empty",
        "path": str(path),
        "expected_schema": "w1ax_verify_logits_v1",
        "schema_confirmed": bool(parsed_rows) and malformed_lines == 0 and all(
            isinstance(row, dict) and row.get("schema") == "w1ax_verify_logits_v1"
            for row in parsed_rows
        ),
        "observed_schemas": observed_schemas,
        "sha256": sha256_bytes(raw),
        "bytes": len(raw),
        "event_lines": len(nonempty_lines),
        "parsed_rows": len(parsed_rows),
        "malformed_lines": malformed_lines,
        "requested_positions": [109],
    }


def verifier_trace_environment(cell: Path) -> dict[str, str]:
    return {
        "W1AX_VERIFY_TRACE_JSONL": str(cell / "verifier-trace.jsonl"),
        "W1AX_VERIFY_TRACE_POSITIONS": "109",
    }


def first_difference(reference: list[int], candidate: list[int]) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return {
                "zero_based_position": index,
                "target_only_token_id": left,
                "ordinary_eagle_token_id": right,
                "target_only_length": len(reference),
                "ordinary_eagle_length": len(candidate),
            }
    if len(reference) != len(candidate):
        index = min(len(reference), len(candidate))
        return {
            "zero_based_position": index,
            "target_only_token_id": reference[index] if index < len(reference) else None,
            "ordinary_eagle_token_id": candidate[index] if index < len(candidate) else None,
            "target_only_length": len(reference),
            "ordinary_eagle_length": len(candidate),
        }
    return None


def generated_ids(response: dict[str, Any]) -> list[int] | None:
    return BASE.generated_token_ids(response)


def api_logprob_entry(
    response: dict[str, Any], position: int, generated_token_ids: list[int] | None
) -> dict[str, Any]:
    """Associate API completion logprobs only when their token positions align."""
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return {"status": "unavailable", "reason": "response has no completion choice"}
    logprobs = choices[0].get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        return {"status": "unavailable", "reason": "completion logprobs content is absent"}
    if generated_token_ids is None:
        return {
            "status": "unavailable",
            "reason": "generated token IDs are absent, so logprob positions cannot be aligned",
            "logprob_content_count": len(content),
        }
    if len(content) != len(generated_token_ids):
        return {
            "status": "misaligned",
            "reason": "completion logprob count differs from generated token ID count",
            "logprob_content_count": len(content),
            "generated_token_id_count": len(generated_token_ids),
        }
    if position < 0 or position >= len(content):
        return {
            "status": "unavailable",
            "reason": "mismatch position is outside the aligned completion logprob sequence",
            "logprob_content_count": len(content),
        }
    entry = content[position]
    if not isinstance(entry, dict):
        return {"status": "unavailable", "reason": "completion logprob entry is not an object"}
    if isinstance(entry, dict):
        for key in ("id", "token_id"):
            if key in entry:
                exposed_id = entry[key]
                if (
                    not isinstance(exposed_id, int)
                    or isinstance(exposed_id, bool)
                    or exposed_id != generated_token_ids[position]
                ):
                    return {
                        "status": "misaligned",
                        "reason": f"logprob entry {key} does not match the generated token ID",
                        "logprob_content_count": len(content),
                        "generated_token_id": generated_token_ids[position],
                        "logprob_entry_token_id": exposed_id,
                    }
    return {
        "status": "available",
        "entry": entry,
        "position": position,
        "generated_token_id": generated_token_ids[position],
        "logprob_content_count": len(content),
        "generated_token_id_count": len(generated_token_ids),
    }


def api_logprobs(response: dict[str, Any]) -> Any:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    return choices[0].get("logprobs")


def verifier_row_logits(response: dict[str, Any], position: int) -> dict[str, Any]:
    """Only accept explicitly named verifier-logit fields, never API logprobs."""
    found: list[dict[str, Any]] = []

    def visit(value: Any, path: str = "$") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = key.lower()
                if normalized in {"verifier_row_logits", "raw_verifier_logits"}:
                    found.append({"path": f"{path}.{key}", "value": child})
                visit(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(response)
    for row in found:
        value = row["value"]
        if isinstance(value, list) and position < len(value):
            return {"status": "available", "source_path": row["path"], "position": value[position]}
        return {"status": "available", "source_path": row["path"], "value": value}
    return {
        "status": "unavailable",
        "reason": "chat-completions response did not expose an explicitly named raw verifier-row logits field",
    }


def comparison_record(
    target_response: dict[str, Any], eagle_response: dict[str, Any], prompt_id: str
) -> dict[str, Any]:
    target_ids = generated_ids(target_response)
    eagle_ids = generated_ids(eagle_response)
    if target_ids is None or eagle_ids is None:
        difference = None
    else:
        difference = first_difference(target_ids, eagle_ids)
    position = difference["zero_based_position"] if difference else None
    return {
        "prompt_id": prompt_id,
        "target_only_token_ids_available": target_ids is not None,
        "ordinary_eagle_token_ids_available": eagle_ids is not None,
        "target_only_token_count": len(target_ids) if target_ids is not None else None,
        "ordinary_eagle_token_count": len(eagle_ids) if eagle_ids is not None else None,
        "target_only_token_ids_sha256": sha256_bytes(json.dumps(target_ids, separators=(",", ":")).encode()) if target_ids is not None else None,
        "ordinary_eagle_token_ids_sha256": sha256_bytes(json.dumps(eagle_ids, separators=(",", ":")).encode()) if eagle_ids is not None else None,
        "exact_match": target_ids == eagle_ids if target_ids is not None and eagle_ids is not None else None,
        "first_mismatch": difference,
        "api_completion_logprobs_at_first_mismatch": {
            "target_only": api_logprob_entry(target_response, position, target_ids) if position is not None else {"status": "unavailable", "reason": "no mismatch position"},
            "ordinary_eagle": api_logprob_entry(eagle_response, position, eagle_ids) if position is not None else {"status": "unavailable", "reason": "no mismatch position"},
            "interpretation": "OpenAI-compatible selected completion-token logprobs; not raw full verifier-row logits",
        },
        "verifier_row_logits": {
            "target_only": verifier_row_logits(target_response, position) if position is not None else {"status": "unavailable", "reason": "no mismatch position"},
            "ordinary_eagle": verifier_row_logits(eagle_response, position) if position is not None else {"status": "unavailable", "reason": "no mismatch position"},
        },
        "historical_reference": {
            **EXPECTED_HISTORICAL_MISMATCH,
            "is_expected_diagnostic_context_only": True,
            "matches_observed_first_mismatch": (
                prompt_id == EXPECTED_HISTORICAL_MISMATCH["prompt_id"]
                and difference is not None
                and all(
                    difference.get(key) == EXPECTED_HISTORICAL_MISMATCH[key]
                    for key in ("zero_based_position", "target_only_token_id", "ordinary_eagle_token_id")
                )
            ),
        },
    }


def configure_request(config: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    body = BASE.request_body(config, prompt)
    body["logprobs"] = True
    body["top_logprobs"] = 5
    return body


def execute_variant(
    *, config: dict[str, Any], paths: dict[str, Path], variant: str,
    prompt: dict[str, Any], cell: Path,
) -> dict[str, Any]:
    cell.mkdir()
    command = BASE.command_for(config, paths, variant)
    server = config["server"]
    if not BASE.available_port(server["host"], server["port"]):
        raise RuntimeError(f"configured server port is occupied: {server['host']}:{server['port']}")
    log_path = cell / "server.log"
    environment = {key: os.environ[key] for key in BASE.SAFE_INHERITED_ENV if key in os.environ}
    environment.update(config.get("environment", {}))
    for selector in ("GGML_CUDA_W1A1_MMA", "GGML_CUDA_W8A8_MMA", "GGML_CUDA_W4A4_MMA", "GGML_W1AX_ACT_BITS", "GGML_W1AX_A4_KERNEL"):
        environment.pop(selector, None)
    trace_environment = verifier_trace_environment(cell)
    trace_path = Path(trace_environment["W1AX_VERIFY_TRACE_JSONL"])
    environment.update(trace_environment)
    body = configure_request(config, prompt)
    json_write(cell / "request-with-logprobs.json", body)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        base_url = f"http://{server['host']}:{server['port']}"
        result = None
        trace_report = None
        try:
            BASE.wait_ready(process, base_url, server["startup_timeout_s"])
            status, raw = BASE.request_json(
                base_url + "/v1/chat/completions", body, server["request_timeout_s"]
            )
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            logprobs_requested = True
            if status != 200 and ("logprobs" in raw.lower() or "top_logprobs" in raw.lower()):
                (cell / "logprobs-rejected-response.json").write_text(raw)
                body = {key: value for key, value in body.items() if key not in ("logprobs", "top_logprobs")}
                json_write(cell / "request.json", body)
                status, raw = BASE.request_json(
                    base_url + "/v1/chat/completions", body, server["request_timeout_s"]
                )
                logprobs_requested = False
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
            if logprobs_requested:
                json_write(cell / "request.json", body)
            (cell / "response.json").write_text(raw)
            if status != 200 or not isinstance(parsed, dict):
                raise RuntimeError(f"{variant} request failed with HTTP {status}; see {cell / 'response.json'}")
            result = {
                "variant": variant, "command": command,
                "request_sha256": sha256_file(cell / "request.json"),
                "response_sha256": sha256_file(cell / "response.json"),
                "server_log_sha256": sha256_file(log_path),
                "server_response_id": parsed.get("id"),
                "http_status": status,
                "logprobs_requested": logprobs_requested,
                "logprobs_available": api_logprobs(parsed) is not None,
                "generated_token_ids": generated_ids(parsed),
                "generated_token_ids_available": generated_ids(parsed) is not None,
                "completion_tokens_reported": (parsed.get("usage") or {}).get("completion_tokens"),
                "finish_reason": ((parsed.get("choices") or [{}])[0]).get("finish_reason"),
                "completion_content": ((parsed.get("choices") or [{}])[0].get("message") or {}).get("content"),
                "parsed_response": parsed,
            }
        finally:
            BASE.stop_server(process)
            trace_report = inspect_verifier_trace(trace_path)
            trace_report["environment"] = trace_environment
            json_write(cell / "verifier-trace-status.json", trace_report)
        if result is None:
            raise RuntimeError(f"{variant} produced no result")
        log.flush()
        result["server_log_sha256"] = sha256_file(log_path)
        result["verifier_trace"] = trace_report
        json_write(cell / "result.json", {key: value for key, value in result.items() if key != "parsed_response"})
        return result


def run(args: argparse.Namespace) -> Path:
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = tomllib.loads(config_path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    evaluation = config["evaluation"]
    required = {
        "max_output_tokens": 128,
        "temperature": 0.0,
        "max_draft_tokens": 5,
        "min_draft_probability": 0.0,
        "enable_thinking": False,
    }
    for key, expected in required.items():
        actual = evaluation.get(key)
        if actual != expected:
            raise ValueError(f"config evaluation.{key} must be {expected!r}, found {actual!r}")
    variants = BASE.selected_variants(evaluation)
    if "target_only" not in variants or "ordinary_eagle" not in variants:
        raise ValueError("config must use the eight-path W1Ax matrix to resolve its paired anchors")
    model_paths = dict(config["models"])
    paths = {
        name: BASE.resolve(ROOT, value)
        for name, value in {
            "binary": config["server"]["binary"],
            "target": model_paths["target"],
            "ordinary_draft": model_paths["ordinary_draft"],
        }.items()
    }
    prompt_path = BASE.resolve(ROOT, evaluation["prompt_file"])
    paths["prompt_file"] = prompt_path
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    prompt_rows = BASE.load_prompts(prompt_path)
    prompt = next((row for row in prompt_rows if row["id"] == args.prompt_id), None)
    if prompt is None:
        raise ValueError(f"prompt id {args.prompt_id!r} not found in {prompt_path}")
    if not BASE.available_port(config["server"]["host"], config["server"]["port"]):
        raise RuntimeError("configured llama-server port is already occupied")
    run_id = args.run_id or datetime.now(UTC).strftime("w1ax-target-divergence-%Y%m%dT%H%M%SZ")
    destination = args.results.resolve() / run_id
    destination.mkdir(parents=True, exist_ok=False)
    json_write(destination / "prompt.json", prompt)
    outcomes = {}
    for variant in ("target_only", "ordinary_eagle"):
        outcomes[variant] = execute_variant(
            config=config, paths=paths, variant=variant, prompt=prompt,
            cell=destination / variant,
        )
    comparison = comparison_record(
        outcomes["target_only"]["parsed_response"],
        outcomes["ordinary_eagle"]["parsed_response"], prompt["id"],
    )
    comparison["verifier_trace"] = {
        variant: outcomes[variant]["verifier_trace"]
        for variant in ("target_only", "ordinary_eagle")
    }
    json_write(destination / "comparison.json", comparison)
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "purpose": "historical greedy target-only versus ordinary EAGLE divergence replay; no timing claim",
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "prompt": {"id": prompt["id"], "path": str(prompt_path), "source_file_sha256": sha256_file(prompt_path), "selected_prompt_sha256": sha256_bytes(json.dumps(prompt, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())},
        "models": {
            "target": {"path": str(paths["target"]), "sha256": sha256_file(paths["target"]), "precision": "FP16"},
            "ordinary_draft": {"path": str(paths["ordinary_draft"]), "sha256": sha256_file(paths["ordinary_draft"])},
            "server_binary": {"path": str(paths["binary"]), "sha256": sha256_file(paths["binary"])},
        },
        "settings": {"max_output_tokens": 128, "temperature": 0.0, "seed": evaluation["seed"], "thinking": False, "speculative_draft_length": 5, "min_draft_probability": 0.0, "variants_sequential": ["target_only", "ordinary_eagle"]},
        "raw": {"target_only": "target_only/response.json", "ordinary_eagle": "ordinary_eagle/response.json", "comparison": "comparison.json"},
        "comparison": comparison,
        "verifier_trace": {
            "expected_schema": "w1ax_verify_logits_v1",
            "raw_jsonl_by_variant": {
                "target_only": "target_only/verifier-trace.jsonl",
                "ordinary_eagle": "ordinary_eagle/verifier-trace.jsonl",
            },
            "status_by_variant": comparison["verifier_trace"],
        },
        "api_response_verifier_row_logits": {
            "status": "unavailable unless the endpoint supplies an explicitly named raw verifier-logits field",
            "api_logprobs_are_not_raw_verifier_logits": True,
        },
        "timing_claim": None,
    }
    json_write(destination / "manifest.json", manifest)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-id")
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--prompt-id", default=DEFAULT_PROMPT_ID)
    args = parser.parse_args()
    destination = run(args)
    print(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())

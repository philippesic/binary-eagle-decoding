#!/usr/bin/env python3
"""Compare a pinned CPU native first-seed FFN graph with candidate-D stages.

The archived reasoning capture is a byte-exact control for the new graph.
This one-column operator diagnostic does not certify recurrent behavior,
choose an error tolerance, run an optimizer, or use a GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit_recurrent_binary_capture import read_jsonl, sha256  # noqa: E402
from capture_w1ax_activations import generated_token_ids  # noqa: E402
from check_recurrent_ffn_from_graph import (  # noqa: E402
    EXPECTED_COLUMN,
    EXPECTED_EXECUTION,
    FFN_BASES,
    INTERMEDIATE,
    WIDTH,
    _array_sha256,
    _cpu_model,
    _metrics,
    join_first_seed,
)
from compare_recurrent_draft_graph import _column, _read_graph  # noqa: E402
from export_binary_rescue import Model  # noqa: E402
from load_recurrent_binary_init import D_SHA256, load_candidate_d_arrays  # noqa: E402
from run_binary_head_capture import TARGET_F16_SHA256, TRAIN_PROMPTS_SHA256  # noqa: E402

from w1a1_eagle.recurrent_binary import GroupedBinaryLinear  # noqa: E402

STAGE_TAPS = {
    "gate": "ffn_gate-0",
    "up": "ffn_up-0",
    "silu_gate": "ffn_gate_silu-0",
    "mul": "ffn_mul-0",
    "down": "ffn_out-0",
}
STAGE_ORDER = tuple(STAGE_TAPS)
BASELINE_MANIFEST_SHA256 = "99b9003698bcdf5505421a1667dac4d21a8545471d91330c04181c9f2a931d40"
CONTROL_FILES = (
    "heads.draft_graph.jsonl",
    "heads.draft_graph.f32",
    "heads.jsonl",
    "heads.f32",
    "prompt.json",
    "request.json",
    "response.json",
)


def verify_capture(capture_dir: Path, *, require_stages: bool) -> tuple[dict, dict]:
    """Validate every sealed file's exact size/hash and the CPU capture identity."""
    capture_dir = Path(capture_dir)
    manifest_path = capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("capture manifest lacks a file ledger")
    if (
        manifest.get("schema") != "recurrent_cpu_native_diagnostic_v1"
        or manifest.get("execution_device") != "cpu"
        or manifest.get("training_eligible") is not False
        or manifest.get("prompt_id") != "qat-revisit-train-reasoning-rate-and-work-01"
    ):
        raise ValueError("capture is not the frozen CPU reasoning diagnostic")
    sources = manifest.get("source_sha256")
    if (
        not isinstance(sources, dict)
        or sources.get("target") != TARGET_F16_SHA256
        or sources.get("draft") != D_SHA256
        or sources.get("frozen_train_prompts") != TRAIN_PROMPTS_SHA256
    ):
        raise ValueError("capture target, draft or training split is not pinned")
    ledger = manifest["files"]
    if not set(CONTROL_FILES).issubset(ledger):
        raise ValueError("capture manifest lacks required graph/head/request files")
    if require_stages and not {"heads.draft_graph.jsonl", "heads.draft_graph.f32"}.issubset(ledger):
        raise ValueError("stage graph is absent from capture manifest")
    actual_files = {path.name for path in capture_dir.iterdir() if path.is_file()} - {
        "manifest.json"
    }
    if actual_files != set(ledger):
        raise ValueError("capture file set differs from sealed manifest")
    for name, entry in ledger.items():
        path = capture_dir / name
        if (
            Path(name).name != name
            or not isinstance(entry, dict)
            or type(entry.get("bytes")) is not int
            or entry["bytes"] < 0
            or not isinstance(entry.get("sha256"), str)
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"capture file size or SHA256 disagrees with manifest: {name}")
    response = json.loads((capture_dir / "response.json").read_text())
    ids = generated_token_ids(response)
    if (
        not isinstance(ids, list)
        or len(ids) != 8
        or any(type(token) is not int or token < 0 for token in ids)
        or manifest.get("generated_token_ids") != ids
    ):
        raise ValueError("raw eight-token response differs from capture manifest")
    return manifest, {"manifest": sha256(manifest_path), "files": ledger, "raw_ids": ids}


def read_first_seed(capture_dir: Path, candidate_d: Path, *, require_stages: bool) -> dict:
    """Uniquely join execution 2/column 0 and inspect its F32 FFN tensors."""
    capture_dir = Path(capture_dir)
    records, values, footer = _read_graph(
        capture_dir / "heads.draft_graph.jsonl", capture_dir / "heads.draft_graph.f32"
    )
    heads = read_jsonl(capture_dir / "heads.jsonl")
    first = [row for row in heads if row.get("round_index") == 0 and row.get("depth") == 0]
    if (
        len(first) != 1
        or first[0].get("schema") != "eagle_head_state_v1"
        or first[0].get("state_dim") != WIDTH
        or first[0].get("state_boundary") != "native_output_norm_f32_before_head_operand_conversion"
        or type(first[0].get("state_row")) is not int
        or not 0 <= first[0]["state_row"] < len(heads)
    ):
        raise ValueError("capture lacks one valid first-round seed head state")
    states_path = capture_dir / "heads.f32"
    if states_path.stat().st_size != len(heads) * WIDTH * 4:
        raise ValueError("head states have the wrong exact F32 byte count")
    states = np.memmap(states_path, dtype="<f4", mode="r").reshape(-1, WIDTH)
    norm = np.asarray(Model(candidate_d).tensors["output_norm.weight"].data, dtype=np.float32)
    native_input, native_output, join = join_first_seed(
        records, values, states[first[0]["state_row"]], norm
    )
    selected = [
        row
        for row in records
        if row.get("group_kind") == "decoder" and row.get("group_execution") == EXPECTED_EXECUTION
    ]
    stages = {}
    if require_stages:
        stage_records = {row["tensor_name"]: row for row in selected}
        if not set(STAGE_TAPS.values()).issubset(stage_records):
            raise ValueError("joined graph lacks all four FFN stage taps and final down")
        token_count = stage_records["ffn_out-0"]["n_tokens"]
        for stage, name in STAGE_TAPS.items():
            row = stage_records[name]
            width = WIDTH if stage == "down" else INTERMEDIATE
            if (
                row.get("dtype") != "f32"
                or row.get("n_tokens") != token_count
                or row.get("token_width") != width
                or row.get("ne") != [width, token_count]
                or row.get("token_axis") != 1
            ):
                raise ValueError(f"native {name} stage has wrong F32 graph geometry")
            vector = _column(row, values, EXPECTED_COLUMN)
            if vector.shape != (width,) or not np.isfinite(vector).all():
                raise ValueError(f"native {name} stage is not one finite F32 vector")
            stages[stage] = np.array(vector, copy=True)
        if not np.array_equal(stages["down"], native_output):
            raise ValueError("native final FFN tap differs from joined graph output")
    return {
        "input": np.array(native_input, copy=True),
        "output": np.array(native_output, copy=True),
        "stages": stages,
        "head_state": np.array(states[first[0]["state_row"]], copy=True),
        "join": join,
        "footer": footer,
    }


def verify_ancestry(baseline: dict, candidate: dict, old: dict, new: dict) -> dict:
    """Reject a changed prompt, response prefix, seed state or FFN boundary."""
    old_files, new_files = old["files"], new["files"]
    for name in ("prompt.json", "request.json"):
        if old_files[name]["sha256"] != new_files[name]["sha256"]:
            raise ValueError(f"new graph changed the frozen {name} bytes")
    if old["raw_ids"] != new["raw_ids"]:
        raise ValueError("new graph changed the first eight raw output IDs")
    for key in ("head_state", "input", "output"):
        if baseline[key].tobytes() != candidate[key].tobytes():
            raise ValueError(f"new graph changed first-seed {key} F32 bytes")
    return {
        "status": "first_seed_boundary_and_eight_raw_ids_bitwise_identical",
        "prompt_sha256": old_files["prompt.json"]["sha256"],
        "request_sha256": old_files["request.json"]["sha256"],
        "first_eight_raw_ids": new["raw_ids"],
        "head_state_sha256": _array_sha256(candidate["head_state"]),
        "ffn_input_sha256": _array_sha256(candidate["input"]),
        "ffn_output_sha256": _array_sha256(candidate["output"]),
    }


def verify_baseline_identity(seal: dict) -> None:
    if seal["manifest"] != BASELINE_MANIFEST_SHA256:
        raise ValueError("baseline is not the archived frozen reasoning manifest")


def replay_stages(
    native_input: np.ndarray,
    arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    arithmetic: str,
    *,
    expected_intermediate: int = INTERMEDIATE,
) -> dict[str, np.ndarray]:
    """Replay gate, up, Torch SiLU, product and down from one native input."""
    native_input = np.asarray(native_input)
    if (
        native_input.ndim != 1
        or native_input.dtype != np.float32
        or not np.isfinite(native_input).all()
    ):
        raise ValueError("FFN input must be one finite F32 vector")
    if set(arrays) != set(FFN_BASES) or arithmetic not in {"native_order", "group_matmul"}:
        raise ValueError("FFN requires gate/up/down arrays and known arithmetic")
    gate_packed, gate_scales = arrays[FFN_BASES[0]]
    up_packed, up_scales = arrays[FFN_BASES[1]]
    down_packed, down_scales = arrays[FFN_BASES[2]]
    if (
        gate_packed.shape[0] != expected_intermediate
        or up_packed.shape[0] != expected_intermediate
        or down_packed.shape[0] != native_input.size
        or down_packed.shape[1] * 32 != expected_intermediate
    ):
        raise ValueError("FFN packed arrays have the wrong gate/up/down geometry")
    linears = {
        "gate": GroupedBinaryLinear.from_packed(
            gate_packed, gate_scales, in_features=native_input.size, arithmetic=arithmetic
        ),
        "up": GroupedBinaryLinear.from_packed(
            up_packed, up_scales, in_features=native_input.size, arithmetic=arithmetic
        ),
        "down": GroupedBinaryLinear.from_packed(
            down_packed, down_scales, in_features=expected_intermediate, arithmetic=arithmetic
        ),
    }
    with torch.no_grad():
        x = torch.from_numpy(np.array(native_input, copy=True))
        gate = linears["gate"](x)
        up = linears["up"](x)
        silu_gate = F.silu(gate)
        mul = up * silu_gate
        down = linears["down"](mul)
    tensors = {"gate": gate, "up": up, "silu_gate": silu_gate, "mul": mul, "down": down}
    if any(
        value.dtype != torch.float32 or not torch.isfinite(value).all()
        for value in tensors.values()
    ):
        raise ValueError("FFN replay produced a nonfinite or non-F32 stage")
    return {name: value.numpy().copy() for name, value in tensors.items()}


def compare_stages(native: dict[str, np.ndarray], replay: dict[str, np.ndarray]) -> dict:
    if set(native) != set(STAGE_ORDER) or set(replay) != set(STAGE_ORDER):
        raise ValueError("comparison requires all five ordered FFN stages")
    metrics = {stage: _metrics(replay[stage], native[stage]) for stage in STAGE_ORDER}
    return {
        "first_divergent_stage": next(
            (
                stage
                for stage in STAGE_ORDER
                if metrics[stage]["exact_elements"] != metrics[stage]["elements"]
            ),
            None,
        ),
        "versus_native": metrics,
        "native_stage_sha256": {stage: _array_sha256(native[stage]) for stage in STAGE_ORDER},
        "replay_stage_sha256": {stage: _array_sha256(replay[stage]) for stage in STAGE_ORDER},
    }


def audit(
    capture_dir: Path,
    baseline_dir: Path,
    target_gguf: Path,
    candidate_d: Path,
    native_source: Path,
    native_binary: Path,
    cmake_cache: Path,
    train_prompts: Path,
) -> dict:
    paths = [target_gguf, candidate_d, native_source, native_binary, cmake_cache, train_prompts]
    if any(not Path(path).is_file() for path in paths):
        raise ValueError("target, D, native source/binary/build and train split files are required")
    if Path(native_binary).resolve() != (Path(cmake_cache).parent / "bin/llama-server").resolve():
        raise ValueError("native binary does not belong to supplied CMake build")
    if sha256(target_gguf) != TARGET_F16_SHA256 or sha256(train_prompts) != TRAIN_PROMPTS_SHA256:
        raise ValueError("target GGUF or frozen training split differs from pinned source")
    arrays, d_audit = load_candidate_d_arrays(candidate_d)
    if d_audit["gguf_sha256"] != D_SHA256:
        raise ValueError("candidate D differs from pinned GGUF")
    old_manifest, old = verify_capture(baseline_dir, require_stages=False)
    verify_baseline_identity(old)
    new_manifest, new = verify_capture(capture_dir, require_stages=True)
    if new_manifest["source_sha256"].get("binary") != sha256(native_binary) or new_manifest[
        "source_sha256"
    ].get("cmake_cache") != sha256(cmake_cache):
        raise ValueError("new capture does not identify supplied native binary/build")
    baseline = read_first_seed(baseline_dir, candidate_d, require_stages=False)
    candidate = read_first_seed(capture_dir, candidate_d, require_stages=True)
    ancestry = verify_ancestry(baseline, candidate, old, new)
    ffn_arrays = {base: arrays[base] for base in FFN_BASES}
    results = {}
    for arithmetic in ("native_order", "group_matmul"):
        replay = replay_stages(candidate["input"], ffn_arrays, arithmetic)
        results[arithmetic] = compare_stages(candidate["stages"], replay)
    return {
        "schema": "recurrent_binary_ffn_stage_parity_cpu_v1",
        "execution_device": "cpu",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
        },
        "precision": "A16 cast before each W1 group; F32 sums, scales, Torch SiLU and product",
        "scope": (
            "one unchanged first-seed reasoning column; no tolerance, training, "
            "GPU or target forward"
        ),
        "join": candidate["join"],
        "ancestry": ancestry,
        "native_graph_footer": candidate["footer"],
        "results": results,
        "source_sha256": {
            "target_gguf": TARGET_F16_SHA256,
            "candidate_d": D_SHA256,
            "frozen_train_prompts": TRAIN_PROMPTS_SHA256,
            "native_eagle3_cpp": sha256(native_source),
            "native_binary": sha256(native_binary),
            "cmake_cache": sha256(cmake_cache),
            "stage_audit_source": sha256(Path(__file__)),
            "ffn_replay_source": sha256(
                Path(__file__).with_name("check_recurrent_ffn_from_graph.py")
            ),
            "binary_linear_source": sha256(
                Path(__file__).resolve().parents[1] / "src/w1a1_eagle/recurrent_binary.py"
            ),
            "baseline_manifest": old["manifest"],
            "new_capture_manifest": new["manifest"],
            "baseline_graph_index": old["files"]["heads.draft_graph.jsonl"]["sha256"],
            "baseline_graph_values": old["files"]["heads.draft_graph.f32"]["sha256"],
            "new_graph_index": new["files"]["heads.draft_graph.jsonl"]["sha256"],
            "new_graph_values": new["files"]["heads.draft_graph.f32"]["sha256"],
        },
        "capture_file_ledgers": {
            "baseline": old["files"],
            "new": new["files"],
        },
        "native_source_bytes": {
            "eagle3_cpp": Path(native_source).stat().st_size,
            "binary": Path(native_binary).stat().st_size,
            "cmake_cache": Path(cmake_cache).stat().st_size,
        },
        "native_revisions": {
            "baseline": old_manifest.get("native_revision"),
            "new": new_manifest.get("native_revision"),
        },
        "software": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_num_threads": torch.get_num_threads(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--target-gguf", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--native-binary", type=Path, required=True)
    parser.add_argument("--cmake-cache", type=Path, required=True)
    parser.add_argument("--train-prompts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report must be a new file")
    torch.set_num_threads(1)
    report = audit(
        args.capture_dir,
        args.baseline_dir,
        args.target_gguf,
        args.candidate_d,
        args.native_source,
        args.native_binary,
        args.cmake_cache,
        args.train_prompts,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {mode: report["results"][mode]["first_divergent_stage"] for mode in report["results"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

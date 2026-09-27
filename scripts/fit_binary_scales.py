#!/usr/bin/env python3
"""Fit the frozen four-way fixed-sign A16 binary scale screen and export GGUF.

The capture manifest lists exactly 96 ``prompts``, each with ``id``,
``capture_dir`` and optionally ``files`` (capture basenames). Inputs must already
be post-F16-cast values in W1AXACT1. No prompt text or final-set data is read.
Fitting uses uncentered features, original BF16 weights promoted to F32, and
F32 BLAS (TF32 disabled). This fitting arithmetic is a least-squares surrogate;
the separately validated native evaluator uses its explicit sequential F32 sums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/llama.cpp/gguf-py"))

from audit_eagle_w1a1_gguf import SOURCE_NAMES, gguf_qk_row_order, sha256  # noqa: E402
from select_w1ax_captures import HEADER, parse_capture, stratified_indices  # noqa: E402

PROTOCOL = {
    "training_prompts": 96,
    "rows_per_prompt_layer": 32,
    "selection": "ordered bucket centers over all invocation rows per prompt/layer",
    "output_row_batch": 32,
    "group_size": 128,
    "mean_absolute_reduction": "F32 CPU torch mean; legacy 128-output-row chunks",
    "ridge_relative": 1e-4,
    "ridge_definition": "lambda=1e-4*mean(diag(Z.T@Z/N)); penalty=lambda*||s-s0||^2",
    "max_iterations": 512,
    "projected_gradient_relative_tolerance": 1e-6,
    "solver": "monotone restarted accelerated projected gradient, nonnegative orthant",
    "initialization": "mean-absolute feasible baseline s0",
    "fallback": "per-row mean-absolute baseline if direct F32-BLAS residual SSE increases",
    "arithmetic": (
        "F32 BLAS with TF32 disabled; F64 residual sum; native arithmetic audited separately"
    ),
}


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def canonical_weight(checkpoint, base: str) -> np.ndarray:
    source = checkpoint.get_tensor(SOURCE_NAMES[base])
    if source.dtype != torch.bfloat16:
        raise ValueError(f"{base}: pinned original source must be BF16")
    weight = source.float().numpy()
    if base == "blk.0.attn_q":
        weight = gguf_qk_row_order(weight, 32)
    elif base == "blk.0.attn_k":
        weight = gguf_qk_row_order(weight, 8)
    if weight.ndim != 2 or not np.isfinite(weight).all():
        raise ValueError(f"{base}: source must be a finite matrix")
    return np.ascontiguousarray(weight)


def pack_signs(weight: np.ndarray) -> np.ndarray:
    signs = weight >= 0
    if weight.shape[1] % 32:
        signs = np.pad(signs, ((0, 0), (0, 32 - weight.shape[1] % 32)))
    return np.packbits(signs, axis=1, bitorder="little").view("<i4")


def baseline_scales(weight: np.ndarray, group_size: int) -> np.ndarray:
    # Match conversion/llama.py's legacy F32 CPU torch reduction, including its
    # 128-output-row chunking. NumPy mean can differ by a few F32 ULPs.
    size = group_size or weight.shape[1]
    groups = (weight.shape[1] + size - 1) // size
    result = np.empty((len(weight), groups), dtype=np.float32)
    for first in range(0, len(weight), 128):
        chunk = torch.from_numpy(weight[first : first + 128])
        for group, start in enumerate(range(0, weight.shape[1], size)):
            result[first : first + len(chunk), group] = (
                chunk[:, start : start + size].abs().mean(1).numpy()
            )
    return result[:, 0].copy() if group_size == 0 else result


def select_prompt_rows(prompt: dict) -> tuple[dict[str, np.ndarray], list[dict]]:
    directory = Path(prompt["capture_dir"])
    if "files" in prompt:
        if len(prompt["files"]) != len(set(prompt["files"])):
            raise ValueError(f"{prompt['id']}: duplicate capture files")
        if any(Path(name).name != name for name in prompt["files"]):
            raise ValueError("capture files must be basenames")
        paths = [directory / name for name in prompt["files"]]
    else:
        paths = sorted(directory.glob("op-*.bin"))
    grouped = defaultdict(list)
    for path in paths:
        capture = parse_capture(path)
        if capture["bits"] != 16 or not capture["name"].endswith(".weight"):
            raise ValueError(f"expected native dense A16 capture: {path}")
        base = capture["name"].removesuffix(".weight")
        if base not in SOURCE_NAMES:
            raise ValueError(f"unexpected captured tensor {base}")
        grouped[base].append(capture)
    if set(grouped) != set(SOURCE_NAMES):
        raise ValueError(f"{prompt['id']}: missing layers {set(SOURCE_NAMES) - set(grouped)}")
    arrays, records = {}, []
    for base, captures in sorted(grouped.items()):
        captures.sort(key=lambda capture: capture["sequence"])
        if len({capture["sequence"] for capture in captures}) != len(captures):
            raise ValueError(f"{prompt['id']}/{base}: duplicate sequence")
        if len({(capture["K"], capture["M"]) for capture in captures}) != 1:
            raise ValueError(f"{base}: inconsistent capture shape")
        count = sum(capture["N"] for capture in captures)
        chosen = stratified_indices(count, PROTOCOL["rows_per_prompt_layer"])
        offset, selected = 0, []
        for capture in captures:
            indices = [
                index - offset for index in chosen if offset <= index < offset + capture["N"]
            ]
            offset += capture["N"]
            if not indices:
                continue
            data = np.memmap(
                capture["path"],
                mode="r",
                dtype="<f4",
                offset=HEADER.size,
                shape=(capture["N"], capture["K"]),
            )
            rows = np.array(data[indices], dtype=np.float32)
            if not np.isfinite(rows).all():
                raise ValueError(f"{capture['path']}: nonfinite captured inputs")
            if not np.array_equal(rows, rows.astype(np.float16).astype(np.float32)):
                raise ValueError(f"{capture['path']}: capture is not actual post-F16-cast input")
            selected.append(rows)
            records.append(
                {
                    "prompt_id": prompt["id"],
                    "layer": base,
                    "path": str(capture["path"]),
                    "sha256": sha256(capture["path"]),
                    "rows": indices,
                    "sequence": capture["sequence"],
                    "N": capture["N"],
                    "K": capture["K"],
                    "M": capture["M"],
                }
            )
        arrays[base] = np.concatenate(selected)
    return arrays, records


def load_inputs(manifest_path: Path) -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads(manifest_path.read_text())
    prompts = manifest["prompts"]
    if len(prompts) != PROTOCOL["training_prompts"]:
        raise ValueError("expected exactly 96 training prompts")
    if len({prompt["id"] for prompt in prompts}) != len(prompts):
        raise ValueError("duplicate training prompt IDs")
    # Reject overlapping invocation ownership when one server writes a shared directory.
    owned = set()
    arrays, records = defaultdict(list), []
    for prompt in prompts:
        selected, prompt_records = select_prompt_rows(prompt)
        for record in prompt_records:
            key = str(Path(record["path"]).resolve())
            if key in owned:
                raise ValueError(f"capture is shared by multiple prompts: {key}")
            owned.add(key)
        for base, values in selected.items():
            arrays[base].append(values)
        records.extend(prompt_records)
    stacked = {base: np.concatenate(values) for base, values in arrays.items()}
    return stacked, {
        "manifest_sha256": sha256(manifest_path),
        "selected": records,
        "inputs": {
            base: {"shape": list(x.shape), "sha256": array_hash(x)} for base, x in stacked.items()
        },
    }


def projected_ridge_nnls(z: torch.Tensor, y: torch.Tensor, baseline: torch.Tensor):
    """Solve batched min ||Zs-y||²/N + lambda||s-s0||² with s>=0.

    Z has shape [output rows, examples, scale groups]. Every iterate is feasible;
    projection is part of the iterative constrained solve, not clipping a dense
    unconstrained solution. The finite fixed budget can yield approximate fits.
    """
    count = z.shape[1]
    gram = z.transpose(1, 2).bmm(z) / count
    rhs = z.transpose(1, 2).bmm(y.unsqueeze(2)).squeeze(2) / count
    ridge = PROTOCOL["ridge_relative"] * gram.diagonal(dim1=1, dim2=2).mean(1)
    hessian = gram + torch.diag_embed(ridge[:, None].expand_as(baseline))
    linear = rhs + ridge[:, None] * baseline
    lipschitz = hessian.abs().sum(2).amax(1).clamp_min(torch.finfo(z.dtype).tiny)

    def multiply(value):
        return hessian.bmm(value.unsqueeze(2)).squeeze(2)

    def objective_change(proposed, current):
        # Evaluate the quadratic *difference* directly: subtracting two large
        # F32 objectives otherwise stalls even a two-variable solve near optimum.
        delta = proposed - current
        return (delta * (multiply(current) - linear + 0.5 * multiply(delta))).double().sum(1)

    scale = baseline.clone()
    extrapolated = scale.clone()
    momentum = torch.ones_like(lipschitz)
    iterations = 0
    for step in range(PROTOCOL["max_iterations"]):
        proposed = (
            extrapolated - (multiply(extrapolated) - linear) / lipschitz[:, None]
        ).clamp_min(0)
        restart = objective_change(proposed, scale) > 0
        safe_step = (scale - (multiply(scale) - linear) / lipschitz[:, None]).clamp_min(0)
        proposed = torch.where(restart[:, None], safe_step, proposed)
        keep = objective_change(proposed, scale) <= 0
        proposed = torch.where(keep[:, None], proposed, scale)
        next_momentum = (1 + torch.sqrt(1 + 4 * momentum.square())) / 2
        next_momentum = torch.where(restart | ~keep, torch.ones_like(momentum), next_momentum)
        extrapolated = proposed + ((momentum - 1) / next_momentum)[:, None] * (proposed - scale)
        extrapolated = torch.where(restart[:, None], proposed, extrapolated)
        scale, momentum = proposed, next_momentum
        iterations = step + 1
        if iterations % 32 == 0:
            residual = (
                (scale - (scale - (multiply(scale) - linear) / lipschitz[:, None]).clamp_min(0))
                .abs()
                .amax(1)
            )
            relative = residual / scale.abs().amax(1).clamp_min(1e-12)
            if bool((relative <= PROTOCOL["projected_gradient_relative_tolerance"]).all()):
                break

    initial_error = (z.bmm(baseline.unsqueeze(2)).squeeze(2) - y).double().square().sum(1)
    fitted_error = (z.bmm(scale.unsqueeze(2)).squeeze(2) - y).double().square().sum(1)
    fallback = (
        (~torch.isfinite(scale).all(1))
        | (~torch.isfinite(fitted_error))
        | (fitted_error > initial_error)
    )
    scale = torch.where(fallback[:, None], baseline, scale)
    gradient = multiply(scale) - linear
    projected = (scale - (scale - gradient / lipschitz[:, None]).clamp_min(0)).abs().amax(1)
    relative = projected / scale.abs().amax(1).clamp_min(1e-12)
    kkt = torch.where(scale > 0, gradient.abs(), (-gradient).clamp_min(0)).amax(1)
    return scale, {
        "iterations": iterations,
        "fallback": fallback,
        "ridge": ridge,
        "projected_gradient_relative": relative,
        "kkt_max_abs": kkt,
        "converged": relative <= PROTOCOL["projected_gradient_relative_tolerance"],
    }


def group_features(x: torch.Tensor, signs: torch.Tensor, size: int) -> torch.Tensor:
    """F32 signed features [output rows, examples, groups], no centering."""
    width = x.shape[1]
    padding = (-width) % size
    if padding:
        x = torch.nn.functional.pad(x, (0, padding))
        signs = torch.nn.functional.pad(signs, (0, padding))
    groups = x.shape[1] // size
    inputs = x.reshape(x.shape[0], groups, size).permute(1, 0, 2)
    weights = signs.reshape(signs.shape[0], groups, size).permute(1, 2, 0)
    return inputs.bmm(weights).permute(2, 1, 0).contiguous()


def fit_layer(weight: np.ndarray, inputs: np.ndarray, device: str) -> tuple[dict, dict, dict]:
    if inputs.shape[1] != weight.shape[1] or not np.isfinite(inputs).all():
        raise ValueError("activation width mismatch or nonfinite inputs")
    if not np.array_equal(inputs, inputs.astype(np.float16).astype(np.float32)):
        raise ValueError("inputs must be actual F16 values promoted to F32")
    x = torch.from_numpy(inputs).to(device=device, dtype=torch.float32)
    scales = {"A": baseline_scales(weight, 0), "B": baseline_scales(weight, 128)}
    scales["C"], scales["D"] = np.empty_like(scales["A"]), np.empty_like(scales["B"])
    errors = {name: [] for name in "ABCD"}
    diagnostics = defaultdict(list)
    target_energy = 0.0
    for first in range(0, len(weight), PROTOCOL["output_row_batch"]):
        last = min(first + PROTOCOL["output_row_batch"], len(weight))
        w = torch.from_numpy(weight[first:last]).to(device)
        signs = torch.where(w >= 0, 1.0, -1.0)
        y = (x @ w.T).T.contiguous()
        target_energy += float(y.double().square().sum())
        row_features = (x @ signs.T).T.contiguous().unsqueeze(2)
        grouped = group_features(x, signs, PROTOCOL["group_size"])
        for initial, fitted, features in (("A", "C", row_features), ("B", "D", grouped)):
            base = (
                torch.from_numpy(scales[initial][first:last]).to(device).reshape(last - first, -1)
            )
            if initial == "A":
                # Exact scalar nonnegative ridge minimizer, including zero-feature rows.
                diag = features.squeeze(2).square().mean(1)
                rhs = (features.squeeze(2) * y).mean(1)
                ridge = PROTOCOL["ridge_relative"] * diag
                denominator = diag + ridge
                fit = torch.where(
                    denominator > 0,
                    (rhs + ridge * base[:, 0])
                    / denominator.clamp_min(torch.finfo(torch.float32).tiny),
                    base[:, 0],
                )
                fit = fit.clamp_min(0)[:, None]
                grad = denominator * fit[:, 0] - (rhs + ridge * base[:, 0])
                kkt = torch.where(fit[:, 0] > 0, grad.abs(), (-grad).clamp_min(0))
                residual = kkt / denominator.clamp_min(torch.finfo(torch.float32).tiny)
                relative = residual / fit[:, 0].abs().clamp_min(1e-12)
                diag_report = {
                    "iterations": 1,
                    "ridge": ridge,
                    "kkt_max_abs": kkt,
                    "projected_gradient_relative": relative,
                    "converged": relative <= PROTOCOL["projected_gradient_relative_tolerance"],
                }
            else:
                fit, diag_report = projected_ridge_nnls(features, y, base)
            initial_error = (
                (features.bmm(base.unsqueeze(2)).squeeze(2) - y).double().square().sum(1)
            )
            fitted_error = (features.bmm(fit.unsqueeze(2)).squeeze(2) - y).double().square().sum(1)
            fallback = ~torch.isfinite(fitted_error) | (fitted_error > initial_error)
            fallback |= diag_report.get("fallback", torch.zeros_like(fallback))
            fit = torch.where(fallback[:, None], base, fit)
            fitted_error = torch.where(fallback, initial_error, fitted_error)
            diag_report["fallback"] = fallback
            if initial == "A":
                # Residuals describe the exported point, including any SSE fallback.
                grad = denominator * fit[:, 0] - (rhs + ridge * base[:, 0])
                kkt = torch.where(fit[:, 0] > 0, grad.abs(), (-grad).clamp_min(0))
                residual = kkt / denominator.clamp_min(torch.finfo(torch.float32).tiny)
                relative = residual / fit[:, 0].abs().clamp_min(1e-12)
                diag_report.update(
                    {
                        "kkt_max_abs": kkt,
                        "projected_gradient_relative": relative,
                        "converged": relative <= PROTOCOL["projected_gradient_relative_tolerance"],
                    }
                )
            errors[initial].extend(initial_error.cpu().tolist())
            errors[fitted].extend(fitted_error.cpu().tolist())
            scales[fitted][first:last] = fit.cpu().numpy().reshape(scales[fitted][first:last].shape)
            for key, value in diag_report.items():
                diagnostics[f"{fitted}_{key}"].extend(
                    value.cpu().tolist()
                    if isinstance(value, torch.Tensor)
                    else [value] * (last - first)
                )
        print(
            json.dumps({"event": "fit_batch", "rows_done": last, "rows": len(weight)}), flush=True
        )
    detail = {f"{key}_sse": np.asarray(value) for key, value in errors.items()}
    detail.update({key: np.asarray(value) for key, value in diagnostics.items()})
    summary = {
        "rows": len(weight),
        "K": weight.shape[1],
        "examples": len(inputs),
        "target_energy": target_energy,
        "variants": {},
    }
    for name, values in scales.items():
        entry = {
            "sse": float(np.sum(errors[name])),
            "scale_sha256": array_hash(values),
            "scale_min": float(values.min()),
            "scale_max": float(values.max()),
            "zero_scales": int(np.count_nonzero(values == 0)),
        }
        entry["relative_squared_error"] = entry["sse"] / max(target_energy, 1e-300)
        if name in "CD":
            baseline = scales["A" if name == "C" else "B"]
            entry.update(
                {
                    "scale_delta_mean_abs": float(np.abs(values - baseline).mean()),
                    "scale_delta_max_abs": float(np.abs(values - baseline).max()),
                    "fallback_rows": int(detail[f"{name}_fallback"].sum()),
                    "converged_rows": int(detail[f"{name}_converged"].sum()),
                    "max_projected_gradient_relative": float(
                        detail[f"{name}_projected_gradient_relative"].max()
                    ),
                }
            )
        summary["variants"][name] = entry
    return scales, summary, detail


def export_variant(f16_path: Path, destination: Path, arrays: dict, variant: str) -> dict:
    from gguf import GGMLQuantizationType, GGUFReader, GGUFWriter

    if variant not in ("A", "B", "C", "D"):
        raise ValueError("unknown scale variant")
    for base, data in arrays.items():
        if data[variant].dtype != np.float32:
            raise ValueError(f"{base}: scales must be F32")
        if not np.isfinite(data[variant]).all() or (data[variant] < 0).any():
            raise ValueError(f"{base}: scales must be finite and nonnegative")
    reader = GGUFReader(f16_path)
    writer = GGUFWriter(destination, reader.fields["general.architecture"].contents())
    for key, field in reader.fields.items():
        if (
            key.startswith("GGUF.")
            or key.startswith("eagle3.w1a1")
            or key == "general.architecture"
        ):
            continue
        writer.add_key_value(
            key, field.contents(), field.types[0], field.types[-1] if len(field.types) > 1 else None
        )
    prefix = "eagle3.w1a1"
    writer.add_uint32(f"{prefix}.version", 2)
    writer.add_uint32(f"{prefix}.scale_group_size", 128 if variant in "BD" else 0)
    writer.add_array(f"{prefix}.groups", ["fusion", "attention", "ffn", "head"])
    writer.add_array(f"{prefix}.tensors", sorted(f"{base}.weight" for base in SOURCE_NAMES))
    writer.add_string(f"{prefix}.bit_order", "little")
    writer.add_string(f"{prefix}.sign_rule", "nonnegative_is_one")
    writer.add_string(
        f"{prefix}.scale_rule",
        "f32_mean_abs" if variant in "AB" else "f32_nonnegative_least_squares",
    )
    writer.add_string(f"{prefix}.arithmetic", "f32")
    selected, preserved = [], []
    for tensor in reader.tensors:
        base = tensor.name.removesuffix(".weight")
        if tensor.name.endswith(".weight") and base in SOURCE_NAMES:
            data = arrays[base]
            key = tensor.name.replace(".", "_")
            writer.add_uint32(f"{prefix}.tensor.{key}.logical_k", int(data["K"]))
            writer.add_string(f"{prefix}.tensor.{key}.packed", f"{base}.w1a1_packed")
            writer.add_string(f"{prefix}.tensor.{key}.scale", f"{base}.w1a1_scale")
            writer.add_tensor(
                f"{base}.w1a1_packed", data["packed"], raw_dtype=GGMLQuantizationType.I32
            )
            writer.add_tensor(
                f"{base}.w1a1_scale", data[variant], raw_dtype=GGMLQuantizationType.F32
            )
            selected.append(base)
        else:
            writer.add_tensor(tensor.name, tensor.data, raw_dtype=tensor.tensor_type)
            preserved.append(tensor.name)
    if set(selected) != set(SOURCE_NAMES):
        raise ValueError("F16 GGUF does not contain exactly the expected nine dense layers")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    exported = GGUFReader(destination)
    observed = {tensor.name: tensor for tensor in exported.tensors}
    for tensor in reader.tensors:
        if tensor.name in preserved:
            actual = observed[tensor.name]
            if actual.tensor_type != tensor.tensor_type or not np.array_equal(
                actual.data, tensor.data
            ):
                raise ValueError(f"nonselected tensor changed during export: {tensor.name}")
    for base in selected:
        for suffix, expected, kind in (
            ("packed", arrays[base]["packed"], GGMLQuantizationType.I32),
            ("scale", arrays[base][variant], GGMLQuantizationType.F32),
        ):
            actual = observed[f"{base}.w1a1_{suffix}"]
            if actual.tensor_type != kind or not np.array_equal(actual.data, expected):
                raise ValueError(f"export mismatch {base}/{suffix}")
        if f"{base}.weight" in observed:
            raise ValueError("dense shadow in binary GGUF")
    return {
        "path": str(destination),
        "sha256": sha256(destination),
        "audited_packed_layers": len(selected),
        "unchanged_nonselected_tensors": preserved,
    }


def main() -> None:
    from gguf import GGMLQuantizationType, GGUFReader
    from safetensors import safe_open

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--f16-gguf", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "protocol": PROTOCOL,
        "source_sha256": sha256(args.source),
        "f16_gguf_sha256": sha256(args.f16_gguf),
        "torch_version": torch.__version__,
        "device": args.device,
        "layers": {},
        "artifacts": {},
    }
    if args.device == "cuda":
        report["gpu"] = torch.cuda.get_device_name()
    inputs, report["capture"] = load_inputs(args.capture_manifest)
    tensors = {tensor.name: tensor for tensor in GGUFReader(args.f16_gguf).tensors}
    arrays = {}
    with safe_open(args.source, framework="pt", device="cpu") as checkpoint:
        for base in SOURCE_NAMES:
            weight = canonical_weight(checkpoint, base)
            dense = tensors[f"{base}.weight"]
            rounded = weight.astype(np.float16)
            if dense.tensor_type != GGMLQuantizationType.F16 or not np.array_equal(
                dense.data, rounded
            ):
                raise ValueError(f"{base}: source/layout does not exactly match original F16 GGUF")
            if not np.isfinite(rounded).all():
                raise ValueError(f"{base}: nonfinite source after F16 cast")
            scales, layer, details = fit_layer(weight, inputs[base], args.device)
            layer["source_f32_sha256"] = array_hash(weight)
            layer["source_to_f16_max_abs_error"] = float(
                np.abs(weight - rounded.astype(np.float32)).max()
            )
            layer["source_to_f16_sign_changes"] = int(
                np.count_nonzero((weight >= 0) != (rounded >= 0))
            )
            layer["packed_sha256"] = array_hash(pack_signs(weight))
            arrays[base] = {"K": weight.shape[1], "packed": pack_signs(weight), **scales}
            artifact = args.output_dir / f"{base}.npz"
            np.savez(artifact, **arrays[base], **details)
            layer["scale_artifact_sha256"] = sha256(artifact)
            report["layers"][base] = layer
            (args.output_dir / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({"event": "fit_layer_complete", "layer": base, "summary": layer}),
                flush=True,
            )
    for variant in "ABCD":
        report["artifacts"][variant] = export_variant(
            args.f16_gguf, args.output_dir / f"{variant}.gguf", arrays, variant
        )
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"event": "complete", "report": str(args.output_dir / "report.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()

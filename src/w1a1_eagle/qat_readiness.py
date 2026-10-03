"""Fail-closed admission of measured QAT optimization receipts; no device queries.

This validator does not manufacture evidence. A separately authorized GPU
producer (scripts/check_qat_optimization_readiness.py) must run the actual full
model forward/backward without real optimizer updates, native decision checks,
native learned-quantizer packing/correction execution, and peak memory checks.
The launch caller supplies freshly queried actual CUDA hardware and runtime
identity. CPU plans, estimates and test fixtures cannot authorize CUDA.

Schema qat_optimization_readiness_v1 binds source_sha256, training_runtime,
recipe, native_commit, backend and hardware. Its gates are full_model,
native_decisions and memory; learned_quantizers/fusion_correction are additionally
required when selected. Required execution/pass flags must be literal true.
The configured optimization_readiness locator is {path, sha256}; the checksum
binds the exact file bytes. This is provenance validation of a trusted measured
receipt, not a cryptographic attestation of its producer. No speedup is required
or inferred, and no receipt overrides live resource checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path

SCHEMA = "qat_optimization_readiness_v1"

# These change scheduling/retention/admission, not the measured training math.
# Unknown/new fields stay in the recipe so old evidence cannot grant new options.
_OPERATIONAL_FIELDS = {
    "optimization_readiness",
    "device",
    "allow_accelerator",
    "max_steps",
    "max_tokens",
    "max_seconds",
    "max_epochs",
    "checkpoint_every",
    "keep_checkpoints",
    "diagnostics_every",
    "development_every",
    "development_lifecycle",
    "min_free_disk_bytes",
    "max_cuda_reserved_bytes",
    "min_cuda_free_bytes",
    "min_host_available_bytes",
    "log_max_bytes",
    "log_backups",
}


def _canonical(value):
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("readiness identity must be finite JSON data") from error


def _config(config) -> dict:
    value = asdict(config) if is_dataclass(config) and not isinstance(config, type) else config
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError("readiness config must be a dataclass or string-keyed mapping")
    return dict(value)


def recipe_identity(config) -> dict:
    """Canonical math/recipe fields, excluding receipt and operational caps.

    CUDA indices are bound semantically by the hardware context rather than the
    recipe. Future unknown config fields are deliberately retained.
    """
    value = _config(config)
    return _canonical({k: v for k, v in value.items() if k not in _OPERATIONAL_FIELDS})


def optimization_requires_receipt(config) -> bool:
    """Default launches retain their existing gates; new options need this gate."""
    value = _config(config)
    correction = value.get("fusion_correction")
    affine = value.get("affine_weights")
    if affine is not None:
        affine = _config(affine)
    if correction is not None:
        correction = _config(correction)
    return any(
        (
            value.get("a1_computation", "reference") != "reference",
            value.get("activation_quantization", "fixed") != "fixed",
            value.get("binary_optimization") is not None,
            correction is not None and correction.get("enabled", False) is not False,
            affine is not None and affine.get("enabled", False) is not False,
            value.get("depth_loss_decay", 1.0) != 1.0,
            value.get("optimize_cache", False) is not False,
            value.get("optimize_head", False) is not False,
            value.get("persistent_sign_diagnostics", False) is not False,
            value.get("curriculum") is not None,
            "stages" in value,
        )
    )


def _hash(value, name, *, lengths=(64,)):
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or not re.fullmatch(r"[0-9a-f]+", value)
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal hash")
    return value


def _number(value, name, *, positive=True, integer=False):
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError(f"{name} must be a nonboolean number")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _true(value, name):
    if value is not True:
        raise ValueError(f"{name} must be literal true")


def _same(actual, expected, name):
    # JSON comparison distinguishes true/1 and includes all exact identity keys.
    if json.dumps(_canonical(actual), sort_keys=True) != json.dumps(
        _canonical(expected), sort_keys=True
    ):
        raise ValueError(f"readiness {name} differs from the requested launch")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate readiness JSON field")
        result[key] = value
    return result


def validate_optimization_readiness(
    config,
    *,
    source_sha256,
    runtime_identity,
    native_commit,
    backend,
    hardware,
    device=None,
    max_cuda_reserved_bytes=None,
    min_cuda_free_bytes=None,
) -> dict | None:
    """Validate CUDA admission against explicit current identity and memory caps.

    CPU returns None before receipt access or CUDA-context validation. For CUDA,
    every call requires a receipt, even for a default recipe; callers use
    optimization_requires_receipt to preserve existing baseline admission.
    Memory caps may come from config or explicit caller arguments (needed by
    curriculum configs). Device/hardware are passed by the launch owner; this
    helper imports no Torch and never discovers or initializes an accelerator.
    """
    config = _config(config)
    requested_device = str(device if device is not None else config.get("device", "cpu"))
    device_type = requested_device.split(":", 1)[0]
    if device_type == "cpu":
        return None
    if device_type != "cuda" or backend != "cuda":
        raise ValueError("optimization readiness requires the actual CUDA backend")
    _hash(source_sha256, "source_sha256")
    _hash(native_commit, "native_commit", lengths=(40, 64))
    hardware = _object(_canonical(hardware), "hardware")
    if (
        hardware.get("device_type") != "cuda"
        or not isinstance(hardware.get("name"), str)
        or not hardware["name"].strip()
    ):
        raise ValueError("explicit actual CUDA hardware name/device required")
    capability = hardware.get("compute_capability")
    if not isinstance(capability, list) or len(capability) != 2:
        raise ValueError("actual CUDA compute capability required")
    _number(capability[0], "compute capability major", integer=True)
    _number(capability[1], "compute capability minor", positive=False, integer=True)
    total = _number(hardware.get("total_memory_bytes"), "hardware memory", integer=True)
    runtime_identity = _object(_canonical(runtime_identity), "training_runtime")
    if runtime_identity.get("device_type") != "cuda":
        raise ValueError("CPU runtime evidence cannot grant CUDA admission")
    for field in ("python_version", "torch_version", "numpy_version"):
        if not isinstance(runtime_identity.get(field), str) or not runtime_identity[field]:
            raise ValueError(f"actual runtime {field} required")
    sources = _object(runtime_identity.get("math_source_sha256"), "math_source_sha256")
    if not sources:
        raise ValueError("actual math-source identity required")
    for name, digest in sources.items():
        _hash(digest, f"math source {name}")
    cuda_math = _object(runtime_identity.get("cuda_math"), "cuda_math")
    if cuda_math.get("float32_matmul_precision") not in ("highest", "high", "medium"):
        raise ValueError("actual CUDA matmul precision required")
    for field in ("matmul_allow_tf32", "cudnn_allow_tf32"):
        if type(cuda_math.get(field)) is not bool:
            raise ValueError(f"actual CUDA {field} must be boolean")
    locator = _object(config.get("optimization_readiness"), "optimization_readiness")
    if (
        set(locator) != {"path", "sha256"}
        or not isinstance(locator["path"], str)
        or not locator["path"]
    ):
        raise ValueError("readiness locator must contain only path and sha256")
    expected_sha = _hash(locator["sha256"], "receipt sha256")
    try:
        data = Path(locator["path"]).read_bytes()
    except OSError as error:
        raise ValueError("readiness receipt cannot be read") from error
    if len(data) > 4 * 1024**2 or hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("readiness receipt byte hash mismatch or oversized file")
    try:
        receipt = json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite readiness JSON constant: {value}")
            ),
        )
    except (ValueError, UnicodeError) as error:
        raise ValueError("invalid readiness receipt JSON") from error
    receipt = _object(receipt, "receipt")
    if receipt.get("schema") != SCHEMA or receipt.get("evidence_device_type") != "cuda":
        raise ValueError("current-schema measured CUDA receipt required")
    if receipt.get("fixture_only", False) is not False:
        raise ValueError("test fixtures cannot grant CUDA admission")
    for field, expected in (
        ("source_sha256", source_sha256),
        ("training_runtime", runtime_identity),
        ("recipe", recipe_identity(config)),
        ("native_commit", native_commit),
        ("backend", backend),
        ("hardware", hardware),
    ):
        _same(receipt.get(field), expected, field)
    gates = _object(receipt.get("gates"), "gates")
    full = _object(gates.get("full_model"), "full_model")
    _true(full.get("passed"), "full_model.passed")
    _true(full.get("finite_gradients"), "full_model.finite_gradients")
    if full.get("model_scope") != "full_model":
        raise ValueError("tiny/estimated model evidence cannot grant full-model admission")
    for field in ("forward_calls", "backward_calls"):
        _number(full.get(field), f"full_model.{field}", integer=True)
    for field in (
        "later_state_gradient_norm",
        "later_key_gradient_norm",
        "later_value_gradient_norm",
    ):
        _number(full.get(field), f"full_model.{field}")
    native = _object(gates.get("native_decisions"), "native_decisions")
    for field in ("passed", "executed"):
        _true(native.get(field), f"native_decisions.{field}")
    _number(native.get("cases"), "native_decisions.cases", integer=True)
    changes = _number(
        native.get("changed_choice_count"), "changed choices", positive=False, integer=True
    )
    material = _number(
        native.get("material_choice_changes"), "material choices", positive=False, integer=True
    )
    if material != 0 or material > changes:
        raise ValueError("material native decision changes cannot grant admission")
    for field, limit in (
        ("max_changed_choice_margin", 0.02),
        ("max_state_relative_rms", 0.10),
        ("max_logit_relative_rms", 0.10),
    ):
        if _number(native.get(field), f"native_decisions.{field}", positive=False) > limit:
            raise ValueError(f"native_decisions.{field} exceeds the existing numeric gate")
    memory = _object(gates.get("memory"), "memory")
    _true(memory.get("passed"), "memory.passed")
    allocated, reserved, free = (
        _number(memory.get(field), f"memory.{field}", integer=True)
        for field in ("peak_allocated_bytes", "peak_reserved_bytes", "min_free_bytes")
    )
    ceiling = _number(
        max_cuda_reserved_bytes
        if max_cuda_reserved_bytes is not None
        else config.get("max_cuda_reserved_bytes"),
        "reserved memory ceiling",
        integer=True,
    )
    floor = _number(
        min_cuda_free_bytes
        if min_cuda_free_bytes is not None
        else config.get("min_cuda_free_bytes"),
        "free memory floor",
        integer=True,
    )
    if not allocated <= reserved <= min(ceiling, total) or not floor <= free <= total:
        raise ValueError("measured CUDA memory does not fit current launch limits")
    if config.get("activation_quantization", "fixed") == "learned":
        learned = _object(gates.get("learned_quantizers"), "learned_quantizers")
        for field in ("passed", "executed", "exact_pack"):
            _true(learned.get(field), f"learned_quantizers.{field}")
        _number(learned.get("cases"), "learned_quantizers.cases", integer=True)
        _hash(learned.get("artifact_sha256"), "learned quantizer artifact")
    correction = config.get("fusion_correction")
    if correction is not None and _config(correction).get("enabled", False) is not False:
        correction = _object(gates.get("fusion_correction"), "fusion_correction")
        for field in (
            "passed",
            "raw_fc_executed",
            "zero_identity_passed",
            "nonzero_forward_passed",
            "raw_input_ancestry_passed",
        ):
            _true(correction.get(field), f"fusion_correction.{field}")
        _number(correction.get("cases"), "fusion_correction.cases", integer=True)
        _hash(correction.get("artifact_sha256"), "fusion correction artifact")
    affine_config = config.get("affine_weights")
    if affine_config is not None and _config(affine_config).get("enabled", False) is not False:
        affine = _object(gates.get("affine_weights"), "affine_weights")
        for field in (
            "passed",
            "executed",
            "mu_zero_identity_passed",
            "alpha_zero_nonzero_mu_passed",
            "exact_code_sum",
        ):
            _true(affine.get(field), "affine_weights." + field)
        fixtures = {"mu_zero_identity", "alpha_zero_nonzero_mu", "quantized_code_sum"}
        if (
            not isinstance(affine.get("fixture_coverage"), list)
            or set(affine["fixture_coverage"]) != fixtures
        ):
            raise ValueError("affine weights require zero/nonzero/same-code native fixtures")
        _number(affine.get("cases"), "affine_weights.cases", integer=True)
        _hash(affine.get("artifact_sha256"), "affine weights artifact")
        expected = {"fc"}
        if _config(affine_config).get("coverage") == "all":
            expected.update(
                {
                    "output",
                    "blk.0.attn_q",
                    "blk.0.attn_k",
                    "blk.0.attn_v",
                    "blk.0.attn_output",
                    "blk.0.ffn_gate",
                    "blk.0.ffn_up",
                    "blk.0.ffn_down",
                }
            )
        if (
            not isinstance(affine.get("projection_bases"), list)
            or set(affine["projection_bases"]) != expected
        ):
            raise ValueError("affine native fixture coverage differs from the declared layers")
    if config.get("curriculum") is not None:
        schedule = _object(config["curriculum"], "curriculum")
        expected_bits = [stage["activation_bits"] for stage in schedule["stages"]]
        measured = _object(gates.get("curriculum_stages"), "curriculum_stages")
        _true(measured.get("passed"), "curriculum_stages.passed")
        if measured.get("activation_bits") != expected_bits:
            raise ValueError("readiness does not cover the complete ordered precision schedule")
        stages = measured.get("stages")
        if not isinstance(stages, list) or len(stages) != len(expected_bits):
            raise ValueError("readiness precision stage inventory differs")
        for bits, stage in zip(expected_bits, stages):
            if (
                not isinstance(stage, dict)
                or type(stage.get("activation_bits")) is not int
                or stage["activation_bits"] != bits
            ):
                raise ValueError("readiness precision stage identity differs")
            full_stage = _object(stage.get("full_model"), "stage.full_model")
            for field in ("passed", "finite_gradients"):
                _true(full_stage.get(field), "stage.full_model." + field)
            if full_stage.get("model_scope") != "full_model":
                raise ValueError("curriculum stage is not an actual full model")
            for field in (
                "forward_calls",
                "backward_calls",
                "later_state_gradient_norm",
                "later_key_gradient_norm",
                "later_value_gradient_norm",
            ):
                _number(
                    full_stage.get(field),
                    "stage.full_model." + field,
                    integer=field.endswith("calls"),
                )
            stage_memory = _object(stage.get("memory"), "stage.memory")
            _true(stage_memory.get("passed"), "stage.memory.passed")
            stage_allocated, stage_reserved, stage_free = (
                _number(stage_memory.get(field), "stage.memory." + field, integer=True)
                for field in ("peak_allocated_bytes", "peak_reserved_bytes", "min_free_bytes")
            )
            if (
                not stage_allocated <= stage_reserved <= min(ceiling, total)
                or not floor <= stage_free <= total
            ):
                raise ValueError("curriculum stage memory does not fit the configured bounds")
            native_stage = _object(stage.get("native_gates"), "stage.native_gates")
            decisions = _object(native_stage.get("native_decisions"), "stage.native_decisions")
            for field in ("passed", "executed"):
                _true(decisions.get(field), "stage.native_decisions." + field)
            _number(decisions.get("cases"), "stage.native_decisions.cases", integer=True)
            stage_changes = _number(
                decisions.get("changed_choice_count"),
                "stage.changed_choice_count",
                positive=False,
                integer=True,
            )
            stage_material = _number(
                decisions.get("material_choice_changes"),
                "stage.material_choice_changes",
                positive=False,
                integer=True,
            )
            if stage_material != 0 or stage_material > stage_changes:
                raise ValueError("curriculum stage has material native choice changes")
            for field, limit in (
                ("max_changed_choice_margin", 0.02),
                ("max_state_relative_rms", 0.10),
                ("max_logit_relative_rms", 0.10),
            ):
                if (
                    _number(decisions.get(field), "stage.native_decisions." + field, positive=False)
                    > limit
                ):
                    raise ValueError("curriculum stage exceeds the native numeric gate")
            for optional in ("learned_quantizers", "fusion_correction", "affine_weights"):
                if optional in gates:
                    proof = _object(native_stage.get(optional), "stage." + optional)
                    fields = {
                        "learned_quantizers": ("passed", "executed", "exact_pack"),
                        "fusion_correction": (
                            "passed",
                            "raw_fc_executed",
                            "zero_identity_passed",
                            "nonzero_forward_passed",
                            "raw_input_ancestry_passed",
                        ),
                        "affine_weights": (
                            "passed",
                            "executed",
                            "mu_zero_identity_passed",
                            "alpha_zero_nonzero_mu_passed",
                            "exact_code_sum",
                        ),
                    }[optional]
                    for field in fields:
                        _true(proof.get(field), "stage." + optional + "." + field)
                    _number(proof.get("cases"), "stage." + optional + ".cases", integer=True)
                    _hash(proof.get("artifact_sha256"), "stage." + optional + " artifact")
    return receipt


def native_checkout_commit() -> str:
    """Read this checkout's actual native revision; no network or accelerator."""
    import subprocess

    native_root = Path(__file__).resolve().parents[2] / "third_party" / "llama.cpp"
    try:
        result = subprocess.run(
            ["git", "-C", str(native_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(
            "cannot bind optimization admission to the actual native checkout"
        ) from error
    return _hash(result.stdout.strip(), "native checkout commit", lengths=(40, 64))


def require_measured_cuda_readiness(
    config,
    *,
    source_sha256,
    runtime_identity,
    device=None,
    max_cuda_reserved_bytes=None,
    min_cuda_free_bytes=None,
) -> dict | None:
    """Launch adapter; called only at an authorized real optimizer boundary."""
    value = _config(config)
    device = str(device if device is not None else value.get("device", "cpu"))
    if device.split(":", 1)[0] == "cpu":
        return None
    if value.get("optimization_readiness") is None:
        raise ValueError(
            "optimized CUDA QAT requires a fresh measured optimization readiness receipt"
        )
    import torch

    properties = torch.cuda.get_device_properties(device)
    hardware = {
        "device_type": "cuda",
        "name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": properties.total_memory,
    }
    return validate_optimization_readiness(
        config,
        source_sha256=source_sha256,
        runtime_identity=runtime_identity,
        native_commit=native_checkout_commit(),
        backend="cuda",
        hardware=hardware,
        device=device,
        max_cuda_reserved_bytes=max_cuda_reserved_bytes,
        min_cuda_free_bytes=min_cuda_free_bytes,
    )


def curriculum_readiness_config(qat, runner, curriculum) -> dict:
    """Bind one complete precision schedule rather than a single isolated stage."""
    result = _config(qat)
    result["runner"] = _canonical(_config(runner))
    result["curriculum"] = _canonical(curriculum.manifest())
    return result

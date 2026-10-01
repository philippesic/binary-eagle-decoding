"""Training runtime/math identity without accelerator discovery on import."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path

import numpy as np
import torch

MATH_FILES = (
    "recurrent_qat.py",
    "recurrent_binary.py",
    "native_step.py",
    "recurrent_rollout.py",
    "recurrent_loss.py",
    "recurrent_trace.py",
    "recurrent_provider.py",
    "continuous_qat.py",
    "qat_state.py",
    "qat_readiness.py",
    "affine_binary.py",
    "qat_optimization.py",
    "qat_curriculum.py",
    "learned_activation.py",
    "fusion_correction.py",
)


ADMISSION_FILES = (
    "w1ax_capture_provider.py",
    "w1ax_multishard_provider.py",
    "w1ax_continuous_stages.py",
    "check_continuous_w1ax_readiness.py",
    "export_recurrent_binary.py",
    "train_continuous_w1ax.py",
    "check_qat_optimization_readiness.py",
    "check_curriculum_qat_readiness.py",
    "collect_qat_native_evidence.py",
    "train_qat_curriculum.py",
)


def training_runtime_identity(device: str, *, source_root: Path | None = None) -> dict:
    """Record critical math hashes and exact Python/Torch/NumPy versions.

    No Git revision is included: documentation commits do not affect numerical
    resume. CUDA precision getters run only for the explicit CUDA training path.
    This rejects changed runtime/code; it does not promise cross-backend parity.
    """
    source_root = Path(source_root) if source_root is not None else Path(__file__).parent
    result = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "device_type": torch.device(device).type,
        "math_source_sha256": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in MATH_FILES
        },
    }
    scripts_root = Path(__file__).resolve().parents[2] / "scripts"
    result["admission_source_sha256"] = {
        name: hashlib.sha256((scripts_root / name).read_bytes()).hexdigest()
        for name in ADMISSION_FILES
    }
    if torch.device(device).type == "cuda":
        result["cuda_math"] = {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        }
    return result

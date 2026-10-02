"""Reproduce bounded CPU identity/receipt checks; creates no real GPU receipt."""

import copy
import hashlib
import json
import platform
import unittest
from pathlib import Path
from unittest.mock import patch

import test_qat_readiness as fixtures
import torch

from w1a1_eagle.continuous_runtime import training_runtime_identity

torch.set_num_threads(1)
identity = training_runtime_identity("cpu")
context = copy.deepcopy(fixtures.CONTEXT)
# The historical receipt fixture predates required helper closure. Supply the
# complete actual source inventory to both synthetic maps for its guard tests.
context["runtime_identity"]["math_source_sha256"] = identity["math_source_sha256"]
modules = [
    "test_parallel20261002_activation_identity",
    "test_continuous_qat",
    "test_qat_curriculum_runner",
    "test_qat_readiness",
]
suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
with patch.dict(fixtures.CONTEXT, context, clear=True):
    result = unittest.TextTestRunner(verbosity=1).run(suite)
root = Path(__file__).resolve().parents[3]
proof = {
    "device": "CPU",
    "platform": platform.platform(),
    "torch_version": str(torch.__version__),
    "tests_run": result.testsRun,
    "passed": result.wasSuccessful(),
    "receipt_scope": "synthetic CUDA schema context only; no actual GPU admission",
    "historical_fixture_alignment": "scoped complete-source map in both synthetic maps",
    "math_source_sha256": identity["math_source_sha256"],
    "preserved_feature_sha256": {
        name: hashlib.sha256((root / "src/w1a1_eagle" / name).read_bytes()).hexdigest()
        for name in ("activation_reuse.py", "learned_activation.py", "native_step.py")
    },
}
print(json.dumps(proof, indent=2, sort_keys=True))
raise SystemExit(not result.wasSuccessful())

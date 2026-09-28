"""Small native CPU graph and metric checks for the layer-0 probe."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.probe_target_layer0_projection import compile_helper, metrics

ROOT = Path(__file__).resolve().parents[1]


class Layer0ProjectionProbeTests(unittest.TestCase):
    def test_metrics_exact_and_different(self) -> None:
        reference = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        self.assertEqual(metrics(reference, reference)["exact_elements"], 4)
        shifted = reference.copy()
        shifted[0, 1] += 1.0
        result = metrics(shifted, reference)
        self.assertEqual(result["exact_elements"], 3)
        self.assertEqual(result["max_abs"], 1.0)

    @unittest.skipUnless(
        os.environ.get("LAYER0_LLAMA_ROOT") and os.environ.get("LAYER0_CPU_BUILD"),
        "native ggml CPU source and build paths are required",
    )
    def test_native_cpu_graph_fixture(self) -> None:
        llama_root = Path(os.environ["LAYER0_LLAMA_ROOT"])
        cpu_build = Path(os.environ["LAYER0_CPU_BUILD"])
        with tempfile.TemporaryDirectory(prefix="layer0-probe-test-") as temporary:
            directory = Path(temporary)
            binary = directory / "native_layer0_projection"
            compile_helper(
                ROOT / "scripts/native_layer0_projection.cpp", llama_root, cpu_build, binary
            )
            x = np.array([[1, 2, 3, 4], [2, 0, -2, 0]], dtype="<f4")
            x.tofile(directory / "embedding.f32")
            np.ones(4, dtype="<f4").tofile(directory / "attn_norm.f32")
            np.eye(4, dtype="<f2").tofile(directory / "q.f16")
            np.array([[1, 0, 0, 0]], dtype="<f2").tofile(directory / "k.f16")
            np.array([[0, 0, 0, 1]], dtype="<f2").tofile(directory / "v.f16")
            arguments = [str(binary), str(directory), "2", "4", "4", "1", "1"]
            subprocess.run(arguments, check=True)
            norm = np.fromfile(directory / "native_norm.f32", dtype="<f4").reshape(2, 4)
            expected = x / np.sqrt(np.mean(x * x, axis=1, keepdims=True) + 1e-6)
            np.testing.assert_allclose(norm, expected, atol=2e-7)
            for label, expected_projection in (
                ("q", expected),
                ("k", expected[:, :1]),
                ("v", expected[:, -1:]),
            ):
                actual = np.fromfile(directory / f"native_{label}.f32", dtype="<f4")
                np.testing.assert_allclose(
                    actual.reshape(expected_projection.shape), expected_projection, atol=1e-3
                )
            (directory / "q.f16").write_bytes(b"bad")
            self.assertNotEqual(subprocess.run(arguments, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()

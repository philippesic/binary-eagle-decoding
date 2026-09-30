"""Fixture-only Linux/WSL resource admission; no real host/GPU probing."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from w1a1_eagle.continuous_qat import ContinuousConfig
from w1a1_eagle.continuous_resources import (
    checkpoint_host_buffer_bytes,
    lane_storage_bytes,
    linux_host_memory,
    model_storage_bytes,
    require_host_memory,
)
from w1a1_eagle.recurrent_qat import RowBinaryLinear, W1AxContract


class HostResourceTests(unittest.TestCase):
    def test_linux_units_sources_and_current_rss_from_fixtures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "self").mkdir()
            (root / "meminfo").write_text(
                "MemTotal: 33554432 kB\nMemFree: 1 kB\nMemAvailable: 12582912 kB\n"
            )
            (root / "self/status").write_text("Name: fixture\nVmRSS: 1048576 kB\n")
            result = linux_host_memory(root)
        self.assertEqual(result["host_available_bytes"], 12 * 1024**3)
        self.assertEqual(result["host_total_bytes"], 32 * 1024**3)
        self.assertEqual(result["process_rss_bytes"], 1024**3)
        self.assertIn("Linux/WSL", result["host_memory_source"])

    def test_bad_missing_units_and_unavailable_proc_fail_safely(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, "Linux/WSL"):
                linux_host_memory(root)
            (root / "self").mkdir()
            (root / "self/status").write_text("VmRSS: 5 kB\n")
            for source in (
                "MemTotal: 20 kB\nMemAvailable: 3 MB\n",
                "MemTotal: 20 kB\nMemFree: 3 kB\n",
                "MemTotal: 20 kB\nMemAvailable: -3 kB\n",
                "MemTotal: 20 kB\nMemAvailable: 3 kB\nMemAvailable: 3 kB\n",
                "MemTotal: 20 kB\nMemAvailable: 30 kB\n",
            ):
                (root / "meminfo").write_text(source)
                with self.assertRaises(RuntimeError):
                    linux_host_memory(root)

    def test_admission_adds_offload_to_floor_before_allocation(self):
        stats = {"host_available_bytes": 8 * 1024**3, "process_rss_bytes": 1024**3}
        report = require_host_memory(
            stats, floor_bytes=2 * 1024**3, additional_bytes=6 * 1024**3, stage="dual offload"
        )
        self.assertEqual(report["host_required_available_bytes"], 8 * 1024**3)
        with self.assertRaisesRegex(RuntimeError, "dual offload"):
            require_host_memory(
                dict(stats, host_available_bytes=8 * 1024**3 - 1),
                floor_bytes=2 * 1024**3,
                additional_bytes=6 * 1024**3,
                stage="dual offload",
            )

    def test_unique_alias_storage_and_optimizer_moments_counted_once(self):
        shared = nn.Parameter(torch.ones(4), requires_grad=False)
        models, lanes = [], []
        for _ in range(2):
            model = nn.Module()
            model.shared = shared
            model.selected = nn.Parameter(torch.ones(3))
            model.register_buffer("view", shared[:2])
            optimizer = torch.optim.AdamW([model.selected], foreach=False)
            optimizer.state[model.selected]["exp_avg"] = torch.zeros(3)
            optimizer.state[model.selected]["exp_avg_sq"] = torch.zeros(3)
            lanes.append(SimpleNamespace(drafter=model, optimizer=optimizer))
            models.append(model)
        with patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU query")):
            self.assertEqual(lane_storage_bytes(lanes), 4 * 4 + 2 * 3 * 4 * 3)
            self.assertEqual(model_storage_bytes(models[0], device_type="cpu"), (4 + 3) * 4)
            self.assertEqual(lane_storage_bytes(lanes, device_type="cuda"), 0)

    def test_checkpoint_export_bound_and_config_floor(self):
        module = RowBinaryLinear(torch.ones(3, 4), torch.ones(3), W1AxContract(8))
        lanes = [SimpleNamespace(linears={"fixture": module})]
        self.assertEqual(
            checkpoint_host_buffer_bytes(lanes), 2 * (12 + 3) * 4 + 12 * 4 + 16 * 1024**2
        )
        with self.assertRaisesRegex(ValueError, "min_host_available_bytes"):
            ContinuousConfig(min_host_available_bytes=0)
        with self.assertRaisesRegex(ValueError, "min_host_available_bytes"):
            ContinuousConfig(min_host_available_bytes=float("nan"))


if __name__ == "__main__":
    unittest.main()

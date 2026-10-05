"""CPU development source fixtures never grant production/CUDA admission."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_nine_model_train_data as capture  # noqa: E402
import test_nine_model_train_capture as fixtures  # noqa: E402
from check_block_capture_portability import NativeCaptureGoldens  # noqa: E402

from w1a1_eagle.block_data import BlockDataset, file_sha256  # noqa: E402


class CpuTeacher(fixtures.SyntheticNativeTeacher):
    def make(self, tokens, taps, mode, ancestry):
        result = super().make(tokens, taps, mode, ancestry)
        result.update(
            hardware=["CPU: fixture CPU"],
            executed_result_buffers=["CPU"],
            target_storage_buffers={"CPU": 1},
            gpu_layers=0,
        )
        return result


class CpuCaptureTests(unittest.TestCase):
    persist = fixtures.CaptureTests.persist

    def setUp(self):
        fixtures.CaptureTests.setUp(self)
        self.plan["execution_profile"] = "development_CPU"
        self.plan["native"]["gpu_layers"] = 0
        self.plan["native"]["expected_compute_capability"] = None
        caps = self.plan["caps"]
        caps.update(
            max_host_rss_bytes=12 * 1024**3,
            min_host_available_bytes=12 * 1024**3,
            min_host_available_live_bytes=4 * 1024**3,
        )
        cache = self.root / "CMakeCache.txt"
        cache.write_text(
            "GGML_CPU:BOOL=ON\n" + "".join(key + ":BOOL=OFF\n" for key in capture.CPU_OFF_OPTIONS)
        )
        dylibs = []
        for name in (
            "libllama.0.dylib",
            "libggml.0.dylib",
            "libggml-cpu.0.dylib",
            "libggml-base.0.dylib",
        ):
            path = self.root / name
            path.write_bytes(b"synthetic resolved library fixture")
            dylibs.append({"path": str(path), "sha256": file_sha256(path)})
        links = self.root / "otool.txt"
        links.write_text("\n".join("@rpath/" + Path(p["path"]).name for p in dylibs))
        self.plan["cpu_build"] = capture.write_json(
            self.root / "cpu-build.json",
            {
                "schema": "nine_model_cpu_teacher_build_v1",
                "binary_sha256": self.plan["native"]["binary"]["sha256"],
                "source_revision": self.plan["native"]["source_revision"],
                "cmake_cache": {"path": str(cache), "sha256": file_sha256(cache)},
                "dylibs": dylibs,
                "otool_links": {"path": str(links), "sha256": file_sha256(links)},
                "library_directory": str(self.root),
            },
        )
        self.persist()

    def run_(self, *, available=32 * 1024**3, clock=None, teacher=CpuTeacher):
        return capture.run_capture(
            self.path,
            self.pin,
            self.root / "output",
            execute=True,
            execution_profile="development_CPU",
            teacher_factory=teacher,
            device_query=lambda: {
                "name": "fixture CPU",
                "backend": "CPU",
                "compute_capability": None,
            },
            rss_query=lambda: 0,
            available_query=lambda: available,
            **({"clock": clock} if clock else {}),
        )

    def test_cpu_fixture_source_imports_and_goldens_cannot_grant_cuda(self):
        result = self.run_()
        self.assertEqual(result["status"], "PASS", result["failure"])
        self.assertEqual(result["artifact_kind"], "synthetic_fixture")
        self.assertEqual(result["production_data_status"], "SYNTHETIC_ONLY")
        self.assertEqual(result["sm120_readiness"], "PENDING")
        manifest = result["manifests"]["dspark"]
        dataset = BlockDataset(manifest["path"], expected_sha256=manifest["sha256"])
        self.assertIn("CPU", dataset.manifest["producer"]["hardware"])
        with self.assertRaisesRegex(ValueError, "profile"):
            NativeCaptureGoldens(
                result["eagle_goldens"]["path"], expected_sha256=result["eagle_goldens"]["sha256"]
            )
        self.assertTrue(result["producer_closed"])

    def test_default_cuda_mode_refuses_cpu_plan(self):
        with self.assertRaises(ValueError):
            capture.prepare_plan(self.path, self.pin)

    def test_weak_floor_or_excess_caps_refuse(self):
        self.plan["caps"]["min_host_available_live_bytes"] = 1
        self.persist()
        with self.assertRaisesRegex(ValueError, "caps|floors"):
            capture.prepare_plan(self.path, self.pin, execution_profile="development_CPU")

    def test_cpu_build_metal_on_refuses(self):
        proof_path = Path(self.plan["cpu_build"]["path"])
        proof = json.loads(proof_path.read_text())
        cache = Path(proof["cmake_cache"]["path"])
        cache.write_text(cache.read_text().replace("GGML_METAL:BOOL=OFF", "GGML_METAL:BOOL=ON"))
        proof["cmake_cache"]["sha256"] = file_sha256(cache)
        proof_path.write_text(json.dumps(proof))
        self.plan["cpu_build"]["sha256"] = file_sha256(proof_path)
        self.persist()
        with self.assertRaisesRegex(ValueError, "OFF"):
            capture.prepare_plan(self.path, self.pin, execution_profile="development_CPU")

    def test_cpu_execution_proof_rejects_gpu_result(self):
        class GpuResult(CpuTeacher):
            def make(self, *args):
                result = super().make(*args)
                result["executed_result_buffers"] = ["Metal"]
                return result

        result = self.run_(teacher=GpuResult)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["producer_closed"])

    def test_cpu_low_available_refuses_before_teacher(self):
        result = self.run_(available=8 * 1024**3)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(fixtures.SyntheticNativeTeacher.instances, [])

    def test_live_floor_failure_closes_and_preserves_cpu_data(self):
        observations = iter([32 * 1024**3, 8 * 1024**3, 3 * 1024**3])
        result = capture.run_capture(
            self.path,
            self.pin,
            self.root / "output",
            execute=True,
            execution_profile="development_CPU",
            teacher_factory=CpuTeacher,
            device_query=lambda: {
                "name": "fixture CPU",
                "backend": "CPU",
                "compute_capability": None,
            },
            rss_query=lambda: 0,
            available_query=lambda: next(observations),
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["producer_closed"])
        self.assertIn("floor", result["failure"]["message"])
        self.assertTrue((self.root / "output/receipts/000000-block.json").exists())

    def test_stop_and_wall_cap_close_owned_cpu_producer(self):
        fixtures.SyntheticNativeTeacher.fault = "stop"
        result = self.run_()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["failure"]["type"], "InterruptedError")
        self.assertTrue(result["producer_closed"])

    def test_wall_cap_refuses_before_cpu_model(self):
        ticks = iter([0, 101, 102])
        result = self.run_(clock=lambda: next(ticks))
        self.assertEqual(result["failure"]["type"], "TimeoutError")
        self.assertEqual(fixtures.SyntheticNativeTeacher.instances, [])

    def test_mac_memory_metric_exact_and_missing_refuses(self):
        output = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free: 2.\nPages inactive: 3.\nPages speculative: 4.\n"
        )
        with patch.object(capture.subprocess, "check_output", return_value=output):
            self.assertEqual(capture.mac_available_bytes(), 9 * 16384)
        with patch.object(capture.subprocess, "check_output", return_value="bad"):
            with self.assertRaises(ValueError):
                capture.mac_available_bytes()

    def test_dyld_log_missing_or_changed_pin_refuses(self):
        import types

        build = capture.validate_cpu_build(self.plan["cpu_build"], self.plan["native"], self.root)
        log = self.root / "dyld.log"
        teacher = types.SimpleNamespace(
            log=types.SimpleNamespace(name=str(log)), process=types.SimpleNamespace(pid=42)
        )
        log.write_text("not a loaded-library proof")
        with self.assertRaisesRegex(ValueError, "dyld"):
            capture.cpu_loaded_library_proof(teacher, build)
        log.write_text("\n".join("dyld[42]: " + p["path"] for p in build["dylibs"]))
        proof = capture.cpu_loaded_library_proof(teacher, build)
        self.assertTrue(proof["actual_dyld_paths_checked"])
        self.assertFalse(proof["actual_linux_mapping_checked"])


if __name__ == "__main__":
    unittest.main()

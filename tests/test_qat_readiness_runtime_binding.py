"""Tiny hashed schema5 runtime joins; no models, native execution or GPU proof."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import check_continuous_w1ax_readiness as gate  # noqa: E402
import test_continuous_readiness as legacy  # noqa: E402
import test_qat_recipe_provider as recipes  # noqa: E402
import w1ax_continuous_stages as stages  # noqa: E402


class RuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.current = self.runtime("current")
        self.historical = self.runtime("historical")
        self.gates = {}
        for bits in (8, 1, 4):
            self.gates[str(bits)] = self.recipe_gate(bits)
        self.common = self.gates["1"]["common_source_sha256"]
        self.ready = {
            "schema": stages.RECIPE_READINESS_SCHEMA,
            "training_eligible": True,
            "objective": "hard_ce",
            "scale_layout": "row",
            "common_source_sha256": self.common,
            "unresolved_gates": [],
            "native_binary_sha256": self.current["binary"]["sha256"],
            "native_runtime": self.current["runtime"],
            "precisions": {
                bits: self.write("gate-" + bits + ".json", report)
                for bits, report in self.gates.items()
            },
        }

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value) if isinstance(value, bytes) else path.write_text(json.dumps(value))
        return stages.file_record(path)

    def runtime(self, name):
        binary = self.write(name + "/llama-server", (name + " server fixture").encode())
        libraries = [
            self.write(name + "/" + lib, (name + lib).encode())
            for lib in ("libllama.so", "libggml-cuda.so")
        ]
        manifest = self.write(
            name + "/manifest.json",
            {"fixture_only": True, "binary": binary, "libraries": libraries},
        )
        return {
            "binary": binary,
            "runtime": {
                "schema": "CPU_runtime_fixture",
                "directory": str(self.root / name),
                "immutable_manifest": manifest,
                "libraries": libraries,
                "ld_library_path": str(self.root / name),
            },
        }

    def recipe_gate(self, bits):
        fixture = legacy.ReadinessTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        report = fixture.report(bits)
        actor = recipes.RecipeProviderTests()
        actor.setUp()
        self.addCleanup(actor.doCleanups)
        binding, manifest, _, _ = actor.actor(bits=bits, schema=5)
        counts = {"sign": 9, "scale": 9, "activation": 6, "fusion": 3, "midpoint": 9}
        report.update(
            schema=gate.RECIPE_SCHEMA,
            recipe=gate.checkpoint_recipe(manifest),
            deployment_state_sha256="c" * 64,
            trainable_parameter_counts=counts,
            native_binary_sha256=self.current["binary"]["sha256"],
            native_runtime=copy.deepcopy(self.current["runtime"]),
        )
        for root in report["roots"]:
            root["finite_gradient_tensors"] = sum(counts.values())
        report["evidence"].update(binding)
        for name in ("native_cell", "native_capture_manifest"):
            path = Path(report["evidence"][name]["path"])
            data = json.loads(path.read_text())
            data["draft_sha256"] = binding["export"]["sha256"]
            if name == "native_cell":
                data["binary_sha256"] = report["native_binary_sha256"]
                log = path.parent / "server.log"
                log.write_text(
                    {
                        1: "CUDA packed W1A1 XOR/POPCOUNT dispatch",
                        4: "CUDA packed W1A4 BITSERIAL dispatch",
                        8: "CUDA packed W1A8 INT8 dispatch",
                    }[bits]
                )
                data["files"] = {"server.log": {"sha256": stages.sha256(log)}}
            path.write_text(json.dumps(data))
            report["evidence"][name] = stages.file_record(path)
        gate.validate_gate_report(report, bits, fixture.common)
        return report

    def validate(self, ready=None, bits=1):
        record = self.write("ready.json", self.ready if ready is None else ready)
        with patch("torch.cuda.is_available", side_effect=AssertionError("GPU discovery")):
            return stages.validate_readiness(
                record, activation_bits=bits, common_hashes=self.common
            )

    def replace_gate(self, ready, bits, report):
        ready["precisions"][str(bits)] = self.write("replacement-" + str(bits) + ".json", report)

    def test_same_runtime_schema5_all_independent_precisions_and_subsets(self):
        for bits in (8, 1, 4):
            with self.subTest(bits=bits):
                self.assertEqual(self.validate(bits=bits), self.ready)
                solo = copy.deepcopy(self.ready)
                solo["precisions"] = {str(bits): solo["precisions"][str(bits)]}
                self.assertEqual(self.validate(solo, bits), solo)

    def test_wrong_binary_is_rejected_even_with_internally_valid_gate(self):
        report = copy.deepcopy(self.gates["8"])
        report["native_binary_sha256"] = self.historical["binary"]["sha256"]
        path = Path(report["evidence"]["native_cell"]["path"])
        cell = json.loads(path.read_text())
        cell["binary_sha256"] = report["native_binary_sha256"]
        path.write_text(json.dumps(cell))
        report["evidence"]["native_cell"] = stages.file_record(path)
        gate.validate_gate_report(report, 8, self.common)
        ready = copy.deepcopy(self.ready)
        self.replace_gate(ready, 8, report)
        with self.assertRaisesRegex(ValueError, "different native runtimes"):
            self.validate(ready, 1)  # Nonrequested A8 must also join the A1 readiness.

    def test_historical_top_level_current_gates_hybrid_is_rejected(self):
        ready = copy.deepcopy(self.ready)
        ready["native_binary_sha256"] = self.historical["binary"]["sha256"]
        ready["native_runtime"] = self.historical["runtime"]
        with self.assertRaisesRegex(ValueError, "different native runtimes"):
            self.validate(ready)

    def test_manifest_library_directory_and_loader_path_substitutions_rejected(self):
        substitutions = {
            "immutable_manifest": self.historical["runtime"]["immutable_manifest"],
            "libraries": self.historical["runtime"]["libraries"],
            "directory": self.historical["runtime"]["directory"],
            "ld_library_path": self.historical["runtime"]["ld_library_path"],
        }
        for field, value in substitutions.items():
            with self.subTest(field=field):
                report = copy.deepcopy(self.gates["4"])
                report["native_runtime"][field] = value
                gate.validate_gate_report(report, 4, self.common)
                ready = copy.deepcopy(self.ready)
                self.replace_gate(ready, 4, report)
                with self.assertRaisesRegex(ValueError, "different native runtimes"):
                    self.validate(ready, 4)

    def test_missing_runtime_binary_or_library_inventory_cannot_join_by_absence(self):
        for field in ("native_binary_sha256", "native_runtime"):
            ready = copy.deepcopy(self.ready)
            ready.pop(field)
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "full runtime inventory"),
            ):
                self.validate(ready)
        ready = copy.deepcopy(self.ready)
        ready["native_runtime"]["libraries"] = []
        with self.assertRaisesRegex(ValueError, "full runtime inventory"):
            self.validate(ready)
        ready = copy.deepcopy(self.ready)
        report = copy.deepcopy(self.gates["1"])
        report.pop("native_runtime")
        self.replace_gate(ready, 1, report)
        with self.assertRaisesRegex(ValueError, "different native runtimes"):
            self.validate(ready)

    def test_matching_but_replaced_runtime_file_bytes_fail_hash_validation(self):
        for record in [
            self.current["runtime"]["immutable_manifest"],
            self.current["runtime"]["libraries"][0],
        ]:
            path = Path(record["path"])
            before = path.read_bytes()
            path.write_bytes(b"replaced after readiness publication")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.validate()
            path.write_bytes(before)

    def test_v1_historical_readiness_has_no_new_cross_runtime_constraint(self):
        fixture = legacy.ReadinessTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        old = {
            **self.ready,
            "schema": stages.READINESS_SCHEMA,
            "common_source_sha256": fixture.common,
            "native_binary_sha256": self.historical["binary"]["sha256"],
            "native_runtime": self.historical["runtime"],
            "precisions": {
                str(bits): self.write("legacy-" + str(bits) + ".json", fixture.report(bits))
                for bits in (8, 1)
            },
        }
        # Original v1 numeric/cache gates keep their own runtime evidence. The
        # v2 join must not impose a current actor/server runtime on this path.
        self.assertEqual(self.validate(old, 8), old)
        old.pop("native_binary_sha256")
        old.pop("native_runtime")
        self.assertEqual(self.validate(old, 1), old)


if __name__ == "__main__":
    unittest.main()

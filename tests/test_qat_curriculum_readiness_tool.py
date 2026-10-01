"""CPU boundary mocks and tiny real autograd; no native/GPU/training evidence."""

import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "curriculum_readiness_tool", ROOT / "scripts/check_curriculum_qat_readiness.py"
)
tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tool)


def spec():
    return {
        "schema": "qat_curriculum_experiment_v1",
        "curriculum": {
            "stages": [
                {"activation_bits": b, "gpu_seconds": 1000, "max_updates": 1} for b in (8, 4, 1)
            ]
        },
        "qat": {"device": "cuda:0", "allow_accelerator": True, "contract": {"activation_bits": 8}},
        "hardware": {"device_name": "CPU_TEST_CONTEXT_STUB", "compute_capability": [12, 0]},
        "native": {"expected_commit": "a" * 40},
    }


class CurriculumReadinessToolTests(unittest.TestCase):
    def test_default_plan_never_imports_runtime_models_or_providers(self):
        with (
            mock.patch.object(tool, "runtime_api", side_effect=AssertionError("GPU/model import")),
            mock.patch.object(tool, "run_cuda", side_effect=AssertionError("GPU query")),
            contextlib.redirect_stdout(io.StringIO()) as stream,
        ):
            self.assertEqual(tool.main([]), 0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["evidence_device_type"], "cpu")
        self.assertFalse(result["receipt_emitted"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["cuda_discovered"])
        self.assertEqual(result["activation_bits"], [8, 4, 1])

    def test_explicit_cuda_inputs_and_group_order_required(self):
        for argv in (
            ["--allow-cuda"],
            ["--groups", "2"],
            ["--groups", "1", "1"],
            ["--groups", "1", "4", "2"],
        ):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                tool.main(argv)

    def test_config_plan_and_unique_output_preserve_existing_attempts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            config = path / "config.json"
            config.write_text(json.dumps(spec()))
            output = path / "output"
            with (
                mock.patch.object(tool, "runtime_api", side_effect=AssertionError("runtime")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(tool.main(["--config", str(config), "--output", str(output)]), 0)
            self.assertTrue((output / "plan.json").is_file())
            self.assertFalse((output / "readiness.json").exists())
            with self.assertRaises(FileExistsError):
                tool.main(["--output", str(output)])

    def test_invalid_precision_hardware_or_commit_rejected(self):
        import copy

        for kind in ("stages", "hardware", "commit", "permission"):
            value = copy.deepcopy(spec())
            if kind == "stages":
                value["curriculum"]["stages"] = [
                    {"activation_bits": 4, "gpu_seconds": 1, "max_updates": 1}
                ]
            elif kind == "hardware":
                value["hardware"]["compute_capability"] = [True, 0]
            elif kind == "commit":
                value["native"]["expected_commit"] = "short"
            else:
                value["qat"]["allow_accelerator"] = 1
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                tool.validate_spec(value, native_required=True)

    def test_full_model_gate_requires_five_repeats_and_real_later_norms(self):
        record = {
            "forward_calls": 1,
            "backward_calls": 1,
            "later_gradients": [
                {
                    "later_state_gradient_norm": 0.2,
                    "later_k_gradient_norm": 0.3,
                    "later_v_gradient_norm": 0.4,
                }
            ],
        }
        self.assertEqual(tool.full_model_gate([record] * 5)["forward_calls"], 5)
        with self.assertRaises(ValueError):
            tool.full_model_gate([record] * 4)
        record["later_gradients"][0]["later_k_gradient_norm"] = True
        with self.assertRaises(ValueError):
            tool.full_model_gate([record] * 5)

    def test_native_preflight_failure_precedes_runtime_or_model_import(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, evidence = root / "config.json", root / "native.json"
            config.write_text(json.dumps(spec()))
            evidence.write_text("{}")
            args = SimpleNamespace(config=config, native_evidence=evidence, groups=[1])
            with (
                mock.patch.object(
                    tool.common,
                    "preflight_native",
                    side_effect=ValueError("native evidence missing"),
                ) as preflight,
                mock.patch.object(
                    tool, "runtime_api", side_effect=AssertionError("runtime loaded")
                ),
                self.assertRaisesRegex(ValueError, "native evidence missing"),
            ):
                tool.run_cuda(args, root)
            self.assertEqual(preflight.call_args.kwargs["expected_lanes"], ("A8", "A4", "A1"))

    def test_stale_binding_rejected_before_full_model_construction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, evidence = root / "config.json", root / "native.json"
            config.write_text(json.dumps(spec()))
            evidence.write_text(json.dumps({"source_sha256": "stale"}))
            provider = SimpleNamespace(source_metadata={"split": "train"}, allowed_prompt_ids={"p"})
            qat = SimpleNamespace(device="cuda:0")
            api = SimpleNamespace(
                load_config=lambda _: (spec(), None, qat, None),
                provider_factory=lambda _: lambda _: provider,
                validate_readiness=lambda *_: {"source": provider.source_metadata},
                native_checkout_commit=lambda: "a" * 40,
                CurriculumRunner=mock.Mock(side_effect=AssertionError("model construction")),
            )
            args = SimpleNamespace(config=config, native_evidence=evidence, groups=[1])
            value = spec()
            value["provider"] = {
                "factory": "CPU_TEST_FACTORY_STUB",
                "manifest": "CPU_TEST_MANIFEST",
            }
            config.write_text(json.dumps(value))
            binding = {key: {} for key in tool.common.BINDINGS}
            binding["source_sha256"] = "a" * 64
            with (
                mock.patch.object(tool.common, "preflight_native"),
                mock.patch.object(tool, "runtime_api", return_value=api),
                mock.patch.object(tool.common, "cuda_environment", return_value={}),
                mock.patch.object(tool, "runtime_binding", return_value=(binding, {})),
                self.assertRaisesRegex(ValueError, "before model load"),
            ):
                tool.run_cuda(args, root)
            api.CurriculumRunner.assert_not_called()

    def test_tiny_fixture_rejected_by_production_shape_admission(self):
        from test_qat_curriculum_runner import make

        with tempfile.TemporaryDirectory() as folder:
            runner = make(Path(folder))
            with self.assertRaisesRegex(ValueError, "tiny fixture"):
                tool.common.validate_model_shapes([tool.current_lane(runner)])

    def test_actual_cpu_all_stage_gradients_state_restoration_and_step_guard(self):
        import torch
        from test_qat_curriculum_runner import make

        from w1a1_eagle.continuous_qat import restore_rng
        from w1a1_eagle.qat_curriculum_runner import _cpu_tree
        from w1a1_eagle.qat_state import deployment_state_sha256

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            runner = make(output / "fixture", recipe=True, affine=True)
            before = tool.common.deterministic_state_sha256(runner.linears)
            batch = next(runner.provider.rounds())
            api = SimpleNamespace(
                torch=torch,
                cpu_tree=_cpu_tree,
                deployment_state_sha256=deployment_state_sha256,
                restore_rng=restore_rng,
            )
            evidence = {"lanes": {f"A{b}": {} for b in (8, 4, 1)}}
            calls = []

            def cpu_measure(_api, _trainer, lane, rounds, flags):
                report = runner._smoke_round(rounds[0][0])
                calls.append((lane.name, flags))
                memory = {"peak_allocated_bytes": 1, "peak_reserved_bytes": 2, "free_bytes": 3}
                return {
                    "group_size": len(rounds),
                    "forward_calls": len(rounds),
                    "backward_calls": 1,
                    "later_gradients": [report],
                    "memory_before": memory,
                    "memory_after": memory,
                }, []

            with (
                tool.forbid_optimizer_updates(torch),
                mock.patch.object(
                    torch.cuda, "is_available", side_effect=AssertionError("CUDA discovery")
                ),
                mock.patch.object(tool.common, "validate_model_shapes"),
                mock.patch.object(
                    tool.common,
                    "validate_native_evidence",
                    return_value={"CPU_TEST_NATIVE_STUB": True},
                ),
                mock.patch.object(tool.common, "measure_lane", side_effect=cpu_measure),
            ):
                report = runner.prepare(batch)
                self.assertEqual([r["activation_bits"] for r in report], [8, 4, 1])
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    runner.optimizer.step()
                records, stages, skipped, states, _, initial = tool.measure_stages(
                    api,
                    runner,
                    runner.curriculum,
                    [(batch, None)],
                    evidence,
                    output,
                    {},
                    output,
                    [1],
                )
            self.assertEqual(initial, before)
            self.assertEqual(tool.common.deterministic_state_sha256(runner.linears), before)
            self.assertEqual([s["activation_bits"] for s in stages], [8, 4, 1])
            self.assertEqual(set(states), {"A8", "A4", "A1"})
            self.assertEqual(len(records), 15)
            self.assertEqual(len(calls), 18)  # One warmup + five repeats per stage.
            self.assertEqual(skipped, [])
            self.assertEqual(runner.optimizer.state, {})
            self.assertEqual(runner.state.global_updates, 0)

    def test_resident_state_layout_adam_and_sgd_never_steps(self):
        import torch

        parameter = torch.nn.Parameter(torch.ones(3))
        for optimizer, count in (
            (torch.optim.AdamW([parameter]), 2),
            (torch.optim.SGD([parameter], lr=0.01), 1),
        ):
            with tool.resident_optimizer_buffers(SimpleNamespace(torch=torch), optimizer) as info:
                self.assertEqual(info["buffers_per_parameter"], count)
                self.assertEqual(info["resident_bytes"], 12 * count)
                self.assertEqual(optimizer.state, {})

    def test_validator_rejection_never_publishes_receipt_or_locator(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            api = SimpleNamespace(
                validate_optimization_readiness=mock.Mock(side_effect=ValueError("gate failed"))
            )
            config = {"device": "cuda:0"}
            runner_config = SimpleNamespace(max_cuda_reserved_bytes=10, min_cuda_free_bytes=1)
            binding = {
                "source_sha256": "a" * 64,
                "training_runtime": {},
                "native_commit": "b" * 40,
                "hardware": {},
            }
            with self.assertRaisesRegex(ValueError, "gate failed"):
                tool.publish_receipt(
                    api, config, runner_config, binding, {"CPU_FIXTURE_ONLY": True}, output
                )
            self.assertTrue((output / "readiness-candidate.json").exists())
            self.assertFalse((output / "readiness.json").exists())
            self.assertFalse((output / "receipt-locator.json").exists())


if __name__ == "__main__":
    unittest.main()

"""CPU-only harness contracts; never discover CUDA, load weights or read captures."""

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

TOOL_PATH = Path(__file__).resolve().parents[1] / "scripts/check_qat_optimization_readiness.py"
spec = importlib.util.spec_from_file_location("qat_readiness_tool", TOOL_PATH)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def config_spec():
    return {
        "schema": "continuous_w1ax_experiment_v1",
        "training": {"device": "cuda:0"},
        "model_shapes": list(tool.PINNED_SHAPES.values()),
        "native": {"expected_commit": "c" * 40},
        "hardware": {"device_name": "expected actual GPU", "compute_capability": [12, 0]},
    }


def binding(learned=False, fusion=False, affine=False, coverage="fusion"):
    return {
        "source_sha256": "1" * 64,
        "training_runtime": {"device_type": "cuda", "fixture": True},
        "recipe": {
            "activation_quantization": "learned" if learned else "fixed",
            "fusion_correction": {"enabled": fusion},
            "affine_weights": {"enabled": affine, "coverage": coverage},
        },
        "native_commit": "c" * 40,
        "backend": "cuda",
        "hardware": {
            "device_type": "cuda",
            "name": "fixture hardware metadata only",
            "compute_capability": [12, 0],
            "total_memory_bytes": 16 * 1024**3,
        },
    }


def native_fixture(directory, *, learned=False, fusion=False, affine=False, coverage="fusion"):
    """Metadata validator fixtures only; not a receipt or GPU evidence."""
    bound = binding(learned, fusion, affine, coverage)
    states = {"A8": "8" * 64, "A1": "a" * 64}
    evidence = {
        "schema": tool.NATIVE_SCHEMA,
        "split": "train",
        "fixture_only": False,
        "optimizer_updates": 0,
        **bound,
        "lanes": {},
    }
    for lane, state in states.items():
        lane_data = {"deployment_state_sha256": state}
        common = {
            "schema": tool.MEASUREMENT_SCHEMA,
            "split": "train",
            "activation_bits": int(lane[1:]),
            "deployment_state_sha256": state,
            **bound,
        }
        ancestry = {
            "prompt_id": "eligible-train",
            "round_index": 1,
            "capture_id": "fixture-capture",
        }
        kinds = {
            "native_decisions": [
                {
                    **ancestry,
                    "state_relative_rms": 0.01,
                    "logit_relative_rms": 0.02,
                    "torch_choice": 3,
                    "native_choice": 3,
                    "native_margin": 0.04,
                }
            ]
        }
        if learned:
            kinds["learned_quantizers"] = [
                {
                    **ancestry,
                    "boundary": boundary,
                    "packed_expected_sha256": "e" * 64,
                    "packed_native_sha256": "e" * 64,
                    "native_output_relative_rms": 0.01,
                }
                for boundary in sorted(tool.BOUNDARIES)
            ]
        if fusion:
            kinds["fusion_correction"] = [
                {
                    **ancestry,
                    "raw_input_sha256": "d" * 64,
                    "native_raw_input_sha256": "d" * 64,
                    "zero_identity_max_abs": 0,
                    "expected_correction_delta_norm": 0.03,
                    "native_correction_delta_norm": 0.03,
                    "nonzero_forward_relative_rms": 0.01,
                }
            ]
        if affine:
            bases = tool.AFFINE_BASES if coverage == "all" else {"fc"}
            kinds["affine_weights"] = [
                {
                    **ancestry,
                    "fixture": fixture,
                    "projection_base": base,
                    "expected_code_sum_sha256": "3" * 64,
                    "native_code_sum_sha256": "3" * 64,
                    "expected_output_sign_sha256": "4" * 64,
                    "native_output_sign_sha256": "4" * 64,
                    "expected_tail_sha256": "5" * 64,
                    "native_tail_sha256": "5" * 64,
                    "native_output_relative_rms": 1e-6,
                    "identity_max_abs": 0,
                    "alpha": 0.0,
                    "mu": 0.0 if fixture == "mu_zero_identity" else -0.02,
                    "expected_output_norm": 0.5,
                    "native_output_norm": 0.5,
                }
                for fixture in sorted(tool.AFFINE_FIXTURES)
                for base in sorted(bases)
            ]
        for kind, cases in kinds.items():
            path = Path(directory) / (lane + "-" + kind + ".json")
            extra = {"projection_bases": sorted(bases)} if kind == "affine_weights" else {}
            tool.write_new(path, {**common, "kind": kind, "cases": cases, **extra})
            lane_data[kind] = {"path": str(path), "sha256": tool.sha256(path)}
        evidence["lanes"][lane] = lane_data
    return evidence, bound, states


class ReadinessToolTests(unittest.TestCase):
    def test_default_plan_never_imports_provider_or_discovers_cuda(self):
        with (
            patch.object(tool, "runtime_api", side_effect=AssertionError("runtime import")),
            patch.object(
                tool.importlib, "import_module", side_effect=AssertionError("provider import")
            ),
            patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA discovery")),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(tool.main([]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["schema"], "qat_optimization_readiness_plan_v1")
        self.assertFalse(result["passed"])
        self.assertFalse(result["receipt_emitted"])
        self.assertFalse(result["model_loaded"])

    def test_explicit_cuda_requires_all_inputs_and_missing_native_fails_before_runtime(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            tool.main(["--allow-cuda"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.json"
            tool.write_new(config, config_spec())
            native = root / "native.json"
            tool.write_new(native, {"schema": "CPU invented flags", "passed": True})
            output = root / "new-run"
            args = [
                "--allow-cuda",
                "--provider",
                "fixture:factory",
                "--provider-manifest",
                str(root / "manifest"),
                "--config",
                str(config),
                "--native-evidence",
                str(native),
                "--output",
                str(output),
            ]
            with (
                patch.object(
                    tool, "runtime_api", side_effect=AssertionError("should not reach CUDA")
                ),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(tool.main(args), 1)
            self.assertTrue((output / "failure.json").is_file())
            self.assertFalse((output / "readiness.json").exists())

    def test_output_is_unique_and_json_duplicate_nonfinite_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-plan"
            with contextlib.redirect_stdout(io.StringIO()):
                tool.main(["--output", str(output)])
            original = (output / "plan.json").read_bytes()
            with self.assertRaises(FileExistsError):
                tool.main(["--output", str(output)])
            self.assertEqual(original, (output / "plan.json").read_bytes())
            self.assertFalse((output / "readiness.json").exists())
            for data in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
                path = Path(directory) / "bad.json"
                path.write_text(data)
                with self.assertRaises(ValueError):
                    tool.read_json(path)

    def test_full_model_claim_rejects_tiny_config_and_cpu_masters(self):
        spec = config_spec()
        spec["model_shapes"][0] = [4, 12]
        with self.assertRaisesRegex(ValueError, "full-model"):
            tool.validate_config_spec(spec)
        with self.assertRaisesRegex(ValueError, "projection inventory"):
            tool.validate_model_shapes([SimpleNamespace(linears={"fc": nn.Linear(12, 4)})])
        # CPU tensors cannot become a CUDA full-model claim by metadata flags.
        modules = {
            name: SimpleNamespace(
                out_features=shape[0],
                in_features=shape[1],
                latent_sign=nn.Parameter(torch.ones(1)),
                scale_offset=nn.Parameter(torch.ones(1)),
            )
            for name, shape in tool.PINNED_SHAPES.items()
        }
        with self.assertRaisesRegex(ValueError, "actual CUDA"):
            tool.validate_model_shapes([SimpleNamespace(linears=modules)])

    def test_state_hash_chunking_alias_order_and_changes(self):
        module = nn.Linear(7, 3)
        module.scalar = nn.Parameter(torch.tensor(0.5))
        first = {"z": module, "a": nn.Linear(3, 2)}
        second = dict(reversed(list(first.items())))
        one = tool.deterministic_state_sha256(first, chunk_elements=1)
        self.assertEqual(one, tool.deterministic_state_sha256(second, chunk_elements=20))
        with torch.no_grad():
            module.scalar.add_(0.01)
        self.assertNotEqual(one, tool.deterministic_state_sha256(first))

    def test_optimizer_step_is_protected_without_any_update(self):
        parameter = nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW([parameter])
        lane = SimpleNamespace(optimizer=optimizer)
        original = parameter.detach().clone()
        with tool.zero_updates([lane]):
            tool.require_zero_progress([lane])
            with self.assertRaisesRegex(RuntimeError, "forbidden"):
                optimizer.step()
        self.assertTrue(torch.equal(original, parameter))
        optimizer.state[parameter]["step"] = torch.tensor(1.0)
        with self.assertRaisesRegex(ValueError, "optimizer progress"):
            tool.require_zero_progress([lane])

    def test_native_case_derivation_and_optional_gates_need_real_artifact_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence, bound, states = native_fixture(directory, learned=True, fusion=True)
            gates = tool.validate_native_evidence(
                evidence, directory, bound, states, {"eligible-train"}
            )
            self.assertEqual(gates["native_decisions"]["cases"], 2)
            self.assertTrue(gates["learned_quantizers"]["exact_pack"])
            self.assertTrue(gates["fusion_correction"]["raw_input_ancestry_passed"])
            bad = copy.deepcopy(evidence)
            del bad["lanes"]["A1"]["learned_quantizers"]
            with self.assertRaisesRegex(ValueError, "missing actual"):
                tool.validate_native_evidence(bad, directory, bound, states, {"eligible-train"})
            bad = copy.deepcopy(evidence)
            bad["fixture_only"] = True
            with self.assertRaisesRegex(ValueError, "real zero-update"):
                tool.validate_native_evidence(bad, directory, bound, states, {"eligible-train"})

    def test_native_source_checkpoint_hash_and_margin_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence, bound, states = native_fixture(directory)
            for field in ("source_sha256", "native_commit", "backend", "hardware", "recipe"):
                bad = copy.deepcopy(evidence)
                bad[field] = "wrong"
                with self.assertRaisesRegex(ValueError, "binding differs"):
                    tool.validate_native_evidence(bad, directory, bound, states, {"eligible-train"})
            with self.assertRaisesRegex(ValueError, "checkpoint differs"):
                tool.validate_native_evidence(
                    evidence,
                    directory,
                    bound,
                    {"A8": "b" * 64, "A1": states["A1"]},
                    {"eligible-train"},
                )
            with self.assertRaisesRegex(ValueError, "outside.*train"):
                tool.validate_native_evidence(evidence, directory, bound, states, {"final-only"})
            record = evidence["lanes"]["A8"]["native_decisions"]
            path = Path(record["path"])
            measurement = tool.read_json(path)
            measurement["cases"][0]["torch_choice"] = 4
            path.write_text(json.dumps(measurement))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                tool.validate_native_evidence(
                    evidence, directory, bound, states, {"eligible-train"}
                )
            record["sha256"] = tool.sha256(path)
            with self.assertRaisesRegex(ValueError, "material native decision"):
                tool.validate_native_evidence(
                    evidence, directory, bound, states, {"eligible-train"}
                )

    def test_native_learned_pack_boundary_and_fusion_raw_ancestry_failures(self):
        for kind, field, value, error in (
            ("learned_quantizers", "packed_native_sha256", "b" * 64, "packed codes differ"),
            ("fusion_correction", "native_raw_input_sha256", "b" * 64, "raw-input ancestry"),
            ("fusion_correction", "zero_identity_max_abs", 0.001, "zero identity"),
            ("fusion_correction", "native_correction_delta_norm", 0, "positive native delta"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                evidence, bound, states = native_fixture(directory, learned=True, fusion=True)
                record = evidence["lanes"]["A8"][kind]
                path = Path(record["path"])
                measurement = tool.read_json(path)
                measurement["cases"][0][field] = value
                path.write_text(json.dumps(measurement))
                record["sha256"] = tool.sha256(path)
                with self.assertRaisesRegex(ValueError, error):
                    tool.validate_native_evidence(
                        evidence, directory, bound, states, {"eligible-train"}
                    )

    def test_affine_all_projection_coverage_exact_sum_and_nonzero_mu_are_required(self):
        for field, value, reason in (
            ("native_code_sum_sha256", "f" * 64, "code_sum differs"),
            ("native_output_sign_sha256", "f" * 64, "output_sign differs"),
            ("native_tail_sha256", "f" * 64, "tail differs"),
            ("native_output_norm", 0, "positive native output"),
            ("native_output_relative_rms", 0.001, "numeric gate"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                evidence, bound, states = native_fixture(directory, affine=True, coverage="all")
                gates = tool.validate_native_evidence(
                    evidence, directory, bound, states, {"eligible-train"}
                )
                self.assertEqual(
                    set(gates["affine_weights"]["projection_bases"]), tool.AFFINE_BASES
                )
                self.assertEqual(gates["affine_weights"]["cases"], 54)
                record = evidence["lanes"]["A8"]["affine_weights"]
                path = Path(record["path"])
                measured = tool.read_json(path)
                row = next(c for c in measured["cases"] if c["fixture"] == "alpha_zero_nonzero_mu")
                row[field] = value
                path.write_text(json.dumps(measured))
                record["sha256"] = tool.sha256(path)
                with self.assertRaisesRegex(ValueError, reason):
                    tool.validate_native_evidence(
                        evidence, directory, bound, states, {"eligible-train"}
                    )
        with tempfile.TemporaryDirectory() as directory:
            evidence, bound, states = native_fixture(directory, affine=True, coverage="all")
            record = evidence["lanes"]["A8"]["affine_weights"]
            path = Path(record["path"])
            measured = tool.read_json(path)
            measured["cases"] = [c for c in measured["cases"] if c["projection_base"] == "fc"]
            path.write_text(json.dumps(measured))
            record["sha256"] = tool.sha256(path)
            with self.assertRaisesRegex(ValueError, "all selected projection"):
                tool.validate_native_evidence(
                    evidence, directory, bound, states, {"eligible-train"}
                )

    def test_mocked_factory_requires_eligible_train_bound_manifest_and_equal_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            manifest.write_text("{}")
            source = {"split": "train", "provider_manifest_sha256": tool.sha256(manifest)}
            provider = SimpleNamespace(
                training_eligible=True,
                full_body_qat_eligible=True,
                split="train",
                data_split="train",
                readiness_scope="full_body_qat",
                allowed_prompt_ids={"train"},
                source_metadata=source,
            )
            config = SimpleNamespace(qat=lambda bits: {"bits": bits})
            calls = []

            def creator(lane_config, path):
                calls.append((lane_config, path))
                return provider

            with patch.object(
                tool.importlib, "import_module", return_value=SimpleNamespace(create=creator)
            ):
                self.assertIs(tool.make_provider("mock:create", manifest, config), provider)
            self.assertEqual([c[0]["bits"] for c in calls], [8, 1])
            for field, bad in (
                ("data_split", "sealed_final"),
                ("training_eligible", False),
                ("readiness_scope", "row_a16_hard_ce_100_steps"),
            ):
                previous = getattr(provider, field)
                setattr(provider, field, bad)
                with (
                    patch.object(
                        tool.importlib,
                        "import_module",
                        return_value=SimpleNamespace(create=creator),
                    ),
                    self.assertRaisesRegex(ValueError, "calibration/finals"),
                ):
                    tool.make_provider("mock:create", manifest, config)
                setattr(provider, field, previous)

    def test_cuda_boundary_uses_only_explicit_fake_and_enforces_hardware_and_math(self):
        fake = SimpleNamespace(
            device=lambda device: SimpleNamespace(type="cuda"),
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda _: SimpleNamespace(
                    major=12, minor=0, total_memory=16 * 1024**3
                ),
                get_device_name=lambda _: "expected actual GPU",
            ),
            set_float32_matmul_precision=lambda value: self.assertEqual(value, "highest"),
            backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                cudnn=SimpleNamespace(allow_tf32=True),
            ),
        )
        result = tool.cuda_environment(
            SimpleNamespace(torch=fake), SimpleNamespace(device="cuda:0"), config_spec()
        )
        self.assertEqual(result["compute_capability"], [12, 0])
        self.assertFalse(fake.backends.cuda.matmul.allow_tf32)
        self.assertFalse(fake.backends.cudnn.allow_tf32)
        fake.cuda.get_device_name = lambda _: "different hardware"
        with self.assertRaisesRegex(ValueError, "actual CUDA hardware"):
            tool.cuda_environment(
                SimpleNamespace(torch=fake), SimpleNamespace(device="cuda:0"), config_spec()
            )

    def test_memory_ceiling_and_free_floor_are_measured_and_not_fit_claims(self):
        fake = SimpleNamespace(
            cuda=SimpleNamespace(
                mem_get_info=lambda _: (1024, 4096),
                max_memory_allocated=lambda _: 512,
                max_memory_reserved=lambda _: 1024,
                memory_allocated=lambda _: 256,
                memory_reserved=lambda _: 1024,
            )
        )
        config = SimpleNamespace(
            device="cuda:0", max_cuda_reserved_bytes=2048, min_cuda_free_bytes=512
        )
        self.assertEqual(
            tool.memory_snapshot(SimpleNamespace(torch=fake), config)["free_bytes"], 1024
        )
        config.min_cuda_free_bytes = 2048
        with self.assertRaises(tool.MemoryAdmissionError) as caught:
            tool.memory_snapshot(SimpleNamespace(torch=fake), config)
        self.assertEqual(caught.exception.measurement["free_bytes"], 1024)

    def test_failed_consumer_validation_preserves_candidate_without_publishing_locator(self):
        @dataclass
        class Config:
            device: str = "cuda:0"
            max_cuda_reserved_bytes: int = 12 * 1024**3
            min_cuda_free_bytes: int = 1024**3
            optimization_readiness: object = None

        api = SimpleNamespace(
            validate_optimization_readiness=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("consumer rejected actual receipt")
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, "consumer rejected"):
                tool.publish_receipt(
                    api, Config(), binding(), {"schema": "fixture candidate only"}, output
                )
            self.assertTrue((output / "readiness-candidate.json").exists())
            self.assertFalse((output / "readiness.json").exists())
            self.assertFalse((output / "receipt-locator.json").exists())

    def test_measurement_retains_later_graphs_but_never_updates_cpu_fixture(self):
        # Explicit fake CUDA instrumentation around a tiny CPU graph. This
        # function result is not a published receipt or full-model evidence.
        sys.path.insert(0, str(Path(__file__).parent))
        from test_qat_cache_head import audit, batch, drafter

        from w1a1_eagle.continuous_qat import ObservedAdapter, later_gradient
        from w1a1_eagle.native_step import NativeStepAdapter
        from w1a1_eagle.recurrent_loss import supported_prefix_ce
        from w1a1_eagle.recurrent_provider import forward_torch_round
        from w1a1_eagle.recurrent_qat import shared_round_hard_signs

        adapter = NativeStepAdapter(drafter(8))
        parameters = [p for m in adapter.linears.values() for p in (m.latent_sign, m.scale_offset)]
        optimizer = torch.optim.AdamW(parameters)
        lane = SimpleNamespace(
            name="A8", linears=adapter.linears, adapter=adapter, optimizer=optimizer
        )
        config = SimpleNamespace(
            device="cpu",
            optimize_cache=True,
            optimize_head=True,
            context_chunk_size=2,
            depth_loss_decay=1.0,
            max_cuda_reserved_bytes=1000,
            min_cuda_free_bytes=1,
        )
        trainer = SimpleNamespace(
            config=config,
            lanes=[lane],
            provider=SimpleNamespace(draft_vocab_size=7),
            resources=lambda: {"fixture_only": True},
        )
        fake_torch = SimpleNamespace(
            stack=torch.stack,
            isfinite=torch.isfinite,
            cuda=SimpleNamespace(
                reset_peak_memory_stats=lambda _: None, synchronize=lambda _: None
            ),
        )
        api = SimpleNamespace(
            torch=fake_torch,
            shared_round_hard_signs=shared_round_hard_signs,
            ObservedAdapter=ObservedAdapter,
            forward_torch_round=forward_torch_round,
            later_gradient=later_gradient,
            depth_weighted_supported_ce=lambda logits, a, _depths, decay: supported_prefix_ce(
                logits, a
            ),
            joint_parameter_families=lambda modules: {
                "sign": [m.latent_sign for m in modules.values()],
                "scale": [m.scale_offset for m in modules.values()],
                "activation": [],
                "fusion": [],
            },
        )
        memory = {
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
            "free_bytes": 800,
            "total_bytes": 1000,
        }
        state = tool.deterministic_state_sha256(adapter.linears)
        with (
            tool.zero_updates([lane]),
            patch.object(tool, "memory_snapshot", return_value=memory),
            patch.object(
                torch.cuda, "is_available", side_effect=AssertionError("actual CUDA discovery")
            ),
        ):
            row, outputs = tool.measure_lane(
                api, trainer, lane, [(batch(1), audit()), (batch(7), audit())], (True, True)
            )
        self.assertEqual(row["group_size"], 2)
        self.assertEqual(row["backward_calls"], 1)
        self.assertEqual(row["optimizer_updates"], 0)
        self.assertEqual(len(row["later_gradients"]), 2)
        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(p.grad is None for p in parameters))
        self.assertEqual(state, tool.deterministic_state_sha256(adapter.linears))


if __name__ == "__main__":
    unittest.main()

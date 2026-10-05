"""Torch receipt/live UUID formatting never relaxes physical device equality."""

import json
import sys
import tempfile
import unittest
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import train_nine_model_qat as producer  # noqa: E402

from w1a1_eagle.qat_admission import VerifiedTrainingAdmission, _digest  # noqa: E402

BARE = "44ceb8b5-b67a-a317-fee3-f01c9201994e"
CANONICAL = "GPU-" + BARE
OTHER = "55ceb8b5-b67a-a317-fee3-f01c9201994e"


@dataclass(frozen=True)
class FixtureQAT:
    device: str = "cuda:0"
    optimize_cache: bool = True
    optimize_head: bool = True
    activation_quantization: str = "fixed"

    @property
    def contract(self):
        return SimpleNamespace(activation_bits=8)


class UUIDTests(unittest.TestCase):
    def test_real_bare_prefixed_and_uppercase_uuid_share_one_canonical_value(self):
        for value in (BARE, CANONICAL, "GPU-" + BARE.upper()):
            self.assertEqual(producer.canonical_gpu_uuid(value), CANONICAL)
        for value in (
            "",
            "cuda:0",
            "GPU-44ceb8b5",
            "GPU-GPU-" + BARE,
            "MIG-" + BARE,
            BARE.replace("-", ""),
            BARE + "\n",
            None,
        ):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "full physical"):
                producer.canonical_gpu_uuid(value)

    def test_actual_configure_cuda_emits_canonical_hardware_uuid_from_live_properties(self):
        for value in (BARE, CANONICAL):
            props = SimpleNamespace(uuid=value, major=12, minor=0, name="NVIDIA GeForce RTX 5080")
            with (
                patch.object(producer.torch, "set_float32_matmul_precision"),
                patch.object(producer.torch.cuda, "is_available", return_value=True),
                patch.object(
                    producer.torch.cuda, "get_device_properties", return_value=props
                ) as queried,
                patch.object(producer.torch.cuda, "reset_peak_memory_stats"),
            ):
                hardware = producer.configure_cuda(zero_updates=True)
            queried.assert_called_once_with("cuda:0")
            self.assertEqual(hardware["gpu_uuid"], CANONICAL)
            self.assertEqual(hardware["compute_capability"], [12, 0])
            self.assertFalse(hardware["tf32"])

    def test_real_live_typed_admission_preserves_exact_uuid_equality(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config_path = root / "config.json"
            config_path.write_text("source/config fixture, no models or GPU")
            config_sha = producer.sha256(config_path)
            config = FixtureQAT()
            record = {
                "schema": "nine_model_training_admission_v1",
                "status": "PASS",
                "artifact_kind": "production",
                "candidate": "eagle_a8",
                "bundle_sha256": "a" * 64,
                "config_sha256": config_sha,
                "compute_capability": [12, 0],
                "optimizer_updates": 0,
                "gpu_uuid": CANONICAL,
                "source_files": producer.training_source_identity(),
                "checks": dict.fromkeys(producer.CHECKS, "PASS"),
                "executed_paths": {"8": {"context_cache_calls": 1, "effective_batched": True}},
            }
            path = root / "admission.json"
            path.write_text(json.dumps(record))
            admission = VerifiedTrainingAdmission(
                path,
                producer.sha256(path),
                config_path,
                config_sha,
                "a" * 64,
                "eagle_a8",
                frozenset({_digest(asdict(config))}),
            )
            for value in (BARE, CANONICAL):
                props = SimpleNamespace(uuid=value, major=12, minor=0)
                with patch.object(producer.torch.cuda, "get_device_properties", return_value=props):
                    self.assertEqual(admission.require_for_qat(config)["gpu_uuid"], CANONICAL)
            for value in (OTHER, "GPU-" + OTHER, "cuda:0"):
                props = SimpleNamespace(uuid=value, major=12, minor=0)
                with (
                    patch.object(producer.torch.cuda, "get_device_properties", return_value=props),
                    self.assertRaises(ValueError),
                ):
                    admission.require_for_qat(config)
            # Stored receipts are never reinterpreted or modified by this fix.
            with self.assertRaises(ValueError):
                producer.require_admission(
                    path, "a" * 64, config_sha, candidate="eagle_a8", gpu_uuid=BARE
                )
            self.assertEqual(json.loads(path.read_text()), record)

    def test_backward_consumer_stays_strict_for_canonical_wrong_and_legacy_bare_receipts(self):
        import test_nine_model_sm120_admission as fixtures

        from w1a1_eagle.nine_model_admission import backward_summary

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            plan = fixtures.AdmissionTests().fixture(root)
            candidate = plan["candidates"]["dspark_a8"]
            directory = root / "dspark_a8/backward"
            fixtures.Runner(plan).run([], directory=directory)
            record = json.loads((directory / "receipt.json").read_text())
            record["hardware"]["gpu_uuid"] = producer.canonical_gpu_uuid(BARE)
            backward_summary(record, candidate, "a" * 64, fixture=True, gpu_uuid=CANONICAL)
            for value in (BARE, "GPU-" + OTHER):
                record["hardware"]["gpu_uuid"] = value
                with self.assertRaisesRegex(ValueError, "GPU UUID differs"):
                    backward_summary(record, candidate, "a" * 64, fixture=True, gpu_uuid=CANONICAL)


if __name__ == "__main__":
    unittest.main()

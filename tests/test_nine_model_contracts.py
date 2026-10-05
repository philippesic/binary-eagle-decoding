"""Independent CPU contracts for block data ancestry and fusion arithmetic."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from train_nine_model_qat import load_spec
from w1a1_eagle.block_data import TAPS, BlockCursor, BlockDataset, file_sha256
from w1a1_eagle.block_fusion import FusionFitConfig, fit_fusion, project, quantize
from w1a1_eagle.block_qat import (
    BlockBinaryLinear,
    BlockDrafter,
    BlockQATConfig,
    block_optimizer,
    block_train_step,
)
from w1a1_eagle.block_training import (
    BlockCursor as TrainCursor,
)
from w1a1_eagle.block_training import (
    export_block_checkpoint,
    load_block_checkpoint,
    save_block_checkpoint,
    transition_a8_to_a1,
)

from w1a1_eagle.recurrent_qat import W1AxContract


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(root: Path, name: str, value: dict) -> dict[str, str]:
    path = root / name
    path.write_text(json.dumps(value, sort_keys=True))
    return {"path": name, "sha256": file_sha256(path)}


def _write_array(root: Path, name: str, value: np.ndarray) -> dict[str, str]:
    path = root / name
    np.save(path, value)
    return {"path": name, "sha256": file_sha256(path)}


def _block_fixture(root: Path, *, with_logits: bool = True) -> Path:
    producer = {
        "kind": "synthetic_fixture",
        "native_revision": "a" * 40,
        "binary_sha256": "b" * 64,
        "target_sha256": "c" * 64,
        "target_precision": "F16",
        "runtime_sha256": "d" * 64,
        "hardware": "CPU fixture",
    }
    tokens = np.arange(14, dtype=np.int64)
    features = np.arange(14 * 5 * 3, dtype=np.float32).reshape(14, 5, 3)
    logits = np.arange(14 * 19, dtype=np.float32).reshape(14, 19)
    token_record = _write_array(root, "tokens.npy", tokens)
    feature_record = _write_array(root, "features.npy", features)
    logits_record = _write_array(root, "logits.npy", logits) if with_logits else None
    prompt_hash = _sha(b"TRAIN prompt with fixed ancestry")
    chain = {
        "chain_id": "train-prose-0001",
        "prompt_id": "authentic-train-id",
        "prompt_sha256": prompt_hash,
        "domain": "prose",
        "split": "train",
        "prompt_length": 3,
        "tokens": token_record,
        "features": feature_record,
        "logits": logits_record,
        "anchors": [2, 4],
        "native_receipt": None,
    }
    inventory = _write_json(
        root,
        "inventory.json",
        {
            "schema": "block_train_inventory_v1",
            "prompts": {
                "authentic-train-id": {
                    "sha256": prompt_hash,
                    "domain": "prose",
                    "split": "TRAIN",
                }
            },
        },
    )
    receipt_chains = {
        chain["chain_id"]: {
            "tokens_sha256": token_record["sha256"],
            "features_sha256": feature_record["sha256"],
            "logits_sha256": logits_record["sha256"] if logits_record else None,
            "prompt_id": chain["prompt_id"],
            "prompt_sha256": prompt_hash,
            "prompt_length": chain["prompt_length"],
            "native_receipt_sha256": None,
        }
    }
    receipt = _write_json(
        root,
        "capture-receipt.json",
        {
            "schema": "block_target_capture_receipt_v1",
            "producer": producer,
            "train_inventory_sha256": inventory["sha256"],
            "taps": list(TAPS),
            "tap_semantics": "native_layer_input_f32",
            "vocab_size": 19,
            "capture_mode": "target_only_autoregressive_train",
            "chains": receipt_chains,
        },
    )
    manifest = {
        "schema": "native_block_train_v1",
        "family": "dspark",
        "vocab_size": 19,
        "target_width": 3,
        "mask_token_id": 18,
        "taps": list(TAPS),
        "tap_semantics": "native_layer_input_f32",
        "layout": "author_anchor_first",
        "producer": producer | {"receipt": receipt},
        "train_inventory": inventory,
        "chains": [chain],
    }
    return root / "manifest.json" if _write_json(root, "manifest.json", manifest) else root


def _open_fixture(path: Path, **kwargs) -> BlockDataset:
    return BlockDataset(
        path,
        expected_sha256=file_sha256(path),
        allow_synthetic=True,
        **kwargs,
    )


def _arithmetic_oracle(raw: np.ndarray, signs: np.ndarray, scales: np.ndarray, bits: int):
    """Small scalar-row oracle independent of block_fusion.quantize/project."""
    codes = np.empty(raw.shape, dtype=np.int16)
    beta = np.empty((len(raw),), dtype=np.float32)
    for row_index, row in enumerate(raw):
        if bits == 1:
            beta[row_index] = np.float32(np.mean(np.abs(row).astype(np.float64)))
            codes[row_index] = np.asarray([-1 if x < 0 else 1 for x in row], dtype=np.int16)
        else:
            peak = np.float32(np.max(np.abs(row)))
            beta[row_index] = np.float32(peak / np.float32(127))
            inverse = np.float32(np.float32(127) / peak) if peak else np.float32(0)
            codes[row_index] = np.clip(np.rint(row * inverse), -127, 127).astype(np.int16)
    dots = codes.astype(np.float32) @ signs.T.astype(np.float32)
    output = dots * scales[None, :].astype(np.float32)
    output = output * beta[:, None]
    return codes, beta, output.astype(np.float32)


def _a8_finite_reciprocal_overflow_oracle(raw: np.ndarray):
    """Scalar oracle for v2's finite reciprocal-overflow normalization path."""
    codes = np.empty(raw.shape, dtype=np.int16)
    beta = np.empty((len(raw),), dtype=np.float32)
    denominator = float(np.float32(127))
    for row_index, row in enumerate(raw):
        peak = max(abs(float(value)) for value in row)
        beta[row_index] = np.float32(peak / denominator)
        if peak == 0:
            codes[row_index].fill(0)
            continue
        normalized = np.asarray(
            [np.float32((float(value) / peak) * 127.0) for value in row], dtype=np.float32
        )
        codes[row_index] = np.clip(np.rint(normalized), -127, 127).astype(np.int16)
    return codes, beta


def _model_fixture(*, bits: int = 8, profile: str = "ffn15_fusion"):
    config = BlockQATConfig(
        "dspark",
        bits,
        hidden_size=8,
        intermediate_size=12,
        num_heads=4,
        num_kv_heads=2,
        head_dim=4,
        vocab_size=19,
        mask_token_id=18,
        profile=profile,
        conditioning="captured_prefix",
    )
    generator = torch.Generator().manual_seed(481)
    tensors = {
        "token_embd.weight": torch.randn(19, 8, generator=generator),
        "output.weight": torch.randn(19, 8, generator=generator),
        "fc.weight": torch.randn(8, 40, generator=generator),
        "enc.output_norm.weight": torch.ones(8),
        "output_norm.weight": torch.ones(8),
        "markov_w1.weight": torch.randn(19, 3, generator=generator),
        "markov_w2.weight": torch.randn(19, 3, generator=generator),
    }
    shapes = {
        "attn_norm": (8,),
        "attn_q_norm": (4,),
        "attn_k_norm": (4,),
        "ffn_norm": (8,),
        "attn_q": (16, 8),
        "attn_k": (8, 8),
        "attn_v": (8, 8),
        "attn_output": (8, 16),
        "ffn_gate": (12, 8),
        "ffn_up": (12, 8),
        "ffn_down": (8, 12),
    }
    for layer in range(5):
        for name, shape in shapes.items():
            tensors[f"blk.{layer}.{name}.weight"] = (
                torch.ones(shape)
                if len(shape) == 1
                else torch.randn(shape, generator=generator) * 0.1
            )
    batch = SimpleNamespace(
        context_features=torch.randn(3, 5, 8, generator=generator),
        prefix_tokens=(1, 2, 3, 4),
        input_tokens=torch.tensor([4, 18, 18, 18, 18, 18, 18]),
        positions=torch.arange(3, 10),
        labels=torch.tensor([5, 6, 7, 8, 9, 10, 11]),
        predecessor_ids=torch.tensor([4, 5, 6, 7, 8, 9, 10]),
        loss_mask=torch.ones(7, dtype=torch.bool),
        attention_allowed=torch.ones(7, 10, dtype=torch.bool),
        teacher_logits=torch.randn(7, 19, generator=generator),
    )
    return config, tensors, batch


def _write_block_gguf(root: Path, *, family="dspark", reduced_vocab=False, unsupported=False):
    from gguf import GGUFWriter

    hidden, intermediate, vocab = 8, 12, 19
    if reduced_vocab:
        vocab -= 1
    path = root / f"{family}.gguf"
    writer = GGUFWriter(path, "dflash")
    writer.add_name("bounded source-loader fixture")
    writer.add_block_count(5)
    writer.add_block_size(7)
    writer.add_target_layers([2, 10, 18, 26, 34])
    writer.add_sample_from_anchor(True)
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(intermediate)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_key_length(4)
    writer.add_value_length(4)
    writer.add_rope_dimension_count(4)
    writer.add_rope_freq_base(1_000_000.0)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_mask_token_id(18)
    writer.add_tokenizer_model("llama")
    writer.add_token_list([f"token{index}" for index in range(vocab)])

    generator = np.random.default_rng(121)
    for name in ("token_embd.weight", "output.weight"):
        writer.add_tensor(name, generator.normal(size=(vocab, hidden)).astype(np.float32))
    for name in ("enc.output_norm.weight", "output_norm.weight"):
        writer.add_tensor(name, np.ones(hidden, dtype=np.float32))
    writer.add_tensor("fc.weight", generator.normal(size=(hidden, 5 * hidden)).astype(np.float32))
    shapes = {
        "attn_norm": (hidden,),
        "attn_q_norm": (4,),
        "attn_k_norm": (4,),
        "ffn_norm": (hidden,),
        "attn_q": (16, hidden),
        "attn_k": (8, hidden),
        "attn_v": (8, hidden),
        "attn_output": (hidden, 16),
        "ffn_gate": (intermediate, hidden),
        "ffn_up": (intermediate, hidden),
        "ffn_down": (hidden, intermediate),
    }
    for layer in range(5):
        for name, shape in shapes.items():
            values = (
                np.ones(shape, dtype=np.float32)
                if len(shape) == 1
                else (generator.normal(size=shape).astype(np.float32))
            )
            writer.add_tensor(f"blk.{layer}.{name}.weight", values)
    if family == "dspark":
        writer.add_tensor("markov_w1.weight", np.ones((vocab, 3), dtype=np.float32))
        writer.add_tensor("markov_w2.weight", np.ones((vocab, 3), dtype=np.float32))
        writer.add_tensor("markov_w2.scale", np.asarray(0.25, dtype=np.float32))
    if unsupported:
        writer.add_tensor("selector.weight", np.ones((2, 2), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class BlockDataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = _block_fixture(self.root)

    def test_anchor_prefix_noise_labels_masks_and_full_vocab_teacher_align(self):
        dataset = _open_fixture(self.manifest)
        batch = dataset.load_block("train-prose-0001", 0, require_teacher=True)
        tokens = np.arange(14, dtype=np.int64)
        features = np.arange(14 * 5 * 3, dtype=np.float32).reshape(14, 5, 3)
        logits = np.arange(14 * 19, dtype=np.float32).reshape(14, 19)
        np.testing.assert_array_equal(batch.context_features, features[:2])
        self.assertEqual(batch.prefix_tokens, (0, 1, 2))
        np.testing.assert_array_equal(batch.input_tokens, [2, 18, 18, 18, 18, 18, 18])
        np.testing.assert_array_equal(batch.labels, tokens[3:10])
        np.testing.assert_array_equal(batch.predecessor_ids, tokens[2:9])
        np.testing.assert_array_equal(batch.positions, np.arange(2, 9))
        self.assertEqual(batch.attention_allowed.shape, (7, 9))
        self.assertTrue(batch.attention_allowed.all())
        self.assertTrue(batch.loss_mask.all())
        np.testing.assert_array_equal(batch.teacher_logits, logits[2:9])

    def test_exact_cursor_serialization_and_foreign_dataset_reject(self):
        dataset = _open_fixture(self.manifest)
        batch0, cursor = dataset.next_block(dataset.cursor(seed=27), require_teacher=True)
        restored = BlockCursor(**json.loads(json.dumps(cursor.payload())))
        batch1, next_cursor = dataset.next_block(restored, require_teacher=True)
        self.assertEqual((batch0.chain_id, batch0.block_index), (batch1.chain_id, 0))
        self.assertEqual(batch1.block_index, 1)
        self.assertEqual(next_cursor.epoch, 1)
        wrong = BlockCursor(**(cursor.payload() | {"dataset_sha256": "f" * 64}))
        with self.assertRaisesRegex(ValueError, "identity"):
            dataset.next_block(wrong)

    def test_missing_teacher_and_corrupt_artifact_fail_closed(self):
        no_logits = _block_fixture(self.root, with_logits=False)
        dataset = _open_fixture(no_logits)
        self.assertIsNone(dataset.load_block("train-prose-0001", 0).teacher_logits)
        with self.assertRaisesRegex(ValueError, "absent"):
            dataset.load_block("train-prose-0001", 0, require_teacher=True)
        (self.root / "features.npy").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            _open_fixture(self.manifest)

    def test_synthetic_capture_cannot_admit_production(self):
        with self.assertRaisesRegex(ValueError, "synthetic"):
            BlockDataset(self.manifest, expected_sha256=file_sha256(self.manifest))


class FusionArithmeticContractTests(unittest.TestCase):
    def test_a1_and_a8_projection_matches_independent_arithmetic_oracle(self):
        raw = np.asarray(
            [[-0.0, 0.0, -2.0, 2.0], [0.25, -0.5, 1.0, -1.5]],
            dtype=np.float32,
        )
        signs = np.asarray([[1, -1, 1, -1], [-1, -1, 1, 1]], dtype=np.float32)
        scales = np.asarray([0.5, 1.25], dtype=np.float32)
        for bits in (1, 8):
            with self.subTest(bits=bits):
                expected_codes, expected_beta, expected = _arithmetic_oracle(
                    raw, signs, scales, bits
                )
                actual_codes, actual_beta = quantize(raw, bits)
                np.testing.assert_array_equal(actual_codes, expected_codes)
                np.testing.assert_array_equal(actual_beta, expected_beta)
                np.testing.assert_array_equal(
                    project(actual_codes, actual_beta, signs, scales), expected
                )

    def test_zero_scale_orientation_rescue_is_opt_in_and_improves_a8_a1_objective(self):
        raw = np.asarray([[1, 2, 3, -4], [2, 3, 4, -5]], dtype=np.float32)
        reference = np.ones((1, 4), dtype=np.float32)
        for bits in (1, 8):
            with self.subTest(bits=bits):
                codes, beta = quantize(raw, bits)
                teacher = -project(codes, beta, reference, np.asarray([2.0], dtype=np.float32))
                control = fit_fusion(raw, teacher, reference, FusionFitConfig(bits))
                rescued = fit_fusion(
                    raw,
                    teacher,
                    reference,
                    FusionFitConfig(bits, zero_scale_orientation_rescue=True),
                )
                self.assertEqual(float(control["scale"][0]), 0.0)
                self.assertGreater(float(rescued["scale"][0]), 0.0)
                self.assertLess(
                    rescued["report"]["candidate"]["sse"],
                    control["report"]["candidate"]["sse"],
                )
                self.assertEqual(
                    rescued["report"]["admission"],
                    "requires_independent_native_trajectory_and_quality_gates",
                )

    def test_invalid_fusion_values_and_profiles_refuse(self):
        with self.assertRaises(ValueError):
            FusionFitConfig(4)
        for invalid in (np.nan, np.inf, -np.inf):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                quantize(np.asarray([[invalid]], dtype=np.float32), 8)

    def test_tiny_a8_finite_reciprocal_overflow_matches_scalar_oracle(self):
        raw = np.array(
            [
                [1, 3, 254, 0x80000003, 0x80000000],
                [0x007FFFFF, 0x807FFFFF, 0, 0, 0],
            ],
            dtype=np.uint32,
        ).view(np.float32)
        expected_codes, expected_beta = _a8_finite_reciprocal_overflow_oracle(raw)
        actual_codes, actual_beta = quantize(raw, 8)
        np.testing.assert_array_equal(actual_codes, expected_codes)
        np.testing.assert_array_equal(actual_beta.view(np.uint32), expected_beta.view(np.uint32))
        np.testing.assert_array_equal(actual_codes[0], [0, 2, 127, -2, 0])
        np.testing.assert_array_equal(actual_codes[1], [127, -127, 0, 0, 0])


class BlockQATContractTests(unittest.TestCase):
    def test_source_gguf_loader_pins_geometry_vocabulary_and_private_tensors(self):
        from w1a1_eagle.block_training import load_block_gguf

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for family in ("dspark", "dflash"):
                path = _write_block_gguf(root, family=family)
                for bits in (1, 8):
                    config = BlockQATConfig(
                        family,
                        bits,
                        hidden_size=8,
                        intermediate_size=12,
                        num_heads=4,
                        num_kv_heads=2,
                        head_dim=4,
                        vocab_size=19,
                        mask_token_id=18,
                    )
                    tensors = load_block_gguf(path, file_sha256(path), config)
                    model = BlockDrafter(tensors, config)
                    self.assertEqual(tuple(model.token_embd.shape), (19, 8))
                    self.assertEqual(tuple(model.output.shape), (19, 8))
                    self.assertEqual(len(model.binary_linears()), 16)
                    if family == "dspark":
                        self.assertEqual(tuple(model.markov_w1.shape), (19, 3))
                    else:
                        self.assertFalse(hasattr(model, "markov_w1"))
                    self.assertNotEqual(
                        model.token_embd.data_ptr(), tensors["token_embd.weight"].data_ptr()
                    )
            dflash = BlockQATConfig(
                "dflash",
                8,
                hidden_size=8,
                intermediate_size=12,
                num_heads=4,
                num_kv_heads=2,
                head_dim=4,
                vocab_size=19,
                mask_token_id=18,
            )
            wrong_vocab = _write_block_gguf(root, family="dflash", reduced_vocab=True)
            tensors = load_block_gguf(wrong_vocab, file_sha256(wrong_vocab), dflash)
            with self.assertRaisesRegex(ValueError, "source operand"):
                BlockDrafter(tensors, dflash)
            wrong_hash = _write_block_gguf(root, family="dflash", unsupported=True)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_block_gguf(wrong_hash, file_sha256(wrong_hash), dflash)
            with self.assertRaisesRegex(ValueError, "hash"):
                load_block_gguf(wrong_hash, "0" * 64, dflash)

    def test_launcher_parses_bounded_direct_and_a8_to_a1_profiles(self):
        config, _, _ = _model_fixture(bits=1)
        base = {
            "schema": "nine_model_qat_training_v1",
            "family": "dspark",
            "candidate": "dspark_a1",
            "device": "cuda:0",
            "qat": asdict(config),
            "limits": {"max_steps": 3, "max_seconds": 5.0},
            "checkpoint_every": 1,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            path.write_text(json.dumps(base))
            self.assertEqual(load_spec(path)["candidate"], "dspark_a1")

            transitioned = dict(base)
            transitioned["candidate"] = "dspark_a1"
            transitioned["precision_stage"] = "a8_to_a1"
            transitioned["a8_warmup_steps"] = 2
            path.write_text(json.dumps(transitioned))
            self.assertEqual(load_spec(path)["precision_stage"], "a8_to_a1")

            invalid = dict(transitioned)
            invalid["qat"] = asdict(replace(config, activation_bits=8))
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "final A1"):
                load_spec(path)

            invalid = dict(base)
            invalid["limits"] = {"max_steps": True}
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "training limit"):
                load_spec(path)

    def test_a8_to_a1_config_requires_nonzero_final_precision_budget_and_quotas(self):
        config, _, _ = _model_fixture(bits=1)
        spec = {
            "schema": "nine_model_qat_training_v1",
            "family": "dspark",
            "candidate": "dspark_a1",
            "device": "cuda:0",
            "qat": asdict(config),
            "precision_stage": "a8_to_a1",
            "a8_warmup_steps": 2,
            "limits": {"max_steps": 3, "max_seconds": 5.0},
            "checkpoint_every": 1,
            "min_a1_updates": 1,
            "min_a1_supervised_tokens": 1,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "profile.json"
            path.write_text(json.dumps(spec))
            self.assertEqual(load_spec(path)["min_a1_updates"], 1)

            invalid = dict(spec, limits={"max_steps": 2, "max_seconds": 5.0})
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "final A1 updates"):
                load_spec(path)

            invalid = dict(spec, min_a1_updates=0)
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "final A1 exposure quota"):
                load_spec(path)

            invalid = dict(spec, min_a1_supervised_tokens=True)
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "final A1 exposure quota"):
                load_spec(path)

    def test_a8_hard_forward_is_deployed_integer_dot_with_ste_gradient(self):
        weight = torch.tensor([[0.5, -0.5, 0.5, 1.0], [-1.0, 0.25, 0.5, -0.75]])
        module = BlockBinaryLinear(weight, torch.tensor([0.75, 0.25]), W1AxContract(8))
        inputs = torch.tensor([[1.0, -0.5, 0.0, 0.25], [-0.125, 0.25, 0.5, -1.0]])
        output = module(inputs)
        raw = inputs.float()
        maximum = raw.abs().amax(-1, keepdim=True)
        codes = torch.round(raw * torch.where(maximum > 0, 127 / maximum, 0)).clamp(-127, 127)
        signs = torch.where(module.latent_sign < 0, -1.0, 1.0)
        expected = (torch.nn.functional.linear(codes, signs) * module.effective_scales()) * (
            maximum / 127
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.sum().backward()
        self.assertTrue(torch.isfinite(module.latent_sign.grad).all())
        self.assertTrue((module.latent_sign.grad != 0).any())

    def test_block_model_owns_only_binary_trainables_and_rejects_mask_drift(self):
        config, source, batch = _model_fixture()
        model = BlockDrafter(source, config)
        optimizer = block_optimizer(model)
        report, output = block_train_step(model, optimizer, batch, update=False)
        self.assertFalse(report["optimizer_updated"])
        self.assertEqual(tuple(output.logits.shape), (7, 19))
        self.assertEqual(len(model.binary_linears()), 16)
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertFalse(model.output.requires_grad)
        self.assertNotEqual(model.output.data_ptr(), source["output.weight"].data_ptr())
        batch.attention_allowed[0, -1] = False
        with self.assertRaisesRegex(ValueError, "bidirectional"):
            model(batch)

    def test_checkpoint_resume_update_is_exact_and_a8_transition_resets_moments(self):
        source = {
            "base_gguf_sha256": "a" * 64,
            "data_manifest_sha256": "b" * 64,
            "bundle_sha256": "c" * 64,
            "synthetic": True,
        }
        config, tensors, batch = _model_fixture()
        uninterrupted = BlockDrafter(tensors, config)
        uninterrupted_optimizer = block_optimizer(uninterrupted)
        block_train_step(uninterrupted, uninterrupted_optimizer, batch)
        clone = BlockDrafter(tensors, config)
        clone_optimizer = block_optimizer(clone)
        with tempfile.TemporaryDirectory() as folder:
            receipt = save_block_checkpoint(
                uninterrupted,
                uninterrupted_optimizer,
                TrainCursor(step=1, supervised_tokens=7, presented_tokens=7),
                source,
                Path(folder),
            )
            restored = load_block_checkpoint(clone, clone_optimizer, source, receipt)
            self.assertEqual(restored.step, 1)
            block_train_step(uninterrupted, uninterrupted_optimizer, batch)
            block_train_step(clone, clone_optimizer, batch)
            for name, module in uninterrupted.binary_linears().items():
                for key, value in module.state_dict().items():
                    self.assertTrue(
                        torch.equal(value, clone.binary_linears()[name].state_dict()[key])
                    )
            exported = export_block_checkpoint(clone, source, Path(folder) / "export")
            manifest = json.loads(Path(exported["manifest"]).read_text())
            self.assertEqual(len(manifest["projections"]), 16)
            self.assertFalse(exported["native_graph_admitted"])

        a1, fresh_optimizer, transition = transition_a8_to_a1(
            uninterrupted, source_checkpoint_sha256="d" * 64
        )
        self.assertEqual(a1.config.activation_bits, 1)
        self.assertFalse(transition["exact_resume"])
        self.assertEqual(fresh_optimizer.state_dict()["state"], {})
        for name, module in uninterrupted.binary_linears().items():
            self.assertTrue(torch.equal(module.latent_sign, a1.binary_linears()[name].latent_sign))

    def test_sparse_initializer_checkpoint_resume_keeps_moments_without_reapplying(self):
        source = {
            "base_gguf_sha256": "a" * 64,
            "data_manifest_sha256": "b" * 64,
            "bundle_sha256": "c" * 64,
            "synthetic": True,
        }
        config, tensors, batch = _model_fixture()
        reference = BlockDrafter(tensors, config)
        original_fc = reference.fc.latent_sign.detach().clone()
        fitted_latent = torch.where(original_fc < 0, 1.0, -1.0) * original_fc.abs()
        fitted_scale = torch.full_like(reference.fc.initial_scale, 0.17)
        initialization = {"fc": (fitted_latent, fitted_scale)}
        original = BlockDrafter(tensors, config, binary_initializer=initialization)
        optimizer = block_optimizer(original)
        block_train_step(original, optimizer, batch)
        cursor = TrainCursor(step=1, supervised_tokens=7, presented_tokens=7)

        with tempfile.TemporaryDirectory() as folder:
            receipt = save_block_checkpoint(original, optimizer, cursor, source, Path(folder))
            resumed = BlockDrafter(tensors, config, binary_initializer=initialization)
            resumed_optimizer = block_optimizer(resumed)
            restored = load_block_checkpoint(resumed, resumed_optimizer, source, receipt)
            self.assertEqual(restored, cursor)
            self.assertTrue(torch.equal(resumed.fc.latent_sign, original.fc.latent_sign))
            self.assertTrue(torch.equal(resumed.fc.initial_scale, original.fc.initial_scale))
            original_moment = optimizer.state_dict()["state"][0]["exp_avg"]
            resumed_moment = resumed_optimizer.state_dict()["state"][0]["exp_avg"]
            self.assertTrue(torch.equal(resumed_moment, original_moment))
            # A post-resume initializer application would erase the committed signs,
            # scales or AdamW moments; the next real optimizer step must still match.
            block_train_step(original, optimizer, batch)
            block_train_step(resumed, resumed_optimizer, batch)
            for name, module in original.binary_linears().items():
                for key, value in module.state_dict().items():
                    self.assertTrue(
                        torch.equal(value, resumed.binary_linears()[name].state_dict()[key])
                    )


if __name__ == "__main__":
    unittest.main()

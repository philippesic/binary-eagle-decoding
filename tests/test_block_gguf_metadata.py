"""Actual GGUF F32 scalar serialization through the production block loader."""

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from test_nine_model_training import block_fixture

from w1a1_eagle.block_qat import BlockDrafter
from w1a1_eagle.block_training import load_block_gguf
from w1a1_eagle.continuous_qat import sha256


class BlockGGUFMetadataTests(unittest.TestCase):
    def test_declared_python_epsilon_matches_exact_deployed_f32_metadata(self):
        sys.path.insert(
            0, str(Path(__file__).resolve().parents[1] / "third_party/llama.cpp/gguf-py")
        )
        from gguf import GGUFWriter

        for family in ("dspark", "dflash"):
            config, tensors, _ = block_fixture(family)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "source.gguf"
                writer = GGUFWriter(str(path), "dflash")
                writer.add_uint32("dflash.block_size", 7)
                writer.add_bool("dflash.sample_from_anchor", True)
                writer.add_array("dflash.target_layers", [2, 10, 18, 26, 34])
                for name, value in {
                    "dflash.embedding_length": config.hidden_size,
                    "dflash.block_count": 5,
                    "dflash.feed_forward_length": config.intermediate_size,
                    "dflash.attention.head_count": config.num_heads,
                    "dflash.attention.head_count_kv": config.num_kv_heads,
                    "dflash.attention.key_length": config.head_dim,
                    "dflash.attention.value_length": config.head_dim,
                    "tokenizer.ggml.mask_token_id": config.mask_token_id,
                }.items():
                    writer.add_uint32(name, value)
                writer.add_float32("dflash.rope.freq_base", config.rope_theta)
                writer.add_float32("dflash.attention.layer_norm_rms_epsilon", config.norm_eps)
                for name, tensor in tensors.items():
                    writer.add_tensor(name, tensor.numpy())
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                pin = sha256(path)
                # Actual released BF16 bases omit optional rope.dimension_count.
                loaded = load_block_gguf(path, pin, config)
                self.assertEqual(len(BlockDrafter(loaded, config).binary_linears()), 16)
                with self.assertRaisesRegex(ValueError, "rms_epsilon"):
                    load_block_gguf(path, pin, replace(config, norm_eps=config.norm_eps * 10))

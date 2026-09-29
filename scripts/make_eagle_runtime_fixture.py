#!/usr/bin/env python3
"""Write a tiny synthetic CPU EAGLE3 fixture for native runtime contract checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--packed", action="store_true", help="store all nine linears as row-scale W1"
    )
    parser.add_argument(
        "--llama-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "third_party" / "llama.cpp",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.llama_dir.resolve() / "gguf-py"))
    import gguf
    import numpy as np

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(args.output), "eagle3")
    writer.add_name("tiny-cpu-runtime-fixture")
    writer.add_tokenizer_model("none")
    writer.add_vocab_size(128)
    writer.add_context_length(128)
    writer.add_embedding_length(16)
    writer.add_block_count(1)
    writer.add_feed_forward_length(32)
    writer.add_head_count(2)
    writer.add_head_count_kv(1)
    writer.add_rope_dimension_count(8)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_target_layers([0, 1, 2])
    writer.add_target_hidden_size(16)
    rng = np.random.default_rng(4242)
    shapes = {
        "token_embd.weight": (128, 16),
        "fc.weight": (16, 48),
        "output.weight": (32, 16),
        "blk.0.attn_q.weight": (16, 32),
        "blk.0.attn_k.weight": (8, 32),
        "blk.0.attn_v.weight": (8, 32),
        "blk.0.attn_output.weight": (16, 16),
        "blk.0.ffn_gate.weight": (32, 16),
        "blk.0.ffn_up.weight": (32, 16),
        "blk.0.ffn_down.weight": (16, 32),
    }
    if args.packed:
        writer.add_uint32("eagle3.w1a1.version", 1)
        writer.add_array("eagle3.w1a1.groups", ["fusion", "attention", "ffn", "head"])
        writer.add_array(
            "eagle3.w1a1.tensors", [name for name in shapes if name != "token_embd.weight"]
        )
        writer.add_string("eagle3.w1a1.bit_order", "little")
        writer.add_string("eagle3.w1a1.sign_rule", "nonnegative_is_one")
        writer.add_string("eagle3.w1a1.scale_rule", "f32_mean_abs")
        writer.add_string("eagle3.w1a1.arithmetic", "f32")
    for name, shape in shapes.items():
        dense = rng.normal(0, 0.1, shape).astype(np.float32)
        if not args.packed or name == "token_embd.weight":
            writer.add_tensor(name, dense)
            continue
        rows, width = shape
        signs = np.zeros((rows, (width + 31) // 32), dtype=np.uint32)
        for column in range(width):
            signs[:, column // 32] |= (dense[:, column] >= 0).astype(np.uint32) << (column % 32)
        packed_name = name.removesuffix(".weight") + ".w1a1_packed"
        scale_name = name.removesuffix(".weight") + ".w1a1_scale"
        prefix = "eagle3.w1a1.tensor." + name.replace(".", "_")
        writer.add_uint32(prefix + ".logical_k", width)
        writer.add_string(prefix + ".packed", packed_name)
        writer.add_string(prefix + ".scale", scale_name)
        writer.add_tensor(packed_name, signs.view(np.int32))
        writer.add_tensor(
            scale_name, np.abs(dense).mean(axis=1, dtype=np.float64).astype(np.float32)
        )
    for name in (
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_norm_2.weight",
        "blk.0.ffn_norm.weight",
    ):
        writer.add_tensor(name, np.ones(16, np.float32))
    writer.add_tensor("d2t", np.arange(31, -1, -1, dtype=np.int64) * 3)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()

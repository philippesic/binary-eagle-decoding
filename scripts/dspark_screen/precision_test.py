import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from precision_q4 import EXPECTED, command, inspect, policy_type


class PrecisionPolicyTests(unittest.TestCase):
    def test_exact_fifteen_matrix_scope(self):
        self.assertEqual(len(EXPECTED), 15)
        for name in EXPECTED:
            self.assertEqual(policy_type(name), "Q4_0")
        for name in ("blk.5.ffn_up.weight", "fc.weight", "output.weight", "token_embd.weight",
                     "markov_w1.weight", "markov_w2.weight", "conf_proj.weight", "blk.0.attn_q.weight"):
            self.assertEqual(policy_type(name), "BF16")

    def test_quantized_default_and_ordered_override(self):
        cmd = command(Path("quantizer"), Path("source.gguf"), Path("output.gguf"))
        patterns = [cmd[i+1] for i, arg in enumerate(cmd) if arg == "--tensor-type"]
        self.assertEqual(patterns[-1], "^.*$=BF16")
        self.assertTrue(patterns[0].endswith("=Q4_0"))
        self.assertEqual(cmd[-2:], ["Q4_0", "4"])
        self.assertIn("--leave-output-tensor", cmd)
        self.assertIn("--token-embedding-type", cmd)

    def test_source_overwrite_rejects(self):
        with self.assertRaises(ValueError):
            command(Path("q"), Path("same"), Path("same"))

    @unittest.skipUnless(os.environ.get("DSPARK_TEST_QUANTIZER") and os.environ.get("DSPARK_TEST_LLAMA"), "optional actual CPU quantizer fixture")
    def test_actual_cpu_quantizer_preserves_all_scaffolding(self):
        llama = Path(os.environ["DSPARK_TEST_LLAMA"])
        sys.path.insert(0, str(llama / "gguf-py"))
        import numpy as np
        from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, output = root / "toy.gguf", root / "toy-q4-ffn.gguf"
            writer = GGUFWriter(str(source), "dflash")
            writer.add_name("CPU toy policy fixture; not released-model proof")
            writer.add_block_count(5)
            writer.add_embedding_length(32)
            writer.add_feed_forward_length(64)
            writer.add_head_count(4)
            writer.add_head_count_kv(2)
            writer.add_context_length(2048)
            writer.add_rope_dimension_count(8)
            writer.add_layer_norm_rms_eps(1e-6)
            writer.add_target_layers([2, 10, 18, 26, 34])
            writer.add_block_size(7)
            writer.add_sample_from_anchor(True)
            writer.add_tokenizer_model("gpt2")
            writer.add_token_list([f"token{i}" for i in range(32)])
            rng = np.random.default_rng(42)
            for name in sorted(EXPECTED):
                shape = (32, 64) if "ffn_down" in name else (64, 32)
                raw = (rng.normal(size=shape).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
                writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.BF16)
            for name in ("fc.weight", "token_embd.weight", "output.weight", "markov_w1.weight", "markov_w2.weight", "blk.0.attn_q.weight"):
                raw = (rng.normal(size=(32, 32)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
                writer.add_tensor(name, raw, raw_dtype=GGMLQuantizationType.BF16)
            writer.add_tensor("blk.0.ffn_norm.weight", np.ones(32, dtype=np.float32))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            completed = subprocess.run(command(Path(os.environ["DSPARK_TEST_QUANTIZER"]), source, output, 2),
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = inspect(source, output, llama)
            self.assertEqual(len(report["selected"]), 15)
            self.assertEqual(len(report["preserved"]), 7)
            self.assertTrue(report["non_ffn_immutable"])
            self.assertFalse(report["passed"], "toy fixture must not issue a released-source admission")
            head = next(t for t in GGUFReader(str(output)).tensors if t.name == "output.weight")
            with output.open("r+b") as file:
                file.seek(head.data_offset)
                byte = file.read(1)
                file.seek(head.data_offset)
                file.write(bytes([byte[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "non-FFN source bytes/type changed"):
                inspect(source, output, llama)


if __name__ == "__main__":
    unittest.main()

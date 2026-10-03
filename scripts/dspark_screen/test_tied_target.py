import json
import tempfile
import unittest
from pathlib import Path

from check_export import sha256
from compare_frozen import tied_target_proof


class TiedTargetTests(unittest.TestCase):
    def fixture(self, root):
        config = root / "config.json"
        config.write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": True}))
        loader = root / "src/models/qwen3.cpp"
        loader.parent.mkdir(parents=True)
        loader.write_text('if (output == NULL) {\noutput = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);\n}')
        return config, loader

    def test_exact_config_and_loader_prove_tied_source_role(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, loader = self.fixture(root)
            proof = tied_target_proof(config, sha256(config), root, "qwen3")
            self.assertEqual(proof["target_head_source_tensor"], "token_embd.weight")
            self.assertEqual(proof["target_loader_sha256"], sha256(loader))

    def test_missing_or_wrong_pin_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = self.fixture(root)
            for cfg, digest in ((None, None), (config, "wrong")):
                with self.assertRaisesRegex(ValueError, "pinned target"):
                    tied_target_proof(cfg, digest, root, "qwen3")

    def test_false_tie_flag_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _ = self.fixture(root)
            config.write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"], "tie_word_embeddings": False}))
            with self.assertRaisesRegex(ValueError, "proven Qwen3"):
                tied_target_proof(config, sha256(config), root, "qwen3")

    def test_missing_loader_fallback_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, loader = self.fixture(root)
            loader.write_text("no fallback")
            with self.assertRaisesRegex(ValueError, "native Qwen3 loader"):
                tied_target_proof(config, sha256(config), root, "qwen3")


if __name__ == "__main__":
    unittest.main()

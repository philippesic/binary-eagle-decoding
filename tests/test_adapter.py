"""CPU checks for selective wrapping of the pinned AngelSlim drafter graph."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from w1a1_eagle import (  # noqa: E402
    GROUP_PATHS,
    DrafterStructureError,
    W1A1Config,
    W1A1Linear,
    fake_binary_linear,
    install_w1a1,
)


class MockAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 4, bias=False)
        self.k_proj = nn.Linear(8, 4, bias=False)
        self.v_proj = nn.Linear(8, 4, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class MockMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 6, bias=False)
        self.up_proj = nn.Linear(4, 6, bias=False)
        self.down_proj = nn.Linear(6, 4, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class MockMidlayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MockAttention()
        self.mlp = MockMLP()

    def forward(self, emb, fused):
        state = fused + self.self_attn(torch.cat((emb, fused), dim=-1))
        return state + self.mlp(state)


class MockTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.lm_head = nn.Linear(4, 7, bias=False)


class MockDrafter(nn.Module):
    """The official direct module names, scaled down to tiny dimensions."""

    def __init__(self, borrowed_embedding):
        super().__init__()
        self.config = SimpleNamespace(pretraining_tp=1)
        self.embed_tokens = borrowed_embedding
        self.fc = nn.Linear(12, 4, bias=False)
        self.midlayer = MockMidlayer()
        self.lm_head = nn.Linear(4, 7, bias=False)

    def forward(self, token_ids, features):
        emb = self.embed_tokens(token_ids)
        _ = self.fc.weight.dtype  # The pinned drafter reads this before calling fc.
        fused = self.fc(features)
        return self.lm_head(self.midlayer(emb, fused))


def module_at(root, path):
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


class DrafterAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.target = MockTarget()
        self.drafter = MockDrafter(self.target.embed_tokens)
        self.tokens = torch.tensor([[1, 2, 3]])
        self.features = torch.randn(1, 3, 12)

    def test_group_coverage_and_borrowed_embedding_untouched(self):
        originals = {
            path: module_at(self.drafter, path) for paths in GROUP_PATHS.values() for path in paths
        }
        target_head = self.target.lm_head
        target_embedding = self.target.embed_tokens
        handle = install_w1a1(self.drafter, GROUP_PATHS, target=self.target)

        self.assertEqual(set(handle.wrappers), set(originals))
        for path, original in originals.items():
            wrapper = module_at(self.drafter, path)
            self.assertIsInstance(wrapper, W1A1Linear)
            self.assertIs(wrapper.linear, original)
        self.assertIs(self.drafter.embed_tokens, target_embedding)
        self.assertIs(self.target.embed_tokens, target_embedding)
        self.assertIs(self.target.lm_head, target_head)
        self.assertNotIsInstance(target_head, W1A1Linear)

    def test_selected_group_and_disabled_path_are_exact(self):
        dense = self.drafter(self.tokens, self.features)
        original_fc = self.drafter.fc
        original_head = self.drafter.lm_head
        config = W1A1Config(zero_sign=-1, weight_scale="rms", activation_scale="unit")
        handle = install_w1a1(self.drafter, ["feature_fusion"], config, enabled=False)
        self.assertIs(handle.wrappers["fc"].linear, original_fc)
        self.assertIs(self.drafter.lm_head, original_head)
        torch.testing.assert_close(self.drafter(self.tokens, self.features), dense, rtol=0, atol=0)

        handle.set_enabled(True)
        expected_fc = fake_binary_linear(self.features, original_fc.weight, config=config)
        actual = self.drafter(self.tokens, self.features)
        expected = original_head(
            self.drafter.midlayer(self.drafter.embed_tokens(self.tokens), expected_fc)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(actual, dense))

        handle.set_enabled(False)
        torch.testing.assert_close(self.drafter(self.tokens, self.features), dense, rtol=0, atol=0)
        handle.uninstall()
        self.assertIs(self.drafter.fc, original_fc)
        torch.testing.assert_close(self.drafter(self.tokens, self.features), dense, rtol=0, atol=0)

    def test_bad_group_or_module_fails_before_any_edit(self):
        original_fc = self.drafter.fc
        with self.assertRaisesRegex(ValueError, "unknown W1A1 group"):
            install_w1a1(self.drafter, ["feature_fusion", "not_a_group"])
        self.assertIs(self.drafter.fc, original_fc)

        self.drafter.midlayer.self_attn.k_proj = nn.Identity()
        with self.assertRaisesRegex(DrafterStructureError, "midlayer.self_attn.k_proj"):
            install_w1a1(self.drafter, ["feature_fusion", "attention"])
        self.assertIs(self.drafter.fc, original_fc)

        del self.drafter.midlayer.mlp
        with self.assertRaisesRegex(DrafterStructureError, "midlayer.mlp.gate_proj"):
            install_w1a1(self.drafter, ["ffn"])

    def test_tensor_parallel_bypass_is_rejected(self):
        self.drafter.config.pretraining_tp = 2
        for group in ("attention", "ffn"):
            with (
                self.subTest(group=group),
                self.assertRaisesRegex(DrafterStructureError, "pretraining_tp == 1"),
            ):
                install_w1a1(self.drafter, [group])
        self.assertNotIsInstance(self.drafter.midlayer.self_attn.q_proj, W1A1Linear)

    def test_target_owned_head_is_rejected(self):
        self.drafter.lm_head = self.target.lm_head
        with self.assertRaisesRegex(DrafterStructureError, "target-owned"):
            install_w1a1(self.drafter, ["lm_head"], target=self.target)
        self.assertIs(self.drafter.lm_head, self.target.lm_head)

    def test_reinstall_and_external_replacement_fail_clearly(self):
        handle = install_w1a1(self.drafter, ["feature_fusion"])
        with self.assertRaisesRegex(DrafterStructureError, "expected nn.Linear"):
            install_w1a1(self.drafter, ["feature_fusion"])
        self.drafter.fc = nn.Linear(12, 4)
        with self.assertRaisesRegex(DrafterStructureError, "cannot uninstall"):
            handle.uninstall()


if __name__ == "__main__":
    unittest.main()

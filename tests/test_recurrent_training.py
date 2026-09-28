"""CPU-only joint-parameter ownership and checkpoint interface tests."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH, GroupedBinaryLinear
from w1a1_eagle.recurrent_trace import TraceAudit
from w1a1_eagle.recurrent_training import CHECKPOINT_NAMES, save_training_checkpoint, train_step


def linears():
    latent = torch.tensor([[0.5, -0.5], [0.5, 0.5]], device="cpu")
    scale = torch.ones((2, 1), device="cpu")
    return {path: GroupedBinaryLinear(latent, scale) for path in CANDIDATE_D_BASE_TO_PATH.values()}


def audit():
    return TraceAudit(
        draft_labels=(-1, 0),
        valid_mask=(True, True),
        supported_mask=(False, True),
        ce_mask=(False, True),
        denominator_mask=(True, True),
        target_to_draft=(0, 1, -1),
        counts={"total": 2, "valid": 2, "unsupported": 1},
        per_depth={},
    )


class RecurrentTrainingTests(unittest.TestCase):
    def test_later_loss_updates_body_and_binary_head(self):
        modules = linears()
        params = [parameter for module in modules.values() for parameter in module.parameters()]
        optimizer = torch.optim.SGD(params, lr=0.01)
        x = torch.tensor([0.5, 0.25], device="cpu")
        body, head = modules["fc"], modules["lm_head"]
        first = body(x)
        first.retain_grad()
        later = body(first)
        logits = torch.stack((head(first), head(later)))
        before_body = body.latent_sign.detach().clone()
        before_head = head.latent_sign.detach().clone()
        loss = train_step(modules, logits, audit(), optimizer)
        self.assertGreater(loss, 0)
        self.assertIsNotNone(first.grad)
        self.assertGreater(float(first.grad.abs().sum()), 0)
        self.assertFalse(torch.equal(body.latent_sign.detach(), before_body))
        self.assertFalse(torch.equal(head.latent_sign.detach(), before_head))
        self.assertTrue(bool((body.effective_scales() >= 0).all()))

    def test_optimizer_rejects_extra_target_parameter(self):
        modules = linears()
        extra = torch.nn.Parameter(torch.ones(1, device="cpu"))
        params = [parameter for module in modules.values() for parameter in module.parameters()]
        optimizer = torch.optim.SGD([*params, extra], lr=0.01)
        with self.assertRaisesRegex(ValueError, "only the nine"):
            train_step(
                modules, torch.zeros((2, 2), device="cpu", requires_grad=True), audit(), optimizer
            )

    def test_checkpoint_matches_exporter_schema_and_copies_arrays(self):
        modules = linears()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "trained.npz"
            manifest_path = Path(temporary) / "trained.json"
            result = save_training_checkpoint(modules, "a" * 64, checkpoint, manifest_path)
            self.assertEqual(result["projections"], 9)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(set(manifest["projections"]), set(CHECKPOINT_NAMES))
            with np.load(checkpoint, allow_pickle=False) as archive:
                self.assertEqual(len(archive.files), 18)
                for base, name in CHECKPOINT_NAMES.items():
                    path = CANDIDATE_D_BASE_TO_PATH[base]
                    latent, scales = modules[path].training_arrays()
                    np.testing.assert_array_equal(archive[name + ".latent"], latent)
                    np.testing.assert_array_equal(archive[name + ".scale"], scales)
            with self.assertRaises(FileExistsError):
                save_training_checkpoint(modules, "a" * 64, checkpoint, manifest_path)


if __name__ == "__main__":
    unittest.main()

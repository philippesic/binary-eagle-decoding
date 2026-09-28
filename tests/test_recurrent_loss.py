"""CPU-only differentiable recurrence and CE-mask checks."""

import unittest

import torch

from w1a1_eagle.recurrent_loss import supported_prefix_ce
from w1a1_eagle.recurrent_trace import TraceAudit


def audit(labels, mask):
    return TraceAudit(
        draft_labels=tuple(labels),
        valid_mask=(True,) * len(labels),
        supported_mask=tuple(mask),
        ce_mask=tuple(mask),
        denominator_mask=(True,) * len(labels),
        target_to_draft=(0, 1, -1),
        counts={"total": len(labels)},
        per_depth={},
    )


class RecurrentLossTests(unittest.TestCase):
    def test_unsupported_early_label_keeps_later_gradient_path(self):
        weight = torch.tensor(0.7, dtype=torch.float32, device="cpu", requires_grad=True)
        seed = torch.tensor(1.25, dtype=torch.float32, device="cpu")
        first_state = weight * seed
        second_state = weight * first_state
        logits = torch.stack(
            (torch.stack((first_state, -first_state)), torch.stack((second_state, -second_state)))
        )
        loss = supported_prefix_ce(logits, audit((-1, 1), (False, True)))
        loss.backward()
        self.assertTrue(torch.isfinite(weight.grad))
        self.assertGreater(abs(weight.grad.item()), 0.01)
        self.assertEqual(logits.grad_fn is not None, True)

    def test_only_supported_rows_contribute_direct_ce(self):
        logits = torch.tensor([[100.0, -100.0], [0.1, 0.2]], device="cpu", requires_grad=True)
        loss = supported_prefix_ce(logits, audit((-1, 1), (False, True)))
        expected = torch.nn.functional.cross_entropy(
            logits[1:].float(), torch.tensor([1], device="cpu")
        )
        self.assertTrue(torch.allclose(loss, expected))
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[0], torch.zeros(2)))
        self.assertGreater(abs(logits.grad[1, 0].item()), 0.01)

    def test_rejects_missing_support_and_bad_shape(self):
        logits = torch.zeros(2, 2, device="cpu")
        with self.assertRaisesRegex(ValueError, "no supported"):
            supported_prefix_ce(logits, audit((-1, -1), (False, False)))
        with self.assertRaisesRegex(ValueError, "one row"):
            supported_prefix_ce(logits[:, :1], audit((0, 1), (True, True)))


if __name__ == "__main__":
    unittest.main()

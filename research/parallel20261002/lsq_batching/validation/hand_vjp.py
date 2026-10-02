#!/usr/bin/env python3
"""Independent handwritten LSQ VJP and shared-consumer batching audit.

CPU only; uses deterministic synthetic tensors and the public learned_activation
API. The handwritten expressions below implement the documented surrogate
derivatives separately from autograd's custom backward.
"""

from __future__ import annotations

import math
import json
from pathlib import Path

import torch

from w1a1_eagle.learned_activation import learned_activation

CONTROL_PATH = Path("/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json")


def check_control():
    """Honor the shared research stop/reset gate before each precision case."""
    control = json.loads(CONTROL_PATH.read_text())
    if control.get("research_stop") or control.get("reset_observed"):
        raise SystemExit(
            "parallel research stop/reset is set; stopping before the next precision case"
        )


def hand_vjp(x: torch.Tensor, upstream: torch.Tensor, bits: int, parameter: float,
             valid: torch.Tensor, normalization_count: int):
    """Return analytical STE input/parameter VJPs from the contract equations."""
    x = x.float()
    upstream = upstream.float()
    valid_e = valid.unsqueeze(-1)
    if bits == 1:
        beta = x.abs().double().mean(-1, keepdim=True).float()
        threshold = torch.tensor(parameter, dtype=torch.float32) * beta
        shifted = x if parameter == 0 else x - threshold
        support = (shifted.abs() <= beta) | (beta == 0)
        support &= valid_e
        dx = upstream * support
        dc = (upstream * (-beta * support)).sum() / math.sqrt(normalization_count)
    else:
        qmax = (1 << (bits - 1)) - 1
        absmax = x.abs().amax(-1, keepdim=True)
        limit = absmax * torch.tensor(parameter, dtype=torch.float32)
        support = (x.abs() <= limit) & valid_e
        # Hard forward: round-to-even and symmetric signed saturation.
        scale = limit / qmax
        inv = torch.where(limit > 0, qmax / limit, torch.zeros_like(limit))
        codes = torch.round(x * inv).clamp(-qmax, qmax)
        values = codes * scale
        dvalue_dc = torch.where(support, (values - x) / parameter, values / parameter)
        dx = upstream * support
        dc = (upstream * dvalue_dc * valid_e).sum() / math.sqrt(
            normalization_count * qmax
        )
    return dx, dc


def check_close(actual, expected, label):
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7, msg=label)


def shared_qkv_case(bits: int):
    gen = torch.Generator().manual_seed(8241 + bits)
    # Q/K/V consume the same activation tensor and validity mask. Distinct
    # cotangents make the shared scalar receive three different VJPs.
    x0 = torch.randn((3, 5), generator=gen) * 0.73 + 0.08
    gs = [torch.randn((3, 5), generator=gen) for _ in range(3)]
    mask = torch.tensor([True, True, False])
    xs = [x0] * 3
    masks = [mask] * 3
    value = 0.22 if bits == 1 else 0.63
    # Default normalization is per invocation: each consumer counts its own
    # valid feature elements, while gradients accumulate into one tied scalar.
    p = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    loss = torch.zeros(())
    expected_dx, expected_dp = [], torch.zeros(())
    actual_x = []
    for x0, g, mask in zip(xs, gs, masks):
        x = x0.clone().requires_grad_(True)
        actual_x.append(x)
        result = learned_activation(x, bits, p, valid_mask=mask)
        loss = loss + (result.values * g).sum()
        local_n = max(int(mask.sum()) * x.shape[-1], 1)
        dx, dp = hand_vjp(x0, g, bits, value, mask, local_n)
        expected_dx.append(dx)
        expected_dp = expected_dp + dp
    loss.backward()
    check_close(p.grad, expected_dp, f"A{bits} tied QKV scalar sum")
    for i, (x, hand_dx) in enumerate(zip(actual_x, expected_dx)):
        check_close(x.grad, hand_dx, f"A{bits} tied QKV input {i}")

    # Invocation-local normalization is observably different from treating Q,
    # K, V as a pooled activation population. Compare with the incorrect pooled
    # formula as a guard that the synthetic case actually distinguishes them.
    pooled_dp = torch.zeros(())
    pooled_n = sum(int(m.sum()) * x.shape[-1] for m, x in zip(masks, xs))
    for x, g, mask in zip(xs, gs, masks):
        _, dp = hand_vjp(x, g, bits, value, mask, pooled_n)
        pooled_dp = pooled_dp + dp
    assert not torch.isclose(p.grad, pooled_dp, rtol=1e-4, atol=1e-6), (
        f"A{bits}: per-invocation and pooled normalizers unexpectedly agree"
    )

    # Explicit chunking with the full invocation's normalizer must reproduce
    # the unchunked input and parameter gradients for each consumer.
    for x0, g, mask in zip(xs, gs, masks):
        x_full = x0.clone().requires_grad_(True)
        p_full = torch.tensor(value, requires_grad=True)
        full = learned_activation(x_full, bits, p_full, valid_mask=mask)
        (full.values * g).sum().backward()
        x_chunk = x0.clone().requires_grad_(True)
        p_chunk = torch.tensor(value, requires_grad=True)
        n = max(int(mask.sum()) * x0.shape[-1], 1)
        # One row at a time; each chunk mask is scalar token shape.
        for row, row_g, row_valid in zip(x_chunk.split(1), g.split(1), mask.split(1)):
            chunk = learned_activation(
                row, bits, p_chunk, valid_mask=row_valid,
                normalization_count=n,
            )
            (chunk.values * row_g).sum().backward()
        check_close(x_chunk.grad, x_full.grad, f"A{bits} chunk input")
        check_close(p_chunk.grad, p_full.grad, f"A{bits} chunk parameter")
    print(f"A{bits}: handwritten VJP matches; tied QKV accumulates three local-normalized consumers; chunk parity passed")


def zero_controls(bits: int):
    value = 0.22 if bits == 1 else 0.63
    # Padding-only invocation: neither inputs nor the shared parameter receive
    # gradients, even though the hard forward has ordinary values.
    x_invalid = torch.tensor([[0.4, -0.2, 0.7], [-0.3, 0.1, -0.5]], requires_grad=True)
    p_invalid = torch.tensor(value, requires_grad=True)
    invalid = learned_activation(
        x_invalid, bits, p_invalid, valid_mask=torch.tensor([False, False])
    )
    invalid.values.sum().backward()
    check_close(x_invalid.grad, torch.zeros_like(x_invalid), f"A{bits} all-invalid input")
    check_close(p_invalid.grad, torch.zeros_like(p_invalid), f"A{bits} all-invalid parameter")

    # A valid zero row keeps the specified identity input STE while inventing
    # no clip/threshold parameter gradient.
    x_zero = torch.zeros((1, 3), requires_grad=True)
    p_zero = torch.tensor(value, requires_grad=True)
    zero = learned_activation(x_zero, bits, p_zero)
    zero.values.sum().backward()
    check_close(x_zero.grad, torch.ones_like(x_zero), f"A{bits} zero-row input")
    check_close(p_zero.grad, torch.zeros_like(p_zero), f"A{bits} zero-row parameter")
    print(f"A{bits}: all-invalid and valid zero-row controls passed")


def main():
    torch.set_num_threads(1)
    for bits in (1, 4, 8):
        check_control()
        shared_qkv_case(bits)
        zero_controls(bits)
    print("PASS: independent CPU handwritten LSQ/threshold VJP audit")


if __name__ == "__main__":
    main()

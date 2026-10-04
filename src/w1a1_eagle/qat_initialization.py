"""Sparse calibrated signs/scales installed only on fresh binary student modules."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def apply_binary_initialization(
    linears: Mapping, initialization: Mapping | None, *, policy="preserve_reference_magnitudes"
) -> dict:
    if initialization is None:
        return {}
    if policy not in {"preserve_reference_magnitudes", "unit_probe"}:
        raise ValueError("explicit calibrated latent magnitude policy required")
    report = {}
    if not initialization or set(initialization) - set(linears):
        raise ValueError("initializer must select existing student binary projections")
    for name, (latent, scale) in initialization.items():
        module = linears[name]
        if getattr(module, "_round_hard_signs", None) is not None:
            raise ValueError("initializer cannot mutate an active attached hard-forward graph")
        if (
            not isinstance(latent, torch.Tensor)
            or not isinstance(scale, torch.Tensor)
            or latent.dtype != torch.float32
            or scale.dtype != torch.float32
            or latent.shape != module.latent_sign.shape
            or scale.shape != module.initial_scale.shape
            or not bool(torch.isfinite(latent).all())
            or not bool(torch.isfinite(scale).all())
            or bool((scale < 0).any())
        ):
            raise ValueError("calibrated initializer shape/dtype/value contract differs: " + name)
        if bool((torch.signbit(latent) & (latent == 0)).any()):
            raise ValueError("calibrated negative-zero latent cannot represent negative hard sign")
        reference = module.latent_sign.detach().cpu().abs()
        expected = (
            reference if policy == "preserve_reference_magnitudes" else torch.ones_like(reference)
        )
        if not torch.equal(latent.detach().cpu().abs(), expected):
            raise ValueError(
                "calibrated latent magnitudes differ from declared reference/probe policy"
            )
        import hashlib

        import numpy as np

        report[name] = {
            "policy": policy,
            "reference_sha256": hashlib.sha256(
                np.ascontiguousarray(reference.numpy(), dtype="<f4").tobytes()
            ).hexdigest(),
        }

    # Validate every projection before modifying any module.
    with torch.no_grad():
        for name, (latent, scale) in initialization.items():
            module = linears[name]
            module.latent_sign.copy_(latent.to(module.latent_sign.device))
            module.initial_scale.copy_(scale.to(module.initial_scale.device))
            module.scale_offset.zero_()

    return report

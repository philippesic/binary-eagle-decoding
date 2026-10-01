"""Canonical attached-recipe state and strict shared-parameter resume validation."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def recipe_state(linears: Mapping) -> dict:
    bank_state = None
    if any(getattr(m, "activation_quantizer", None) is not None for m in linears.values()):
        from .learned_activation import LearnedActivationBank

        bank_state = LearnedActivationBank.from_attached(linears).checkpoint()
    correction = getattr(linears["fc"], "fusion_correction", None)
    affine = None
    if any(getattr(m, "affine_binary", None) is not None for m in linears.values()):
        from .affine_binary import AffineBinaryBank

        affine = AffineBinaryBank.from_attached(linears).state_payload()
    return {
        "activation_bank": bank_state,
        "fusion_correction": None if correction is None else correction.state_payload(),
        "affine_binary": affine,
    }


def validate_resume_state(linears: Mapping, states: Mapping, recipes: Mapping) -> None:
    """Validate every tensor and tied alias before changing a live model."""
    if not isinstance(states, Mapping) or set(states) != set(linears):
        raise ValueError("resume projection inventory differs")
    current_recipe = recipe_state(linears)
    if not isinstance(recipes, Mapping) or set(recipes) != set(current_recipe):
        raise ValueError("resume recipe inventory differs")
    for kind in current_recipe:
        live, saved = current_recipe[kind], recipes[kind]
        if (live is None) != (saved is None):
            raise ValueError("resume attached recipe differs")
        if live is not None and saved.get("identity") != live["identity"]:
            raise ValueError("resume attached recipe identity differs")
    for name, module in linears.items():
        saved, expected = states[name], module.state_dict()
        if not isinstance(saved, Mapping) or set(saved) != set(expected):
            raise ValueError("resume parameter inventory differs: " + name)
        for key, reference in expected.items():
            value = saved[key]
            if isinstance(reference, torch.Tensor):
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != reference.shape
                    or value.dtype != reference.dtype
                    or not bool(torch.isfinite(value).all())
                ):
                    raise ValueError("resume invalid parameter: " + name + "." + key)
                if key in ("initial_scale", "frozen_bias") and not torch.equal(
                    value.cpu(), reference.cpu()
                ):
                    raise ValueError("resume changed frozen operands: " + name)
            elif value != reference:
                raise ValueError("resume parameter contract differs: " + name + "." + key)
    if recipes["activation_bank"] is not None:
        from .learned_activation import BOUNDARY_PATHS, LearnedActivationBank

        bank = LearnedActivationBank.from_attached(linears)
        saved = recipes["activation_bank"]
        if set(saved) != {"identity", "parameters"} or set(saved["parameters"]) != set(
            BOUNDARY_PATHS
        ):
            raise ValueError("resume learned boundary inventory differs")
        for boundary, paths in BOUNDARY_PATHS.items():
            quantizer = bank.quantizers[boundary]
            key = "activation_quantizer." + (
                "threshold_delta" if quantizer.bits == 1 else "clip_ratio"
            )
            value = saved["parameters"][boundary]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != torch.Size([])
                or value.dtype != torch.float32
                or not bool(torch.isfinite(value))
            ):
                raise ValueError("resume learned scalar invalid")
            if quantizer.bits != 1 and not 2**-16 <= float(value) <= 1:
                raise ValueError("resume clipping scalar outside contract")
            if any(not torch.equal(states[path][key].cpu(), value.cpu()) for path in paths):
                raise ValueError("resume inconsistent shared activation aliases")
    if recipes["fusion_correction"] is not None:
        from .fusion_correction import FusionCorrection

        correction = linears["fc"].fusion_correction
        # Tiny factors only; validate payload against an isolated module first.
        probe = FusionCorrection(correction.in_features, correction.out_features, correction.config)
        probe.load_payload(recipes["fusion_correction"])
        for key, value in recipes["fusion_correction"]["state"].items():
            if not torch.equal(states["fc"]["fusion_correction." + key].cpu(), value.cpu()):
                raise ValueError("resume inconsistent correction aliases")

    if recipes["affine_binary"] is not None:
        from .affine_binary import AffineBinaryBank, AffineBinaryMidpoint

        bank = AffineBinaryBank.from_attached(linears)
        saved = recipes["affine_binary"]
        probes = {
            p: AffineBinaryMidpoint(row.midpoint.numel(), bank.config)
            for p, row in bank.midpoints.items()
        }
        AffineBinaryBank(bank.config, probes, bank.dimensions).load_payload(saved)
        for path, value in saved["state"].items():
            if not torch.equal(states[path]["affine_binary.midpoint"].cpu(), value.cpu()):
                raise ValueError("resume inconsistent affine midpoint alias")


def deployment_state_sha256(linears: Mapping) -> str:
    """Fingerprint the effective native representation in checkpoint row order.

    Latent magnitudes, scale offsets and F32 factor masters are training state;
    the native drafter uses packed signs, effective F32 scales and rounded F16
    factors. Source identity separately binds frozen norms/embeddings. This
    lets an NPZ/native replay prove the same deployed forward without claiming
    that it reconstructs the training master's backward or optimizer state.
    """
    import hashlib
    import json

    import numpy as np

    digest = hashlib.sha256()

    def add(name, array):
        array = np.ascontiguousarray(array)
        header = json.dumps(
            {"name": name, "shape": list(array.shape), "dtype": array.dtype.str},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(header.encode() + b"\n")
        digest.update(memoryview(array).cast("B"))

    for path in sorted(linears):
        module = linears[path]
        digest.update(
            json.dumps(
                {
                    "path": path,
                    "activation_bits": module.contract.activation_bits,
                    "shape": list(module.latent_sign.shape),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        negative = (module.latent_sign.detach() < 0).cpu().numpy().reshape(-1)
        add(path + ".packed_signs", np.packbits(negative, bitorder="little"))
        add(path + ".scale", module.effective_scales().detach().cpu().numpy())
        if module.frozen_bias is not None:
            add(path + ".bias", module.frozen_bias.detach().cpu().numpy())
        affine = getattr(module, "affine_binary", None)
        if affine is not None:
            add(path + ".midpoint", affine.midpoint.detach().cpu().numpy())
    if any(getattr(m, "activation_quantizer", None) is not None for m in linears.values()):
        from .learned_activation import LearnedActivationBank

        parameters = LearnedActivationBank.from_attached(linears).native_parameters()
        digest.update(
            json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            + b"\n"
        )
    correction = getattr(linears["fc"], "fusion_correction", None)
    if correction is not None:
        descriptor, tensors = correction.native_payload()
        digest.update(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            + b"\n"
        )
        for name, tensor in sorted(tensors.items()):
            add(name, tensor.detach().cpu().numpy())
    return digest.hexdigest()

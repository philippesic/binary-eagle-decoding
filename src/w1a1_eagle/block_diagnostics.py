"""Bounded observational block diagnostics, never native acceptance metrics."""

from __future__ import annotations

import copy
from contextlib import contextmanager

import torch

SAMPLE_SIZE = 128
SAMPLE_INTERVAL = 100


def fixed_indices(size, device):
    count = min(size, SAMPLE_SIZE)
    return torch.arange(count, device=device, dtype=torch.int64) * size // count


def slot_metrics(output, batch, config):
    from .block_qat import block_supported_mask

    with torch.no_grad():
        valid = block_supported_mask(batch, config).to(output.logits.device)
        correct = output.logits.detach().argmax(-1) == batch.labels.to(output.logits.device)
        contiguous = valid.to(torch.int64).cumprod(0).bool()
        survival = correct.to(torch.int64).cumprod(0).bool() & contiguous
        return {
            "slot_supported": valid.to(torch.int64).cpu().tolist(),
            "teacher_forced_top1_hits": (correct & valid).to(torch.int64).cpu().tolist(),
            "captured_prefix_survival_supported": contiguous.to(torch.int64).cpu().tolist(),
            "captured_prefix_survival_hits": survival.to(torch.int64).cpu().tolist(),
            "diagnostic_semantics": (
                "teacher_forced_argmax_and_predicted_prefix_survival_not_native_acceptance"
            ),
        }


def exposure_metadata(dataset, batches):
    chains = getattr(dataset, "chains", {})
    records = []
    for batch in batches:
        source = chains.get(batch.chain_id, {}) if isinstance(chains, dict) else {}
        domain = source.get("domain")
        length = source.get("input_tokens")
        supported = batch.loss_mask & (batch.labels >= 0) & (batch.predecessor_ids >= 0)
        records.append(
            {
                "domain_declared": domain if domain in {"prose", "code", "reasoning"} else None,
                "source_input_tokens_declared": length
                if type(length) is int and length >= 0
                else None,
                "context_tokens_measured": len(batch.prefix_tokens) - 1,
                "supervised_slots": int(supported.sum()),
                "slot_supported": [int(value) for value in supported.tolist()],
            }
        )
    return records


def length_bucket(length):
    if length is None:
        return "unavailable"
    if length <= 64:
        return "0_to_64"
    if length <= 256:
        return "65_to_256"
    if length <= 512:
        return "257_to_512"
    return "over_512"


def accumulate_exposure(history, records, metrics):
    value = history.setdefault(
        "exposure",
        {"domain_declared": {}, "source_input_tokens_declared": {}, "context_tokens_measured": {}},
    )
    for record in records:
        for field, key in (
            ("domain_declared", record["domain_declared"] or "unavailable"),
            ("source_input_tokens_declared", length_bucket(record["source_input_tokens_declared"])),
            ("context_tokens_measured", length_bucket(record["context_tokens_measured"])),
        ):
            counts = value[field].setdefault(key, {"blocks": 0, "supervised_slots": 0})
            counts["blocks"] += 1
            counts["supervised_slots"] += record["supervised_slots"]
            depth = counts.setdefault("slot_supported", [0] * 7)
            for index, count in enumerate(record["slot_supported"]):
                depth[index] += count
    slots = history.setdefault("slot_totals", {})
    for field in (
        "slot_supported",
        "teacher_forced_top1_hits",
        "captured_prefix_survival_supported",
        "captured_prefix_survival_hits",
    ):
        if field in metrics:
            counts = slots.setdefault(field, [0] * 7)
            for i, count in enumerate(metrics[field]):
                counts[i] += count
    slots["semantics"] = "teacher_forced_argmax_and_predicted_prefix_survival_not_native_acceptance"


class BlockDiagnosticSession:
    """Fixed 128-coordinate observations every100 completed optimizer updates.

    Flip-back is between sampled interval endpoints, not every update. State
    retains initial/previous signs and previous endpoint changes for exact resume.
    A8 saturation denotes hard-code endpoint occupancy, not excess clipping.
    """

    def __init__(self, model, history, step):
        self.step = step
        self.sample = step % SAMPLE_INTERVAL == 0
        self.state = copy.deepcopy((history or {}).get("diagnostic_sample_state", {}))
        self.activation = {}
        for name, module in model.binary_linears().items():
            if name not in self.state:
                signs = (
                    (
                        module.latent_sign.detach().reshape(-1)[
                            fixed_indices(module.latent_sign.numel(), module.latent_sign.device)
                        ]
                        < 0
                    )
                    .cpu()
                    .tolist()
                )
                self.state[name] = {
                    "latent_count": module.latent_sign.numel(),
                    "initial": signs,
                    "previous": signs,
                    "changed_previous": [False] * len(signs),
                    "previous_step": step - 1,
                }
            if self.state[name]["latent_count"] != module.latent_sign.numel():
                raise ValueError("diagnostic sample inventory differs from checkpoint")

    @contextmanager
    def capture(self, model):
        previous = {}
        if self.sample:
            for name, module in model.binary_linears().items():
                previous[name] = getattr(module, "_diagnostic_a8", None)

                def collect(codes, name=name):
                    sampled = codes.detach().reshape(-1)[fixed_indices(codes.numel(), codes.device)]
                    values = self.activation.setdefault(
                        name, {"sampled_codes": 0, "abs_code_127": 0}
                    )
                    values["sampled_codes"] += sampled.numel()
                    values["abs_code_127"] += int((sampled.abs() == 127).sum())

                module._diagnostic_a8 = collect
        try:
            yield
        finally:
            for name, module in model.binary_linears().items():
                if name in previous:
                    module._diagnostic_a8 = previous[name]

    @torch.no_grad()
    def finish(self, model):
        reports = {}
        if self.sample:
            for name, module in model.binary_linears().items():
                values = module.latent_sign.detach().reshape(-1)[
                    fixed_indices(module.latent_sign.numel(), module.latent_sign.device)
                ]
                signs = (values < 0).cpu().tolist()
                state = self.state[name]
                changed = [a != b for a, b in zip(signs, state["previous"])]
                scale_indices = fixed_indices(
                    module.initial_scale.numel(), module.initial_scale.device
                )
                scales = (
                    module.initial_scale[scale_indices] + module.scale_offset[scale_indices]
                ).clamp_min(0)
                activation = self.activation.get(name)
                reports[name] = {
                    "latent_samples": values.numel(),
                    "negative": int((values < 0).sum()),
                    "near_zero_abs_le_0_001": int((values.abs() <= 0.001).sum()),
                    "bound_abs_ge_0_999": int((values.abs() >= 0.999).sum()),
                    "sampled_endpoint_sign_changes": sum(changed),
                    "sampled_endpoint_flip_backs": sum(
                        a and b for a, b in zip(changed, state["changed_previous"])
                    ),
                    "signs_equal_initial": sum(a == b for a, b in zip(signs, state["initial"])),
                    "previous_sample_step": state["previous_step"],
                    "scale_samples": scales.numel(),
                    "zero_scales": int((scales == 0).sum()),
                    "scale_min": float(scales.min()),
                    "scale_mean": float(scales.mean()),
                    "scale_max": float(scales.max()),
                    "a8_hard_code_endpoint_occupancy": activation,
                    "a8_status": "measured_fixed_coordinate_sample"
                    if activation
                    else "unavailable",
                }
                state.update(previous=signs, changed_previous=changed, previous_step=self.step)
        return {
            "state": self.state,
            "sample": reports or None,
            "sample_step": self.step,
            "semantics": (
                "fixed_coordinate_sample; interval_endpoint_flip_back; A8_abs_code_127_occupancy"
            ),
        }

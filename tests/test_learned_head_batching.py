"""Production round/adapter/linear path on tiny CPU graphs, no real training."""

import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_qat_cache_head import batch, drafter  # noqa: E402

from w1a1_eagle.continuous_qat import ObservedAdapter  # noqa: E402
from w1a1_eagle.learned_activation import LearnedActivationBank  # noqa: E402
from w1a1_eagle.native_step import NativeStepAdapter  # noqa: E402
from w1a1_eagle.recurrent_binary import CANDIDATE_D_BASE_TO_PATH  # noqa: E402
from w1a1_eagle.recurrent_loss import supported_prefix_ce  # noqa: E402
from w1a1_eagle.recurrent_provider import forward_torch_round  # noqa: E402
from w1a1_eagle.recurrent_qat import shared_round_hard_signs  # noqa: E402
from w1a1_eagle.recurrent_trace import TraceAudit  # noqa: E402


def learned_adapter(bits, *, defaults=False):
    model = drafter(bits)
    linears = {name: model.get_submodule(name) for name in CANDIDATE_D_BASE_TO_PATH.values()}
    bank = LearnedActivationBank(
        bits, {name: module.in_features for name, module in linears.items()}
    )
    bank.attach(linears)
    if not defaults:
        with torch.no_grad():
            for quantizer in bank.quantizers.values():
                quantizer.parameter.fill_(0.125 if bits == 1 else 0.625)
    return NativeStepAdapter(model)


def chain(depth, mask="all", terminal=True):
    original = batch(parent=3, terminal=False)
    seed = original.prefix_token_ids[-1]
    rows = []
    for index in range(depth + int(terminal)):
        valid = index < depth
        proposal = (seed + 1) % 11 if valid else None
        rows.append(
            {
                **original.rows[0],
                "depth": index,
                "input_position": 4 + index,
                "input_token_id": seed,
                "proposed_token_id": proposal,
                "valid": valid,
            }
        )
        seed = proposal
    valid = (True,) * depth + ((False,) if terminal else ())
    ce = tuple(
        index < depth and (mask == "all" or index == depth - 1) for index in range(len(rows))
    )
    labels = tuple((index + 2) % 7 if supported else -1 for index, supported in enumerate(ce))
    audit = TraceAudit(labels, valid, ce, ce, valid, tuple(range(7)), {}, {})
    return replace(original, rows=tuple(rows)), audit


class LearnedHeadBatchingTests(unittest.TestCase):
    def test_trainable_learned_round_gradients_clipping_and_updates_match_serial(self):
        for bits in (1, 4, 8):
            for depth in (2, 4):
                for mask in ("all", "last"):
                    for optimizer_type in (torch.optim.SGD, torch.optim.AdamW):
                        with self.subTest(
                            bits=bits, depth=depth, mask=mask, optimizer=optimizer_type.__name__
                        ):
                            left = learned_adapter(bits)
                            right = NativeStepAdapter(copy.deepcopy(left.drafter))
                            result = []
                            for adapter, optimized in ((left, False), (right, True)):
                                b, audit = chain(depth, mask, terminal=mask == "last")
                                b.raw_target_features.requires_grad_()
                                observer = ObservedAdapter(adapter)
                                metadata, calls = {}, []
                                hook = adapter.drafter.lm_head.register_forward_pre_hook(
                                    lambda _, args: calls.append(tuple(args[0].shape))
                                )
                                parameters = [
                                    p for p in adapter.drafter.parameters() if p.requires_grad
                                ]
                                optimizer = optimizer_type(parameters, lr=0.03)
                                try:
                                    with shared_round_hard_signs(adapter.linears):
                                        logits = forward_torch_round(
                                            b,
                                            observer,
                                            7,
                                            optimize_head=optimized,
                                            execution_metadata=metadata,
                                        )
                                        loss = supported_prefix_ce(logits, audit)
                                        first = observer.first
                                        attached = torch.autograd.grad(
                                            loss,
                                            (
                                                first.pre_norm,
                                                first.cache.key,
                                                first.cache.value,
                                                b.raw_target_features,
                                            ),
                                            retain_graph=True,
                                        )
                                        loss.backward()
                                    grads = [p.grad.detach().clone() for p in parameters]
                                    torch.nn.utils.clip_grad_norm_(parameters, 0.1)
                                    optimizer.step()
                                    result.append(
                                        (
                                            logits.detach(),
                                            loss.detach(),
                                            grads,
                                            attached,
                                            [p.detach().clone() for p in parameters],
                                        )
                                    )
                                finally:
                                    hook.remove()
                                self.assertEqual(calls, [(16,)] * depth)
                                self.assertEqual(metadata["path"], "serial")
                                self.assertEqual(metadata["saturation_scope"], "last_valid_row")
                                self.assertEqual(adapter.head_saturation_scope, "last_valid_row")
                                if optimized:
                                    self.assertEqual(
                                        metadata["reason"], "trainable_learned_activation"
                                    )
                                    self.assertIsNotNone(
                                        adapter.linears[
                                            "lm_head"
                                        ].activation_quantizer.parameter.grad
                                    )
                            for index in (0, 1):
                                torch.testing.assert_close(
                                    result[0][index], result[1][index], rtol=0, atol=0
                                )
                            for index in (2, 3, 4):
                                for serial, requested in zip(result[0][index], result[1][index]):
                                    torch.testing.assert_close(serial, requested, rtol=0, atol=0)
                            # Later-supported rows keep state/K/V and raw input attached.
                            for gradient in result[1][3][:3]:
                                self.assertTrue(bool(torch.isfinite(gradient).all()))
                            self.assertGreater(float(result[1][3][-1][-1].norm()), 0)

    def test_default_parameter_initialization_is_unchanged(self):
        for bits in (1, 4, 8):
            adapter = learned_adapter(bits, defaults=True)
            self.assertEqual(
                float(adapter.linears["lm_head"].activation_quantizer.parameter.detach()),
                0.0 if bits == 1 else 1.0,
            )
            reference = NativeStepAdapter(copy.deepcopy(adapter.drafter))
            b, audit = chain(3, "last")
            results = []
            for model, requested in ((reference, False), (adapter, True)):
                metadata = {}
                logits = forward_torch_round(
                    b,
                    ObservedAdapter(model),
                    7,
                    optimize_head=requested,
                    execution_metadata=metadata,
                )
                supported_prefix_ce(logits, audit).backward()
                results.append(
                    (logits, model.linears["lm_head"].activation_quantizer.parameter.grad)
                )
            torch.testing.assert_close(results[0][0], results[1][0], rtol=0, atol=0)
            torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
            self.assertEqual(metadata["reason"], "trainable_learned_activation")

    def test_fixed_frozen_learned_and_no_grad_still_batch(self):
        for bits in (1, 4, 8):
            for mode in ("fixed", "frozen_learned", "no_grad"):
                with self.subTest(bits=bits, mode=mode):
                    adapter = (
                        NativeStepAdapter(drafter(bits))
                        if mode == "fixed"
                        else learned_adapter(bits)
                    )
                    if mode == "frozen_learned":
                        adapter.linears["lm_head"].activation_quantizer.parameter.requires_grad_(
                            False
                        )
                    serial = NativeStepAdapter(copy.deepcopy(adapter.drafter))
                    b, _ = chain(3, terminal=True)
                    calls, metadata = [], {}
                    hook = adapter.drafter.lm_head.register_forward_pre_hook(
                        lambda _, args: calls.append(tuple(args[0].shape))
                    )
                    context = torch.no_grad() if mode == "no_grad" else torch.enable_grad()
                    try:
                        with context:
                            expected = forward_torch_round(b, ObservedAdapter(serial), 7)
                            actual = forward_torch_round(
                                b,
                                ObservedAdapter(adapter),
                                7,
                                optimize_head=True,
                                execution_metadata=metadata,
                            )
                    finally:
                        hook.remove()
                    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                    if mode != "no_grad":
                        _, a = chain(3, terminal=True)
                        supported_prefix_ce(expected, a).backward()
                        supported_prefix_ce(actual, a).backward()
                        for (name, left), (_, right) in zip(
                            serial.drafter.named_parameters(), adapter.drafter.named_parameters()
                        ):
                            if left.grad is None:
                                self.assertIsNone(right.grad, name)
                            else:
                                torch.testing.assert_close(
                                    left.grad, right.grad, rtol=5e-5, atol=2e-6, msg=name
                                )
                    self.assertEqual(calls, [(3, 16)])
                    self.assertTrue(metadata["effective_batched"])
                    self.assertEqual(metadata["path"], "batched")
                    self.assertEqual(adapter.head_saturation_scope, "valid_chain_mean")

    def test_paired_measurement_reports_effective_fallback_without_update(self):
        import importlib.util
        from types import SimpleNamespace

        from w1a1_eagle.continuous_qat import later_gradient
        from w1a1_eagle.recurrent_qat import joint_parameter_families

        spec = importlib.util.spec_from_file_location(
            "learned_head_readiness", ROOT / "scripts/check_qat_optimization_readiness.py"
        )
        tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tool)
        adapter = learned_adapter(8)
        parameters = [p for p in adapter.drafter.parameters() if p.requires_grad]
        lane = SimpleNamespace(
            name="A8",
            adapter=adapter,
            linears=adapter.linears,
            optimizer=torch.optim.AdamW(parameters),
        )
        config = SimpleNamespace(device="cpu", context_chunk_size=2, depth_loss_decay=1.0)
        trainer = SimpleNamespace(
            config=config,
            lanes=[lane],
            provider=SimpleNamespace(draft_vocab_size=7),
            resources=lambda: {"fixture_only": True},
        )
        fake_torch = SimpleNamespace(
            stack=torch.stack,
            isfinite=torch.isfinite,
            cuda=SimpleNamespace(
                reset_peak_memory_stats=lambda _: None, synchronize=lambda _: None
            ),
        )
        api = SimpleNamespace(
            torch=fake_torch,
            shared_round_hard_signs=shared_round_hard_signs,
            ObservedAdapter=ObservedAdapter,
            forward_torch_round=forward_torch_round,
            later_gradient=later_gradient,
            depth_weighted_supported_ce=lambda logits, audit, _depths, decay: supported_prefix_ce(
                logits, audit
            ),
            joint_parameter_families=joint_parameter_families,
        )
        before = tool.deterministic_state_sha256(adapter.linears)
        with (
            tool.zero_updates([lane]),
            patch.object(tool, "memory_snapshot", return_value={"fixture_only": True}),
            patch.object(torch.cuda, "is_available", side_effect=AssertionError("GPU discovery")),
        ):
            row, _ = tool.measure_lane(api, trainer, lane, [chain(3, "last")], (False, True))
        self.assertTrue(row["head_optimization_requested"])
        self.assertFalse(row["head_optimized"])
        self.assertEqual(row["head_execution"][0]["reason"], "trainable_learned_activation")
        self.assertEqual(row["head_execution"][0]["saturation_scope"], "last_valid_row")
        self.assertEqual(row["optimizer_updates"], 0)
        self.assertEqual(dict(lane.optimizer.state), {})
        self.assertEqual(tool.deterministic_state_sha256(adapter.linears), before)

    def test_observed_wrapper_missing_head_visibility_fails_closed(self):
        class HiddenHead:
            def __init__(self, adapter):
                self.adapter = adapter

            def __getattr__(self, name):
                if name == "linears":
                    raise AttributeError(name)
                return getattr(self.adapter, name)

        adapter = learned_adapter(4)
        observer = ObservedAdapter(HiddenHead(adapter))
        metadata = {}
        with patch.object(
            adapter, "decode_head", side_effect=AssertionError("invisible fast head")
        ):
            result = forward_torch_round(
                chain(3)[0], observer, 7, optimize_head=True, execution_metadata=metadata
            )
        self.assertEqual(result.shape, (4, 7))
        self.assertIsNotNone(observer.first)
        self.assertEqual(metadata["reason"], "head_contract_unavailable")
        self.assertFalse(metadata["effective_batched"])

    def test_single_valid_and_all_invalid_paths_are_reported(self):
        adapter = learned_adapter(1)
        metadata = {}
        forward_torch_round(
            chain(1)[0], adapter, 7, optimize_head=True, execution_metadata=metadata
        )
        self.assertEqual(metadata["reason"], "trainable_learned_activation")
        b, _ = chain(1)
        b = replace(b, rows=({**b.rows[0], "valid": False, "proposed_token_id": None},))
        with (
            patch.object(adapter, "decode_step", side_effect=AssertionError("invalid body")),
            patch.object(adapter, "decode_head", side_effect=AssertionError("invalid head")),
        ):
            logits = forward_torch_round(
                b, adapter, 7, optimize_cache=True, optimize_head=True, execution_metadata=metadata
            )
        self.assertTrue(torch.equal(logits, torch.zeros(1, 7)))
        self.assertEqual(metadata["path"], "skipped")
        self.assertEqual(metadata["saturation_scope"], "none")


if __name__ == "__main__":
    unittest.main()

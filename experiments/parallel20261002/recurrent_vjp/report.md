# Current-student recurrence and cache VJP audit

The composed CPU training graph passed the declared equivalence gate. No
production patch is proposed. Context-cache chunking, per-chain head batching,
round-shared signs, and the A1 single-forward computation retained the serial
oracle's hard logits, later-depth loss, input VJP, and all 18 binary parameter
VJPs within `atol=2e-6, rtol=5e-5`. This falsifies a graph-detach concern for the
tested fixed-activation path; learned activation normalization belongs to the
separate activation-batching study.

## Tested source contract

Source base: `6c613039d2c6bef3f6f7b5c6e9bf5f8cbccac2e8`. The exact SHA-256
manifest and per-case scalar errors are in [results.json](results.json).
Production source was read-only. The constructor supplies all nine real
`RowBinaryLinear` projections to `NativeStepAdapter`; no attention/body/cache
stand-in replaces its graph. Hidden width is 8, heads are 2, KV heads are 1,
intermediate width is 13, vocabulary is 7, and fixed PRNG seeds are 811/912.

`forward_torch_round` exposes `optimize_cache`, `optimize_head`, and positive
`context_chunk_size`. The `f32` NativeStep mode supports both optimizations.
It rebuilds shifted accepted-context K/V under `no_grad`; those context values
are an intentional truncation boundary. The deferred seed feature is encoded
with gradients, and each proposal consumes the preceding attached pre-norm
state and F16-rounded attached K/V append. All trainable sign and row-scale
parameters are included in the VJP comparison. Learned clips/thresholds are
absent, and embeddings/norm weights are frozen by the adapter.

There is no supported activation-checkpointing flag in this path. The tested
checkpoint boundary is a model/optimizer state restoration between updates.
The real `CurriculumRunner._forward` is called with its required fields in a
small synthetic runner stub; this does not test the runner's complete durable
checkpoint, curriculum, budget, provider, or ancestry state machine.

Diagnostic attention modes do not support optimized cache/head paths:
`native_forward_f32_backward` requires its separately validated native oracle,
and `native_cpu_diagnostic` is forward-only. This study does not substitute a
synthetic native oracle and make a native numerical claim. Tree masks, holes,
and nonterminal invalid rows are unsupported by the actual graph. Rollout
accepts one exact captured contiguous chain per round, with terminal invalid
padding; it does not implement verifier branch selection or a tree rollback.

The independent serial oracle explicitly loops context rows under `no_grad`,
then loops valid proposals carrying state/cache. It does not call provider,
prefix-rebuild, or rollout helpers. Adapter arithmetic remains common so that
this is an integration/topology audit, building on the existing independent
operand checks in `test_qat_computation.py`, rather than a new derivative claim
for hard quantizers. Finite differences of their hard forward would not test
the declared STE and were not used.

## Results

All 216 comparisons passed on Apple arm64 CPU, PyTorch 2.14.0, F32 arithmetic
with the actual F16 K/V round trip. Each precision has 72 comparisons:
prefix lengths 0/1/3, valid depths 1/3, terminal padding off/on, six controls
covering serial, cache only, head only, both, chunks 1/2/64, shared signs
off/on, and A1 reference/single-forward. The single-forward setting is inert
for A4/A8. Every scalar loss is the same CE at the last valid depth, avoiding
a changed loss normalization or update cadence.

| Fixed activation | Comparisons | Max absolute logit error | Max absolute VJP error |
|---|---:|---:|---:|
| A1 | 72 | 0 | 2.3841858e-7 |
| A4 | 72 | 5.9604645e-8 | 0 |
| A8 | 72 | 5.9604645e-8 | 0 |

Accepted-context raw-feature gradients were zero; deferred seed gradients were
nonzero in every case. Terminal invalid logits were exact zero. The independent
validator separately requires nonzero early state/K/V VJPs, detects state and
cache detach while preserving hard logits, and checks invalid-row decoding and
the context truncation boundary; see its [report](../../../research/parallel20261002/recurrent_vjp/validation/independent-vjp.md).

Two synthetic clipped AdamW updates were compared for each precision. Both
updates agree with explicit serial execution, including binary signs and row
scales; restored second-step gradients/parameters agree exactly with the
uninterrupted path. Optimizer moments have step 2 for every parameter.
Maximum second-step discrepancy is 0 for A1, 5.9604645e-8 for A4, and
1.1920929e-7 for A8. This uses tiny synthetic optimizer updates only.

The controls also demonstrate detection sensitivity:

- A deliberate V-scale perturbation of +0.08 changes rebuilt accepted-context
  cache values by 0.4770508. Reusing the old detached context changes logits by
  0.03090015, exceeding the declared gate. The production provider rebuilds.
- A length-4 proposal cache is excluded when rebuilding a length-1 accepted
  prefix for the next captured round. The next round gets a fresh detached
  context-cache object and identical fresh logits. Attempting to feed the
  length-4 cache at position 1 fails closed. This checks the supported rebuild
  boundary without claiming a serving verifier or branching implementation.

## Reproduction and integration

From the repository root with CPU Torch installed:

```sh
PYTHONPATH=src:. python research/parallel20261002/recurrent_vjp/reference/audit.py
PYTHONPATH=src:. python -m unittest discover -s tests -p 'test_parallel20261002_recurrent_vjp*.py' -v
```

`audit.py` checks the root's shared research-stop control before its bounded
chunks and writes only synthetic scalar report data. No model weights,
captures, raw tensors, real optimizer updates, GPU/Metal, SSH, paid credits,
reserved-final data, or production recipe/source changes are involved.

Final acceptance: the combined reference/independent suite passed **7/7** in
1.395 seconds on the shared CPU PyTorch 2.14.0 environment; Ruff lint and
format checks passed for all four Python files. The separate audit invocation
passed all 216 comparisons. Every pinned source hash also matched the root's
main checkout when checked before integration.

Integrate the two uniquely named tests and research/report files after review.
No live source change is needed. The useful regression gates are the composed
mode matrix, restored second step, stale-context detection, and independent
detach/mask checks. Runtime-native trajectory correctness, actual CUDA
acceptance/throughput versus Q4_0, GPU memory/timing, real-model shape regimes,
learned activation VJPs, and full durable runner resume remain separate gates.

The team checkpoint for the active QAT optimization readiness goal is this
report plus `results.json` and the independent report. Root owns the shared
`docs/goals/qat-optimization-readiness.md` integration checkpoint, avoiding
concurrent edits to that goal file.

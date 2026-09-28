# Joint binary body and head: CPU preparation and frozen trial protocol

**Status:** CPU implementation in progress. The user has prohibited all GPU and
accelerator access for this goal. No real-model training, capture, native model
run, or quality measurement is authorized under that restriction. The first
future GPU step below is conditional on an explicit change to that restriction.

## Objective and representation

Start from candidate D of the [scale screen](binary-scale-fitting-5080.md):
the pinned one-layer Qwen3-4B EAGLE drafter with all nine selected matrices
represented by packed binary signs and F32 group-128 scales. Keep the pinned
FP16 Qwen3-4B target and verifier, target feature taps, draft vocabulary map,
borrowed target embedding, norms, RoPE, and nonselected tensors fixed. Train
the fusion, attention Q/K/V/output, FFN gate/up/down, and vocabulary head
**together**. This is W1A16; activation binarization is outside this goal.

Every trained forward must use hard signs with `sign(0)=+1`, nonnegative F32
group scales, and the native A16 boundary: each selected linear receives
`F32 -> F16 -> F32` input values. For each output row and 128-input group,
sum signed input values in F32, multiply by that group's F32 scale, then sum
groups in order in F32. Retain the exact D sign convention and canonical GGUF
Q/K row permutation. FP32 latent sign parameters may exist in a training
checkpoint but may never become an ordinary dense forward or deployment
tensor. A declared clipped straight-through surrogate supplies sign gradients;
scale gradients use the actual hard-binary forward and project scales to
`>=0` after the optimizer step. D's fitted representation permits zero scales,
so a strictly positive parameterization would change its initialization.

The current native v2/v3 loader recognizes `f32_nonnegative_least_squares`
as D's **origin**, not the learned scale rule of a trained export. A trained
GGUF must use a truthful new scale-rule value and a loader contract change;
until then, native trained-model loading is a stop gate. Export only packed
I32 sign words, F32 group scales, the unchanged nonselected tensors and
explicitly declared exceptions (none in the primary trial). Audit each
tensor's shape, type, bits, source hash, mapping and packed roundtrip. The
frozen-body fitted FP16 head is a diagnostic control, not a training baseline
or an allowable final head.

## Recurrent state and supervision contract

Use the native greedy sample-and-match chain, maximum five proposals per
round. At memory/RoPE position `P`, the draft decoder receives
`(token[P+1], fused target features[P])`; its **pre-norm** hidden state feeds
the next proposal step. The saved head-only captures hold output-normalized
head inputs and cannot substitute for the decoder input, pre-norm state, or
draft K/V cache. Within a short unroll, student pre-norm states and the new
student K/V entries must stay differentiable. Preserve the pinned attention
mask, absolute positions, prefix ancestry, target-feature tap order, deferred
seed pair, and accepted-prefix cache truncation. Reset/detach only at the
bounded unroll boundary, never between its proposal positions.

The target is frozen and supplies a next-token label/logit row for the **exact
proposal prefix**. Use the native verifier sampler cloned and advanced with
the proposed token even after a rejection; raw target argmax can differ from
the sampler label. A training record must join prompt, round, parent node,
prefix IDs, input position, label position, verifier row, proposal depth,
validity/reached/EOS masks, and draft-to-target vocabulary mapping. For
depth `d` under parent position `P`, input position is `P+d+1`, label
position is `P+d+2`, and the next row's prefix must append the previous
proposal token. In the checkpoint `d2t[i]` is an **offset**, so target ID is
`i+d2t[i]`; GGUF stores absolute IDs. Unsupported target labels and target
probability mass outside the 32,000-token draft vocabulary must be counted
and reported. CE applies only to supported, valid, reached rows; unsupported
rows remain in all coverage and agreement denominators. Padding, pruned
branches, post-EOS rows and bonus target tokens cannot silently become
training labels. No head-only cached-state or teacher-draft KL loss replaces
this target-verifier supervision.

Use scheduled teacher-forced proposal prefixes from the native capture for a
bounded, differentiable unroll. This gives exact-prefix labels and allows a
later-position loss to backpropagate through earlier student body states and
K/V. Captured D prefixes become off-policy if training changes proposal IDs;
their labels must never be reused as labels for a different current-student
prefix. It does **not** by itself measure live-chain acceptance: export and run
the native sampler to check that. Capture must include sufficient target
features, token IDs, positions, mask/KV boundary records, verifier labels and
logits to replay each prefix without reading frozen final prompts.

## Bounded proposed first trial and stop gates

The existing prompt manifests stay fixed: 96 training prompts, 24 development
prompts, 24 reserved final prompts. Training and development SHA256 values are
`80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74`
and `a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885`.
Do not load or evaluate the reserved final prompts in this exploratory stage.
Development evidence is reused and has no new-prompt generalization claim.

**Proposed budget, to freeze before any future training:** one initialization
and one optimizer recipe; at most 8,160 training unroll starts selected
deterministically and evenly over the 96 prompts, each at most five proposal
positions; at most 500 optimizer steps or 45 optimizer minutes, whichever
comes first. Save step 0, 100, 250 and 500 plus a finite time-cap endpoint.
No hyperparameter search or training-state refresh is in this trial. Record
the selected-state manifest and all settings before optimization. The old
500-step/45-minute **head-only** budget does not automatically authorize this
larger trial; the user owns a budget amendment before GPU execution.

Stop before optimization if a captured prefix/verifier join fails, mapping or
support accounting is inconsistent, a student recurrence disagrees with a
fixed-prefix native reference in masks/positions/KV boundaries, or CPU/native
operator/export parity fails. Stop during optimization for nonfinite values,
wrong trainable/frozen parameter ownership, or a time/step cap. Save an audit
checkpoint on every stop. Select only by pooled online development accepted
drafts per complete round; target CE breaks exact ties. If no trained export
improves D's `909/2139 = 0.4250` development accepted/round, stop the recipe.
For a promoted candidate, compare raw IDs, acceptance, complete-round latency
and full-request throughput on matched native runs. **Q4_0 EAGLE is primary:**
its reference development acceptance is `1555/1493 = 1.0415`, and its clean
five-repetition full-request rate was 135.1 tokens/s. FP16 EAGLE and
target-only are secondary controls. A candidate cannot be called a Q4_0 win
until matched native GPU tests, including graph behavior, are authorized and
completed. No existing CPU or 5080 result establishes SM75 performance.

## CPU work and remaining gate

CPU tests now validate hard-sign/group-scale forward and backward, gradients
from a later two-call unroll loss to earlier body states and functional
causal-cache key/value entries, exact-prefix/offset/mask joins on synthetic
traces, sign/scale serialization, and scalar replay.
The repaired scale surrogate passes gradient at an exact zero scale and
projects any negative update back to zero. The CPU reference also matches
430/430 **archived D native sampled outputs** across all nine projections;
these are previously recorded training-capture operands, not a trained model.
The sequential per-feature PyTorch forward is deliberately an exact CPU
reference and is too slow for the full nine-linear model. An explicitly
selected grouped-F32-matmul training path retains the hard signs, A16 casts,
F32 group scales and ordered group accumulation, with different in-group
reduction order. On the same 430 archived D outputs it matched 395 exactly;
maximum absolute discrepancy was `0.0001220703125` and maximum
`|delta|/(1+|native|)` was `7.422315e-7`. This is a numerical diagnostic,
not evidence of equal proposal IDs or acceptance near ties. The training
checkpoint records which arithmetic path was used. Replacing the exact
forward with grouped matmul for a full run remains a documented research
choice requiring a fixed-input native argmax/trajectory parity gate.

A CPU prefix-rebuild helper now recomputes the current student cache from
accepted-prefix token `t[j+1]` and raw target feature `f[j]` at decoder
position `j`, then defers `f[P]` and seed `t[P+1]` for position `P`. It
recreates context under an explicit truncated-gradient boundary; new proposal
states and K/V stay differentiable. Native `input_position=P+d+1` is the
shifted **token** index, while decoder memory/RoPE position is `P+d`.
Synthetic tests cover this pairing and stale-cache elimination. Native
feature extraction and actual cache parity remain unverified.

This is necessary
preparation, not evidence about real-model quality or throughput. The native
model capture of raw target features, an efficient hard-binary training
forward with accepted numerical differences, student cache parity, the
trained-scale GGUF loader amendment, real-model training, and matched native
evaluation remain unverified. The first future GPU-owner operation,
**only if the user changes the no-GPU restriction and approves the budget,**
begins with the CPU preflight command below, followed by a training-split-only
native capture/alignment gate using the pinned
target, D draft and exact commands/versions recorded in a new run directory;
validate prefix ancestry, unsupported-label mass and fixed-prefix recurrence
before starting optimizer steps. No remote command is prescribed while the
required capture implementation is incomplete.

```sh
CUDA_VISIBLE_DEVICES='' PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src \
  .venv/bin/python -m unittest discover -s tests -p 'test_recurrent_*.py'
```

Once a native capture exists, its first CPU metadata gate is
`scripts/audit_recurrent_binary_capture.py --manifest <capture-manifest>
--prompts data/qat-revisit/train.jsonl --output <new-audit-json>`. That gate
requires the frozen 96-prompt train hash and checks proposal ancestry,
offset-form `d2t` with the `t2d` inverse mask, exact-prefix verifier labels, masks and mapped probability
mass where full logits were saved. It explicitly leaves target-feature and
draft-cache parity unverified; those need separate fixed-prefix model checks.

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

The former native v2/v3 loader recognized `f32_nonnegative_least_squares`
as D's **origin**, not the learned scale rule of a trained export. Fork commit
`7f23c89b3` now also accepts truthful `f32_learned_nonnegative` metadata;
eight CPU-only native loader fixtures pass, including rejection of an unknown
scale rule. This validates loader declaration, not numerical behavior of a
trained model. Export only packed
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
and reported. CE applies to supported, valid exact-prefix proposal rows,
including cloned-verifier rows beyond an earlier live rejection when their
target logits were actually computed. `verifier_reached` remains a separate
diagnostic mask for online acceptance and must not silently censor those
teacher-forced later-position losses. Unsupported
rows remain in all coverage and agreement denominators. Padding, pruned
branches, post-EOS rows and bonus target tokens cannot silently become
training labels. No head-only cached-state or teacher-draft KL loss replaces
this target-verifier supervision.

Fork commit `c282087a9` adds an opt-in bounded raw target-logit file beside
the existing head capture: `EAGLE_CAPTURE_TARGET_LOGITS=1` writes
`<EAGLE_CAPTURE_PREFIX>.target_logits.f32`, with a default 32-row cap and
`EAGLE_CAPTURE_TARGET_LOGITS_LIMIT` override. It copies target logits before
the cloned sampler runs and writes a distinct row index/source in each head
record. The existing `EAGLE_CAPTURE_FULL_LOGITS` file contains mapped
**draft-head** logits and remains separate. The CPU audit joins only the
raw-target file by row index, dimensions and exact provenance; malformed,
missing or substituted files fail. A CPU-only server build passed, but no
real-model capture has exercised this new writer.

Use scheduled teacher-forced proposal prefixes from the native capture for a
bounded, differentiable unroll. This gives exact-prefix labels and allows a
later-position loss to backpropagate through earlier student body states and
K/V. Captured D prefixes become off-policy if training changes proposal IDs;
their labels must never be reused as labels for a different current-student
prefix. It does **not** by itself measure live-chain acceptance: export and run
the native sampler to check that. Capture must include sufficient target
features, token IDs, positions, mask/KV boundary records, verifier labels and
logits to replay each prefix without reading frozen final prompts.

The CPU capture gate expects two feature-ledger files in addition to proposal
rows, round anchors, offset `d2t` and boolean `t2d`: a hashed F32 `.npy` matrix
of 7,680-wide raw target-feature rows and a hashed JSONL with one row per
matrix row. Each ledger row declares its prompt, absolute target position,
accepted-prefix token IDs through that position, ordered taps `[2,18,33]`,
and boundary `native_target_block_inputs_concat_before_draft_fc`. Every
anchor must find feature rows for all positions `0..P`; no unjoined or
rejected-prefix feature row is admitted. The last row `f[P]` is deferred for
the seed, while earlier rows rebuild the current student context cache.
This metadata gate cannot prove the numerical feature values came from the
pinned target; native fixed-prefix replay must do that.

Fork commit `ddcf2a608` adds an opt-in raw target-feature stream under
`EAGLE_CAPTURE_TARGET_FEATURES=1` and `EAGLE_CAPTURE_PREFIX`. It writes
`<prefix>.target_features.f32` and `.target_features.jsonl` after successful
target decode and before EAGLE fusion. Each decoded row records task, slot,
decode ordinal, local and global batch indices, absolute token position,
complete through-token prefix, ordered tap IDs and F32 source boundary.
A disposition event records whether that input joined retained history. If
`A` drafts were accepted, verifier input rows `j=0..A` (seed and accepted
draft inputs) are retained and later rows are rejected suffix. The default
4,096-row budget fails the run on overflow; a positive
`EAGLE_CAPTURE_TARGET_FEATURES_LIMIT` can be set before capture. The first
capture contract rejects context shift, prompt reuse, multimodal inputs,
checkpoint replay and multiple slots. A CPU-only server compilation passed,
but no real-model run has validated row order or feature values.

The CPU preparers now join native head/round records to prompt IDs,
offset-form vocabulary maps and optional raw target logits; separately, they
filter the raw feature stream by disposition and exact accepted-prefix
ancestry into the feature ledger required above. A request cell manifest can
verify task-to-prompt ownership and file hashes. A plain task map alone is
reported as unverified and is insufficient to start training. Complete
request/terminal continuity, pinned model identity in captured rows, and
actual target/drafter numerical parity remain stop gates.

The `recurrent-train` capture-runner mode prepares the exact frozen 96-prompt
D/D own-history cell. It pins target, draft and absolute D vocabulary-map
hashes, requests both raw target streams with explicit positive limits,
disables context shift, prompt-cache reuse and checkpoints, and records
request task/ranges and raw-file hashes. This is a future run protocol only:
its tests use mocked CPU server files, and it has not been used to launch a
native model. Its cell manifest and task map feed both native preparers.
`scripts/audit_recurrent_continuity.py` then checks prefill, each speculative
input/disposition and successive accepted prefixes against canonical round
records. It deliberately leaves the initial target sample, terminal emission,
stopping, request completeness and numerical feature values unproved.
`scripts/build_recurrent_capture_bundle.py` joins the cell and preparer
hashes, checks pinned target/draft file hashes and the canonical D map
digest, runs the final CPU capture audit and publishes a self-contained
preparation bundle marked `training_eligible: false`. No current CPU result
certifies what a future server executed, native training eligibility or model
quality.

A [one-prompt CPU-only model capture](recurrent-binary-cpu-capture-smoke.md)
now exercises the native writers end to end with the pinned target and D
draft. It exposed a missing `GGML_W1AX_ACT_BITS=16` setting in the runner;
the corrected diagnostic passed continuity, both preparers and the final
bundle audit. This validates capture format and the local joins for one real
request. It does not establish numerical target-feature or drafter-cache
parity, full 96-prompt capture completeness, training quality or Q4_0 gates.

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

The CPU `NativeStepAdapter` now runs the pinned one-layer graph structure
without calling AngelSlim's inference-only tree generator: frozen F16 borrowed
embeddings, frozen F32 RMS norms, embedding-first concatenation, binary
Q/K/V/output and FFN, RoPE, F16-rounded K/V cache writes before causal
attention, residuals, output norm and binary head. Its two-step and
prefix-rebuild-to-optimizer synthetic tests pass. The actual pinned model's
numeric parity with native GGML remains a stop gate; PyTorch norm, RoPE,
attention and reduction arithmetic may differ. A strict CPU audit of the
unchanged D GGUF found all nine packed pairs and 17,005 exact-zero scales;
this preserves D initialization, not model quality.
The frozen-input adapter verifies the pinned FP16 target GGUF and D GGUF
hashes, memory-maps the target's F16 token embedding, copies only requested
rows, and binds copied F32 draft norm vectors. A CPU read of one real token
row passed without a target forward or full embedding copy. This is operand
identity evidence, not target-feature capture or model-output parity.

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
trained-export native numerical parity, real-model training, and matched native
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

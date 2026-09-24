# INT4 and INT8 drafter acceptance plan

**Status:** implementation and held-out CUDA run pending. This is a follow-on
acceptance measurement, not a native INT4/INT8 speed claim.

## Question and fixed comparison

Measure accepted draft tokens per verification round for post-training INT4
and INT8 simulations on the pinned RTX 5080 target/drafter pair. Reuse the
same 12 held-out prompts, greedy non-thinking template, tree settings, target
weights, and AngelSlim verifier as the W1A1 CUDA experiment. Run ordinary BF16
EAGLE again in the same process as the low-bit variants. Preserve the strict
target-only parity result and label the low-bit counts verifier-relative if
the known BF16 tree-versus-prefix mismatch persists.

The initial interpretation is **W4A4 and W8A8** on all nine drafter-owned
candidate linears: symmetric signed uniform codes with ranges `[-7, 7]` and
`[-127, 127]`, per-output-row weight and per-input-token activation absmax
scales, nearest-even rounding, and explicit zero-vector behavior. Accumulate
the integer-valued operands in floating point for the PyTorch simulation and
cast the linear result back to the drafter dtype. Record the exact simulated
math and compare it with a small independent numerical reference. The user has
been asked whether they instead intend weight-only W4A16/W8A16; the runner
should make that choice explicit and keep the two protocols distinct.

The selected drafter modules are feature fusion, attention Q/K/V/O, FFN
gate/up/down, and the drafter-owned vocabulary head. Embeddings, normalizations,
attention softmax, residuals, target verification, and other scaffolding remain
at their existing precision. This is post-training quantization; no QAT is in
scope.

## Completion evidence

- A selectable INT4/INT8 fake-quantized drafter path with focused numerical
  tests for scale broadcasting, clipping/rounding, zero vectors, and disabled
  wrapper parity. Record exact operand and accumulation precision.
- One pinned 12-prompt RTX 5080 run with ordinary BF16 EAGLE and both low-bit
  variants under matched settings. Preserve per-prompt accepted/proposed nodes,
  rounds, output lengths, target-greedy mismatch flags, config, model and code
  revisions, environment, exact commands, raw artifact hashes, and GPU cleanup.
- A compact report comparing accepted-per-round values with ordinary BF16 and
  the prior W1A1 result. Separate observations under the BF16 verifier from
  any target-equivalence or real-hardware throughput claim.

## Work ownership

1. Core quantizer implementation and CPU numerical tests in an isolated
   worktree; no GPU use.
2. Orchestrator integration of selectable wrappers/config/runner and review of
   the quantizer; no concurrent edits to the core quantizer files.
3. One Luna GPU operator owns the 5080 for the supervised held-out run after
   current code is pushed and a fresh idle check passes.

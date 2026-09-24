# INT4 and INT8 drafter acceptance plan

**Status:** implementation integrated; held-out CUDA run pending. This is a follow-on
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
keeps that mode explicit and distinct. The W4A4/W8A8 CUDA run disables TF32
matmul shortcuts during FP32 numerical simulation; this setting is recorded
in the environment manifest.

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

## Current checkpoint

- `01ce2fb` on pushed `main` contains the fake uniform quantizer, selective
  reversible adapter, explicit per-variant protocol metadata, and the matched
  `configs/pytorch_int4_int8.toml` config. The old W1A1 config and strict
  parity default remain available.
- `make check` passes all 30 tests. The new core's independent scalar checks
  cover 4/8-bit codes, nearest-even rounding, saturation, zero rows/vectors,
  scale broadcasting, BF16 output, weight-cache refresh, and exact disabled
  wrapper parity. Temporary feature worktrees/branches were integrated and
  removed; `main` is clean.
- GPU acceptance has not run. Use a fresh host/process/memory check via tmux
  MCP, regenerate a model manifest for the new config hash, validate one
  prompt, then run all 12 prompts if memory and the verifier path are stable.
- Luna operator `/root/cuda_acceptance_operator` exclusively owns the RTX 5080
  measurement. Initial Windows/WSL idle gate passed: no Fortnite or project
  job, three 0% samples, 3,108 MiB used / 12,870 MiB free. It is fetching the
  pushed low-bit code and will hold a fixed exact revision across smoke and
  full sweep.
- Remote code is clean at `92c91a4cecdd0123a2e1d7bf7c6d394fa673dabe`.
  New config SHA256 is
  `7ac30ee431e9b2583598ab68cb28731daca6edb5e453ebd1c300578621752c3a`;
  model manifest SHA256 is
  `872cc50776ce1e5adb9494e1822cadaec37fbc81338dd1a7b455057907c1f807`.
  All 18 pinned model files matched prior hashes, and the prompt SHA was
  unchanged.
- Supervised one-prompt `cuda-int4-int8-smoke-20260924` finished exit 0.
  The disabled wrapper matched ordinary EAGLE exactly; all three settings
  recorded the known target-greedy mismatch at index 3. On `prose-01`,
  accepted/round was ordinary 84/47 = 1.787, W4A4 25/105 = 0.238, and W8A8
  80/52 = 1.538. The environment confirms BF16 target/drafter outputs, FP32
  simulated accumulation, TF32 disabled, and highest FP32 matmul precision.
  GPU memory returned to its 3,108 MiB Windows idle baseline. These are
  exploratory one-prompt counts; the 12-prompt sweep is next.
- After a fresh idle gate, supervised `cuda-int4-int8-12prompt-20260924` is
  running all 12 prompts for ordinary, W4A4, and W8A8 under fixed remote code
  `92c91a4cecdd0123a2e1d7bf7c6d394fa673dabe`. Supervisor PID/PGID 866
  is active. Ordinary completed 12 rows; W4A4 is progressing. WDDM sampled
  15,123 MiB used / 855 MiB free at 01:54:39 with no OOM/error. The operator
  is watching memory; do not start another GPU job concurrently.

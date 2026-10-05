# Exact indexed block teacher storage

Bounded source implementation for the active nine-model QAT goal. Native base
`cc9cab3c64f61580cf63e5ef050b075b11cd1fb9`; additive native commit
`ecff6d4e74814c631801df2e74f562d4ed6bd0eb` is published on the user's llama.cpp
fork as `feature/indexed-block-teacher`. Parent branch has the same name, in
`/tmp/binary-eagle-indexed-teacher`; root reviews and integrates it. The live
5080 QAT source, configuration, lane, target/verifier precision and F16 KV remain
unchanged. No GPU, SSH, real model loading or new real capture was performed.

## Storage contract

Dense APIs remain the default: native `all/last/none`, client defaults and plans
without `teacher_logits_layout` retain their previous behavior. An externally
SHA-pinned nine-model exact-soft plan can explicitly select
`"teacher_logits_layout": "indexed"`. The planner rejects that option for hard
CE. This implementation chooses no new source subset, prompt exposure or compute
budget; those decisions remain with the user.

Native indexed mode preserves `batch.logits[i]=true` for every token, identical
prefill/greedy decode partitioning, F16 KV and the original full-vocabulary head.
Selection occurs only at the raw F32 output write after native computation and
finite checking; values are neither quantized nor recomputed. All exact tokens
and all five-tap features remain retained, including the complete prompt/context.
No current-student prefix substitution is introduced.

Replay requests provide a sorted, unique, in-range integer `logits_indices`
list of absolute token positions. Generated requests provide
`logits_selection={"kind":"block_anchors","stride":7,"horizon":7}`. The native
helper and client derive anchors from the same existing rule:
`range(prompt_length - 1, len(tokens) - 7, 7)`. Output rows are the sorted union
of `range(anchor, anchor + 7)` for every selected anchor, even if a static DSpark
prefix currently diverges or a slot currently has no loss. Generated output
writes candidate rows from `prompt_length - 1`, then truncates the incomplete
final horizon after the actual continuation terminates, preserving the previous
anchor boundary policy. Receipts expose the explicit row map and final shape.

The client binds returned maps to the issued replay positions or generated
selection plus actual native prompt/token boundaries. Import preserves the native
map in each chain and binds it into the target producer receipt. Dataset admission
rejects absent, duplicated, unsorted, noninteger, out-of-range, wrong-position and
source-misaligned maps; requires exactly the complete selected-anchor union; and
requires each stored row to have the complete vocabulary. Per-block lookup uses
absolute native positions and copies individual rows into the same seven-row F32
teacher tensor, preserving the existing per-block allocation bound. Tokens,
labels, predecessors, complete context features, positions, attention, prefix
hashes, static divergence masking and loss mathematics are unchanged.

## Costs and limits

For native width 2560 and vocabulary 151936, each complete five-tap feature row
is 51,200 bytes and each full-vocabulary F32 teacher row is 607,744 bytes.
Indexed generated capture retains `7 * floor(actual_generated_tokens / 7)`
teacher rows under the current anchor rule; it can temporarily write
`actual_generated_tokens + 1` teacher rows before final truncation. Cost evidence
separates full feature bytes, retained teacher rows/bytes and peak teacher rows,
and adds bounded metadata headroom for the seven persisted index-map copies.
The existing request/source/host/disk/time caps remain enforced. Source selection
is still the external pinned plan. Cost row counts concern the selected plan,
not an unselected complete corpus.

At 3.9 million token rows, features alone consume 199,680,000,000 bytes before
metadata. Dense feature-plus-teacher rows would consume 2,569,881,600,000 bytes.
Indexed output removes unconsumed teacher rows (principally prompt logits) while
retaining all potentially eligible block rows. It cannot establish complete
corpus feasibility against roughly 197 GB currently reported free storage, and
it does not reduce dense native head computation or certify production readiness.
A full-corpus/exposure/capture-duration decision is outside this implementation.

## Acceptance evidence

All checks ran on local macOS arm64 CPU, with synthetic arrays/API receipts or
model-free native code. They do not certify actual target capture, CUDA/SM120 or
SM75 performance, native drafter acceptance, throughput or a Q4_0 EAGLE win.
Q4_0 EAGLE remains the primary eventual comparison baseline.

- New `tests/test_indexed_block_teacher.py`: 13/13 passed. Dense/indexed batches
  match every anchor's teachers, context, labels, predecessors, positions,
  attention, inputs and prefix hashes. Identical source `block_loss` outputs and
  logit gradients checked for all eight first-divergence choices, all 128 padding
  masks and both anchors (2048 cases; cases without eligible labels reject).
  Malformed maps, source/producer map mismatches, generated-client actual-final
  boundary binding, completed admission, exact retained/peak cost counts,
  synthetic capture/import and model-free C++ protocol checks passed.
- Existing tests: block_data 21/21, nine_model_train_capture 22/22 and
  block_capture_portability 14/14 passed. Five actual-native model fixture tests
  in block_teacher_capture were explicitly skipped because no target capture was
  authorized for this task.
- CPU-only `llama-block-teacher` CMake/Ninja build succeeded with
  `GGML_METAL=OFF`, `GGML_BLAS=OFF`, `GGML_ACCELERATE=OFF` and Release configuration.
  Model-free C++ tests compile/execute index parsing and unions across prompt
  lengths 1, 2, 255, 256, 257 and 1024 and continuation lengths 0 through 32;
  source checks confirm full-head computation flags and post-head write selection.
  The binary's no-argument usage check returned expected exit 1 without loading a
  model. Existing unrelated compiler warnings did not fail the build.
- Ruff and parent/native `git diff --check` passed.
- Independent experiment_operator source review and final new-suite rerun passed
  13/13. Raw logs are ignored at
  `/tmp/binary-eagle-indexed-teacher/results/indexed-storage-qa-final-20261005/`;
  initial logs under `results/indexed-storage-qa-20261005/` record one subsequently
  corrected brittle error-message regex and unavailable pytest. Checks use
  unittest with `/Users/pippo/github/binary-eagle-decoding/.venv/bin/python`.

Integration requires root review/cherry-pick of the parent feature commit after
fetching the published native commit. Any future real indexed capture requires a
new source/client/runtime-pinned plan and root's resource/ancestry admission;
this patch never changes a frozen running execution checkout. Preserve QA logs
if needed before retiring the temporary worktree. No main merge was performed.

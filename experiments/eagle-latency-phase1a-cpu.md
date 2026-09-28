# EAGLE Phase 1A latency accounting and bounded runtime patch

CPU-only preparation on 2026-09-28. No model inference, accelerator profiling,
or timing validation was run for the patch. The archived timing evidence below
was collected earlier on RTX 5080; those observations are historical and do not
establish current device availability or SM75 behavior.

## Reproduce offline accounting

```sh
python3 scripts/account_eagle_latency.py \
  --analysis results/binary-rescue-head-5080/round-analysis.json \
  --output results/eagle-latency-accounting.json
```

The raw archive remains ignored. The analyzer verifies SHA256 for every indexed
round and stage file, reconciles nonoverlapping round intervals, and validates
contiguous draft-stage partitions. The report gives each stage's CPU-wall time,
actual proposed/accepted/emitted counts, unmatched draft calls, process parts,
and explicit residual time. Summed round-row time can duplicate batched work
shared by slots. Nested stage values must not be added to round totals.

For the archived instrumented primary run, representative pooled values are:

| Variant | Round rows | Round ms/row | Draft ms/row | Process ms/row | Proposals | Accepted | Emitted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_0 | 1,493 | 15.751 | 2.957 | 0.603 | 7,320 | 1,555 | 3,048 |
| D group-128/A16 | 2,139 | 27.232 | 14.556 | 1.051 | 10,449 | 909 | 3,048 |
| FP16 | 1,496 | 16.946 | 4.882 | 0.708 | 7,329 | 1,552 | 3,048 |

The D trace has 2,124 matched draft-stage calls. Its `seed_sync_retrieve` plus
`recurrent_sync_retrieve` spans total about 8.993 ms per matched call. These
CPU-wall spans include waiting for pending backend work, so they cannot identify
which projection, attention operation, transfer, or kernel incurred that work.
The Q4_0 and FP16 values for the same span are about 1.174 and 2.603 ms per
matched call. The variants have different proposal acceptance and round counts;
these historical rows do not measure the effect of any new patch.

Missing attribution remains explicit: no per-projection activation pack/scales,
dot and output timings; no independent attention/norm/state-transfer split; no
per-node CUDA graph attribution; no complete-request prefill/decode alignment in
this archive. The draft stages cover seed and recurrent decode, synchronization,
sampling, input copying and bookkeeping. The round trace covers `draft()`,
target decode/sync, `process()` cache catch-up, checking and repair. The
`process()` scalars cover feature copy, encoder, batch construction and draft
decode, with an unassigned remainder. Whole-request timing and operator
profiles must be measured in a separate paired run with their own scope.

## Source changes ready for later A/B

`GGML_EAGLE_PRUNE_UNUSED_HEAD=1` skips EAGLE output norm, head projection and
`d2t` full-vocabulary expansion only when the decoder graph requests zero logits
and normalized-head embeddings are not requested. The prenorm state and cache
path remain in the graph. This affects eligible `process()` catch-up calls, not
the old `draft()` span. It applies to FP16, Q4_0 and W1Ax EAGLE through their
shared graph. The default path is unchanged. CPU compilation checks the source;
native paired output and cache equivalence remain a later GPU gate.
The zero-logit branch is in the decoder after `t_h_nextn` is marked as an output.
The generic sampling builder skips sampling when `t_logits` is null, and the
context copies logits only when `n_outputs > 0`. The cached-attention builder
adds K/V cache-write nodes to the graph before the branch. This source review
supports graph construction but does not replace native trajectory validation.

The loader now accepts learned row-scale v2/v3 packed models for A16/A8/A4/A1,
while group-128 stays A16-only. A v2/v3 model may pin
`eagle3.w1a1.activation_bits` in GGUF; when present, it must equal the runtime
`GGML_W1AX_ACT_BITS` selector. Existing files without that key still load. This
is a representation compatibility gate, not a measured latency change.

## Finite later profile and A/B protocol

After the user restores GPU access, run the standard supervised remote job on
the pinned host and build. Use fixed development prompt IDs, context length,
target precision, KV type, seed, draft length, graph setting and warmup. Use
Q4_0 as the primary control, FP16 as a diagnostic, and each supported W1Ax
representation as labeled. Copy the benchmark TOML to two immutable files;
set `GGML_EAGLE_PRUNE_UNUSED_HEAD` to `0` and `1` in `[environment]`. Run each
with `python3 scripts/benchmark_native_eagle.py --config <config> --run-id
<unique-run-id>` under `scripts/remote_job.py`, alternating order across at
least three paired repetitions. Keep timed serving runs distinct from a short
trace run with `round_trace=true` and `EAGLE_DRAFT_STAGE_JSONL` set to an
ignored output path.

Before comparing time, require exact emitted token IDs, proposals, accepted
counts, round counts and native state/cache behavior on the paired inputs.
Report full request wall and server decode throughput, draft ms/proposed token,
`process()` ms/round, and accepted/emitted totals, including absolute times and
confidence intervals. Profile representative fixed-shape calls with GPU events
or Nsight ranges for pack/scales, dot, output, attention, norms, state transfers
and graph replay; retain unassigned GPU and CPU-wall time. Do not subtract
isolated operator medians or lifetime kernel sums from unrelated request times.
If the behavior gate fails, disable the opt-in path and inspect the mismatch.
If it passes, retain the patch only if paired request or process time improves
without reducing acceptance or throughput; CPU compilation alone is no speedup
claim.

## CPU checks

Apple arm64 CPU build, `GGML_CUDA=OFF`, `GGML_METAL=OFF`, target `llama` passed.
The native loader fixture tests passed for learned row v2/v3 across four widths,
group-128 lower-width rejection, activation-selector mismatch and prior malformed
GGUF cases. Offline archive accounting and two synthetic partition tests passed.

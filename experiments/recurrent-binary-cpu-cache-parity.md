# CPU EAGLE draft-cache byte and mask diagnostic

**2026-09-28 UTC.** The native fork at published commit `0abe6e586` added an
opt-in capture after each completed EAGLE decoder graph. It reads the actual
F16 K/V cache rows at physical slots after `ggml_set_rows`, plus the causal
attention-mask input for that execution. The hook accepts only one CPU-host
F16 draft cache and one sequence, caps itself at 512 rows/64 MiB, and fails
closed on an unsupported layout. It synchronizes the graph before reading.

Two eight-token requests used frozen **training** prompts, pinned FP16 target
SHA256 `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`
and candidate-D SHA256
`10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`.
The Apple M3 Max CPU build disabled CUDA, Metal, Vulkan, SYCL, HIP, RPC,
OpenCL, BLAS and Accelerate; target and draft GPU-layer counts were zero.
Flash Attention was `auto` and enabled. No GPU/accelerator, remote model run,
training, development/final prompt or Q4_0 performance test occurred. Both
server process groups stopped with return code zero.

| Train prompt | Decoder executions | Stored rows | Exact F16 key bytes as elements | Exact F16 value bytes as elements | Exact-prefix masks | Rewritten physical positions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose: urban waterways 01 | 26 | 69 | 70,656/70,656 | 70,656/70,656 | 69/69 | 8 |
| Reasoning: rate and work 01 | 20 | 76 | 77,824/77,824 | 77,824/77,824 | 76/76 | 8 |

Each stored cache row matched its same-execution, same-column native
`Kcur_rope-0` or `Vcur-0` projection rounded to F16. The capture includes
two dummy reserve rows per request; the remaining 67 and 74 rows are the
live decoder executions. Every row used physical slot equal to decoder
position. Its mask had zeros exactly in slots `0..position` and negative
infinity elsewhere, including executions after positions were rewritten.
This verifies the simple contiguous-prefix mask and post-write bytes for
these two native CPU requests. It does not inspect the cache at an arbitrary
later time, prove a general multi-sequence rollback policy, or reconcile the
remaining CPU adapter attention-output drift. A full 96-prompt capture,
target-feature exactness and trained Q4_0 acceptance/latency/throughput gates
remain open.

Both runs reproduced the prior eight raw response IDs. Their graph index,
graph F32 values, raw target-feature index and values, forced rounds, native
head index/states and raw target logits were each bitwise identical to the
previous uninstrumented prose/reasoning CPU captures. Their continuity and
raw-response audits passed. The new stored-cache auditor also rejects
altered bytes, slots and masks on three synthetic fixtures; two CPU runner
policy tests passed.

Raw files remain ignored under
`results/recurrent-binary-cpu-cache-20260928-final-prose/` and
`results/recurrent-binary-cpu-cache-20260928-final-reasoning/`. Manifest
SHA256s are
`4ad342145a1b77deb5f48cac4e00671e26b691850f398be56687ef05830b7169` and
`99b9003698bcdf5505421a1667dac4d21a8545471d91330c04181c9f2a931d40`;
audit SHA256s are
`ca2f5d39fe2f3c57872115bb720386ff7c86c49218edf3a4f7209893a8f81325` and
`464e2fe84849ced2b28d31b2b06ff1cd35b53142d8a98b68d813a64a1acdd01e`.
Both manifests pin server binary SHA256
`0c78b4b3c6042a1f3a0a520cd957f4bb5715106ecb01ae7a548993c6b8791024`.
The earlier first-attempt output under `...-prose/` failed before serving;
intermediate outputs under `...-prose-b/` and `...-reasoning/` preceded the
final committed-source rebuild and are excluded from this result.

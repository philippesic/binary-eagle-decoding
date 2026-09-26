# All-layer W1Ax activation precision on RTX 2080 Ti

**State:** historical and development matrices complete; diagnostics in progress.
**Protocol:** [frozen W1Ax study](w1ax-activation-precision-plan.md).
**Hardware:** NVIDIA RTX 2080 Ti, SM75, 11,264 MiB VRAM under Ubuntu 24.04 WSL2.

## Frozen comparison

The same FP16 Qwen3-4B target and verifier serve target-only, ordinary FP16
EAGLE, Q8_0 EAGLE, Q4_0 EAGLE, and four all-nine-linear W1Ax drafts. The W1Ax
modes use byte-identical packed weight signs and F32 row scales from one GGUF;
`GGML_W1AX_ACT_BITS` changes only the activation path. A16 explicitly rounds
incoming F32 values to FP16 before signed accumulation. A8/A4 use per-token
absmax quantization with signed ranges ±127/±7 and nearest-even rounding. A4
uses the four-plane bit-serial CUDA kernel; A1 uses sign packing and mean
absolute scaling. The target, embeddings, normalization, attention scores,
residuals, and verifier remain at their ordinary precision.

The FP16 target GGUF SHA256 is
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`.
The shared all-nine W1A1 draft GGUF SHA256 is
`098e1ecbb299aa16e2c968663acc49e60c0fcf16b053766d9f558114f79d011c`,
matching the prior all-row source audit. The benchmark uses published llama.cpp
fork commit `3792aa79c` and project checkout `dc4ccdd`, CUDA 12.8 on SM75,
F16 KV, context 2048, concurrency one, D=5, confidence floor zero, greedy
decoding, two warmups, five measured repetitions, and one server at a time.
Each run manifest preserves exact commands and model/binary/config hashes.
The primary runner's GPU snapshots were unavailable because its nonlogin WSL
PATH omitted `nvidia-smi`; they are not zero-valued measurements. Hardware and
dispatch were independently verified by the SM75 gates and explicit WSL GPU
queries. A separate full device snapshot is preserved in
`runs/w1ax-device-manifest-20260925/` (stdout SHA256
`bd87139e8ce478aef42c4b17fda2345891be9cd1c4ad02b826c47dee417af0c8`).
The probe is being corrected for separate telemetry diagnostics; primary
per-variant peak-memory/clock claims are not made. Q8_0/Q4_0 name standard GGUF block weight formats; their live
activation path is Q8_1, not the research W8A8/W4A4 format.
Acceptance is the pinned native server's target-sample-and-match count, not
classical probability-ratio/residual speculative sampling. All comparisons
use that same verifier and its unchanged target sampling path.

## Correctness gates

- Local CPU operator tests passed 88/88. On the actual 2080 Ti, native A1,
  A16, A8, and A4 bit-serial CUDA tests passed 88/88 with an independent
  INT32-dot assertion for A8/A4. The conventional A4 CUDA comparator passed
  another 88/88. Distinct mode dispatch markers were observed.
- A three-request diagnostic captured 4,203 real activations from all nine
  selected linears. A frozen selector retained 147 captures over every
  observed tensor/shape group; its full invocation histogram has 73 rows.
  Identical-input replay completed 147 × 4 native W1Ax operators with scalar
  output parity, and a separate assertion run passed exact A8/A4 INT32 dots
  on those real inputs. The diagnostic assertion requires CUDA graphs disabled;
  it is excluded from timing.
- One frozen 32-token request per mode gave identical raw CPU-draft and
  CUDA-draft token IDs for A1/A4/A8/A16, with the FP16 target on GPU in both
  paths. The selected-token log probabilities also matched exactly on that
  request. `--spec-draft-device none` was required to keep the reference
  draft on CPU; setting its GPU layer count to zero alone still dispatched the
  custom operation on CUDA.

Real-input replay measured the complete ggml operation, including activation
quantization/packing, native dot, output, graph dispatch and synchronization.
At the observed head N=2 shape, medians were 978.7/331.4/117.8/94.8 µs for
A16/A8/A4/A1. At N=37 they were 29,443/5,470/1,637/919 µs. These are
synchronized host-wall operator timings, not isolated CUDA kernel events or
serving rates. Across 145 selected head token rows, A8/A4/A1 agreed with the
same-binary-weight A16 top choice 97.2%/72.4%/33.1% of the time. The rows are
correlated diagnostic inputs, not independent prompt examples.

## Historical paired matrix

The 12 frozen historical prompts × five repetitions × eight paths completed
480/480 requests on the RTX 2080 Ti with no failed request. All four W1Ax
loader, graph-selector, and CUDA-dispatch gates passed in every repetition;
every response included raw generated token IDs. The GPU was idle afterward.

| Draft | Decode tok/s | Request tok/s | Accepted drafts/round | Decode vs FP16 | Decode vs Q4_0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 60.56 | 58.61 | — | — | — |
| FP16 EAGLE | 80.67 | 75.71 | 1.168 | 1.000× | 0.899× |
| Q8_0 EAGLE | 86.16 | 80.47 | 1.168 | 1.068× | 0.960× |
| Q4_0 EAGLE | 89.73 | 83.57 | 1.182 | 1.112× | 1.000× |
| W1A16 EAGLE | 29.43 | 28.73 | 0.106 | 0.365× | 0.328× |
| W1A8 EAGLE | 42.12 | 40.70 | 0.102 | 0.522× | 0.469× |
| W1A4 EAGLE | 44.44 | 42.89 | 0.041 | 0.551× | 0.495× |
| W1A1 EAGLE | 45.39 | 43.74 | 0.055 | 0.563× | 0.506× |

Paired prompt/repetition bootstrap 95% intervals for W1Ax decode rate versus
Q4_0 were 0.304–0.354 (A16), 0.434–0.508 (A8), 0.454–0.541 (A4), and
0.466–0.549 (A1), using 2,000 resamples. Repetitions estimate timing
variation; the 12 prompts, not the 60 repeated requests, supply the quality
examples.

All speculative variants emitted identical raw IDs on all 60 paired
prompt/repetition requests. They differed from target-only on the same
`reasoning-02` request in each repetition, first at generated ID position
109. Ratios to target-only are timing observations, not strict lossless speedup
claims. W1A16 preserves more activation precision than A8/A4/A1, yet accepts
only 0.106 drafts/round versus FP16 EAGLE's 1.168. Its slow sign-add kernel
also makes it the slowest W1Ax serving path. The development matrix below provides the next quality check; the 24 QAT-final prompts remain reserved.

Historical raw artifacts reside on the 2080 Ti under ignored
`runs/w1ax-project-src/results/w1ax-historical-matrix-20260925/`. The
`records.json` SHA256 is
`160c2bced47d98cf641bc4e5fecb7b46caf23c3442ad2e75cb6d7891f11869e7`;
`report.json` is
`83299ab3ce612e75dcae839ab4221b8514dc8792bc1b6073d041369d97307fbf`;
the 2,000-resample paired analysis is
`b646e3df7bbb04764067386ecc1bf868011cc0fb764296debb47ddc20661a7f2`.
The paired intervals, per-prompt data, actual proposal/acceptance counts,
model hashes and dispatch evidence are preserved in those files. The selected
operator replay JSONL SHA256 is
`a06654a31f000bacf5988cda91201a100b57575fbead9548b66d614093c1afe5`.

## Development paired matrix

The frozen 24 development prompts × five repetitions × eight paths completed
960/960 requests with exit 0. All dispatch gates passed, and every speculative
variant matched target-only raw token IDs on all 120 paired requests. Each
request reached the 128-token cap. The primary build and policy were unchanged.

| Draft | Decode tok/s | Request tok/s | Accepted drafts/round | Decode vs FP16 | Decode vs Q4_0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 60.45 | 58.58 | — | — | — |
| FP16 EAGLE | 76.55 | 72.40 | 1.047 | 1.000× | 0.912× |
| Q8_0 EAGLE | 81.56 | 76.94 | 1.046 | 1.065× | 0.972× |
| Q4_0 EAGLE | 83.93 | 79.05 | 1.036 | 1.096× | 1.000× |
| W1A16 | 29.86 | 29.20 | 0.117 | 0.390× | 0.356× |
| W1A8 | 42.49 | 41.19 | 0.111 | 0.555× | 0.506× |
| W1A4 | 44.68 | 43.26 | 0.047 | 0.584× | 0.532× |
| W1A1 | 45.36 | 43.91 | 0.055 | 0.593× | 0.540× |

Paired prompt/repetition bootstrap 95% decode-ratio intervals versus Q4_0
were A16 0.343–0.368, A8 0.488–0.523, A4 0.509–0.555, and A1 0.520–0.559.
Against FP16 they were 0.375–0.404, 0.534–0.575, 0.558–0.608, and
0.570–0.613, respectively (2,000 resamples, seed 42).

The same acceptance loss persists with A16: restoring activation precision does
not recover the all-layer binary draft's quality. None of the W1Ax paths
approaches either throughput anchor on this development set.

Raw artifacts: remote `runs/w1ax-project-src/results/w1ax-development-matrix-20260925/`.
SHA256: `records.json`
`a3ac1f91d249a3edd7c54d07ff69a457d49dbd0efaabb317a45a4c0149ab04f8`;
`report.json` `f653acbb50bf490da3d589932d201f9d7f59bb4e795968eade0e9344de07e7ea`;
2,000-resample `analysis.json`
`f175094050ca1d807e2f3a6b963b44f97b14b8a7e2d39ab8d2eee5d2f49f9592`;
`manifest.json` `e7c2c55fca7a609bce162fa348d37bc93da2181707b5eb24d21c0be6b7a22cab`.

## Correctness and vocabulary diagnostics

A separate diagnostic runtime (`feba15698`, 88/88 SM75 gate passed) preserves
raw target logits before sampling. On historical `reasoning-02` at zero-based
position 109, target-only ranks token 12 first by 0.00348663 logit units; the
emitted speculative verifier row ranks 29208 first by 0.00539780. Its draft
29208 is accepted because it matches the target verifier. A preceding
hypothetical row was neither sampled nor emitted. The output difference is
therefore supported by a target-logit ranking reversal; the underlying
numerical cause remains unproven. These instrumented requests are excluded
from throughput. Raw traces and comparison are under
`runs/w1ax-project-src/results/w1ax-verifier-divergence-20260926/`.

Across both primary sets, 22,460/22,800 target-only emitted IDs (98.51%) lie
in the 32,000-entry draft vocabulary. The category fractions are 97.79% code,
99.13% prose and 98.63% reasoning. Repetitions are repeated trajectories, not
independent quality samples. This coverage audit does not measure target
probability mass, nor does it establish acceptance. The ignored raw report is
`runs/w1ax-vocab-coverage-20260926/report.json`, SHA256
`c341f882579b85b116c93e57d5b436a58565940eb17e0b69cf232a9846a7e922`.

## Matched operator controls

The extended same-input replay completed all 147 captures against all four
W1Ax modes and native FP16/Q8_0/Q4_0 operators. An explicit FP16 activation
roundtrip changed none of the ordinary FP16 operator outputs on these inputs;
all 145 head token rows retained identical top-1 and top-5. This bounds the
cast's observed effect on these captured inputs, not every possible state.

Representative head synchronized full-operation medians, microseconds:

| N | FP16 | Q8_0 | Q4_0 | W1A16 | W1A8 | W1A4 bit-serial | W1A1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 331.3 | 199.3 | 140.7 | 470.5 | 246.6 | 103.1 | 70.8 |
| 2 | 513.2 | 258.2 | 188.8 | 1,376.1 | 594.2 | 188.1 | 125.5 |
| 37 | 524.6 | 275.7 | 225.1 | 31,192.1 | 5,730.1 | 1,687.3 | 966.4 |

These are paired operator inputs with differing weight formats, not model
trajectory comparisons. A separate conventional-A4 replay measured
196.0/347.1/7,636.8µs at N=1/2/37; that run is a separate timing batch.
Raw operator JSONL SHA256:
`208b2e8f9caa526aeab4e9d66bfb88d0a375b14b1afbf7f95c864bc0f042f8cb`.

Separate CUDA software profiles preserve packing/quantization and dot kernels.
The profiles pool correctness, warmup and sampled launches; dot rescaling is
fused. Matching graph-disabled unprofiled replay showed median per-capture
profiler wall overhead factors 1.010/1.123/1.157/1.079 for A16/A8/A4/A1.
Do not substitute these traced times for the uninstrumented serving results.

## Activation error on captured inputs

All 588 W1Ax replay rows have validated element counts and activation
diagnostics. Element-weighted summaries are preserved for each of the nine
linears and each observed shape. A4 zero-code rates range from 29.99% at
attention output to 98.31% at fusion; FFN-down is 88.88%. Fusion's A8
zero-code rate is 63.57%. At the head, A8/A4 zero-code rates are 2.22%/37.43%,
and activation RMSE is 0.000291/0.022248/0.402891/0.866936 for A16/A8/A4/A1.
A1 has sign codes, so a zero-code rate does not apply.

Clipping was zero for A16/A8/A4 on these inputs; absmax quantization can
produce substantial rounding-to-zero error while avoiding clipping. These
statistics weight the selected, correlated capture elements, not live operator
invocation frequency or independent prompts. They do not isolate the cause of
whole-model acceptance loss. Full report:
`runs/w1ax-activation-error-analysis-20260926/report.json`, SHA256
`b0917ab80e0c4418e4a381466031e018edbabc8e69db0f34206c225095bf1580`.

## CUDA packing and dot stages

Each low-bit profile contains 1,176 validated adjacent same-stream packing/dot
pairs, with zero unpaired launches. Kernel rescaling is fused into the dot.
The head's launch grids are identified from the frozen replay's unique
32,000-row tensor and the runtime grid formula; the trace itself does not carry
capture or layer IDs. Median microseconds for these head launch shapes:

| N | Mode | Packing/quantization | Dot + rescale | Paired kernel sum |
| ---: | --- | ---: | ---: | ---: |
| 2 | A8 | 5.40 | 310.01 | 315.35 |
| 2 | A4 | 8.17 | 104.31 | 112.46 |
| 2 | A1 | 7.53 | 67.45 | 74.97 |
| 37 | A8 | 7.04 | 7,643.47 | 7,650.00 |
| 37 | A4 | 8.46 | 1,863.28 | 1,871.69 |
| 37 | A1 | 8.76 | 1,187.30 | 1,196.04 |

Paired sums are medians of per-invocation sums, not sums of marginal medians.
There are 24 pairs at N=2 and eight at N=37 per precision, pooling scalar-check,
warmup and timing executions. Dot-only is an already-packed diagnostic. Host
allocation/dispatch and inter-kernel gaps are excluded from these sums. The
A16 fused cast/sign-add/rescale kernel measured 933.31µs at N=2 and
30,327.11µs at N=37 in its separate profile. All four use CUDA graphs disabled
and include profiling overhead; they must not be subtracted from serving time.

The anchor profile confirms Q8_1 conversion plus MMVQ/MMQ for Q8_0/Q4_0;
ordinary FP16 includes floating matmul and Turing FP16 GEMM kernels. Full
symbols and unknown classifications remain preserved. Traces/SQLite exports:
`runs/w1ax-nsys-a{16,8,4,1}-20260926/` and
`runs/w1ax-nsys-anchors-20260926/`. Pair-analysis report SHA256s:

- A16: `453641e3ca6fcdb28dedb23dc6626d4017b07a96079fceae96187600c7ff3389`
- A8: `9ea1ae1ed79f046411deea26f07bbc03001b386b35969f7f658f1b7d25c53105`
- A4: `7c8ee8d8bf8f6bc657d5dac7afc6f23a18b921e8c4369a25feb6443608a2d036`
- A1: `05245e57db6ebf8854ab9b1fd814856b251d6569a899aa447a4833abbe2c7061`

## Complete round trace and conditional limits

The separate 480-request historical trace completed with all 480 output-ID
sequences matching their corresponding uninstrumented primary requests. All
420 speculative measured requests were mapped to task groups, with two warmups
per server excluded. The trace contains 37,695 measured round events and omits
exactly one leading output token per request. Native accepted/proposed/verified-
round counters reconcile exactly. Full trace counts additionally include
`no_proposal` events: 20 for FP16/Q8_0, 15 for Q4_0, and 55 per W1Ax mode.
The primary tables' acceptance denominators are native verification rounds.
FP16 and Q8_0 each accepted five tokens that were not emitted at stopping
boundaries; counterfactuals use accepted tokens actually emitted.

All spans passed bounds, union, overlap and unclamped-residual checks: zero
invalid, nested, overlapping, out-of-bounds or clamped-residual rows. Mean CPU
wall milliseconds over complete traced round events:

| Draft | Round mean | Round p95 | draft() | Target decode + sync | process() |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP16 | 26.827 | 28.872 | 6.730 | 18.138 | 1.076 |
| Q8_0 | 25.273 | 27.480 | 5.238 | 18.083 | 1.055 |
| Q4_0 | 24.542 | 26.632 | 4.435 | 18.152 | 1.047 |
| W1A16 | 37.603 | 38.668 | 17.447 | 18.138 | 1.466 |
| W1A8 | 26.398 | 27.516 | 6.546 | 18.075 | 1.230 |
| W1A4 | 23.585 | 24.948 | 3.961 | 18.043 | 1.054 |
| W1A1 | 23.441 | 24.708 | 3.815 | 18.061 | 1.034 |

The remaining means include proposal checking (0.340–0.703ms), residual host
work (0.172–0.190ms), checkpoints, repair and the accept hook. `begin` is
outside the round and retained separately in raw summaries. These CPU spans
are distinct from CUDA kernel events and from uninstrumented request timing.

For a deliberately fixed trace trajectory, remove only exclusive `draft()`
spans, retaining `process()` and every other measured cost. Let C be total
round time, D those removable spans, E traced emitted IDs, A accepted drafts
actually emitted, P proposed drafts, and B=E−A. The span-removal proxy is
E/(C−D). At fixed C, the acceptance threshold against primary rate R is
A_required=R×C−B. All times in these formulas use seconds.

| W1Ax | Rate after removing draft() spans | Required accepted/trace round vs FP16 | vs Q4_0 |
| --- | ---: | ---: | ---: |
| A16 | 54.85 tok/s | 2.033 | 2.374 |
| A8 | 55.44 tok/s | 1.129 | 1.369 |
| A4 | 53.04 tok/s | 0.903 | 1.116 |
| A1 | 53.76 tok/s | 0.891 | 1.103 |

Both primary anchors (80.67/89.73 tok/s) exceed these removal proxies. This
indicates that optimizing the recorded draft() region alone is insufficient
under this fixed trajectory. **These are neither measured speedups nor
uninstrumented bounds:** the seed, begin/inter-round work and prefill lie
outside C; process/catch-up work remains; changing acceptance changes round
counts, batch shapes, stopping and costs. The report also preserves raw
thresholds for C−D and fixed-proposal ceilings without clamping.

Artifacts and SHA256:

- `runs/w1ax-project-src/results/w1ax-round-matrix-20260926/records.json`: `a6c0c458c2cf86807cd3e7cc6f9965eae337b4d72e75504ad17e1c853fa079a3`
- `runs/w1ax-round-analysis-20260926/report.json`: `7d7186cdb14d6aeffd153142328d5d3d0c418a7bcfde189585e12db1d0e3b2f3`
- `runs/w1ax-break-even-analysis-20260926/report.json`: `166790c62ccff4b6eb1d18a8e161a4d4078dd477d75415a796b22c07adeb3bad`

## Full-server CUDA profile

A separate three-prompt, 32-token diagnostic ran all eight paths for five
repetitions, both unprofiled and under Nsight (120 measured requests each, two
warmups/server). All 120 profiled raw output sequences match their unprofiled
partners. Observed profiled/unprofiled decode-time ratios were:

| Path | Decode-time ratio |
| --- | ---: |
| Target-only | 1.0084× |
| FP16 EAGLE | 1.0309× |
| Q8_0 | 1.0304× |
| Q4_0 | 1.0179× |
| W1A16 | 1.0053× |
| W1A8 | 1.0186× |
| W1A4 | 1.0198× |
| W1A1 | 1.0209× |

These descriptive ratios include timing variability; they are not a pure
hardware-independent instrumentation tax. The primary throughput trials remain
uninstrumented. Both short diagnostic runs retain CPU round tracing.

The verified export contains 813,285 kernel activities and 29,470 aggregate
CUDA graph activities across 40 server PIDs, plus transfer/memset activities.
All 40 processes associate with the recorded five-by-eight serial schedule
under noninterleaving activity-span and mode-signature checks. This is
**chronological schedule inference**, supported by per-server mode logs; the
runner did not directly record server PIDs. It gives process-level pooling, not
exact request/round/layer CUDA attribution.

Graphs remain enabled. Replays are recorded at whole-graph granularity; there
are no replay-node kernel rows. The separate graph-disabled operator profiles
supply individual packing/dot breakdowns. Combined activity intervals are
unioned so graph envelopes and child activities are not naively added. Their
coverage includes possible gaps inside graphs and is not physical GPU busy
time. Kernel-only summaries remain separate.

The first profile attempt had valid request records but no CUDA activity:
the server environment filter removed profiler injection variables. That
attempt is preserved and excluded from profiling claims. A tested profile-only
wrapper adds exactly the observed Nsight variables to a fresh configuration,
validates library paths, and records hashes without editing the measured runner.
A two-request smoke confirmed kernels and aggregate graph activity; the full
retry explicitly requested graph granularity.

Verified raw profile/SQLite:
`runs/w1ax-project-src/runs/w1ax-server-nsys-retry-supervisor-20260926/`.
Paired request directories: `results/w1ax-server-profile-baseline-20260926/`
and `results/w1ax-server-nsys-retry-20260926/` within the runner worktree.
Combined analysis: `runs/w1ax-server-cuda-analysis-20260926/report.json`, SHA256
`1be5bdef18d1356ccf300ea29d23e1027c78c121d17a1381495eb6bee365b37b`.

## Conditional secondary controls

The plan's optional genuine W8A8/W4A4 same-run controls were unavailable in
the frozen combined build. Their previously audited all-nine GGUFs exist, but
W1Ax runtime `feba15698` lacks the W8A8/W4A4 loaders/operators from separate
revision `d0724427b`, which is not its ancestor. Their historical results remain
separate; they are not substituted for new measurements. FP16, Q8_0 and Q4_0
mandatory controls are present throughout the primary and diagnostic matrices.

## Outstanding measurements

Context/output-cap diagnostics, D/p_min policy grid and streaming latency/telemetry
remain in progress. Do not treat the historical
screen as a final trained-QAT result or use the reserved 24-prompt final set.

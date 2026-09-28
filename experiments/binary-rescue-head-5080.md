# Binary rescue and frozen-body head adaptation on RTX 5080

Status: **in progress**. Quality, body/head diagnosis, the bounded head fit,
profiling and clean repeated primary performance are complete. The resumed
full-prompt instrumented run completed all 312 requests with CUDA graph
launches in every block; round-cost analysis is pending. The frozen
longer-context comparison is running, and final artifact handoff remains
pending. No candidate has been promoted.

## Frozen comparison

Q4_0 EAGLE is the primary baseline. The target/verifier remains the pinned FP16
Qwen3-4B. This study uses RTX 5080 only, the existing 96 training prompts and 24
development prompts. The 24 reserved-final prompts remain untouched. All primary
requests use native greedy sample-and-match, D=5, p_min=0, context2048, F16 KV,
seed42, no thinking, no prompt reuse, one client, and at most128 output tokens.

The [goal checkpoint](../docs/goals/binary-rescue-head-5080.md) records ownership,
protocol, jobs and pending work. Earlier scale-fitting and SM75 reports remain
sealed. Development data is reused exploratory evidence; timing repetitions do
not create additional independent quality prompts.

## Single-rescue quality

One deterministic pass of24 development prompts per path, plus two excluded
warmups/path. All240 measured requests completed. All paths matched both Q4_0
and target-only raw output IDs on24/24 prompts. These are quality findings;
single-pass instrumented times are not the repeated performance result.

| Path | Accepted / proposed drafts | All completed rounds | Accepted/round | No-proposal rounds |
| --- | ---: | ---: | ---: | ---: |
| **Q4_0 primary** | 1555 / 7320 | 1493 | **1.0415** | 11 |
| D, unchanged fitted group scales | 909 / 10449 | 2139 | 0.4250 | 15 |
| C, fitted row scales | 685 / 11535 | 2363 | 0.2899 | 19 |
| D + attention Q8_0 | 1154 / 9262 | 1894 | 0.6093 | 14 |
| D + fusion Q8_0 | 1060 / 9736 | 1988 | 0.5332 | 14 |
| D + FFN-down Q8_0 | 977 / 10127 | 2071 | 0.4718 | 15 |
| D + output-head Q8_0 | 1028 / 9876 | 2020 | 0.5089 | 15 |
| D + original FP16 head | 1031 / 9861 | 2017 | 0.5112 | 15 |
| FP16 diagnostic | 1552 / 7329 | 1496 | 1.0374 | 11 |

No bonus target tokens enter accepted counts. The server's Prometheus round
counter excludes no-proposal rounds; this table uses the complete native trace
denominator. An independent audit reconciled every measured and warmup request.
Q4_0, D and C reproduce the sealed earlier development acceptance counts exactly.

The frozen combination rule admits at most two subsets whose individual gains
are at least10% over D and improve at least16/24 prompts, choosing the two largest
pooled gains. Attention improves43.37% and fusion25.47%, each on24/24 prompts.
Their one combined rescue reached 1251/1797 = **0.6962 accepted drafts/round**
(24 prompts), versus a repeated same-run Q4 control of 1555/1493 = 1.0415.
Both used verified CUDA graphs. The combination remains below the frozen 80%
Q4 quality threshold for admitting an extra depth screen.
The original FP16 head's0.5112 is the untrained control the fitted head must beat.
Each path emitted 3,072 output IDs. That is 2.058 emitted tokens per complete
round for Q4_0, 1.436 for D and 1.709 for the combined rescue, counting the
target token as an emission but never as an accepted draft. Accepted/proposed
draft fractions are 21.24%, 8.70% and 14.20% respectively. These fractions
and accepted/round answer different questions: the latter also reflects extra
verification rounds caused by early rejection and no-proposal behavior.

Later drafts remain the main quality deficit. Conditional acceptance is the
number surviving a depth divided by the number actually reached at that depth;
terminal-censored positions and rounds without proposals are not counted as
failures at later depths. The selected fitted head uses its separately audited
24-prompt selection pass.

| Path | Depth 2, survived/reached | Depth 3, survived/reached | Zero-accept rounds | No-proposal rounds |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | 431/860 (50.1%) | 179/425 (42.1%) | 628 | 11 |
| D | 159/720 (22.1%) | 21/157 (13.4%) | 1,412 | 15 |
| D + attention Q8_0 | 266/807 (33.0%) | 61/263 (23.2%) | 1,081 | 14 |
| D + fusion Q8_0 | 205/815 (25.2%) | 28/203 (13.8%) | 1,165 | 14 |
| D + fitted FP16 head | 234/769 (30.4%) | 62/231 (26.8%) | 1,185 | 13 |

The full quality manifests retain every proposal-length and depth numerator,
including eligible but unreached positions; these conditional figures must not
be interpreted as unconditional five-token survival.

## Storage and precision audit

Each rescue replaces only the named D projections with standard Q8_0 payloads
derived from the original FP16 checkpoint. Every other tensor is byte-identical
to D, including signs, F32 group128 scales, normalization and vocabulary mapping.
All four artifacts passed reference-quantization and serialized-type audits.

| Q8_0 subset | Share of selected weights | Binary share remaining | Selected payload bits/weight | Entire file bits/model parameter |
| --- | ---: | ---: | ---: | ---: |
| Fusion | 9.009% | 90.991% | 1.903 | 2.132 |
| Attention Q/K/V/output | 19.219% | 80.781% | 2.643 | 2.872 |
| FFN down | 11.411% | 88.589% | 2.077 | 2.306 |
| Vocabulary head | 37.537% | 62.463% | 3.971 | 4.200 |

Each named projection has the following logical shape and tensor payload.
All D entries store packed signs with F32 group128 scales and run the custom
F16-input/F32-sign-add path; Q8 entries, only where tested, include standard
block scales and run Q8_1 conversion plus MMVQ/MMQ. Payloads are measured GGUF
tensor bytes rather than a percentage-based runtime estimate.

| Projection | Output × input | Rescue subset | D MiB | Q8 MiB |
| --- | ---: | --- | ---: | ---: |
| Fusion `fc` | 2,560 × 7,680 | Fusion | 2.930 | 19.922 |
| Attention Q | 4,096 × 5,120 | Attention | 3.125 | 21.250 |
| Attention K | 1,024 × 5,120 | Attention | 0.781 | 5.312 |
| Attention V | 1,024 × 5,120 | Attention | 0.781 | 5.312 |
| Attention output | 2,560 × 4,096 | Attention | 1.562 | 10.625 |
| FFN gate | 9,728 × 2,560 | None | 3.711 | — |
| FFN up | 9,728 × 2,560 | None | 3.711 | — |
| FFN down | 2,560 × 9,728 | FFN down | 3.711 | 25.234 |
| Vocabulary output | 32,000 × 2,560 | Head | 12.207 | 83.008 |

Gate and up stayed D in every rescue; — is an untested replacement, not a
zero-cost tensor. The FP16 head uses 156.25 MiB for its 81.92M weights; its
native one-token MMVF has FP16 partial and F32 final reduction, while larger
shapes can use MMF or cuBLAS with separately audited accumulation.

Selected-weight denominators are218,234,880 logical matrix weights. Binary
payload includes packed signs plus F32 scales (1.25bits/weight); Q8_0 includes
block scales (8.5bits/weight). Norms remain F32 and d2t I64. Full-file figures
include GGUF metadata, padding and mapping bytes; model-parameter denominators
exclude integer maps. Parameter percentages are not runtime percentages. This is owned drafter payload;
the unchanged target embeddings remain shared higher-precision inputs, and the
fixed FP16 target is included in total serving memory and end-to-end timing.
Standard Q8_0's Q8_1 activation conversion differs from D's F16-cast activation
path: these are practical mixed-path interventions, not isolated weight-bit tests.
Actual per-projection execution paths and selective kernel costs are detailed below.

The admitted attention+fusion combination retains 71.77% binary selected
weights, uses 3.2965 selected payload bits/weight, and has a 91.71 MiB GGUF.
The selected fitted-FP16-head model retains 62.46% binary selected weights but
uses 6.7868 selected payload bits/weight and 182.51 MiB on disk. The unchanged
D and Q4_0 draft files are 38.47 and 123.01 MiB respectively. These are draft
file footprints; the shared FP16 target dominates total loaded memory. The
dense fitted head exceeds Q4_0's entire draft file size and is diagnostic.

## Correctness and graph gates

- 112/112 binary operator cases passed on actual RTX5080 CUDA, plus CPU gates.
- Native loader fixtures accepted valid v2/v3 models and rejected wrong types,
  overlapping/omitted coverage and packed/dense shadow tensors.
- Historical3-prompt Q4/D capture off/on gates matched raw IDs and proposal
  digests; canonical Q4 self-replay and all five crossed arms passed exact round,
  verifier-label and output checks. Same-body cross-head states matched by bytes.
- Initial FP16-head export roundtrip matched raw IDs and captured body states.
- 32 real FP16-head states /1,024,000 logits gave max absolute error 0.0035923
  and p95 0.0012449 when compared with a CPU F32 surrogate; no argmax mismatch.
  The training computation uses
  F16-rounded inputs/weights with F32 accumulation; native MMVF uses half2
  partial accumulation and F32 reduction. It is an audited approximation.

All ten quality blocks report actual CUDA graph launches, with zero disabled or
incompatible direct fallbacks. Binary/mixed blocks report custom-operation graph
launches. Counters include setup and warmups; capture/recapture costs and direct
warmups are retained. They do not alone establish per-projection kernel timing.

The final runtime's opt-in CUDA dispatch audit observed all nine drafter
projections for each path. Q4_0 executes Q4_0 matrix paths with Q8_1 activation
conversion; D/C execute custom group/row A16 sign-add paths; each Q8 rescue
executes MMVQ/MMQ only for its declared subset. The original and fitted dense
heads execute FP16 MMVF/MMF/cuBLAS as shape dictates. The audit records the
actual weight name/type, logical shape, activation operand, accumulation and
fused gate/up sharing; it is an inventory at graph construction, not a graph
replay counter. Full records and per-server logs are retained locally and on
the RTX 5080 host.

Nsight Systems 2025.5.2 captured real CUDA graph nodes and kernels on three
historical prompts plus two warmups for each path. Exact process-PID joins
separate the twelve sequential servers. **Summed GPU kernel duration over the
whole diagnostic server lifetime** was:

| Path | Kernel sum (s) | Custom A16 subset (s) |
| --- | ---: | ---: |
| Q4_0 | 3.785 | — |
| D | 11.044 | 5.916 |
| C | 8.682 | 3.073 |
| D + attention Q8 | 7.180 | 2.621 |
| D + fusion Q8 | 10.094 | 5.328 |
| D + FFN-down Q8 | 9.618 | 4.696 |
| D + head Q8 | 9.522 | 4.558 |
| D + original FP16 head | 9.708 | 4.536 |
| Attention + fusion Q8 | 6.657 | 2.351 |
| D + fitted FP16 head | 9.527 | 4.460 |
| FP16 EAGLE | 4.330 | — |
| Target only | 6.492 | — |

A dash means the path has no custom binary A16 operation. For the
observed one-token binary-head kernel shape, D group128 scales had a 206 µs
median over 2307 launches versus 76 µs over 2545 launches for C row scales.
The 32,000-row head attribution uses its unique grid shape in this frozen
model, not a runtime tensor ID. Standard Q4_0's Q8_1 pack kernels summed
24.9 ms and its quantized matmul kernels 275.2 ms over its whole trace.
These per-path sums have different round trajectories and include startup and
warmups; they are not request latency or paired speed ratios. CUDA kernel
interval unions, observed grids, graph-node IDs and ambiguity limits are in
the preserved per-process analysis. No GPU and CPU span sums are combined.

## Common-history body/head diagnostic

All 144 requests completed: canonical Q4 recording plus five forced arms across
24 development prompts. Each arm scored 7,320 identical proposal prefixes and
round boundaries, preserving its own body recurrence and cache progression.
Every same-body cross-head state comparison was byte-identical; forced outputs,
accept counts and reached cloned-verifier labels matched exactly. The target
label was unsupported by the draft vocabulary for 148/7,320 states (2.02%).
Those labels remain failures in unconditional agreement below.

| Body / head | Position 1 agreement | Position 2 | Position 3 | Position 4 | Position 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 / Q4 | 58.37% | 43.15% | 29.44% | 20.29% | 14.38% |
| Q4 / D | 52.43% | 36.91% | 25.55% | 18.02% | 12.03% |
| D / D | 33.60% | 19.88% | 12.30% | 10.39% | 8.37% |
| D / Q4 | 38.60% | 23.07% | 13.93% | 9.70% | 8.23% |
| D / original FP16 | 38.46% | 23.81% | 13.52% | 9.97% | 8.23% |

These are per-position predictions on forced histories, not live acceptance or
conditional prefix survival. The verifier-token rank and signed margin
(`logit(label) − max(other logits)`) among supported labels deteriorate with
depth; off-policy proposal prefixes are included:

| Body / head | Median rank at positions 1 / 3 / 5 | Mean margin at positions 1 / 3 / 5 |
| --- | ---: | ---: |
| Q4 / Q4 | 1 / 3 / 8 | +0.745 / −1.938 / −3.902 |
| Q4 / D | 1 / 4 / 10 | +0.236 / −2.288 / −4.041 |
| D / D | 3 / 16 / 37 | −1.482 / −3.879 / −4.916 |
| D / Q4 | 2 / 12 / 26 | −0.978 / −3.475 / −4.580 |
| D / original FP16 | 2 / 12 / 26 | −0.980 / −3.473 / −4.596 |

Full per-depth ranks, margins, validity/support denominators and prefix
records are saved.

Under this canonical Q4 forced-prefix distribution, the larger loss follows
the D body, which includes fusion and its own recurrent cache: swapping a D
head onto Q4 body costs about 5.9 percentage points at position 1, while
swapping D body under Q4 head costs 19.8 points. Later-position deterioration
also persists under a dense head. These are controlled interventions, not an
additive decomposition or live acceptance estimate. They do not prove
irrecoverable information loss or predict the fitted head's live-chain result.

## Head fit and performance

The one approved fit completed 500 steps in 19.83 optimizer seconds. It used
8,160 supported states selected evenly from 40,815 aligned training states;
1,110 source labels were unsupported and excluded from CE while retained in
coverage denominators. No selected state had an unsupported label.
There were 31,904 state presentations (3.9098 effective passes; each selected
state appeared three or four times). All optimization/export checks were finite.
CUDA native/surrogate parity passed with max error 0.003593 and no argmax change
on the frozen 32-state audit. Live native selection chose **step 500**: 1088/1960 = **0.5551 accepted drafts
per round**, versus step 0 at 0.5112, step 100 at 0.5449 and step 250 at 0.5472.
Every checkpoint matches Q4_0 and target-only raw IDs on all 24 prompts. The
selected head improves 8.60% beyond the untrained swap, but reaches only 53.30%
of Q4_0 acceptance. This is a dense-head mixed-precision diagnostic, not an
all-binary result. Neither finalist meets the extra depth-screen threshold.

The approved fit freezes D body/signs/scales/norm/vocabulary and all
target parameters; one original-FP16-head initialization, true native-verifier
CE and a fixed initialization regularizer. Only supported valid states from the
96 training prompts may train, with explicit unsupported denominators, at most
8160 selected states and500 steps/45 optimizer minutes. Native development
acceptance selects among0/100/250/500 (plus a finite time-cap endpoint).

## Five-repetition native performance

The clean primary matrix completed 1,440 measured requests: 12 paths × 24
development prompts × five balanced, alternating repetitions, with 120 separate
warmups. Every response emitted 128 raw token IDs; all 120 measured outputs per
path match Q4_0 and target-only. The same RTX 5080, final CUDA binary, model
artifacts, target, prompt manifest and policy were used throughout. Each of the
60 server blocks reported actual CUDA graph launches (912,625 in aggregate),
with no disabled or incompatible direct fallback. Graph recaptures and direct
warmup work remain included in the observed serving behavior.
An independent local audit compared all 1,440 measured requests: each path's
raw IDs matched the Q4_0 path for the same prompt, and all five repetitions
within each path/prompt matched stable proposal/output digests and native
accepted/proposed/round counters. The proposal digest is noncryptographic
equality evidence; raw output ID comparison is exact. Task IDs and completion
ordinals are server-specific and were excluded from digest equality. The audit
is retained beside the compact timing manifest.

Client throughput divides 15,360 emitted tokens per path by the sum of its
120 full request wall times; server decode throughput uses the server's decode
time only. The ratio interval is a paired 95% bootstrap over 24 prompt IDs,
retaining all five repetitions within each sampled prompt. It describes this
selected development workload, not uncertainty over new prompts or a
reserved-final result.

| Path | Client tok/s | Client / Q4_0 (95% interval) | Server decode tok/s | Request p50 / p95 (s) | TTFT p50 / p95 (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Q4_0** | **135.1** | **1.000** | **142.1** | **0.935 / 1.111** | 0.031 / 0.109 |
| Target only | 97.7 | 0.723 (0.700–0.748) | 100.0 | 1.310 / 1.315 | 0.029 / 0.035 |
| FP16 EAGLE | 121.6 | 0.900 (0.891–0.910) | 127.2 | 1.053 / 1.222 | 0.031 / 0.107 |
| D unchanged | 53.8 | 0.398 (0.387–0.408) | 54.9 | 2.409 / 2.568 | 0.033 / 0.113 |
| C row scales | 65.3 | 0.483 (0.468–0.498) | 66.9 | 1.954 / 2.089 | 0.032 / 0.110 |
| D + attention Q8_0 | 77.3 | 0.572 (0.557–0.585) | 79.6 | 1.673 / 1.835 | 0.033 / 0.111 |
| D + fusion Q8_0 | 58.6 | 0.434 (0.425–0.443) | 59.9 | 2.219 / 2.410 | 0.031 / 0.105 |
| D + FFN-down Q8_0 | 60.1 | 0.445 (0.433–0.455) | 61.5 | 2.149 / 2.271 | 0.033 / 0.109 |
| D + output-head Q8_0 | 59.9 | 0.443 (0.431–0.455) | 61.3 | 2.112 / 2.385 | 0.033 / 0.109 |
| D + original FP16 head | 59.1 | 0.437 (0.424–0.449) | 60.4 | 2.139 / 2.421 | 0.033 / 0.111 |
| D + attention/fusion Q8_0 | 83.3 | 0.616 (0.602–0.631) | 85.9 | 1.542 / 1.721 | 0.031 / 0.106 |
| D + fitted FP16 head | 60.8 | 0.450 (0.437–0.462) | 62.2 | 2.154 / 2.406 | 0.033 / 0.107 |

The admitted mixed rescue is the fastest binary-body candidate, but its full
request throughput is 0.616× Q4_0 and 0.852× target-only. The selected fitted
head reaches 0.450× Q4_0 and 0.622× target-only. Neither candidate wins against
Q4_0 on any of the 24 paired prompt IDs; their prompt-level client ratios span
0.535–0.747 and 0.399–0.511 respectively. The Q4_0 advantage is therefore
larger than the observed repetition and prompt variation in this workload.

Streaming per-token intervals are available for 85/120 requests per path
(17/24 prompts × five repetitions); seven code prompts have batched or absent
per-event raw IDs and are excluded from that interval statistic. All 120
requests/path have inter-chunk intervals, which are not equivalent to per-token
latency when chunks contain multiple tokens. On the valid subset, Q4_0's
inter-token p50/p95 were 0.289/14.755 ms, versus 19.505/20.164 ms for the
combined rescue and 24.931/25.573 ms for the fitted head. Q4_0 often emits
accepted drafts in a burst, making its median very low; the p95 captures
slower verification cycles. Full-request wall time, server decode time and
stream intervals have different boundaries and cannot be added.

The first timing attempt was interrupted after a simultaneous CPU-only Nsight
SQLite export overlapped its initial Q4_0 block. All its results were excluded;
the clean 60-block matrix above ran without that competing export. Separate
full-prompt round instrumentation was interrupted at the user's request for all
RTX 5080 host resources and is excluded from final cost calibration. The frozen
longer-context diagnostic has not started. Neither can replace the clean
primary timings.

Whole-device telemetry was sampled once per second during each server block,
including setup, warmups, measured requests and shutdown. “Loaded” is the
post-load snapshot, identical across the five blocks for each path; “peak” is
the largest sampled memory use, not an allocator high-water mark. Utilization,
power and SM clock are medians of each block's median across the five blocks.

| Path | Loaded / sampled peak VRAM (MiB) | GPU utilization | Power (W) | SM clock (MHz) |
| --- | ---: | ---: | ---: | ---: |
| Q4_0 | 11,182 / 11,306 | 81% | 138.4 | 2,921 |
| Target only | 10,446 / 10,548 | 95% | 178.7 | 2,917 |
| FP16 EAGLE | 11,482 / 11,700 | 80% | 147.5 | 2,917 |
| D | 11,098 / 11,220 | 91% | 147.4 | 2,917 |
| C | 11,092 / 11,214 | 88% | 143.2 | 2,917 |
| D + attention Q8_0 | 11,134 / 11,258 | 88% | 151.2 | 2,917 |
| D + fusion Q8_0 | 11,114 / 11,238 | 91% | 146.3 | 2,917 |
| D + FFN-down Q8_0 | 11,120 / 11,244 | 90% | 147.8 | 2,917 |
| D + output-head Q8_0 | 11,168 / 11,292 | 90% | 142.5 | 2,917 |
| D + original FP16 head | 11,242 / 11,460 | 90% | 144.3 | 2,917 |
| D + attention/fusion Q8_0 | 11,150 / 11,274 | 88% | 141.3 | 2,917 |
| D + fitted FP16 head | 11,242 / 11,460 | 90% | 142.7 | 2,917 |

The pre-load whole-device baseline was about 2,115 MiB. Sampling includes
other graph and runtime allocations; subtracting one row from another does
not isolate draft weight storage, and one-second polling can miss short peaks.
The 2,048-token logical FP16 KV allocation is 288 MiB for the 36-layer target
and 8 MiB for the one-layer drafter. The file payload comparison above is the
more direct drafter storage measure.

## Interim assessment

Neither tested endpoint merits promotion as a Q4_0 replacement on the
development workload: both accept fewer drafts, run slower end to end, and
trail target-only despite matching its raw output IDs. The head fit's positive
gain over its untrained dense-head control does not erase the body gap. The
common-history intervention associates about 19.8 points of first-position
agreement loss with the D body under Q4 head, versus 5.9 points with the D
head under Q4 body. A future bounded target-aligned recurrent body-plus-head
adaptation with fixed binary/A16 deployment and native Q4_0 gates is therefore
the most directly motivated research option. Its quality and speed remain
unproven; the observed A16 execution cost is a separate obstacle. This
recommendation starts no new training or GPU work. The longer-context and
full-prompt round diagnostics must still be completed before closing this
goal; reserved-final evaluation requires a separate user decision.

## Reproducibility checkpoint

The frozen primary development file is SHA-256
`a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885`;
the 96-prompt training file is
`80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74`.
The deterministic six-prompt long-context file was independently regenerated
locally at
`a03bc4469eb7bedd75048e4fa538a0404f5801ccdb14dd6aa9f4623d8b113966`.
Reserved-final prompts were not loaded or evaluated.

The clean primary run used published llama.cpp runtime revision
`70ec890961cf368cb6657a7859ac4122c72b778a` and parent source revision
`0c796ea`; subsequent parent revisions update analysis and reporting without
changing the benchmark binary. The executable wrapper SHA-256 was
`d541495500448f909de505fc36b58528c4c22422d66dc3c15970a4867eb96631`.
The target GGUF was
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`;
Q4_0 draft
`2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`,
D draft
`10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`,
combined rescue
`da36b46ee38469680c90ac4760fc79cd797f081e135ecc5427164b0b77bc331f`,
and fitted-head draft
`a5b35de1ed2bce7606cc5be41872a0ed08b94110261436ef49d4e36d6ff4e795`.
The primary manifest retains hashes for every other draft and exact block order.
The measured GPU was an RTX 5080 (compute capability 12.0, 16,303 MiB) under
WSL with driver/KMD 616.92 and CUDA user-mode driver 13.4. No RTX 2080 Ti
performance is inferred.

Quality runtime `a38e9d428218fa5f845363eb25499e1a51a274ca`, parent
`1cb78ed`. Remote project
`/home/philip/binary-eagle-decoding/rescue-head-20260927` retains raw outputs,
source/config/model hashes, failed and repaired gates, benchmark logs, and
supervisor states. Primary quality run: `rescue-quality-20260927`, directory
`results/rescue-quality`. Clean timing run:
`rescue-timed-primary-clean-20260927`, directory
`results/timed-primary-clean`. The interrupted full-prompt instrumented run is
`rescue-instrumented-primary-20260927` and is excluded from completed results.
Local ignored evidence is under `results/binary-rescue-head-5080`, including
the clean 34 MiB timing manifest, Nsight SQLite and per-process kernel analysis.
The partial local artifact index hashes 14 current evidence files (644,203,716
bytes in total) and identifies the preserved remote project; it will be
extended after the remaining diagnostics and raw archive are complete.
The remote checkout and all raw artifacts remain preserved for resumption;
the final archive/index and remaining diagnostic results are pending.

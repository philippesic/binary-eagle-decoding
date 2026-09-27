# Binary rescue and frozen-body head adaptation on RTX 5080

Status: **in progress**. Quality, body/head diagnosis and the bounded head fit are complete; repeated
performance measurements remain pending. No promotion decision yet.

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

Selected-weight denominators are218,234,880 logical matrix weights. Binary
payload includes packed signs plus F32 scales (1.25bits/weight); Q8_0 includes
block scales (8.5bits/weight). Norms remain F32 and d2t I64. Full-file figures
include GGUF metadata, padding and mapping bytes; model-parameter denominators
exclude integer maps. Parameter percentages are not runtime percentages. This is owned drafter payload;
the unchanged target embeddings remain shared higher-precision inputs, and the
fixed FP16 target is included in total serving memory and end-to-end timing.
Standard Q8_0's Q8_1 activation conversion differs from D's F16-cast activation
path: these are practical mixed-path interventions, not isolated weight-bit tests.
Per-projection executed-kernel profiling remains pending.

## Correctness and graph gates

- 112/112 binary operator cases passed on actual RTX5080 CUDA, plus CPU gates.
- Native loader fixtures accepted valid v2/v3 models and rejected wrong types,
  overlapping/omitted coverage and packed/dense shadow tensors.
- Historical3-prompt Q4/D capture off/on gates matched raw IDs and proposal
  digests; canonical Q4 self-replay and all five crossed arms passed exact round,
  verifier-label and output checks. Same-body cross-head states matched by bytes.
- Initial FP16-head export roundtrip matched raw IDs and captured body states.
- 32 real FP16-head states /1,024,000 logits gave max absolute surrogate error
  0.0035923, p95 0.0012449 and no argmax mismatch. The training computation uses
  F16-rounded inputs/weights with F32 accumulation; native MMVF uses half2
  partial accumulation and F32 reduction. It is an audited approximation.

All ten quality blocks report actual CUDA graph launches, with zero disabled or
incompatible direct fallbacks. Binary/mixed blocks report custom-operation graph
launches. Counters include setup and warmups; capture/recapture costs and direct
warmups are retained. They do not alone establish per-projection kernel timing.

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
conditional prefix survival. At position 1, supported-label median ranks are
1, 1, 3, 2 and 2 respectively; mean signed margins (label minus best other)
are +0.745, +0.236, -1.482, -0.978 and -0.980 logits. Full per-depth ranks,
margins, validity/support denominators and off-policy prefix records are saved.

The larger loss follows the D body: swapping a D head onto Q4 body costs about
5.9 percentage points at position 1, while swapping D body under Q4 head costs
19.8 points. Later-position deterioration also persists under a dense head.
This motivates the bounded compensation fit; it does not prove irrecoverable
information loss or predict the fitted head's live-chain result.

## Head fit and performance

The one approved fit completed 500 steps in 19.83 optimizer seconds. It used
8,160 states selected evenly from 40,815 aligned training states; 1,110 labels
were unsupported and excluded from CE while retained in coverage denominators.
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

Repeated performance will use matched graph-enabled builds, warmups, at least
five alternating repetitions, client wall latency/TTFT/stream arrivals, server
prefill/decode, paired distributions and uncertainty against Q4_0. Heavy state
and round dumps remain outside primary timed runs. Separate behavior-matched
round/kernel traces will support cost and conditional headroom interpretation.

## Reproducibility checkpoint

Quality runtime `a38e9d428218fa5f845363eb25499e1a51a274ca`, parent `1cb78ed`.
Remote project `/home/philip/binary-eagle-decoding/rescue-head-20260927` retains
all raw outputs, source/config/model hashes, failed gates and repaired gates.
Primary quality run `rescue-quality-20260927`; results `results/rescue-quality`.
Local compact evidence is under `results/binary-rescue-head-5080` in the main
checkout. Final artifact index, repeated timing and recommendation are pending.

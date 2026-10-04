# DSpark and DFlash native screen on RTX 2080 Ti

The seven-token released drafters beat the primary Q4_0 EAGLE baseline in every
paired repeat. DFlash D7 reached **132.37 request tokens/s, +45.8%**, and DSpark
D7 reached **129.49, +42.6%**. Their FFN-only Q4 variants also beat the paired
baseline by 55.9% and 55.6%. All four D7 conditions matched primary Q4_0 EAGLE
complete output IDs on all 144 measured requests per condition.

The supported comparison is complete: two native six-arm/six-repeat phases,
1728 measured requests plus 144 warmups. Genuine W1A8/W1A1 is unimplemented for
these models; its proposed 15-FFN exporter/loader/graph extension remains a
human scope choice. No activation-only flag is reported as binary-weight data.

## Request rate, decode rate, latency and acceptance

Released-reference phase:

| Arm | Request tok/s | Decode tok/s | Request rate / Q4 EAGLE | Mean HTTP request ms | Useful accepted / active draft round |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 63.75 | 66.23 | 0.702 | 1994.64 | — |
| Q4_0 EAGLE D5 | 90.80 | 96.12 | 1.000 | 1400.49 | 1.135 |
| DSpark D3 | 99.30 | 105.91 | 1.094 | 1280.65 | 2.087 |
| DSpark D7 | 129.49 | 140.87 | 1.426 | 982.08 | 3.254 |
| DFlash D3 | 107.13 | 114.88 | 1.180 | 1187.02 | 1.995 |
| DFlash D7 | 132.37 | 144.18 | 1.458 | 960.73 | 2.903 |

FFN-Q4 phase (only the fifteen gate/up/down matrices across five layers change):

| Arm | Request tok/s | Decode tok/s | Request rate / Q4 EAGLE | Mean HTTP request ms | Useful accepted / active draft round |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target-only | 58.34 | 60.57 | 0.705 | 2179.76 | — |
| Q4_0 EAGLE D5 | 82.75 | 87.59 | 1.000 | 1536.71 | 1.135 |
| DSpark D3 | 99.14 | 106.26 | 1.198 | 1282.69 | 2.089 |
| DSpark D7 | 128.77 | 141.12 | 1.556 | 987.52 | 3.266 |
| DFlash D3 | 105.23 | 113.45 | 1.272 | 1208.50 | 1.993 |
| DFlash D7 | 128.99 | 141.28 | 1.559 | 985.87 | 2.884 |

Each arm has 144 measured requests. Rates pool output counts and actual times;
latency is local nonstreaming HTTP wall after load/warmup. Useful acceptance
counts only the consumed accepted prefix, excluding verifier matches past EOS.
Its denominator is active complete draft rounds; no-proposal records and the
possibly untraced initial seed remain separate. Raw counts, p50/p95 latency,
conditional position survival, rejection locations and per-repeat ratios are
retained in the analysis artifacts. TTFT and maximum concurrent serving capacity
were not measured. Regular native round tracing is included in both phases;
CUDA events, graph splitting, logit hooks and admission traces are excluded.

**The phased screen does not isolate a causal quantization speed effect.**
Q4-phase target-only and EAGLE request rates were 8.5% and 8.9% lower than their
reference-phase rates, while the D7 candidate absolute rates were nearly flat
(DSpark−0.6%, DFlash−2.6%). Every candidate beat its own phase's paired baseline
in all six repeats. Existing hardware metadata is retained; no unmeasured cause
for the cross-phase anchor change is asserted. An interleaved precision study
would be needed before attributing a normalized-ratio change to quantization.

## Precision, layout and memory

Actual RTX2080Ti/SM75, 11264MiB, driver 610.74, CUDA 12.8.93; F16 Qwen3-4B target,
F16 target/draft KV, context 2048, batch/microbatch 32, concurrency 1. Fixed 24
unsealed prompts comprise eight prose, eight code and eight reasoning tasks.
Both phases use greedy seed 42, nonthinking, max 128 outputs, two warmups and
six Williams-balanced repeats. No reserved-final or training data was used.

Released DSpark/DFlash matrices retain original BF16 source storage with F32
norm widening. BF16 storage is not a BF16 Tensor Core claim on SM75. Both
private original embeddings and full heads are retained: their canonical source
values differ slightly from the target, so approximate borrowing was rejected.
All teacher embedding/head bytes and identities remain unchanged in native
admission. Explicit draft-dspark mode, author slot 0, seven bidirectional noise
rows, clean-cache reinjection and rejection repair are preserved for D3 and D7.
Native fcdf582 reserves seven draft outputs for the short read without changing
its target verifier's four-output capacity.

FFN-Q4 is weight-format coverage on exactly 15 matrices, not full-model Q4 or
W4A4. Embedding, full head, attention, fusion, Markov, confidence/norms and target
retain their admitted bytes/types. Native kernel/activation details are separate
profile evidence. FFN payload falls from 747,110,400 to 210,124,800 bytes: 512.11MiB
saved. Actual probe peaks fell DSpark 10509→9977MiB and DFlash 10361→9827MiB;
full timing snapshots show roughly 512MiB lower residency per corresponding arm.
No lower target precision or new binary kernel was used to fit the models.

## Correctness limits

All divergence indices are zero-based generated-token positions. D7 in both
formats matches primary Q4_0 EAGLE on all 144 requests per condition. Primary
EAGLE and all D7 variants differ from target-only at prose-01/index89. Its
source-bound raw-logit gate passed, with opposing margins 0.008221/0.003122 and
centered common-top-five difference 0.008618. This supports bounded native
near-tie sensitivity, not bit parity or a proved batch-arithmetic cause.

D3 variants additionally differ from primary on prose-01/index94 and
prose-06/index90. The94 gate passed with opposing margins 0.001274/0.005501 and
centered difference 0.004618. The one original-order BF16 p90 diagnostic
reproduced all 16 complete warmup/prompt requests; its reported opposing margins
are 0.002527/0.002306 and centered difference 0.005583, with both competing IDs
finite and in both top-five rows. Final source-bound CPU receipt is pending
readback; no universal output-equivalence claim is made for short arms.

The Q4 admission preserved its initial strict output/checker failures. A scoped
receipt binds all 20 probe candidate outputs and terminations to complete
admitted reference paths. The token-only first-three comparison was corrected:
all five first noise blocks per architecture have matching recorded numerical
injection histories and matching first-three proposals. Later token-equal but
unequal-history joins numbered DSpark 122 and DFlash 146, with 2 and 6 proposal
changes (one and three unique repeated paths). Matching-history disagreements
still reject. These records do not prove full fused-KV byte equality or harmless
rounding; target-feature batch histories differ before the changed decisions.
Masks, cache ancestry, target immutability and native greedy verification pass
independently. No target near-tie threshold is transferred to draft logits.

## Evidence and measured budget

Public native source: fcdf5822c5b78f9dbcfd1f5c7106f09c3f0c9b1a;
CUDA binary: 1bd67cdf74d6ced49454ca2546d9a462e71d4e695b8fb5e52d7359fa68473822.
Frozen protocol: 367431663597d312bf4cc75544d3e8c1994260a0d3cd266558cdd388b48ceca7.
Asset/release/loader/export ancestry and exact commands are in operator.md and
native-runtime-admission.md; model weights and raw runs remain outside Git.

Reference parent a9dded1; measurements
74b9c9dfc4f16bf549502efe0fe779f22ac7e1a8db5839e008a9ee655aa31381;
analysis 07c85d24f846c06fd9eea4a1b9188d28dd9d5548efb008fa8620d87472b78f99.
Q4 parent a7c3f50; measurements
92c4d132df8ce330bfbc0a1dfab13e849fd5b9ce5a8752c3910ada75e457433b;
analysis 138be59ba135f5e421afc49ed8afc1dedc219cec7e17d263fbb513db7380c91e.
Q4 numerical receipt d9e75322b43b79ea147fe5fb2a1eb72847f169671eaeb3172c0b4a422112cb79;
history-aware admission b3055c5ee2c76d20aff23b425027530ba81ff36ad91bc9a70b79372d0f29704e.

Reference request inference 1246.091775789s + Q4 inference 1306.184010578s =
**2552.275786367s of the shared 7200s limit**, with 4647.724213633s unused.
Q4 config hashes and carries the original reference receipt. Failed timed
requests would remain charged; both phases finished exit0. Hashing, model load,
conversion/builds, admission and separate diagnostic profiles are overhead
outside this measured-request allowance. Q4 supervisor started 03:59:17.479593Z,
ended 04:25:43.704522Z; all owned groups/contexts stopped before profiling.

No W1A8/W1A1 inference, training/QAT, RTX5080 action, other-chat contact or
baseline change is implied. The remaining human decision is the concrete
FFN-only W1 exporter/loader/graph extension documented in precision-admission.md.

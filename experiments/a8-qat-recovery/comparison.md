# Matched A8 QAT comparison — interim

The matched comparison is still running on RTX5080/SM120. Both arms have real
optimizer updates and exact positive-step restore evidence. At 5,000 and 10,000 updates,
the combined candidate trails the fixed A8 reference; both
remain below Q4_0. The matched cumulative-budget endpoints are still pending.
This report will be replaced with final measured results after both 7,200-second
cumulative training caps and final native development evaluations finish.

## Frozen comparison

Both arms read from the same authenticated corpus of 10,000 TRAIN prompts and
3,899,930 supervised rows. Actual unique training coverage will be reported at
the endpoints. Both use seed 8101, hard CE, captured native target features/labels,
fixed cache/mask/vocabulary ancestry and the same data order. No A1, Bop, curriculum, refresh,
fusion correction or architecture change. Reference: fixed A8/symmetric binary
weights, latent magnitude 0.5. Candidate: learned A8, all-nine affine midpoints,
latent magnitude 0.1. Both use existing AdamW, sign LR 0.001, scale/activation/
midpoint LR 0.00001 where present, warmup 100, norm clip 1.

Both request cache/head optimization. Actual cache size 1/chunk size 64 and reference
batched head are observed. Candidate training intentionally uses serial head
execution to preserve invocation-local learned-quantizer gradients; inference
uses batched/native execution. Every candidate tensor had finite/nonzero gradients
and measured sampled movement: sign 9/9, scale 9/9, quantizer 6/6, midpoint 9/9.

Execution source 3bd4837850915cb7d308290e573e8d8a08eee1a2 is a helper-only fix atop
583480c79f3ca090de0952deca8278dcb7f15c2d. Training runtime hashes are unchanged.
Native runtime 9e2c7a90051e738751aab7d7bd7c2d8201fb76e3, CUDA RTX5080/SM120,
Torch 2.14.0+cu130. Target weights and target/draft KV are F16; A8 uses native packed
INT8 dispatch. Q4_0 EAGLE is the primary baseline with native Q8_1 activation
conversion. These results do not establish SM75 performance.

## Available interim measurements

All measurements use the same 24 unsealed development prompts
(SHA 131a3db7958ff6aa818b23019297654507d5b80bed3c298349417b7e3b2ba081).
Timing has five measured repetitions /120 requests per variant, with warmups excluded,
A8/Q4_0/target-only order alternated. Request rate counts actual output tokens /
complete HTTP wall time including prefill; decode uses server predicted_ms.
Native acceptance is one instrumented pass per checkpoint; timing uses five
separate repetitions. TTFT is unavailable for this nonstreaming endpoint.
Evaluation wall time is separate overhead and is never a serving-rate denominator.

| Checkpoint | Development CE | Accepted drafts/round | Acceptance rate | Request tokens/s | Relative to Q4_0 |
|---|---:|---:|---:|---:|---:|
| Reference step 0 | 7.5459 | 0.122376 | 2.5012% | 66.991 | 0.4949 |
| Candidate step 0 | 7.5463 | 0.122376 | 2.5012% | 63.524 | 0.4703 |
| Reference step 5,000 | 5.1176 | 0.377184 | 7.7106% | 81.477 | 0.6013 |
| Candidate step 5,000 | 5.5689 | 0.217877 | 4.4587% | 68.342 | 0.5056 |
| Reference step 10,000 | 4.2628 | 0.500529 | 10.2403% | 88.069 | 0.6519 |
| Candidate step 10,000 | 6.0285 | 0.180342 | 3.6933% | 66.145 | 0.4898 |
| Reference step 15,000 | 3.9946 | 0.581382 | 11.8942% | 92.340 | 0.6856 |
| Q4_0 at reference step 5,000 | n/a | 1.306255 | 26.5917% | 135.491 | 1.0000 |

Accepted drafts per round is the primary acceptance metric. Acceptance rate uses
proposed drafts as its denominator and is reported separately. Original native
capture counts are:

| Capture | Accepted | Proposed | Rounds | Emitted tokens |
|---|---:|---:|---:|---:|
| Reference step 0 | 309 | 12,354 | 2,525 | 2,834 |
| Candidate step 0 | 309 | 12,354 | 2,525 | 2,834 |
| Reference step 5,000 | 777 | 10,077 | 2,060 | 2,834 |
| Candidate step 5,000 | 507 | 11,371 | 2,327 | 2,834 |
| Reference step 10,000 | 946 | 9,238 | 1,890 | 2,834 |
| Candidate step 10,000 | 433 | 11,724 | 2,401 | 2,834 |
| Reference step 15,000 | 1,043 | 8,769 | 1,794 | 2,834 |
| Q4_0, each capture | 1,608 | 6,047 | 1,231 | 2,834 |

Both zero checkpoints have identical native acceptance counts. Each A8 report
uses 24 development prompts, 48 loss rounds and 229 loss labels; all 24 native
response sequences match Q4_0. Each timing report has 120 matched request token-ID
sequences per comparison. Native capture counts and the separately measured
request-timing workload have distinct ledgers.

For reference step 5,000, median/p95 full-request latency is 1.543/1.811 seconds
for A8 versus 0.909/1.092 seconds for Q4_0. These are pooled request distributions
across different prompt lengths, not confidence intervals. Decode rate is
84.305 versus 143.511 tokens/s (ratio 0.5874). Target-only request rate is
89.439 tokens/s, also above the current A8 result.

Candidate step 5,000 has median/p95 full-request latency 1.868/2.070 seconds
versus Q4_0 at 0.913/1.102 seconds. Decode rate is 70.431 versus 143.253 tokens/s
(ratio 0.4917). Target-only request rate is 88.642 tokens/s. Each A8 throughput
ratio uses the Q4_0 measurement from the same evaluation. The candidate evaluation
completed exit 0, with 24/24 native response matches and 120/120 timing matches
for both A8 and target-only. Its supervised runtime was 761.223 seconds; raw
reports and captures are archived with verified hardlinks.

At reference step 10,000, request rate is 88.069 versus Q4_0 at 135.098 and
target-only at 88.782 tokens/s. Decode rate is 91.444 versus 142.974 tokens/s
(Q4_0 ratio 0.6396). Median/p95 full-request latency is 1.427/1.675 seconds versus
Q4_0 at 0.914/1.096 seconds. Each timing variant generated exactly 14,290 returned
token IDs over 120 measured requests, with 105 length finishes and 15 stop finishes.
All 24 native response sequences and all 120 timing sequences match Q4_0.
The evaluation completed exit 0 in 692.183 supervised seconds; raw evidence is
hardlink archived. Acceptance and throughput improved, but remain below Q4_0.

At candidate step 10,000, acceptance fell from 0.217877 at step 5,000 to
0.180342 accepted drafts/round, while CE rose from 5.5689 to 6.0285. Request rate
is 66.145 versus Q4_0 at 135.031 and target-only at 88.594 tokens/s; decode rate
is 68.043 versus 142.875 tokens/s. Median/p95 request latency is 1.965/2.189 seconds
versus Q4_0 at 0.911/1.098 seconds. The complete evaluation passed in 783.216
supervised seconds, with 24/24 native and 120/120 timing matches. Each timing
variant generated 14,290 returned token IDs (105 length finishes, 15 stop finishes).
These results show a regression on this fixed development set; they do not identify
which combined candidate feature caused it. The frozen experiment continues.

Reference step 15,000 reached 0.581382 accepted drafts/round and 92.340 request
tokens/s versus Q4_0 at 134.686 and target-only at 88.556. Decode rate is 96.162
versus Q4_0 at 142.763 tokens/s. The complete evaluation passed in 683.692
supervised seconds, with 24/24 native and 120/120 timing matches. Each timing
variant generated 14,290 returned IDs with 105 length and 15 stop finishes.
Reference exceeds target-only throughput in this measurement, while remaining
below the primary Q4_0 baseline; no significance claim is made from pooled rates.

Candidate's 1,047.285 trainer seconds differ from the reference's 764.725 seconds
at the same update count; these checkpoint measurements are intermediate observations. The final comparison
requires both independent 7,200-second cumulative endpoints. No quality, latency
or total-throughput win against Q4_0 is claimed. This combined candidate does not isolate
individual learned-quantizer, midpoint or latent-inertia effects.

## Resume and recovery evidence

Reference exact restore 859→912 and candidate 105→1200 passed with optimizer/RNG/
cursor/recipe/probe/telemetry preserved. Human pause saved candidate step 2081/cursor 2084
(checkpoint ac8af0f058b7d1d...), budget 441.281998629 seconds, no active attempt;
human resume restored it and reached step 5000/cursor 5009. Cumulative candidate trainer
budget is 1047.284754942 seconds, settled with no active attempt; its scheduled
5,000 evaluation passed and is archived. Pause downtime is excluded.
Reference step 5,000 budget settled at 764.725 seconds; after candidate evaluation,
exact resume advanced the reference to step 5,411 and 829.816 cumulative seconds.
Each arm retains its own 7,200-second
cap; standalone evaluations are bounded at 1,200 seconds and arms alternate at natural
5,000-update development boundaries. Intermediate snapshots save every 1,000 updates.

Two preserved pretraining execution incidents recovered within the two-retry
limit: unbalanced TRAIN gate selection was replaced with already-declared balanced
TRAIN gate prompts using the authenticated existing bootstrap; a timing-helper
ID-prefix mistake was fixed to accept exact frozen prepared development bytes.
No dataset recapture, recipe/precision/budget change or discarded optimizer work.
Checkpoint and successful raw evaluation archives use verified same-filesystem
hardlinks before pruning. Original historical paired step 1,000 remains untouched.

Reference training subsequently segfaulted at step 9,681. The exact step 9,000
checkpoint and raw failure were preserved; the first same-recipe retry restored
that checkpoint and has positive updates. Built-in recovery retained 1,085.811
seconds for the failed attempt, making the settled reference budget 1,850.536
seconds. About 379.457 seconds of that charge is conservative post-crash downtime.
The original cap remains 7,200 trainer-accounted seconds; it is not a claim of
exactly two hours of productive optimizer work after such recovery. Uncommitted
updates 9,001–9,681 were rolled back and replayed from the checkpoint; their first
attempt remains charged. Retry 1 passed the failure point and reached step 10,000/
cursor10,017, exit 0, with budget settled at 2,005.419 seconds. Its scheduled native
evaluation passed and is archived; candidate resume reached step 10,000/cursor10,017, checkpoint6898e9b1,
with settled budget2,064.260seconds. Its scheduled native evaluation passed and is archived.
Reference resumed from step 10,000 and reached the scheduled step 15,000 boundary
with budget settled at 2,767.995 seconds; its native evaluation passed and is
archived. Candidate resumed from 10,000 and has positive optimizer updates toward
15,000. Both final endpoints remain pending. Fault cause is
unexplained; equivalent recurrence retains the same retry limit (one of two used).

## Endpoint reporting method

Final coverage will use the authenticated checkpoint's committed `unique_prompts`
and `unique_rows` sets, with the counts exposed as `unique_prompts` and
`unique_supervised_rows` in status. `presented_supervised_tokens` includes replay;
epoch replay does not inflate distinct counts. Abandoned work rolled back after a
crash can remain charged without appearing in committed coverage.

Exact cumulative per-update sign flips are separate from sampled diagnostic sign
flips and flip-backs. Final telemetry will state its observation step/gap; the
latest sample can precede the endpoint by up to 99 updates. Aggregate near-zero
telemetry uses threshold 0.01, while layer diagnostics use 0.05. Per-family
movement admission samples one maximum-gradient element per tensor and proves
observed movement; it does not measure full-tensor displacement.

Each endpoint must have a settled 7,200-second budget with no active attempt,
authenticated final checkpoint and matching final development request/result.
Accounting can conservatively retain failed work and clamp at the cap while an
in-flight update or serialization finishes. Endpoint timing is trainer-accounted
time; startup, evaluation and physical process-release evidence remain separate.

## Artifact locations and pending completion

Reference step 15,000 identities: checkpoint
`73429bd0ca2302147a2ee7cf883664a63323da5e229f0e3611a6f5e044a940eb`, report
`cc5341b12d5733465f5b4ceecc10b9b8c0671ab33a39435703c1cfe23057af74`, result
`8490312be6eafca3c0c7233271eb140a4089c6115019eeaf28c5af1f0f9cde33`, timing
`0c1ade69a26b2d4cec65c096ea97231aae1e6fbd6dc3b9954b5589f3fa59d317`.
Archive locator: `evidence-archive/development/reference-step15000-locator-map.json`.

Candidate step 10,000 identities: checkpoint
`6898e9b1cb6cb0303e1a38340a23decb3eeabf14d215afd761d7e0c8aa7496ae`, report
`e561e03ce99c049458d815239f7c0d8ea3433abcaf5be63cbc31ad032fa5acbd`, result
`3bbbb913eac5c0bb3880815a0da3bf8221f320a835696dadd78cb479dbafbb74`, timing
`fc12102cd0637c36848e05954a0b7f84879d16a01b7e7fd0d4887a2340f6a8d9`.
Archive locator: `evidence-archive/development/candidate-step10000-locator-map.json`.

Reference step 10,000 identities: checkpoint
`281546e75c0d1d010a32b5477890788ad77077e1ec6819ceae745a2ac5f24f78`, report
`3481aeda0ac35e28a6579c3fee9b25ef8e554e459504c5a3212abaa039de2d7c`, result
`0d2d7fcfe05f658b5573c13b4ac03bd92f910458cdbf88aaa24acd84dfc57599`, timing
`b21cf1b3ce16c83e934eacd14fc964bc8876610018744e8dd82f3b052bf6a278`.
Archive locator: `evidence-archive/development/reference-step10000-locator-map.json`.

Candidate step 5,000 identities: checkpoint
`893e205d4e64fa04457c3281980570bb5ad13b3fb803cc1fe5f30d2d8b51ad10`, report
`b5845c6340d8a70fb5f3661d48ee469ad54a7ba69d1039b563dab81882ce54f9`, result
`be7072ca05466a7bf850fce4a99580e20d96b09651448679064b7cdbbdfc1e2b`, timing
`c8876de6f06b703a8d0d1d7c3ffe20f0b95f71d60a9c0fc8cb425bdfc62da779`.
Original archive locator: `evidence-archive/development/candidate-step5000-locator-map.json`.

Raw runs/archives live outside Git beneath
`/home/philip/binary-eagle-decoding/checkouts/a8-qat-run-583480c7/runs/qat-a8-comparison-20261003-01`.
Evaluator/supervisor source is the separate immutable 3bd checkout. Exact
original report/result/timing/checkpoint hashes and aggregate counts are
also transcribed in ignored `runs/qat-a8-recovery/comparison-interim-summary.json`.
Exact commands, birth ticks, hashes, budget ledgers, archive locator maps and
incidents are in
ignored local `runs/qat-a8-recovery/operator-ledger.json`. Durable current state:
[goal](../../docs/goals/a8-qat-recovery-and-comparison.md).

Pending: later scheduled and both final reports, both cumulative cap endpoints,
final training coverage and telemetry, complete timing distributions and final
resource-release proof. Monitor stays active until completion or a human pause. No sealed-final prompts are accessed.

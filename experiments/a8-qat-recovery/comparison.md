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
| Candidate step 15,000 | 5.9516 | 0.209471 | 4.2856% | 68.261 | 0.5003 |
| Reference step 20,000 | 3.6999 | 0.613546 | 12.5495% | 93.850 | 0.7001 |
| Candidate step 20,000 | 5.7818 | 0.205183 | 4.1920% | 67.751 | 0.4989 |
| Reference step 25,000 | 3.5502 | 0.637782 | 13.0419% | 95.171 | 0.7092 |
| Candidate step 25,000 | 5.8161 | 0.230803 | 4.7259% | 68.954 | 0.5111 |
| Reference step 30,000 | 3.5495 | 0.623711 | 12.7413% | 95.036 | 0.7028 |
| Candidate step 30,000 | 5.5156 | 0.268456 | 5.4950% | 70.905 | 0.5247 |
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
| Candidate step 15,000 | 491 | 11,457 | 2,344 | 2,834 |
| Reference step 20,000 | 1,078 | 8,590 | 1,757 | 2,834 |
| Candidate step 20,000 | 483 | 11,522 | 2,354 | 2,834 |
| Reference step 25,000 | 1,104 | 8,465 | 1,731 | 2,834 |
| Candidate step 25,000 | 532 | 11,257 | 2,305 | 2,834 |
| Reference step 30,000 | 1,089 | 8,547 | 1,746 | 2,834 |
| Candidate step 30,000 | 600 | 10,919 | 2,235 | 2,834 |
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

Candidate step 15,000 recovered slightly to 0.209471 accepted drafts/round,
with CE 5.9516 and 68.261 request tokens/s. Q4_0 measured 136.437 and target-only
89.328 tokens/s. Decode rate is 69.893 versus Q4_0 at 143.172 tokens/s. Median/p95
full-request latency is 1.863/2.090 seconds versus Q4_0 at 0.899/1.085 seconds.
All 24 native and 120 timing sequences match Q4_0. Every timing variant generated
14,290 returned IDs (105 length, 15 stop finishes). Supervisor wall time was
773.704 seconds; the reported evaluation phase was 769.607 seconds. Terminal
absence was observed later at 09:35:23 UTC. The result remains below reference
step 15,000 and Q4_0; no individual feature effect is inferred.

Reference step 20,000 reached 0.613546 accepted drafts/round, CE 3.6999 and
93.850 request tokens/s. Q4_0 measured 134.046 and target-only 88.441 tokens/s.
Decode rate is 97.899 versus Q4_0 at 142.269; median/p95 full-request latency is
1.347/1.599 seconds versus 0.921/1.112 seconds. Every timing variant generated
14,290 returned IDs with 105 length and 15 stop finishes, and all 24 native/
120 timing sequences match Q4_0. The evaluation passed in 686.691 supervisor-wall
seconds (681.765 reported evaluation-phase seconds). Raw evidence is archived.
Reference remains below Q4_0 on acceptance and full-request throughput.

Candidate step 20,000 measured 0.205183 accepted drafts/round, CE 5.7818 and
67.751 request tokens/s versus Q4_0 at 135.812 and target-only at 88.851.
Decode rate is 69.619 versus 143.280 tokens/s. Median/p95 full-request latency is
1.859/2.126 seconds versus Q4_0 at 0.906/1.087 seconds. The complete evaluation
passed in 776.723 supervisor-wall seconds (771.597 evaluation-phase seconds),
with 24/24 native and 120/120 timing matches. Each timing variant generated
14,290 returned IDs with 105 length and 15 stop finishes. The candidate remains
well below reference step 20,000 and the primary Q4_0 baseline.

Reference step 25,000 reached 0.637782 accepted drafts/round, CE 3.5502 and
95.171 request tokens/s versus Q4_0 at 134.190 and target-only at 88.418.
Decode rate is 99.411 versus Q4_0 at 142.781 tokens/s. The complete evaluation
passed in 683.708 supervisor-wall seconds (678.563 evaluation-phase seconds),
with 24/24 native and 120/120 timing matches. Each timing variant generated
14,290 returned IDs with 105 length and 15 stop finishes. Reference continues
improving but remains below the primary Q4_0 acceptance and throughput baseline.

Candidate step 25,000 measured 0.230803 accepted drafts/round, CE 5.8161 and
68.954 request tokens/s. Q4_0 measured 134.899 and target-only 88.712 tokens/s;
decode rate was 70.994 versus Q4_0 at 142.655. The complete evaluation passed in
767.206 supervisor-wall seconds (762.473 evaluation-phase seconds), with 24/24
native and 120/120 timing matches. Each timing variant generated 14,290 returned
IDs with 105 length and 15 stop finishes. Candidate acceptance improved slightly
from 20,000, while remaining well below reference step 25,000 and Q4_0.

Reference step 30,000 measured 0.623711 accepted drafts/round, CE 3.5495 and
95.036 request tokens/s versus Q4_0 at 135.229 and target-only at 88.711.
Decode rate was 98.995 versus Q4_0 at 143.191 tokens/s. The complete evaluation
passed in 674.689 supervisor-wall seconds (669.613 evaluation-phase seconds),
with 24/24 native and 120/120 timing matches. Each timing variant generated
14,290 returned IDs with 105 length and 15 stop finishes. Acceptance is slightly
lower than at reference 25,000, while throughput is close; both remain below Q4_0.

Candidate step 30,000 measured 0.268456 accepted drafts/round, CE 5.5156 and
70.905 request tokens/s versus Q4_0 at 135.129 and target-only at 88.534.
Decode rate was 73.055 versus Q4_0 at 142.891 tokens/s. The evaluation passed
in 769.214 supervisor-wall seconds (764.960 evaluation-phase seconds), with
24/24 native and 120/120 timing matches. Every timing variant generated 14,290
returned IDs with 105 length and 15 stop finishes. This is the candidate's best
scheduled acceptance so far, but remains below reference and Q4_0. Candidate
training has used 6,219.893 of 7,200 seconds; its final endpoint is pending.

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
archived. Candidate resumed from 10,000 and reached 15,000/cursor15,024 with budget settled
at 3,093.120 seconds; its scheduled native evaluation passed and is archived. Both final endpoints remain pending. Fault cause is
unexplained; equivalent recurrence retains the same retry limit (one of two used).

Reference subsequently reached step 20,000/cursor20,028 and exited cleanly,
with budget settled at 3,541.358 seconds; its scheduled native evaluation passed
and is archived. Candidate resumed from 15,000 and has positive optimizer updates
toward 20,000. Candidate remains evaluated at 15,000 and 3,093.120 seconds. Both final
7,200-second endpoints remain pending.

Candidate subsequently reached step 20,000/cursor20,028 and exited cleanly,
with budget settled at 4,132.197 seconds. Its scheduled native evaluation passed
and is archived. Both final 7,200-second endpoints remain pending.

Reference subsequently reached step 25,000/cursor25,044 and saved checkpoint
e3948dbd, with budget 4,324.038 seconds. Its scheduled native evaluation passed
and is archived. Candidate resumed from 20,000 and has positive optimizer updates
toward 25,000. Both final 7,200-second endpoints remain pending.

Candidate subsequently reached step 25,000/cursor25,044 and exited cleanly,
with budget settled at 5,181.483 seconds. Its scheduled native evaluation passed
and is archived. Reference resumed from 25,000 and has positive optimizer updates
toward 30,000. Both final 7,200-second endpoints remain pending.

Reference subsequently reached step 30,000/cursor30,054 and exited cleanly,
with budget settled at 5,097.991 seconds. Its scheduled native evaluation passed
and is archived. Candidate resumed from 25,000 with positive optimizer updates
toward 30,000. Both final 7,200-second endpoints remain pending.

Candidate subsequently reached step 30,000/cursor30,054 and exited cleanly,
with budget settled at 6,219.893 seconds and 980.107 seconds remaining. Its
scheduled native evaluation is running. The next candidate training segment must
stop at the original cap and receive a final native evaluation, even if the
endpoint precedes step 35,000. Both final 7,200-second endpoints remain pending.

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

Candidate step 30,000 identities: checkpoint
`f68255ace54f4803ba8f856f72b65b1ae86fbf9c06ee805ab5cc54bea4dc9c8f`, report
`a1325ecd864e1ea0253af5a280b0a855ebc1528d6057cfb98cd929dc46852424`, result
`c36cebb927d39de8ec05241d1bf112f14d8875dc350db1c6964a0af5c520bc03`, timing
`b1a35a60a09e2831009d21305c2600c0f1a524c4b557478a3954ae0c63e5811a`.
Archive locator: `evidence-archive/development/candidate-step30000-locator-map.json`.

Reference step 30,000 identities: checkpoint
`108968085f4b99ad3ae3fd600b14bee702aa821885c742120d41d988d351940c`, report
`faf585b9e9f35c15ff0b2fdba68158cfa06a9157a9aaddeecefb00c2494cc6b5`, result
`cb038ef2254cd8131a500db8b2a93c70fe138598612963d63af05530e2678043`, timing
`8e580a5bf94d8935c570975d619c4b0a81d9738ff34487fb397a2cc11f428d75`.
Archive locator: `evidence-archive/development/reference-step30000-locator-map.json`.

Candidate step 25,000 identities: checkpoint
`b9936e38bee428cbfb8999fe8bb153fa0b014347561543b5d76c09dc87900cdb`, report
`4220d6295c2b72919c8822a528511b15f91eab8fbfb1d7d1de83cf7e4a3fb5a3`, result
`e581a54ed357a6e9cee52482f9b5738a15811c914922725cddb64889c75dc8b1`, timing
`3f17c3204220b1a06328e2105a338d037c1d68c59cb8040f89f69ad0f742760b`.
Archive locator: `evidence-archive/development/candidate-step25000-locator-map.json`.

Reference step 25,000 identities: checkpoint
`e3948dbd55be4b443d1d0ff8bb2c6e4b5c08b3e8308e69c435b8692b5c3088a8`, report
`d80978e41eb18231f59eb1be1ddb9aa0554e609624802d7d9b4d84c5182ebc60`, result
`d99f100e0a2359af26477dcf82b5cc510bbe2bf7bd55855ee6e70e1944b3aba7`, timing
`d44833b524a2dfb47593ecb8a9c71d6c86cf1751a251d1e8281d2a0ee81de8a9`.
Archive locator: `evidence-archive/development/reference-step25000-locator-map.json`.

Candidate step 20,000 identities: checkpoint
`21f5278f9489b166067a59d83f22f454cacca975e419b0a010a15cdf7df75865`, report
`624c8e3397bd5c5c6a6deb6e88431e268ba9af80bcdaaa5f4412767b220a3d64`, result
`c2c9280adf5f3f91c717e05a1b6c1fda6534229bb7ca08bf6002c9f0a1c3a954`, timing
`a505b9868f1338097fe5ad14ff2f6d4914b6166f2bd8a87b55296408fe6ef36c`.
Archive locator: `evidence-archive/development/candidate-step20000-locator-map.json`.

Reference step 20,000 identities: checkpoint
`1fab73c38bab8992fb0c92a0a56691472674ec56a0d465f1fe7b6efa5169ab8d`, report
`3e4d03de6864109075dfb356dfa66e78ec9f326fb461e69ae53f3b8524c20721`, result
`57a60fb72032eb8c49ac2643f333f22e3d4fb82eb3cad2faf7bd5705a7c26d8a`, timing
`d9291b9980466ccf7c989051ce4739c5397e875a0f827633c5081b6807a9a5b4`.
Archive locator: `evidence-archive/development/reference-step20000-locator-map.json`.

Candidate step 15,000 identities: checkpoint
`50e0764ce0a7c343366c04827605b6c12b5fc82f87dec5063ccea316bf12f102`, report
`445e3af8925102393dc4ac0cd86d93463d69010a5f2d2e76eb2ee9a53e9a19b9`, result
`276352166478317be789dee4bb5683b00a3a1588419df8c49d8a8e626ef7cda9`, timing
`1667c7dc07b174ae9eb7915594b84fd7422980b6d8c083bd1652f228bc28fe23`.
Archive locator: `evidence-archive/development/candidate-step15000-locator-map.json`.

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

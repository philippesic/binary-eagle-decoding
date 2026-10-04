# Matched A8 QAT comparison — final

Both arms completed the original **7,200 cumulative trainer-accounted seconds**
on **RTX5080/SM120** and passed matching final native development evaluations.
**Neither arm beat the primary Q4_0 EAGLE baseline** on acceptance, request latency
or complete-request throughput. Reference reached **0.638568 accepted drafts/round**
and **95.689 request tokens/s (70.47% of paired Q4_0)**; the combined candidate reached
**0.247581** and **69.623 tokens/s (51.68%)**. The candidate also trails reference on
this fixed development set. The combined recipe does not isolate feature effects.

## Equal-budget endpoints and actual training coverage

| Arm | Accounted seconds | Committed updates | Cursor | Unique TRAIN prompts | Unique supervised rows | Presented supervised tokens |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 7,200.0 | 43,203 | 43,287 | 531 | 206,163 | 206,163 |
| Candidate | 7,200.0 | 34,731 | 34,797 | 426 | 165,733 | 165,733 |

Both budget ledgers are `7200/max7200/active_attempt=null`. Final status is
`completed/final_training_complete=true`; latest checkpoint, final development
request and completed result agree. CPU-only reads of the original checkpoints
join step/cursor/coverage/token/sign-flip counters to final status. Both epochs are
zero. Available corpus is 10,000 TRAIN prompts / 3,899,930 supervised rows;
reference consumed 5.31% of available prompts and 5.2863% of rows,
candidate 4.26% and 4.2496%. These are committed checkpoint counts.
Abandoned work after a crash can remain charged outside committed coverage.

Reference's unexplained SIGSEGV at step 9,681 restored step 9,000 on retry 1 of max 2.
The failed attempt retained 1,085.811 seconds, including about 379.457 seconds of
conservative post-crash downtime. Uncommitted updates 9,001–9,681 were rolled back
and replayed; their first attempt remains charged. No refund/topup, new incident
label or extra training block. This comparison uses the agreed cumulative
accounting policy. The original historical paired A8/A1 step 1,000 remains intact.

## Frozen recipe and effective execution

Same seed 8101, data order, hard CE, captured native target features/labels,
cache/mask/vocabulary ancestry and update cadence. Reference uses fixed A8,
symmetric binary weights and latent initialization 0.5. Candidate combines learned
A8 quantizers, all-nine affine midpoints and latent initialization 0.1. Both use
AdamW, sign LR 0.001, small-family LR 0.00001, warmup 100 and norm clip 1. A1/Bop/
fusion/curriculum/refresh and additional training were excluded.

Both requested cache/head optimization: cache size 1/chunk 64 is observed. Reference
training uses the batched head; learned-quantizer candidate training intentionally
uses serial head calls to retain invocation-local gradients. Candidate inference
uses the native/batched path. Actual admission established finite/nonzero gradients
and sampled maximum-gradient-element movement for every tensor: sign 9/9,
scale 9/9, learned A8 activation 6/6, midpoint 9/9. That admission demonstrates observed
movement, not full-tensor displacement. Both arms have exact positive-step
optimizer/RNG/cursor/recipe/probe/telemetry restore evidence.

Training math is frozen at `583480c79f3ca090de0952deca8278dcb7f15c2d`; executed helper
source `3bd4837850915cb7d308290e573e8d8a08eee1a2` only repairs evaluator integration.
Native runtime `9e2c7a90051e738751aab7d7bd7c2d8201fb76e3` uses shared pack and unused-head
pruning. CUDA RTX5080/SM120, Torch 2.14.0+cu130; target/verifier weights and both KV
caches are F16. Drafter is native row W1A8 with packed INT8 dispatch. Q4_0 draft uses
native Q8_1 activation conversion. These results do not establish SM75 performance.
FP16 EAGLE was not measured in this frozen comparison; Q4_0 is the primary baseline.

## Final native acceptance and development loss

| Arm | Development CE | Accepted drafts | Proposed drafts | Speculative rounds | Accepted/round | Acceptance rate | Emitted/round |
|---|---:|---:|---:|---:|---:|---:|---:|
| Reference | 3.696147 | 1,106 | 8,463 | 1,732 | 0.638568 | 13.0687% | 1.636259 |
| Candidate | 5.534134 | 563 | 11,113 | 2,274 | 0.247581 | 5.0661% | 1.246262 |
| Q4_0 (both final captures) | unavailable | 1,608 | 6,047 | 1,231 | 1.306255 | 26.5917% | 2.302193 |

One instrumented native pass per checkpoint uses 24 unsealed development prompts;
all 24 A8 response sequences match Q4_0 and each variant emits 2,834 tokens. CE uses
229 supported labels / 48 rounds / 24 prompts. Accepted drafts/round uses reported
native accepted-draft counters; acceptance rate divides by proposed drafts and
excludes the extra target token. Reported accepted versus actually emitted accepted
tokens are distinct at the output cap: reference 1106/1105, candidate 563/563 and
Q4_0 1608/1606. Both are preserved in the compact evidence. Native acceptance is
separate from speculative counters in the five-repeat request-timing workload.
The verifier is the frozen greedy target-sample-and-match path; no stochastic
sampling-distribution claim is made.

## Final complete-request throughput and latency

| Final arm | A8 request tokens/s | Paired Q4_0 request tokens/s | A8/Q4_0 | A8 decode tokens/s | Q4_0 decode tokens/s | Target-only request tokens/s |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 95.689 | 135.787 | 70.47% | 99.509 | 143.311 | 88.898 |
| Candidate | 69.623 | 134.728 | 51.68% | 71.873 | 143.113 | 88.468 |

Each final evaluation has five alternating measured repetitions, 24 prompts per
variant: 120 A8, 120 Q4_0 and 120 target-only requests. One warmup per server is
excluded (15 warmups total). Each variant returns exactly 14,290 token IDs with
105 length finishes /15 stop finishes; A8 and target-only match all 120 Q4_0
sequences. Fixed greedy seed 42, thinking disabled, max 128 outputs, draft max 5/
p-min 0, concurrency 1/context 2048 and frozen F16 KV/cache policy.

Request throughput is summed actual returned token IDs / summed full HTTP wall,
including prefill. Decode is the same numerator / summed server `predicted_ms`;
server `prompt_ms` is excluded. Evaluation/startup wall is separate overhead.
These are concurrency-one request rates; maximum online serving capacity is
unmeasured. TTFT is unavailable at the nonstreaming endpoint. Per-round draft/
verification/component profile was not added to this bounded comparison.

| Final arm | A8 median request s | A8 p95 s | A8 max s | Q4_0 median s | Q4_0 p95 s | Q4_0 max s |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 1.297 | 1.593 | 1.653 | 0.906 | 1.094 | 1.190 |
| Candidate | 1.827 | 2.042 | 2.142 | 0.912 | 1.108 | 1.212 |

These are pooled distributions across differing prompt lengths, not confidence
intervals. Complete min/median/p95/max request/decode values for all three variants
are in [final-report-metadata.json](final-report-metadata.json).

One CPU-only aggregation of the **existing two final timing manifests** recomputed
all five 24-request aggregates, verified token-ID parity and cross-checked original
pooled totals. No extra native requests. Independent arithmetic review passed 472
checks with zero failures; ratios use counts/time sums, not mean request/repetition
rates. All five paired request ratios remain below 1:

| Repetition | Reference request/Q4_0 | Reference decode/Q4_0 | Candidate request/Q4_0 | Candidate decode/Q4_0 |
|---|---:|---:|---:|---:|
| 1 | 0.695472 | 0.685988 | 0.519661 | 0.507172 |
| 2 | 0.708651 | 0.698368 | 0.516911 | 0.502851 |
| 3 | 0.706385 | 0.696506 | 0.510793 | 0.496717 |
| 4 | 0.708136 | 0.697396 | 0.517122 | 0.501405 |
| 5 | 0.705039 | 0.693656 | 0.519387 | 0.502955 |

Reference request ratio range 0.695472–0.708651; candidate 0.510793–0.519661.
Pooled ratios are 0.704702 and0.516767. Descriptive ranges carry no significance claim.
Reference request rate is 1.076393× target-only; candidate is 0.786987×.
[Per-repetition totals](final-timing-repetitions.json) and
[independent QA](final-timing-qa.json) preserve exact sums, ratios and source hashes.

## Exact versus sampled movement telemetry

| Arm | Exact committed cumulative sign flips | Diagnostic sample step / lag | Sampled cumulative flips | Sampled cumulative flip-backs | Sampled net disagreement | Near-zero fraction |
|---|---:|---:|---:|---:|---:|---:|
| Reference | 137,097,380 | 43,201 / 2 | 83,568,564 | 35,514,241 | 12,540,082 | 0.8535% |
| Candidate | 1,119,471,657 | 34,701 / 30 | 570,747,555 | 267,824,891 | 35,097,773 | 5.3715% |

| Arm | Mean latent magnitude | Minimum distance to zero | Scale L1 since initialization | Activation L1 | Midpoint L1 |
|---|---:|---:|---:|---:|---:|
| Reference | 0.494697 | 1.057e-08 | 854.692429 | 0.000000 | 0.000000 |
| Candidate | 0.119146 | 5.956e-10 | 4159.718524 | 0.080039 | 899.166947 |

Exact sign flips come from each committed optimizer update. Sampled history observes
intervals of 100 updates and can miss changes between observations; a flip-back means
return to the initial sign. Saved diagnostic `last_step`, observation count and
cumulative counters join the small telemetry fields. Candidate has 348 observations,
reference 433. Aggregate near-zero threshold is 0.01; per-layer diagnostic threshold
is 0.05. Small-family L1 norms cover those family parameters at the sample. These
statistics demonstrate movement, and do not attribute acceptance or useful decisions
to any particular sign change. Full scalar diagnostics and exact checkpoint/status
joins are in [final-endpoint-audit.json](final-endpoint-audit.json).

## Scheduled evidence and recovery lineage

All 18 original step-zero/scheduled/final reports are archived; final endpoints lead
this comparison. Equal-update checkpoints are intermediate observations:

| Capture | Development CE | Native accepted | Proposed | Rounds | Accepted/round |
|---|---:|---:|---:|---:|---:|
| Reference step 0 | 7.5459 | 309 | 12,354 | 2,525 | 0.122376 |
| Candidate step 0 | 7.5463 | 309 | 12,354 | 2,525 | 0.122376 |
| Reference step 5,000 | 5.1176 | 777 | 10,077 | 2,060 | 0.377184 |
| Candidate step 5,000 | 5.5689 | 507 | 11,371 | 2,327 | 0.217877 |
| Reference step 10,000 | 4.2628 | 946 | 9,238 | 1,890 | 0.500529 |
| Candidate step 10,000 | 6.0285 | 433 | 11,724 | 2,401 | 0.180342 |
| Reference step 15,000 | 3.9946 | 1,043 | 8,769 | 1,794 | 0.581382 |
| Candidate step 15,000 | 5.9516 | 491 | 11,457 | 2,344 | 0.209471 |
| Reference step 20,000 | 3.6999 | 1,078 | 8,590 | 1,757 | 0.613546 |
| Candidate step 20,000 | 5.7818 | 483 | 11,522 | 2,354 | 0.205183 |
| Reference step 25,000 | 3.5502 | 1,104 | 8,465 | 1,731 | 0.637782 |
| Candidate step 25,000 | 5.8161 | 532 | 11,257 | 2,305 | 0.230803 |
| Reference step 30,000 | 3.5495 | 1,089 | 8,547 | 1,746 | 0.623711 |
| Candidate step 30,000 | 5.5156 | 600 | 10,919 | 2,235 | 0.268456 |
| Reference step 35,000 | 3.5963 | 1,119 | 8,398 | 1,718 | 0.651339 |
| Candidate final | 5.5341 | 563 | 11,113 | 2,274 | 0.247581 |
| Reference step 40,000 | 3.6013 | 1,107 | 8,467 | 1,730 | 0.639884 |
| Reference final | 3.6961 | 1,106 | 8,463 | 1,732 | 0.638568 |

Two pretraining execution incidents were retained: an unbalanced TRAIN gate selection
was replaced with already-declared balanced authenticated TRAIN gate prompts; the
timing helper's invented ID-prefix check was repaired to accept exact frozen prepared
24 prompt bytes. Neither recaptured data nor changed math/precision/budget. Subsequent
reference SIGSEGV and charged retry remain preserved. Raw failures, step-zero,
positive-resume proofs and scheduled/final captures remain outside Git. Metadata
scope corrections and CPU script transport retries changed no experiment artifacts.

## Artifact identity and completion audit

Reference final:

- Checkpoint: `aebb8032c03f2d5263f608d67a65849b3239c85d6ccd372e09a7d9efa507579d`
- Checkpoint manifest: `a0f92778d094f352fb86e65275a99c328e592d27539319fb9b3c8d7675464b58`
- Report: `2cc27b3f0286f92cb40d642f2733179f5fd539dfc046ea7827a891e0e2296a4b`
- Result: `18c3c406fb0daa5d6217823eb25266de754a681bfb2d90b18a50060cd4b1f303`
- Timing manifest: `f45e31915137104d8441b9b03bbe7b23e034396a26d959b954066d6ab9043902`

Candidate final:

- Checkpoint: `bda021d3f09e2db6f9261a1d5d1b7a680aaa826b68874639696de2a4e2cfcd43`
- Checkpoint manifest: `04b23e0e808c473676853cba06db11e3d84acc4c20f28a592cf7372e5b721bfa`
- Report: `fcd0c0100d07d23aeff14f7cad7a715677b8852f1389bf16b4f7025120692c28`
- Result: `0a5ae30b2be6872f33b9efa4874e4622d3a34f30e6ccf6dd4b33e7bc3bc992f9`
- Timing manifest: `01fbc50b9fb73d5dd0391fd78cd0952c200c61d04a820009254174640a54454d`

Frozen prompt bytes: `131a3db7958ff6aa818b23019297654507d5b80bed3c298349417b7e3b2ba081`;
derived development manifest: `fc18f400a776161fd0ff40b30d58d69233be723e9e26b858a60160a2b802d091`.
Native server SHA: `1ca0c1d9ea62d15ec52e8f32bb50b8a7d31427ab009f4a00ce388f4cf9cef072`;
runtime inventory: `c472e36da5d441cae76a813f076cac13f5eff45f7d0e5ec03d75d9deaadc8ba0`.
Teacher/capture binary ancestry stays separately pinned to
`b5093749d67888bc2cafdb6a65c479f4c182f0a904820f1dae4870b6ae66d41c`.

All report/result/timing/checkpoint hashes and exact archive locators are in
[final-comparison-evidence.json](final-comparison-evidence.json). Raw runs are beneath
`/home/philip/binary-eagle-decoding/checkouts/a8-qat-run-583480c7/runs/qat-a8-comparison-20261003-01`;
frozen supervisor/evaluator cwd is the separate `a8-eval-recovery-3bd4837` checkout.
Model weights, datasets, captures and raw runs remain outside Git. Native split is
unsealed development; sealed finals were not accessed. No recapture, topup or
regression-driven experiment occurred.

Endpoint CPU audit loaded original checkpoints only with map_location CPU and
CUDA_VISIBLE_DEVICES empty; all joins passed, peak RSS 3.572 GB. Analysis utilities
[aggregate_existing_timing.py](aggregate_existing_timing.py) (SHA b7421232...) and
[audit_final_endpoints.py](audit_final_endpoints.py) (SHA c7a70733...) reproduce the
bounded metadata checks on the preserved raw artifacts.

Physical inventory at 16:03–16:04UTC verified all 34 final native server PIDs/groups
and all owned supervisor/trainer/evaluator groups absent, every final server stop/
process_group_gone flag true, runtime socket empty, no `/dev/dxg` holders and
RTX5080 idle baseline 2766 MiB/0%. See [final-physical-release.json](final-physical-release.json).
The first transport closed 16:05:50UTC; the bounded remaining three-report metadata
read changed no hashes or GPU state, and its transport closed 16:17:24UTC. No owned
experiment remains. Final publication, worktree cleanup and SAME heartbeat pause
are recorded in the [goal completion checkpoint](../../docs/goals/a8-qat-recovery-and-comparison.md).

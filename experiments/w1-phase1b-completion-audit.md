# Bounded GPU Phase 1B completion audit

The active thread objective is complete on RTX 5080/SM120: fixed native
quality/latency A/B, CUDA deployment checks, first bounded data shard and
the permitted 100-step real-model calibration. Q4_0 remains the primary
comparison. This is completion of the bounded task, not the broader research
claim that a binary drafter beats Q4_0 or that full-body training is eligible.

## Objective requirements and inspected evidence

| Requirement | Evidence inspected | Conclusion |
| --- | --- | --- |
| Fixed Q4_0/D/FP16/target-only pruning quality A/B | Native comparator SHA256 `891e8cfb979752cb49c77f5c3fa130142a2d40f568e94abfe5dc84ab079c5f63`; all four variants, 24 requests each, 96 exact pairs | Passed |
| Same-device latency A/B with order control | Two native comparators SHA256 `44368dc46ced468d42d23f77bd4bed7410e11aa290066b824d8c78730a8521df` and `e452374ceaa9c125310ff79eb9f1ec25424916e9f253455a2567c93499104488`; each 480 matched pairs, zero behavior mismatch | Passed |
| CUDA deployment contracts | Original SM120 run terminal exit zero with 112/112 operator cases; later combined runtime compiled and passed 133/133 + three fanout cases, integer assertions, real CUDA source-copy fixtures, row A1/A4/A8/A16 checks and exact native selector quality | Passed within reported cases |
| First bounded larger-data shard capture/audit | Manifest SHA256 `3ee7a8f4526f1dbcca1b6e0ea0756e42eff81333d7a88137afebe7213d3c1947`; independent audit `832325813eefea67dc97dc0d251b1e37b3a7b4a7349a4e26eb3aa9663198508b`; 31 train prompts, 12,610 labels, 12,250 supported, 7,796 native feature rows, source/emission audits retained | Passed; preparation eligibility unchanged |
| 100-step joint W1Ax calibration if gates permit | All five versioned gates passed under scoped report `103396b1f25ca127da8ab2b13326a3db085e1629dcdedb479bc100698fbf1dcd`; training report `ff6a31e923f1b6152111ed90db01e44e6f2793b2df43bcd912a6783b3e0c7273` has 100/100 steps, finite losses/gradients and all 18 active gradient tensors per step | Passed, row-A16 hard CE only |
| Preserve hashes/data/one GPU owner | Native/model/runtime/input/output hashes are recorded; raw source and checkpoints retained outside Git; root was sole GPU owner and all jobs supervised; sealed final data unused; no remaining project process after final audit | Preserved |
| Durable project state | STATUS, goal checkpoints and linked experiment reports are committed/pushed; published native gitlink and verified recovery bundles retained | Preserved |

Reports: [quality](eagle-prune-quality-5080.md), [timing](eagle-prune-timing-5080.md),
[first shard](w1ax-shard0000-capture-5080.md),
[calibration](w1ax-row-a16-calibration-5080.md),
[runtime engineering](eagle-runtime-optimizations.md).

## Additional authorized engineering

| Bounded item | Delivered evidence |
| --- | --- |
| Shared Q/K/V and FFN activation packing | Explicit pack nodes, three fanout graphs, A1/A4/A8 fixtures, real A4 quality and order-balanced timing |
| Direct mapped draft-vocabulary sampler | Guarded compact CPU sampling/fallback; six exact quality pairs and 60 timing pairs; Q4_0 decode 1.04028x on bounded workload |
| Bounded resident state | One-row backend state, mutation/position/sequence guards, original-placement scheduler-copy fix, 19 routing tests, real CUDA markers, exact quality and order-balanced timing; about 1% slower |
| Evidenced kernel cleanup | Default-off warp reduction, CUDA integer assertions over 133 cases/three fanouts, exact real A4 quality and bounded timing; no clear general gain |
| K/V-only catch-up | Strict eligibility/fallback, exact used-cache/following-state fixtures, exact real quality and 60 timing pairs; Q4_0 decode 1.00817x on bounded workload |
| Fine GPU measurement | Default-off intrusive event/node/transfer accounting, graph/direct/off fixtures, actual model process/draft traces, null captured timing, 15 request joins, zero reference/cap errors; no intrusive throughput claim |
| Versioned compact/label-only storage | V2 builder/converter/auditor tested and real 31-prompt conversion independently audited; avoids 7.66 GB duplicate logits; original raw data retained and eligibility false |
| Bounded student refresh preparation | Exact-prefix planner/auditor and native trace bridge, checkpoint/export/model/cache bindings, capture templates/provider preparation; 20 CPU tests and independent review. Execution authorization/loader/eligibility stay gated |

The runtime final timing audit matched 180 additional pairs and 120 graph
blocks; earlier compact/cache-only timing matched 120 pairs and 80 blocks.
No individual improvement is summed into an unmeasured combined gain.
All selectors remain opt-in; no default promotion or production deployment
was requested or performed.

## Research work outside this completed task

Full 2k capture, actual trained-student refresh collection, changed-prefix
teacher data/provider eligibility, substantive multi-width training, trained
native quality promotion and sealed final evaluation require their own
research decisions/budgets. The refresh bridge and label-only storage are
preparation tools, not evidence that these future steps executed. Original
bundle/full-body eligibility remains false, with unresolved gates retained.
The trained calibration checkpoint requires native export validation before
quality claims. No SM75 performance conclusion follows from these SM120 runs.

The project's joint binary EAGLE goal therefore continues at a user decision
boundary; the scoped Phase 1B thread objective has no required execution left.

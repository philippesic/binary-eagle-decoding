# Proposed next run: DSpark W1A8

This is the concrete run I recommend for review, not a launched or admitted run.
It combines full-pool data, calibrated fixed-A8 binary training, an explicit
author-informed objective, better scheduling and observability. It does not
claim an optimal recipe or a guaranteed Q4 win. Every implementation gap below
must be closed and checked before launch; existing EAGLE checkpoints stay frozen.

## Data and validation

Use all 10,000 original project TRAIN prompts, assigning each exactly one role:

| Role | Prompts | Prose / code / reasoning | Use |
|---|---:|---|---|
| Optimizer TRAIN | 9,856 | 3,286 / 3,285 / 3,285 | Gradient updates |
| Fusion calibration fit | 96 | 32 / 32 / 32 | Initial sign/scale fitting |
| Calibration validation | 48 | 16 / 16 / 16 | Calibration selection and fixed teacher-forced development checks |

The proposed selector has been computed and checked against all ten original
TRAIN-index SHA pins. SHA256:
`6de06976383246d4b512558fa45cf67780546ae11f121ed31f376e81ee1d7ea5`.
Ignored artifact: `results/dspark-next-run-plan-20261006/full-train-singleton-calibration-selector.proposed.v2.json`.
It is not a production capture manifest. All original prompts are retained;
none is silently deduplicated, length-filtered or promoted from held-out data.

Seed8101 selects calibration records from globally singleton source components
and content hashes. Original source grouping joins shared group **or topic**
transitively: there are9,969 components, including25 multi-record components
with56 prompts. All those multi-record components stay entirely in optimizer
TRAIN. Calibration therefore has no group/topic/content overlap with training.
Within each domain, fit samples16 from the lower length half,12 from the next40%
and4 from the upper10%; validation samples8/6/2. Source lengths above512 remain:
161 optimizer prompts,5 fit and1 validation. Actual native lengths are checked
before the final capture envelope is frozen.

Prefer replaying the existing F16 target's committed continuation token IDs after
proving complete generation ancestry, excluding rejected Q4 proposal branches.
Only the missing five-tap/full-distribution tensors are then recaptured. If that
ancestry cannot be established, disclose and freeze a fresh target-generated
corpus variant before training: maximum512 new tokens with natural EOG, original
prompt text, pinned Qwen3-4B F16 target/tokenizer/template, thinking off, greedy.
Do not silently mix replacement generations into an existing capture identity.
This reuses prompts/splits, not EAGLE's three-tap proposal-trace tensors.
Capture capacity4096 tokens preserves
the complete native prompt plus continuation; any over-cap case causes review
and capacity adjustment, never silent truncation or substitution. Training
retains full context. Native comparison remains context2048, independently.
The512 generation limit belongs to the fresh-generation variant. Existing
validated committed continuations retain their actual length; do not cut them
to512 to claim exact reuse. Size shards from their actual tensors, not an assumed
universal512-token response.

Target features are full F32 native layer inputs at2/10/18/26/34, width2560 each.
DSpark soft teachers retain all151,936 vocabulary values in F32 at every required
TRAIN loss-bearing position. No top-k/uniform-tail surrogate, teacher quantization,
borrowed head or stale changed-prefix label. Main forward still computes seven
noise slots, with chronological stride-seven anchors. Supervise all seven where
targets exist; mask only unavailable positions after actual EOG/length end.
Retain an observed EOS decision as a label, not repeated fabricated EOS targets.
This explicit partial-tail/short-response loss contract needs data/Markov-
predecessor/teacher-row-map validation code: current importer demands complete
horizons. Test full-block equivalence and short/EOG cases before claiming every
selected prompt can contribute. Never silently discard short-response prompts
or force generation past EOS. Eligible9856 versus actually loss-bearing/consumed
counts are separately reported. Domain
interleaving prevents first-shard-only training. Freeze balanced, whole-group
shard membership using metadata/byte bounds; shuffle shard IDs per epoch and
shuffle chains within each active shard with seed8101. This is locality-aware
shard/chain shuffling, not a claim of one uniform global prompt permutation.
Preserve chronology within each chain and save exact shard/prompt/anchor order.
Random-anchor sampling is accounted for below but not silently substituted.

The original unsealed development pool has1,002 prompts; it is separate from
the96/48 TRAIN-derived calibration roles. The frozen24-prompt native suite
(8/domain) remains the primary longitudinal comparator. Freeze an additional
96-prompt,32/domain, group-disjoint unsealed confirmation selector excluding
those24 and all training/calibration groups. Use it for initial/final quality
confirmation, not repeated recipe selection. Its actual IDs must be authenticated
before launch; no sealed-final bodies are used. Final sealed evaluation stays
reserved for the original final-nine protocol and its authority.

## Storage and source ownership

The full soft-teacher pool will not be materialized on the GPU host at once.
Implement an immutable global selector/logical cursor plus byte-bounded capture
shards. Initial limits:8GiB per raw shard, two active shards plus one publication
staging shard (24GiB total). Keep compact input token IDs, original generation
receipts, complete row maps, per-file hashes, source pins and exposure ledgers.
Training and teacher capture use the GPU serially under root's sole ownership.

Rotating caches must preserve the original actual capture identity. Recreate
evicted bytes only after deterministic replay admission and verify each recreated
tensor against its original SHA before consumption; retain original tokens rather
than silently generate a new continuation. If that exact replay contract fails,
use verified archival/restoration or obtain sufficient storage instead of dropping
prompts or relaxing teacher precision. This is a real prelaunch engineering gate,
not an existing feature. Per-shard filesystem admissions are refreshed after
restore; old inode/stat receipts are not relabeled.

A shard preparation boundary must checkpoint exact state and suspend the training
clock without resetting it. It is an ordinary resumable publication, not a new
human STOP/protected checkpoint for every shard. Producer/trainer process birth,
boot, run, source and ownership are bound; release the old GPU group before
another phase. Keep a global cursor/exposure identity across physical cache
movement. No concurrent target/student residency is assumed.
Measure a real small-shard capture/train/close/save/restore/replay cycle before
committing to this lifecycle. At512 outputs, logits alone are roughly296MiB per
complete chain; an8GiB shard holds only a few dozen ordinary chains. Hundreds
of transitions can cause substantial model reload/checkpoint traffic. Publish
measured phase costs, expected first-pass coverage, storage and total wall-time
projection with uncertainty. Twelve trainer-hours may take much longer in elapsed
time; excluding preparation from the clock does not make it free. Reuse validated
committed tokens and avoid gratuitous recapture where it is source-safe.

## Model, initialization and forward arithmetic

Use the pinned original DSpark snapshot3457dff1417cb84927f6098a5fcb7cee85c934b7.
Profile `ffn15_fusion`: one-bit weights/eight-bit activations in five layers'
gate/up/down projections and the five-tap fusion FC,16 selected matrices.
Train F32 latent signs and row-scale offsets. Preserve source latent magnitudes
with fitted signs/scales overlaid; no unit or0.1-inertia reset.

Private151,936-vocabulary embedding/head, attention, norms and Markov matrices/
scale remain original frozen values. The target/verifier/KV stay F16. Training
uses the admitted F32 reference for floating exceptions and F16 cache casts,
highest F32 matmul precision and TF32 off; export/native trajectories must admit
that reference-to-deployment difference. No autocast or low-precision moments.
Use the actual five-layer, seven-slot, anchor-first author layout, mask151669,
absolute positions, bidirectional noise/full context and declared target-token
Markov predecessors. Do not infer a different layout from the family name.
Frozen parameters do not imply detached student gradients: retain credit through
the private head/attention and evolving student noise states into the trainable
projections. Only teacher targets are detached. Historical original DSpark Q4
uses FFN15 with floating fusion; this FC16 candidate is a deployment comparison,
not an isolated activation-precision attribution or a redefined Q4 control.

Fit fusion on32 representative context rows per calibration-fit prompt
(3,072 rows), score the48 validation prompts with the actual following RMSNorm
and raw residual amplitude as well as SSE/cosine. Compare original row-mean
scales, converged scale-only and bounded discrete-fit candidates (up to8 flips/
row, explicit CPU wall/memory bounds). Preserve original gamma/F32 epsilon bits.
Set fit `rows_per_chain=32`, `max_total_rows=3072`, array planning cap2GiB and
CPU wall cap1800s; measure owner RSS separately rather than treating an array
estimate as its proof. The initial96/48 calibration-development population is
not enough evidence to claim universal fusion improvement.
Select one whole initializer; no validation-selected per-row mixing portrayed as
independent confirmation. Native zero-update trajectories remain the quality gate.
Orientation reversal has prior negative postnorm evidence and stays off by
default; it is a separately labeled calibration probe, not an assumed rescue.

## Objective and optimizer

Proposed main loss, over supported positions only:

`L = sum_i w_i [0.1 CE_i + 0.9 sum_vocab |p_draft_i - p_target_i|] / sum_i w_i`

`w_i = exp(-i/4)`, for slots0–6. This preserves the full horizon and explicitly
normalizes weighted valid labels. It is author-informed restricted binary QAT,
not a reproduction of full-precision confidence/Markov training. The current
`CE + 9L1` option is not identical: exact0.1/0.9 normalization needs new code
and gradient/clipping checks. Report CE and L1 separately and do not equate L1
overlap formulas with this native greedy verifier's acceptance.
Teacher/student objective softmax temperature is1; target generation temperature
is0. These are different settings. Do not train the immutable target accidentally
through shared module aliases.

| Setting | Proposed value |
|---|---|
| Optimizer | AdamW, F32 masters and both moments |
| Betas / epsilon / weight decay |0.9 /0.999;1e-8;0 |
| Peak latent-sign / scale LR |1e-3 /1e-5 |
| Warmup | First2% of cumulative trainer time:864s/14.4min |
| Decay | Time-based cosine after warmup to10% of peak:1e-4/1e-6 at12h |
| Gradient clipping | Global norm1, record pre-clip norm and clipping fraction |
| Projection after update | Latents[-1,1]; effective row scales nonnegative |
| Effective optimizer batch |2 seven-slot blocks from distinct chains/groups, prefer different domains;14 supported labels when complete |
| Microbatch | Start1 with accumulation2; vectorized2 only after equivalent-effective-batch gradient/state/memory/rate checks |
| RNG | Explicit Python/NumPy/Torch CPU/CUDA seed8101; all states in resume |

Warmup/decay/batching are proposed choices, not proven optimal settings. They need
source implementation and an exact-resume contract; warmup never restarts at a
shard/evaluation boundary. Accumulation normalizes over the total weighted valid
labels, clips/steps once per effective batch, persists partial state or checkpoints
only at completed optimizer transactions. Throughput comparisons use the same
effective batch so execution optimization is not confused with a recipe change.
Reserve both logical blocks before starting accumulation; never silently take
adjacent correlated blocks from one chain and call them independent. A leftover
block at a shard/epoch boundary is carried by its exact descriptor/capture identity
until a distinct-group partner is available; checkpoint completed transactions
rather than lose half a batch. RNG and cursor preserve this pairing.

## Duration, telemetry and checkpoints

Budget43,200 cumulative trainer-seconds, no smaller step/token/epoch cutoff.
Continue through12h while numerical/resource/correctness gates remain healthy,
even if early acceptance is poor. Do not promise a full epoch or an optimum from
elapsed time. Bulk teacher capture, reconstruction, startup and standalone native
evaluation are separate preparation/evaluation costs. Active-loop data transfer,
optimization and normal periodic checkpoint overhead are charged; overshoot from
a completed update is recorded honestly, never retroactively erased.

Write compact persistent minute/window history, not just overwrite status. Record
optimizer steps, unique prompts/groups/anchors/teacher conditions, repeated
presentations, per-domain/category/length/depth exposure, CE/L1/weighted total,
teacher-forced top1 by slot, predicted-prefix survival, pre-clip norm/clipping
fraction, per-layer signs/flips/flip-back sample/net changes, near-zero/bound
occupancy, scales/collapse, A8 saturation, actual rates, GPU/host memory and phase.
All coverage definitions and denominators are named. Diagnostics have a bounded
sample/cadence; profile their cost without removing finite gates.

Health at5 and15 minutes after optimization begins, then every2h of operational
wall time, including truthful preparation/evaluation phase reports. Notify on
meaningful progress/failure/required input. Check exact process birth/boot/source,
finite losses/gradients, active update rate, checkpoint freshness, memory/disk and
remaining allocation. No status-only loop described as training.

Periodic recovery checkpoint every15 trainer-minutes plus every shard boundary,
incident and4/8/12h native milestone. Implement time cadence explicitly; do not
inherit EAGLE's250-step multi-GB write cadence. Keep latest3 ordinary generations;
protect initial and4/8/12h/endpoint/incident anchors. Initial proposed per-lane
cap10 payloads/56GiB includes the next writer and failed staging; actual bytes
must pass admission and may require a reviewed larger cap. Whole-run exports,
logs and captures have separate budgets. Keep at least10GiB disk and4GiB host
available; initial GPU policy ≤12GiB reserved and≥1GiB free, verified against
actual full optimizer moments/backward/save/resume peaks.

## Execution optimization coverage

| Item previously discussed | Disposition for this run |
|---|---|
| Grouped GQA without repeated K/V | Already core block implementation; preserve grouping/masks/F32 softmax/F16 casts, measure actual CUDA benefit |
| Fused F32 AdamW | Implement production packet selection; enable only after same-gradient moments/parameters/step/resume/quality and complete-step rate checks; serial fallback |
| Batching and accumulation | Effective batch2 as above; vectorization is gated execution of that same batch, not an undeclared larger optimizer batch |
| Immutable raw teacher input/upload reuse | Profile and add only source/hash/lifetime-safe reuse; never reuse student FC/K/V/cache across changed optimizer weights |
| Shared activation packing/gate-up reuse | Preserve current valid hard-forward sharing; new fusion only if actual graph/gradient and step profiling support it |
| Sign-snapshot clone removal/scalar transfer consolidation | Audit and adopt only exact bookkeeping changes preserving I64 counters/pre-update finite checks |
| Status/fsync and recipe-audit write cadence | Add block-specific bounded history/publication; direct-EAGLE flag is not a DSpark implementation |
| Provider/startup integrity audit reuse | Reuse only complete source/actor/data/mask/backend-bound receipts; fresh actor admission never bypassed |
| Projected context/chunked KV | Only within unchanged input/model lifetime and correct gradient graph; EAGLE one-layer catch-up does not automatically transfer to five layers |
| Pinned prefetch | Off until uploads demonstrably stall; bounded DMA ownership and no extra whole shard residency |
| Static confidence suppression/logit concat assembly | Separate native profile-gated paths; p_min0 fixed, apply eligible math-preserving runtime changes to applicable Q4 controls too |
| Adaptive draft length/confidence scheduler | Off; fixed seven computed rows, no trained confidence head or overlap-to-greedy calibration claim |
| Native head Q4/floating-format dispatch | Off in this candidate; changes precision coverage and original teacher/control comparability |
| Fused gate/up native kernel, SM75 binary MMA, shared pack/warp/resident state | New kernel work deferred absent actual bottleneck; do not claim2080performance or transplant small/negative EAGLE wins |
| Blanket compile/autocast/checkpointing/low-precision moments/optimizer-in-backward | Off; only revisit an observed bottleneck with exact gradient/clip/cache contracts |

## Learning options accounted for, not stacked blindly

Execution-option promotion is predeclared, not decided by a faster kernel alone:
exact source/slot/mask/prefix/quantized-code ownership, counters/RNG and restore
contracts; FP32 reference loss atol1e-6/rtol1e-5, VJP/gradient atol1e-6/rtol1e-4
with aggregate relative L2 error≤1e-4, optimizer parameter/moment atol1e-7/rtol1e-5;
no native decision regression on the fixed admission cases after matched test
updates. These are proposed FP32 execution checks, not universal bit equality
or replacement tolerances for native F16/model portability. Freeze them before
performance inspection. Reject an optimized option if they fail; don't weaken
them to obtain a speed result. Serial accumulation remains the usable fallback.
Measure three repeated warm complete-step workloads at the same effective batch,
data and restored initialization. Require≥5% median complete-step improvement
outside observed timing noise, with full memory/save/restore checks. Record the
actual benefit;5% is a proposed adoption threshold, not a promised speedup.
Preflight/benchmark updates are logged separately and restored away; the main
run still receives its full43,200 trainer-seconds from its admitted initializer.

| Prior idea | Disposition/reason |
|---|---|
| Learned A8 clipping; affine fusion/all-layer midpoints; latent±0.1; weight-unit/fan-in-normalized gradients; SGD | Off main. Current DSpark adapters do not expose them; combined EAGLE learned/affine/low-inertia arm lost. Individual gains unproved |
| Fusion bias/rank1/rank4 residual; fusion affine center | Off main: extra representation/floating work and export contract; separately test only a localized defect/speed headroom |
| Local norm gains; attached1–4 accepted-context tail | EAGLE-first probes, not silently copied to block model. Existing recurrent credit is not generically missing |
| Orientation rescue; energy-weighted/raw/postnorm/gradient-guided fusion fit; another-layer discrete fit | Raw/postnorm scoring included; orientation/alternative objectives/another layer gated or off, with whole-initializer confirmation |
| Randomized anchors | Explicitly off main; requires dense eligible teachers, RNG/cursor/mask/cache admission and a controlled sampling comparison |
| Live student-prefix refresh; old/fresh whole-round replay | Off initial main; diagnose stale-prefix disagreement first. Exact target query/prefix identities and costs required; no spliced stale teachers |
| Soft CE/KL/dense-drafter auxiliary features or Q4 auxiliary distillation | Off main; distinct objectives, not interchangeable with selected full-L1. Q4 drafter never replaces authoritative F16 target labels |
| Confidence objective/Markov unfreezing/private-head learning | Off in restricted-QAT scope; author training has these routes, but this candidate freezes them explicitly |
| EWGS, late dampening, Bop, first-mismatch/rank-preservation, broad survival objectives | Off absent a diagnosed optimization failure. Sign movement alone is not quality; no rejected-update accounting loophole |
| SFDD/teacher-flatness, AdaSPEC regret filtering, VAT weighting | No corpus filtering in this run. Separate prefix-valid controls if later justified; preserve original exposure and count masks honestly |
| Feature-noise augmentation/ReCU | Not supported by a measured missing error distribution/dead-zone diagnosis; off |
| A8→A1 warm-start; learned A1 thresholds; A4 ladders/Q1/ternary | Not part of DSpark A8. Later A1 stays the requested direct-A1 route unless separately changed, with all stages charged |
| Head factorization/shortlists, Hydra/Medusa, smaller backbones, FFN structural pruning, broad mixed precision, blind rotations/cold SVD/failed two-plane init | Separate/deferred architecture work; no scope change disguised as this optimization package |

This inventory comes from the October4 research plan/eight-report synthesis,
implemented profile declarations and the October6 Astra audit. None is omitted
merely because it was once discussed; none is enabled merely because it exists
in an EAGLE-specific JSON or a CPU prototype.

## Native validation and launch gate

Before paid training: freeze actual implemented config/source/data/initializer/
backend and all tensor precision/trainability manifests; run the real small-shard
capture/import/fusion/export/native trajectory/full backward/moment-memory/save/
restore checks. Verify masks, tap semantics, exact prefix joins, slot mapping,
packing/scales, vocabulary, teacher immutability and current CUDA dispatch. Timed
and shard boundaries must reproduce optimizer/RNG/cursor/scheduler/clock state.
No tiny synthetic pilot is the production dataset, and no build proof is model
admission. Calibrated zero-update DSpark is the improvement baseline.

At0/4/8/12 trainer-hours, serialize export and native comparison against calibrated
zero-update DSpark, immutable Q4 EAGLE and target-only. Same5080, F16 target/KV,
original24 prompts,5 repetitions,2 warmups/cell, context2048,batch/ubatch32,
greedy seed42,thinkingoff,max128 outputs, DSpark draft7/p_min0 and Q4 EAGLEdraft5.
Report accepted/proposed, accepted per round, slot-conditioned/early-prefix
survival, full request tokens/time and per-domain/prompt distributions, output
lengths and complete raw token/finish parity. Keep timing instrumentation separate
from diagnostic profiling; same-device Q4 is always the primary denominator.

Between milestones, fixed48 calibration-development prompts/four complete
anchor positions where available provide cheap teacher-forced CE/L1/top1 checks,
explicitly not native acceptance or independent confirmation, because this set
also selected the initializer. Initial/final96 supplemental unsealed prompts
give a separate confirmation view. The original DSpark Q4 exact-file comparison
is added when its authentic bytes are available; don't re-quantize new bytes and
call them the frozen control.

The existing EAGLE/Q4 target-only parity failure remains FAILED. It cannot silently
authorize a changed gate. Before launch, explicitly reconcile the new runtime's
baseline behavior and controller response so a known numerical issue is not
hidden as PASS or rediscovered at4h. Unexplained mask/cache/decision failures pause
with exact state preserved. Any research-only continuation policy needs an
explicit reviewed distinction from deployment correctness; the current fail-closed
receipt is not already such a policy. Poor acceptance alone is not a reason to
shorten a healthy12h allocation. Maximum two exact-state repairs per incident;
preserve raw failures and show any unresolved launch blocker.

## What the EAGLE gap says

Current trained EAGLE10.3306% versus Q426.4939% is a16.16-point gap, requiring
about2.56× current acceptance to match Q4, plus recovery from0.64536× Q4 request
throughput. No omitted knob has demonstrated closing it. Historical body/head
factorial evidence implicated body/fusion/attention damage more strongly than
head-only damage; it is a lead for the current checkpoint, not its fresh diagnosis.

The matched2h fixed-A8 reference reached0.638568 accepted/round, while combined
learned-A8/all-nine-affine/low-inertia reached0.247581. That package therefore
was not a known winner accidentally omitted; its components were not isolated.
Earlier head-only KL improved validation KL while BF16 PyTorch/AngelSlim
held-out acceptance worsened; that was not a native llama.cpp measurement.
More consumed data, better LR/clipping behavior, depth weighting and current-
prefix alignment are plausible opportunities. Current14.43h has no intermediate
native curve, and an older separately initialized fixed arm plateaued; joining
those runs as one curve or promising that another12h closes the gap is unjustified.
Preserve the existing checkpoint and next-model ordering; any additional EAGLE
diagnostic/training is a separately selected task, not a hidden part of DSpark.

## Implementation status

Already present: original data/source metadata, proposed full-pool role selector,
fixed-A8 block math/grouped GQA, indexed native capture, fusion fitter, serial
AdamW/fused probe, bounded one-chain mappings, protected checkpoint machinery,
4/8/12h evaluation/resume source and audited native primitives.

Required before this proposed run: full-pool canonical-group validation/selector
support, exact partial-tail/short-response target masks, rotating immutable
shard/provider/clock/replay lifecycle, exact normalized
loss, time LR scheduler, effective-batch2/accumulation/seeding contracts, block
history/time-checkpoint cadence, production fused-backend selection/gating,
actual storage/teacher/calibration/model/native/full-moment admission, supplemental
unsealed selector and explicit treatment of the existing parity/controller gate.
This plan does not represent those as toggles already switched on.

# Additional research for the nine model campaign

October 4, 2026. Eight agents completed the requested research: four started
from primary literature and four derived ideas from project evidence before
intermittent primary-source validation. Opus 5.5 audited all eight complete
reports in four pairs, then audited this reconciliation. No model changes,
training, GPU/Metal/SSH, new goal or compute allocation occurred. Agents wrote
only their uniquely assigned reports after the read-only research phase.

Recommend a small engineering package, two refinements inside the existing
fusion screen, and a few profile/diagnostic-gated training probes. The output
remains nine models. Extra findings share existing fusion, refresh, optimizer,
data and profiling allowances; they do not create one long run per idea.

## Evidence and report index

| Report | Method | Scope |
| --- | --- | --- |
| [01](01-literature-quantizers.md) | Literature first, Sol high | EWGS adaptation, oscillation controls, ReCU rejection |
| [02](02-literature-training-data.md) | Literature first, Sol high | SFDD, AdaSPEC, rejection-anchored VAT |
| [03](03-literature-train-systems.md) | Literature first, Sol high | Snapshots, input upload, grouped attention, fused optimizer |
| [04](04-literature-inference.md) | Literature first, Sol high | Static confidence, logit assembly, SM75 floating dispatch |
| [05](05-reason-root-acceptance.md) | Reason first, Astra medium | Fusion orientation, post-norm diagnostics, norm gains |
| [06](06-reason-trajectory.md) | Reason first, Astra medium | Attached context tail, replay, loss normalization |
| [07](07-reason-latency.md) | Reason first, Astra medium | Telemetry, durable publication, optimizer, output assembly |
| [08](08-reason-cost-quality.md) | Reason first, Astra medium | Row rescue, example weighting and capacity limits |

Each reasoning-first agent sent an initial source-derived hypothesis before
its first web validation. [Report manifest](report-manifest.json) records file
SHA256, source revision and word counts. Original reports are retained; the
qualifications below govern adoption when an agent or reviewer overstates a
premise. Paper results identify mechanisms, not local quality or speed gains.

A bounded read of existing fitter metadata confirms all 383 scale-zero control
rows have positive continuous derivative at zero, hence negative fitting
correlation. The 2,560 solver records are from the existing authenticated real
A8 fit, not a new experiment. The source SHA matches the committed original
report pin. See [census](zero-scale-census.json). It establishes an alternating
solver trap, not improved native acceptance or an A1 result.

## Engineering package and execution probes

| Item | Decision | Applicability and proof |
| --- | --- | --- |
| Delete intermediate FP32 sign-snapshot clone | Incorporate as the first small engineering change when implementation starts | Existing Boolean comparison owns its result; retain scale snapshots, exact counts and finite checks. Rediscovered previously CPU-prototyped work, not a novel algorithm. Training only; frozen Q4 training control is not applicable. |
| Consolidate diagnostic scalar transfers | Optional part of the same edit | Keep loss/gradient checks before updates and exact I64 counters. Profile fresh reduction/transfer work; do not assume either a huge synchronization bottleneck or zero cost because a previous gate drained the queue. |
| Reuse immutable captured-feature upload | Incorporate only for compatible paired lanes | Hoist one upload for genuinely identical batches on the same device; keep student outputs/caches independent. No benefit for separate sequential A8-to-A1 stages. Cursor/RNG/STOP and input mutation proof required. |
| Live status publication cadence | Profile-gated | Measure actual status/fsync wall cost first; retain immediate admission/failure/stop/checkpoint records and independent checkpoint/budget durability. Bounded time cadence must fit watchdog tolerance. Recipe-audit writes are probe-gated, not every update. |
| Fused FP32 AdamW backend | Bounded execution probe before long jobs | Keep FP32 masters/moments and the optimizer formula. Fix CPU-versus-device step-state initialization and resume, record backend identity, check moments/projections/hard sign choices. No assumed percentage saving or exact-resume equivalence to historical unfused runs. |
| Grouped F32 attention without repeated K/V | Probe if memory or attention time is material | Preserve group mapping, F32 scale/softmax, masks, attached K/V and F16 cache casts. Compare VJPs and later gradients; no automatic SDPA/backend/precision substitution. Training-only tensor-size savings are not an SM75 inference result. |
| Suppress unused DSpark confidence at p_min zero | Add to low-cost native engineering shortlist after binary admission | Explicit confidence-output request and graph compatibility key; preserve positive-p_min and diagnostic paths. Prove proposal/cache behavior, confirm node/copy removal and time fairly on DSpark Q4 and admitted binaries. Savings are unmeasured and may be small. |
| Balanced DSpark concat tree / identity layout skip | Low-priority profile-gated probe | Preserve serial Markov predecessor, slot and multiblock layout. Seven-column growing concat writes 27 column-equivalents versus a balanced tree's 20, excluding final layout work. This arithmetic is not latency evidence. Defer aliased preallocation. |
| Bounded pinned round prefetch | Defer unless actual upload/provider stalls matter | Training only, with host-memory and DMA/cursor lifetime proofs; no counterpart in the three frozen Q4 baselines. Do not retain another whole capture shard. |
| BF16-to-F16 floating exception dispatch on SM75 | Conditional diagnostic outside frozen controls | Source values must be audited, and activation/reduction paths still change. Preserve target and immutable teacher. Test matching Q4 diagnostics if applicable; prefer the existing head-Q4 experiment when bytes dominate. No global cuBLAS precision override. |

Snapshot arithmetic is cumulative traffic/allocation, not simultaneous peak:
218,234,880 F32 signs imply an avoided 832.5 MiB clone payload per lane/update;
the largest individual clone is 312.5 MiB. Actual peak and timing require a
measurement. Grouped attention avoids explicit repeated K/V tensors, but the
reported 64 MiB per depth is not a guaranteed peak-memory reduction. Fused
AdamW retains the two FP32 moments; it does not reduce their persistent storage.

## Calibration and learning refinements

| Item | Decision | Minimal useful test |
| --- | --- | --- |
| Zero-scale whole-row orientation reversal | First new fusion initialization probe | For negative TRAIN correlation, reverse the row and refit a nonnegative scale; accept only improvement in the exported F32 objective. Record global reversals separately from local bit-move caps. Compare the whole candidate on independent prompts and native prefix survival. |
| Post-norm plus raw fusion scoring | Incorporate diagnostics in existing fusion screen | Score initializer, scale-only and fitted candidates at the actual following normalization and retain raw residual-amplitude error. Current tiny cosine difference alone does not justify a new objective. |
| Per-example energy-weighted fusion SSE | Conditional single objective control | If the diagnostics support it, vary fixed example weights using TRAIN-only floors/caps; hold initialization/move allowance fixed. It is a relative-error proxy, not the coupled post-RMSNorm Jacobian objective. |
| One existing drafter norm gain | Conditional EAGLE A8 probe | Localize channel/outlier error first. Clone lane-owned parameters and extend optimizer/checkpoint/export/hash contracts. No added inference operation; no automatic A1 rescue claim. |
| Attach a short accepted-context K/V tail | One EAGLE A8 probe if added gradient is material | Preserve forward/casts/cache and quantizer invocation normalization while attaching a predeclared 1–4 context rows. Verify new context-attention credit, memory/rate and native survival. Existing fusion/seed/proposal gradients already exist; this is extra credit assignment, not a missing generic gradient fix. |
| Mix old and newly recaptured whole rounds | Incorporate replay/source accounting; probe the mixture within existing refresh | Separate actor/prefix identities, deduplicate unique exposure and preserve complete chains. Existing recapture is shared; do not create a second capture job merely for replace-versus-pool. |
| Loss denominator and Markov predecessor declarations | Incorporate into batch/trainer admission | State round-mean versus token-mean, weighted masks, accumulation denominators and whether Markov predecessors are captured, teacher-forced or student-generated. No silent recipe switch. |
| Update-resolution reversal sampling | Add bounded diagnostics if chatter is suspected | A fixed small coordinate sample can distinguish actual reversals from return-to-initial-sign counters sampled every 100 updates. Its cost is bounded but not zero; sampling cannot certify all parameters. |
| Round-aggregated EWGS sign backward | Lower-priority gradient probe after effect measurement | Name residual coordinates, preserve shared-sign aggregation, measure perturbation under Adam before allocating an arm. Original CNN efficacy does not establish native drafter gains. |
| Late oscillation dampening | Defer until actual harmful chatter is observed | Matched late continuation only; no early default, new full-weight EMA state or gradient-zero-only freezing under Adam. Less chatter is not itself better acceptance. |

The orientation probe is performed before QAT initialization, with fresh
optimizer state. No moment conversion is needed at that point. Do not silently
perform it during a resumed run. A future mid-training intervention would need
a separate latent/moment/scale/auxiliary-state transition contract; merely
negating one Adam moment is not proof that the altered learning recipe is valid.

Current QAT scales have a trainable additive offset and derivative one at exact
zero. Thus zero sign gradients do not prove permanently dead channels. The
saved least-squares derivative describes the calibration objective, not the
QAT CE gradient. A small CE-gradient diagnostic can determine whether the same
rows tend to revive; neither possibility removes the need to test initialization.

Do not select per-row fallback signs on a validation set and then call its
error unbiased held-out evidence. Select on TRAIN or a named selection split,
then use untouched prompt-disjoint confirmation and native evaluation. The
383-row census is specific to A8; A1 needs its own operands and validation.

Positive constant output-row weights do not change a truly independent row's
LS minimizer. Shared move budgets, fixed stopping tolerances or a coupled
objective can change allocation/termination, but that is another mechanism.
Per-example weights and exact normalized-output optimization must not be
misrepresented as the same objective. EAGLE's raw residual and normalized
attention inputs need both magnitude and direction.

For A1, strictly positive multiplicative channel rescaling directly before a
zero-threshold sign boundary preserves sign bits, though the shared meanabs
amplitude changes. This does not assume all original norm weights are positive:
the rescaling must preserve their signs. An unconstrained norm update crossing
zero, or interaction with learned thresholds, is a different case. This limits
a blanket norm-only remedy while leaving an A8 capacity probe plausible.

For the proposed EWGS residual (z minus sign(z))/2, magnitude is at most 0.5:
delta 0.001 changes the multiplier by at most 0.05%. Without the explicit half
normalization, the bound is 0.1%. Neither is a mathematical no-op; Adam can
reduce its practical effect. Do not blindly raise delta to 0.1–1 based on a
reviewer's suggestion; the source paper itself reports collapse at one large
factor in a different setting. Gradient history, memory and native quality
need evidence before promotion.

## Data filtering and rejection weighting

SFDD teacher-flatness selection is a conditional data-order/selection control
inside the existing data allowance, not a default reduction of the long corpus.
It requires exact full-vocabulary probabilities or a validated approximation;
uniform-tail top-k cannot recover sum of squared probabilities. The reported
greedy EAGLE2 speedup drops from 2.80 full-data to 2.62 at 50% retention and
2.70 at 70% retention. These show a compute/quality tradeoff, not a binary-QAT
gain. Near-tie examples may be noisy rather than most useful for binary roots.
Stratify any ordering/retention experiment and compare with random ordering at
common exposure and common total charged compute. Do not assume all historical
full logits remain available or cost nothing to stream. [Report 02](02-literature-training-data.md).

AdaSPEC CE-regret filtering is lower priority until a well-covered matched
reference exists. The method prioritizes large student-minus-reference loss
gaps, not simply easy examples. Do not start another long reference solely
for the filter. Fixed masks can encode the reference's limited early exposure;
loss masking saves no recurrent forward work and must preserve earlier states
needed by later supported losses. Report presented versus loss-bearing tokens.

VAT rejection-anchored weighting is a possible second control within the
existing depth-weight experiment. The paper explicitly includes greedy training;
ratio-overlap criticism alone does not reject it. But current forced captured
tokens are not automatically the student's actual trajectory. Known-prefix
validity requires equality of current/captured previous tokens and the correct
teacher-label join. Censor when the paths diverge or labels become unavailable;
never treat an unknown later rejection as observed. A bounded TRAIN ancestry
audit can establish whether a censored weighting proxy is useful. The complete
survival classifier and soft-label recipe stay deferred until the actual
current-prefix capture path is admitted. Weak-root models may make adaptive
weighting nearly identical to fixed decay; count eligible cases before opening
another training arm.

Reject generic feature-noise augmentation for now: the original motivation
was an older feature-prediction regime, while hard QAT and current-state
recurrence already introduce the relevant local distortions. A new noise
model would require a measured missing error distribution. ReCU, broad compile
/checkpoint changes, low-precision moments, aliased logit output assembly and
new head/backbone/formats remain deferred under the existing evidence.

## Peer audit reconciliation

Five successful peer-review MCP calls used claude-opus-5-5-high. Calls 1–4
covered reports (01,05), (02,06), (03,07), (04,08); call 5 audited root decisions
after source checks and the existing metadata census. Raw texts stay outside
Git; [peer manifest](peer-manifest.json) records model, pair coverage and SHA256.
No Fable substitution or unavailable-call scientific feedback is claimed.

Accepted corrections include guarded recipe-audit publication, paired-lane
applicability, optimizer step-state placement, finite/I64 metric safety, actual
scale revival semantics, the limits of cheap post-norm proxies, explicit loss
and Markov contracts, and training-only versus runtime Q4 applicability.

Qualified or rejected reviewer claims:

- No unmeasured 1%, 5–10%, microsecond or bandwidth estimate is a local gain.
  Profile the stage and complete request/step; graph envelopes include idle.
- A prior finite gate draining CUDA does not prove that later newly launched
  metric reductions/transfers cost nothing. Their marginal cost is unmeasured.
- Exact sign-clone removal does not need another quality-study arm, but retain
  focused mutation/diagnostic/finite-order proof from the existing prototype.
- Scale-zero sign gradients are not proof of permanent QAT death; source gives
  scales nonzero surrogate derivative at zero. Least-squares signs do not
  determine the CE-gradient sign.
- Whole-row rescue here initializes a fresh QAT run. A review's mandatory
  ongoing-optimizer moment transformation does not apply to this design.
- A review's 0.1% EWGS correction assumes an unnormalized residual; the actual
  report explicitly halves it and has the 0.05% bound stated above.
- Validation-based row selection is selection, not independent confirmation.
- Loss means and supported lengths matter with accumulation, clipping, moments
  and objective masks. Adam is not a proof of complete invariance; audit
  induced weights rather than presuming negligible variation.
- Attached-tail gradient size/cosine is a diagnostic, not an arbitrary 1% or
  0.99 automatic quality gate; small gradients can still matter.
- Prefetch/upload/optimizer/telemetry are training-only. The frozen Q4 models
  have no training counterpart; applicable native graph changes do get Q4
  development and final comparisons.

## Practical order

1. Preserve binary deployment/capture, balanced coverage, family-specific QAT
   and the existing high-priority A1/fusion experiments. New findings do not
   delay those engineering prerequisites indefinitely.
2. Prepare the one-line snapshot change, explicit loss/replay/Markov contracts
   and root/fusion diagnostics. Record previously prototyped code rather than
   claiming its rediscovery is a new measured optimization.
3. Put orientation rescue and post-norm/raw scoring into the existing fusion
   screen; combine only after singleton controls.
4. Use one short attribution pass for optimizer, status, transfers and memory.
   Decide fused AdamW, cadence, grouped attention and static confidence from
   meaningful measurements; preserve correctness before any launch.
5. Allocate at most a few additional learning probes inside the already
   reserved allowance: short context tail, localized A8 norm gain, or one
   filtered/reweighted data control when diagnostics justify them.
6. Confirm the best compatible combination against the strongest singleton
   and the unchanged binary control, then run long balanced QAT. Nine final
   artifacts remain three frozen Q4 controls and six selected binary candidates.

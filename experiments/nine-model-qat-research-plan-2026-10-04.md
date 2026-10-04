# Research selection for nine Q4 W1A8 and W1A1 drafters

October 4, 2026. Recommendation for the human's proposed EAGLE, DSpark and
DFlash campaign. This is a planning report, not a new active goal, GPU launch,
training budget allocation or successful model result. Existing completed goals
and sealed evaluation data are preserved.

The intended deliverables are three frozen Q4 controls and six trained binary
candidates. Small development experiments can produce additional temporary
checkpoints without expanding the final nine-model comparison.

Recommend calibrated fusion initialization, serious balanced data exposure,
architecture-specific hard-forward QAT, direct-A1 versus A8-to-A1, learned A1
thresholds, depth weighting and one own-prefix refresh probe. Add optimizer
changes and small fusion corrections only when their isolated development
evidence supports them. Integrate the strongest compatible recipe for each candidate only after its development test;
do not put every untested idea into one long training run.

## Evidence that determines the priorities

The October 4 fixed-A8 EAGLE reference completed 43,203 updates and 206,163
supervised tokens in 7,200 trainer-accounted seconds. Native acceptance improved
from 0.122 to 0.639 drafts per round, but request throughput remained 95.7 tokens/s
versus its Q4 control's 135.8. The combined learned-A8, all-nine-midpoint and
lower-inertia arm completed 34,731 updates and 165,733 tokens, with acceptance
0.248 and throughput 69.6. These factors were not isolated. Increased sign flips
are not evidence of useful changes. See [final comparison](a8-qat-recovery/comparison.md).

The reference reached 0.638 at step 25,000, peaked at 0.651 at step 35,000,
and finished at 0.639. Its existing development curve was already fairly flat
late in the bounded run. More hours alone are not demonstrated to fix the gap;
more balanced exposure, initialization and trajectory alignment must be measured.

Those arms consumed 5.29% and 4.25% of the available 3,899,930 supervised rows.
An arithmetic extrapolation of their recorded average rates gives about 37.8
and 47.1 accounted hours per corpus pass, respectively. This is a conditional
projection, not a forecast: prefix length, domain order, batching, resource
behavior and the reference's charged crash/downtime affect it. It does not
estimate DSpark or DFlash training time.

The real fusion fitter reduced validation reconstruction error by 26.98% versus
converged scales on four independent TRAIN prompts, but some direction/ranking
measures still trail the initializer. Validation has no code prompts. Actual
native proposal quality remains unmeasured. The fit used fixed A8 arithmetic;
its results cannot establish A1 reconstruction quality. See [fusion evidence](fusion-binary-real-a8-2026-10-03/README.md).

Released DSpark/DFlash seven-token models have a positive RTX2080Ti native
reference/Q4 study. Binary export/load/graphs and QAT are missing. Their current
Q4 scope is only fifteen FFN matrices; fusion, attention and their large private
full-vocabulary heads stay at original precision. Existing EAGLE training
features and 32k vocabulary masks are insufficient for these five-tap block
models. See [results](dspark-sm75-20261003/results.md) and [precision admission](dspark-sm75-20261003/precision-admission.md).

## Define the nine models before training

| Family | Frozen control | Trained candidate | Trained candidate |
| --- | --- | --- | --- |
| EAGLE | Original Q4 EAGLE | W1A8 with calibrated fusion and QAT | W1A1 with calibrated fusion and QAT |
| DSpark | Released-model Q4 control | W1A8 with its own calibrated fusion and QAT | W1A1 with its own calibrated fusion and QAT |
| DFlash | Released-model Q4 control | W1A8 with its own calibrated fusion and QAT | W1A1 with its own calibrated fusion and QAT |

These names are abbreviated deployment labels, not assertions that every tensor
uses the named bit width. Freeze a tensor-by-tensor coverage and trainability
manifest, packed payload bytes, exceptions, quantizer arithmetic and actual
kernel dispatch for each model. Target/verifier weights, sampler and KV precision
are common and immutable within each device track. Use captured native teacher
features/logits and bounded native trajectories as the admission reference.
Timebox cross-backend diagnostics unless they change labels, draft decisions,
gradients or conclusions; exact Hugging Face/native floating agreement is not
a prerequisite in its own right.

Profile the admitted block models before freezing binary coverage: the large
full head and DSpark Markov/output work were material in the bounded intrusive
profiles, although those totals do not predict request throughput. If FFN work
is a small part of clean round cost, FFN-only binary cannot be sold as a likely
speed win merely from saved bytes. A native cost/acceptance gate comes first.

Recommended implementation order is existing all-nine EAGLE support, then the
fifteen FFN projections for both block models. Admit each new native path before
adding block fusion or other projections. Fusion calibration is then an explicit
additional scope step. If block fusion becomes binary while the historical Q4
control retains floating fusion, that main result is a trained deployment
comparison, not an isolated precision effect. Use a matching-coverage Q4 or
floating-fusion binary ablation as a development diagnostic outside the final
nine; do not silently redefine the historical baseline. Full block attention,
Markov and output-head quantization are separate conditional experiments.

The human still owns the final block coverage, floating exceptions and total
compute allocation. A pure all-major-linear W1 objective and a best-performing
mixed-precision objective can select different models. A hybrid candidate must
be labeled explicitly if selected for a final cell.

## Incorporate into the training and deployment foundation

1. **Architecture-specific native data.** Reuse prompt splits and verified token
   ancestry where possible, not missing feature arrays or incompatible masks.
   Capture ordered five-tap target features for the block models, their full
   vocabulary labels/distributions as required, correct anchors, bidirectional
   noise masks, target injection, cache rollback and actual slot mapping.
   Preserve original private head values and an immutable teacher path. Do not
   borrow approximate target heads or accidentally train the target via aliases.
2. **Balanced training coverage.** Shuffle/interleave whole recurrent rounds or
   blocks across prose, code and reasoning, preserving chronology inside a
   chain. Count supported labels, presented tokens, unique rows/prompts and
   source/depth/length coverage. Repeated easy examples are not an epoch.
   Expand fusion fitting and validation to all three domains and longer contexts.
3. **Fusion fit as initialization.** Preserve fitted signs and scales exactly in
   the latent model and checkpoint. Use the deployed A8 or A1 hard arithmetic
   and each architecture's fusion/norm ordering. Treat an A8-fitted warm start
   for A1 as an initialization experiment, not an A1-calibrated result. Include
   mean-absolute and converged-scale controls. Promote calibration on native
   early-prefix survival and throughput, not reconstruction SSE alone.
4. **Hard deployment forward and architecture-specific objectives.** EAGLE
   retains attached student-state recurrence and audited token-prefix labels.
   DFlash/DSpark retain their own block layout and conditioning. Compare the
   block recipe's objective with a hard-target-CE QAT control before committing
   long compute. Do not transfer EAGLE's scalar loss blindly to DSpark.
5. **Sufficient final-precision exposure.** Define a coverage target, maximum
   trainer-accounted time and evaluation allowance after measuring each actual
   model, including detached evaluator and host-memory lifecycle. Use token/coverage-based checkpoints and a stable learning-rate plan.
   Keep fixed-A8 AdamW as the current reference, not the underperforming combined
   recipe. Select learning-rate decay/warmup as a labeled recipe; the existing
   constant-LR run does not establish long-run convergence behavior.
6. **Training speed and memory.** Retain validated computation/cache/head paths
   where their semantic assumptions hold. Share sibling quantization during
   training after current-source gradient checks. Measure chunk sizes, modest
   independent batches or accumulation, peak GPU/host memory and supervised
   tokens/s. A larger optimizer batch changes the recipe and needs a matched
   learning comparison. Block K/V injection can reuse projected target context;
   EAGLE's one-layer catch-up shortcut is not blindly portable to five layers.
7. **Native evaluation and attribution.** Use periodic lightweight acceptance
   tests, less frequent clean timing of promising checkpoints, and final paired
   same-device request tests. Record early-position survival, root acceptance,
   emitted tokens per round, packing/draft/head/fusion/cache/verification/host
   costs and memory. Keep intrusive GPU profiles separate from throughput runs.
   Apply common runtime improvements to every applicable Q4 control too.

The released DeepSpec config at author revision
005e03b81cec38b7da6399833d609ee89a2587f2 uses DSpark CE plus probability L1 and confidence
training, while its DFlash control uses CE without Markov/confidence. Both have
position decay. These are different released recipes, not one loss with a
renamed architecture. Keep confidence scheduling disabled in the initial main
matrix: author overlap confidence is not automatically the local greedy
verifier's prefix survival. If soft teachers are needed, stream exact logits or
validate compact-teacher tail accounting; do not call approximate top-k data
full-vocabulary L1. [DSpark config](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dspark/dspark_qwen3_4b.py),
[DFlash config](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/config/dflash/dflash_qwen3_4b.py),
[loss implementation](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/modeling/dspark/loss.py).

## Development experiments worth including

The order below is a dependency order, not a promise that every experiment
needs to run for every family. Use a predeclared development allowance. Screen
one change at a time, then verify the selected combination against the best
singleton and the fixed control at meaningful coverage before allocating long
compute. Interactions such as fusion initialization with warm-start or depth
weighting with refresh are plausible; a singleton screen cannot establish that
stacking the winners is beneficial. A very short run can reject instability
or a broken contract, but cannot reject a slow-learning binary recipe solely
because it has not yet crossed a sign boundary.

| Priority | Experiment | Promotion evidence |
| --- | --- | --- |
| First | Fusion fitting at A8 and A1 for each family | Improves native prefix survival without losing complete-request performance; all-domain data and correct deployment arithmetic |
| First | Direct A1 versus short A8 to A1 | Better native quality at common exposure and charged total compute; substantial hard-A1 phase remains |
| First | Learned A1 thresholds alone | Improves native acceptance and throughput versus the same surrogate with frozen threshold; exact native packing/export |
| First | Mild early-position CE weighting | Better useful prefix survival without collapsing later proposals; retain full horizon and fixed normalized token accounting |
| Next | Gradient/inertia and scale normalization controls | Stable useful decision changes, not merely flips; isolate initialization, gradient rule and clipping rather than stacking them |
| Next | Student-prefix refresh | Include one bounded development probe after its native recapture/data path is valid; prioritize if capture/student mismatch grows or offline loss separates from native quality; compare against matched no-refresh continuation |
| Conditional | Fusion output bias, then rank-one residual | Calibration/QAT leaves a localized fusion defect with measured speed headroom; actual quality improvement exceeds added operator cost |
| Conditional | Fusion-only affine centers | Isolate from learned quantizers and inertia; integer-sum metadata and real quality/cost evidence |
| Conditional | Architecture-appropriate soft-label distillation | Exact teacher/label/prefix join and a controlled benefit over hard labels; full probability storage/compute accounted |
| Conditional | Discrete sign/scale fitting for another sensitive layer | A measured layer defect supports it; fit actual current inputs and verify native decisions before broadening calibration |

Use cheap EAGLE screens for shared quantizer/optimizer mechanisms where useful,
but retain architecture-specific confirmation and loss/coverage checks. EAGLE
state recurrence, the block model geometry and DSpark Markov objectives differ;
EAGLE results cannot select a block recipe by themselves. Avoid a full factorial
search across all six candidates. Keep the main selected combination small.

For A8 to A1, charge source A8 training to the A1 lineage even when another
candidate reuses it. Report marginal shared-campaign cost separately. Preserve
latent magnitudes, signs and scale representation; activation bits changing is a
new stage, not exact resume. The current curriculum code recreates optimizer
moments and activation parameters at transition. Use an A1-to-A1 reset control
if isolating warm-start precision from optimizer-reset effects. Moment carryover
is a possible extra development arm only after its explicit transition contract
is implemented and admitted; the current runner does not provide it. Do not
claim moment preservation that the implementation does not provide.

The learned A1 threshold experiment changes both hard bits and its surrogate.
Include a fixed-threshold version of the same surrogate to separate learned
threshold capacity from backward changes. Preserve invocation-local LSQ
normalization for batching/reuse, or explicitly admit a new normalization recipe.

A low-rank floating fusion correction can be a useful best-model fallback, but
is not a strict binary-linear layer. Compare its zero-initialized identity,
trained bias control and measured correction cost. Keep a pure binary candidate
as the control; do not describe the hybrid's gain as binary arithmetic alone.

## Engineering experiments after profiling

| Item | Disposition |
| --- | --- |
| Immediate sibling activation reuse | Incorporate after actual source/VJP and memory/time tests; reduced CPU prototype is promising correctness evidence |
| Learned-head batching | Preserve the serial safeguard by default; implement invocation-preserving batching only if it reproduces gradients, otherwise study the new normalization separately |
| Provider/audit startup reuse | Incorporate bounded trusted reuse if it reduces measured startup work; preserve actor-specific admission and corpus/mask/source hashes |
| Correct precompute caps | Worth implementing and testing for eligible sequential paths; block models still compute their fixed trained noise block unless the graph changes |
| Compact vocabulary selection | Apply existing exact selector where compatible; verify full-head block behavior rather than transplanting EAGLE's 32k map |
| Cost-aware stopping / calibrated confidence | Explore on trained candidates only if prefix-survival and verifier cost leave achievable headroom; fixed seven-row block work is not saved merely by returning fewer tokens; same controller tests on Q4 |
| Full-vocabulary head Q4 control | High-value conditional block-model test if profiling identifies head bandwidth/cost; generic Q4 conversion exists, so test this before learning a compressed head; preserve the separate immutable target teacher and label head precision in binary candidates |
| Fused gate/up dot operations | Profile first; investigate only if launch/weight-load costs matter after shared packing |
| Existing SM75 binary-MMA code on block shapes | Conditional reuse/dispatch experiment after W1 quality and native graph admission; older head-only tie does not settle seven-row FFN performance, and routing the current W1Ax contract is not automatically implemented |
| Shared pack, warp reduction, resident state | Already tested with small or negative bounded gains; no blanket adoption or assumed combined benefit |
| Unused head pruning and eligible K/V catch-up | Use existing validated paths; test changes on each actual trained model and applicable Q4 control |

See [runtime results](eagle-runtime-optimizations.md), [training reuse prototype](parallel20261002/activation_reuse/report.md)
and [LSQ batching audit](parallel20261002/lsq_batching/report.md).

## Keep out of the initial long training recipe

Bop, the toy first-mismatch/rank-preservation optimizer, and extensive
acceptance-survival loss searches can be bounded research probes if ordinary
training demonstrates an optimization failure. They are not defaults for six
expensive jobs. Candidate-audit and rejected-update costs must be charged;
reject-only guards can stall before latent sign changes.

Defer learned Q1_0 deployment, A4 curriculum ladders, two-bit/ternary controls,
FFN structural pruning, smaller DSpark backbones, Hydra/Medusa heads, learned
head factorization/shortlists, broad mixed-precision sweeps and teacher-state
regression unless the initial batch establishes a specific capacity or runtime
bottleneck. These change representation, architecture or objective substantially.
They can form the next research batch without blocking this one.

Do not revive the tested cold-SVD head, centroid shortlist router, semantic
fusion-plus-QKV group-A1 proposal, blind rotations or failed two-plane initializer
without a new hypothesis/data screen. They already have negative evidence.
The underperforming combined learned-A8/midpoint/inertia run is completed
negative evidence, not a reason to combine those factors again unexamined.

## Long training and final comparison

Recommend a staged allocation rather than an arbitrary identical hour count:
measure real throughput and memory, reserve about 10-15% of a proposed total
compute allowance for bounded recipe selection, run diagnostic development ablations,
freeze six candidates, then train through substantial balanced corpus coverage.
The 10-15% selection allowance is a planning proposal, not a validated optimum
or permission to spend compute. Stop opening new recipe branches when that
allowance is exhausted; retain explicit no-result outcomes. A full available-corpus
pass is a reasonable planning target for EAGLE, with
further passes conditional on progress; it is not a sufficiency claim or an
allocation approved by this report. Block models require their own new corpus
and rate pilot. Record token, prompt and anchor exposure because an EAGLE update
and a block update supervise different amounts of data.

Extend beyond the first pass only when native development quality is improving
and the model has achievable request-speed headroom. Use predeclared patience
and minimum relevant exposure before declaring a plateau. Do not automatically
cancel at an arbitrary fraction of Q4 acceptance or a fit of acceptance against
log(tokens): the sparse, changing-domain curve is not a reliable future capacity
bound. Use measured costs and current trajectory limits to guide a human
checkpoint decision, not to declare that further training cannot work. A1 can
have different cost/quality tradeoffs, so A8 underperformance alone does not
justify cancelling A1. Numerical failures,
invalid data ancestry, resource failure or verifier/cache defects require a
checkpoint and repair, not budget erasure. Do not stop a slow learner purely
for failing to beat Q4 early. Do stop expending large deployment engineering
budgets when a measured acceptance/cost model rules out useful headroom.

Reserve independent prompt-disjoint development coverage for recipe selection
and for later confirmation. Repeatedly choosing across many recipes on the
same 24 prompts invites overfitting. Use the original 24 as regressions;
predeclare a larger balanced selection/confirmation split without accessing
sealed finals. Freeze models, policies and selectors before the final set.

Example planning rungs are 50k, 100k and 200k supervised tokens for screening,
then 25%, 50% and 100% of the declared balanced corpus for long-run decisions.
These are proposed exposure milestones, not approved budgets or proof that
all binary variants learn within the early rungs. Record effective loss masks
and weighted exposures rather than treating padded/no-loss tokens as training.

Use common supervised-token rungs within each family for learning comparisons,
plus total-compute curves for practical efficiency. Neither replaces the other.
Different per-step cost is a real outcome for an equal-time deployment experiment;
it does not invalidate the completed October 4 comparison. Cross-family counts
also need prompt/anchor/length coverage because token labels have different roles.

Measure training variance separately from repeated evaluation timing. Recommend
two seeds at meaningful baseline/probe rungs, and reserve second-seed confirmation
for selected long recipes where cost allows or a close/new-method claim requires
it. A single final training lineage can still supply one deployable artifact,
but five timing repeats do not establish training-seed robustness.

Choose on the selection split, then confirm the frozen recipe/checkpoint on an
independent confirmation split. Repeatedly selecting checkpoints on that split
would turn it into another selection set. Use paired prompt-level confidence
intervals where the sample supports them, and keep training-seed variability
separate from prompt/timing uncertainty.

Trained binary versus original released Q4 measures deployable improvement,
not the causal advantage of the bit format. A small same-data/coverage F16 or
Q4-fine-tuning diagnostic with the same trainable layer scope can reveal domain
adaptation. Include it in bounded exploration if attribution matters; a full
matched adaptation control is necessary for a strong final causal claim. It is
outside the nine frozen-control deliverables and costs must be recorded.

The final output remains nine models, with per-family Q4/A8/A1 tables and one
cross-family table on each common feasible GPU. Use the same target, prompt
bytes, sampling, output/context lengths, concurrency and KV policy. At least
five balanced timing repetitions, paired raw IDs and counts, request throughput,
latency, emitted tokens/round, peak memory and tensor coverage are required.
Target-only is an additional diagnostic, not a tenth trained candidate. Keep
SM120 and SM75 results separate; one device cannot establish the other's speed.
If a model does not fit a common device under fixed settings, mark that cell
unavailable rather than reducing target precision for only that model.

## Additional findings from the eight agent research slate

The human requested four literature-first and four reasoning-first agents to
find easy additions, then peer review every report. All eight completed; five
successful Opus 5.5 calls audited the reports and reconciliation. See the
[additional research synthesis](nine-model-research-slate-2026-10-04/synthesis.md)
for sources, costs, all dispositions and qualifications. The nine final models
and original selection allowance remain unchanged. These are implementation
recommendations and experimental options, not changes already made or gains
already measured.

**Add to the engineering preparation:** remove only the unnecessary F32
latent clone before each Boolean sign snapshot; optionally consolidate typed
metric reads while retaining finite-before-update checks and exact I64 counts.
This revives an existing CPU prototype. Share an immutable raw-feature upload
only where paired lanes consume the same data/device. Measure live status/fsync
time before changing publication cadence; recipe-audit output is probe-gated,
not every update. Checkpoint, budget, failure-event and watchdog durability stay
intact. Training-only work has no counterpart in the three frozen Q4 models.

**Add to the existing fusion screen:** census zero-scale rows, propose whole-row
sign reversal and refit nonnegative scale for negative TRAIN correlation, then
accept only finite exported-F32 objective improvement. A read-only census of
the authenticated existing A8 report confirms all 383 scale-zero rows have
negative correlation. This is a fitting opportunity, not proof of dead QAT
channels: the current trainable scale offset has derivative one at zero and can
revive. The rescue initializes fresh QAT; it does not edit a resumed optimizer.
Count global row reversals separately, fit A1 using its own arithmetic, and
confirm the whole candidate on prompt-disjoint native evaluation. Do not choose
per-row signs on the same set later advertised as independent validation.

Also score existing fusion candidates after the actual following normalization
and retain raw residual-amplitude error. If these metrics explain native
failures, screen one per-example-energy-weighted SSE control. Constant positive
weights on independent output rows do not change their LS minimizers; example
weighting is not equivalent to the coupled post-RMSNorm objective. Keep these
changes as singletons before selecting any combination.

**Prepare a bounded execution probe:** fused FP32 AdamW with explicit backend
identity, correct CPU/device step-state initialization and exact resume layout.
Keep formula, masters, moments, clipping and projection order; examine hard sign
changes and targeted native consequences of floating reassociation. Use actual
optimizer-stage and whole-step measurements. Grouped F32 attention avoiding
physical K/V repetition is another probe if memory or its stage cost matters;
retain group layout, cache casts, masks and later gradients. Bounded prefetch
waits for real provider/upload stalls.

**Add cheap native options while working on block graphs:** suppress DSpark's
unused confidence output at p_min zero with an explicit request/reuse key and
positive-p_min/diagnostic fallback. Test actual node/copy removal, semantics and
clean timing on DSpark Q4 and its binary candidates. A balanced logit concat
tree and the one-block identity-layout fast path are lower-priority options
only if assembly cost is material. Keep sequential Markov conditioning; defer
aliased preallocation. These are small possible savings, not replacements for
acceptance improvement or the binary deployment prerequisite.

**Learning probes that fit existing allowances:** one localized existing
drafter norm gain on A8 if channel errors justify it, and one short attached
accepted-context K/V tail if new context-attention gradients are material.
Norm ownership/export must stop sharing that parameter across lanes. Positive
channel rescaling before a zero-threshold A1 sign quantizer preserves bits,
though meanabs amplitude changes; crossing zero or learned thresholds is a
different interaction. The context-tail probe starts on EAGLE; do not port the
one-layer gradient/cache shortcut blindly to the block models.

**Refine refresh and batch admission:** retain original and recaptured complete
rounds as separately identified replay sources; report unique and repeated
exposure. Declare round-mean/token-mean/weighted valid-loss denominators,
especially with accumulation or loss masks. Declare DSpark's predecessor-token
conditioning and why teacher labels remain valid. Root-label support, signed
margins and conditional survival by depth are diagnostic prerequisites. A
bounded update-resolution sign sample can test actual chatter before any late
dampening; the existing return-to-initial-sign counter is not that statistic.

**Keep later research conditional:** SFDD flatness ordering/subsets require
exact teacher statistics and a balanced matched control; their published greedy
results show a quality/compute tradeoff rather than a binary-QAT win. AdaSPEC
CE-regret needs a well-covered matched reference and is not worth a new long
reference job. Censored rejection-anchored weighting may refine the existing
depth-loss probe only where current/captured prefix identity and teacher labels
are proven; never label an unknown continuation as observed rejection. Full
VAT heads/soft labels wait for valid current-prefix capture.

EWGS is a lower-priority sign-gradient adaptation requiring coordinate and
round-aggregation identity; measure its effect under Adam before a training
arm and do not escalate the factor blindly. Late oscillation dampening waits
for harmful reversal evidence. SM75 BF16-to-F16 floating-exception dispatch
stays outside frozen baseline policies until value and reduction-path checks;
head Q4 is the more direct byte-reduction control when bandwidth dominates.
Reject a global cuBLAS-F16 override, unsupported feature noise, broad low-bit
moments/compile/checkpoint changes and the previously failed representation
branches as automatic additions.

## Peer review and decision status

Two focused peer-review MCP calls completed with requested model
`claude-opus-5-5-high` (Opus 5.5). One reviewed model/training choices; the other
reviewed experiment design and budgets. The broader max-reasoning call timed out
at the bridge's 180-second limit and supplies no scientific feedback. No Fable
fallback was requested or used. The initial dual-model Gemini leg failed its
bridge's invalid max-effort argument and supplies no scientific feedback.

Accepted review changes: put binary deployment and actual rate/memory pilots
on the critical path; profile the full block head before choosing coverage;
expand development selection/confirmation; measure seed variance; use token
rungs plus total-cost accounting; test the best combined recipe against the
best singleton/control; promote one own-prefix-refresh development probe;
add the domain-adaptation-control option and honest deployment attribution.

Qualifications and rejected suggestions:

- Do not transfer an EAGLE winner to block models without architecture-specific
  confirmation; their objective, masks and vocabulary differ.
- Do not treat adaptive stopping as a math-preserving activation-reuse patch.
  It changes proposal/verification trajectories and needs native quality and
  clean timing tests on both candidates and controls.
- Do not use an arbitrary 75% of Q4 acceptance or a sparse log-token fit as an
  automatic training futility gate. The minimum meaningful exposure, plateau
  evidence and achievable measured cost/quality tradeoff guide decisions.
- Current DSpark/DFlash study uses an F16 target and F16 KV with BF16 drafter
  source storage; a review's shorthand BF16 does not change target precision.
- Fusion direction/ranking improved versus converged scale-only, but still
  trailed the original initializer on some metrics. A review's statement that
  no such improvement was shown was too broad. Fit has eight fitting prompts
  and four validation prompts, with 384 rows total, not 384 fitting rows.
- Equal-time and equal-token answer different questions. Different training
  rates do not invalidate the deliberately frozen equal-time experiment.

Raw reviews are retained outside Git under
`runs/nine-model-qat-planning-20261004/opus-5-5-high-review-{1,2}.txt` in the
primary workspace. SHA256 values:

- Model/training review: `30ce067f87ea9680b5ae0ccfec5ea96a138a4a9889b5c5fc224f139cb896122c`.
- Design/budget review: `cdd8a1f519d2a77e8e1d850f9a281fa7f812325e132d850a06c3129e6cc5252c`.

All scientific claims remain tied to the measured reports
and source contracts rather than to reviewer authority.

The human's nine-model direction is accepted as planning intent. Final coverage,
compute allocation, recipe promotions and experiment launch remain unselected.
This report starts no goal, monitoring, remote operation, training or budget.

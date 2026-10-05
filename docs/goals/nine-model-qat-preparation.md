# Nine model QAT preparation

## Current hardware boundary: Mac only

Latest direct human instruction: **“pause all 5080 usage continue mac only.”**
Root immediately ran `agent_env.py pause rtx5080`. All feature/QA owners were
notified; preparation remains active for Mac source/CPU work. Do not connect,
query, stage, build or run on either remote GPU host until renewed authorization.

The sole operator confirmed its dependency supervisor was already terminal
(`nineprep-5080-uvsync-20261004`, exit0,23:45:20.701665UTC). No compiler probe,
CUDA build, model load or capture had started. Its SSH invocations returned
and MCP transport `nine-model-5080` session256 was closed. No other team's
processes, stale state, dirty root checkout or untracked artifacts were changed.
No remote call was made after cleanup/closure. The last device snapshot is
historical23:42:18.969940UTC:13,495MiB free, zero compute rows/DXG holders;
this is not a new claim about current whole-device occupancy.

Preserve the isolated remote checkout pinned b998f4f/native874cd2b, locked
Python3.11.15/CMake3.31.10/Torch2.14.0 environment,4.31GB cache and raw logs.
They are preparation artifacts, not CUDA execution evidence. Actual CUDA build,
model/backward/memory and materialized native TRAIN/golden checks remain PENDING.

## Historical RTX5080 development authorization — October 4, 2026

The human explicitly announced: **“5080 is open use that for development work.”**
This supersedes the initial RTX5080 no-query/pause boundary below. Root resumed
the machine-local RTX5080 flag; `/root/cuda_operator` remains the sole operator
and is now dispatched to fresh RTX5080 identity, process/context and resource
checks through tmux MCP using the shared registry.

Authorized there: isolated source staging/builds, bounded synthetic optimizer
fixtures, native conversion/operator/graph checks, actual model forward/backward
with zero real-model optimizer updates, full training-memory reservations without
updates, and necessary target-only TRAIN data/calibration/golden preparation.
Long QAT, recipe-selection quality/performance runs, held-out and final evaluation
still await the human's recipe/coverage/compute decision. No new goal/operator or
RTX2080Ti operation is created. Preserve other teams' jobs.

The native Goal's original pause phrase describes the initial boundary; this
authorization and durable project record control current work. The operator
records actual SM120 hardware, source/binary/model hashes and each owned detached
supervisor/group/context/resource closeout. SM75 claims remain pending/excluded
if only SM120 is exercised.

## Initial objective and authorization

October 4, 2026. The human requested a separate independent Codex task and
agent team to implement and test preparation for the nine-model campaign:
EAGLE, DSpark and DFlash, each with original frozen Q4 and fusion-calibrated
long-QAT W1A8/W1A1 candidates. Prepare code, data, configurations, artifacts
and launch/evaluation orchestration so no further integration is needed when
the RTX5080 becomes available. Training and model-quality/performance evaluation
are reserved for RTX5080. This objective authorizes preparation implementation
and testing, not starting a training/evaluation job now.

The human said they will open RTX2080Ti for development, including CUDA
testing/preflight. Until it is actually available, progress on local CPU/source
work. A stale unpaused host flag is not evidence of availability. Once available,
one operator may use it for bounded CUDA development tests, builds, native
pack/load/graph/operator checks, actual-model forward/backward smoke without
real-model optimizer updates, and necessary TRAIN teacher-data preparation.
Target-only TRAIN feature/logit capture is data preparation, not a drafter
quality/performance evaluation; record producer hardware and bound its resource
use. Do not run the long QAT, recipe-selection quality runs, held-out benchmark
or final evaluation on RTX2080Ti. Synthetic optimizer/save-resume fixtures are
development tests, not real-model training. Sealed-final content stays untouched.

RTX5080 is currently unavailable. Parent set its machine-local pause flag true,
which blocks new runs; no remote connection, interruption or availability
verification was performed. Do not query, stage to, resume or run on RTX5080
until the human announces availability. Preserve that pause and other teams'
processes, checkouts, captures and existing results. Read host addresses only
from the shared registry in AGENT_OPERATIONS; never persist IPs here or guess.

## Ownership and durable record

Sole preparation coordinator is `01a10903-1c7a-71b1-abb1-0de3ecc046b8`,
claimed October 4 from published handoff `da097e5`. Parent
recap/research chat is a launcher only and will stop editing these records after
dispatch. Do not reuse the completed A8 team's ownership or restart its monitors.
The previous A8 goal and eight-agent research slate remain COMPLETE.

Coordinator owns STATUS, this goal file, recipe/coverage decisions and integration.
Use the configured Sol-high coordinator/feature owners, Luna-high tests/operator,
Astra-medium focused advice, and Opus 5.5 peer review with exact requested model
claude-opus-5-5-high. Assign each mutable file/native module once; use isolated
worktrees, review/tests, frequent short commits and pushed main. Native commits
must be pushed to the fork before the parent gitlink. Workers are not alone and
must preserve each other's edits. Keep one RTX2080Ti operator and serialize GPU
use. Every SSH connection must use tmux MCP; detached jobs use the Linux tmux
socket and remote_job supervisor prescribed in AGENT_OPERATIONS.

## Authoritative inputs

Read the current main version and these records; do not reload every historical
goal document at startup:

- [Main nine-model plan](../../experiments/nine-model-qat-research-plan-2026-10-04.md).
- [Eight-agent additions and peer reconciliation](../../experiments/nine-model-research-slate-2026-10-04/synthesis.md).
- [Completed A8 comparison](../../experiments/a8-qat-recovery/comparison.md).
- [DSpark/DFlash native results](../../experiments/dspark-sm75-20261003/results.md) and
  [precision gap](../../experiments/dspark-sm75-20261003/precision-admission.md).
- [Real fusion fitting](../../experiments/fusion-binary-real-a8-2026-10-03/README.md) and
  [source-pinned zero-scale census](../../experiments/nine-model-research-slate-2026-10-04/zero-scale-census.json).
- [Evaluation protocol](../EVALUATION.md), [operations](../AGENT_OPERATIONS.md) and
  relevant [lessons](../USER_LESSONS.md).

Research-only baseline main d9b07a3 contains all reports. It is not a deployable
nine-model training bundle. Most block binary paths and training/data bridges
are still missing. Existing optimized EAGLE forward/backward/cache/head paths
have actual hardware evidence; old completion-audit pending claims can be stale.
The underperforming learned-A8/midpoint/low-inertia combination is not the default.

## Work packages

The independent coordinator should launch bounded feature owners and reviewers,
not another literature-only slate. Recommended ownership partitions, refined
against current files before edits:

1. Native DSpark/DFlash W1 export, loader and graph with honest selected-tensor
   manifests and CUDA dispatch. Start with fifteen FFN matrices, then admit
   architecture-specific fusion; prepare supported coverage options without
   pretending an activation flag makes dense weights binary. Preserve private
   embeddings/head, anchor-first slot mapping, Markov semantics, masks, cache
   repair and immutable target/verifier. No approximate head borrowing.
2. Training integration for EAGLE and block models: hard deployment-forward
   quantization, own-state/token-prefix contracts, correct full/pruned vocabulary,
   family-specific objective controls, parameter ownership and finite gradients,
   checkpoint/optimizer/RNG/cursor exact resume, export and stage transitions.
   Fixed reference, direct A1 and A8-to-A1 are supported profiles; optional probes
   must remain explicit and off by default until admitted, not stacked silently.
3. Data and fusion preparation: reuse authenticated existing EAGLE inputs;
   new five-tap/full-vocabulary block data where needed, balanced prompt-disjoint
   splits, bounded teacher storage, whole-chain order and label/cache ancestry.
   Fit/calibrate A8 and A1 using their actual arithmetic, with scale-only controls,
   row-orientation rescue and post-norm/raw diagnostics. Source derivative-one
   scales do not prove permanently dead QAT channels. Preserve historical fits.
4. Prepared pipeline and evaluator lifecycle: current-package source/config/data
   bindings, launch guards, cheap reuse of completed admission/audits, training
   ledgers, detached evaluator, process/context/resource release, result joins
   and automatic sequencing on successful committed endpoints. Avoid the old
   multi-hour repeated hashing/audits, in-process host-RAM evaluation failure,
   invented development-ID prefix checks and raw-tensor metadata collection.
5. Training/native performance engineering where applicable: one-line sign clone
   removal, typed metrics, paired-only upload reuse, source-correct logging
   cadence, admitted fused-optimizer option, grouped attention and static
   confidence. Profile-gated refinements are configurations with truthful
   evidence, not blanket assertions of benefit. Do not block readiness on the
   deferred formats/architectures or implementing every one of the new ideas.
6. Independent QA: own tests/acceptance ledger and review evidence separately
   from feature authors. A sole Luna operator dispatches coordinated CUDA tests;
   QA never creates a second host owner. Astra/Opus advise on difficult contracts.

## Readiness acceptance

Maintain a requirement-by-profile ledger: PASS with original evidence/hash and
hardware, PENDING with exact missing check, or EXCLUDED for a deliberately
unsupported option. Never claim a percentage from test count or that synthetic
fixtures prove real-model quality, movement or SM120 memory/performance.

Before declaring portable preparation complete, require:

- All declared profiles/configs parse and execute actual source APIs; loader/
  vocabulary/shape/padding/precision/export and bad-input tests pass. No mock
  evaluator or dense fallback can grant readiness.
- CPU and allowed SM75 pack/operator/graph and forward/backward fixtures,
  later-state/K/V gradients, serialization/resume/RNG and stage-transition
  tests pass with model/target storage ownership. Actual-model SM75 smoke uses
  zero real-model optimizer updates and no held-out quality evaluation.
- Data is materialized or its exact required capture stage and cost are
  exposed; missing block taps or full-distribution teachers cannot be hidden.
  Captures bind the actual producer GPU/native target and exact prefix.
  Native-captured teachers plus bounded trajectories are the numeric reference;
  timebox cross-backend diagnostics unless labels/decisions/gradients change.
- Memory and out-of-process lifecycle tests cover host/GPU floors, source/data
  integrity, STOP/caps/failures and supervisor/context cleanup. Preserve all
  protected files and failures. Empty WSL compute-app lists do not prove release.
- End-to-end synthetic pipeline tests prove checkpoint -> export -> resource
  release -> evaluator -> report, with frozen Q4 controls and target-only
  diagnostic; paused/unavailable/stale-source/failed-export scenarios refuse
  correctly. Real measured results remain reserved for RTX5080.
- Full applicable suites, build/lint and independent review pass on integrated
  source. Commit/push native and parent artifacts coherently; preserve raw
  models/captures/runs outside Git and record versions/hashes.

Prepare a frozen launch bundle and one concrete start command with no code edits
needed. It should run the minimal fresh RTX5080 source, resource, kernel/model/
backward/memory and capture-portability checks once availability is announced,
then start the approved QAT recipe/budget and automatically run native evaluation
after successful committed training and verified resource return. A failed guard
must retain the checkpoint and report the precise failure, not waive it or
blindly start evaluation. Clear ambient experimental flags and bind the actual
binaries/runtime libraries and current recipe. Use separately bounded checkpoint
quality and final timing stages, not intrusive profiler timings as throughput.

SM75 testing cannot certify SM120 execution or resource headroom. Report portable
software/data readiness separately from the pending fresh RTX5080 admission.
Do not repeat completed full corpus audits or synthetic suites just because the
host changes; reuse source-bound evidence where valid and perform only the
necessary fresh target-device checks. Any late dataset or artifact dependency
that prevents prompt QAT launch is a visible remaining preparation item.

## Decisions and launch boundary

Implement the user-authorized foundation and selected feasible preparation
options autonomously. Prepare explicit proposed tensor/exception coverage,
objective/quantizer profiles, balanced token/epoch milestones, wall caps and
evaluation allowance; record evidence and options in DECISIONS. The human owns
major recipe/coverage choices and long training compute allocation. Do not
quietly choose an unbounded budget or promote unmeasured recipe combinations.
The current task stops short of real QAT/evaluation while RTX5080 is unavailable.
When the human starts the prepared campaign with availability and a selected
budget, automatic training-to-evaluation sequencing is already authorized by
this request; no redundant confirmation or integration step is needed.

Keep brief milestone updates; update this file before handoffs/context rotation.
Final preparation report must show completed work/tests, exact commits/artifacts/
command, any excluded probes, and the narrowly defined remaining SM120 checks.
Never claim literal 100% hardware/quality certainty or immediate execution
without a fresh resource check. Complete all work that can be done independently
while a host or scientific decision is pending.

## Implementation team started — October 4, 2026

Native Goal is active in the coordinator chat. Source baseline parent `da097e5`,
native `fcdf5822`. All original untracked overnight files and older worktrees
are preserved. No host connection or training/evaluation was started.

Temporary worktree root: `/private/tmp/nine-model-qat-20261004/`.
Each author owns its isolated branch; coordinator reviews/tests/cherry-picks to
main and pushes before retiring any merged worktree.

| Native subagent | Worktree / branch suffix | Exclusive responsibility |
| --- | --- | --- |
| `/root/native` (Sol high) | native | native fork export/load/graphs, block export and TRAIN teacher capture producer |
| `/root/training` (Sol high) | training | block QAT, EAGLE continuous/recurrent QAT, exact resume/stages, training CLI |
| `/root/data_fusion` (Sol high) | data-fusion | five-tap/full-vocabulary data bridge, deployed-arithmetic fusion fitting |
| `/root/pipeline` (Sol high) | pipeline | frozen configs/launcher, process/resource/evaluator lifecycle and report aggregation |
| `/root/qa` (Luna high) | qa | independent requirement/profile ledger, CPU suites and failure-path tests |
| `/root/cuda_operator` (Luna high) | own report worktree pending | sole coordinated RTX2080Ti operator, initially local planning only |
| `/root/contracts_advisor` (Astra medium) | read-only | native architecture/data/mask/ownership advice |

Coordinator exclusively owns STATUS, this goal, DECISIONS and integration.
Training ownership additionally includes `recurrent_qat.py` for sign clone and
typed diagnostics. Workers must request overlapping expansions. Peer reviews
use exact `claude-opus-5-5-high` with bounded focused requests.

Early concrete dependency: existing EAGLE three-tap/pruned-vocabulary data cannot
supply block five-tap/full-vocabulary teachers. Native owns the real capture
producer; data owner defines ancestry/storage/cursor contracts. Materialized
block TRAIN captures and producer hardware/portability evidence remain PENDING.
The sole operator must plan their cost and identify source model dependencies.

RTX5080 remains locally paused/unavailable; no query/staging/resume authorized.
RTX2080Ti availability clarification requested; stale registry flags are ignored.
No remote operation until direct human availability, followed by sole-operator
fresh resource/ownership checks. CPU/source implementation progresses meanwhile.

Next: settle cross-owner interfaces, land focused tested implementation chunks,
review native arithmetic and block conditioning, build independent PASS/PENDING/
EXCLUDED ledger. Portable readiness and fresh SM120 admission remain separate.

## Contract milestone — October 4, 2026

Ownership and decisions published parent commits30791d0,1ce1a9d; CUDA
local-only plan df9b6f7; slot lesson c1f2fdd; Opus native contract review c276740.
Parent main is pushed. Native/source implementation remains in author worktrees,
not yet integrated/admitted. Operator worktree is
`/private/tmp/nine-model-qat-20261004/cuda-operator`, branch
`prep/nine-model-cuda-operator`; no remote tmux/session/job exists.

Astra source review resolved both released families to author slot0, compute7,
explicit `draft-dspark` driver. Schema/tests must bind driver/slot instead of
infer from family. DSpark native sequential argmax conditions Markov predecessors;
full-distribution teacher supervision past a student prefix mismatch needs
current-prefix frozen native teacher rows. Static captures are replay/censored
profiles, never silently called own-prefix supervision. Native capture owner
is implementing a real target-only producer decoupled from EAGLE-specific state.

Exact Opus5.5 bounded review completed on native fcdf5822, saved in
`experiments/nine-model-qat-preparation/opus-native-contract-review.md`. It covers
selected packed-only loader/graph paths, metadata arithmetic, shared activation
boundaries, zero sign, block/cache invariants and raw-fusion outliers. The
reviewer's strictly-positive-scale recommendation is qualified: finite zero
scales are legal and additive scale gradients/rescue are preserved.

QA initial baseline: full unittest discovery1096 tests,59 errors,2 skips in
incomplete worker environment; many errors are missing gguf/yaml/package
metadata. Full Ruff check1183 baseline findings,format73 files. QA is normalizing
dependencies and submodule initialization before final failure classification.
No baseline error has been waived or classified as new regression yet. All
changed/profile-applicable files must pass their suites/build/lint; baseline
repository debt must stay explicit if it remains.

Storage risk exposed to data/native/training: full151936-vocabulary float32
teachers cost607744bytes per supervised row (~2.37TB for3.9million rows),
before five taps/metadata. Bounded streaming/chunk caps and any admitted reduced
storage precision are explicit profile decisions; no hidden whole-corpus
dense-teacher materialization. Actual producer rates/storage and SM75 model
backward memory remain unmeasured pending hardware availability.

No human2080Ti availability reply yet; no host operation or real-model update.
Next: land tested author APIs, independent contract/failure reviews, actual
producer integration and storage/resource cost plan; then serialized allowed
SM75 checks once direct availability is supplied.

## Tested integration milestone — October 4, 2026

Main integrated/pushed through `fa0ce7d`, nativefork `874cd2b04` (pushedbefore
parentgitlinks). Worker ownbranches retained for activefollowups; no sourcework
or olduntrackedresearch dropped. Main includes:

- Block data/fusion `aada40c,cc87403`: TRAIN inventory/native receipt/prefix/
  five-tap/full-vocabulary joins; prompt/content-disjoint calibration; whole-chain
  balanced domain cursor; bounded mmap teachers; source-bound initialauditreuse
  with current file identity and per-consumed-block finite checks.
- Native `1702368,51f831d,bf054bf`: selected15FFN/optionalFC packed exporter,
  strict loader and bothgraph routes; private head/embedding retained; source
  hardware/precision-bound persistent native teacher; exact paired geometry;
  same-PGID teacher and actual SIGTERM/reap fixture.
- Training foundation `dc5d694`: block own-state hard-forward graph, prefix
  conditioning/masks/fullprivatehead, exact optimizer/RNG/data-cursor checkpoints
  and reset-only stage transition; direct EAGLEA1, sign-clone removal and charged
  paired feature-transfer reuse. Full adapters/transitions/initializers pending.
- Pipeline `b4a1031,87707ce,e8e7b45,67040f0,fa0ce7d`: supervised detached
  stages; exact source/config/artifact pins; source-aware retry/resume and
  process/kernel identities; context/DXG/resource return before export/eval;
  fresh native evaluator/report builder; sealed-final prompt reads deferred;
  explicit90second cleanup allowance; EAGLEactivation env derives from audit
  after ambient flag clearing. Actualbundle cannot grantready withmissingdata.

Independent QA ran current native export/loader/graph/teacher10methods PASS,
including malformed/dense-shadow/paired-geometry and realCPU STOP/reap. Expanded
backend source oracle260/260 AppleM3Max CPU cases PASS, including actual
K2560/4096/7680/9728/12800, bits1/8, one/seven-token forms. Correctfilter is
`-o W1A1_MUL_MAT`; symbolic lower-case filter returnedexit0/zero cases and is
rejected. Source-based filter inference was corrected before anyGPU use.
Root integrated61focused tests PASS(twoLinux-only skips), then30pipeline/EAGLE
preflight tests PASS(twoLinux skips). Author39data/fusion/portability tests and
20existing fusion regressions PASS; independent18contracts/failure fixtures PASS
(oneLinuxcheck skipped). Final integrated/full applicability sweep is pending.
CPU/synthetic evidence does not certify realblocktrajectories, CUDA or quality.

Actual saved TRAIN EAGLE CPU fusion artifacts were fit/hashed independently:
256fit/128validation rows, previous nativeRTX5080capture ancestry; newM3CPU
calibration. AllfourNPZ/source/receipt/report SHAjoins passed independent QA.
A8 row-orientation rescue rawvalidationRSE.07450→.05027, butpostnorm
.9582→1.0766; A1 raw.74985→.64237, butpostnorm1.2852→1.6850. Rescue remains
off by default, no nativequality/throughput claim. Historicalvalidation lacks
code, so all-domain final calibration remains PENDING.

Astra found unit±1 fitlatents preserve hardforward butchange AdamW sign inertia.
Actual EAGLEreference init is±.5 (not source magnitudes); block newbaseline
sourceweight magnitudes. Training/data owners bind explicit calibrated hard
signs+scales to reference magnitude policy, with unit as a named off-default
recipe. Negative-zero sign preservation must be rejected or explicitly floored.
HistoricalfourNPZ artifacts preserved with original unitpolicy; no silent
recipe promotion or post-resume reinitialization.

Fresh SM120 orchestration owned by root in new `nine_model_admission.py` and
`admit_nine_model_sm120.py`; source-only implementation/failure fixtures in
coordination worktree pending final API join. It serializes actualkernel/native
model/zero-updatebackward+memory/boundedcaptureportability producers, and emits
source/config/bundle/GPU-bound per-candidate admissions. Fixture receipts cannot
grant production readiness. Block/EAGLE capture-portability producers are
implemented in dataowner followups, actual three-domain goldens missing.

New subagent `/root/data_fusion/capture_orchestration` (Solhigh) owns only
`scripts/capture_nine_model_train_data.py` and focused tests in its ownworktree.
Native owns original prompt native-tokenizer/greedychain generation; data
owner owns materialization/selector/storage ancestry. Existing token-only
producer was insufficient to make production chains from originalTRAIN text.
This is an explicit remaining artifact/source integration dependency.

No human2080Ti availability reply yet, no remote tmux/job/context exists, no
GPUoperation or real-modeloptimizer update. RTX5080 remains paused/unqueried.
Next: land completed training/data/native proof fixes, join root freshgate and
resolved six-config generator, independent Opus/QA reviews; once2080opened,
serial actualmodel CUDA/zero-update checks, nativeTRAIN captures and goldens.
Human still owns coverage/recipe/exposure/wall/eval budget. No readyclaim yet.

## RTX5080 development dispatch — direct human steering

Same operator dispatched after the direct availability announcement. Local
`agent_env.py resume rtx5080` succeeded; no root remote command was issued.
Operator starts one-shot fresh resource/identity/artifact/toolchain inspection
while source owners finish matched generation/trace/initializer/config commits.
Development runs use reviewed published source; never execute mutable author
worktrees. Real-model optimizer updates and quality/evaluation remain zero.

All native/training/data/pipeline/QA owners were notified. Once fresh admission
passes, prioritize actual source CUDA compilation, deployed-width binary operator
oracles, Linux STOP/resource/resume fixtures, authentic bounded balanced TRAIN
capture and zero-update model/memory checks. The prepared campaign remains
PENDING until those artifacts and independently tested production contracts exist.

## Mac-only continuation checkpoint

Current published main658bead, native50ca2676. Integrated75nine-model source
tests PASS(twoLinux-only skips),55block/capture tests PASS and ownedRuff PASS.
Independent QA actual source-loader checks caught strict Python-vs-GGUF F32
epsilon mismatch;658bead canonicalizes exactF32 metadata and adds realGGUF
round-trip tests without broad tolerance or changing model math.

All remote development is now paused. Local followups: training8a16cea
curriculum/normalized smoke contract/audit reuse, pipeline final-counter/sampler/
paused-start fixes, root fresh-admission API join, independent ledger and applicable
full suites. Native/data/training are addressing a concrete finite-subnormal
A8 reciprocal-overflow risk with boundedCPU tests, explicit revised arithmetic
and unchanged normal-domain behavior; CUDA behavior remains unverified.

Actual released DSpark/DFlash weights are missing locally; native owner may
materialize exact public source-pinned snapshots outsideGit and exercise actual
CPU export/load/graph with synthetic activations. Heavy CPU jobs must be serialized
against Mac memory. No quality/throughput or SM120 claim follows from CPU evidence.
Native Goal remains active; the human paused GPU usage, not source preparation.


## Actual-model Mac milestone and first compaction checkpoint

Published main `01a24db`, native `624f50e74`; root coordination rebased onto
that integrated source. No unmerged worker artifact or original overnight
untracked record was removed. Native Goal stays ACTIVE; Mac-only boundary above
is authoritative, with no remote query after operator closure.

All source packages are integrated: normalized zero-update smoke/stages,
calibrated reference-magnitude initialization, optimizer save/load validation,
source/config/producer-bound fresh admission and captured-prefix replay. Final
root nine-model discovery ran140 tests OK with5 Linux/device-only skips.
Independent QA refresh is assigned after source quiescence; full-repo stopped
research-reference failures and pre-existing Ruff debt remain separately recorded.

Native original-model Mac evidence is published in
`experiments/nine-model-qat-preparation/native-cpu-artifacts.md`: four actual
FC16 CPU export/load/graphs and representative DSpark A8/frozen F16 target
co-load, separate private head/embedding pointers and exact geometry/taps.
The machine-readable summary is
`models/nine-model-original-snapshots/actual-cpu-summary.json`, SHA256
`d484a70ab08e3f9a397725bd6edf175d6a66a8a4616c5b27f55c0023575d2628`.
Fresh BF16 conversions do not replace inaccessible frozen block Q4 controls.
Data owner extracted actual BF16 FC/F32 gamma references with source/epsilon
pins; fitting awaits authentic five-tap TRAIN captures. Historical EAGLE fitting
shows raw improvement but postnorm degradation for orientation rescue, so rescue
remains off by default. CUDA finite-subnormal arithmetic is still unverified.

Training owner completed all4 actual original block CPU cells (DSpark/DFlash,
A8/A1), full private heads/16 binary projections:32/32 selected gradient tensors
finite/nonzero, later-state/noise K/V gradients nonzero, parameters unchanged,
optimizer updates0/moments empty. Synthetic features/labels only. Peak RSS
14.266–15.050GB stayed below16GiB; minimum available host RAM12.811–15.331GB
exceeded4GiB live floor. Each cell took7.29–8.47seconds. PTYs16269/51603 exited0,
owned processes/helpers absent; heavy slot released. Reports are under ignored
`results/nine-model-qat-preparation/actual-cpu-training/`; owner is publishing
guard/structured report and exact hashes, not a real-training admission.

Next bounded work: native owner uses the released heavy CPU slot for actual
FFN15-only graphs, sequentially with prior caps; training publishes its results;
QA refreshes exact source/artifact joins and requirement ledger. Root reviews,
independently tests, integrates/pushes and prepares final Mac checkpoint. Missing
production captures/calibration/frozen control files, fresh hardware checks and
human coverage/recipe/long budget remain PENDING; no completed-goal declaration.


## Mac data continuation and prelaunch gate correction

Integrated/pushed main `b17d9c7` includes the actual block CPU training guard
and four-cell report. Root independently ran69 block tests PASS,140 nine-model
tests OK(5 Linux/device-only skips),98 QAT/export/fusion regressions PASS;
new guard Ruff/format and diff-check pass. Four actual raw CPU report SHA256s
and all core source pins match integrated source. Logs remain ignored in
`results/nine-model-qat-preparation/root-final-mac/`.

Native completed four additional actual FFN15-only CPU graphs, both families
and A1/A8: exactly15 selected I32/W1 operations, original dense BF16 FC bytes,
correct bits/MASK/seven noise slots, finite full-vocabulary outputs. Six stages
exited0 with owned groups empty; export peak6.93GB/graph peak2.29GB within8GiB.
Source unchanged; report publication pending. Separate summary SHA256
`5868db9d37ed55168150c9ef516ee74b530df72ca56d27f47b277643113cdbc6` at
`models/nine-model-original-snapshots/actual-cpu-ffn15-summary.json`.
Heavy CPU slot is released. Training/operator/data-capture merged worktrees
and branches were retired after checkpoint, patch-equivalent rebase and needed
raw-log preservation; other teams' worktrees remain untouched.

Root/pipeline review found a concrete prelaunch deadlock: bundle builder gated
on aggregate profile PASS, while QA aggregate completion also requires trained
checkpoints/final exports/evaluation. Pipeline is correcting this to explicit
strict prelaunch_status plus portable initial-model/data/export/source evidence;
aggregate campaign results remain independent PENDING. Missing prelaunch status
refuses; current missing production inputs are not waived. QA coordinates schema
and negative tests. Fresh SM120 remains an authorized runtime gate.

A read-only data-owner audit established that NativeTeacher(gpu_layers=0) and
raw importer support authentic frozen-F16 CPU TRAIN captures; only the full pilot
controller/portability execution deliberately require CUDA. Root authorized a
separate explicit development_CPU path to finish useful Mac artifacts without
relaxing default CUDA/readiness gates. Data owns implementation/tests and will
publish source and exact plan for root review before any heavy launch. Bound:
9 original prompt-disjoint pilot chains plus6 separate CPU goldens, prompt≤512,
new≤32/chain≤544, retained≤8GiB, RSS≤12GiB, prelaunch available≥12GiB/live≥4GiB,
wall≤1800s, one persistent target-only CPU producer, STOP/failure close/reap.
No optimizer updates, quality evaluation, GPU/SSH or changed controls. If capture
succeeds, fit actual block A8/A1 FC references for both families separately, with
rescue off and≤96 calibration rows/1GiB workspace, only after producer closure.
CPU artifacts/goldens never grant CUDA/SM120 readiness. Source/capture progress
and failures must be preserved; pipeline/QA defer final source pins accordingly.


## Published CPU capture implementation and launch-review boundary

Main `a28cb36`/native `624f50e74` are pushed. Prelaunch/campaign circular gate
and synthetic-scope loophole fixed in `32f514c,bd28e17`; actual preparation
requires explicit prelaunch evidence while aggregate training/quality stays
separate. Real draft inspection exits0/PENDING, noGPUquery/no updates, with
62 explicit missing dependencies; inventory/interim receipts remain in ignored
`results/nine-model-qat-preparation/final-draft-packet/`. Final QA snapshot waits
source quiescence, now announced to QA/pipeline.

CPU capture source `df36c95,f016001,11faf22,a28cb36` adds explicit Mac-only
`--development-cpu`: default CUDA plan/execution guards unchanged, distinct CPU
 golden schema, original TRAIN/content/role/runtime joins, exact native replay,
2second RSS/free/wall/STOP watcher and15second typed progress. Root/author tests
include a truly blocked8second fixture request interrupted by real SIGINT in
<4seconds; producer closed, FAIL report retained, watcher joined/no late signal.
Final capture suite33 tests PASS. QA additionally rejects CPU receipts/goldens
as SM120 readiness and checks backend-library closure independently.

CPU build is pinned: CPU ON, all device/BLAS/backend-DL off. Exactly four project
libraries (llama,ggml,ggml-base,ggml-cpu) and exact canonical loaded paths required;
no extra backend/project copy accepted. Native reconfigured/rebuilt only teacher
against clean624, exit0; old packet/binaries/libs backed up. Fresh packet
`runs/nine-model-native-cpu-20261004/cpu-native-packet/rebuilt-624/packet.json`
SHA256 `9782bb05e7fa43acb54471f5ad32519ee8673be13676556734c7fa7b07c78c1a`.
Teacher SHA b7ab98ae..., base dylib SHA cb5f310...; actual loaded-path proof will
be checked at capture. Native Mac Linux-map-unchecked status remains unchanged.
Native build/worktree must stay intact until capture finishes.

Data owner now uses `/private/tmp/nine-model-qat-20261004/data-cpu-pilot`, branch
`prep/nine-model-data-cpu-pilot`; old data-fusion tree remains clean/checkpointed.
Actual pilot plan is under main ignored
`results/nine-model-qat-preparation/development-cpu-pilot-plan-20261004-01/plan.json`,
SHA prefix36306ad4,9 existing original TRAIN rows/3domains×3roles,6 separate CPU
 golden requests. Frozen F16 target/KV and original native chat template; no
optimizer, drafter-quality, held-out or GPU work. Bounds remain9+6 requests,
512prompt/32new/544chain,8GiB retained,12GiB parent+producer sampled RSS,
12GiB prelaunch/4GiB live available,1800seconds,1 persistent target-only CPU
producer. CPU available metric is vm_stat free+inactive+speculative pages;
RSS aggregate is an explicit conservative guard, not allocator peak.

Root reviewed all source; no target has loaded yet. Data must run real plan-only
inspection on publishedmain and report exact argv/cost/resource snapshot before
root heavy-slot GO. Once approved, update progress/cleanup/artifact pins and fit
bounded actual A8/A1 block FCs only after producer close. GPU pause stays in force.

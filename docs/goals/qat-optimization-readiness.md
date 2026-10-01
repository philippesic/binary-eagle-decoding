# QAT optimization readiness

## Objective and boundaries

User requested on 2026-10-01 via start-goal-team: implement and fully test faster
QAT computation, batching/cache construction, binary-weight optimization,
learned A4/A8 clipping and A1 thresholds, and curricula/trajectory refresh,
until ready for real GPU QAT. This is the one active project goal, continuing
Phase 1 of the one-bit plan. The previous [body/head goal](recurrent-binary-body-head.md)
retains historical experiment/data ancestry and preparation job ownership.

Preparation only: no real-data optimizer updates, no sealed-final reads, no
changes to frozen target/verifier or existing experiment source/config identity.
Tiny synthetic optimizer tests are permitted. RTX2080Ti remains paused.
RTX5080 belongs to the existing preparation operator in chat
01a0f47a-e246-75e1-a299-fcac42d34f8a; new GPU checks require explicit ownership
coordination and idle proof. Do not modify or stop that job from this goal.

## Completion evidence

- Computation: remove duplicate A1 forward work and validate custom backward,
  hard/native sign and scale order, zeros/subnormals, attached recurrent gradients.
- Cache/head: chunked K/V-only context and per-chain head batching, with exact
  token/feature/position ancestry, F16 cache casts, masks and detach boundaries.
  Investigate larger batches using measured memory/time; preserve update cadence
  by default and label experiments that change it.
- Binary optimizer: selectable initialization, LR/STE/optimizer controls,
  persistent-sign diagnostics, strict optimizer ownership and exact resume.
- Learned quantizers: effective trainable A4/A8 clipping and A1 thresholds;
  explicit checkpoint/export/native support or fail-closed deployment boundary.
  No simulation-only quantizer can receive native-ready status.
- Curricula: bounded A8→A1 transfer, depth weights, staged precision, fresh
  trajectory ancestry; deterministic resume and common-budget accounting.
- Independent CPU suite and integration review; actual RTX5080 bounded
  forward/backward, native export/decision gates and memory/timing evidence for
  enabled deployment options, without real optimizer updates. Preserve failures.
- Tested commits integrated/pushed on main, worker worktrees cleaned only after
  verification. Publish report and launch runbook with remaining limits stated.

100% ready is a testable gate, not an accuracy or speed guarantee. Q4_0 EAGLE
remains the acceptance/latency/throughput comparison target. Learned options may
require explicit experimental deployments; no unsupported feature silently
falls back to a different recipe. Measured GPU savings and quality improvements
must not be inferred from CPU tests or arithmetic counts.

## Initial team and ownership

Checkpoint baseline: main 5676254. Source audit:
experiments/training-optimization-audit-2026-10-01.md.

Planned independent temporary worktrees:
1. Computation owner: recurrent_qat.py A1 autograd implementation and its tests.
2. Cache/head owner: recurrent rollout/provider and native Torch adapter/tests.
3. Binary/curriculum owner: new optimization/curriculum modules and tests.
4. Learned-quantizer owner: new quantizer module, export contract and tests.
5. Luna: independent CPU integration checks and GPU readiness plan; no GPU use
   until assigned after existing preparation owner releases it.
Root owns continuous integration/config/checkpoints, reports, goal/status.
Every worker stops at a reviewed bounded deliverable, records checks/findings
under experiments/qat-optimization-readiness/, and commits its own files.

## Current state

Implementation is integrated and published on main eb66093; 952 CPU tests
pass with four skips. Native8025a0777 is published and CPU-tested. Both GPUs
remain paused; actual CUDA gates and complete data preparation are pending.
See the final integration checkpoint below. No real-data optimizer updates.

## Added scope: tiny fusion correction

Companion chat 01a0f90d-acc2-77b3-a8e5-383695f54087 forwarded the human's
explicit implementation request on 2026-10-01. Add one optional rank-1/rank-4
feature-fusion correction from raw pre-quantization features, preserving the
binary core. Label F16 factor storage/F32 accumulation as mixed precision.
Provide a bounded fusion output-bias control, train-only provenance-bound fitting
entry point, parameter/target ownership checks, exact resume, versioned native
export/load/execution and later state/K/V gradient tests. No real calibration
fitting or GPU ownership change is authorized by the forwarded scope.
Worker /root/fusion_correction owns new module/tests/CLI/report only; native
owner will add export/runtime support after its learned-quantizer commit.

## Team launched

Checkpoint f502d4a pushed before workers. Native subagents: /root/computation,
/root/cache_head, /root/recipes, /root/learned_activations, /root/native_quantizers,
/root/validation (baseline completed), /root/advice (focused correctness),
/root/fusion_correction. Feature owners are GPT-6.1 Sol/high; validation is
Luna/high; focused advisor Astra/medium. Root integration worktree:
/private/tmp/eagle-qat-integration, feature/qat-integration. Native standalone
worktree /private/tmp/eagle-native-learned, feature/learned-w1ax. CPU baseline:
114 passing tests, Apple M3 Max CPU, Torch 2.14.0/NumPy 2.4.6/Python 3.11.15.
No new GPU checks or real optimizer updates.

## CPU implementation integration milestone

In isolated root worktree, integrated computation 1e8a012, cache/head 84fae13,
learned activation be824f6, binary/curriculum helpers aeed6fe, raw fusion
correction dc0ed58 and native/export learned support parent3e92989. Native
cce962891 is published on the user's fork feature/learned-w1ax. Native CPU
187 operator cases +24 EAGLE load/encode fixtures passed; CUDA not yet compiled
or run. Correction native/export and runnable staged curriculum are finishing.
Root added explicit config/optimizer families, deduped learned sharing, config
source identity, schema3/4 checkpoint payloads and canonical alias/frozen-operand
resume validation. Eight combined recipe integration tests pass, including
exact uninterrupted/resumed learned+correction optimizer state. One tiny
learned-A1 fixture has legal exact zero gate/up scale gradients: smoke correctly
rejects it; a separately seeded nonsingular fixture passes without relaxing gates.
No real model optimizer updates/GPU/finals/calibration fitting occurred.

Existing preparation owner reported healthy supervisor07 after disconnect at
2026-10-01 20:44:55 UTC; --prepare-only in cmdline, retained327manifests,
10,000train/224dev andzerooptimizerupdates. This is historical observed state,
not a fresh GPU-release claim. Ownership remains with that operator.

Next: finish runner/correction support, independent whole-package review,
config/readiness launch gating and actual GPU no-update validation after ownership
release. Learned or mixed-precision options cannot inherit the old frozen source
readiness silently; their new source and native contracts need fresh gates.

## Added scope: asymmetric binary weight codebooks

Companion chat01a0f90d-acc2-77b3-a8e5-383695f54087 forwarded explicit human
authorization to implement per-output-row trainable weight midpoints on
2026-10-01. Representation w_ri=mu_r+alpha_r*sign(z_ri), alpha>=0, mu initialized0.
Opt-in fusion-only and all-nine coverage, separate midpoint LR and mild
regularization; same quantized input as the binary term. Native arithmetic:
(D*alpha)*beta+(S*mu)*beta before bias, with exact same-code S shared per input
boundary; A16 uses the F16 boundary values and beta1. Do not materialize a dense
center matrix. Versioned schema5/native admission, midpoint permutation,
checkpoint/resume, zero-scale/input and later-state/K/V/ownership tests required.
This is affine binary-core weights, distinct from learned activation thresholds
and the raw-input rank correction. No training-quality or speed claim.

Worker /root/affine_weights owns only new module/tests/report10; root owns
training/config/state integration; /root/native_quantizers owns sequential
native/export support. Existing GPU preparation remains untouched and soleowner.

## Validation and admission milestone

Combined CPU source now includes runnable curriculum85e0ab6 and native correction
parent ea6d60c. Published native head dc6d178b5 includes learned-only CUDA
subnormal arithmetic fixes; GPU compilation still absent. Independent Luna
review found and root repaired boolean numeric/accelerator-option acceptance
and a zero-update guard that previously ignored nonzero Adam/SGD moments.
Weighted depth CE no longer computes and discards a duplicate CE. Broad suite
858tests had20 baseline-only fixture errors:8 unsafe fake process-groupcalls and
12missingignoredpromptfixtures; same failures independently reproduced on main.
Luna owns isolated test-only repairs; assertions and capture behavior preserved.
A CUDA readiness receipt validator and full-model zero-update profiling harness
are being integrated, binding source/runtime/recipe/native/hardware and actual
pack/decisions/memory evidence. CPU proof cannot authorize CUDA. Provider/A4 and
refresh validation are being extended with new independently validated schemas;
old source/readiness cannot silently admit new recipes.

## Shared GPU pause checkpoint — 2026-10-01 23:15 UTC

Existing preparation owner in chat01a0f47a-e246-75e1-a299-fcac42d34f8a executed
the human “Pause gpu”. Supervisor07 gracefully interrupted/exit0; verified
23:15:28UTC both owned groups1204/1205 absent,no project processes or GPUcompute
apps. Retained10,000train/224dev across327complete manifests,zeroQATupdates;
preparation incomplete. Both shared hostpause flags and existing monitor PAUSED.
GPU ownership may not be reassigned from this idle proof while humanpause is
active. CPUteam work can continue; GPU use requires NEW human resume/ownership
coordination. Existing stop-before-QAT boundary and all frozenidentity preserved.
Exact evidence and resumeprocedure remain in the prior body/head goal and
ignored pause-preparation-only-20261001T231528Z.json monitor snapshot.

## Full CPU suite and user GPU pause

The integrated package in /private/tmp/eagle-qat-integration passed the full
CPU unittest discovery:924tests,4expectedskips,104.080seconds,AppleM3Max,
Python3.11.15/Torch2.14.0(CPU-only)/NumPy2.4.6. Rawlog:
runs/qat-optimization-readiness/cpu-full-suite.log in that worktree (ignored).
Subsequent strict curriculum and workflow changes have targeted green checks;
final whole-package rerun is pending those last tooling commits.

Native published head524427ed3 supports explicit --backendCUDA with fallback
rejection. CPU native proof:220operators,108loaders,53encodergraphs and33exact
packfixtures; GPU compilation/runtime is still absent. Integration branch
feature/qat-integration is published through0244308. No real-data optimizer
updates,real calibration fitting,remote/GPU work or sealed-final reads occurred.

Root integrated canonical μ/activation/factor state, exact paired/curriculum
resume, schema2–5 exports/providers, independent A4 admission, typed options,
unchanged default cache/head controls, all-trainer measured CUDA admission and
complete schedule gates. Fixed raw correction addition order to native base+
raw_delta+bias, and separated detached/attached shared-sum cache keys. Learned
and affine tiny multi-bit inputs use the explicit safe normalization rule.

User GPU pause checkpointd172f42 takes precedence. Both pause flags aretrue;
supervisor07 terminal/released23:15UTC,10,000train/224of1,002dev retained,
zerooptimizerupdates; corpus preparation remains incomplete. No automatic
resume or CUDA launch is permitted until the human explicitly resumes. The
earlier async GPU scheduling question is superseded by this pause. CPU-only
implementation and test work continues.

Last bounded workers: /root/computation deliveredb846fd9 curriculum receipt
producer(11CPUtests); /root/cache_head is finishing real-measurement native
evidence collection/generic stage support; /root/native_quantizers is adding
actual JSON fixture reports on CPU only. All earlier workers completed and their
commits are integrated on the published branch. Root must finish review/full
CPU rerun, integrate/pushmain, preserve unmerged worker outputs and clean only
verified integrated worktrees. Then checkpoint pending actual CUDA/native
decision/memory/timing gates without marking the goal complete.


## Final CPU implementation milestone

All workers completed and root integrated their owned changes on
feature/qat-integration through e050d74, followed by final source binding and
formatting. Native gitlink8025a07773b7828bdeb4f3e0b834c8b54cb65c66 is already
published on the user's fork feature/learned-w1ax. Main integration/push follows
the final CPU rerun; no native gitlink points at an unpublished commit.

Full suite952tests/fourskips/104.553s passed on AppleM3Max with CPU-only
Torch2.14.0,Python3.11.15,NumPy2.4.6. Final formatting/source-inventory rerun passed952tests/fourskips/104.517s;
Ruff and whitespace checks pass. Outcomes are recorded in report11 and ignored runs/qat-optimization-readiness.
Native CPU220operators,57exactbytepairs,36projections,114loaders,59encodergraphs,
218auditedCPUarithnodes pass. Native actual JSON is correctly rejected as CUDA
proof. Strong every-stage curriculum smoke, real-evidence collector and paired/
complete-schedule receipt producers now bind the entire deployable recipe.
Report11 supersedes interim worker test/smoke counts; no performance gain claimed.

Remaining: actual CUDA compile/operator/exact-pack/nonzero-option/native-decision
and full-shape backward/memory/timing gates, with fresh eligible train ancestry
and every enabled stage. Both GPUs remain paused by the human; no queries or
remote actions occurred. Existing preparation remains incomplete at10,000train/
224of1,002dev/327manifests/zerooptimizerupdates. The scheduling question is
superseded; wait for NEW human resume, assign one operator, then follow runbook.
Goal remains active and incomplete; GPU-resource pause is not a whole-goal pause.


## Published main and cleanup checkpoint

Implementation main **eb66093** is pushed; native published head is8025a0777.
Final CPU rerun952tests/fourskips/104.517s and changed-Python Ruff/whitespace
checks pass. Main and the primary native checkout are clean. Reviewed worker
history was recorded in d10a228 before cleanup, preserving all owned commits.
Nine worker worktrees and root integration worktree were removed after
ancestry/cleanliness checks. Logs were copied into the main ignored
runs/qat-optimization-readiness directory before removal. Native JSON and logs
are preserved under its native-cpu subdirectory with SHA256 inventory; JSON SHA
c9bb1abf6fc10555c81b39b7efdbd66f5dc24c27e0e7690e12e5bef19c827076.

Two independent read-only reviews confirmed the recipes worktree had no unique
uncommitted owned work. Its15borrowed dependency copies, binary diff and SHA256
inventory were preserved under ignored runs/qat-optimization-readiness/worker-archives
before removing that merged worktree. All parent worker branches were retained
in main history before removal. Native worktree
/private/tmp/eagle-native-learned and published feature/learned-w1ax remain for
pending actual CUDA validation; CPU build /private/tmp/eagle-native-learned-build
and source are preserved. No workers own remote GPU jobs. Existing unrelated
worktrees were left intact. GPU pause flags stilltrue; no resume is authorized.

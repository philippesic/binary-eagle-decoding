# QAT optimization readiness

## Objective and boundaries

User requested on 2026-10-01 via start-goal-team: implement and fully test faster
QAT computation, batching/cache construction, binary-weight optimization,
learned A4/A8 clipping and A1 thresholds, and curricula/trajectory refresh,
until ready for real GPU QAT. This is the one active project goal, continuing
Phase 1 of the one-bit plan. The previous [body/head goal](recurrent-binary-body-head.md)
retains historical experiment/data ancestry and preparation job ownership.

Latest human authorization: root owns actual validation and subsequent training
after the current preparation owner releasesRTX5080. Zero real-data optimizer
updates during validation; training may begin after all current launch gates
pass, without another confirmation. No sealed-final reads or changes to frozen
target/verifier or existing preparation source/config identity. Tiny synthetic
optimizer tests are permitted. RTX2080Ti remains paused. The earlier
preparation-only boundary is historical; see the latest handoff checkpoint.
RTX5080 belongs to the existing preparation operator in chat
01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed; new GPU checks require explicit ownership
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
pass with four skips. Native8025a0777 is published and CPU-tested. RTX5080 is assigned to the existing corpus preparation resume after the human
continued it; RTX2080Ti remains paused. New implementation CUDA gates and complete
data preparation are pending.
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


## Completion audit continuation

Previous turn classified progress: implementation/main publication,952CPUtests,
native CPU proof and verified cleanup. This turn revalidated main2ff983c/native
8025a0777 and actual CPU CLI planning for all12profiles plus standalone curriculum
and native collector. All plans explicitly remain GPU-readyfalse with zero
updates,no provider/model/native/CUDA work. Generated configs,plans,hash inventory:
main ignored runs/qat-optimization-readiness/completion-audit-bc6d012367/audit.json.
Requirement-by-requirement report12 separates CPU proof from missing CUDA,
full-model backward,native decisions,memory/timing,larger-batch measurements and
fresh every-stage providers. No completion claim or real training-quality claim.

Shared host pause flags re-read and stilltrue. Same human GPU pause blocker on
consecutive goal turn2(counting original implementation turn); no live process
wait. Current goal remains active. No remote query,host action,model/data/final
read or real optimizer update occurred. Next meaningful endpoint action requires
NEW human GPU resume and sole-owner assignment; do not automatically restart.


## Blocked checkpoint

The previous turn was progress: actual preparation/planning CLI checks for all12
profiles and published requirement auditbd88554. This continuation revalidated
clean mainbd88554, the audit, native Goal state and both durable host pause flags.
The same human GPU pause has now blocked required actual CUDA evidence across
three consecutive goal turns, counting the original implementation turn. No live
job handle is being waited on; no remote query or experiment occurred.

Native Goal tool returned **blocked**. CPU implementation,952tests/fourskips,
native CPU fixtures and executable plans are preserved. Completion remains
unproven for CUDA compilation/packing,full-model backward,native decisions,
memory/timing,larger-batch measurements and every-stage fresh eligible providers.
There is no remaining meaningful independent CPU action toward those measured
requirements. The scope and no-real-data-update/final/precision boundaries remain
unchanged. Resume only after NEW human GPU authorization and sole-operator
assignment; start a fresh blocked audit if the goal is resumed. Runbook and
report12 give the exact next gates. No automatic GPU/monitor restart.

## Existing corpus preparation resume reservation — 2026-10-02 05:15 UTC

Human in preparation owner chat01a0f47a-e246-75e1-a299-fcac42d34f8a requested
“Continue. Also why have you not made more prompt gen progress”. OnlyRTX5080
sharedpause flag cleared;2080Ti paused. `/root/resume_prep_after_2315` has sole
resume/preparation ownership, SAME frozen originaldata/maths/config/runtime and
--prepare-only stop-before-QAT boundary. New optimization-feature CUDA checks
must await separate ownership coordination and verified idle release; no local
new feature/source bytes may be pulled into the frozen corpus preparation.

Fresh preflight/CPUpartial preservation/new unique supervisor+hostsession and
postdisconnect proof underway; monitor remainsPAUSED until verified. Retained
10,000train/224dev in327completed manifests,zerooptimizer. Code review of the
original stage loop confirms everyresume starts at firstcapture and audits
existing label manifests; it skips native recapture but has no saved auditcursor.
Previous supervisor07 ran20:43→23:15UTC and reachedreaudit220/353, before newdev
capture. Human was told the restartcost; no data loss or newtraining claim.
No gate/identity weakening to skipchecks is authorized. Keep GPUreservation
until sameprep pipeline readiness/exit and verified release; actual new feature
GPU readiness remains unestablished.

## Existing preparation resume verified — 2026-10-02 05:32 UTC

Human continuation remains preparation-only,no QAT optimizerupdates. Sole Luna
`/root/resume_prep_after_2315` completed freshpreflight,CPUpartialpreservation,
one new supervisor08 launch and one disconnect/reconnect proof. Fresh21checks
passed: old07terminal/groups1204/1205gone,no ownedprocesses/computeapps,RTX5080
SM120free12,749MiB,MemAvailable19,755,324KiB,disk424,602,206,208B. Original frozen
parent7547d253/nativeb4e366d4,eight exactmath/config/stages/runtime/resolved/launcher
SHA identities match. Only launcher control delta remains allowed. No latest
local optimization-feature source/native bytes deployed to the old data run.
CPUrecoverpartial exit0 at05:26:49UTC,records:[],processes_started:false,
raw_artifacts_deleted:false; no quarantine/deletion or new experiment.

Same pipeline launched05:28:08UTC via remote_job300secgrace, NEW supervisor
`luna-supervisor-a8-a1-native-order-20261002-08`, host socketbinary-eagle-runtime /
session`continuous-a8-a1-native-order-20261002-08`, SAME experiment dir
`runs/luna-continuous-a8-a1-native-order-20260930`, child --start --allow-cuda
--resume --prepare-only. Firsthealth05:29:26UTC passed at1/353. FirstLOCAL
transportclosed, ONCEfreshLOCALMacBookreconnect: CPUhealth **05:31:39.846713UTC**
passed at4/353,independentownership **05:32:52.175302UTC** passed12checks and
reauditadvanced6/353. Hostserver673,supervisor674/PGID674,child676/PGID676;
live --prepare-only verified. Retained327complete manifests /10,000train /
224dev,optimization_started:false,steps:{},readyreceipt:null. Auditordinalreset
is not lostdata or newlycapturedprompts. BothLOCALtransports closed/absent.
HostGPU jobremainsrunning;2080Ti remains paused. No failure-budget charge.

Ignored rawoperation `runs/luna-continuous-a8-a1-20260929/resume-operation-20261002-after-2315.json`
contains authorization,fullcommands/rawpreflight/recovery/launch/health/ownership/
cleanup (SHA256e77fd345prefix). Current registration canonical08state/hosttmux,
monitor queryJSONSTRINGlength2,983 and binding matched. Querysource ONLY statepath
07→08; sourceSHA256
6e8b3470a3b3138cc2949101c47024d22318f07c0972281031d44f1276511f92.
Parent last-health normalized to remote CPUchecktimestamp,ownership timestamp
separate. Existing monitor reactivated via native tool with SAMEprep-onlyprompt,
ID,target,15mincadence. Prepared terminal endpoint must have zero global/A8/A1
steps+readyreceipt and ownedgroupsgone/GPUreleased before notification/pause.
No QAT start,frozenmath/runtime/config/data/precision/nullcaps/finals unchanged.

Userasked why promptgeneration hasn't advanced. Code read of original
scripts/w1ax_continuous_stages.py confirms existing nativecaptures aren't regenerated,
but audit_native_labels is called unconditionally for each completed manifest,
starting at firstshard on EVERYresume; no saved auditcursor. Previous07 run
20:43→23:15UTC spent about2.5hours auditing and reached220/353 before humanpause,
so new developmentcaptures remained224. Textpromptpool alreadyprepared; remaining
work is native response/features/labels and audit. User told this restartcost and
that it should have been clearer. No audit bypass/weakening/sourcechange made.

Existing preparation owns RTX5080 until ready/verifiedrelease or humanpause. New
optimization-feature GPU gates remain unverified and require ownership coordination,
not an automatic deploy into this frozen run. Same activeprojectgoal retained;
no new goal/architecture/experiment/schedule. Operatorcomplete,no activeLOCAL
transport. Docs-only checkpoint follows8971bb6,diffcheck appropriate verification.


## Human assigns validation and training ownership

Latest direct human instruction in this chat: “from now on you will be in charge
of training ... Monitor the other agent for when it releases the gpu ... then
start your validation and later training.” Root is designated validation/training
owner after current dataset preparation releasesRTX5080. This supersedes the
old no-real-data-update restriction ONLY after current actual validation gates
pass. No new goal created; full readiness objective persists, followed by the
new authorized training responsibility. Frozen target/verifier, sealed finals,
exact data/cache semantics and all validation gates remain unchanged.

Current registry re-read: RTX5080unpaused,RTX2080Ti paused. Existing preparation
owner retains sole5080assignment,supervisor08 recordedlive, --prepare-only.
Root will not modify or interrupt it or deploy new source into its frozen run.
A fresh bounded read-only Luna handoff observation is in progress, owning only
its separate local transport and ignored handoff-observation JSON; no GPUlaunch.

Native heartbeat **qat-validation-and-training-handoff** created and confirmed
ACTIVE every15minutes,targetthischat01a0f934-dd65-7e33-a5bf-0ba591e713a4. It watches
preparationchat01a0f47a-e246-75e1-a299-fcac42d34f8a usingcompactwaitcursor and local
registration; existingprepmonitor remainsuntouched. Silent healthy observation;
notifyverifiedrelease/failure/requiredaction, then start already-authorized
validation and training after allrecipe/source/native/hardware/memory gates.
Handoffphase/authorization/cursor saved in ignored
runs/qat-optimization-readiness/training-handoff-registration.json. Exact operator
sequence and latestauthorization live in docs/QAT_TRAINING_HANDOFF.md.

Training recipe will be selected/recorded from actual passing controls before
launch; noneselected orstartednow. No real-dataoptimizerupdates in this turn.
The old nativeGoalblocked status during the humanGPU pause is historical; durable
projectstate is waitingforcurrentpreparationrelease with monitoringactive. Goal
API provides no agent-owned resume transition; do not create a duplicate goal.


Fresh handoff observation **2026-10-02 06:03:52.658738UTC**: one existing CPU-only
checker invocation exit0/healthy,statuspreparing,phaseteacher_capture_audit,
reaudit52/353. Retained327completedmanifests /10,000train /224dev; no newdata
completion inferred from the auditordinal. optimization_started:false,steps:{},
readyreceipt:null. Currentregisteredowner stilllive: hostserver673,supervisor674/
PGID674,child676/PGID676. Release notproven. No GPU query,hostjobaction,recovery,
sourcechange,data/model/finalread oroptimizerupdate. Exactrawcheck/command/paths/
cleanup in ignored runs/qat-optimization-readiness/handoff-observation-20261002.json.
Lunaobservercompleted; its ownLOCALtransportclosedandabsenceverified. No local
command remains live. Nextscheduledobservation uses savedcursor/freshregistration;
only successful fullprep+terminalgroups+freshresourceidle proof advances to
newrecipeCUDAvalidation. Rootheartbeatconfirmedactive before checkpoint.


## User-authorized overnight research teams and usage supervisor

The human requested the three proposed long investigations and one single-agent
usage monitor every5minutes, extending useful research until <2%weekly allowance
remains. The existing QAT owner01a0f934-dd65-7e33-a5bf-0ba591e713a4 must continue
allnight and may use credits; this supervisor must never pause it or its GPU job.
Research must stop at exhaustion or reset and must not burn the new allowance.
This is prospective supporting research within the existing active goal, including
explicit authorization for reduced block-drafter research, not a live pivot.

Assignments, message/interrupt authority, CPU-only boundaries and cutoff policy
are in docs/OVERNIGHT_RESEARCH.md. Initial68%used/32%remaining; account reset
1791049896 conflicts with the human’s upcoming-morning expectation. Conservative
hard cutoff2026-10-02 16:55UTC is earlier. Registry and usage log will be retained
under ignored runs/overnight-research-20261002; supervisor launches leaders in its
own collaboration tree for direct stop control. Research-specific work is
isolated from live QAT/runtime source and sealed finals. Launch IDs and verified
control state will follow in the next checkpoint.


## Overnight protection verified

Incoming overnight coordinator message was checked against direct human text in
chat01a0fb34-e010-7b11-ac9a-f72cf2367c6c, not treated as standalone permission.
Human requires the QAT agent continue overnight,includingavailablecredits,while
research alone stops near exhaustion/reset. Rootheartbeat was updated IN PLACE
and native tool returnedACTIVE, preserving15mincadence/name/target/notifications.
Its protection applies to thischat,agents,ownedGPUjobs and monitor; all source/
recipe/data/frozenprecision/native/memory/ownership gates remain intact. No paid
credit purchase or usage-reset redemption performed orauthorized.

Read docs/OVERNIGHT_RESEARCH.md: three CPUresearchteams cannot change liveQAT
source/recipe orclaimGPUownership. Existing5080preparationowner stayssoleowner.
Compactwait snapshot advancedsavedcursor to b7452744-5e68-4e46-b168-50b5b2f76802:9;
owner'ssavedhealth06:13:55.765738UTC healthy/nonterminal,reaudit67/353,retained
10,000train/224dev,zerooptimizer. No releaseproof,newremotecheck,model/data/final
read,source/runtimechange oroptimizerupdate here. RootmonitorACTIVE; nextphase
stillwaitingforverifiedfullprep and GPUrelease,thenactualnewrecipevalidation and
alreadyauthorizedtraining. Ordinaryhealthywait remainsquiet; notifymeaningful
release/failure/requiredaction/actualoptimizerstart.


Overnight launch checkpoint: supervisor01a0fb45-5d1e-7bc3-a3a0-133d9942d822
launched three Sol/high leaders and has ACTIVE five-minute heartbeat
`overnight-research-usage-control`. All new research tasks currently await
sandbox approval for Git writes; unattended progress is NOT verified. Parent
requested the necessary app permission resolution and opened the supervisor.
Protected QAT task/15-minuteheartbeat remain active; owner publishedf11b3b4
with the human’s overnight/oncredits authorization. IDs, managed worktree paths,
stop controls and next action: docs/overnight20261002/launch.md. No permission
settings were changed and no GPU experiment was launched by research.


## Preparation owner rotation — 2026-10-02 06:42 UTC

This checkpoint transfers ONLY existing corpus preparation supervision from
chat `01a0f47a-e246-75e1-a299-fcac42d34f8a` to the fresh successor recorded
below and in ignored monitor registration. The one active objective remains QAT
optimization readiness; separate validation/training owner
`01a0f934-dd65-7e33-a5bf-0ba591e713a4` waits for preparation's verified release.
Existing frozen preparation must finish data, audit, readiness, coverage, paired
CUDA forward/backward smoke and initial checkpoint, then stop BEFORE optimizer
updates. The human's preparation-only instruction remains binding for this job.
No new goal, recipe, source deployment, experiment or optimizer launch is needed.

Completed: prep-only launcher commit95428e0c3f0daaa7c1a168d1cce62827e7c8c217,
integration5676254;21launcher+12health CPU tests, Ruff/diff check passed. Launcher
SHA25682f0185ab9ac38bc622749d2ca5e5c5a297a72494dcbf3d16d2b08df249cdade is the only
allowed tracked remote delta. Resume proof and audit-cost explanation were pushed
in79b8876. Broader implementation eb66093/native8025a0777 has952CPUtests/fourskips;
actual new-feature GPU gates remain unverified and belong to the other owner
AFTER release. This rotation changes documentation/ownership records only.

Latest single CPU health observation **2026-10-02T06:42:04.636798+00:00**:
healthy/nonterminal, preparing `teacher_capture_audit`, re-audit109/353;327
completed label manifests retain10,000train/224development prompts, zero optimizer
updates, no preparation-ready receipt. This ordinal is resumed audit progress,
not new captures or data loss. Every resume audits completed shards from the
beginning before unfinished dev capture; text prompt pool already exists.
Remaining capture target is1,002development prompts. No healthy GPU query was
made, so this is not an idle/resource-release claim.

Remote SAME frozen project `/home/philip/binary-eagle-decoding`, run
`runs/luna-continuous-a8-a1-native-order-20260930`; status `${run}/status.json`.
Current supervisor `luna-supervisor-a8-a1-native-order-20261002-08`, state
`/home/philip/binary-eagle-decoding/runs/luna-supervisor-a8-a1-native-order-20261002-08/state.json`.
Detached Linux host tmux socket `binary-eagle-runtime`, session
`continuous-a8-a1-native-order-20261002-08`:server673/supervisor674(PGID674)/child676
(PGID676). State pid/pgid mean child. Launch `remote_job.py` with300secondgrace,
`.venv/bin/python scripts/train_continuous_w1ax.py --start --allow-cuda --resume
--prepare-only --stages-manifest runs/continuous-preparation/stages.json --run-dir`
SAMErun. Ownership/live prep flag verified after one reconnect05:32:52UTC.
WSL20GB and instanceIdleTimeout=-1 unchanged. RTX5080 sole preparation assignment;
RTX2080Ti paused. Derive fresh host from ~/.config/binary-eagle-decoding/hosts.toml
and CURRENT ownership/paths from registration.experiment, never historical fields.

Ignored authoritative files in `runs/luna-continuous-a8-a1-20260929/`:
monitor-registration.json,last-health.json,recovery-budget.json,
monitor-query-source.py,monitor-query-command.json. Current query JSON is a
STRING, ASCII2,983characters, source SHA256
6e8b3470a3b3138cc2949101c47024d22318f07c0972281031d44f1276511f92;
binding08. Latest exact four-key raw snapshot
`health-snapshot-20261002T064204Z.json` validated command byte equality, stdout,
JSON and Git ignore. Resume evidence `resume-operation-20261002-after-2315.json`.
Recovery budget old24hwindow began2026-09-30T21:55UTC, used1/manualsupervisor03;
now expired but healthy checks do not reset it. Evaluate actual timestamps/fresh
pause flags before any authorized failure recovery; human resumes do not charge.

No live child agent/local command/transport remains: sole tick operator
`/root/health_20261002_0639` completed and local session absence verified. All
previous preparation agents are completed. GPU supervisor continues independently;
do NOT interrupt it for rotation. No unmerged worker output or owned worktree
requires transfer; root's docs-only temporary worktree is integrated/removed
before retirement. Other project/team worktrees and jobs are outside this owner.

Exact next actions: fresh successor read STATUS,this goal,AGENT_OPERATIONS and
CONTINUOUS_W1AX_MONITOR; inspect ignored registration/last-health/budget and
registry/pause flags LOCALLY, acknowledge that it can see current08 job and
prep-only boundary. Do not run another remote checker for this completed tick.
Parent transfers SAME existing ACTIVE15min automation
`a8-a1-health-check-enable-after-manual-start` to successor with unchanged prompt,
cadence/name/preferences and records native confirmation; no duplicate schedule.
Next due tick: exactly one experiment_operator Luna/high with fork_turns none,
no launch/recovery operator active; one CPU checker through fresh LOCAL tmux MCP
transport, full saved JSON command passed directly without printed/truncated
base64. SSH bridge uses Windows OpenSSH then wsl.exe -e python3. Salvage executed
LOCAL pane output if wrapper capture fails, without rerunning checker. Never
attach/send keys/kill HOST job tmux. Count completed prompt_count bytrain/dev,
exclude partial/gate cells. Persist last-health and keep ordinary progress quiet;
manual status direct compact read. No healthy extra logs/GPU query.

New failure: notify verified error+retained counts BEFORE any recovery. Only
allowed native transport failure/failedprocess or CUDAunknown sync after GPU
usable qualifies. Fresh pause/STOP/budget, same math/native/runtime/data/config/
precision identity, terminal supervisor/all owned groups gone and actual hardware/
resource floors are mandatory; CPU recover-partial preserves bytes. Persist
budget BEFORE at most one new supervisor/session launch per tick, SAMErun with
--prepare-only and300secondgrace; close/reconnect once and verify actual health.
Never retry gate/numeric/cache/data/eligibility/corruptcheckpoint/OOM/identitychange/
intentionalstop/ownership ambiguity. UnknownSSH is unknownhealth, not failed
training or freeGPU; notify once. No driver reset/reboot/source pull/finals.

Endpoint: stopped+preparation_complete:true+successful matching SHA readyreceipt,
zero global/A8/A1/Adam optimizer counters, checkpoint zero, full frozen coverage
and supervisor terminalexit0. Then fresh proof both owned groups and project
processes gone/GPU compute apps empty through LOCALtmuxMCP, before notifyingready
and pausing SAMEmonitor. Save release proof for separate training owner; do not
start its validation/training in this frozen checkout. Positive optimizer updates
are unauthorized: notify and gracefully stop verified owned supervisor (STOP and
SIGINT,300secgrace; never hosttmux panes/unrelatedwork). Explicit humanpause marks
shared flag and pausesmonitor immediately, then verifies graceful cleanup before
reporting GPUfree. Resume only new humanrequest. No unresolved human decision is
needed for existing preparation continuation; no new feature recipe is selected.


Preparation-owner transfer confirmed: successor chat
`01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed` completed local initialization, acknowledged
supervisor08/current paths/server673/supervisor674/child676,06:42:04UTC healthy
109/353,retained10,000train/224dev/327manifests,zerooptimizer/no receipt,
RTX5080unpaused/2080Tipaused and no new remote check. Native automation update
confirmed SAME `a8-a1-health-check-enable-after-manual-start` ACTIVE;
TOML readback verifies successor target and unchanged every15minute cadence/prompt.
Ignored coordination-rotation-20261002.json and monitor-registration.json record
acknowledgment/previous owner/native confirmation and current preparation owner.
The old owner now retires; no live subagent/command transferred and no GPU job
interrupted. New task reads checkpoint6285415 plus this confirmation; next action
is the next scheduled single CPU check. Separate validation/training owner stays
01a0f934-dd65-7e33-a5bf-0ba591e713a4 and must derive latest prepowner from registration,
not assume the historical old chat. Stop-before-QAT boundary unchanged.


## Validation owner follows current preparation supervisor

QATowner verified the acknowledged preparation rotation to successorchat
01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed against coordination_handoff and the existing
prepheartbeat target. It read both old and new compactwait snapshots, retained
per-chat cursors, and updated its SAME ACTIVE qat-validation-and-training-handoff
heartbeat to derive future acknowledged preparation-owner changes. Its own
trainingowner target/name/15mincadence/notification preferences are unchanged.

Currentowner'ssavedrawhealthsnapshot06:56:05.023881UTC proves healthy/nonterminal,
reaudit130/353,retained10,000train/224dev,zerooptimizer,readyreceipt:null and SAME
supervisor08running. No GPUreleaseclaim,duplicateremotequery,source/runtimechange
oroptimizerupdate. Per-tick evidence remains ignoredhandoff-observations.jsonl;
phasewaitingforfullprepandreleasedGPU. This internal supervision rotation does
not change the successful-preparation/actual-CUDA-before-training requirements.


## Human corrects overnight reset policy — October2

The human clarified that reset may be later than10a.m. and must be monitored,
not replaced by a time-only cutoff. Remove the conservative October2 09:55PDT
cutoff from ACTIVE instructions, heartbeat and control; legacy fields become
null. Continue useful research until <=1%remaining or the actual monitored
original-window reset, then latchstop without burning the refreshed allowance.
Track corrected reset estimates when usedusage has not reset; a timestamp change
alone is not enough to identify a new allowance. QAT task/monitor remainexempt
and mayuseavailablecredits. Parent updates the existing5minschedule in place,
not duplicate monitors. This supersedes cutoff language in historical checkpoints.

Preparation explanation from source scripts/w1ax_continuous_stages.py: each
resume executes audit_native_labels for every completed manifest, rebuilding
selected features and checking source/labels/prefixjoins/hashes on CPU. The
10,000traincaptures are retainedcomplete;224of1002devcaptured. Fresh07:41:20UTC
healthhealthyreaudit200/353,327retainedmanifests,zerooptimizer,stillpreparing.
Recent22shards/14minutes suggests~80minutes to re-audit remaining completed
shards, conditional on unchangedpace; unfinisheddevcapture and laterreadiness/
pairedsmoke/checkpointzero are additional,unestimatedwork. The human’s question
changes no frozen runningjob or validationgate. Preparationowner asked for
source-grounded timing/bottleneck explanation usingexistinghealth evidence.

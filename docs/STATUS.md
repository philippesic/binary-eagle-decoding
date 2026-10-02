# Current project status

**Comparison target:** Q4_0 EAGLE is the baseline to beat for acceptance,
latency and total throughput. FP16 EAGLE is secondary diagnostic context.
The target/verifier model precision remains as frozen for each experiment.

**Active goal:** [QAT optimization readiness](goals/qat-optimization-readiness.md).
All five requested controls plus raw fusion correction and affine binary weights
are implemented. Final CPU suite:952tests/fourskips; native CPU packing/operator/
encoder checks pass. Actual CUDA compilation, full-model backward, native decisions
and memory/timing remain unverified. [Integration report](../experiments/qat-optimization-readiness/11-integration.md).
All12 profile preparation/planning CLIs also pass on the published checkout;
[completion audit](../experiments/qat-optimization-readiness/12-completion-audit.md)
records requirement-specific CPU proof and missing GPU evidence.
The human now authorizes early GPU validation during the preparation CPU audit,
subject to a coordinated exclusive GPU reservation, fresh resource proof and a
safe handback before preparation resumes GPU work. Training still requires full
verified data/current gates and GPU handoff. [Overlap policy](QAT_GPU_AUDIT_OVERLAP.md). RTX2080Ti remains paused.
**Early validation lease granted:** RTX5080 is exclusively reserved for QAT
owner01a0f934 until October2 02:40:04PDT (09:40UTC). Preparation child676 is
intentionally held in the same live process at audit244/353; supervisor674 stays
live,10,000train/224dev retained,zerooptimizer. Fresh08:10UTC proof showed empty
GPUcomputeapps and13.37GBGPU/19.47GBhost free. Actual CUDA build completed; the first RTX5080 fixture executed at08:56:45UTC
and failed `pack beta mismatch` (activation scale). Backend-ops/model readiness
and QAT remain unproved. Failed evidence is preserved; a targeted native-owner
investigation is underway without gate weakening. Validationgroups2254/2256
were gone and GPUcomputeappsempty at08:57:31UTC. Validation teardown cutoff02:30PDT, then preparation owner
requires fresh ownedgroupsgone/GPUempty proof before sameprocessSIGCONT. Both
hold-awareprep and QAT15minmonitors ACTIVE. [Lease checkpoint](goals/qat-optimization-readiness.md#exclusive-early-validation-lease-granted--october2).
Training still requires complete verified data and current passing CUDA gates.
**09:18UTC decision:** QAT owner is returning the early-validation lease after
an additional diagnostic launcher failed before the fixture on a private-header
path error. Exact failed beta case remains unclassified. Fresh final cleanup and
same-process preparation resume are pending verification; no more GPU attempts
or lease renewal. The readiness objective remains incomplete.

**Overnight CPU research authorized (October1):** three teams investigate W1A1
representation geometry, native accepted-prefix objectives and block-parallel
drafter design, with one usage supervisor. QAT task and its monitor are protected
and may use credits. Latest human correction: stop research at <=1% remaining or the actual monitored
reset; the former09:55PDT cutoff is removed. Do not burn the new allowance. No new active goal,
GPU ownership or frozen-training change. [Assignments and control](OVERNIGHT_RESEARCH.md).

**Preparation supervision rotation (2026-10-02 06:42 UTC):** latest single CPU
check healthy, re-audit109/353;327completed manifests retain10,000train/224dev,
zerooptimizerupdates,no readyreceipt. SAME supervisor08 remains soleRTX5080
preparation owner with --prepare-only;2080Ti paused. All tick agents/transports
finished. Acknowledged successor `01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed` owns the SAME ACTIVE15min monitor; GPU
job unchanged; [exact handoff](goals/qat-optimization-readiness.md#preparation-owner-rotation--2026-10-02-0642-utc)
records source identity,job ownership,stopprocedure,tests and nextactions.
Separate validation/training owner still waits for verified fullprep/GPUrelease.

**Existing corpus preparation resumed and verified:** SAME --prepare-only run under supervisor08. CPU health **2026-10-02 05:31:39 UTC** (October1,10:31p.m.PDT), independent ownership **05:32:52UTC** confirms server673/supervisor674/child676 and live prep-only flag afterdisconnect/reconnect. Retained327manifests /10,000train /224dev,zeroQATsteps; re-audit4/353 then6/353 is not lostdata. Existing15min monitorACTIVE;2080Ti paused. No newcaptures yet: everyresume restarts the audit loop over completed data before unfinished dev capture. Same fullgates and automatic stop-before-QAT endpoint; new optimization-feature GPU checks still await ownership release.

**Historical existing corpus preparation pause:** Verified **2026-10-01 23:15:28 UTC** (4:15 p.m. PDT): supervisor07 interrupted/exit0,owned groups1204/1205 absent,no project processes or GPU compute apps. Retained327completed manifests /10,000train /224of1,002dev,zeroQATsteps; preparation remains incomplete. Data/partials/stop-before-QAT boundary preserved. No auto-resume until new human resume instruction.
**Historical preparation observation before pause:** [joint binary EAGLE body and head](goals/recurrent-binary-body-head.md), Phase 1 of the [one-bit research plan](W1_RESEARCH_PLAN.md). **Preparation-only RTX5080 run healthy; QAT optimizer updates disabled.** User endpoint is complete data/QAT prep, then stop. At **2026-10-01 20:44:55 UTC** (1:44 p.m. PDT), supervisor07 passed post-disconnect CPU health and live cmdline includes --prepare-only; server1203/supervisor1204/child1205 verified. Retained327manifests /10,000train /224dev,zerooptimizer; re-audit ordinal1/353 is not lost data. Same capture/audit/readiness/coverage/paired CUDA smoke and initial zero-update save precede automatic exit before optimizer loop. Existing15min monitorACTIVE with prep-only prompt; budget1of2 unchanged. Only launcher stopping control changed; math/native/runtime/config/data/precision/nullcaps and finals remain frozen. RTX2080Ti paused. See latest scope checkpoint.

**Broader CPU-first research audit complete (2026-10-01):** ten user-requested
GPT-6.1 Sol/high agents researched model compression, binary representations,
corrections, direct fitting, low-bit formats, draft policies, output heads,
native execution, data selection and alternative architectures. The
[plain-language ranked list](../experiments/broader-project-audit-2026-10-01.md)
recommends a bounded fusion sign/scale fitter plus a cheap correction control;
learned Q1_0 export and coupled FFN pruning are independent preparation options.
All ten reports are complete. This changes documentation only; no code,
model/test/GPU/remote run or sealed-final action occurred. No new experiment
or architecture was selected, and no fresh live health observation is claimed.
The user requests simpler explanations with high-impact summaries from now on.
See the [checkpoint](goals/recurrent-binary-body-head.md#ten-agent-broader-research-audit-completed).

**Training optimization audit complete (2026-10-01):** five user-requested
GPT-6.1 Sol/high agents reviewed source and primary research. The strongest
implementation candidates are bulk K/V-only Torch prefix construction, a
stacked per-round head and an A1 native-only no-gradient shortcut. Source
arithmetic permits removing 77.88% of prefix-stage linear MACs; no training
speedup or acceptance gain was measured. Binary optimizer, learned quantizer
and curriculum changes remain bounded proposals. Reports and the
[ranked synthesis](../experiments/training-optimization-audit-2026-10-01.md)
are published; all workers completed. No code, remote/GPU, live experiment,
monitor or sealed-final action occurred. See the
[audit checkpoint](goals/recurrent-binary-body-head.md#five-agent-training-optimization-audit-completed).

**Bounded GPU Phase 1B complete (2026-09-29):** fixed four-variant
quality/timing A/B, CUDA deployment checks, first-shard capture/audit and
100-step real row-A16 calibration are verified. All additional bounded
selector quality/timing and tracing gates completed; integrated worker
worktrees are archived, and the GPU is idle. Binary acceptance still trails
Q4_0; no full-body eligibility or final evaluation is claimed. The broader
project continues with the user-authorized CPU preparation below. See the
[completion audit](../experiments/w1-phase1b-completion-audit.md) and
[goal checkpoint](goals/recurrent-binary-body-head.md#bounded-gpu-phase-1b-complete).

## Continuous A8/A1 preparation complete (2026-09-29)

CPU preparation is implemented and independently audited in task
`01a0f01d-65c6-7af0-9660-99c07e95cacd`, continuing the same project goal.
The initial tier has10,000 train/1,002 dev/1,002 sealed-test prompts and24,507
reserve; all sealed payloads remain unopened. A verified22MB launch packet,
continuous dual-model engine, exact CPU resume/recovery, label-only v2 native
stages, refreshed-source binding, fixed Q4_0 dev evaluation and manual runbook
are published. **117 guarded CPU tests pass**; actual CUDA/training/acceptance
and teacher tensor volume remain unverified USER-start gates.

The hourly heartbeat is **PAUSED**, with the per-heartbeat Luna model override
limitation recorded. No native inference/capture/training or WSL GPU action was
launched. A legacy CPU fixture's accelerator RNG API calls were identified and
corrected; the final audit guards library backend APIs. See the
[preparation report](../experiments/continuous-w1ax-preparation.md),
[manual runbook](CONTINUOUS_W1AX_RUNBOOK.md), and
[goal checkpoint](goals/recurrent-binary-body-head.md#continuous-a8a1-cpu-preparation).
The user has now authorized Luna to start the prepared native gates, capture and
continuous paired W1A8/W1A1 training on RTX5080, then monitor with durable spaced
checks. This supersedes the preparation-only boundary; RTX2080Ti remains paused.
One Luna operator owns the GPU. No tight polling/sleep loop is authorized.
See the [launch checkpoint](goals/recurrent-binary-body-head.md#luna-launch-authorization).

Luna operator `/root/luna_launch` connected through tmux MCP, verified no
competing project process and RTX5080 compute capability12.0. Initial baseline
was2,617MiB/2%utilization and764GiB free disk. Packet/frozen artifact hashes match.
A Unicode JSONL delimiter bug was fixed/tested/pushed as`41e230d`, preserving
all data bytes; host config resolves10,000 train/1,002 dev with622,868,961,328B
capture forecast. First supervisor stopped on the unchanged14GiB host-RAM guard.
Corrective validation`2b87b5b` passed the bounded W1A8 native/CUDA gate, then
stopped at A1 initialization: available12.94GiB after reclamation,1.06GiB below
admission; WSL exposes15.21GiB total. No optimizer/full corpus capture ran.
Both attempts/evidence are preserved and no project process remains. Hourly
heartbeat is **PAUSED**; it dispatches pinned Luna/high once a live run is
verified. User-reported memory change now verified: `[wsl2] memory=20GB`,
WSL total20,971,151,360B/available20,237,811,712B at post-restart preflight.
Supervisor03/04 passed A1 memory admission but failed deployment-relevant state/
decision checks. Retained head replay exactly matches native output; CPU tests
confirm scaled-sum cancellation can flip the next A1 sign. Corrected native-order
A1 forward/reference code is integrated as`f0566aa`/`119e418`, retaining local STE
gradients and unchanged other widths/criteria.144 guarded CPU checks pass. Luna
completed both fresh gates (seven checks/six roots each) in new experiment
`luna-continuous-a8-a1-native-order-20260930`, preserving old failures/captured
bytes. The first hourly check at10:23UTC found supervisor interrupted/exit0 at
09:12UTC and trainer stopped on signal15. The stop predates monitor activation;
the earlier live handoff report relied on stale startup evidence. Monitor is
PAUSED, logs/data preserved, no restart. Signal origin and partial capture
recovery require investigation before resume. See [monitor setup](CONTINUOUS_W1AX_MONITOR.md)
and the historical interruption checkpoint. The repair now uses WSL
`instanceIdleTimeout=-1` plus host-side tmux`binary-eagle-runtime`, proved over
90s with all clients closed. Complete32promptshard retained, incomplete611MB
capture quarantined intact. Supervisor02 is running under host tmux847 with
supervisor848/trainer855, verified after disconnect at19:35:12UTC. Capture is
healthy; no optimizer steps. See the
[current checkpoint](goals/recurrent-binary-body-head.md#post-disconnect-real-resume-verified-live).

## Historical checkpoints

**Final selector timing live (2026-09-29):** packed shared/warp quality
pairs each passed 6/6, completing the bounded selector quality gates. The
validated resident/shared/warp runtime is now in an immutable order-balanced
timing queue. Ratios and cleanup remain pending; the goal stays active. See
the [checkpoint](goals/recurrent-binary-body-head.md#final-selector-quality-passed-and-frozen-timing-live).

**Resident and event/warp CUDA gates passed (2026-09-29):** the fixed
resident path matched 6/6 real quality pairs with accepted CUDA0 copies and
no transfer fallback. Warp integer assertions passed 133 operator/three
fanout cases. Event fixtures and actual-model tracing audited without reference
or cap errors. Remaining packed-selector quality/timing and cleanup are
recorded in the [checkpoint](goals/recurrent-binary-body-head.md#resident-fix-and-eventwarp-cuda-gates-passed).

**Runtime timing complete (2026-09-29):** all 120 paired requests and
80 CUDA-graph blocks passed. On the bounded three-prompt workload, Q4_0
order-balanced decode ratios were compact **1.04028** and K/V-only **1.00817**;
row-A16 was 1.02666/1.08174. A combined CUDA build is now live for the
resident-copy fix and event/warp gates. See the [checkpoint](goals/recurrent-binary-body-head.md#runtime-timing-finished-and-combined-cuda-build-started).

**Resident fix integrated (2026-09-29):** the guarded scheduler-copy fix
passed 19 CPU allocator/routing cases and is integrated with default-off event
and warp diagnostics. CUDA proof is queued after the live immutable timing
run. Compact timing has two behavior-matched blocks; K/V-only timing continues.
See the [checkpoint](goals/recurrent-binary-body-head.md#scheduler-copy-candidate-and-combined-runtime-integrated).

**Passing-selector timing underway (2026-09-29):** compact sampling and
K/V-only catch-up are in a supervised order-balanced, five-repetition timing
queue on the frozen three-domain sample and immutable runtime/libraries.
Resident state remains off while its scheduler-copy fix is CPU-tested.
See the [checkpoint](goals/recurrent-binary-body-head.md#scheduler-evidence-and-frozen-timing-queue).

**Resident diagnostic (2026-09-29):** restoring original input placement
restored all six request pairs exactly, isolating forced placement as the
trigger. The passing path still uses host traffic and is not a speed claim.
One bounded scheduler-log probe is live to audit device-only admission; the
resident selector remains gated. See the [checkpoint](goals/recurrent-binary-body-head.md#resident-placement-isolated).

**Native quality gate (2026-09-29):** compact sampling and K/V-only
catch-up each matched 6/6 request pairs exactly. Resident state preserved
outputs but changed row-A16 proposals/acceptance and failed the gate; it
remains off pending a focused placement diagnostic. The GPU is idle. See
the [checkpoint](goals/recurrent-binary-body-head.md#real-selector-quality-results-and-resident-divergence).

**CUDA runtime fixtures passed (2026-09-29):** corrected dense and all six
packed A1/A4/A8 sharing off/on fixtures passed, alongside 133 operator and
three fanout cases. Matched real-model compact/cache-only/resident quality
pairs are running sequentially on the frozen three-domain sample. The native
refresh bridge is integrated with 20 CPU tests; capture/training permissions
remain false. See the [checkpoint](goals/recurrent-binary-body-head.md#corrected-cuda-fixtures-and-native-quality-queue).

**Stable CUDA checks (2026-09-29):** dense runtime fixture, 133/133 binary
operator cases and three shared-pack fanout graphs passed on RTX 5080. Packed
fixture reruns await a tested encoder batch-metadata initialization fix;
no numerical or cache assertion is being relaxed. Real selector quality and
event/warp validation remain pending. See the [checkpoint](goals/recurrent-binary-body-head.md#stable-cuda-operator-and-dense-fixture-milestone).

**Stable runtime validation started (2026-09-29):** shared packing and
bounded resident state are integrated as native `8fd9b399a` / parent `0261d1b`
after CPU checks. The supervised CUDA build is live; actual CUDA fixtures and
selector comparisons are next. See the [checkpoint](goals/recurrent-binary-body-head.md#stable-runtime-integration-after-calibration).

**Real-model calibration complete (2026-09-29):** the gated RTX 5080 run
finished 100/100 row-A16 hard-CE steps with finite loss and all 18 gradients.
Mean/median synchronized step time was 0.923/0.737 s; CUDA allocator peaks
were 8.494 GiB allocated and 10.049 GiB reserved. The GPU is idle with no
remaining run process. Original/full-body eligibility remains false; no
trained native quality claim has been made. The authorized engineering backlog
remains active. See the [report](../experiments/w1ax-row-a16-calibration-5080.md)
and [checkpoint](goals/recurrent-binary-body-head.md#real-model-calibration-completed).

**Calibration started (2026-09-29):** all five readiness gates passed,
including nine ordered cache/head bridges. The sole GPU owner launched the
authorized supervised 100-step row-A16 hard-CE calibration on RTX 5080.
Completion and timing/memory measurements remain pending. Original data
and full-body training eligibility remain false. See the
[launch checkpoint](goals/recurrent-binary-body-head.md#calibration-readiness-passed-and-optimizer-launched).

**Readiness adapter integrated (2026-09-29):** the selected-head prenorm
observer adapter passed 13 CPU tests and is pushed as `fd8e1a1`. The supervised
CPU readiness retry is running against the existing capture; no optimizer step
has started. See the [checkpoint](goals/recurrent-binary-body-head.md#selected-head-observer-adapter-integrated).

**Pilot cache/backward gates passed (2026-09-29):** nine roots have finite
all-nine gradients and exact frozen operands; the CUDA audit matched
5,533,696 key and value elements each and 5,404 causal masks. Q4_0 and
row-A16 response IDs match on the three prompts. Versioned readiness
assembly is completing the ordered sequence/task/cache ancestry join before
any optimizer run;
original eligibility remains false. Real label-only v2 conversion/audit
passed on the 31-prompt shard without copying 7.66 GB of raw logits. See the
[latest evidence](goals/recurrent-binary-body-head.md#phase-1b-completed-backward-and-native-cache-evidence)
and [storage/refresh checkpoint](goals/recurrent-binary-body-head.md#compact-storage-and-refresh-engineering-checkpoint).

**Parallel engineering resumed (2026-09-29):** the user's companion-chat
authorization starts the remaining CPU runtime, compact-storage and refresh
work within the same goal. The bounded pilot remains first: numeric checks
passed, while backward-only gradient and reduced CUDA cache capture checks
are still in progress. A full graph capture reached its 512-MiB cap and is
incomplete; it does not authorize training. Calibration-only evidence and
measurement guards are pushed; the original bundle stays ineligible. See the
[assignments and current job](goals/recurrent-binary-body-head.md#parallel-engineering-authorization-and-pilot-continuation).

**Torch CUDA pilot gate (2026-09-29):** nine selected roots passed the
frozen numerical check with zero proposal disagreements. Maximum state/logit
relative RMS was `3.55e-5`/`7.77e-5`, below `0.10`; the three Q4_0 response
ID pairs matched exactly. The original bundle remains ineligible until
cache/mask, gradient and versioned provider gates pass. The sole coordinator
is building the opt-in CUDA cache diagnostic after preserving the original
runtime. See the
[checkpoint](goals/recurrent-binary-body-head.md#phase-1b-torch-cuda-numerical-gate).

**Pilot revalidation (2026-09-29 18:45 UTC):** fresh first-shard re-audit
passed byte-identically, all pinned provider/model snapshot hashes matched,
and all nine exported row-A16 sign-bit/scale pairs matched checkpoint zero.
The GPU was idle at the fresh check. The Torch/native numeric check and
selected-root CUDA cache/mask evidence remain required before calibration;
the original bundle is still ineligible. See the
[checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-pilot-revalidation-and-implementation).

**Pilot resumed (2026-09-29 18:17 UTC):** the user chose the short practical
native gate followed by one 100-step row-A16 hard-CE calibration if it passes.
The original capture remains ineligible until a versioned gate record proves
the focused checks. RTX 5080 access remains resumed; a fresh check found 0%
utilization, about 2.9 GiB baseline use and no project process. The
[decision](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices)
and [goal checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-short-pilot-resume)
record the limited scope and stop condition.
The three-domain native row-A16 diagnostic has now completed with exact
response-ID matches to the candidate-D capture and 199 shared proposal roots
across the three prompts. The focused Torch-versus-native check is next; no
eligibility flag has changed.

**Prior decision boundary:** the native GPU Phase 1B task has completed its
fixed A/B, CUDA, first-shard capture/audit, synthetic device gate and untrained
checkpoint-zero quality checks. The first-shard provider still rejects
captured-data QAT because its readiness flag is false. The user owns the
[practical-native versus strict-parity and initialization choice](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices)
before a real-model 100-step calibration. The native task is blocked on that
choice after three consecutive goal turns; the project goal and RTX 5080
access flag are unchanged. No GPU process is active.

**RTX 5080 resume (2026-09-29 06:54 UTC):** a fresh tmux MCP session found 0% utilization, about 2.9 GiB baseline memory and no project process. The local RTX 5080 pause flag was resumed. Two full timed pairs in opposite orders completed, with 480 behavior-matched requests and 20 verified graph blocks per condition. The [timing report](../experiments/eagle-prune-timing-5080.md) gives order-balanced on/off server decode ratios of Q4_0 1.0085×, D 1.0513× and FP16 1.0112×; target-only moved 1.0013×. D remains far slower than Q4_0 overall. The first bounded 31-prompt native capture finished: 12,610 raw verifier-logit rows, 7,796 selected feature rows, byte-identical independent bundle audit, compact teacher and 31/31 response audit. A 100-step model-independent row-A4 CUDA trainer fixture exited zero with finite metrics and 18 gradient tensors per step. The [capture report](../experiments/w1ax-shard0000-capture-5080.md) records hashes and limits. Its manifest remains preparation-only and training-ineligible, so real QAT still requires a documented readiness decision. The GPU has returned to baseline with no project process. The [goal checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-first-shard-audit) records run IDs, owner and next gate. The prior pause checkpoint below remains historical.

**Row checkpoint-zero gate (2026-09-29):** pinned dense weights produced one
untrained nine-linear row checkpoint; A4 and A16 exports passed serialization
and native CUDA graph execution. On 24 old development prompts, A4 accepted
131 drafts over 2,917 rounds and A16 accepted 323 over 2,725; matched Q4_0
accepted 1,555 over 1,493. Both row variants emitted the same raw IDs as Q4_0
on all 24 pairs. The [native report](../experiments/w1ax-checkpoint-zero-native-5080.md)
records hashes and limits. This is low checkpoint-zero acceptance, not a QAT
or timing result. The first-shard provider remains ineligible; the
[readiness/initialization choice](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices)
is pending. No project GPU process is running and RTX 5080 is at baseline.
The first-shard CPU domain check confirmed all 12,610 audited rows across
prose (4,815), reasoning (4,020) and code (3,775), with 12,250 supported
labels in total; see the capture report. This does not change eligibility.

**GPU pause checkpoint (2026-09-29 05:55 UTC):** an earlier reply misread “No pause gpu work actually” as a direction to continue. The coordinator corrected that interpretation, set the RTX 5080 pause flag, verified every Phase 1B supervisor terminal and no project `llama-server`, trainer or supervisor process on WSL, and closed tmux MCP session `$34`. The GPU still has Windows game/display workload; no project GPU process remains. The completed CUDA build, 112/112 operator check and exact 96-pair quality comparison are preserved. Timed A/B, first-shard capture and training calibration did not run. See the [goal checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-pause-checkpoint).

**GPU Phase 1B startup (2026-09-28):** the coordinator resumed the local RTX 5080 flag and owns the sole GPU experiment through tmux MCP. Pinned target/FP16/Q4_0/D/config hashes matched; verified links resolve archived remote artifacts. The supervised SM120 CUDA rebuild completed, and 112/112 W1A1 matrix-operation cases passed. The remote 2k candidate freeze and 65-shard plan now match their original hashes exactly. The paired off/on native quality check matched all 96 requests in generated IDs, speculative counters and checked round semantics; the [report](../experiments/eagle-prune-quality-5080.md) records hashes and limits. The first-shard runner preflight passed without inference. With all project processes stopped, sustained non-project GPU load persists above 90% utilization and 4.8 GiB use; uncontended timing and the memory-heavy capture await resource clarification. The RTX 5080 pause flag remains resumed. The [goal checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-startup) records session, runs and limits.

**CPU Phase 1A complete (2026-09-28):** the fresh team's latency analyzer,
opt-in shared runtime patch, four-format row-scale joint QAT with a guarded
Torch device path, eligible-capture/multi-shard provider, pinned candidate
2k/192/192 data freeze and bounded 65-shard capture plan are reviewed,
CPU-tested, committed and pushed on main through `d9f3ae9`; native gitlink
`14c188e` is published in the user's fork. The
[completion checkpoint](goals/recurrent-binary-body-head.md#cpu-phase-1a-completion-and-gpu-only-boundary)
records hashes, limits and the first supervised GPU commands. All temporary
team worktrees were archived after preserving ignored data. The same project
goal remains active; native acceptance, CUDA timing/training and SM75 claims
wait for explicit restored access. Full-tier raw-logit storage and final
representation/objective budgets are user-owned choices before their runs.

**CPU Phase 1A team setup (historical):** fresh task
`01a0e9d0-1273-70a1-972e-8d1381f72701` resumed the same goal with separate
runtime, joint QAT, data and CPU verification workers. Their file ownership
and worktrees are recorded in the [goal checkpoint](goals/recurrent-binary-body-head.md#fresh-team-execution-cpu-phase-1a).
Row-scale W1Ax is the actionable common-format implementation default while
candidate D group-128/A16 stays separately labeled; the native learned-row
loader gate now passes CPU metadata tests for all four widths. The options and
provisional practical numeric gate are in [DECISIONS.md](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices).

The archived-latency and capture-readiness report, larger-data preparers and
compact-teacher schema, and synthetic joint W1Ax QAT path are now integrated
and pushed (`e5bd3dc`, `0b5fce7`, `6db186f`). The corrected native runtime
patch and latency analyzer were integrated as `112693f`, with native gitlink
`14c188e` published in the user's fork. Subsequent data/capture/provider
commits and final CPU checks are summarized in the completion checkpoint.

**Candidate data freeze (2026-09-28):** pinned Dolly/GSM8K/MBPP source files
and a hashed local manifest now supply 2,000 candidate train prompts plus
192 independent development and 192 sealed new final prompts, balanced across
three broad domains (`c4f6764`). Raw prompts remain ignored under `data/`; the
new final text was not opened. This is not yet tokenized, captured or approved
as sufficient training coverage. The [source report](../experiments/w1a-public-source-freeze.md)
records hashes, terms and limitations. A Luna worker accidentally queried
local CUDA/MPS availability once during environment discovery; no accelerator
operation or model inference ran, and verification resumed on explicit CPU.

**Trainer integration (2026-09-28):** the joint QAT CLI now accepts an audited
native-prefix provider (`29f0e96`). It checks eligibility before model loading,
rechecks trace and exact-prefix compact-teacher ancestry, and retains recurrent
state/cache gradients in CPU fixtures. The concrete future-capture factory is
integrated as `83e72d6`; it still awaits eligible larger captures. A guarded
device-aware Torch rollout is integrated as `4d0684d`; bounded raw-logit
sharding is integrated as `08c70b0` (65 capped train shards), and the
capture-tool prompt-contract extension is integrated as `a18e5a9`. The v1 bundle still
requires raw logits for re-audit; the full-tier compact/label-only storage
choice is recorded in [DECISIONS.md](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices).
Native training quality and throughput remain unmeasured.

**Pre-team handoff (historical):** the user requested an overarching plan and a fresh team to continue the existing EAGLE work with no GPU access. Parent starting point was `d111335`, following the predecessor's `5110257`; native gitlink then was `21f617d4ef3f5dc383d3ab8dc619daaa87db7ff8`. Both host pause flags were set. The plan superseded the old next action to continue target block-14 parity. The old 96 prompts are smoke/regression data, not an adequate full-body QAT corpus. The [handoff](goals/recurrent-binary-body-head.md#fresh-team-handoff-eagle-w1-cpu-phase) and [completion checkpoint](goals/recurrent-binary-body-head.md#cpu-phase-1a-completion-and-gpu-only-boundary) preserve the state across tasks.

**Latest completed goal:** [mixed precision rescue and frozen-body head adaptation on RTX 5080](goals/binary-rescue-head-5080.md), completed 2026-09-28 UTC. The preceding [binary scale fitting goal](goals/binary-scale-fitting-5080.md) completed 2026-09-27 UTC.

**Pre-handoff checkpoint (2026-09-28 UTC, historical):** the output-preserving RTX 5080
block-14 stage capture was compared with source HF CUDA/F16 eager on
the frozen 29-token training prefix. At position 3, accumulated HF
error grows from 1.2157% at FFN input to 12.1527% at FFN branch
output and 4.1478% at complete block output. Replacing the full
block-14 input with native rows cast to F16 leaves 0.1015%, 0.5045%
and 0.1491% at those boundaries; the earlier intervention controls
reproduce exactly. This locates the observed amplification chiefly
in the FFN path on that row, without identifying the causal FFN
operation. The [stage report](../experiments/recurrent-target-block14-stage-intervention.md),
[safe capture](../experiments/recurrent-target-block14-safe-stages.md)
and [goal checkpoint](goals/recurrent-binary-body-head.md#forty-seventh-goal-turn-block-14-stage-attribution-gpu-paused)
record hashes and limits. Our supervised job finished and process
group 417 is absent. New RTX 5080 runs are locally blocked; a later
read showed 95% GPU utilization and 8,011 MiB whole-device use from
a workload outside this project run. The Apple CPU drafter forward
has 18 exact diagnostic depths. Full target-feature parity, a
training numeric policy and all-body budget remain open. Training,
final-set and Q4_0 evaluation remain gated.

**Prior CPU arithmetic checkpoint (2026-09-28 UTC):** corrected CPU RMS norm and RoPE
frequency arithmetic matched all 619,520 F16 fused-input and 123,904 raw
F32 K/V projection elements across three post-acceptance joins. F16 projected
value writes matched 123,904/123,904; projected key writes matched
123,900/123,904. Native stored cache bytes and attention arithmetic were
still unverified at that checkpoint. The [goal file](goals/recurrent-binary-body-head.md#twelfth-goal-turn-native-style-cpu-norm-and-rope)
records the checks, report hashes, 5080 access change and next gate.

**Stored-cache checkpoint (2026-09-28 UTC):** an opt-in native CPU capture
verified actual F16 draft-cache writes and exact-prefix mask inputs on two
frozen training requests. Key and value bytes each matched all 148,480
captured F16 elements, including two reserve rows per request; all 145
decoder mask rows allowed exactly slots through their query position. The
[cache report](../experiments/recurrent-binary-cpu-cache-parity.md) and
[active goal checkpoint](goals/recurrent-binary-body-head.md#thirteenth-goal-turn-actual-stored-draft-cache-and-mask)
record hashes, commits and limits. Attention arithmetic, general cache
behavior, full 96-prompt capture, training and Q4_0 quality/speed gates
remain open. No GPU run had started at that checkpoint.

**RTX 5080 capture checkpoint (2026-09-28 UTC):** the authorized eight-token
CUDA smoke on one frozen training prompt captured raw target logits and
features, passed continuity/response audits and reproduced the CPU raw IDs.
Its supervised run stopped and the GPU returned idle. The full 96-prompt
capture is prepared with expanded explicit row limits and an isolated
remote checkout; it has not started. The [goal checkpoint](goals/recurrent-binary-body-head.md#fourteenth-goal-turn-cuda-capture-path-and-full-run-setup)
records the owner, tmux session, remote directory, commits, source hashes
and stop procedure. No training budget or final-set use has been approved.

**Full frozen-training capture (2026-09-28 UTC):** the supervised RTX 5080
run finished all 96 training requests with 40,815 joined head/verifier-logit
rows, 52,297 raw target-feature rows and 8,295 native rounds. Its process
group stopped and the GPU is free. Internal continuity and native row/feature
preparers passed; the bundle and all-request response audits remain in
progress. The [active goal checkpoint](goals/recurrent-binary-body-head.md#fifteenth-goal-turn-frozen-96-prompt-cuda-capture-audit-pending)
records counts, hashes, limitations and the next CPU checks. The raw capture
is not yet training-eligible; no optimization or final-set evaluation ran.

**Final capture preparation checkpoint (2026-09-28 UTC):** the frozen
96-prompt bundle and all-request audit now pass. All 40,815 raw verifier
logit rows and 15,042 retained feature rows are joined; 96/96 responses
match 12,251 native emissions, including one final EOS stop with an
un-emitted canonical suffix. Mean mapped target probability mass is 0.972
on captured candidate-D histories. The bundle remains explicitly
`training_eligible: false`. A CPU attention arithmetic ablation explains
most first-seed drift but leaves residual numerical differences. The
[active goal checkpoint](goals/recurrent-binary-body-head.md#sixteenth-goal-turn-full-capture-audited-training-still-gated)
and [full capture report](../experiments/recurrent-binary-full-capture-5080.md)
record hashes, checks, failed first audit attempts and remaining gates.
The RTX 5080 is free; no training or final-set use occurred.

**Target-feature checkpoint (2026-09-28 UTC):** independent Hugging Face
forwards on Apple M3 Max CPU and, separately, RTX 5080 CUDA/F16 used frozen
training prefixes and checked source embeddings. Their raw target-feature
rows still differed from native execution by median relative row L2 errors
of roughly 0.3–0.6% across taps 2, 18 and 33; the largest checked row
reached 2.088% in the CPU reasoning capture. The
[target-feature report](../experiments/recurrent-binary-target-feature-parity.md)
records per-tap measurements, hardware, versions and source hashes. Switching
the independent CUDA forward from eager to SDPA attention barely changed
the differences. Both CUDA diagnostics stopped and the GPU is free. Exact
target-feature and
whole-drafter state/logit parity remain open; the capture bundle is still
training-ineligible pending a user-owned numeric gate and training budget.

**Full-prefix feature checkpoint (2026-09-28 UTC):** the sealed 96-request
capture supplied 3,112 complete training prefill rows. On RTX 5080 CUDA/F16,
independent eager forwards differed from native tap-2/18/33 features by
median relative row L2 of 0.293/0.538/0.488%. One tap-18 row reached
10.014%; the same row reached 12.135% with independent SDPA attention.
An Apple M3 Max ggml CPU layer-0 operator probe found near-exact RMS norm
but nonzero pre-attention Q/K/V projection differences from F32 references.
The native capture also has 103/427 cross-request prefill-row pairs with
identical token prefixes but different feature bytes, beginning after the
first token; the cause needs investigation. A bounded recapture of the
tap-18 outlier prompt matched all 222,720 native prefill F32 values bitwise,
so that row is reproducible under the same native CUDA binary.
The [goal checkpoint](goals/recurrent-binary-body-head.md#eighteenth-goal-turn-full-training-prefix-feature-distribution-and-layer-0-probe)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md)
record hashes, hardware and limits. Both supervised GPU comparisons exited
zero and released the 5080. Exact target-feature and whole-drafter parity,
numeric gate, training budget and Q4_0 evaluation remain open.

**Tap-2 intervention checkpoint (2026-09-28 UTC):** on the reproducible
29-token outlier training prompt, substituting captured native tap-2 input
into an independent RTX 5080 F16 forward cut position-3 tap-2 error from
0.324% to 0.020%, while tap-18 error rose from 10.014% to 11.659%.
Thus the early tap-2 mismatch alone does not explain the later outlier.
The [goal checkpoint](goals/recurrent-binary-body-head.md#nineteenth-goal-turn-native-tap-2-input-intervention)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md#native-tap-2-input-intervention-on-the-outlier)
record bounds, source hash and limits. The supervised GPU run stopped;
target layer-by-layer parity, the numeric gate, training and Q4_0 evaluation
remain open.

**Target ladder checkpoint (2026-09-28 UTC):** a bounded native RTX 5080
capture on the 29-token outlier prompt recorded target layer inputs
`0–18,33`. Existing taps 2/18/33 matched the same-run and sealed full96
feature bytes exactly. The independent F16 forward's position-3 error had
its largest adjacent rise across target block 14: 1.209% at layer-14 input
to 4.148% at layer-15 input, with absolute RMS error 0.01202→0.04218.
The [active goal checkpoint](goals/recurrent-binary-body-head.md#twentieth-goal-turn-target-layer-input-ladder-localizes-the-outlier)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md#native-target-layer-input-ladder-on-the-outlier)
record source hashes, hardware and limits. Both supervised runs stopped and
the GPU is free. The responsible block operation, exact target parity,
training gate and Q4_0 evaluation remain open.

**Block-14 intervention checkpoint (2026-09-28 UTC):** substituting captured
native layer-14 input into the independent RTX 5080 F16 forward reduced
the outlier's layer-15 relative error from 4.148% to 0.149% and absolute
RMS error from 0.04218 to 0.001516. Block 14 amplifies earlier drift in
this comparison; its same-input operator mismatch is much smaller. The
[active goal](goals/recurrent-binary-body-head.md#twenty-first-goal-turn-block-14-amplifies-upstream-drift)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md#same-input-block-14-intervention)
record the source hash and limits. The supervised run stopped and the GPU
is free. The first material local operator difference and training gate
remain open.

**Local target-block screen (2026-09-28 UTC):** on the same frozen 29-token
prefix, independent RTX 5080 F16 forwards supplied each block `0–17` its
own captured native input. At the position-3 outlier, every same-input
block-output error was at most 0.272% relative row L2, compared with the
accumulated 10.014% layer-18 error. Small local backend differences are
amplified through later blocks; the all-row ranking differs and no numeric
gate follows from one prompt. The [goal checkpoint](goals/recurrent-binary-body-head.md#twenty-second-goal-turn-local-target-blocks-and-amplified-state-drift)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md#same-input-local-block-screen)
record source hashes and limits. The supervised run stopped and the GPU is
free. Full-drafter parity, training budget and Q4_0 evaluation remain open.

**CPU drafter checkpoint (2026-09-28 UTC):** a standalone ggml replay using
actual stored F16 draft K/V bytes, captured masks and native queries matched
all **593,920** F32 attention output elements bitwise across 46 prose and
reasoning decoder executions on Apple M3 Max. A separate no-optimizer
real-size two-step D probe applied CE only at the later proposal and found
nonzero gradients in the earlier pre-norm state and appended K/V rows, with
zero gradient on the earlier logits. The [goal checkpoint](goals/recurrent-binary-body-head.md#twenty-third-goal-turn-native-cpu-attention-oracle-and-real-causal-gradient),
[attention report](../experiments/recurrent-binary-cpu-attention-oracle.md) and
[gradient report](../experiments/recurrent-binary-real-later-gradient.md)
record hashes, hardware and limits. The Python student still uses a different
F32 attention forward, and exact whole-drafter parity, training budget and
Q4_0 evaluation remain open. No GPU was used for these two checks.

**Optional native student-attention checkpoint (2026-09-28 UTC):** the CPU
student can now use the pinned ggml attention result in an explicit
diagnostic forward while retaining an F32 surrogate backward. Correcting
Python/native Q/K row order made the prose first-seed attention output
4,096/4,096 F32 elements bitwise equal; reasoning's remaining maximum
attention error is 0.002172 from stored K/V operand differences. First
normalized-state maximum error fell to 0.000184 prose and 0.000511
reasoning, and a real-size later-only loss still reached earlier state/K/V
through the surrogate without an optimizer step. The [goal checkpoint](goals/recurrent-binary-body-head.md#twenty-fourth-goal-turn-optional-native-forward-student-attention)
and [diagnostic report](../experiments/recurrent-binary-native-attention-student.md)
record exact hashes and limits. This mode is not a chosen training recipe;
full-drafter parity, all-body budget and Q4_0 evaluation remain open.

**Same-input FFN checkpoint (2026-09-28 UTC):** on a captured native FFN
input, candidate D's native-order CPU arithmetic matched all 2,560 prose
output values bitwise; grouped matmul differed by at most 1.4305e-6.
On reasoning, both modes remained 6.1035e-5 from native while differing
from each other by at most 1.9073e-6. The [goal checkpoint](goals/recurrent-binary-body-head.md#twenty-fifth-goal-turn-same-input-binary-ffn-boundary)
and [FFN report](../experiments/recurrent-binary-ffn-same-input.md) record
hashes and limits. This narrows the remaining reasoning FFN gap beyond
grouped reduction order; exact whole-drafter parity, training budget and
Q4_0 evaluation remain open. No GPU was used.

**Handoff checkpoint (2026-09-28 06:35 UTC):** the active goal file records
the current objective, pushed commits, CPU tests, projected K/V write
comparison, pending 5080 clarification and exact next actions. Code round 2
matched 37,681/37,888 F16-rounded key operands and 37,723/37,888 value
operands across 37 reconstructed context positions; native stored K/V bytes
remain unread. No agent, model server, local experiment or remote job is
running. The RTX 5080's availability has not changed the explicit CPU-only
restriction. Continue from [the active goal handoff](goals/recurrent-binary-body-head.md#rotation-handoff-2026-09-28-0635-utc).

**Current goal milestone (2026-09-28 UTC):** the [joint-training protocol](../experiments/recurrent-binary-qat-plan.md) records the candidate-D W1A16 representation, exact-prefix recurrent supervision, proposed bounded trial and stop gates. The CPU hard-binary core, nine-linear installer, exact-prefix trace validator, masked recurrent loss, differentiable proposal-chain interface, optimizer ownership/checkpoint step and learned-scale GGUF serializer are integrated. The focused CPU gates pass, including a training-checkpoint-to-GGUF synthetic roundtrip and a two-step causal-cache gradient test. A new scalar CPU replay matched 430/430 archived candidate-D native samples across all nine projections. This is arithmetic evidence on old training captures, not a trained-model quality result. Existing cached head states cannot train the body; a real feature/cache/verifier capture, full-drafter numeric parity and trained-export native validation remain necessary. No accelerator, remote host, model training or final-set prompt has been used for this goal. Q4_0 remains the primary future acceptance, latency and throughput gate; no new quality or speed claim exists.

**Second CPU milestone:** native decoder memory position is one behind the
shifted input-token index; the rollout and tests now use that position.
Accepted-prefix cache rebuilding uses raw target features with a fresh cache
and an explicit truncated-gradient boundary. The trace rejects draft-head
logits mislabeled as raw target verifier logits. The CPU capture gate now
requires a hashed raw-feature ledger joined to every accepted-prefix anchor,
with 7,680-wide F32 rows and frozen target tap order. An explicit grouped-F32-matmul
training option matched 395/430 archived native D outputs exactly; its maximum
absolute difference was `0.0001220703125`, so real-model argmax and cache
parity remain gates. Training checkpoint manifests record the arithmetic mode.
No current accelerator operation was performed.

**Third CPU milestone:** the forked llama.cpp loader now recognizes truthful
`f32_learned_nonnegative` metadata at published submodule commit `7f23c89b3`.
A CPU-only `libllama` build and eight native loader fixtures passed. This
checks metadata and tensor coverage, not numerical parity of a trained GGUF.
The pinned AngelSlim forward's floating-weight dtype read and F32 K/V cache
were identified as incompatible with the proposed hard-binary/F16-KV path.
An explicit CPU decoder-step adapter now runs all nine binary linears through
two synthetic proposal steps and a masked optimizer update, including F16
K/V cache writes. A frozen-D GGUF initialization audit passed all nine
tensor pairs and found 17,005 exact-zero scales. Native full-drafter numeric
parity remains unverified. A read-only CPU GGUF view verified the pinned FP16
target and D draft hashes, memory-mapped the frozen target embedding and
extracted one F16 row plus four F32 draft norms. The adapter can consume these
operands without copying the full target embedding or changing target weights.

**Fourth CPU milestone:** fork commit `c282087a9` adds an opt-in 32-row-capped
file of raw target verifier logits, copied before sampler processing and
separate from the existing mapped draft-head logit file. A CPU-only
`llama-server` build passed with GPU and optional Accelerate/BLAS backends
disabled; runtime capture was not exercised. The CPU capture audit checks
that file's hash, target-vocabulary width, source label and unique exact-prefix
row joins. Live `verifier_reached` and teacher-forced valid/support masks are
now recorded separately, so later-position loss is not censored merely by
an earlier live rejection. Accepted-prefix raw target-feature capture and
real-model parity remain open.

**Fifth CPU milestone:** fork commit `ddcf2a608` adds opt-in bounded raw
target-feature rows and one retention disposition per decoded row. The native
source captures ordered target layer-input taps before EAGLE fusion, records
exact token ancestry and marks speculative input rows `j<=A` retained after
`A` accepted drafts. A CPU-only server build passed with optional accelerator
and BLAS backends disabled. Parent CPU preparers now validate native
head/round/label/map joins and select accepted-prefix feature rows; 59
recurrent tests pass. No real-model feature capture or microbatch row-order
parity has run, and Q4_0 quality/throughput gates remain untouched.

**Sixth CPU milestone:** a pinned `recurrent-train` capture mode now prepares
the frozen 96-prompt D/D own-history run with explicit raw target-logit and
feature budgets, request ownership/ranges, and hashes for each raw stream.
It has only been exercised with mocked CPU server output. A separate CPU
continuity audit checks complete captured prefill, every speculative input,
the accepted-prefix retention rule and consecutive round prefixes/seeds;
initial sampling, terminal emission and request completeness remain
unverified. A bundle builder joins both native preparers and the final
capture audit, checks pinned target/draft hashes and the D map digest, and
always marks output `training_eligible: false`. The focused 73 recurrent
and 16 capture-runner CPU tests pass. No real-model
capture, training, accelerator or Q4_0 quality/throughput gate ran.

**Seventh CPU milestone:** the first actual pinned FP16-target/candidate-D
model request ran on an accelerator-disabled Apple M3 Max CPU server. A
missing A16 activation setting initially stopped draft loading; the runner
now sets it explicitly. The eight-token frozen-training-prompt diagnostic
captured four native rounds, 16 head and raw target-logit rows, and 53 raw
target-feature rows. Internal continuity, both preparers and the final
preparation-only bundle audit passed. A response audit joined all eight
output IDs to the seed, four round emissions and the terminal no-proposal
trace; the final sample lacks an independent logit check. See the
[CPU capture report](../experiments/recurrent-binary-cpu-capture-smoke.md)
for hashes and limits. A one-row microbatch repeat matched all captured
feature, head-state and target-logit values bitwise. The focused 77 recurrent,
16 capture-runner and nine native-adapter CPU tests pass. No training,
GPU/accelerator or Q4_0 comparison ran.
The real-model CPU adapter now accepts the pinned 2,560-hidden/4,096-Q
attention geometry and replays five first-round proposals. Both exact-order
and grouped-matmul paths matched all five native mapped top IDs; normalized
state drift reached about 0.0034 in one element, so numerical parity remains
unverified. A separate diagnostic two-position CPU SGD step produced finite
nonzero sign and scale gradients in all nine linears, exported a learned-scale
GGUF, and loaded it in the CPU native server for one matching eight-token
request. This verifies the joint gradient/checkpoint/export/loader path at
real dimensions, not training quality or Q4_0 performance.

**Eighth CPU milestone:** the same pinned one-prompt run with Flash Attention
disabled preserved all output IDs but changed native raw target features.
Replaying each capture's own features reduced first/fifth-depth state drift
under the non-flash path, without closing numeric parity. An independent
local Hugging Face CPU forward checked all 32 prompt embedding rows and
sampled target FFN weights bitwise against GGUF, then compared ordered layer
2/18/33 inputs with the native feature stream. Median relative row L2
differences against the non-flash stream were 0.579% / 0.382% / 0.280%.
The [CPU capture report](../experiments/recurrent-binary-cpu-capture-smoke.md)
records the per-row limits and hashes. These are alignment diagnostics,
not exact target-feature parity or SM75 results. The focused 80 recurrent,
16 capture-runner and nine adapter CPU tests pass. No GPU/accelerator ran.

**Ninth CPU milestone:** the forked native server now has a bounded opt-in
draft graph callback, published first in fork commit `b4df1b547`. On one
CPU request its completed trace left existing output IDs, head states,
target features and verifier logits byte-identical. A numeric join to the
first proposal shows decoder inputs, fusion norm and Q/K/V bitwise exact
after Q/K row conversion; RoPE differs by at most `3.55e-6`. The first
material gap is attention output: max `0.009986` with native Flash
Attention auto, `0.005002` with it off. Replaying attention from native
Q/K/V reproduces that gap. The [CPU capture report](../experiments/recurrent-binary-cpu-capture-smoke.md)
has source hashes and the mapped draft-logit comparison. This localizes
the drift but does not certify full parity or Q4_0 quality/throughput. The
focused 86 recurrent, 16 capture-runner and ten adapter CPU tests pass.

**Tenth CPU milestone:** a pinned accelerator-disabled diagnostic runner
captured one code and one reasoning training prompt alongside the earlier
prose prompt. Exact-prefix CPU cache rebuilding and grouped-matmul proposal
unrolls matched all **44/44** native mapped top IDs across **11 rounds**;
the largest normalized state-element difference was `0.004617`. Native
graph traces again put the first material gap at attention. The
[three-prompt report](../experiments/recurrent-binary-cpu-broader-diagnostic.md)
and ignored hashed summary retain per-category counts and limits. This
does not establish the 96-prompt capture, training quality or Q4_0
acceptance/throughput. No GPU or accelerator was used.
The focused 90 recurrent, 16 capture-runner and ten adapter CPU tests pass.

**Eleventh CPU milestone:** a later post-acceptance round from each of the
three training categories was joined to its native decoder graph. Rebuilt
CPU prefix inputs and unrotated Q/K/V matched native bitwise or within
`2.4e-7` at the fused-input boundary. The first material mismatch again
appeared at attention (maximum `0.00984`–`0.01175`); mapped seed top IDs
matched in all three joins. The [broader report](../experiments/recurrent-binary-cpu-broader-diagnostic.md)
records the individual rounds and hashes. Native stored K/V bytes remain
unread, so cache parity is not established. The recurrent CPU suite now
passes 91 tests. No accelerator was used.

**Current milestone:** the [final rescue/readout report](../experiments/binary-rescue-head-5080.md) records a negative result against Q4_0 EAGLE. Q4_0 reached 1.042 accepted drafts/round and 135.1 full-request tokens/s; the admitted attention+fusion Q8_0 rescue reached 0.696 and 83.3 (0.616×), and the fitted frozen-D-body FP16 head reached 0.555 and 60.8 (0.450×). All 120 primary measured raw outputs per path matched Q4_0 and target-only. The 264-request speculative round calibration, 92 graph-verified server blocks across final primary/diagnostic runs, and separate six-prompt longer-context comparison are complete. The longer-context client rates were 127.0 Q4_0, 68.5 combined rescue, 47.1 fitted head and 93.8 target-only tokens/s. Neither endpoint beat target-only. Full raw results, logs, binaries and model artifacts are indexed in the remote archive named in the report. Final RTX 5080 check found no owned process or compute app, 1,916 MiB whole-device use and 0% utilization. The user owns any future body-aware QAT or final-set decision; none was started.

The approved four-way A16 screen completed all **168 requests** (7 paths × 24 development prompts). A/B/C/D accepted **0.119 / 0.161 / 0.290 / 0.425 drafts per round**, versus **1.037 FP16** and **1.042 Q4_0**. Fitted group scales improved 3.585× over row-mean A, closing 33.20% of the Q4_0 gap. D beat all other binary candidates on every prompt but trailed both controls on every prompt. All seven paths matched target-only raw IDs on all 24 prompts. See the [report](../experiments/binary-scale-fitting-5080.md) for counts, depth survival, calibration, artifacts and limitations.

The scale-screen recommendation at that time was to retain D for a common-history body/readout diagnostic; the completed goal above supplied it. The [all-layer W1Ax suite](goals/w1ax-activation-precision-suite.md) and [RTX 2080 Ti quantization suite](goals/rtx2080ti-quantization-suite.md) remain completed and sealed; the prior [W1A1 research goal](goals/full-w1a1-eagle-project.md) remains checkpointed. The next scope/budget choice is user-owned.

**Final W1Ax result:** the twelve-cell development policy grid completed and
validated all **11,520 requests**, alongside the sealed historical/development,
operator, round, profiling, streaming and 720-request context measurements.
Every W1Ax path had lower pooled decode and full-request throughput than both
same-cell anchors. The largest observed decode ratios were 0.707× FP16 and
0.684× Q4_0. The [main report](../experiments/w1ax-activation-precision-results.md)
and [complete policy appendix](../experiments/w1ax-policy-grid-results.md) retain
fixed versus development-selected policies, raw-output differences, timing-control
variation and measurement limits. The interrupted 120-request attempt remains
preserved and excluded from completed-cell rates. Final verification found no
owned jobs or GPU compute apps; the 2080 Ti was idle. The goal file records exact
artifacts, hashes, checks and completion evidence.

Ubuntu 24.04 WSL2 and SSH are reachable through the shared host registry. The
RTX 2080 Ti (SM75) passed five native W1A1 CUDA backend
cases, a standalone 21-case/880-dot binary-MMA probe, and both integrated
portable/MMA eight-case backend gates. All five W1A1 draft GGUFs passed
source-row audits; the FP16 target track fits in 11,264 MiB VRAM.

The full [nine-variant Turing comparison](../experiments/rtx2080ti-quantization-suite.md)
completed 540/540 matched requests across 12 prompts and five repetitions.
Ordinary EAGLE decoded at 82.49 tokens/s. Native head W1A1 reached 74.81
(0.907× ordinary), and all-group W1A1 46.31 (0.561×). Q4_0 and Q8_0 draft
controls reached 91.51 and 87.73 (1.109× and 1.064×). Fusion, attention,
and FFN W1A1 also trailed ordinary. All eight speculative variants produced
identical text on all 60 paired requests; each differed from target-only on
one prompt. Ratios to target-only are timing observations, not strict
lossless speedups. Q4_0/Q8_0 stored types are verified, and a later isolated
trace confirmed their Q8_1 activation conversion and MMVQ/MMQ dispatch.

A separate 240-request four-path comparison found integrated binary-MMA head
W1A1 at 74.72 decode tok/s versus portable head W1A1 at 74.59, a 1.0017×
ratio with paired 95% interval 0.9973–1.0062. Both paths emitted identical
text on all 60 paired requests; no end-to-end MMA gain was resolved. The
genuine native W8A8/W4A4 300-request vector comparison has also finished
with verified CUDA dispatch. W8A8 decoded at 80.18 tok/s versus ordinary
EAGLE's 80.99 (0.990×; paired 95% interval 0.975–1.005), so no difference
was resolved. W4A4 decoded at 41.40 tok/s (0.511×), with accepted drafts
collapsing to 0.092/round versus ordinary's 1.168. The seven-path Tensor
Core comparison completed 420/420 requests with identical text and acceptance within
each default/MMA pair. W8A8 MMA decoded at 0.821× its default DP4A path
(paired 95% interval 0.817–0.824); W4A4 MMA at 0.880× its default vector
(0.876–0.884). Specialized matrix instructions were slower for this EAGLE
workload. A separate executed-path trace confirmed Q4_0/Q8_0 use Q8_1
activation quantization with MMVQ during one/two-token work and MMQ during an
observed 38-token operation. The goal file has exact commits, owners, and raw
hashes. The [final synthesis](../experiments/rtx2080ti-synthesis.md) and
packing-inclusive CUDA traces are preserved; final cleanup confirmed all
supervisors stopped and the GPU released.

## RTX 5080 result

A dedicated packed W1A1 operation, EAGLE output-head GGUF exporter/loader,
and CUDA dispatch were integrated in the 5080-tested llama.cpp revision
`92bc706`. The expanded five-setting revision is `8d2b18a`.
CUDA backend correctness passed 5/5 cases, all 32,000 packed head rows were
audited against the published BF16 source, and every packed benchmark server
logged actual CUDA XOR/POPCOUNT dispatch. See the
[integration report](../experiments/ggml-w1a1-cuda-5080.md).
A separate [captured-input parity run](../experiments/real-head-parity-5080.md)
checked 8 real drafter inputs against all 32,000 packed head rows on the
5080: exact sign packing and 256,000 integer dots, with no scaled-output
tolerance failures.

The [matched five-repetition comparison](../experiments/native-end-to-end-5080.md)
completed 180 requests on 12 fixed prompts with the same FP16 target and
server settings:

| Variant | Request tokens/s | Decode tokens/s |
| --- | ---: | ---: |
| Target-only | 96.67 | 99.39 |
| Ordinary EAGLE | 123.96 | 132.88 |
| Packed-head W1A1 EAGLE | 115.38 | 122.94 |

Packed-head W1A1 achieved **0.931× ordinary request throughput** and
**0.925× ordinary decode throughput**. Draft generation became faster per
round (4.386→3.515 ms), but accepted draft tokens fell (1.161→0.892 per
round), requiring 480 extra verification rounds. All five repetitions and
all three prompt categories favored ordinary EAGLE. The packed draft used
148 MiB less GPU memory while loaded. Both speculative paths exceeded
target-only throughput, but their outputs differed from target-only on two
prompts, so that ratio is not a clean lossless speedup claim.
An integrated-kernel Nsight Compute attempt was limited by
`ERR_NVGPUCTRPERM`; separate standalone packing-inclusive CUDA-event timings
are preserved, but no integrated per-kernel trace is claimed.

Ordinary and packed EAGLE decoded texts matched on all 60 paired requests.
A separate raw-token check found identical ordinary/packed IDs on the two
target-only mismatch prompts. An isolated [raw verifier-logit
trace](../experiments/native-verifier-trace-5080.md) reproduced both: at the
emitted rows, the target verifier itself ranked the speculative output first,
by 0.008074 and 0.000729 raw-logit units. The drafts were rejected, so these
were not wrongly accepted tokens. Target-only ranked the opposite IDs first
by 0.000963 and 0.016508 nats. Numerical sensitivity is plausible, but the
precise native baseline mismatch cause remains unproven. The earlier BF16
PyTorch verifier trace separately identified a tree-versus-incremental target
logit tie; see the [acceptance
report](../experiments/pytorch-w1a1-cuda-acceptance.md).

## Other completed gates

The [bounded head-only QAT pilot](../experiments/qat-head-pilot-results.md)
improved validation KL but reduced fixed held-out W1A1 acceptance to 1.565
drafts/round from the untrained 1.677. That recipe was stopped without
held-out tuning. The user's requested [W4A4/W8A8 accepted-per-round
comparison](../experiments/pytorch-int4-int8-cuda-acceptance.md) measured
0.2882 and 2.1816 respectively under the BF16 PyTorch verifier; those are
numerical simulations, not native INT4/INT8 timing.

A standalone SM75 binary-MMA probe cross-compiled to `BMMA.88128.XOR.POPC`
and later passed 21 cases/880 exact integer dots on the RTX 2080 Ti; see the
[2080 Ti suite](../experiments/rtx2080ti-quantization-suite.md). A focused
[related-work note](../experiments/related-work-note.md) keeps novelty claims
narrow: quantized EAGLE and native QAT already exist.
An opt-in [integrated binary-MMA
candidate](../experiments/integrated-binary-mma-5080.md) also passed 8/8
scalar-reference backend cases on the 5080, matched all 85 packed-draft
tokens in one model request, and compiled to SM75 SASS containing the exact
binary-MMA instruction. It subsequently passed the real SM75 backend gate and
tied portable W1A1 in the matched head comparison above.
The paired benchmark runner and analysis now support an opt-in fourth MMA
variant with separate selector/dispatch records and same-device MMA/portable
speed ratios. Local fake-server and analysis checks passed 15/15; the
[2080 Ti runbook](RTX2080TI_RUNBOOK.md) specifies the required run.

## Next research gate

The [post-suite consolidation](../experiments/one-bit-next-steps-2026-09-27.md)
updates the earlier agent brainstorms using the completed policy and round
measurements. It recommends recovering useful binary weights with wider
activations first: finish missing graph/state parity checks, then a train-only
fixed-sign row/group scale-fitting screen, with readout/body adaptation
conditional on its result. W1A1 remains the research endpoint. The subsequently approved scale-fitting screen is now complete (report above).
Body/readout adaptation and any expanded training budget remain user-owned;
reserved-final evaluation has not begun.

The [all-layer W1Ax study](../experiments/w1ax-activation-precision-results.md)
is complete, including the predeclared policy grid. Its report separates
acceptance, identical-input operator cost, complete round timing and serving
rates, with explicit measurement limits. Use its quality evidence to guide the [QAT revisit
plan](../experiments/qat-revisit-plan.md): audit
drafter-state/target-verifier alignment and target probability mass outside the
draft vocabulary before training, then test one bounded target-aligned recipe
on the frozen new development/final prompts. If acceptance improves, run native
same-device end-to-end comparisons against **both** FP16 EAGLE and Q4_0 EAGLE.
Later, screen non-EAGLE drafters such as block-parallel DFlash/DSpark before
investing in their W1A1 kernels. The completed 2080 Ti and 5080 experiments
are sealed, and the GPUs were released after their runs.

## Local research review (2026-09-25)

Seven Astra-high local-only analyses are collected in the
[ranked synthesis](../experiments/research-review-2026-09-25.md). They identify
target-aligned QAT capture, unused native cache-catch-up graph work, genuine
early draft caps, shared activation packing, structured scales, and a later
DFlash/DSpark quality screen as bounded opportunities. These are advisory
findings, not new performance results or changes to the active W1Ax protocol.
The initial review performed no GPU work or web search. A subsequent
[primary-source cross-reference](../experiments/research-cross-reference-2026-09-25.md)
revised the seven reports: DSpark becomes the lead later architecture candidate
with matched DFlash control; hard CE and structured scales remain hypotheses;
native sample-and-match is distinguished from probability-ratio verification.
The current W1Ax goal, one-run QAT budget and sealed final set are unchanged.
This literature review produced no new model or GPU result.

## PrismML / quantization research (2026-09-25)

The user-requested [PrismML research report](../experiments/prism-quantization-research-2026-09-25.md)
separates ternary task-score claims from true binary-weight execution and our
W1A1 acceptance objective. A full selected-weight geometry audit and a bounded
local CPU Q1 control support investigating representation fitting and
head/body adaptation before new kernel work. Existing Q1_0 support was verified;
the naive grouped control is not an optimized Prism checkpoint.

The active SM75 suite, its GPU owner, frozen protocol and reserved final prompts
are unchanged. New fitting/QAT budgets and any practical weight-only branch
remain proposals for the user, not additional experiments started by this review.

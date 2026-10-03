# A8 QAT recovery and comparison

## Standalone ownership claimed — October 3, 2026

Coordinator `01a103da-0980-7332-a041-3f95aca6a3f5` owns this goal exclusively.
The originating chat and interrupted workers are uninvolved. Existing heartbeat
`a8-qat-recovery-monitor` is retargeted here and ACTIVE at the same 15-minute
cadence. No duplicate monitor. RTX2080Ti, A1 and unrelated research stay paused.

New bounded team:

| Worker | Ownership | Preserved checkout |
|---|---|---|
| `/root/trainer_recovery`, Sol high | Trainer, prepared adapter, standalone evaluation, exact resume and cumulative budget | `/private/tmp/eagle-qat-a8-integration` |
| `/root/recipe_recovery`, Sol high | Matched recipes, effective configuration and parameter-family movement | `/private/tmp/eagle-qat-a8-recipe` |
| `/root/gpu_supervisor`, Luna high | Independent validation, sole tmux MCP/RTX5080 deployment and supervised runs | `/private/tmp/eagle-qat-a8-operator` |
| Root | Readiness review, integration, durable records and existing monitor | `/private/tmp/eagle-a8-standalone` |

Selected-lane readiness commit99a53ee is reviewed and adopted as a8c9986;
all50readiness CPU checks pass, including the6new selected-lane checks. Ownership
and readiness are integrated/published on main3e12ad6. Partial trainer and recipe
implementations are preserved by workera4a66fa and3d5fafa respectively.

A fourth bounded Sol worker `/root/native_request_metrics` owns a NEW native
request timing helper and tests, with trainer owner stitching its API into the
standalone evaluator. Existing tensor captures measure acceptance only; true
full-request timing must be measured separately within the same1200s evaluation
cap, with warmup/five repetitions and alternating variant order.

Fresh sole-operator read-only check confirms same WSLboot517c4a36, RTX5080/SM120,
zero utilization, no remote_job/trainer/native/pytest process or dedicated host
tmux sessions, approximately20.27GB available host RAM and357.34GB disk. Native
binaries and retained preparation-ready/zero-checkpoint artifacts are located;
prior receipt/config identities match. Remote main dirty source and untracked
checkouts remain untouched. No new model or optimizer has run.

Reference latent magnitude0.5 matches original dense-sign initialization;
candidate0.1 retains the same initial deployed signs/scales with zero midpoints
and initial learned clips. Next: complete/test integrated recipe+trainer, then
one decisive actual-model/native/resume validation and the fixed-budget matched
comparison. Failed post-checkpoint work must remain charged.

### Integrated review checkpoint

Recipe worker committed3d5fafa/20c2cad; root adopted23a0f5d/f8c504a.
Root21CPU recipe/cache-head checks plus Ruff/diff pass on Torch2.14.0.
Readiness50checks also pass on the project Torch2.14.0 runtime after supplying
the existing gguf-py module path. Recipe tests establish identical deployed
initial signs/scales; zero-affine CPU logit drift max1.19e-7 is bounded by
1e-6relative/2e-7absolute tolerance and identical argmax. No deployment failure
is inferred from that floating difference.

Root found the historical development evaluator uses frozenb4 while the enabled
learned/affine native support belongs to9e2. The team is adding explicit
`evaluation_native` metadata for current supported executable/libraries and
separately preserving original teacher/capture binary ancestry. Model/verifier
precision and request settings remain frozen. New native evaluation enables
shared packing and unused-head pruning for both arms; Torch cache/head audit
remains distinct, including the learned-head serial-gradient exception.

Crash-budget implementation preserves unresolved trainer charges and excludes
normal standalone evaluation/startup. Conservative uncertain downtime charging
on the same boot, or unresolved remaining reservation consumption after reboot,
must be disclosed if recovery uses it. New optional-family movement evidence is
measured by optimizer hooks, bounded samples and per-tensor finite gradients;
first-update float32 nonmovement during warmup cannot alone prove a broken
parameter family. A bounded early-update family gate retains actual proof.

### Matched gradient arithmetic correction

Sole-operator independent review found `a1_computation=single_forward` also
selects the alternate learned A8 VJP. The prior assumption that this flag is
irrelevant to A8 was wrong. Recipe commit2a3c489, adopted29ee3b3, selects
`reference` for both arms, preserving the agreed gradient baseline and all
requested candidate features. The execution exception for learned-head serial
training remains explicit. Cheap execution observations run each round; dense
recipe audits/packed sign telemetry run at initialization and diagnostic cadence.
Movement admission accumulates finite/nonzero gradients and measured displacement
for every enabled family across at most the first100budgeted updates, preserving
per-tensor evidence and exact checkpoint state.

Native timing helper49c7511 is adopted43ed317; its11CPUchecks are pending root
integration. The mandatory same-target no-speculation timing reference is being
added within the SAME1200s evaluation deadline; Q4_0 remains primary. No extra
training/evaluation budget or new research arm is selected.

### Integration ready for current-model execution

Root integrated trainerf30dad6/3272936 as5e3276a/a16bff5, on top of selected-lane
readiness, corrected recipe, accumulated movement, and native timing/target-only
work. Root82integrated CPUchecks plus7latest launch-contract checks pass on the
project Torch2.14.0 runtime with scripts/tests/src and populated gguf-py imports.
Ruff and diff checks pass. Worker independently reports167affected checks; all
actual GPU evidence is still pending. Current branch progress is published;
main integration publication is the next action before immutable deployment.

Prepared-data fixes preserve exact original receipt/data bytes: source6f's known
stages-wrapper audit hash may be reused only with a pinned historical receipt,
matching all actual data/runtime/dependency bytes, validated historical full-pass
provenance, and unchanged semantic auditor AST. The data child is initialized
under the original fixed teacher contract; actual current candidate linears,
optimizers and admission use the untouched candidate config. TRAIN and
standalone development use this explicit teacher/actor distinction.

Current gate provider: `train_prepared_continuous_w1ax:create_current_provider`.
Its CPU JSON record schema is `current_prepared_provider_v1`, with pinned arm
config locator, optional stages_manifest locator, original prepared_run_dir and
ready SHA, unique validation_run_dir and selected_shard0. Keep the production
spec.provider declaration frozen; pass this factory/binding via gate CLI flags.
Do not use the legacy `train_prepared --validate/--start` source6f launcher or its
`prepared_provider_binding_v1` emission for the new recipes.

Run sequence per arm: current native/model/backward/memory admissions; fresh
`train_continuous_w1ax --start --prepare-only --allow-cuda` with original prepared
corpus flags; native standalone step-zero development; exact resume into actual
7200-second training. First100budgeted updates prove enabled family movement;
then graceful `--stop` saves current state, verify group/context return and exact
`--resume` under UNCHANGED production config before continuing. No temporary
max_steps change or separate disposable trial. All updates remain charged to
the same arm. Scheduled5000/final evaluations retain the same1200s bounds.

Current evaluation runtime metadata is in ignored
`runs/qat-a8-recovery/evaluation-native-runtime-9e2.json`: supported9e2 executable
SHA1ca0c1d9..., exact immutable manifest/library pins and native_commit. This is
static identity evidence, not native CUDA readiness. Sole operator will record
actual process/run/source identities and effective CUDA deployment next.

### Sole-operator pipeline GO

Frozen execution source is published583480c79f3ca090de0952deca8278dcb7f15c2d;
main integration is8583b68. A concurrent independent CPU feature added files to
main; they were preserved, and the A8 deployment stays on the reviewed exact
source583480c. No old-chat coordination occurred. Final malformed-receipt/import
fix passed7root checks. The root integrated82checks and worker167affected checks
remain positive; no actual optimizer/GPU launch is inferred from CPU evidence.

Root authorized `/root/gpu_supervisor` to deploy that source into a new immutable
checkout, perform fresh source/runtime/resource/ownership checks, and progress
through native+model gates, each fresh step-zero native development evaluation,
first100budgeted realupdates/graceful stop/exact resume, and the matched7200s arms
without repeated root confirmation. Existing gates must pass; max2recoveries per
incident, raw failures preserved, exact owned groups/contexts released before
retry, no scientific/feature/budget fallback. The existing monitor remains ACTIVE.
The operator restored the unexpectedly clear RTX2080Ti local pause flag without
connecting to that host; this does not establish remote RTX2080Ti resource state.

## Standalone chat handoff — October 3, 2026

The human corrected the execution structure: “just launch it as a seperate codex
task and dont touch it i dont want any cross incolvement”. The historical
coordinator chat has interrupted all four native workers and PAUSED its
`a8-qat-recovery-monitor`. It will create one standalone project chat with this
handoff, then perform no further coordination, messaging, monitoring or edits
for this goal. The new chat is authorized to own the goal/team, fix integration,
run the comparison and retarget/rewrite the paused heartbeat to itself. It must
claim ownership in STATUS/this file/ignored registration and use its own bounded
workers. Do not contact the old coordinator or reuse its interrupted workers.

Preserved implementation state (do not discard):

- `/private/tmp/eagle-qat-a8-integration`, branch
  `feature/qat-a8-integration-20261003`: uncommitted changes to continuous_qat.py,
  train_continuous_w1ax.py, train_prepared_continuous_w1ax.py and
  w1ax_continuous_stages.py, plus new test_a8_continuous_integration.py. Implements
  activation_bits=(8,), current authenticated PreparedProvider reuse and
  standalone evaluation request/result binding. Targeted tests were in progress.
- `/private/tmp/eagle-qat-a8-recipe`, branch `feature/qat-a8-recipe-20261003`:
  uncommitted configs/qat_a8_comparison.json and qat_recipe_audit.py. Recipe
  materialization, effective attachments/optimizer audit and telemetry draft;
  tests not yet verified. Preserve partial work and review before adoption.
- `/Users/pippo/github/binary-eagle-decoding-a8-readiness`, branch a8-readiness:
  clean committed `99a53ee3dfc6447566e23262c2a40c732ed8b9c7`, four owned files;
  selected-lane readiness/native collector, 59 focused CPU tests plus Ruff/diff
  pass. Not integrated into main. Independent integrated verification remains.
- `/private/tmp/eagle-qat-a8-operator`, branch
  `feature/qat-a8-operator-20261003`: clean at83a3153. Operator's fresh read-only
  evidence found retained corpus and original zero checkpoint intact. Registry
  address was reachable via tmux MCP; RTX5080 idle/no project compute process,
  WSL boot517c4a36-e475-4a5f-9fa6-65de57edc6fe, ~20.08GB host available and
  357.35GB disk. `.wslconfig` has20GB and instanceIdleTimeout=-1. Remote main has
  preexisting dirty source/untracked dirs: preserve and deploy new checkout.

The last operator observations did not start a CUDA model or optimizer. Fresh
availability must still be checked by the new sole operator; do not infer a
perpetual free-GPU claim. RTX5080 local pause flag was resumed for this authorized
goal, RTX2080Ti remains paused. The new team should verify no leftover owned
test/operator commands before acquiring the GPU; old worker interruption is not
a remote process teardown claim. Original data/checkpoints/raw failures and
untracked overnight research remain untouched.

Outstanding root review: ensure failed work since the last checkpoint remains
charged to a durable cumulative time budget; exact resume must not refund it.
Keep the documented learned-head serial-training exception. Do not repeat full
corpus semantic audit on each restart. Reuse existing operator machinery rather
than building a new chain of permission wrappers.

## Human authorization — October 3, 2026

“launch a team to do that for qat, fix it, then run with a monitor that will heal
if things break. starting with a8 first, we will hold out a1.” Implementation,
targeted tests, RTX5080 validation/training and bounded recovery are authorized
without another confirmation. This is the only active goal; the previous goal
remains historical and complete for its fixed reference recipe only.

Coordinator: chat `01a103da-0980-7332-a041-3f95aca6a3f5`. Historical ownership below is superseded by the standalone claim. No A1 model/optimizer/
training/evaluation; RTX2080Ti and unrelated architecture research remain paused.
Demonstrate the effective A8 recipe through actual updates, save/resume/export/
reload and native development comparison against Q4_0, including complete-request
throughput. Frozen target/verifier precision and sealed finals are unchanged.

## Selected experiment contract

- Reuse authenticated 10,000 TRAIN prompts / 3,899,930 supervised rows and
  unsealed development data. No recapture or repeated semantic auditing of trusted
  completed data; fresh integrity checks are allowed. Preserve token/feature/
  teacher/cache/mask/vocabulary ancestry.
- Two fresh A8-only arms with the same seed, data order, hard-CE objective,
  initial deployed signs/scales and update cadence. Preserve the original paired
  step1000 unchanged; do not reinterpret it as a new recipe's exact resume.
- Shared execution: enable cache/head optimization and actual supported A8
  computation savings. A1 single-forward controls are irrelevant to A8. Learned
  activation training currently requires serial head execution to preserve
  invocation-local LSQ gradients; disclose this deliberate effective-path
  difference and validate it rather than silently bypassing the quantizer.
- Control: fixed A8 activations, symmetric binary weights, baseline AdamW latent
  initialization and movement rules. Candidate: learned A8 activations, all-nine
  affine weight midpoints, existing lower-inertia latent magnitude 0.1, AdamW
  sign LR 0.001 / scale LR 0.00001 and norm clip 1. No direct-bit Bop, fusion
  correction, depth weighting, curriculum or trajectory refresh in this initial
  comparison. This tests a combined candidate, not isolated feature attribution.
- Initial training cap: 7,200 cumulative trainer-accounted seconds per arm,
  14,400 total, preserved across standalone evaluations and failure recovery.
  Each evaluation is separately bounded at 1,200 seconds. Evaluate both step-zero
  arms, scheduled checkpoints (proposed every 5,000 updates; save every 1,000)
  and final checkpoints; record actual overhead and distinct prompt/row coverage.
- Fail launch on requested/effective recipe, lane, module or optimizer mismatch.
  Measure sign/scale/midpoint/quantizer gradients and movement, sign flips/
  flip-backs/near-zero distances and effective cache/head execution. Current
  native/backward/memory/save/export/reload/positive-step resume proof is required
  for enabled options, with a short decisive integrated validation path.
- Standalone evaluation releases live trainer/model/optimizer before allocating
  evaluation resources. Preserve source/runtime/hardware/data/config/checkpoint
  identities and raw failures. Never treat evaluation elapsed time as serving
  throughput or CPU evidence as GPU performance.

## Team and ownership

Workers use isolated temporary worktrees and preserve other edits. Root reviews,
integrates and pushes tested commits and owns goal/status/decisions/monitor. No
concurrent GPU use or parallel edits to the same files.

| Worker | Owned responsibility | Checkout |
|---|---|---|
| `/root/a8_integration`, Sol high | Single-lane trainer/CLI, prepared reuse, standalone evaluator/resume and new tests | `/private/tmp/eagle-qat-a8-integration` |
| `/root/a8_recipe`, Sol high | New comparison configs, effective recipe audit, movement telemetry and tests | `/private/tmp/eagle-qat-a8-recipe` |
| `/root/a8_readiness`, Sol high | Existing readiness/native collector selected-lane adaptation and new tests | `/Users/pippo/github/binary-eagle-decoding-a8-readiness` |
| `/root/a8_operator`, Luna high | Independent verification and sole RTX5080 deployment/run/evaluation/recovery operator | `/private/tmp/eagle-qat-a8-operator` |

Consult docs/AGENT_OPERATIONS.md and relevant USER_LESSONS. Reuse integrated
evaluator reconstruction and host-save accounting; relevant research is
experiments/parallel20261002/sign_inertia/report.md. Preserve unmerged research
and ignored artifacts. SSH only through tmux MCP using the shared local registry.

## Monitor and healing contract

ACTIVE heartbeat `a8-qat-recovery-monitor`, every 15 minutes, is attached to this
coordinator and continues from this file and
ignored A8 monitor registration. Stay quiet on unchanged healthy/non-actionable
state; notify verified training start, meaningful milestones, failures/recovery,
completion or required user action. Idle agent turns and disconnected transport
do not establish job failure. New human pause/stop overrides recovery immediately.

On genuine failure, retain raw error and committed checkpoint, verify exact owned
process identities/groups and release GPU contexts before retry. At most two
automatic retries for one incident, charged to the original cumulative budget.
Resume optimizer/RNG/cursor/data/recipe exactly. Infrastructure reconnection and
resource-safe evaluation can recover without math changes. A concrete code bug
can be fixed with targeted tests and newly published source identity; record
lineage and prove resume compatibility. Never silently disable requested
features, increase budget, change precision/objective/seed/data, recapture or
pivot architectures. Persistent deterministic failure or an incompatible
scientific change requires a factual report and user decision.

Use one operator; never duplicate SSH/GPU operators or infer rotated addresses.
On completion verify resource release, publish evidence, pause heartbeat and
mark durable goal complete. Do not automatically archive the human chat.

## Initial launch checkpoint

Main baseline `83a3153`; four workers dispatched. Existing untracked overnight
research is preserved. Operator resumed only RTX5080 using the shared registry.
Fresh WSL boot and RTX5080/SM120 availability verified: no project GPU compute
process, 0% utilization, about 20.08 GB available host RAM and 357.35 GB disk.
Preexisting remote main changes are preserved; deploy a new immutable checkout.
Retained prepared corpus was located. No new CUDA model or optimizer has run.
The original below-Q4_0 result is historical and lacks a matched step-zero control.

Monitor creation succeeded. Registration is saved outside Git at
`runs/qat-a8-recovery/monitor-registration.json`. This local registration records
the worker identities, sole GPU owner, initial budget, retry cap and phase;
actual-model/optimizer flags remain false until observed evidence exists.

## Native request timing helper — standalone bounded feature

`/root/native_request_metrics` owns only the new
`scripts/a8_native_request_metrics.py` and focused tests in
`/private/tmp/eagle-a8-native-metrics` (branch
`feature/a8-native-metrics-20261003`). Existing trainer/stages files are untouched.
The deliverable reuses `benchmark_native_eagle` parsing/count aggregation and
`run_binary_head_capture.server_command` for the exact24development request
policy (F16 target/draft KV, greedy seed42, max128 outputs, draft5/pmin0).
Tensor capture and trace environment hooks are absent. Both native arms require
current `9e2c7a90051e738751aab7d7bd7c2d8201fb76e3` with shared pack and unused-head
pruning; historical teacher b4 source remains separate and is refused for timing.

Callable: `measure_a8_requests(sources, prompts, draft, output,
deadline=caller_absolute_monotonic_deadline, stop_file=STOP)`.
Five alternating repetitions each run both A8/Q4_0 with one warmup per server:
240measured requests and10warmups. The helper shares the evaluator's1200s
aggregate deadline and cannot start a new budget. Raw request/response/counts,
HTTP wall, server prefill/decode spans, IDs, kernel markers and mapped runtime
are retained. Aggregate count/time ratios and distributions lead with Q4_0;
TTFT is explicitly unavailable from the nonstreaming endpoint. Incomplete runs
publish no performance ratios, retain raw failures and exact teardown evidence.

Acceptance check:11focused CPU tests passed, including full synthetic contract,
count/time aggregation, frozen policy, historical-runtime rejection, no capture
hooks, deadline/STOP handling and real local owned-process-group cleanup without
killing an unrelated process. Ruff and diff checks pass. No SSH/GPU or measured
performance result. Remaining integration: trainer calls the helper after export
using derived current evaluation sources (explicit commit/env/runtime inventory),
rejects incomplete timing, binds its manifest and attaches actual hardware evidence;
sole Luna executes within the existing evaluation deadline.

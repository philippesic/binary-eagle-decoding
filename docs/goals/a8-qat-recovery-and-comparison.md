# A8 QAT recovery and comparison

## Human authorization — October 3, 2026

“launch a team to do that for qat, fix it, then run with a monitor that will heal
if things break. starting with a8 first, we will hold out a1.” Implementation,
targeted tests, RTX5080 validation/training and bounded recovery are authorized
without another confirmation. This is the only active goal; the previous goal
remains historical and complete for its fixed reference recipe only.

Coordinator: chat `01a103c6-9cbd-7e70-ab05-7de1c49acf79`. No A1 model/optimizer/
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

One 15-minute heartbeat attached to this coordinator continues from this file and
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

# Early CUDA validation during CPU dataset audit

## Latest human authorization

The human instructed on October2: if preparation is CPU-only, have the optimizer
owner run its GPU tests now, then start QAT once all required training data is
verified. This supersedes the requirement to finish the entire preparation
endpoint before CUDA validation. It does not allow optimizer updates before
current launch gates or concurrent GPU computation without coordination.

QAT/validation owner01a0f934-dd65-7e33-a5bf-0ba591e713a4 and current preparation
owner01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed have received direct coordination
instructions. RTX2080Ti remains paused. All SSH must use tmux MCP and the shared
host registry. Existing QAT overnight/on-credits protection remains in force.

## Validation reservation

Completed shard audit is CPU code, but the preparation process automatically
returns to native GPU capture for missing development shards and later paired
CUDA smoke. A CPU phase/status alone is not proof of released CUDA contexts,
available memory or a safe overlapping resource schedule.

The two owners must establish a bounded exclusive RTX5080 validation reservation
with one coordinated fresh process/GPU/memory/host-resource observation and a
recorded lease owner, deadline, next preparation GPU phase and release procedure.
Continue CPU audit if its host-memory requirements fit alongside validation.
Guarantee validation process groups are stopped before preparation starts GPU
capture/smoke. If a transition hold is necessary, preparation owner may coordinate
a reversible hold of its live process at a CPU boundary, avoiding restart and
repeat audit. Preserve frozen source/config/runtime/data and distinguish a hold
from failure in monitoring/recovery. No killing/restarting the preparation job
or silently modifying its math/gates/source to create a reservation.

If safe overlap cannot be established, state the exact resource/transition
constraint and prepare the new build/launch on CPU while waiting for a valid
GPU slot. This new instruction is not an unconditional claim of GPU availability.

## Tests and later training

Use a separate immutable checkout/run for the published new native runtime and
recipe. Begin with CUDA compilation and exact-pack/operator/nonzero learned,
correction and affine fixtures. Actual-model native decisions and zero-update
full-model backward/memory/timing use independently audited eligible unsealed
TRAIN operands with exact ancestry and provider admission. A partial corpus is
not silently marked fully training-eligible. Preserve failures and all precision,
mask/cache/verifier/ownership gates; no real-data optimizer updates in validation.

After all required corpus capture/audits, current recipe CUDA/native receipts,
full preparation gates and the exclusive training GPU handoff are verified, the
QAT owner starts the already-authorized training in a new validated run without
further confirmation. Preparation run remains --prepare-only and stops before
its optimizer loop. Neither GPU overlap nor an early test pass is data readiness.

The QAT owner owns its existing heartbeat/runbook/handoff and lease evidence.
Parent owns this scope checkpoint plus STATUS/active goal. Fresh actual lease,
validation starts/results and training starts must be checkpointed by the owner;
dispatching these instructions is not proof that GPU tests have started.

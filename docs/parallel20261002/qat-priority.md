# Protected QAT coordination — October 2, 2026

This is supporting work under the active QAT optimization readiness goal, not a
new research goal. Deliverable: restore the acknowledged preparation and QAT
owners and their protected supervision, then pursue the existing safe validation
and training sequence. Acceptance: actual active owner turns, active correctly
targeted recurring supervision, exclusive GPU ownership, and evidence-backed
gates before any CUDA or optimizer claim.

## Restored owners and supervision

The human's latest parallel-team request protects QAT and its required preparation
and GPU supervision through the supporting CPU research usage cutoff and actual
weekly allowance reset. Available paid-credit continuation is authorized;
purchasing credits and redeeming resets are not authorized.

The existing acknowledged owners were archived. They were unarchived and woken
with the current human instruction; app snapshots verify active turns:

| Responsibility | Owner chat | Protected heartbeat |
| --- | --- | --- |
| Validation and later training | `01a0fc3d-bbe1-7e93-a19b-a9200dfa186c` | `qat-validation-and-training-handoff` |
| Sole preparation/preflight operator coordination | `01a0fdd6-8e11-7393-9aed-5c99bd08e428` | `a8-a1-luna-health-and-recovery` |
| Interrupted transport diagnosis only | `01a0fb34-e010-7b11-ac9a-f72cf2367c6c` | No additional heartbeat |

Both old protected automation IDs were absent. App update calls explicitly
reported that the automations did not exist. Replacement creation succeeded:
QAT retained its old ID; preparation received the new ID above rather than the
deleted `a8-a1-health-check-enable-after-manual-start`. Actual local TOML readback
verifies ACTIVE heartbeat kind, 15-minute cadence, and exact owner targets for
both replacements. Neither action received an approval rejection. Preparation
owner owns consistent registration ID/reference updates, preserving old history;
the coordinating agent does not race its ignored registration edits.

Preparation registration readback at 18:34:43 UTC confirms the new ACTIVE
heartbeat ID, correct successor target, cadence, and restoration-proof path.

The QAT coordinator restored scheduling and ownership only. It did not create a
second GPU operator, take over frozen preparation, run SSH, signal a remote
process, change a recipe, or claim a GPU result.

## Actual execution boundary

Latest actual saved remote observation remains historical October 2, 14:06 UTC:
original preparation child 676 was actively auditing the complete 10,000 train /
1,002 development corpus across 353 manifests, with zero optimizer updates and
no final readiness receipt. Subsequent connectivity failures establish unknown
remote health, not GPU availability or training failure.

RTX5080 remains preparation-owned. Fresh local controls permit it; RTX2080Ti
remains paused. Validation still needs fresh exclusive identity/resource/context
proof and a bounded hold. Training separately needs current actual CUDA packing,
full-model/native/gradient/memory/recipe gates, final complete preparation receipt,
terminal preparation supervisor, and fresh GPU handoff.

The successor metadata rebind passed the same 20 local mocked safety tests with
unchanged controller/guard/entry code hashes. That is CPU process-safety evidence,
not CUDA readiness. QAT owner is resuming renewal of ONE fixture-only conditional
authorization, now renewed; preparation owner is mechanically preparing the matching transport
packet. Frozen `--prepare-only` job and numerical/ancestry gates remain intact.

## Transport handoff and next action

The recorded parent-owned LOCAL198 transport diagnosis was still active at first
inspection and blocked all new remote dispatch. Its archived owner was reopened
and asked for explicit completion/cancel/transfer with raw proof. Its resumed
turn captured and closed only its owned local tmux session and explicitly
released the lock. Fresh lease readback at 18:35:29 UTC records diagnosis inactive,
completed, and transferred to the sole preparation operator; operator-active is
false. Raw `parent-transport-diagnosis-20261002.json` SHA256
`6711fe18e9de9992a28113f4ea135f07bae9dd033c6de20aff98322d29198c8f`
was independently checked against its actual bytes. Proof preserves zero
successful SSH connections, no CPU checker or GPU query, no remote job/signal,
and closed LOCAL198. It establishes no successful persistent connection procedure
or physical host-unavailability claim. Both current owners received the explicit
handoff and raw proof reference.

Next: let the SAME sole preparation operator consume the exact handoff and execute the
reviewed packet only after renewed owner authorization and fresh safe exclusive
preflight. Consume its result; preserve any failure. Continue protected owner
supervision and restart needs through credits and allowance reset.

Integration: checkpoint/report branch `feature/qat-priority-20261002` in
`/private/tmp/eagle-qat-priority-20261002`; orchestrator should integrate the report
and link it from the existing active goal/status. Shared goal/status edits remain
with the orchestrator and existing QAT owners.

## Learned-head batching risk — 18:43 UTC assessment

The LSQ research team reports 72 synthetic CPU cases through the actual
`forward_torch_round` / `NativeStepAdapter` / installed `RowBinaryLinear` head:
serial and stacked-head logits/loss match, while trainable activation clip or
threshold gradients differ by the square root of valid depth, with optimizer
step differences up to 0.002749. Invocation-local normalization explains this
computation-control equivalence issue. This is CPU evidence, not CUDA behavior
or a QAT quality result; the research implementation remains isolated.

The frozen preparation config's actual SHA256
`4ee4ce05139e0ae4762dfa35b6546b79a3fafc8c2b91be07b6ef76e9f5579c8e`
matches its registration pin exactly. It omits activation quantization and head
optimization controls, whose current defaults are fixed activations and serial
head. The existing frozen prepare-only job is not exposed to this combination;
no frozen source or recipe change follows.

New training's selected recipe is still null. The prepared `learned-activations`
and `combined-contract-smoke` profiles explicitly enable learned activations and
`optimize_head=true`, so this risk must be resolved before those recipes receive
training admission. Fixed-activation profiles are not implicated by this finding.
The actual all-nine activation bank includes the installed output head, and the
observed adapter forwards linears, making the production path relevant.

Astra's focused advice: the smallest reference-preserving option is serial head
execution when gradient recording is enabled and its attached activation scalar
requires gradients, retaining batching for fixed/frozen/no-grad cases. A shared
chain normalization would deliberately change the serial optimization recipe.
No remedy is selected or integrated here. Current QAT owner received the evidence
and profile assessment; require source/config-bound gradient/update equivalence
or a reference-preserving fallback before enabling the affected combination.
The single packing fixture and existing actual-model/native/memory/full-prep
gates remain separate; no synthetic or CPU result admits optimizer updates.

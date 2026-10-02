# QAT validation and training ownership

## Latest human authorization

The human assigned chat01a0f934-dd65-7e33-a5bf-0ba591e713a4 responsibility for
training: monitor the existing dataset-generation owner for GPU release, then
start validation and later training. This explicitly supersedes the earlier
preparation-only restriction for this chat **after successful current validation**.
It does not authorize optimizer updates before validation, interruption of the
preparation run, changes to frozen target/verifier precision or sealed-final use.

The current preparation owner is derived from the acknowledged
monitor-registration.json coordination_handoff and confirmed against its existing
heartbeat target. Successorchat01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed currently retains
RTX5080 until its preparation endpoint and verified release. Its source/native/
config/data identity stays frozen; it must stop before optimizer updates. The
training owner uses a new source-bound project/run and audited data ancestry.
RTX2080Ti remains paused. No training recipe is selected or launched yet.

## Persistent observation

Heartbeat **qat-validation-and-training-handoff**, ACTIVE every15minutes, targets
the training owner chat. It observes the current preparation-owner chat using compact wait
snapshots and the existing ignored preparation registration/last-health. Cursor,
authorization and phase are saved under ignored
runs/qat-optimization-readiness/training-handoff-registration.json. The existing
preparation health/recovery schedule remains independent; this monitor neither
replaces it nor repeats healthy GPU checks. Stay quiet on unchanged healthy
progress, notify verified release/new failure/required action and actual training
start. An idle chat or completed turn is not GPU release.

## Release and validation sequence

1. Require successful full preparation, audits, coverage, paired smoke and
   preparation-ready receipt, zero global/A8/A1 optimizer counters, terminal
   supervisor, absence of owned process groups and fresh GPU/resource proof.
   Failure, unavailable SSH, stale observation or incomplete data is not release
   to training. Derive current paths and IDs from registration, not historical
   supervisor IDs. Use tmux MCP for every SSH connection.
2. Assign one GPU operator. Use a separate project checkout/run under the
   registry workdir and published parent/native revisions. Do not update the
   live/frozen preparation directory. Keep native8025a0777 source and actual
   CUDA compiler/architecture/binary provenance bound to measurements.
3. Execute the runbook's exact-pack/operator, nonzero-option and actual-model
   native-decision gates. Produce measured zero-update full-model forward/
   backward and memory/timing evidence, including every enabled curriculum
   precision and optional parameter family. Larger batches remain measured
   independent graph probes until any changed optimizer cadence is validated.
4. Resolve failed implementation/numeric/cache/gradient/memory gates and retest;
   never weaken ancestry, claim CPU proof as CUDA or inherit old frozen source
   receipts. Preserve raw failures and exact effective deployment identity.
5. Record the selected passing training recipe, budgets, initialization, dataset
   versions and source/native/runtime/hardware hashes before optimizer launch.
   Enabled asymmetric midpoint/sign/learned quantizer/correction/curriculum
   controls must each be covered by current evidence. Do not silently enable an
   unvalidated option or select a purported winner from unmeasured quality.
6. Start the already-authorized training in a new run after all launch gates
   pass. Use detached Linux host tmux with remote_job.py, exact checkpoint/resume,
   sign/midpoint/quantizer diagnostics and Q4_0 development comparison. Verify
   disconnect/reconnect survival once. This heartbeat then follows owned
   validation/training progress; do not create a duplicate schedule.

The human need not reconfirm training after these gates: this handoff instruction
is the authorization. Any newer human pause/stop takes precedence. Report actual
first optimizer updates and meaningful validation, failures or required actions;
no convergence, acceptance gain or throughput claim follows from launch alone.

## Current baseline

Implementationeb66093, native8025a07773b7828bdeb4f3e0b834c8b54cb65c66;952CPUtests
withfourskips and native CPU fixtures pass. All12 prepared profile CLI plans pass.
Actual optimization-feature CUDA receipts are still absent. Fresh bounded CPU check at2026-10-02 06:03:52.658738UTC was healthy,
reaudit52/353,retained10,000train/224dev/327manifests,zerooptimizer,readyreceipt:null.
Supervisor674 andchild676 remainlive; release is notproven. The ignored
handoff-observation-20261002.json records rawcheck and localtransportcleanup.


## Overnight QAT protection

Confirmed from the direct human message in overnight coordinator chat
01a0fb34-e010-7b11-ac9a-f72cf2367c6c: QAT supervision/validation/training must
continue all night, including available paid credits after included allowance
exhaustion. Supporting research's usage threshold, allowance reset and morning
cutoff do not apply to this chat, its workers, owned GPU jobs or heartbeat.
Never stop this monitor solely because research stops or included usage resets.
No account reset redemption or credit purchase is authorized by this instruction.

The same current validation/data/cache/frozen precision/sealed-final/ownership
requirements still apply. The dataset owner retainsRTX5080 until verified
successful preparation/release. Research teams are isolated CPU work under
docs/OVERNIGHT_RESEARCH.md and cannot change live source/recipe or take GPU
ownership. Newer direct human pause/stop remains authoritative.

Heartbeatqat-validation-and-training-handoff was updated in place with this
protection, preserving its15minute cadence,target,name and notification policy;
confirmedACTIVE. No duplicate schedule or source/runtime change. Saved current
preparation observation from its owner at2026-10-02 06:13:55.765738UTC is healthy,
reaudit67/353,retained10,000train/224development,zerooptimizer,nonterminal. This
local saved observation is not a fresh release proof; validation remains queued.


## Preparation-owner rotation followed

Preparation supervision transferred from01a0f47a-e246-75e1-a299-fcac42d34f8a to
acknowledged successor01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed. Registration confirms
acknowledged/automation_target_verified and native preparation heartbeat target
matches the successor. QATheartbeat was updated IN PLACE to derive future owner
rotations rather than keep polling a retired chat; its own target/cadence/name/
notification preferences and all launch gates are unchanged. Per-chat cursors
are retained in training-handoff-registration.json. No duplicate schedule.

Saved owner health06:56:05.023881UTC is healthy/nonterminal,reaudit130/353,
10,000train/224dev retained,zerooptimizer,no readyreceipt,supervisor08running.
The GPU job remains the same; supervision rotation is not release. No remote
query/sourcechange/training action occurred on this routine handoff observation.

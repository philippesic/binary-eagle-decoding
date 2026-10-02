# A8/A1 health and bounded recovery monitor

## Current preparation ownership — 2026-10-02 09:57 UTC

Both validation reservations are CLOSED. The SAME supervisor08 preparation
child676/startticks85132 is running again under supervisor674/server673, with
`--prepare-only` and zero optimizer updates. RTX5080 belongs to preparation;
RTX2080Ti remains paused. The existing15minute monitor is ACTIVE and uses its
normal registration-bound CPU query when the local lease is returned/resumed.
Future validation requires a new explicit coordinated reservation or verified
full preparation release; a CPU phase alone grants no GPU access.

The initial lease resumed at09:27:19.132939UTC, audit244→245, after all14 validation
groups were absent and GPU compute apps empty. A second prebuilt diagnostic-only
fixture lease held the process at272 from09:45:20UTC, with a09:55:20 deadline.
The fixture obtained the exact failing CUDA case: learnedA1 activation beta at
the smallest positive F32 subnormal became zero; no model/data/optimizer work.
It exited09:49; its groups8656/8658 and compute apps were empty at09:50.

Second release initially sent NO signal because the local guard contained a
65-character checksum copied from an operator message rather than the actual
64-character raw receipt hash. Every other release check passed. The preparation
owner corrected that construction error from parsed raw JSON and independently
hashed canonical receipt bytes, preserving the denied attempt. Corrected fresh
release checks passed and oneSIGCONT resumed the original676 at
09:57:37.646130UTC; the audit advanced272→274 and one45second-postresume CPU check
passed with zero steps. The hold exceeded its deadline by137seconds due to the
guard error; no GPU overlap, implicit renewal, restart or audit reset occurred.
All operator transports are closed. No GPU readiness or training claim follows.

Ignored raw evidence: `lease-return-operation-20261002.json`,
`beta-probe-hold-operation-20261002.json`,
`beta-probe-return-operation-20261002.json` (denied/no signal), and
`beta-probe-return-corrected-operation-20261002.json`, all under
`runs/luna-continuous-a8-a1-20260929/`. The local lease/registration/last-health
record closure and exact process ancestry. Retained327 completed manifests hold
10,000train/224dev; resumed audit counters are separate from retained counts.
Release scripts recorded precise preflight/resume times but no separate checker
timestamp: last-health explicitly bounds postresume health after45seconds until
the next normal combined query replaces it with an exact remote timestamp.

**Proof construction rule:** load expected hashes, paths, lease IDs and deadline
from parsed complete raw operation/receipt JSON or current registration. Validate
SHA256 as exactly64 lowercase hex characters and check exact bytes before use.
Never copy expected hashes from prose, abbreviated output or displayed messages.
Preserve each failed guard attempt and fix its construction without weakening
receipt/identity/data/numerical gates. Derive held-query bindings from the CURRENT
lease: each new cycle has its own receipt, checksum, CPU ordinal and deadline.
The healthy-query contract still forbids extra GPU/resource queries.

## Latest coordinated validation lease — October2

The human now authorizes the separate QAT owner to validate its changes on
RTX5080 before full preparation completes, under an exclusive bounded lease.
This supersedes the requirement below to wait for full release before validation;
training still requires full verified data, current recipe gates and GPU handoff.
Preparation retains the SAME frozen job and `--prepare-only`, with zero updates.

Read ignored `runs/luna-continuous-a8-a1-20260929/gpu-validation-lease.json`
BEFORE each tick. The preparation owner owns the single preflight/hold/release
operator; the validation owner runs no competing reservation checker. Never
dispatch a second operator while one of those operations is active. A pending
lease is not a grant. Initial agreed maximum is90minutes from verified hold,
with fresh coordination required for renewal. Validation uses a separate checkout
and records its owned process groups; common goal/runbook edits belong to it.

Frozen preparation has no next-GPU boundary guard: after auditing existing data,
it automatically starts missing native capture and later CUDA smoke. Therefore
the reservation guard is a reversible SIGSTOP of its verified supervised child
process group only, while the supervisor stays live. This temporarily stops CPU
audit too but preserves its exact in-memory position, avoiding restart/re-audit.
Before grant, prove current CPU audit/complete current shard, exact identities,
all group members stopped, live supervisor, zero optimizer, no GPU compute apps
or resident preparation CUDA contexts, actual hardware and resource floors.
Never signal host job tmux panes/session/server or unrelated processes.

For held ticks, run one bounded Luna/high CPU identity/status/lease proof via
fresh LOCALtmuxMCP. Verify stopped members/starttimes, live supervisor and zero
optimizer, without querying GPU during validation ownership. Save held-state
observations separately from last running health. An intentionally stale audit
heartbeat is not failed training and must not trigger recovery/budget charges,
restart or SIGCONT. Do not suppress unrelated failures or unexpected process
states. Unknown transport is unknown ownership, not a free-GPU claim.

Before expiry the validation owner must tear down its owned GPU groups. Only
after fresh coordinated groups-gone/GPU-empty proof may the preparation owner
verify the original identities and SIGCONT the SAME child group, then verify
resumed CPU health. Deadline expiry alone never permits resume against possibly
live validation. Notify an overdue/unsafe return and coordinate cleanup; retain
the monitor and stopped-state evidence. An explicit human pause takes precedence;
account for a stopped child when performing authorized graceful termination.
All original full preparation/release gates and frozen-source/data rules remain.

Prepared 2026-09-29; **ACTIVE every15minutes for preparation-only run (resume verified2026-10-02 05:32UTC)** with user-authorized failure notification and bounded recovery. Automation ID:
`a8-a1-health-check-enable-after-manual-start`. Target chat:
`01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed`.
Latest human “Continue” resumes SAME --prepare-only pipeline; healthyafter
SSH disconnect/reconnect05:31:39UTC,ownership05:32:52UTC under supervisor08,
hostserver673/supervisor674/child676. Retained10,000train/224dev,327manifests,
zerooptimizer,reaudit4/353 then6/353. No newcapturedprompts yet: completed shards
are fullyre-audited from firstshard eachresume before unfinished devcapture.
ExistingmonitorACTIVE withsameprep-onlyprompt/cadence/target,2080Tipaused.
All launch/recovery MUST--prepare-only. Stopafterfull data/audits/readiness/
coverage/pairedsmoke/checkpointzero BEFOREoptimizer. No QAT updates authorized.
Atstatusstopped/prepcomplete+readyreceipt/zero steps,verifyownedgroupsgone and
GPUreleased,notifyready andpausemonitor. Currentpaths/hosttmux/commandbinding
MUST derive from registration.experiment; no historicalhardcoding. Frozen
math/runtime/native/data/config/precision/caps/finals preserved,onlyapproved
launchercontrolSHA82f0185a delta. NewfeatureGPUchecks mustwaitforownershiprelease.
Earlierpause/automatictraining paragraphs below are HISTORICAL and superseded.

2026-10-02 required context rotation is complete: successor above acknowledged
current08paths/ownership/zero-update scope and saved06:42:04UTC healthy check
(reaudit109/353;327retained manifests/10000train/224dev,no readyreceipt).
Existing automation transferred IN PLACE; native update and TOML readback confirm
ACTIVE15min,same prompt. Previousowner01a0f47a retires with all local operators/
transports completed. GPUjob unchanged. Exacthandoff in activegoal's preparation
owner rotation checkpoint and ignored coordination-rotation-20261002.json.


Coordination transferred to the acknowledged successor on 2026-09-30 after
the required context rotation; the same automation ID, prompt and cadence are
preserved. The GPU supervisor is independent and was untouched.

The latest human authorization supersedes the pause: “Continue. It's all yours
for the night. If prompt gen finishes, start the training. I expect to see you
still going in the morning.” Supervisor05 was verified healthy after disconnect
at 06:37:02 UTC, retaining 119 manifests / 3,736 train / 0 dev prompts, zero
optimizer steps, re-audit ordinal 1/353. Current host ownership is server794,
supervisor795/PGID795, child797/PGID797 in experiment.host_tmux. Existing monitor
reactivated with an overnight authorization paragraph; budget stays 1 of 2 used.
The supervised pipeline enters CUDA smoke and continuous training automatically
after all frozen capture/audit/readiness/coverage gates pass. Capture completion
is not overall run completion. No further approval is needed for that transition;
do not stop for morning or slow learning. Notify actual first optimizer updates,
new verified failure or required action; quiet healthy ordinary progress.
Current monitor_query_command is bound to supervisor08. Source checks must use
exact src/w1a1_eagle paths; a broad filename glob can select a test file.
Both prior pause attempts and all retained/partial data remain preserved.

Current run:`luna-continuous-a8-a1-native-order-20260930`; supervisor
`luna-supervisor-a8-a1-native-order-20261002-08`. Stable local ignored registration:
`runs/luna-continuous-a8-a1-20260929/monitor-registration.json` (preserves the old
failed experiment separately). Derive current remote status/supervisor paths from
its`experiment`object; never reuse a hardcoded earlier attempt. First Luna check
passed while corpus capture/audit was preparing, both fresh precision gates
passed, and operator handoff is ready. Last-health fingerprint is stored beside
registration. Initial activation wasACTIVE/hourly. At10:23UTC on2026-09-30 the
first scheduled Luna check found status stopped/supervisor interrupted(exit0),
ending09:12UTC on signal15. Origin is unknown; the status text “user signal”
does not establish a human stop request. Phase/capture counters and disk/memory
metrics are missing from terminal status; secondary checker warnings are not
proof of missing trained models or low disk. Root paused this existing heartbeat;
tool/saved config verifyPAUSED. No restart, additional schedule or GPU query.
Resume requires investigation of interruption and retained partial capture.

Repair is verified: WSL idle shutdown disabled while preserving20GB, host-side
Linux tmux socket`binary-eagle-runtime`/session
`continuous-a8-a1-native-order-20260930-02` owns the supervisor. A90s CPU disconnect
proof passed, incomplete data preserved by recovery, and real resume health
passed after SSH closure/reconnect (1/353shards audited, next32requests complete,
optimizer not started). Heartbeat reactivated with dynamic supervisor paths.
Monitoring uses separate **local transport** windows/fresh SSH; never attach or
send keys/kill a pane/session/server in the host-side job session. Saved config
verifiesACTIVE. Original interruption and both CPU proofs remain preserved.

## Exact command loading

The07:45UTC unknown observation was caused by copying truncated local tool
output into SSH: the raw pane includes “679 tokens truncated” inside base64.
This is command construction failure; no health checker result or job failure
is established. No second observation or GPU recovery was attempted.

A compact stdlib combined query is locally compiled in ignored
`runs/luna-continuous-a8-a1-20260929/monitor-query-source.py` (SHA256
6e8b3470a3b3138cc2949101c47024d22318f07c0972281031d44f1276511f92).
The current command JSON is a STRING (ASCII length2,983; derive dynamic length
from registration.monitor_query_command_length). Query includes readyreceipt
metadata ifexists. Operator objectwrappers/stalebinding were normalized locally;
validate commandstring equality/current08binding before everydispatch.
The exact current command is JSON in `monitor-query-command.json` and copied
into registration.experiment.monitor_query_command, bound to supervisor08.
Read the command JSON with exec_command max_output_tokens8000, parse it inside
functions.exec, and pass the resulting string directly to tmux execute_command.
Do not print/read/copy displayed command text or reconstruct payloads by hand.
Reject non-ASCII, ellipses or truncation markers before submitting. The shortened
query still runs the same checker once with --check-process and counts completed
label-manifest prompts. It changes no remote file, training configuration or gate.
At08:00:40UTC the next tick passed using direct JSON-to-tool loading: healthy
120/353,120 completed labels / 3,768train / 0dev,zerooptimizer. The prior unknown
is resolved and its evidence remains preserved. No GPU recovery or job action.
Subsequent ticks use the same compact command and direct loading rule.

## Local shell transport requirement

At16:02UTC on2026-10-01 the observation became UNKNOWN because the operator
opened interactive SSH before submitting the full saved SSH command. Raw pane
evidence shows that complete local command nested inside the Windows cmd prompt;
no checker result was returned. No retry/recovery or job action occurred.
Last verified15:45UTC: healthy264/353,8,256train /0dev,zerooptimizer.

Create a fresh LOCAL shell pane and pass the full stored monitor_query_command
from that local shell directly to execute_command. Do not open preliminary
interactive SSH or send this complete local SSH command to a remote shell.
The stored command already handles SSH and the Windows-to-WSL bridge. Local
preflight may verify the new pane is a local shell; job tmux stays untouched.
Monitor remains ACTIVE; next scheduled tick restores observation when possible.
At16:14:59UTC the next tick verified a fresh local MacBook shell before the direct
query: healthy273/353,8,544train /0dev,zerooptimizer. The unknown is resolved;
prior raw evidence preserved. Local transport closed, host job unchanged.

If get_command_result reports an output-capture error, read the already executed
LOCAL transport pane before closing it and preserve any existing checker JSON.
This is reading existing transport output, not a second remote query. Never use
the host job pane. At18:47UTC the wrapper returned no output; health unknown,
last verified18:30UTC9,832train /0dev preserved; no retry/recovery, monitor ACTIVE.
Next tick19:00:52UTC passed: healthy323/353,10,000train /96dev,zerooptimizer;
unknown resolved, prior evidence preserved, host job untouched.

## Windows SSH shell bridge

The registered RTX5080 endpoint is Windows SSH, with Linux files inside WSL.
Every health command must enter WSL explicitly, using `wsl.exe -e python3`
and a compact base64-encoded stdlib query as in the prior successful snapshot.
Do not send Linux `cd`/Python commands directly to the Windows shell. The
22:48:51 UTC observation failed before the CPU checker because that bridge was
omitted; it is unknown health, not evidence of a stopped or failed GPU job.
The next scheduled tick uses the bridge; no second observation or restart was
attempted in the failed tick.

## Model constraint and supported setup

**Current policy supersedes the historical read-only/no-restart contract below.**
On2026-09-30 the human user explicitly objected to slow delegated status reads,
stale/ambiguous progress, and failed work parked without notification/retry.
Manual status questions use one direct compact read. Scheduled checks still
dispatch one pinned Luna/high operator, now every15minutes. A new failure is
reported before recovery. Whitelisted native capture transport/CUDA-unknown
failures may get one guarded recovery per tick, at most2per24h; manual
supervisor03 consumes one. Budget is ignored`recovery-budget.json` beside
registration. Verify terminal ownership, noSTOP/pausedGPU, usable hardware and
unchanged runtime/math/data/config before preserving partial data and resuming
same experiment in a new host tmux/supervisor. Never relax a failed numerical/
data/eligibility gate, change precision/caps/tiers, reboot/reset drivers, touch
unrelated work or final sets. Unsafe/exhausted recovery gets an actionable alert,
not silence. Keep observing/suppress duplicates; pause only for completion or
explicit user stop/pause. Capture, audit and weight updates are separate metrics;
resume loop ordinals can reset while retained manifests remain complete.
First actual optimizer progress is a meaningful one-time notification.

The installed `automation_update` heartbeat schema accepts cadence, prompt,
status and target chat, but **no model or reasoning override**. The heartbeat
therefore coordinates one bounded subagent with
`agent_type="experiment_operator"` per scheduled tick. That role explicitly pins
`gpt-6-luna`/`high` in `.codex/agents/experiment_operator.toml` and the callable
collaboration role definition. Only the Luna operator performs the remote health
check; the parent records the result, suppresses duplicate reports and manages
the schedule. If that pinned role is unavailable, report the configuration
failure without substituting another model for remote monitoring.

This uses the existing chat heartbeat and the user's explicit authorization to
launch Luna for monitoring. No standalone task or invented heartbeat model field
is needed. The earlier standalone-surface question is superseded by this bounded
delegation design. A prompt naming Luna alone is still insufficient; actual
dispatch must use the pinned role.

[Official scheduled-task documentation](https://learn.chatgpt.com/docs/automations)
describes local app availability requirements. Keep the desktop app and host
available for local scheduled checks; the remote supervisor continues running
independently if the app is closed. This setup does not claim that the parent
heartbeat itself runs on Luna.

## Activation and stop

1. Start capture/readiness/training manually using the runbook. Record the
   actual run ID, remote project directory, status and supervisor paths in
   `runs/<run-id>/monitor-registration.json` locally. No guessed IP or run ID.
2. Once the supervisor is verified live and registration is complete, use
   `automation_update` to activate the existing hourly heartbeat with the pinned
   Luna delegation prompt. Preserve its ID, target chat and cadence. The parent
   must wait for the bounded operator's result before ending each check.
3. To stop monitoring, ask “Pause the A8/A1 health check” or pause it in Scheduled.
   Use `automation_update` for programmatic changes, never edit its TOML directly.
   Pausing monitoring does not stop training. A terminal run pauses the schedule
   after reporting completion/failure once.

## Health contract

The checker `scripts/check_continuous_w1ax_health.py` imports no GPU framework,
queries no device and changes no training state. It reads both model steps,
heartbeats, finite-loss/gradient flags, recorded memory metrics, free disk and
checkpoint hash. It detects step imbalance beyond one interleaved batch,
stale per-model heartbeat, unreadable status, nonfinite loss, supervisor failure,
missing/corrupt checkpoint and low disk. On the training host:

```sh
python3 scripts/check_continuous_w1ax_health.py <experiment>/status.json \
  --supervisor runs/<supervisor-id>/state.json --check-process
```

SSH must use the shared host registry and tmux MCP. No remote actions were
performed during preparation. Hashing a full dual checkpoint reads roughly
4.88 GiB; do it after checkpoint publication, never on a pending generation.
Preparation uses stage/per-prompt heartbeats before models exist; the checker
accepts that declared phase and distinguishes it from optimizer progress.
The default stale threshold is 30 minutes; tune only for a declared operation
whose bounded duration actually exceeds it. The monitor compares previous and
current steps, suppresses duplicate reports and stays quiet when healthy.

The Luna health operator may report meaningful failure, completion or required
action to its parent. It may not tune, start capture/evaluation, restart or
terminate training, select another architecture, spawn further agents, or open
final data. The parent may dispatch exactly one pinned operator for each check,
with no sleep/poll loop or overlapping GPU operators. Slow learning alone is
not failure. Immediate nonfinite, disk and memory safeguards belong in training,
because hourly checks are too sparse to provide those safeguards.

## Latest terminal capture failure

Latest user progress check21:28UTC found supervisor02 finished/exit1, native
capture server aborted(return-6) with CUDA unknown error atstream synchronization.
26completed train-label manifests total832prompts; dev0; shard00026 has30native
requests and no labels manifest. No optimizer started. Root paused this existing
schedule with fields preserved. This is a new native/CUDA failure, not the prior
SSH/WSL lifetime stop. No automatic retry/recovery or GPU query occurred.

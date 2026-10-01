# A8/A1 health and bounded recovery monitor

Prepared 2026-09-29; **PAUSED at the human user’s GPU-pause request** with user-authorized failure notification and bounded recovery. Automation ID:
`a8-a1-health-check-enable-after-manual-start`. Target chat:
`01a0f47a-e246-75e1-a299-fcac42d34f8a`.
Coordination transferred to the acknowledged successor on 2026-09-30 after
the required context rotation; the same automation ID, prompt and cadence are
preserved. The GPU supervisor is independent and was untouched.

At 2026-10-01 00:09:53 UTC, the explicit user pause was completed: both shared
GPU controls paused, supervisor03 interrupted after SIGINT, trainer stopped,
owned process group absent, no project processes and no RTX5080 compute apps.
58 completed manifests retain1,832train/0dev prompts; optimizer never started.
STOP is present. Existing monitor pause is verified in saved automation config.
No recovery or automatic restart is permitted until explicit user resume.

Current run:`luna-continuous-a8-a1-native-order-20260930`; supervisor
`luna-supervisor-a8-a1-native-order-20260930-03`. Stable local ignored registration:
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

# A8/A1 hourly health monitor

Prepared 2026-09-29; **PAUSED** after the first hourly check found a terminal SIGTERM stop. Automation ID:
`a8-a1-health-check-enable-after-manual-start`. Target chat:
`01a0f01d-65c6-7af0-9660-99c07e95cacd`.

Current run:`luna-continuous-a8-a1-native-order-20260930`; supervisor
`luna-supervisor-a8-a1-native-order-20260930-01`. Stable local ignored registration:
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

## Model constraint and supported setup

The installed `automation_update` heartbeat schema accepts cadence, prompt,
status and target chat, but **no model or reasoning override**. The heartbeat
therefore coordinates one bounded subagent with
`agent_type="experiment_operator"` per hourly tick. That role explicitly pins
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

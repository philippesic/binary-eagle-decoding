# A8/A1 hourly health monitor

Prepared 2026-09-29; **PAUSED**, no scheduled execution. Automation ID:
`a8-a1-health-check-enable-after-manual-start`. Target chat:
`01a0f01d-65c6-7af0-9660-99c07e95cacd`.

## Model constraint and supported setup

The installed `automation_update` heartbeat schema accepts cadence, prompt,
status and target chat, but **no model or reasoning override**. Its saved
configuration has no model/reasoning field. The available model catalog includes
`gpt-6-luna` with `high`, but this preparation chat currently uses Sol. Therefore
the paused heartbeat is not a guarantee of Luna-only scheduled execution.
A prompt naming Luna cannot select a model. Do not activate it while unresolved.

After manually starting training, select **GPT-6 Luna / High** for this chat in
the desktop model picker, then inspect the Scheduled editor and a manual health
check turn's actual model. If the app exposes model/reasoning controls for this
heartbeat, set them explicitly and verify they persist. Otherwise verify the
heartbeat inherits this chat's Luna/high setting with its first run; keep this
chat on Luna/high thereafter. If the installed scheduler cannot confirm that
behavior, leave the heartbeat paused. A standalone scheduled task supports
explicit model/reasoning in the tool, but requires the user's explicit request
for a standalone task; it was not silently substituted here.

The supported explicit standalone configuration, should the user choose it, is:
local project `binary-eagle-decoding`, hourly, model `gpt-6-luna`, reasoning
`high`, the same read-only health prompt, created with `automation_update` after
`list_projects`. No standalone schedule has been created.

[Official scheduled-task documentation](https://learn.chatgpt.com/docs/automations)
describes explicit model/reasoning selection, local app availability requirements,
and checking selected settings on initial runs; it does not establish this
installed heartbeat's per-model behavior.

## Activation and stop

1. Start capture/readiness/training manually using the runbook. Record the
   actual run ID, remote project directory, status and supervisor paths in
   `runs/<run-id>/monitor-registration.json` locally. No guessed IP or run ID.
2. Resolve Luna/high as above. Ask in that Luna chat: “Enable the prepared A8/A1
   health check hourly for run <run-id>.” The agent must use `automation_update`
   to update the existing ID and preserve its prompt/cadence/target fields.
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
The default stale threshold is 30 minutes; tune only for a declared operation
whose bounded duration actually exceeds it. The monitor compares previous and
current steps, suppresses duplicate reports and stays quiet when healthy.

The hourly agent may report meaningful failure, completion or required action.
It may not tune, start capture/evaluation, restart or terminate training, select
another architecture, spawn agents, or open final data. Slow learning alone is
not failure. Immediate nonfinite, disk and memory safeguards belong in training,
because hourly checks are too sparse to provide those safeguards.

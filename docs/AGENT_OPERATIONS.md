# Agent operations

## Starting and coordinating a goal

The user requests one goal at a time with a rough objective; `$start-goal-team` is
an optional entry point, not a requirement for delegation. The GPT-6.1 Sol high
orchestrator translates it into a native Codex Goal and a concise
`docs/goals/<slug>.md` checkpoint, linked from `docs/STATUS.md`. The Goal sustains
its task; the file sustains the project when tasks or agents change. A native Goal
is scoped to its Codex chat; an explicitly authorized successor resumes the same
objective from the written checkpoint.

The orchestrator is the user's interface: brief technical updates with enough
background to follow decisions, progress, and results. The user owns major
research forks. Record a pending decision with options and current evidence in
`docs/DECISIONS.md`, then continue independent work instead of waiting idle.

Use native subagents for independent features, investigations, and checks. Create
or message a separate user-owned Codex chat only with explicit human authorization
required by the tool. Start only as many workers as can make independent progress.
Assign files and one GPU owner explicitly. GPT-6.1 Sol high owns coordination and
features; Luna high handles tests, logs, and long experiment supervision; Astra
medium is an advisor on hard questions. Feature owners report independent checking
needs to the parent, which schedules workers and remains responsible for
integration and acceptance.

## Checkpoints and context rotation

At each milestone and before a handoff, update the active goal file with what
actually happened: commit IDs, checks, results, decisions, active task IDs,
worktrees, remote tmux sessions/run directories, and the next action. The status
file points to that checkpoint. Keep long logs elsewhere.

Project config asks Codex to compact at 630,000 tokens, 60% of the locally
configured 1,050,000-token context window. Revisit that absolute number if the
model or context window changes. The `SessionStart(source=compact)` hook counts
compactions per session in the shared Git common directory. After the first it
prompts a status refresh; after the second it asks for a checkpoint and worker
rotation at the next safe boundary. It runs **after** compaction because
`PreCompact` cannot launch a replacement agent. Neither hook can transfer subagent
ownership or a running command. The orchestrator must finish or checkpoint active
work and rotate bounded workers. A successor user-owned chat requires explicit
human authorization; when unavailable, continue the objective in the current chat
with the durable checkpoint. For an authorized successor, verify all jobs and
acknowledge ownership before retiring the predecessor. A GPU process supervised
inside tmux may continue across the agent handoff if its ownership and stop
procedure are recorded.

If the hook is unavailable or Codex does not honor its trust state, rotate at a
milestone instead; the written checkpoint is the reliable mechanism. Do not
trust a compacted chat summary as the sole record of an experiment.

## Git and worktrees

`main` is the integration branch. Commit coherent progress without waiting for a
large final batch. Push only when covered by the current user authorization. Use
one short subject line and no coauthors or body. For independent edits, make
temporary worktrees; share a worktree only for read-only work or explicit file
partitioning. Integrate changes into `main` after relevant checks pass, push
`main` when authorized, then remove a merged worktree and its branch. Before
removal check for uncommitted changes, unpublished commits, and live tasks. Do not
run broad `git clean` on a shared checkout.

llama.cpp is a Git submodule whose fetch/push URL is the user's fork. Its
`upstream` remote can track ggml-org. For a runtime change, commit inside the
submodule, push that commit to the fork, then commit the updated gitlink in the
parent repository and push `main`. Keep the upstream revision and experimental
baselines recorded when updating it.

## Shared GPU hosts

All local worktrees read the machine-local
`~/.config/binary-eagle-decoding/hosts.toml`. Initialize it with
`python3 scripts/agent_env.py init`. When the user supplies a rotated address:

```sh
python3 scripts/agent_env.py set-host rtx5080 <host-or-ip> <ssh-user> --port 22
python3 scripts/agent_env.py set-host rtx2080ti <host-or-ip> <ssh-user> --port 22
python3 scripts/agent_env.py status
```

The file stores host, user, port, and a remote project directory; no keys or
passwords. IPs are deliberately outside Git and shared through the home
directory. Never use an old address after the user reports a rotation.

**Every SSH connection to either WSL device must go through tmux MCP.** Do not
run `ssh`, `scp`, or `rsync` through the ordinary shell. Use the tmux MCP session
to log in, run commands, inspect processes, transfer code via Git on the remote
shell, and disconnect. The RTX 5080 is the normal experiment device. Reserve
RTX 2080 Ti for SM75 binary instruction and end-to-end measurements; record the
hardware for every result. A GPU has one active project experiment owner unless
the orchestrator explicitly coordinates sharing.

Keep the remote checkout, models, caches, logs, and outputs under the host file's
`workdir` (default `~/binary-eagle-decoding`). A run has a unique `runs/<run-
id>/`. Connect through tmux MCP, but start the supervisor in a detached **Linux
tmux session on the WSL host**, not foreground inside SSH protected only by local
tmux. Use a dedicated host socket such as`binary-eagle-runtime` and `tmux -L
binary-eagle-runtime new-session -d -s <session> -c <project> 'exec python3
scripts/remote_job.py <run-id> -- <command> [args...]'`. Record the host
socket/session/server and supervisor identities separately from the local MCP
transport window. On this WSL2.7host, verify the user's `.wslconfig`
contains`[general] instanceIdleTimeout=-1`: background Linux services alone did
not prevent distro auto-shutdown after SSH closed. Preserve existing settings and
backup any authorized edit. Before a long GPU launch, require one bounded CPU
disconnect proof beyond the idle boundary and a fresh reconnect; do not claim
durability from a startup snapshot. The supervisor puts the child in its own
process group and places common caches and temporary files inside that run
directory. It records `state.json` and `stdout.log`. Do not start detached GPU
processes outside this supervisor. A run directory can be removed only after its
job has stopped and its artifacts are preserved.

For **“pause GPU work”**, immediately mark the relevant host paused with
`python3 scripts/agent_env.py pause rtx5080` (or `all`) so agents start no new
runs. Via tmux MCP, send an interrupt to each active remote supervisor, wait for
its `state.json` to show stopped, inspect the process group and `nvidia-smi` for
remaining project GPU use, then report that the machine is free. If graceful
stop fails, terminate the recorded process group through the remote tmux MCP
session and verify again. Checkpoint the experiment and task before leaving it.
`pause` alone is only a local request flag; it does not free GPU memory. Resume
new runs only after the user asks to resume, with `python3 scripts/agent_env.py
resume rtx5080` and a fresh host/resource check.

The host registry is machine-local. Historical scaffold defaults are not live
state: consult the current registry and active-goal checkpoint, then verify exact
job identities before acting. Never overwrite configured hosts with blank defaults.

Codex may ask the user to review and trust the repository hook before it runs. The
[Codex hooks reference](https://learn.chatgpt.com/docs/hooks) describes that trust
step and the `SessionStart` behavior. The [subagents
reference](https://learn.chatgpt.com/docs/agent-configuration/subagents) documents
project agent profiles; the [skills
reference](https://learn.chatgpt.com/docs/build-skills) documents repository-local
skills.
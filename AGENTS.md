# W1A1 EAGLE agent rules

This is a bounded research project. Use `docs/PROJECT_OVERVIEW.md` for research
direction, `docs/EVALUATION.md` for measurements, and `docs/AGENT_OPERATIONS.md`
for team, handoff, Git, and GPU procedures. Read the document relevant to the
task; do not load every document for a small edit.

- The user owns major research decisions. Continue independent work when a
  decision is pending; record options and evidence in `docs/DECISIONS.md`.
- Keep one active goal at a time. Its durable state lives in `docs/STATUS.md`
  and its own file under `docs/goals/`. Checkpoint before handing work to a new
  agent. A thread's memory or Goal is not the project record.
- Delegate independent, bounded work. Sol high owns coordination and features;
  Luna high handles tests, logs, and experiment supervision; Astra medium gives
  focused advice on hard problems. Avoid parallel edits to the same files or
  concurrent use of one GPU without explicit coordination.
- For long work, use a Codex task when the user may want to inspect or steer it;
  use a subagent for a short, disposable assignment. Finish or checkpoint a
  worker's output before retiring its worktree.
- Use temporary worktrees for isolated edits. Integrate reviewed, tested work
  into `main`, push it, then remove the merged worktree and branch. Never drop
  unmerged work. Do not leave a parent gitlink pointing to an unpublished
  llama.cpp commit.
- Commit and push coherent progress regularly. Commit subjects are short and
  simple, with no body or coauthor trailers.
- SSH to either WSL GPU host only through the tmux MCP tool. Consult the shared
  local host file described in `docs/AGENT_OPERATIONS.md`; never guess a rotated
  IP. Keep each remote experiment in its own project directory and stop its
  process group before reporting the GPU free. A user request to pause GPU work
  takes priority over new experiments.
- Keep model weights, datasets, captures, and raw runs out of Git. Record
  versions and hashes in experiment reports. CPU/Metal checks do not validate
  SM75 performance; identify actual hardware and precision in every result.

# Goal files

Create one file per user-started goal. Keep the active file short enough for a
fresh agent to read immediately. Suggested fields:

- Objective and evidence required for completion.
- Current stage and last update time.
- Decisions pending with the user; independent work still possible.
- Worker/task owners, branches/worktrees, and exact deliverables.
- Live GPU jobs: host alias, tmux session, run directory, process/status, and
  cleanup responsibility.
- Completed commits, tests, experiment IDs, and observed results.
- Next concrete actions and any blockers.

Update at milestones and before compaction handoff. Keep long data and logs in
`results/` or remote run directories, with links/hashes here.

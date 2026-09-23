---
name: start-goal-team
description: Start one new W1A1 EAGLE project goal in Codex desktop, with a Sol orchestrator and scoped agent team. Use when the user explicitly invokes the skill for a new goal.
---

# Start a project goal

The user manually invokes this skill and supplies a rough objective. A goal is
one coherent research or implementation outcome; do not start unrelated goals.

1. Read `docs/STATUS.md`, `docs/PROJECT_OVERVIEW.md`, and the relevant project
   guidance. Translate the user's prompt into a concrete objective, success
   evidence, boundaries, and first independent work units. Record it in a new
   `docs/goals/<short-slug>.md` and link it from `docs/STATUS.md`. Commit and
   push this checkpoint before launching workers so their worktrees see it.
   Use the native Codex Goal feature for the orchestrator's objective when available; the file
   remains the cross-thread record. Ask about missing major decisions while
   continuing work that does not depend on them.
2. Use Sol high for orchestration and most feature work. Create separate Codex
   tasks for substantial work the user may want to inspect or steer; use native
   subagents for bounded checks, log review, or independent research. Luna high
   handles tests and experiment operations. Consult Astra medium for a focused
   hard question or impasse. Delegate only work that can progress independently,
   name file/GPU ownership, and avoid same-file or same-device contention.
3. Assign every worker a deliverable, check, deadline or stopping boundary, and
   location for its findings. Prefer isolated temporary worktrees for separate
   edits. Integrate verified commits into `main`, push, and clean merged
   worktrees. Checkpoint state at milestones and before an agent rotates.
4. Follow `docs/AGENT_OPERATIONS.md` for handoffs, Git, and GPU resources. On
   the second compaction, the project hook prompts a handoff at a safe boundary;
   it cannot automatically transfer agent state. A successor must read the goal
   file and verify active agents/jobs before resuming.

Keep user updates brief, technical, and clear to a reader with graduate-level
background who has not memorized this codebase. Report evidence, next steps,
and decisions needed from the user. Do not mark a Goal complete until its
stated evidence exists.

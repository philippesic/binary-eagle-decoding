---
name: start-goal-team
description: Start one new W1A1 EAGLE project goal in Codex desktop, with a GPT-6.1 Sol orchestrator and scoped agent team. Use when the user requests a new project goal or invokes this skill; ordinary edits continue existing work.
---

# Start a project goal

The user requests a project goal with a rough objective; skill invocation is
optional. A goal is one coherent research or implementation outcome. Preserve
an unfinished goal unless the user explicitly supersedes it.

1. Read `docs/STATUS.md`, `docs/PROJECT_OVERVIEW.md`, and the relevant project
   guidance. Translate the user's prompt into a concrete objective, success
   evidence, boundaries, and first independent work units. Record it in a new
   `docs/goals/<short-slug>.md` and link it from `docs/STATUS.md`. Preserve this
   checkpoint before launching workers so their worktrees see it. Push only when
   covered by existing user authorization. Use the native Codex Goal feature for
   the orchestrator's objective when available and explicitly requested; the file
   remains the cross-thread record. Ask about missing major decisions while
   continuing work that does not depend on them. 2. Use GPT-6.1 Sol high for
   orchestration and most feature work. Use native subagents for independent
   implementation, checks, log review, or research. Create or message separate
   user-owned chats only with explicit human authorization. Luna high handles
   tests and experiment operations. Consult Astra medium for a focused hard
   question or impasse. Delegate only work that can progress independently, name
   file/GPU ownership, and avoid same-file or same-device contention. 3. Assign
   every worker a deliverable, check, deadline or stopping boundary, and location
   for its findings. Prefer isolated temporary worktrees for separate edits.
   Integrate verified commits into the integration branch, push when authorized,
   and clean merged worktrees after preserving unpublished work. Checkpoint state
   at milestones and before an agent rotates. 4. Follow `docs/AGENT_OPERATIONS.md`
   for handoffs, Git, and GPU resources. On the second compaction, the project
   hook prompts a handoff at a safe boundary; it cannot transfer agent state or
   authorize a new user-owned chat. Rotate bounded workers when useful; continue
   here if successor-chat authorization is unavailable. An authorized successor
   verifies live jobs and acknowledges ownership before acting. Preserve healthy
   GPU jobs during rotation.

Keep user updates brief, technical, and clear to a reader with graduate-level
background who has not memorized this codebase. Report evidence, next steps,
and decisions needed from the user. Do not mark a Goal complete until its
stated evidence exists.

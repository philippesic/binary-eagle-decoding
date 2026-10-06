# W1A1 EAGLE agent rules

This is a bounded research project. Use `docs/PROJECT_OVERVIEW.md` for research
direction, `docs/EVALUATION.md` for measurements, and `docs/AGENT_OPERATIONS.md`
for team, handoff, Git, and GPU procedures. Read the document relevant to the
task; do not load every document for a small edit.

- Q4_0 EAGLE is the primary comparison baseline and the model to beat for
  acceptance, latency, and total throughput. FP16 EAGLE is a secondary diagnostic
  reference, not the primary success target. This does not change target/verifier
  model precision or previously frozen experiments.
- Favor development speed and decisive native acceptance/throughput tests over
  bit-exact cross-backend parity. Timebox numerical diagnostics and require each
  one to address a concrete training or deployment risk. Hugging Face versus
  llama.cpp floating-point drift alone is not a blocker when training uses
  captured native target features/logits and bounded native trajectory checks
  pass. Record the discrepancy and a practical numeric gate, then proceed.
  Investigate further if it changes labels, drafter decisions, gradients, or
  conclusions. Preserve exact data ancestry, cache/mask semantics, verifier
  behavior, and held-out evaluation.
- The user owns major research decisions. Continue independent work when a
  decision is pending; record options and evidence in `docs/DECISIONS.md`.
- Keep one active goal at a time. Its durable state lives in `docs/STATUS.md` and
  its own file under `docs/goals/`. Checkpoint before handing work to a new agent.
  Start a new goal only when the user requests one. A thread's memory or Goal is
  not the project record. Work autonomously within the active goal.
- The orchestrator speaks to the user briefly and technically, explaining terms
  that require project-specific background. Give periodic milestone updates and
  answer check-ins directly. Save detailed state in the goal file.
- Whenever you notice contention between the user and agents (disagreements,
  cooperation difficulties, or confusion), or a concrete user mistake, record it
  in `docs/USER_LESSONS.md`. Keep entries factual and respectful: include context,
  evidence, agent contributions to the problem, and a practical lesson for
  reminders, behavior, or clearer coordination. Distinguish confirmed errors from
  disagreements or uncertainty; update entries when corrected. Consult relevant
  lessons when needed, without treating them as overriding the user's current
  instructions.
- Delegate independent, bounded work. GPT-6.1 Sol high owns coordination and
  features; Luna high handles bounded tests, logs, and experiment supervision;
  Astra medium gives focused advice on hard problems. Avoid parallel edits to the
  same files or concurrent use of one GPU without explicit coordination.
- Use native subagents for independent subtasks. Create or message a separate
  user-owned Codex chat only when the human explicitly authorizes that action.
  Finish or checkpoint a worker's output before retiring its worktree.
- Use temporary worktrees for isolated edits. Integrate reviewed, tested work into
  the integration branch; push when covered by user authorization, then remove the
  merged worktree and branch after checking unpublished work. Never drop unmerged
  work. Do not leave a parent gitlink pointing to an unpublished llama.cpp commit.
- Commit coherent progress regularly; push within existing user authorization.
  Commit subjects are short and simple, with no body or coauthor trailers.
- SSH to either WSL GPU host only through the tmux MCP tool. Consult the shared
  local host file described in `docs/AGENT_OPERATIONS.md`; never guess a rotated
  IP. Keep each remote experiment in its own project directory and stop its
  process group before reporting the GPU free. A user request to pause GPU work
  takes priority over new experiments.
- Keep model weights, datasets, captures, and raw runs out of Git. Record versions
  and hashes in experiment reports. CPU/Metal checks do not validate SM75
  performance; identify actual hardware and precision in every result.

## Acceptance, continuity, and experiment ownership

- Continue the existing active goal and its latest checkpoint; do not restart the
  nine-model work, shorten an authorized training allocation, or infer completion
  from a prepared endpoint/metadata packet. Historical README scaffold claims
  and host defaults are not live state. Reconcile STATUS with actual job evidence.
- Automatically delegate useful independent CPU/source work when permitted,
  normally up to four active workers. Parent schedules topology; workers report
  opportunities without recursively expanding the team. Only the verified remote
  operator controls a GPU lane. Local tests/builds must respect any quiet window.
- Preserve healthy training and passive endpoint watchers through unrelated
  configuration edits or context rotation. Handoffs require exact host/run,
  process birth/boot identity, source/checkpoint pins, owner acknowledgment, and
  stop procedure. A queued message or written checkpoint does not transfer control.
- Require actual drafter/export/admission/native evaluation evidence. Acceptance
  rate or an isolated kernel speedup alone does not prove total-throughput benefit
  over Q4_0 EAGLE. Keep held-out inputs, target/verifier semantics, precision,
  hardware, data ancestry and matched baseline immutable within a comparison.
- Fresh worker contexts receive a bounded assignment and evidence links. Refresh
  durable state after compaction; rotate workers at safe boundaries. A successor
  user-owned chat still requires explicit human authorization. Continue useful
  authorized work in the current chat when automatic replacement is unavailable.

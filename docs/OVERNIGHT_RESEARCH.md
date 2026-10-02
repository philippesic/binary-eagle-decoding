# Overnight research and usage supervision

## Human authorization and active goal

On October 1, 2026, the human authorized three substantial CPU research teams
and one usage supervisor. They should extend useful investigations until less
than 2% weekly usage remains, then stop research. The existing QAT task must
remain supervised through the night and may use credits. Research must not
consume the next allowance after reset. This is supporting research within the
existing QAT optimization readiness goal, not a second active project goal.

The human explicitly authorizes the supervisor to message, extend, restart and
interrupt its research workers, launch additional bounded research when useful,
and message the existing QAT task to preserve its authorized validation/training
and monitor. This does not authorize changing frozen experiments, verifier or
target precision, reading sealed finals, or interfering with GPU ownership.

Protected QAT task: `01a0f934-dd65-7e33-a5bf-0ba591e713a4`,
**Optimize QAT for GPU readiness**, local host. Its existing heartbeat is
`qat-validation-and-training-handoff`, ACTIVE every 15 minutes. Dataset owner
`01a0f47a-e246-75e1-a299-fcac42d34f8a` retains RTX5080 until verified release.
The research supervisor does not own either GPU or preparation recovery.

## Budget and time control

Initial observation at 2026-10-02 06:15 UTC: codex weekly allowance 68% used,
32% remaining. Initial reported reset is Unix1791049896, October3 10:51:36PDT.
The human clarified on October2 that the reset may be later than10a.m. and
must be monitored. **There is no fixed morning cutoff.** Track the live codex
weekly window/reset and stop research before the actual reported original-window
reset, or immediately on observed reset, whichever applies. A timestamp change
alone can be a corrected reset estimate: update the original-window deadline if
usage has not reset; do not mistake a later revised estimate for a fresh allowance.
A new window plus reset usage is a reset and must latch stop. Never spend the
new allowance on research. Initial reset1791049896 remains an observation, not
a fixed override. The former October2 09:55PDT cutoff is superseded.

One single-agent supervisor owns all research team leaders and descendants in
its own collaboration tree, permitting direct interruption. Check usage every
five minutes and log UTC, used/remaining, reset, team state and QAT state in
ignored `runs/overnight-research-20261002/usage.jsonl`. Share a brief usage ping
at each check. Use the account codex bucket, not API rate limits or reset credits.
Never redeem a reset. Do not estimate remaining usage from token counts.

Stop on the first of:

- remaining <=1% (the goal is <2%, leaving a little stop/checkpoint room);
- remaining0%, ordinary allowance disallowed, or a limit reached;
- current time >= the latest verified reset timestamp for the original allowance
  window (checkpoint and stop just before that timestamp);
- reset observed by a new allowance window with reset usage, or an unexplained
  substantial decrease in used percentage. Never reinterpret a fresh allowance as a new
  research budget;
- newer human stop.

At <=5% reduce concurrency and checkpoint; at <=3% use only one bounded team
extension, with frequent worker stop checks, to reduce overshoot. Below2% stop.
If usage is unavailable, permit no new extension and stop after two consecutive
unavailable checks; never assume unavailable means zero consumed.

Before any new agent, phase, expensive reasoning/retrieval batch or CPU run,
workers read the absolute control JSON. CPU experiment loops check it between
iterations and honor the live original-window reset deadline themselves. The
legacy hard_cutoff_utc/unix fields are null; workers must handle null as no fixed
cutoff and read the current reset deadline/override before work. GPU/Metal and real-data
optimizer updates are forbidden in these research teams; tiny synthetic CPU
training is allowed. Agent research, not long CPU busywork, is the priority.

At stop, atomically latch state=stopping/stopped in the control JSON, send all
research workers a checkpoint/stop message, directly interrupt every running
research descendant, and verify no research experiment process remains. Preserve
dirty/unmerged work; do not delete worktrees or launch summarizer agents. A
minimal supervisor terminal pass may use credits if necessary to finish stopping
workers and protect QAT. Disable the research heartbeat once stops are verified.
Do not stop, archive, interrupt, handoff, or pause the QAT task, dataset task,
their agents, remote jobs or their heartbeat. Verify the QAT heartbeat remains
ACTIVE and QAT has a current checkpoint; if idle unexpectedly or its monitor is
missing, send an explicitly authorized continuation to that task. An idle QAT
chat waiting under an active healthy heartbeat is not a stopped GPU job.

The supervisor heartbeat provides five-minute checks and recovery across turn
completion. If a research leader finishes while allowance remains, inspect its
evidence and assign the highest-value unfinished question, attempted
falsification or prototype extension. Do not repeat a survey or manufacture
work to consume tokens. New bounded topics must advance the three research
questions or an evidenced project bottleneck. Never spawn another usage monitor.

## Team1 — representation geometry

Determine whether coordinate choice, grouping or structured factorization can
make all-nine W1A1 EAGLE more learnable without losing its runtime advantage.
Begin with existing representation/discrete-fitting audits and implemented
asymmetric weights, learned thresholds and rank corrections. Go beyond surveys.

Derive transformations on the actual feature fusion, RMSNorm, residual, Q/K/V/O,
RoPE, gate/up/SiLU/down and private head graph. Identify exact invariances and
noncommuting boundaries. Test structured rotations/grouping and candidate
factorizations on small CPU graphs. Demonstrate failure cases and zero-benefit
symmetries; charge floating-point transformations and packing/memory/launch
costs. Use SpinQuant/QuaRot/RBNN source as evidence with explicit transfer limits.
No inference-speed or native-acceptance claim from reconstruction or CPU timing.

Deliver candidate derivations, executable CPU references and comparative
fixtures, counterexamples, real-sample availability/accounting and one actionable
representation proposal or supported rejection. Give the later GPU experiment
only after the mathematical and deployment assumptions are concrete.

Own new `research/overnight20261002/representation/` and
`experiments/overnight20261002/representation/`. Do not edit shared QAT/runtime
source. One Sol/high leader, one Sol/high builder, focused Astra/medium challenger
and Luna/high CPU validation. Separate worker worktrees or explicit disjoint
file ownership. Raw artifacts stay under ignored runs.

## Team2 — accepted-prefix training objective

Determine which trainable objective can improve native accepted prefix length
when hard CE and optimizer improvements fail. Read actual pinned sample-and-match
verifier source and EVALUATION.md. Distinguish greedy/sample-and-match from
probability-ratio overlap verification; derive rather than borrow the formula.
Account for recurrent errors, vocabulary exclusions, supported labels, training
versus deployment prefix distributions and biased/stale trajectory labels.

Build small exactly enumerable teacher/drafter systems on CPU. Compare CE,
distribution matching, prefix-survival losses and refresh. Construct cases where
loss decreases but accepted length worsens. Derive estimators/surrogates, gradient
and variance behavior, and identify when existing captures cannot support an
objective. Investigate primary acceptance-aware training literature, including
arXiv2609.24150, with exact verification semantics and data/compute requirements.

Deliver checked derivations, CPU objective prototypes, falsification fixtures,
one proposed next recipe and exact minimum capture requirements. Do not mutate
current training, consume sealed finals or reuse changed-prefix labels.

Own new `research/overnight20261002/objective/` and
`experiments/overnight20261002/objective/`. Team composition as above. Report
the conditions under which the conclusion would change after QAT results.

## Team3 — block-parallel drafter design

Advance the completed architecture shortlist into executable reduced DSpark/
DFlash references and a source-grounded W1Ax architecture decision. This is an
explicitly authorized prospective research expansion; do not replace EAGLE or
start real new-architecture training/capture.

Inspect pinned author source and current native support. Reconstruct feature
interfaces, dual-context masks, Markov conditioning, head ownership, training
loss, and exact draft/verify/rollback/reinjection behavior. Build CPU references
with first rejection, partial/full acceptance, EOS and mask/alignment tests.
Map W1Ax boundaries and quantify parameter storage, operation counts, borrowed
full-vocabulary head costs and target-feature capture requirements. Prototype
alternatives where expensive head/conditioning defeats the binary premise.

Deliver substantial reference code and tests, an evidence-backed architecture
choice or rejection, integration design and minimum later GPU screen. Existing
EAGLE captures are not automatically eligible for different layer taps/prefixes.
Maintain frozen target/verifier and immutable borrowed tensor ownership.

Own new `research/overnight20261002/architecture/` and
`experiments/overnight20261002/architecture/`. Team composition as above.

## Git and evidence ownership

Leaders own their team worktrees, reviewed commits and progress reports; the
supervisor alone owns control JSON, usage log, team registry and this run's
checkpoint. Every worker must know it is not alone and must not revert others.
Commit/push coherent team branches regularly, retaining raw data outside Git.
Integrate tested isolated research files into main sequentially if clean and
reviewed, without altering live training/runtime source identities. Only clean
worktrees after verified integration; preserve all unmerged work at stop.

The existing goal file links this run. The supervisor records its task ID,
heartbeat ID, all canonical agent IDs, worktree/branch paths, experiment process
ownership and next actions in the ignored registry and a compact project
checkpoint. On handoff read that registry; never infer workers or GPU release
from chat summaries alone.

# Curriculum transition worker checkpoint

Assigned active goal: `docs/goals/qat-optimization-readiness.md`; this is a
supporting CPU audit, not a new goal. Owner `/root/curriculum_transition`,
branch `research/20261002-curriculum-transition`, worktree
`/private/tmp/eagle-parallel-20261002/curriculum-transition`.

Deliverable: source-bound A8→A4→A1 learned-state/reset and crash-resume audit,
synthetic actual-API fixtures, independently checked failure cases and narrowly
scoped patch proposal. No core/live recipe/runtime changes.

Completed: exact resets/persistence measured, post-transition exact checkpoint
round trips, independent finite-difference surrogate oracle; own two new tests
plus 20 existing runner tests pass (22 total). Source/runtime hashes and
measurements are in `experiments/parallel20261002/curriculum_transition/`.

Descendant `/root/curriculum_transition/transition_validation` owns only
`independent_validation.py` and `independent_validation.md`; its revised
PyTorch 2.14.0 CPU run verifies live model/optimizer/RNG round trip, target
pointer failure/source replay and rejection of damaged activation/phase
ancestry. Omitted/partial Adam moments are accepted and change the next
updated model by max `0.0023282766`; NaN/wrong-shape moments pass loading
and fail at update. The ten scenarios and exact commands are documented.
Both workers check shared control at each chunk. No persistent experiment
process, GPU/Metal/SSH or paid credits.

The concrete narrowly scoped optimizer-validation proposal is in
`experiments/parallel20261002/curriculum_transition/PATCH_PROPOSAL.md`.
Activation resets remain current semantics; no transfer recipe is inferred.

Deliverable complete: 22/22 unittest checks, 10 independent scenarios,
`git diff --check` clean. Both synchronous validators have exited; descendant
reports complete and has stopped editing. Coherent branch commit is handed
to root with its hash. No core source is changed.

Remaining: root review, main integration, push and worktree cleanup. Preserve
needed ignored raw cases under `runs/parallel20261002/independent-validation-torch214/`
before retiring this worktree; the superseded system-PyTorch exploratory
directory is separately identified in the independent report.
Root owns the shared active-goal
checkpoint and integration. This worker checkpoint deliberately does not
edit the concurrently maintained shared goal file.

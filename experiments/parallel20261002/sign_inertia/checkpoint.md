# Sign inertia team checkpoint

Supports the existing `docs/goals/qat-optimization-readiness.md`; no new goal.
Owner `/root/sign_inertia`, validator `/root/sign_inertia/inertia_validation`.
Worktree `/private/tmp/eagle-parallel-20261002/sign-inertia`, branch
`research/20261002-sign-inertia`, source base `6c613039`.

Deliverable: 99 fixed-budget synthetic CPU trajectories, exact reachable and
infeasible capacity enumeration, gradient/optimizer-state oracle, independent
validation, compact report, manifest, summary, and unique tests. No core changes.
Main results: default inertia stalls64; lower inertia yields linear useful
flip20, but recurrent7 flips/3backs with only6/16 fully correct tail updates;
XOR is exactly infeasible despite14 flips/6backs; large-scale recurrent flip
worsens margin to-125.15. No recipe selected; telemetry recommendation only.

Remaining integration: orchestrator reviews/cherry-picks this branch, records
the reviewed commit/checks/report link in the active goal and STATUS, pushes
main, and cleans this worktree only after preserving raw artifacts and verifying
merge. Native acceptance/Q4_0 throughput and real recipe decisions remain with
the existing QAT owner. No GPU/SSH/external resource was acquired by this team.

Raw results: main ignored
`runs/parallel20261002/sign_inertia/results.json`; hash in manifest. Independent
logs were copied to main's ignored `runs/parallel20261002/sign_inertia/validation/`
before cleanup. Shared stop control was read
before each grid chunk; ordinary original-window research only, no paid credits.
Final acceptance is 20/20 tests (5 owner, 7 independent, 8 existing), lint passes,
and all 99 stored trajectories pass independent audit. Commit is reported to the
orchestrator separately; no experiment process remains running.

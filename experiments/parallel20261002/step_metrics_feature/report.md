# Step metrics source feature

Source candidate based on `e8569fb`, isolated branch
`research/20261002-step-metrics-feature`, worktree
`/private/tmp/eagle-parallel-20261002/step-metrics-feature`. Source ownership is
only `src/w1a1_eagle/recurrent_qat.py`; trainer, curriculum, evaluator, native
adapter, and activation implementation are read-only. Root/QAT owns any future
integration and adoption. This candidate has not changed the live recipe.

## Deliverable and acceptance

The actual source removes `clone()` from the pre-update sign comparison.
`latent_sign.detach() < 0` allocates the same independent Boolean snapshot;
the original effective-scale snapshots remain unchanged. A private reporting
helper extracts already-computed post-update scalars in groups keyed by tensor
device and dtype. It preserves dictionary insertion order, float precision,
int64 counts, and a Python float norm from the binary optimizer path.

Acceptance requires exact scalar schema/types/values, parameters, gradients,
optimizer moments and recurrent cache/state gradients against the pinned
original source; unchanged error precedence and optimizer mutation boundaries;
compatible continuous/curriculum checkpoint-resume behavior; and an actual
Torch operation/host extraction census on the permitted CPU hardware.

All ownership, objective, finite-loss, finite-gradient, clipping and
finite-parameter gates retain their source order and scans. In particular,
`gradient_tensors = int(active.sum())` remains before the finite-gradient gate
and optimizer update. The finite-parameter gate still runs after projection
and before reporting work. Diagnostic scheduling remains in the unchanged
continuous/curriculum callers. Their metric schema is unchanged.

## Actual source proof

`probe.py` compiles only `joint_train_step` from Git commit `e8569fb` in a
separate namespace, then compares it to the imported edited production
function. It executes fixed A4 and learned A4 with all affine midpoint
parameters plus binary optimization. The latter uses the existing corrected
reference activation/head path. No earlier snapshot prototype is imported.

Both comparisons are exact for the metric dictionary including Python types
and key order; parameter tensors; gradient tensors; optimizer state (including
Adam step/first/second moments); and earlier recurrent state/K/V gradients.
Actual dispatch census:

| Fixture | Sign float snapshot clones | Pre-update local scalar ops | Post-update local scalar ops | Reporting host extractions |
| --- | --- | --- | --- | --- |
| Fixed A4 | 9 → 0 | 23 → 23 | 18 → 10 | F32 vector of 6, int64 vector of 2 |
| Learned/affine A4, binary recipe | 9 → 0 | 152 → 152 | 32 → 25 | F32 vector of 5, int64 vector of 2 |

The binary recipe already computes its norm as a Python float, so it retains
that value without a new tensor allocation. Each fixture adds two reporting
`aten.stack` operations; integer counts never enter a floating vector. The
retained pre-update scale snapshots are nine floating clones in either
version; the removed nine clones are specifically the sign-master copies.

`summary.json` contains every observed dispatch operation by phase, actual
group dtype/device/length, source hash, all exact metric values, and the bounded
CPU conversion microbenchmark. The pre-update phase includes optimizer work
until `optimizer.step()` returns, preserving a clear mutation boundary.

## CPU cost and limits

Hardware: Apple M3 Max, macOS 27 arm64 CPU, Torch 2.14.0, one Torch thread. No CUDA, Metal,
SSH, native target, real weights, captures, final evaluation, or optimizer
updates on the live project were used. Tiny synthetic fixture optimizer
updates exist only in process memory or temporary test directories.

For already-computed six F32 plus two int64 CPU reporting scalars, five trials
of 2,000 calls give median original conversion cost **1.33 µs**, grouped helper
cost **6.71 µs**. Grouping adds approximately **5.38 µs** on this CPU; it is
not a CPU speed improvement. CPU tensors are already on host, so `.cpu()`
performs no device transfer. The two vector host extractions are structurally
verified CPU calls, not measured CUDA transfers. No accelerator speed or
throughput conclusion follows from this result. Adoption of grouped reporting
needs an actual device measurement by the hardware owner. Clone removal can
be reviewed/adopted independently of reporting batching if that cost is not
justified.

## Checks and integration

Owner checks: **42/42** unittest tests pass in 2.588 seconds across
`test_step_metrics_feature`, `test_recurrent_qat`, `test_continuous_qat`, and
`test_qat_curriculum_runner`. These include actual fixture continuous resume
checks for parameters, optimizer moments, RNG and cursor, and curriculum
midphase/boundary resume with optional parameter families. Source/report
runner/test Ruff checks and `git diff --check` pass. The independent validator
owns `test_step_metrics_feature_validation.py` and [VALIDATION.md](VALIDATION.md).
Its final **6/6** checks pass in 0.450 seconds, including mixed floating dtypes
with a float64 value that would detect narrowing, int64 above 2^53, populated
AdamW moment preservation on rejection, ownership/objective precedence and
the post-update finite-parameter gate before scalar extraction. Validator
commits are `2d16b11` and `b947a96`; source implementation is `bdb8b42`.
The final combined **48/48** tests pass in 2.546 seconds, with Ruff and diff
checks passing. No source edit occurred after the actual census proof.

Reproduce the owner check from this worktree:

```sh
PYTHONPATH=src:.:tests /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest test_step_metrics_feature test_recurrent_qat test_continuous_qat test_qat_curriculum_runner
PYTHONPATH=src:.:tests /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest test_step_metrics_feature test_step_metrics_feature_validation test_recurrent_qat test_continuous_qat test_qat_curriculum_runner
PYTHONPATH=src:.:tests /Users/pippo/github/binary-eagle-decoding/.venv/bin/python experiments/parallel20261002/step_metrics_feature/probe.py
```

Deliverable complete. Remaining integration work: root review of whether to
adopt reporting batching given the CPU cost; future device measurement if
batching is admitted. Root owns the active goal checkpoint and integration;
this branch must remain isolated until reviewed and must not be removed while
unmerged or unpublished. No persistent experiment process or device resource
remains after the owner and validator checks.

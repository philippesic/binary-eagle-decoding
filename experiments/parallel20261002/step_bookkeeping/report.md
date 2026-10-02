# Diagnostic sign snapshots: remove redundant float clones

The isolated production-function variant passes the local acceptance gate.
Replace `m.latent_sign.detach().clone() < 0` with
`m.latent_sign.detach() < 0`. The comparison allocates its own Boolean storage;
the subsequent optimizer mutation does not change the captured signs under the
current sequential same-stream step. This is
the only proposed source change. Live core source is untouched.

Scope clearance: `/root/qat_priority` confirmed no equivalent active diagnostic
fix in recorded QAT assignments. Evaluator, memory admission and resume adoption
remain with their existing owners. Source binding and the unapplied one-line
patch are under `research/parallel20261002/step_bookkeeping/reference/`.

## Actual step and operation evidence

Owner checks use the real `joint_train_step`, its real ownership/finite checks,
optimizer, projections and diagnostics. The candidate compiles the same function
in an isolated namespace with exactly the one expression replaced; a pinned
function SHA256 rejects source drift. Both execute the existing three-position
`tiny_joint_fixture`/`tiny_rollout`, using all nine selected modules and attached
earlier state/K/V gradients. No real weights, captures, data or final prompts
were accessed.

| Actual reduced recipe | Sign flips | Active gradient tensors | Snapshot float clones, baseline → candidate | All step clone calls | Boolean snapshots/bytes | Scalar extraction calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed A4 | 68 | 18 | 9 → 0 | 27 → 18 | 9 / 144, unchanged | 41, unchanged |
| Learned A4, affine all nine, declared binary optimizer | 60 | 33 | 9 → 0 | 27 → 18 | 9 / 144, unchanged | 184, unchanged |
| Fixed A4, injected NaN gradient | no update | n/a | 9 → 0 | 27 → 18 | 9 / 144, unchanged | 4, unchanged |

Metrics compare exactly, including loss, token loss, midpoint penalty, gradient
norm, active gradient count, sign flips, scale movement, saturation, and latent
clip violations. Unique parameter gradients, parameters after projection,
optimizer state, retained earlier state/K/V gradients and all before-sign tensors
also compare exactly (`rtol=atol=0`). Optional learned/affine state is exercised
once, with nonzero affine regularization and midpoint values. Both successful
fixtures have nonzero sign flips and scale movement, so equality does not rely
on an idle update.

`TorchDispatchMode` counts actual operations. Snapshot allocation classification
tracks the original master storage and transient clone lineage only until
`zero_grad`; it removes freed clone addresses after comparison to avoid allocator
address reuse misclassification. Every captured Boolean snapshot owns storage
distinct from its source master and from all other snapshots, and remains exact
after the actual update. A direct mutation control also covers negative zero,
positive zero and NaN comparison behavior.

The NaN gradient produces `nonfinite joint QAT gradient` in both functions,
with zero optimizer calls, unchanged parameters and empty optimizer state.
Additional error-order fixtures show invalid optimizer ownership fails before
`zero_grad` and objective validation; forbidden hard-CE teacher input fails
after `zero_grad` but before any update. The replacement moves no safety gate.

Scalar extraction consolidation was deliberately left outside this patch.
These dispatcher counts include safety and recipe validation calls; they are
CPU structural counts, not measured CUDA synchronization or latency.

## Configured allocation arithmetic

`configs/continuous_w1ax.json` has nine shapes totalling 218,234,880 weights per
lane. This calculation reads configuration only and allocates no full-shape
tensor. Per-projection arithmetic and source hashes are in `summary.json`.

| Quantity per lane-step | Bytes |
| --- | ---: |
| Removed F32 copy payload, `218234880 × 4` | 872,939,520 |
| Nominal removed clone read plus write, `218234880 × 8` | 1,745,879,040 |
| Largest removed clone, head `[32000, 2560] × 4` | 327,680,000 |
| Retained Boolean sign snapshots, `218234880 × 1` | 218,234,880 |

The payload total is an aggregate over nine sequential copies, not an aggregate
live-memory reduction. The largest float clone is a transient allocation.
The Boolean comparison and Boolean snapshots remain; actual allocator peak and
hardware memory traffic are not measured here.

## Checks and adoption

Owner: 7/7 focused tests pass on Apple M3 Max, macOS 27 ARM64, PyTorch 2.14.0,
CPU (10 configured Torch threads). Hardware identity was checked with
`sysctl -n machdep.cpu.brand_string`. There is no CUDA, SM75, Metal or GPU
throughput claim. Operation counts are the selected decisive structural gate;
no synthetic timing sweep was needed.

```sh
PYTHONPATH=src:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest research.parallel20261002.step_bookkeeping.reference.test_snapshot_step -v
PYTHONPATH=src:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m research.parallel20261002.step_bookkeeping.reference.audit
/Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff check research/parallel20261002/step_bookkeeping/reference
git apply --check research/parallel20261002/step_bookkeeping/reference/snapshot-only.patch
git diff --check
```

Independent Luna validation uses separately constructed 3×3 A1 fixtures and
seeded existing AdamW moments. Fixed, learned and affine variants compare
exactly across 99, 135 and 144 tensor entries, respectively, on Torch2.8 and
Torch2.14 CPU. Its independent whole-step census confirms latent float clones
9→0, scale clones9→9 and Boolean comparison outputs27→27. Its seeded optimizer
state also remains unchanged after NaN rejection. Commands, raw hashes and
environment details are in [validation.md](validation.md).

Root should integrate this report/prototype/patch only, checkpoint
the existing QAT goal, push main and retire the merged worktree after preserving
ignored raw logs. Any live-core adoption belongs to the QAT source owner and
requires current-source step checks; actual GPU performance is its next gate.

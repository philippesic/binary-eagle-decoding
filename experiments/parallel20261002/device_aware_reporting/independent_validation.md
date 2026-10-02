# Independent CPU validation: device-aware scalar reporting

Validated the reporting fast path in the current worktree against an independent
copy of the grouped scalar conversion pinned at `93a2ff1`. The validation only
used CPU tensors and the tiny synthetic trainer fixture.

## Environment and run directory

- Worktree: `/private/tmp/eagle-parallel-20261002/device-aware-reporting`
- Python: 3.11.15
- PyTorch: 2.14.0
- Platform: macOS 27.0 arm64
- Run directory: `runs/parallel20261002/device-aware-reporting-validation/`
- Test source SHA-256: `2250df6cd29d43bb00fc01ea3c045c99392efa092a1a1706f0b43398b6672510`

## Commands and results

The final focused validation command was:

```sh
PYTHONPATH=src:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  tests/test_parallel20261002_device_reporting_validation.py
```

Result: **4 tests passed in 0.472 seconds**. It checks exact values, order, and
Python scalar types for bool, int64 above `2**53`, float16, bfloat16, float32,
float64, and native Python values. It also executes three seeded real
`joint_train_step` updates with ordinary hard CE and with `depth_loss_decay=0.8`.
The resulting metrics, model parameters, and optimizer state match exactly
against the pinned grouped conversion path.

Error-path checks confirm optimizer ownership is rejected before objective
validation or mutation, hard CE rejects an invalid teacher before optimizer
step, nonfinite gradients reject before optimizer step, and post-update
nonfinite parameter validation runs before reporting conversion.

Additional checks:

```sh
/Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff check \
  tests/test_parallel20261002_device_reporting_validation.py
git diff --check
```

Ruff and `git diff --check` passed. An initial test attempt exposed a missing
`unittest` import in the new test harness; it was fixed before the final run.
Ruff's initial findings were also fixed before the passing check.

## Cleanup and retained output

All test and lint processes exited. There is no persistent process or device
allocation. Raw command output is retained under the ignored run directory in
`final-tests.log`, `final-ruff.log`, and `environment.txt`; the first harness
attempt is retained as `initial.log` and `run01.log`. No weights, captures, or
other model artifacts were created.

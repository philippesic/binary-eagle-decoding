# Step metrics feature: independent validation

Independent validator: `/root/step_metrics_feature/metrics_validation`.

## Scope and environment

Validation exercised the synthetic tiny joint QAT fixture on CPU only. The host
was an Apple M3 Max (`Darwin 27.0.0`, arm64), using Python from
`/Users/pippo/github/binary-eagle-decoding/.venv/bin/python` and PyTorch
`2.14.0`. CUDA was unavailable; MPS was available but not used. No GPU, Metal,
SSH, model weights, captures, live recipe, or remote host was accessed.

The shared research control was checked before each test chunk. It showed 89%
weekly usage, the original reset value `1791049896`, and both `research_stop`
and `reset_observed` false. Remaining allowance was 11%, above the 1% stop
threshold.

## Checks

The added CPU tests cover:

- `_reporting_scalars` key order and mixed Python/tensor scalar types across
  float16, bfloat16, float32, float64, and int64, including exact preservation
  for `2**53 + 137`; the float64 probe includes a bit small enough to be lost
  by narrowing to float32;
- repeated seeded two-step AdamW runs, equality of optimizer moments and model
  parameters, compatibility of the nine-key metrics schema, and JSON plus
  `torch.save`/`torch.load` round trips for checkpoint-shaped metrics;
- error precedence for a rejected hard-CE step and unchanged parameters and
  optimizer moments after rejection;
- nonfinite gradients rejected before `optimizer.step`, with seeded parameters
  and AdamW moments unchanged;
- optimizer-ownership precedence over an invalid teacher, before `zero_grad`;
- nonfinite updated parameters rejected before reporting scalar extraction.

Exact successful commands, run from
`/private/tmp/eagle-parallel-20261002/step-metrics-feature`:

```sh
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest discover -s tests -p test_step_metrics_feature_validation.py -v > /Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/step-metrics-feature-validation/unittest.log 2>&1
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m ruff check tests/test_step_metrics_feature_validation.py > /Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/step-metrics-feature-validation/ruff.log 2>&1
```

Result: all 6 validator tests passed in 0.450 seconds; Ruff passed after
formatting one overlong line. Raw logs
are retained in the ignored run directory above. A first attempt with system
Python had no `pytest`; `uv run --no-sync` created an empty worktree `.venv`
with no Torch. Both attempts stopped before test execution. The temporary
worktree `.venv` was removed. The test's temporary checkpoint file was removed
by `TemporaryDirectory`; no experiment process or GPU allocation remains.

## Limit

The checkpoint round trip checks the produced metrics payload shape and scalar
types; it does not instantiate the full continuous trainer or write its paired
model/export checkpoint. Trainer consumers and checkpoint metric placement
were inspected read-only. This validation establishes reporting compatibility
on CPU and does not establish CUDA synchronization cost or accelerator
performance.

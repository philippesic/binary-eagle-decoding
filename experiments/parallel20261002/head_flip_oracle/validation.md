# Independent CPU validation: row head flip oracle

This run independently reconstructed each nominated candidate by flipping a
copy of the complete integer sign matrix, computing the dense `codes @ signs.T`
head, and applying the F32 epilogue in native order: `(dot * alpha) * beta`,
then `(sum(codes) * midpoint) * beta` when present, then bias. Hard CE and
rank telemetry were recomputed from the resulting logits. No dot-delta formula
or implementation candidate output was used to build the reference.

## Run record

- Worktree and branch: `/private/tmp/eagle-parallel-20261002/head-flip-oracle`,
  `research/20261002-head-flip-oracle`.
- Base revision at validation: `7fcaa84` (owner implementation commit).
- Run directory: `/private/tmp/eagle-parallel-20261002/head-flip-oracle`.
- Raw test output: `runs/parallel20261002/head-flip-oracle-validation/unittest.log`
  (ignored by Git).
- Environment: macOS 27.0 arm64, CPython 3.11.3, PyTorch 2.8.0. The test
  instantiated CPU tensors only; CUDA was unavailable (`torch.cuda.is_available()`
  returned false). No model, dataset, Metal, SSH, or GPU was used.
- Main control at the final run: 93% used, original reset `1791049896`,
  `research_stop=false`, `reset_observed=false` (7% remaining). The absolute
  MAIN `runs/parallel20261002/control.json` was checked before the run and by
  each test case/configuration.

Exact commands, from the worktree root:

```sh
ruff format research/parallel20261002/head_flip_oracle/validation/independent_oracle.py tests/test_parallel20261002_head_flip_validation.py
ruff check research/parallel20261002/head_flip_oracle/validation/independent_oracle.py tests/test_parallel20261002_head_flip_validation.py
ruff format --check research/parallel20261002/head_flip_oracle/validation/independent_oracle.py tests/test_parallel20261002_head_flip_validation.py
python3 - <<'PY'
from research.parallel20261002.head_flip_oracle.validation.independent_oracle import check_control
state = check_control()
print('control_ok remaining_percent=', 100 - state['last_weekly_used_percent'], 'reset=', state['last_reset_unix'])
PY
PYTHONPATH=src python3 -m unittest discover -s tests -p test_parallel20261002_head_flip_validation.py -v 2>&1 | tee runs/parallel20261002/head-flip-oracle-validation/unittest.log
```

## Results

All 5 validator tests passed in 0.051 seconds. Across fixed symmetric,
learned-activation, and fixed-affine-v1 `RowBinaryLinear` heads at A1, A4, and
A8, the audit covered 9 configurations and 27 candidates. It compared 135
candidate-row token logits and 27 dense integer-dot vectors. All integer dots
and F32 candidate logits matched exactly in this fixture; the maximum observed
logit difference was 0.0, within the gate
`8 * eps32 * max(1, abs(reference))`.

For the 108 supported token/candidate CE values, maximum absolute difference
from independent F64 `logsumexp` was `4.718447854656915e-16`, within
`2e-10 + 2e-12 * abs(reference)`. Unsupported labels retained NaN CE and label
margin. The stable LSE check used self-consistent logits with row scales of
`1e28`; all reported CE deltas and top-1 margins remained finite.

There were 89 token decisions outside the per-token margin gate, and all 89
top-1 rows agreed exactly. The remaining 46 tokens were explicitly reported
as ties/near-ties; the exact tie fixture selected the lowest dense row before
mapping it through the deliberately nonidentity `d2t` table. No decision claim
is made for values inside the margin gate.

Baseline max absolute drift between the native-order captured head and the
module forward was:

| Activation rule | A1 | A4 | A8 |
| --- | ---: | ---: | ---: |
| Fixed symmetric | 0 | `9.5367431640625e-07` | `8.344650268554688e-07` |
| Learned | 0 | 0 | 0 |
| Fixed affine v1 | 0 | 0 | 0 |

The fixed-symmetric A4/A8 differences reflect the legacy dense-dot operation
order. They remain below the row-logit gate for this fixture and did not change
any decisive top-1 result. The independent baseline itself uses native-order
F32 arithmetic.

Coverage also included width 131 and flips in the final unaligned columns,
both signed-zero and negative-subnormal inputs, zero activation and weight
scales, a zero weight sign, frozen bias, affine midpoints, an unsupported
absolute target ID, and unchanged module state. The legacy fixed symmetric A4
subnormal reciprocal produced invalid codes and was rejected by capture;
learned A4's specified safe fallback returned finite logits. A separate full
sign-flip example showed the relaxed local CE derivative predicting an
improvement while the actual discrete flip increased CE and reduced accuracy
from 0.75 to 0.25.

Source SHA-256 values at validation:

| File | SHA-256 |
| --- | --- |
| `research/parallel20261002/head_flip_oracle/reference/oracle.py` | `4f009565ad19da8e7165fc130a8c7de2d426d1cae473fbdbaf5101ebbe60f828` |
| `research/parallel20261002/head_flip_oracle/validation/independent_oracle.py` | `be3dd614652ee3de18b3341e774602ebbacc8d7a5deb992d48650f77c423e7b0` |
| `tests/test_parallel20261002_head_flip_validation.py` | `ef5e39b316205fcf6a87d5a2ff7e45157ef280111b24adf32cfddc1c84e7d2fd` |

`ruff check` and `ruff format --check` passed. The one-shot CPU test process
exited normally; there was no persistent process or accelerator allocation to
clean up. The ignored raw test log was retained for review.

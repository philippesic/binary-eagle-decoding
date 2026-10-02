# Sign inertia independent validation

## Scope and environment

This CPU-only check independently enumerated hard sign states and exercised
`qat_optimization.py` through `RowBinaryLinear` and `W1AxContract`. It used no
model weights or datasets. The shared stop file
`/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json`
was read before each run; `research_stop=false` and `reset_observed=false`.

- Worktree: `/private/tmp/eagle-parallel-20261002/sign-inertia`
- Branch/HEAD: `research/20261002-sign-inertia`, `6c613039d2c6bef3f6f7b5c6e9bf5f8cbccac2e8`
- Host/device: macOS 27.0 arm64, CPU
- Python: 3.11.15; PyTorch: 2.14.0; precision: float32
- `PYTHONPATH=src`
- Source SHA-256:
  - `src/w1a1_eagle/qat_optimization.py`: `d2da18a4f4db612dd3cb51916af688109593d2c9d4cf119a21f8129567c305f7`
  - `src/w1a1_eagle/recurrent_qat.py`: `f2b6d83b8929807b65b6f21826cc99097a6ddd70f99ff23ca0836aebe4db1224`
  - `src/w1a1_eagle/recurrent_binary.py`: `7624fa4486bcb130440cbc6b84c2463af01f0750e0145236063a125ab8b3de06`

## Results

For the four 3D inputs, desired weight signs `(+, -, +)` induce labels
`++-+`. All eight sign rows have strict nonzero margins; only `(+, -, +)`
matches all four labels. The other seven predictions are `-+++`, `---+`,
`--+-`, `+-++`, `-+--`, `+++-`, and `+---`. For the no-bias XOR fixture,
none of the four 2D hard sign rows matches labels `+--+`, including under
positive scale values 0.001, 0.5, and 17.0.

The recurrent four-state enumeration used corrected desired signs `(+, -)`.
I initially coded its activation scale as max-absolute and found that its logits
differed from W1Ax A1. I corrected the independent scalar oracle to the
production mean-absolute rule and then matched its logits to
`RowBinaryLinear` for all four sign rows at scales 0.01, 1, and 100 (`atol=1e-5`,
`rtol=1e-6`). There is no remaining discrepancy. Both implementations found
0 feasible states at scale 0.01, where the frozen +0.15 bias dominates, and 1
feasible state at scales 1 and 100.

I also checked the owner's stored 99-trajectory result independently. The 96
AdamW grid cases covered all planned unique configurations, with 3 optimizer
controls. Recomputed per-update useful and wrong flip counts agreed with every
summary: 56 useful events, 32 wrong events, 102 observed flips, and 30 flip
backs. The recurrent scale-1, magnitude-0.02, joint-scale baseline had its
first useful flip on update 20, 3 wrong flips, 22 fully correct updates, and
6 fully correct updates in its final 16 steps. The stored capacity tables also
matched independent enumerations for every scale: linear 1 feasible state per
scale, recurrent 0/1/1 at scales 0.01/1/100, and XOR 0 at all scales.

Constant positive gradient, sign LR 0.125, and projection yielded these latent
paths and hard signs:

| Optimizer | Initial latent | Latents after updates 1, 2, 3 | Hard signs |
| --- | ---: | --- | --- |
| SGD, no momentum | +0.25 | +0.125, 0, -0.125 | +, +, - |
| AdamW | +0.25 | +0.1250000149, +1.4901161e-8, -0.1249999851 | +, +, - |
| SGD, momentum 0.5 | +0.3125 | +0.1875, 0, -0.21875 | +, +, - |

Thus zero retains the positive hard sign. SGD without momentum reaches zero on
update 2; AdamW remains a small positive float32 residual at that update due to
epsilon arithmetic; both cross on update 3. With momentum 0.5, SGD's
displacements are 1/8, 3/16, and 7/32, reaching zero on update 2 and crossing
on update 3. The checks also confirmed that changing latent magnitude among
0.01, 0.1, 0.5, and 1.0 preserved the hard forward exactly; at exactly zero
effective row scale, the `weight_unit` rule changed an assigned sign gradient
of 1 to 0 and the sign did not revive; and observations at steps 2 and 4
reported no flips while missing four intervening alternating sign changes.

## Commands, output, and cleanup

The validation test command was:

```sh
PYTHONPATH=src /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest tests/test_parallel_sign_inertia_validation.py -v > runs/parallel20261002/sign_inertia/validation/test.log 2>&1
```

Result: 7 tests passed in 0.442 seconds. The command's raw stdout/stderr are in
the ignored run directory
`/private/tmp/eagle-parallel-20261002/sign-inertia/runs/parallel20261002/sign_inertia/validation/test.log`.
The test source is
`/private/tmp/eagle-parallel-20261002/sign-inertia/tests/test_parallel_sign_inertia_validation.py`.

The independent stored-results audit command was:

```sh
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python research/parallel20261002/sign_inertia/validation/check_stored_results.py > runs/parallel20261002/sign_inertia/validation/stored_results_check.log 2>&1
```

It passed all 99 trajectory, capacity-table, and useful/wrong summary checks.
Its raw output is retained at
`/private/tmp/eagle-parallel-20261002/sign-inertia/runs/parallel20261002/sign_inertia/validation/stored_results_check.log`.
No process remains running; there were no external resources to release. The
ignored run log is retained as the raw output. The first assertion pass exposed
that AdamW's epsilon leaves a `1.49e-8` positive residual instead of exact zero;
the check was narrowed to test that observed residual and the correct positive
hard sign. No production source was changed.

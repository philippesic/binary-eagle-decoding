# Independent activation reuse validation

## Scope

This CPU-only validation checked the isolated immediate-sibling reuse helper in
the reduced all-nine `NativeStepAdapter` graph. It used learned A1/A4/A8, three
proposal steps, the serial learned head, the existing affine and fusion
correction fixtures, context caching, and shared hard-sign scope. The baseline
and reuse graph outputs and VJPs were compared both to each other and to the
read-only auxiliary-VJP fixture's independent handwritten
`AlgebraicProjection`/`AlgebraicCorrection` oracle.

No production source was changed. No GPU, Metal, SSH, model weights, or data
were used. This run provides CPU structural evidence only; it does not predict
SM75/CUDA performance.

## Results

For each activation width, all 42 compared logits, loss, parameter/input/cache
VJPs matched between baseline and reuse at `atol=2e-6`, `rtol=5e-5`. Maximum
absolute reuse deltas were `5.82e-10` (A1), `1.49e-8` (A4), and `1.49e-8`
(A8). Against the independent handwritten oracle, maximum absolute errors
were `5.96e-8`, `2.38e-7`, and `2.38e-7` respectively.

The saved-storage census inspected the two tensors saved directly by each
`_LearnedActivationSTE` node before backward and deduplicated their backing
storage pointers. Across this three-step graph, QKV support/derivative storage
dropped from 18 storages / 720 bytes to 6 / 240 bytes. Gate/up dropped from 12
/ 240 bytes to 6 / 120 bytes. The graph therefore saved 18 unique support and
derivative storages, or 600 bytes of retained saved-tensor payload. Quantizer
wrapper invocation totals remained 11/11 for QKV (including two no-grad
context calls) and 6/6 for gate/up; each attached proposal group instead showed
one miss and two QKV hits, or one miss and one gate/up hit. Every group ended
with `retained=0`.

A weak-reference control independently confirmed that the cache's result object
became collectible after the immediate group exited and the caller dropped its
reference. A mutation between same-input calls caused two misses and changed
the output. A no-grad call followed by a grad-enabled call also caused two
misses, returned a detached first output and attached second output, and
produced a finite parameter gradient.

One clipped A8 update compared logits, clipped gradients, all 36 parameters,
optimizer tensors, and metrics across 181 tensor values. The gradient norm was
`0.06999999285` for a `0.07` limit; the largest update/state difference was
`3.64e-11`. All reported scalar metrics matched exactly.

## Commands and environment

Worktree: `/private/tmp/eagle-parallel-20261002/activation-reuse`, branch
`research/20261002-activation-reuse`.

Environment: macOS 27.0 arm64, Python 3.11.3, PyTorch 2.8.0, CPU, one Torch
thread. Before and after the run, the absolute control file
`/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/control.json`
reported `research_stop=false`, `reset_observed=false`, reset `1791049896`
unchanged, and 15% weekly allowance remaining.

The exact final validation command was:

```sh
PYTHONPATH=src:. python3 research/parallel20261002/activation_reuse/validation/independent_validation.py > runs/parallel20261002/activation_reuse/validation-independent-01/final.log 2>&1
```

It exited 0. The script also wrote `result.json` into the same ignored run
directory. The script was syntax-checked with:

```sh
python3 -m py_compile research/parallel20261002/activation_reuse/validation/independent_validation.py
```

Lint and formatting checks passed:

```sh
ruff check research/parallel20261002/activation_reuse/validation/independent_validation.py
ruff format --check research/parallel20261002/activation_reuse/validation/independent_validation.py
```

An initial harness attempt exited 1 because its stale-control vector had four
features while the selected gate/up quantizer requires eight. The fixture was
corrected to use the gate/up width; the final run above passed. The initial
trace is retained as `primary.log` in the ignored run directory.

Raw outputs are retained in
`runs/parallel20261002/activation_reuse/validation-independent-01/` and are
ignored by Git. Final `result.json` SHA-256:
`c5ded69bb7aca0c35c5bd5ff7d7a5f3dc38f77b371be09d4464939106cd52d85`.

Source SHA-256 values at validation time:

| File | SHA-256 |
| --- | --- |
| `research/parallel20261002/activation_reuse/reference/prototype.py` | `0fb463b254496036be36210abb22ffba3b55bd1078ba6c83be4476816174d9ed` |
| `research/parallel20261002/activation_reuse/validation/independent_validation.py` | `b6b92a1878d53026536ff935ea65981ce56092a473f5b2ea8d73b674247a68ce` |
| `research/parallel20261002/auxiliary_vjp/reference/audit.py` | `39420bda1987c7f4f54936ad8e0eb78135d9312c882f6f82a8297b4a9e4de958` |

The test process exited, no validation process remains, and the GPU was not
used.

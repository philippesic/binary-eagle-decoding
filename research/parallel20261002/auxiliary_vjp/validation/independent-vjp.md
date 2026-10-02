# Independent combined auxiliary VJP validation

## Scope and derivation

This validation follows packet A in `docs/parallel20261002/astra-research.md`.
It checks the affine-midpoint and raw fusion-correction VJPs through their
public APIs, then independently reruns the owner’s handwritten-algebra audit
through the actual recurrent adapter/provider graph. Inputs are fixed CPU
synthetic tensors; no fitting, optimizer sweep, model, dataset, GPU, Metal, or
SSH was used.

For one affine projection, let `Q` be the attached quantized-value surrogate,
`C` the hard integer codes, `S` the binary sign rows, `a` the row scales, `μ`
the row midpoints, and `β` the activation scale. The hard output is
`((C Sᵀ) ⊙ a) β + (sum(C) ⊙ μ) β + bias`. The API’s declared surrogate gives,
for upstream row gradient `G`,

```text
dQ       = (G ⊙ a) S + (G μ) 1ᵀ
dS       = (G ⊙ a)ᵀ Q
da       = row_sum(G ⊙ (Q Sᵀ))
dμ       = row_sum(G ⊙ sum(Q))
dbias    = row_sum(G)
```

The midpoint branch uses the attached `Q` sum. When two projections consume
the same cached sum, their `dQ` contributions add through the shared node.
For A1/A4/A8, the actual `hard_activation_with_codes` API supplies the hard
codes, scale, and attached-value STE; no hard quantizer is finite-differenced.

For fusion correction, `Uh=F16(U)` and `Vh=F16(V)` are the hard forward
factors, `Z=X Vhᵀ`, and `D=Z Uhᵀ + clamp(bias)`. The declared factor STE passes
the ordinary matrix VJPs to the F32 masters:

```text
dU       = Gᵀ Z
dV       = (G Uh)ᵀ X
dX       = (G Uh) Vh
dbias    = row_sum(G) where bias is inside its clamp bounds
```

When correction and quantization share the raw input, `dX` from correction
adds to the quantizer path. The focused check uses nonzero affine scales,
midpoints, biases, U and V; half-rounded U/V differ from their F32 masters.
Its detached-factor negative control retains a raw-input gradient but produces
no U/V parameter gradients. The actual-graph audit separately checks detached
first-state/cache paths, rejection of a split tied-quantizer alias, and
cross-boundary parameter aliasing.

## Commands and results

Worktree: `/private/tmp/eagle-parallel-20261002/auxiliary-vjp` on
`research/20261002-auxiliary-vjp`. Runtime: macOS arm64 CPU, Python 3.11.3,
PyTorch 2.8.0. The configured project interpreter
`/Users/pippo/github/binary-eagle-decoding/.venv/bin/python` was not accessible
in the validator execution context, so system `python3` was used. No GPU runtime
was queried.

Before each test run, `runs/parallel20261002/control.json` showed
`research_stop=false`, `reset_observed=false`, reset `1791049896` unchanged,
and 18% weekly usage remaining at the first run (82% used). The exact commands
and outcomes were:

```sh
PYTHONPATH=src python3 tests/test_parallel20261002_auxiliary_vjp_validation.py
# Ran 3 tests in 0.019s — OK

PYTHONPATH=src python3 -m unittest discover -s tests \
  -p 'test_parallel20261002_auxiliary_vjp_reference.py' -v
# Ran 3 tests in 1.190s — OK (updated alias controls)
```

The focused tests compare every listed gradient to the equations above for
A1/A4/A8 and confirm shared-boundary reuse. The owner suite rerun covered
48 actual adapter/provider VJP comparisons (three precisions, two depths,
reference/single-forward, cache/chunk/shared-sum modes), three clipped joint
updates, and detach/alias negative controls. All passed. The owner’s
`summary.json` records the PyTorch 2.14.0 run, including the 48 comparison
table, 36-family clipped updates, and both alias controls. The maximum reported
composed VJP error is `2.3841858e-7`; the maximum one-update parameter error is
`9.0379649e-7`, within the declared `atol=2e-6`, `rtol=5e-5`. The raw full
result is at
`/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/auxiliary-vjp/results.json`.
My rerun used PyTorch 2.8.0 and did not overwrite the owner’s report.

Raw outputs are retained at
`/Users/pippo/github/binary-eagle-decoding/runs/parallel20261002/auxiliary_vjp/validation/`
(`focused-tests.log` and `owner-audit.log`). The project `.gitignore` excludes
that `runs/` tree. Both Python processes exited; no process or persistent
resource required cleanup. No production source was changed.

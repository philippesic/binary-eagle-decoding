# Combined auxiliary recurrent VJPs — synthetic CPU

The composed all-nine NativeStepAdapter/provider gate passes all **48** declared
comparisons for learned A1/A4/A8 at draft depths 1 and 3. Every one of the **36
unique trainable tensors** has a nonzero later-loss VJP at depth 3 in each
precision. Hard logits and loss match exactly. Largest VJP error is
`2.384185791015625e-7`, within the
predeclared F32 tolerance `atol=2e-6, rtol=5e-5`. This falsifies a missing
auxiliary-gradient concern for the tested synthetic composition; it does not
establish model quality, Q4_0-relative acceptance, native SM75 timing, or full
provider readiness.

Owner `/root/auxiliary_vjp`; independent Luna validator
`/root/auxiliary_vjp/auxiliary_validation`. Isolated base
`ca02272ae4c5740077c96eb2b3993435c814f444`, branch
`research/20261002-auxiliary-vjp`, worktree
`/private/tmp/eagle-parallel-20261002/auxiliary-vjp`. No core/live source, recipe,
model, data, held-out/final, GPU, Metal or SSH action occurred.

## Fixture and boundaries

The first-slate reduced model and explicit serial topology are imported
read-only and source-hashed. All nine row projections receive learned
activations, affine midpoints and nonzero frozen biases; FC also receives rank-1
fusion correction with nonzero U, V and interior output bias. Six activation
boundaries are unique; Q/K/V and gate/up tie their declared scalar. Midpoints
span `[-0.027, 0.031]`; learned A1 thresholds span `[0.13, 0.18]`, A4/A8 clip
ratios `[0.71, 0.81]`. Correction factors use deterministic seed 1062, with
nonzero scales 0.09/0.12. No fitting or sweep occurs.

Each depth/precision compares eight provider variants with one explicit serial
oracle: computation labels reference/single_forward, shared-round signs and
affine sums on/off, cache-only optimized reconstruction off/on, and cache
chunks 1/2. Learned heads remain **serial** (`optimize_head=False`) to avoid
repeating the already-proven head batching normalization defect. The all-row
affine source branch always dispatches `affine_binary_projection` with
`single_forward=True`; the computation labels therefore do not select two
independent affine kernels. The independently written projection oracle supplies
the actual alternate backward algebra.

Accepted prefix length is 2 and an invalid terminal row is included. Accepted
context is reconstructed under no-grad and intentionally truncated. Proposal
state and F16-roundtrip K/V remain attached. Loss uses only the final valid
depth for the VJP gate, demanding paths through earlier proposals. Raw accepted
context VJPs are exactly zero and the seed raw-feature VJP is nonzero. The table
also covers the first proposal input state and its appended K/V cache tensors.
Depth-3 first-state/key/value VJP norms are all nonzero in A1/A4/A8.

## Independent backward algebra

The oracle shares the production **no-grad hard learned quantizer reference**
and real adapter attention/RMS/RoPE/F16-cache arithmetic. It independently
derives all local sign, scale, learned activation, midpoint and fusion VJPs;
it does not call their production backward implementations. This is a bounded
surrogate composition check, not an independent hard-quantizer forward audit.

For quantized values Q, hard signs S, row scale alpha, midpoint mu and output
cotangent G, the surrogate row is `Q @ (alpha*S + mu).T`. Its VJPs are:

- `dQ = (G*alpha) @ S + sum_rows(G*mu)` broadcast across input features.
- `dS = (G.T @ Q)*alpha`; latent gradients additionally mask `abs(latent)<=1`.
- `dalpha = sum_features((G.T @ Q)*S)`, masking negative raw scales.
- `dmu = sum_tokens(G*sum_features(Q))`.

The hard affine epilogue separately follows integer dot→alpha→beta and integer
sum→mu→beta before bias. Input and learned scalar VJPs use the declared clipped
identity/LSQ surrogate: A1 support is `abs(x-delta*beta)<=beta`, with parameter
derivative `-beta/sqrt(N)` on that support; A4/A8 use support `abs(x)<=M*c` and
parameter derivative `(Q-x)/c` inside or `Q/c` outside, divided by
`sqrt(N*qmax)`. N is each serial invocation's feature count. Shared consumer
contributions accumulate into one scalar exactly once per consumer; this is
distinct from changing its normalization domain.

For correction `Z=x @ V16.T`, `D=Z @ U16.T`, the independently derived VJPs
are `dU=G.T@Z`, `dV=(G@U16).T@x`, `dx=(G@U16)@V16`; F16 factor rounding has
identity surrogate backward. Interior output-bias VJP is the token sum of G.
Both correction factors are nonzero, preventing zero initialization from
hiding an absent V/input gradient.

## Joint update and negative controls

One actual `joint_optimizer`/`joint_train_step` AdamW update per precision
owns all 36 tensors exactly once. Mean CE over the three valid synthetic rows
is used for this update, with zero midpoint regularization to isolate VJPs.
The production joint clipping bound is 0.07:

| Precision | Before clipping | After clipping | Max clipped VJP error | Max parameter error |
| --- | ---: | ---: | ---: | ---: |
| A1 | 2.099605 | 0.069999968 | 9.94e-10 | 9.04e-7 |
| A4 | 5.546794 | 0.069999991 | 1.86e-9 | 1.39e-7 |
| A8 | 5.639069 | 0.069999992 | 9.31e-10 | 1.82e-12 |

All updates pass the same declared tolerance; all 36 gradient tensors are
active. This one-step gate makes no sign-movement, budget-adequacy or optimizer
recommendation. No resume or second-update study repeats prior work.

Intentional state/cache detachment leaves hard logits exactly unchanged but
fails the all-family VJP checker, maximum differences `1.985554` and `0.107273`.
Splitting tied QKV scalar ownership and aliasing head/FC scalar parameters
both fail the declared activation-bank ownership check.

## Reproduction and evidence

Use the existing project CPU Torch 2.14.0 environment on macOS arm64:

```sh
PYTHONPATH=src:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest discover -s tests -p 'test_parallel20261002_auxiliary_vjp*.py' -v
PYTHONPATH=src:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m research.parallel20261002.auxiliary_vjp.reference.audit
```

Owner checks: 3/3 focused tests; Ruff check/format pass. Independent validator's
three hand-VJP tests and owner suite also pass on system Torch 2.8.0; the owner
reran the combined **6/6 tests** with the project Torch 2.14.0 interpreter.
The validator committed evidence as `51df26e` (polished in `f99f03f`); its report records commands,
runtime limitations, independent formulas and ignored logs. The committed
[summary](summary.json) gives six source-bound tables for every unique parameter,
raw/first state/cache VJP and the eight declared modes; it preserves all source
hashes, loss/norm/error values and the full ignored result hash. Full mode
tables stay at `runs/parallel20261002/auxiliary-vjp/results.json` and were copied
to the main workspace for retirement safety. No raw tensors or runs enter Git.
Independent validation is recorded separately under
`research/parallel20261002/auxiliary_vjp/validation/`.

Integration: evidence/test files only. No source patch or recipe change is
proposed. The orchestrator reviews/cherry-picks, checkpoints the active goal,
pushes main, and retires this worktree after preserving ignored artifacts.
Actual-model/native and Q4_0-relative gates remain with their existing owners.

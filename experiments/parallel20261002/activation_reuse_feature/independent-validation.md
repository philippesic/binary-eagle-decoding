# Independent CPU integration validation

This independent check exercised activation reuse through the actual
`NativeStepAdapter.decode_step` graph using synthetic learned A1 and A8
quantizers. It compared two-step logits, intermediate state, K/V cache,
parameter gradients, leaf-feature gradients, and one manual SGD update between
the default adapter and the opt-in adapter. Both bit widths passed at
`atol=2e-6, rtol=5e-5`.

The test counted calls to the hard `learned_activation` computation (the
quantizer module's `forward` still runs once per projection). Over two decode
steps, QKV computations fell from 6 to 2 and gate/up computations from 4 to 2.
The attention-output, down, and head boundaries each remained at 2 computations
on both adapters; FC remained unused by this decode-only graph. Reuse diagnostics
reported the expected 2 QKV hits and 1 gate/up hit per scope, with zero retained
entries. A separate provider-round check confirmed that the default serial
learned-head path kept the same logits, raw-feature and head-quantizer gradients,
and three per-row head calls with reuse enabled. Default, explicit-off, and
enabled adapters also had identical state-dict keys and values.

An injected exception after the first QKV projection confirmed cached
`ActivationResult` objects became collectible during exceptional scope exit.
Weak references also cleared after each successful decode across two steps.
The focused suite passed 5/5 tests in 0.123 seconds. Ruff check, Ruff format
check, and `git diff --check` passed.

## Reproduction and environment

- Worktree: `/private/tmp/eagle-parallel-20261002/activation-reuse-feature`
- Branch: `research/20261002-activation-reuse-feature`
- Feature source commit: `c7b07d9` (`Add opt-in activation reuse`)
- Device: local Apple arm64 CPU; one Torch thread in each test; no GPU, Metal,
  SSH, model weights, captures, or datasets used.
- Runtime: macOS 27.0 arm64, Python 3.11.3, PyTorch 2.8.0.
- Before the run, the shared control file reported `research_stop=false`,
  `reset_observed=false`, reset `1791049896` unchanged, and 11% weekly allowance
  remaining.
- Run directory: `runs/parallel20261002/activation_reuse_feature/independent-validation-01/`
  (ignored by Git).
- Exact test command:

  ```sh
  PYTHONPATH=src:. python3 tests/test_parallel20261002_activation_reuse_feature_validation.py -v > runs/parallel20261002/activation_reuse_feature/independent-validation-01/final.log 2>&1
  ```

- Exact focused static checks:

  ```sh
  ruff check tests/test_parallel20261002_activation_reuse_feature_validation.py
  ruff format --check tests/test_parallel20261002_activation_reuse_feature_validation.py
  git diff --check
  ```

The test process exited normally. Instrumentation hooks and patches were removed
by test cleanup. The ignored run directory retains `final.log`; no process or
GPU allocation remains.

## Source and result hashes

| File | SHA-256 |
| --- | --- |
| `tests/test_parallel20261002_activation_reuse_feature_validation.py` | `4c9ae2fa88169890b8d7c43886da5e1125575e89d469efff011b6ef808081f4b` |
| `src/w1a1_eagle/activation_reuse.py` | `a3bce5484733420588b54d72f696a255d20928594324ff646a87177565102002` |
| `src/w1a1_eagle/native_step.py` | `5be7faa64ff49214c2ce44d5f47d9a0c933e412da1c0ac359fa944a4bba1c634` |
| `src/w1a1_eagle/learned_activation.py` | `458894b5bfa2f15c428d8d9b2c403a01c28ec902e3a95ff90488fd87b7181af9` |
| `runs/parallel20261002/activation_reuse_feature/independent-validation-01/final.log` | `ddf8671c00ef729754d6791c9eb5852c65c3c42ce70e06e7dbeca6b76d5f1962` |

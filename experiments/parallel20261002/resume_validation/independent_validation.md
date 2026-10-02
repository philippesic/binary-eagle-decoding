# Independent staged-resume validation

Date: 2026-10-02. Worktree: `/private/tmp/eagle-parallel-20261002/resume-validation`,
branch `research/20261002-resume-validation`. This validates the isolated
prototype in `research/parallel20261002/resume_validation/reference/staged_resume.py`;
it does not change production resume code.

## Setup and commands

The run used the existing deterministic tiny `make()` fixture from
`tests/test_qat_curriculum_runner.py`, with CPU only. The host was an Apple M3
Max running macOS 27.0 arm64, CPython 3.11.15, and PyTorch 2.14.0. CUDA was
unavailable. The fixture's original dense weights use FP16; trainable binary
parameters and optimizer moments use FP32. No model files, datasets, GPU,
remote host, or production update path were used.

Exact validation command, from the worktree root:

```sh
uv run --group w1a1 python research/parallel20261002/resume_validation/validation/independent_validation.py > runs/parallel20261002/resume-validation-adversarial/final.log 2>&1
```

Lint and format checks:

```sh
uv run --group dev ruff check research/parallel20261002/resume_validation/validation/independent_validation.py
uv run --group dev ruff format --check research/parallel20261002/resume_validation/validation/independent_validation.py
```

The first command passed all 4 tests in 0.792 seconds. Ruff reported all
checks passed, and the format check reported the file already formatted. Raw
output is retained at
`runs/parallel20261002/resume-validation-adversarial/final.log` (SHA256
`d65cf96499ec593f57d3c20dbae45135d36d203634ec1d2035097b307e65c472`).

## Coverage and findings

The malformed checkpoint suite has 11 optimizer-state adversarial cases:
reordered owned IDs, an unowned ID, wrong moment shape and dtype, nonfinite
moments, a negative second moment, fractional, negative, and nonscalar steps,
changed immutable Adam betas, and a warmup LR mismatch. Each is staged against a
fresh combined learned-activation, fusion-correction, and affine fixture. Every
case failed before changing model values, parameter identities or storage,
`requires_grad`/grad state, optimizer, curriculum counters, cursor, RNG,
checkpoint pointer, or runner timing fields.

Valid staging also left the live runner unchanged before `commit()`. Midphase
resume after update 1 and a phase-boundary resume after update 2 both continued
to the same final model and optimizer state as uninterrupted execution, with
identical RNG, cursor, and curriculum update/exposure ancestry. Timing fields
and checkpoint digests were excluded from cross-run equality because they
encode measured wall time and each source checkpoint has a different digest.

The suite additionally accepted a fresh phase with an empty optimizer state,
an AdamW parameter with no saved moments when gradients were absent, SGD with
zero momentum and no state, and SGD with momentum `0.8` and stored
`momentum_buffer` tensors. The SGD fixtures are built through the existing
`make()` fixture with a validated `BinaryOptimizationConfig` injected at
construction.

## Cleanup and limits

All fixture run directories were temporary and removed by the test teardown.
The ignored raw log remains in the run directory above, and `uv`'s ignored
worktree-local `.venv` remains available for repeat runs. The test process
exited; no persistent process or GPU allocation remains. These are tiny CPU
semantic checks and make no performance or GPU-readiness claim.

Control was reread before each execution chunk. At the final run it reported
84% weekly usage, `research_stop=false`, and the original reset timestamp
`1791049896` unchanged.

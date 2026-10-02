# Independent math inventory validation

## Purpose

This CPU-only check exercises the readiness consumer against authentic source
hashes from `training_runtime_identity("cpu")` and, for curriculum, the
authentic `curriculum_runtime("cpu")` inventory. CUDA receipt fields and the
hardware context are explicit test stubs; they represent no measured CUDA run.

## Environment and command

- Worktree: `/private/tmp/eagle-parallel-20261002/required_math_closure`
- Branch/revision: `research/20261002-required_math_closure` at `5a8cd77`
- Python 3.11.3, PyTorch 2.8.0, NumPy 1.26.4
- Host: macOS 27.0 arm64; `torch.cuda.is_available()` returned `False`
- Device use: CPU only; no Metal, model weights, datasets, or SSH
- Command: `PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_required_math_inventory_independent.py' -v`
- Result: 4 tests passed in 0.372 seconds
- Raw output: `runs/parallel20261002_required_math_closure_independent/independent_unittest.log` (ignored)

## Results

The complete synthetic receipt bound all authentic `MATH_FILES` hashes and was
accepted. A caller/receipt pair that mutually omitted either
`learned_activation.py` or `recurrent_qat.py` was rejected against the declared
producer inventory. A curriculum containing unsupported precision stage A2
was rejected. A continuous trainer resume with a changed runtime source map was
rejected before loading the checkpoint; the test compared every drafter
parameter and optimizer state before and after the failed resume and found no
changes.

The readiness receipt and CUDA hardware are schema fixtures only. These checks
establish source-inventory and fail-before-mutation contracts, not CUDA
performance or accelerator readiness. The earlier baseline probe reported by
the coordinating owner accepted mutually incomplete `learned_activation.py`
and `recurrent_qat.py` maps; the closure guard and this independent test now
reject them.

## Cleanup

The test's `TemporaryDirectory` fixtures were removed on completion. The raw
test output remains in the ignored run directory for audit. No model/data/run
artifacts were created elsewhere.

# Protected QAT LSQ CPU validation

Bounded synthetic CPU validation of the learned-head LSQ correction and the
protected regression adaptation. No GPU, Metal, SSH, model weights, captures,
or real training data were used. This is macOS ARM CPU evidence and says
nothing about SM75 latency or GPU throughput.

## Environment and provenance

- Correction worktree: `/private/tmp/eagle-learned-head-serial`, HEAD
  `85a5a7256e18b9bd8253db51254a953b61fa845d`.
- Protected regression worktree: `/private/tmp/eagle-qat-lsq-regression`, HEAD
  `af82c932327605c1adafad4bc76b5dcdb5211657`.
- Interpreter: `/Users/pippo/github/binary-eagle-decoding/.venv/bin/python`,
  Python 3.11.15, PyTorch 2.14.0, macOS 27.0 ARM64.
- CPU thread limits: `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `OPENBLAS_NUM_THREADS=1`, and `torch.set_num_threads(1)`.
- Raw logs and hash snapshots:
  `/Users/pippo/github/binary-eagle-decoding/runs/qat-lsq-correction-validation-20261002/`.

SHA-256 before validation:

| File | SHA-256 |
| --- | --- |
| `src/w1a1_eagle/recurrent_provider.py` | `36c7eda803baede517add00ced7715307035e989730f10edc5e8c98eb3aebd9a` |
| `scripts/check_qat_optimization_readiness.py` | `9f1dab1a0147ffe6e7ce7f35fde5f474482831b7121194a4c3677cd681e218ea` |
| `tests/test_learned_head_batching.py` | `37cfb7d300dda02bc558e9144d749d411aa4a03490760ed060862d0614aeb0fc` |
| `tests/test_parallel20261002_lsq_batching.py` | `35feea2f9cffb8f659ff16a94f9a48dc1f7121a805e1ad8ec4ac5f9082728049` |

SHA-256 after validation:

| File | SHA-256 |
| --- | --- |
| `src/w1a1_eagle/recurrent_provider.py` | `36c7eda803baede517add00ced7715307035e989730f10edc5e8c98eb3aebd9a` |
| `scripts/check_qat_optimization_readiness.py` | `9f1dab1a0147ffe6e7ce7f35fde5f474482831b7121194a4c3677cd681e218ea` |
| `tests/test_learned_head_batching.py` | `80f5283cb445271ff632fcd1751ebb2e2daf831b949e8d4af87eb924a832984b` |
| `tests/test_parallel20261002_lsq_batching.py` | `35feea2f9cffb8f659ff16a94f9a48dc1f7121a805e1ad8ec4ac5f9082728049` |

The provider, readiness implementation, and protected regression test hashes
were stable. The focused test file changed between its before/after snapshots,
so the focused run is provisional with respect to the final test source. No
repeat was made in this bounded pass. Re-run the focused set against the final
committed source before treating it as immutable acceptance evidence.

## Commands and results

An initial attempt to invoke `pytest` with the shared interpreter stopped before
test collection because `pytest` is not installed (`ModuleNotFoundError`). No
tests ran in that attempt. The successful correction command used standard
`unittest` discovery for the same six files:

```sh
cd /private/tmp/eagle-learned-head-serial
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/private/tmp/eagle-learned-head-serial/src:/private/tmp/eagle-learned-head-serial/tests \
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python -c 'import torch; torch.set_num_threads(1); import unittest; loader=unittest.TestLoader(); suite=unittest.TestSuite(); patterns=("test_learned_head_batching.py", "test_qat_cache_head.py", "test_learned_activation.py", "test_recurrent_provider.py", "test_qat_recipe_integration.py", "test_qat_readiness.py"); [suite.addTests(loader.discover("tests", pattern=p)) for p in patterns]; result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
```

Result: **72 tests passed in 1.733 seconds**. The learned-head cases exercised
A1/A4/A8 serial gradients and updates against the requested optimized path,
fixed/frozen/no-grad batching, visible wrapper behavior, and truthful requested
versus effective execution and saturation metadata. Existing cache/head,
learned activation, provider, recipe integration, and readiness tests passed.
Full output is in `focused-qat.log`.

The protected ten-test regression command was:

```sh
cd /private/tmp/eagle-qat-lsq-regression
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=/private/tmp/eagle-learned-head-serial/src:. \
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python -c 'import torch; torch.set_num_threads(1); import unittest; suite=unittest.defaultTestLoader.discover("tests", pattern="test_parallel20261002_lsq_batching.py"); result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
```

Result: **10 tests passed in 2.845 seconds**. This included the audit source
pin, serial parameter VJP and update checks, ragged valid-chain normalization,
terminal padding, live provider fallback, retained fixed/inference batching,
and observer delegation. The live-provider test checked A1/A4/A8 at depths 1,
2, and 4. Full output is in `protected-lsq-regression.log`.

## Cleanup

Both local test processes exited after their runs. No remote process, GPU
allocation, model-data temporary, or tracked run artifact was created. Raw
logs and hash snapshots remain in the ignored run directory named above. The
correction worktree still has the pre-existing pending edits to the provider
and readiness checker and the untracked focused test; this validation did not
modify or clean those files. No commit was made.

## Final combined acceptance run

The final immutable correction commit was cherry-picked after the protected
adaptation. The combined run used regression worktree HEAD
`7d0a9689df129a8b2cdf2ea7456a3827891419b8`, shared Python 3.11.15 / PyTorch
2.14.0, macOS ARM64, and one CPU thread. The only worktree change before or
after the run was this validation document; `git diff` showed no core or test
file changes during the run.

The exact combined command was:

```sh
cd /private/tmp/eagle-qat-lsq-regression
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=src:.:tests \
/Users/pippo/github/binary-eagle-decoding/.venv/bin/python -c 'import torch; torch.set_num_threads(1); import unittest; loader=unittest.TestLoader(); suite=unittest.TestSuite(); patterns=("test_learned_head_batching.py", "test_qat_cache_head.py", "test_learned_activation.py", "test_recurrent_provider.py", "test_qat_recipe_integration.py", "test_qat_readiness.py", "test_parallel20261002_lsq_batching.py"); [suite.addTests(loader.discover("tests", pattern=p)) for p in patterns]; result=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not result.wasSuccessful())'
```

**Result: 82 tests passed in 3.854 seconds, exit code 0.** The run covered the
six correction suites and all ten protected regression tests in a single
process, including learned-head A1/A4/A8 gradient and update equivalence,
fixed/frozen/no-grad batching, readiness metadata, ragged-chain normalization,
and observer/provider fallback behavior.

Before and after hashes matched exactly:

| File | SHA-256 before and after |
| --- | --- |
| `src/w1a1_eagle/recurrent_provider.py` | `36c7eda803baede517add00ced7715307035e989730f10edc5e8c98eb3aebd9a` |
| `scripts/check_qat_optimization_readiness.py` | `9f1dab1a0147ffe6e7ce7f35fde5f474482831b7121194a4c3677cd681e218ea` |
| `tests/test_learned_head_batching.py` | `66975d3a736457f95254acffef705e5ffce34a7d14b2ea2e277d60be0f405256` |
| `tests/test_parallel20261002_lsq_batching.py` | `35feea2f9cffb8f659ff16a94f9a48dc1f7121a805e1ad8ec4ac5f9082728049` |

This final run supersedes the earlier provisional focused result. The complete
output and before/after hash snapshots are retained in
`/Users/pippo/github/binary-eagle-decoding/runs/qat-lsq-correction-validation-20261002/final-combined.log`,
`final-combined-hashes-before.txt`, and `final-combined-hashes-after.txt`.
The test process exited; no GPU, remote process, model data, or temporary
artifact was created. No source or test files were modified and no commit was
made.

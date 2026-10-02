# Activation identity closure: independent CPU validator

## Scope

This validator checks that `activation_reuse.py` remains part of the runtime
identity with activation reuse disabled, that readiness rejects a matching
receipt whose partial math-source map omits the helper, and that a changed
helper/runtime identity rejects resume before checkpoint training state is
restored. The readiness receipts and runner use synthetic CPU fixtures only.

## Run record

- Worktree: `/private/tmp/eagle-parallel-20261002/activation-identity`
- Branch/base: `research/20261002-activation-identity`, based on `f027609`
- Run directory: `runs/parallel20261002/activation_identity/independent-cpu-01/`
- Environment: macOS 27 arm64; Python 3.11.15; PyTorch 2.14.0 CPU; NumPy 2.4.6; 10 Torch threads.
- Command:

  ```sh
  PYTHONPATH=src:tests:. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python tests/test_parallel20261002_activation_identity_independent.py -v > runs/parallel20261002/activation_identity/independent-cpu-01/final.log 2>&1
  ```

- Result: exit 0; 5/5 tests passed in 0.445 seconds.
- Checks: `ruff check tests/test_parallel20261002_activation_identity_independent.py` and `ruff format --check tests/test_parallel20261002_activation_identity_independent.py` passed.
- Raw log SHA-256: `6aac5942de82ec60222fcdb3d0f3237b40026d9553a31df3236daebc13bc2621`.
- Test source SHA-256: `7c63488b93ca02e8994e259bdac3014e14d4acfa224f4d48491fea5420ccb485`.
- Cleanup: temporary source copies and fixture directories were removed by `TemporaryDirectory` cleanup. The raw log is retained in the ignored run directory. No persistent process or device allocation remains; no GPU, Metal, SSH, model, dataset, or capture resources were used.

## Cases

1. The runtime hash map contains the helper and changes when helper bytes change, without an activation-reuse option being enabled.
2. A complete synthetic receipt carrying the helper hash validates.
3. A receipt and current runtime map that both omit the helper are rejected.
4. A malformed helper digest is rejected.
5. A changed helper hash in the resumed runtime contract fails before model parameters, optimizer state, RNG state, phase, or update count are restored.

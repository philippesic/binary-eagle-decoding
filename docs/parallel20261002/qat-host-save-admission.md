# QAT checkpoint host-save admission correction

Protected QAT owner assignment: `src/w1a1_eagle/qat_curriculum_runner.py` save
admission and focused tests only. Current QAT owner's separate worker retains
development evaluator ownership. Frozen preparation, GPU ownership, mathematical
recipe, `_cpu_tree`, resume and moment-probe behavior are unchanged.

The previous CUDA save gate charged retained CPU tensor copies plus 16 MiB.
`detach().cpu().clone()` also keeps the transferred CPU tensor alive while its
retained clone is allocated. The correction adds the largest CUDA tensor in the
actual payload. Tensor aliases are counted per occurrence, matching the existing
recursive clone; CPU inputs receive no transfer charge. The existing host floor
and 16 MiB workspace allowance are preserved. This bounds the identified tensor
overlap; serialization, allocator and other workspace behavior remain unresolved.

For the configured fully populated fixed Adam case, metadata-only tensors reproduce
2,619,863,112 retained bytes, largest transfer 327,680,000 bytes, and ordered peak
2,947,020,836 bytes. The previous charge undercounted that peak by 310,380,508 bytes
(296.002 MiB). The corrected tensor/workspace charge is 2,964,320,328 bytes,
separate from the unchanged host floor. No configured-size backing tensor or
actual CUDA transfer was allocated for this check.

Acceptance: six new tests cover nested aliases/mixed dtypes, empty metadata,
actual tiny CPU clone storage independence, configured Adam event order, refusal
at the former boundary before copy/write/publication, and admission at the exact
new boundary with floor/workspace unchanged. Together with existing curriculum
runner, curriculum readiness and continuous-resource tests, **42/42 pass** on
macOS ARM64 / Python 3.11.15 / PyTorch 2.14.0, one CPU thread, with accelerator
discovery explicitly forbidden. Ruff and whitespace checks pass.

```sh
PYTHONPATH=src:tests OMP_NUM_THREADS=1 \
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest discover \
  -s tests -p test_qat_checkpoint_host_admission.py -v
```

Worktree `/private/tmp/eagle-qat-host-save-admission`, branch
`feature/qat-host-save-admission`, current-main base `b858d33`. Exact final source
and test hashes and independent checks are recorded with the reviewed commit.
The runner is already bound in its runtime `EXTRA_MATH` source identity, so this
source change requires new current receipts rather than reuse of old identities.

Independent Luna acceptance at immutable `53c2eaa`: **42/42 tests passed in
1.457 seconds**, with the runner and all four participating test-file hashes
unchanged before and after, accelerator discovery forbidden, no source/test
diff, and the process exited. Its read-only review confirms the clone-per-
occurrence/one-transfer bound and unchanged copy/resume/math/probe behavior.
Exact source hashes, command, environment and raw paths are in
[the independent validation record](qat-host-save-validation.md).

Remaining: QAT owner review, sole-root integration/push and clean worktree
retirement. Actual source-bound host available/RSS and device allocated/reserved
measurements for save and resume remain mandatory before launch admission. This
fix establishes no full-model CUDA fit, SM75 performance, convergence or Q4_0 gain.
The separate resume loaded-payload and moment-probe peaks are not changed here.

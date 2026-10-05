# Nine model QAT preparation QA ledger

This is the independent QA record for the nine model preparation branch. It
tracks portable evidence separately from actual-model and target-GPU admission.
Fixtures, mocks and CPU/MPS runs never establish native quality, CUDA behavior,
RTX2080Ti readiness or RTX5080/SM120 readiness.

## Campaign profiles

`PENDING` means the required independent evidence has not been recorded here.
Use `PASS` only with a command, raw evidence path, SHA256, source revision and
actual hardware. Use `EXCLUDED` only for a deliberately unsupported profile or
option, with its scope stated. The final nine model set is three original frozen
Q4 controls plus W1A8 and W1A1 candidates for EAGLE, DSpark and DFlash.

| Profile / family | Q4 control | W1A8 | W1A1 | Current disposition |
|---|---|---|---|---|
| EAGLE | PENDING original artifact identity and matched control path | PENDING fixed-reference profile | PENDING fixed-reference profile | No new-profile evidence yet |
| DSpark | PENDING original artifact identity and matched control path | PENDING block QAT/export/native graph | PENDING block QAT/export/native graph | No new-profile evidence yet |
| DFlash | PENDING original artifact identity and matched control path | PENDING block QAT/export/native graph | PENDING block QAT/export/native graph | No new-profile evidence yet |
| Direct A1 profile | n/a | EXCLUDED | PENDING explicit profile/API and independent behavior proof | Supported only if implementation exposes it explicitly |
| A8-to-A1 reset profile | n/a | PENDING A8 state | PENDING exact reset/retention transition proof | Supported only if optimizer/RNG/cursor transition is exact |

## Requirement gates

| Gate | Status | Evidence required for PASS |
|---|---|---|
| Declared config/profile parse and use real source APIs | PENDING | Independent calls with exact config/path types; invalid schema and shape cases reject |
| DSpark/DFlash W1A8/W1A1 pack, load, graph and no-dense-fallback | PENDING | Actual loader/graph APIs, malformed payload rejection, native dispatch evidence; CPU/SM75 and SM120 remain distinct |
| Five-tap/full-vocabulary data ancestry, masks, prefixes and cursor resume | PENDING | Pinned manifest + producer receipt + TRAIN membership joins; bounded full-vocabulary teacher path; independent cursor/mask checks |
| EAGLE/block hard-forward QAT conditioning, parameter ownership and gradients | PENDING | Independent forward/backward tests on actual implementation, including later state/K/V and frozen target storage |
| A8/A1 calibrated fusion arithmetic and diagnostics | PENDING | Independent A8/A1 arithmetic oracle, scale-only comparison, negative zero-scale row rescue and exact finite exported objective checks |
| Checkpoint, optimizer, RNG, cursor and A8-to-A1 exact resume | PENDING | Genuine save/load continuation equivalence; failure on altered config/artifacts; actual model remains a separate gate |
| Export and evaluator sequence | PENDING | Success and failed-export paths, committed checkpoint retention, fresh evaluator process, matched report and target-only diagnostic |
| STOP, wall-cap, process-group and resource cleanup | PENDING | Success, STOP and timeout/failure exercises; cleanup receipt plus independent group/context/resource refusal tests |
| Portable CPU source/build/lint/test suites | PENDING | Applicable suite results on integrated source, lint/format for all changed files, successful CPU build |
| RTX2080Ti/SM75 checks | PENDING | Human opened host, sole operator verifies current hardware; no long QAT or quality/performance runs |
| RTX5080/SM120 admission | PENDING | Human availability plus fresh minimal resource, kernel/model/backward/memory/capture-portability gates |

## Baseline evidence

The baseline was collected from handoff commit `da097e5` in this worktree. The
machine was a MacBook Pro Mac15,10 with Apple M3 Max, 36 GB RAM, Python 3.11.15,
PyTorch 2.14.0, MPS available and CUDA unavailable. These results are local
source/test evidence only.

The full suite command used the established root virtualenv and root's existing
llama.cpp `gguf-py` source tree without modifying that environment:

```sh
PYTHONPATH="/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py:$PWD/scripts:$PWD/src" \
  /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  -m unittest discover -s tests
```

It ran 1,265 tests in 182.006 seconds, with 12 errors and 5 skipped. Each error
was from the pre-existing local research stop controls (`Research stopped by
supervisor control` or the research stop latch); no shared pause or stop state
was changed. Raw output is ignored at
`runs/nine-model-qa-baseline/unittest-root-venv.txt`, SHA256
`c9f3f8205d315e7253d1efd982449eed62be8bafb3687448be268f386efa69be`.

Other baseline commands and results:

| Command | Result | Raw evidence SHA256 |
|---|---|---|
| `make check` | Stops at Ruff: 1,183 lint findings. The follow-on format check independently finds 73 files unformatted. | `ruff-check.txt`: `c5f5d35a5682e0d613bcec7c7101b014423859cbcb44511b6f5f05d4f6c499f6`; `ruff-format.txt`: `65c9781c3ef91046ac6e590c9930b21ce5f08dff62035d0c3beec67d003a27ad` |
| TOML parse command from `Makefile` | PASS | Empty output SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `git diff --check` | PASS | Empty output SHA256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `make build-cpu` | BLOCKED before configure: `third_party/llama.cpp` is not initialized in this worktree; command requests `make setup`. No build was claimed. | `build-cpu.txt`: `16839fa5cbbdbea551261819aac1c3027d234239015e4dbf805ecbc0826449df` |

All raw baseline files are kept under ignored `runs/nine-model-qa-baseline/`.
The first fresh-venv suite attempt (not used as the normalized baseline) is
also preserved there as `unittest.txt`; missing YAML/GGUF/target dependencies
caused import errors. The normalized run above supplies the useful source
baseline while honoring the existing stopped-research controls.

## Per-revision independent evidence

Add one dated section per integrated feature/review revision. Include the exact
commands, environment, run directory, raw-output hashes, result, reviewer
findings and cleanup. Preserve author tests as author evidence; list independent
QA separately. Any remaining target-hardware or production artifact dependency
stays `PENDING` until its own evidence exists.

## October 4 independent feature evidence

These runs used the repository's established `/Users/pippo/github/binary-eagle-decoding/.venv/bin/python` (Python 3.11.15, PyTorch 2.14.0) and the current feature worktrees. The operator did not query or control either WSL GPU. Local hardware remained Apple M3 Max CPU/MPS with CUDA unavailable. Raw command output is in ignored `runs/nine-model-qa-baseline/`.

Latest focused suites at this checkpoint:

| Scope | Command / revision | Result | Evidence |
|---|---|---|---|
| Native block exporter, C++ loader/graph, teacher replay | `BLOCK_TEST_NATIVE=/private/tmp/nine-model-qat-20261004/native-build/bin/test-block-binary BLOCK_TEACHER_NATIVE=/private/tmp/nine-model-qat-20261004/native-build/bin/llama-block-teacher python -m unittest discover -s tests -p 'test_block_*.py' -v`, native parent `3d2b1f2` | 12/12 PASS; CPU fixture GGUFs only | `native-cpu-tests-final.txt`, SHA256 `c7554674bafc0b52dc5a1afa66d35c06184b94550b0442a8f2b9fd75ec748585` |
| Native CPU build | `cmake --build /private/tmp/nine-model-qat-20261004/native-build --target test-block-binary llama-block-teacher -j2`, native parent `3d2b1f2`, llama.cpp fork revision from that worktree | PASS; ARM CPU/Accelerate build, no CUDA | `native-cpu-build-final.txt`, SHA256 `49e5fe1dd37a47d4422f1a697824f1e5f88795048cb436654a8636b666c293e7` |
| Block data, EAGLE three-tap golden contract, native raw import, fusion arithmetic | `PYTHONPATH=src python -m unittest discover -s tests -p 'test_block_*.py' -v`, data/fusion `7c6b6b1` | 43/43 PASS; synthetic capture/receipts | `data-author-current.txt`, SHA256 `c81c00714facf2f965b170f60026f2d419aa6b1352477c469b20a778ef4343d2` |
| TRAIN capture planner/executor | `PYTHONPATH=scripts:src python -m unittest discover -s tests -p test_nine_model_train_capture.py -v`, capture `a1917d9` | 19/19 PASS; synthetic native producer, no CUDA | `capture-author-final.txt`, SHA256 `816cb8ca78724c11bfda83b1e7b1039055f477789e0867dd8a9a91dcba888dbf` |
| QAT block profiles, ownership, optimizer, checkpoint, transition, initialization | `PYTHONPATH=src:scripts python -m unittest discover -s tests -p test_nine_model_training.py -v`, training `4697256` | 21/21 PASS on CPU fixtures | `training-author-current.txt`, SHA256 `b49c5fa8ed9203637dd44a102f7ccdaa682c17b5b6971ae105456e9bae14c045` |
| Campaign, bundle builder, report, training lifecycle fixtures | `PYTHONPATH=src:scripts:<training src/scripts>:<data src> python -m unittest discover -s tests -p 'test_nine_model_*.py' -v`, pipeline `7e3f5be` | 56 tests PASS, 2 Linux `/proc` tests skipped on macOS | `pipeline-author-current.txt`, SHA256 `e6a638e9f4b8c33fb8195cc44baa75c5ac724ad9a11bad5ba380844ebbeaec83` |
| SM120 admission refusal/fixture receipts | `PYTHONPATH=src:<pipeline src> python -m unittest discover -s tests -p test_nine_model_sm120_admission.py -v`, coordination `ca2b1c5` plus working-tree gate | 4/4 PASS with test doubles; no device query | `sm120-admission-author-cpu-fixtures.txt`, SHA256 `2b96a8a5e2d33553457306835b6d5d4f322beff6b04f7b73729a6973cd625032` |
| Independent block/data/fusion/loader/init/resume/pipeline contracts | CPU suite `test_nine_model_*.py`, including actual saved EAGLE A8/A1 fit NPZs | Current attempt has 1 failure in the source-GGUF loader contract (details below); all other 21 executed tests pass and one Linux-only process test is skipped | `independent-nine-model-current.txt`, raw failure preserved |

Independent integration checks also verified all four saved EAGLE reference-magnitude initializer NPZs against the fixed-half contract. The check loaded the actual A8/A1 candidate artifacts, compared their hard sign planes and scale arrays with the original saved fits, ran the adapter, and checked that the sparse FC initializer leaves fifteen unrelated binary linears unchanged. The historical unit-magnitude NPZ requires an explicitly selected `unit_probe`. Two tests passed on the M3 Max CPU: `independent-calibration-artifact-tests.txt`, SHA256 `a95331fd86a7c920d6764cdcd2bdb9a524c09a622d455417ae62db5abd783a65`.

The source-GGUF loader test found a F32 epsilon comparison defect; training commit `452f29a` canonicalized the metadata before exact comparison. The latest independent loader tests now pass for both DSpark and DFlash, as recorded in the October 4 checkpoint below.

The EAGLE fusion report itself is only a calibration initializer: its original target capture is historically reported as RTX5080, while the calibration ran on the M3 Max CPU, with 256 fit rows and 128 prose/reasoning validation rows. Its raw validation fit improves A8/A1 MSE, but post-norm error worsens for both candidates; it proves neither native trajectory nor quality. The fixed-half repack changes latent magnitudes to the known EAGLE reference while retaining the original fitted hard bits and scales. The paired audit is `eagle-fusion-independent-audit.json`; original report SHA256 `4002b8f9b1b514a3b55e2c57a1c703628ff7d06cb1ad331000d19deb3e425bc7`, fixed-half contract JSON SHA256 `c4cd37492b8056597f1eee3889a1ab4bd2d88a4329a4e08d3a25134ac6a1deda`.

Selected-file Ruff checks pass for the QA files; the whole-repository baseline debt remains 1,183 Ruff findings and 73 unformatted files. Feature owners ran their own selected checks. The independent QA checks are recorded under `final-ruff.txt` and `final-format.txt`. No broad formatting changes were applied.

The QA branch adds [independent contract tests](../../tests/test_nine_model_contracts.py), [failure-path tests](../../tests/test_nine_model_failure_paths.py), and [actual initializer artifact tests](../../tests/test_nine_model_calibration_artifact.py). It has exercised CPU success/failure/STOP/resume/export/evaluation chaining and local child-group cleanup. One process-leader/descendant test is intentionally Linux-only because actual `/proc`, `nvidia-smi` and `/dev/dxg` release evidence is a WSL/Linux gate. Portable source readiness and fresh SM120 admission remain separate; all nine production profile statuses and the actual-model native/resource gates are still `PENDING`.

## Latest independent QA checkpoint — October 4

This section supersedes the earlier transitional test notes above. The normalized source GGUF epsilon issue recorded in that checkpoint was fixed by training commit `452f29a`; the latest independent loader checks pass for both DSpark and DFlash. The latest training source is `8a16cea`. The pipeline tests use source commit `86cec63`; the campaign/builder worktree also had follow-up edits, so the recorded SHA pins exact source bytes and must be refreshed after integration.

The final independently authored QA suite has 25 tests: 14 independent data/fusion/training/source-loader/profile tests pass; 8 independent lifecycle, STOP, resource-return, persistent-authorization, export/evaluation and diagnostic-sampler checks pass; two pinned EAGLE calibration-artifact adapter checks pass; one Linux-only process-identity test is skipped on macOS. Exact raw logs and hashes are listed in `runs/nine-model-qa-baseline/qa-independent-final.txt` and `qa-ledger.json`. All tests were CPU-only on Apple M3 Max using the established root Python 3.11.15 / PyTorch 2.14.0 environment. The ignored overlay pinned imports to the exact isolated feature worktrees. QA-owned test Ruff and format checks pass.

Latest author suites independently reviewed: training 25/25 pass; pipeline lifecycle/report suite 32 pass and 2 Linux-only skips; the latest bundle-builder suite 8/8 pass; data/fusion 43/43 pass; TRAIN capture 19/19 pass on synthetic producer fixtures; native CPU fixture tests 12/12 pass with the native CPU build; SM120 admission refusal fixtures 4/4 pass without querying a device. These local checks establish portable implementation behavior only. They are not real-model or production capture evidence.

The pipeline diagnostic memory sampler now publishes sample-start cadence, longest observer duration and whether the bounded sample count was reached. RSS is labeled as sampled evaluator/descendant process maxima with overlap explicit and is not summed. Independent deterministic-clock checks pass. Lifecycle validation now rejects a zero-update training endpoint before export/evaluation; QA's local synthetic stage receipt includes a positive update count and passes the complete success chain. A8-to-A1 launcher configs require positive final-A1 update/token quotas and a budget beyond the warm phase; independent invalid-config checks pass.

All nine aggregate profile statuses remain `PENDING`. The remaining production evidence includes original Q4 pins, resolved candidate configs and human-selected budgets, complete production TRAIN data/admissions, trained checkpoints, final exports, and matched native evaluation artifacts. Human direction on October 4 was “pause all 5080 usage continue mac only.” QA did not query or control the RTX5080; fresh SM120 admission remains `PENDING` until the pause is lifted and a new receipt is produced. RTX2080Ti was also not queried in this QA pass.

A local Linux-container run was also checked as an optional way to exercise the process-identity tests without WSL/GPU access. Docker CLI is installed but its selected local `desktop-linux` Unix socket is absent; Podman is not installed. QA did not start or install a runtime. The Linux-only process cleanup checks remain skipped/pending; discovery evidence is `runs/nine-model-qa-baseline/local-container-runtime-check.txt`, SHA256 `e80ebfdb510f92a0a1a2071e806af9bf54c307918baa9b81af4ad07a6b743530`.

# QAT readiness validation plan and CPU baseline

Date: 2026-10-01  
Checkout: `main` at `f502d4a10fc3594e485e5893b0682bb3d7bb1bfc`  
Purpose: record existing CPU validation and define bounded checks needed before real-data QAT. This is planning and baseline evidence only; no implementation, real-data optimizer update, GPU run, SSH, or source/config mutation was performed.

## Baseline run

Environment: macOS 27, Apple M3 Max (arm64); Python 3.11.15; PyTorch 2.14.0 built without CUDA (`torch.version.cuda is None`); NumPy 2.4.6. `CUDA_VISIBLE_DEVICES=''` was set. Existing tests in continuous readiness/runtime explicitly replace `torch.cuda.is_available` with an assertion failure around the guarded operations. The rest of the selected suite uses CPU fixtures and no CUDA-enabled PyTorch build was available. This result validates CPU contracts only; it gives no SM75/RTX2080Ti or RTX5080 performance evidence.

Run directory: ignored `runs/qat-optimization-readiness/cpu-baseline/`. Raw outputs are retained there as `unittest-corrected.log`, `unittest.log`, and `pytest.log`.

Successful command, from the repository root:

```sh
PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest \
  test_recurrent_qat test_qat_head test_qat_head_export test_qat_head_trainer \
  test_native_step test_continuous_qat test_continuous_readiness \
  test_continuous_runtime test_continuous_resources test_continuous_launcher \
  test_w1ax_continuous_config test_w1ax_continuous_stages test_recurrent_provider \
  > runs/qat-optimization-readiness/cpu-baseline/unittest-corrected.log 2>&1
```

Result: exit 0, 114 tests passed in 2.582 seconds. The temporary directories created by the unit tests were cleaned by their `TemporaryDirectory` fixtures. The ignored run directory and logs remain as raw evidence.

Two setup attempts are also preserved. `.venv/bin/python -m pytest ...` exited 1 because pytest is absent from the pinned environment; no package was installed. The initial unittest invocation used path-style test names without `PYTHONPATH=src:tests`, so two modules failed imports while 86 other tests ran. The corrected command above passed the full selected suite.

## Existing coverage and gaps

The current selected suite checks CPU hard quantization, row binary linear behavior, QAT training arithmetic, export fixtures, native-step structure, recurrent gradients, paired continuous training/checkpoint behavior, readiness report validation, resource estimates, and launcher/config/stage contracts. The readiness and runtime fixtures explicitly fail if `torch.cuda.is_available()` is queried in those CPU paths. Their reports are synthetic fixtures; they cannot qualify a new learned parameter for native deployment.

`src/w1a1_eagle/continuous_runtime.py` defines `MATH_FILES` as an explicit list of eight modules: `recurrent_qat.py`, `recurrent_binary.py`, `native_step.py`, `recurrent_rollout.py`, `recurrent_loss.py`, `recurrent_trace.py`, `recurrent_provider.py`, and `continuous_qat.py`. Runtime identity hashes only those modules, Python/Torch/NumPy versions, and device type; it records CUDA math settings only on an explicit CUDA path. Any new optimizer, quantizer, curriculum, or batching module that can affect math or checkpoint interpretation must be added to the source identity inventory and tested by changing a source fixture and observing a resume rejection.

Continuous checkpoints save each lane's linears and optimizer state, plus paired RNG/cursor metadata. Before enabling learned A4/A8 clipping or A1 thresholds, verify that every such trainable value is part of the model state, optimizer ownership is exact, checkpoint save/load restores it exactly, and exports either preserve it in a native-supported form or reject deployment. A wrapper-only/config value is not sufficient evidence of a learned parameter.

The current `pyproject.toml` sets `[tool.uv] package = false`, and the selected checks import directly from `src` via `PYTHONPATH`; there is no package build/install validation. When the implementation introduces files or command entry points, add an isolated clean-tree/import smoke check that proves the run packet contains all required files and does not rely on a developer checkout's ambient `PYTHONPATH`.

The runbook at `docs/CONTINUOUS_W1AX_RUNBOOK.md` requires an exact resume to use a new supervisor ID and identical source/config/runtime; changed math identity rejects resume. Python processes use imported code loaded at startup, so edits on disk do not update the already-running process. Treat the existing preparation process/run as frozen and do not hot-reload new source into it. New implementation requires a fresh, explicitly authorized launch after the current owner releases the host, with its own run directory, config snapshot, runtime identity, and supervisor.

## Required checks before real-data QAT

1. **CPU integration:** extend the selected unittest suite for duplicate A1 forward elimination and custom backward; native sign/scale order including zero and subnormal cases; recurrent state and K/V gradients; chunked K/V-only context and per-chain batching; exact token, position, feature, mask, cache-cast, and detach ancestry; optimizer ownership, selectable initialization/STE/LR behavior, and exact checkpoint resume; trainable A4/A8 clipping and A1 thresholds; bounded curriculum transfer and refreshed-trajectory ancestry. CPU checks establish structure only.
2. **Config and identity:** test default and explicit experimental configs round-trip with all new fields, validate unsupported combinations fail closed, and confirm each new math/checkpoint-format module appears in the runtime hash inventory. Mutating a listed source/config/runtime must reject exact resume. Confirm manifests name every learned parameter and record shape/type/value hashes or checkpoint hash, optimizer ownership, recipe, data ancestry, code revision, and library versions.
3. **Export and native deployment:** exercise checkpoint-to-export serialization and readback for every learned parameter. Native decision tests must show that a trained non-default clipping/threshold state reaches the native computation and changes the decision when expected. If native kernels cannot represent the value, assert an explicit deployment error; never accept a simulation-only result as native-ready. Cache and mask tests must preserve exact prefix ancestry after chunking/batching.
4. **CUDA forward/backward with zero real updates:** after explicit RTX5080 ownership and idle proof, use only a tiny synthetic fixture and the supervised project runner. Run every enabled recipe through full model-shaped forward/backward (including cache rebuild and all active learned quantizers), check finite losses/gradients and nonzero gradients for each intended learned parameter, then assert optimizer/global counters remain zero and weights/optimizer step counters did not advance. Save/reload the step-zero checkpoint and verify hashes and native export/decision gates. This does not authorize any real-data optimizer step.
5. **Batch/profile and memory admission:** compare the current default and candidate larger microbatches on identical synthetic shapes and hardware, with synchronized timing and peak allocated/reserved memory recorded per lane and for the paired workload. Preserve the current cadence unless an explicitly reviewed experiment changes it. Require measured peak reserved CUDA memory at or below the existing 12 GiB ceiling, at least 1 GiB free at admission, and current host/disk floors (2 GiB host floor and the existing additional 12 GiB pre-load admission) before enabling a larger batch. Include allocated/reserved peaks, step timings, shape/config hashes, GPU model/driver/runtime, and precision. CPU arithmetic estimates do not satisfy this gate.
6. **Held-out/native acceptance boundary:** keep the target/verifier precision and sealed sets frozen. Before claiming acceptance, run the bounded native trajectory/decision check on allowed development prompts against Q4_0 EAGLE as the primary comparison. Report FP16 only as a secondary diagnostic. Throughput/quality benefits remain unproven by a synthetic zero-update smoke.

Exact CPU rerun command is recorded above. Future CUDA work has no safe command to record yet: entry points and files are under active implementation, the RTX5080 owner is active, and RTX2080Ti is paused. Once unblocked, use `docs/AGENT_OPERATIONS.md` and `docs/CONTINUOUS_W1AX_RUNBOOK.md`: host registration only, tmux MCP for SSH, a unique remote run directory, and `scripts/remote_job.py` inside a named host-side tmux session. Stop and verify the supervisor process group and GPU memory before reporting completion. Raw output stays under ignored `runs/qat-optimization-readiness/<run-id>/`.

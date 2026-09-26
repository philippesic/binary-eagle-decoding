# Goal: all-layer W1Ax activation-precision suite on RTX 2080 Ti

**Opened:** 2026-09-25
**State:** active; implementation in progress
**GPU owner:** orchestrator in this task; no remote job started yet

## Objective

Implement, validate, and run the full [W1Ax activation-precision study](../../experiments/w1ax-activation-precision-plan.md) on the RTX 2080 Ti. Compare W1A16, W1A8, W1A4, and W1A1 using identical one-bit weights across all nine drafter linears, with target-only, FP16, Q8_0, and Q4_0 anchors. Separate correctness and dispatch, acceptance, identical-input operator cost, full EAGLE round cost, and serving throughput. Preserve reproducible raw artifacts and a compact report.

## Checkpoint

- Native Codex Goal opened in this task on 2026-09-25.
- The frozen study protocol already exists. No new W1A16/W1A8/W1A4 artifact or measurement exists yet.
- Local `main` was clean at goal start and the goal checkpoint was pushed as `7ae1cc2`. A managed worktree at `/Users/pippo/.codex/worktrees/w1ax-suite/binary-eagle-decoding` holds the `w1ax-suite` branch.
- Shared host registry points to `192.168.4.29`. The first SSH attempt through tmux MCP timed out during banner exchange; the user's requested retry connected. The actual RTX 2080 Ti (SM75) was idle at 0% with 672 MiB used and 10,356 MiB free, and the project process scan found no active supervisor/server/benchmark. Remote project parent was `11a25f5` and its llama.cpp checkout `34e21b7`. No new run has started.
- Bounded workers own native W1Ax runtime, eight-path benchmark runner, and opt-in round tracing in separate files. The orchestrator owns GPU operations and integration. Workers are not to commit or use the GPU.
- Keep the RTX 2080 Ti to one supervised experiment at a time. Use `scripts/remote_job.py` and unique run directories. Do not use the 24 QAT-final prompts in this untrained screen.
- A supervised preflight on the actual RTX 2080 Ti (`runs/w1ax-preflight-w1a1-20260925/`) passed the existing W1A1 CUDA backend gate 5/5, including K=31/32/33/2560 and strided K=33. The supervisor ended with exit 0 and the GPU returned to 0% / 672 MiB used.
- The separate nine-prompt context diagnostic was generated under remote `runs/w1ax-context-20260925/` from committed script `e2ad359`; manifest SHA256 `5653cfe7599e5dd4ae44e057df27b816221f8ee89635056bc0fb24b9f44a21a3`. Qwen3-4B chat-template tokenization with thinking disabled passed all planned bins: short 195–207, medium 521–543, long 1131–1171 tokens. Audit JSON SHA256 `ff23798a72f3a782821008fba4ff43dcbe8a9e86e307fd77aa313b64585e4173`; supervised audit `runs/w1ax-context-audit-final-20260925/` exited 0. The audit script is committed as `69d29d4`. These prompts are a context/latency diagnostic, separate from the QAT development and final sets.
- Frozen QAT-revisit manifests were generated in remote `data/qat-revisit/` by supervised `runs/w1ax-qat-prompts-20260925/` (exit 0). The 24-prompt development manifest SHA256 is `a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885` (8 prose, 8 code, 8 reasoning). The 24-prompt final manifest was generated and sealed; it has not been used in this study.
- The reused all-nine W1A1 GGUF SHA256 is `098e1ecbb299aa16e2c968663acc49e60c0fcf16b053766d9f558114f79d011c`, matching the prior all-row source audit. The fixed FP16 target GGUF SHA256 is `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`.
- Native W1Ax runtime commits `45b76c033`, `4c8767e4e`, `73caaa8ab`, and `3792aa79c` are published on the user's llama.cpp fork branch `w1ax-suite`. Local CPU operator gate passed 88/88; Mac `llama-server` with round tracing compiled. Parent runner, replay harness, and diagnostic analysis commits are on the parent `w1ax-suite` branch. The latest parent gitlink will be updated to `3792aa79c` before a sealed GPU run.
- Remote CUDA source worktree `runs/llama-w1ax-src/` was created from published commit `4c8767e4e`; supervised CUDA build `runs/w1ax-sm75-build-20260925/` is in progress at `build/llama-cuda-w1ax/`. Remote parent runner worktree `runs/w1ax-project-src/` is at parent `b16d72e` with its published submodule gitlink `73caaa8ab` checked out. Its model/build links and resolved historical/development configs are under ignored run paths. Update both source and runner worktrees to their final published commits after the initial compile and before measurement. Only this orchestrator uses the GPU.

## Next actions

1. Verify host and GPU state through tmux MCP, plus local and remote Git/build state.
2. Implement and audit common W1Ax weight/export and numeric contracts; pass CPU/CUDA reference gates and real SM75 dispatch checks.
3. Run the frozen full matrix, operator replay, round tracing, and context/policy diagnostics; analyze both FP16 and Q4_0 comparisons.
4. Write a compact report with raw artifact hashes, update this checkpoint and status, integrate tested code into `main`, and push.

## Native runtime worker checkpoint

- Extended the existing packed-weight GGML operator with activation modes 1, 4, 8, and 16 selected by `GGML_W1AX_ACT_BITS` at EAGLE graph build. Default 1 preserves the old W1A1 path. All modes load the same nine packed signs and F32 row scales, with no dense selected-weight shadow.
- A16 casts each F32 activation to FP16 at the operator boundary, accumulates signed values in F32, then multiplies the row scale. A8/A4 use per-token F32 absmax scales, signed ranges ±127/±7, nearest-even rounding, exact integer dot accumulation, then row-scale and activation-scale multiplication. CUDA A4 defaults to a four-plane bit-serial dot; `GGML_W1AX_A4_KERNEL=conventional` selects a real quantized-code comparator.
- CPU build succeeded locally on Apple M3 Max; focused backend operator tests passed 88/88 across odd K, 32-bit boundaries, dirty tails, N=1/2/3, all-zero activations, negative zero, and strided views. This is a CPU correctness check, not an SM75 performance result. Python conversion tests were not run because the local environment lacks PyTorch.
- CUDA code is written but awaits CUDA compilation and actual 2080 Ti exact-dot/dispatch validation. Operator replay, all-nine real activation capture, full-model logits/IDs, and matched benchmark remain required gates. No GPU was used by this worker.
- Opt-in diagnostic capture: set `GGML_W1AX_CAPTURE_DIR` to an existing directory and disable CUDA graphs. Each `op-%012llu.bin` stores magic `W1AXACT1`, LE uint64 sequence/K/M/N, LE uint32 bits, a 128-byte NUL-padded packed tensor name, then N×K F32 activations. Capture synchronizes the CUDA stream and is excluded from timing.

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

## Next actions

1. Verify host and GPU state through tmux MCP, plus local and remote Git/build state.
2. Implement and audit common W1Ax weight/export and numeric contracts; pass CPU/CUDA reference gates and real SM75 dispatch checks.
3. Run the frozen full matrix, operator replay, round tracing, and context/policy diagnostics; analyze both FP16 and Q4_0 comparisons.
4. Write a compact report with raw artifact hashes, update this checkpoint and status, integrate tested code into `main`, and push.

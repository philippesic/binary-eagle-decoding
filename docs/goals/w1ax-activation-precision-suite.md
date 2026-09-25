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

## Next actions

1. Verify host and GPU state through tmux MCP, plus local and remote Git/build state.
2. Implement and audit common W1Ax weight/export and numeric contracts; pass CPU/CUDA reference gates and real SM75 dispatch checks.
3. Run the frozen full matrix, operator replay, round tracing, and context/policy diagnostics; analyze both FP16 and Q4_0 comparisons.
4. Write a compact report with raw artifact hashes, update this checkpoint and status, integrate tested code into `main`, and push.

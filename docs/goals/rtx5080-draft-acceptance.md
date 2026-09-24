# Goal: RTX 5080 W1A1 draft acceptance

**Opened:** 2026-09-23  
**State:** active  
**Orchestrator:** current Codex task  
**GPU owner:** to be assigned after the host connection is verified

## Objective

Confirm the pinned PyTorch W1A1 EAGLE drafter's greedy draft acceptance on the
RTX 5080, diagnose any target/ordinary EAGLE parity issue, and measure the
drafter's layer cost and binary-eligible coverage. Use the evidence to present
the user with a concrete choice between selective W1A1 coverage and bounded
quantization-aware training (QAT). This goal does not include QAT or native
binary execution.

## Completion evidence

- Record the actual 5080 GPU, CUDA/driver, Python/package versions, project and
  official-code revisions, pinned model snapshots and file hashes, commands,
  and resolved acceptance settings. Keep model files and raw runs out of Git.
- Verify ordinary EAGLE against target-only greedy generation and the disabled
  W1A1 wrapper on CUDA. If parity fails, preserve a token/logit trace and
  identify the cause or a precise blocker before interpreting W1A1 counts.
- Run the fixed 12-prompt held-out suite on the 5080 for ordinary EAGLE and the
  five existing W1A1 layer-group variants. Preserve per-prompt accepted nodes,
  proposed nodes, rounds, output lengths, and the prompt/config manifests.
- Profile actual drafter layer shapes and per-group cost on the 5080, including
  feature fusion, attention, FFN, vocabulary head, and remaining graph work.
  State measurement method, warmups, repetitions, and precision. Identify
  which groups are binary eligible and what share of measured draft cost each
  represents. Profiling is diagnostic, not a native W1A1 speed claim.
- Write a compact report in `experiments/` with raw artifact locations and
  hashes, CUDA versus Metal acceptance, uncertainty or parity limitations,
  and the evidence and tradeoffs for the user's next research decision.

## Boundaries

- Keep target weights, tokenizer, verifier, prompt suite, tree settings, and
  greedy sampling fixed. Do not use held-out prompts for training/calibration.
- Use the RTX 5080 for this phase. The RTX 2080 Ti remains reserved for SM75
  native-binary validation and paired end-to-end comparison.
- Do not infer binary throughput from PyTorch fake quantization or change the
  native runtime in this goal. If a greedy parity issue blocks a clean sweep,
  run a bounded diagnostic and report the limitation accurately.
- The user owns the selective-coverage versus QAT decision. Record options in
  `docs/DECISIONS.md` and continue independent audit work while it is pending.

## First independent work units

1. **GPU experiment operator:** exclusively own the 5080. Verify the current
   host via the local registry and tmux MCP, prepare the remote project and
   pinned environment/models, then run CUDA parity and acceptance under
   `scripts/remote_job.py`. Deliver raw artifacts, commands, hashes, and a
   concise finding. Stop after the fixed sweep or a parity blocker; do not
   start QAT. No other worker uses the 5080 concurrently.
2. **Profiler implementation:** in an isolated worktree, add a focused drafter
   layer-cost measurement tool under `scripts/` and any necessary focused
   tests. Deliver a reviewable commit and usage notes. No GPU or model download.
3. **Orchestrator:** review and integrate the profiler, analyze CUDA results,
   write the experiment report and decision options, and update this goal and
   `docs/STATUS.md` at milestones.

## Current checkpoint

- Goal opened from the completed PyTorch W1A1 EAGLE phase. The fixed prompt and
  model config are `configs/pytorch_w1a1.toml`; the Metal development report is
  `experiments/pytorch-w1a1-metal-acceptance.md`.
- User reports that the RTX 5080 is free and supplied a rotated address. SSH
  username and port are being confirmed; the address belongs only in the
  machine-local host registry. No remote connection or experiment has begun.
- Opening checkpoint `95cf31c` was pushed to `main` before worker setup.
  A Sol high profiling task has been dispatched into an isolated worktree;
  its GPU-free tool/check deliverable is pending. A Luna high GPU-operator
  task has been dispatched and is waiting for host registry details; it is
  the sole intended RTX 5080 owner. Task IDs are pending asynchronous setup.
- `e010c8c` adds an unquantized ordinary EAGLE row to the held-out CUDA
  sweep, making six configurations over 12 prompts. Runner dry-run passes.
  The remote model manifest must be generated against this updated config
  hash.
- `6f32a63` adds a full target-only greedy reference per prompt and checks
  every variant's output against it (allowing only round-boundary overshoot).
  A CUDA mismatch stops the sweep with token IDs preserved. `make check`
  passes all 17 tests. The GPU operator must fetch this commit before running.
- `7957efe` ensures the disabled-wrapper check uses an actual selected group.
  The Sol profiling tool was reviewed and integrated at `bfce552`. It records
  CUDA-event spans for the nine eligible drafter linears and `topK_genrate`,
  shapes, dtypes, group shares, and a residual; no 5080 measurement exists yet.
  The main-branch `make check` gate passes all 22 tests. The profiler's
  temporary worktree remains until its task is confirmed idle and clean.
- The GPU operator should use current `main` at or after `bfce552`, regenerate
  the model manifest against the updated config hash, then run parity,
  acceptance, and layer profiling under separate supervised run IDs.

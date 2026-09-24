# Goal: RTX 5080 W1A1 draft acceptance

**Opened:** 2026-09-23  
**State:** active; RTX 5080 work resumed by user
**Orchestrator:** current Codex task  
**GPU owner:** `/root/cuda_acceptance_operator` (Luna high); fresh idle check in progress

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
- User supplied the RTX 5080 address and SSH usernames for both GPU hosts;
  these are stored only in the machine-local host registry. The 2080 Ti address
  remains unset. No remote experiment has begun.
- Opening checkpoint `95cf31c` was pushed to `main` before worker setup.
  The Sol high profiling task's code was integrated at `bfce552`. A Luna
  high GPU-operator task was dispatched, but its Codex task ID is still pending
  asynchronous setup and no GPU work began. It is the sole intended 5080 owner.
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
  The main-branch `make check` gate passes all 22 tests.
- The GPU operator should use current `main` at or after `df8c6db`, regenerate
  the model manifest against the updated config hash, then run parity,
  acceptance, and layer profiling under separate supervised run IDs.
- A local read of the pinned drafter safetensors header confirmed all nine
  BF16 candidate linear shapes. `experiments/drafter-static-coverage.md`
  records exact parameter counts and distinguishes storage share from the
  pending measured draft time share.
- The user resumed RTX 5080 work. The shared local pause flag is clear. The
  earlier queued GPU task never became visible in the app task list or a GPU
  worktree; no remote work from it has been observed. Luna high subagent
  `/root/cuda_acceptance_operator` was assigned exclusive 5080 ownership for
  host verification and CPU preparation. Its bounded turn finished before
  CUDA runs because the Windows game continued to occupy the GPU.
- The operator reached the registered WSL host through tmux MCP and confirmed
  the RTX 5080 (16,303 MiB), WSL NVIDIA-SMI 615.71.08, Windows driver 616.92,
  and CUDA UMD 13.4. Windows-side inspection identified Fortnite and its
  overlay using 5,587 MiB; three samples showed 43%, 42%, and 41% GPU use.
  No WSL project process was found. Preflight evidence is saved locally at
  `results/preflight-5080-20260924T034000Z.txt` (SHA256
  `10dd5c335349668d80ebe59077c72e6750edd110d02253347c9fcfd659e8cdae`).
  The tmux MCP session was `w1a1-5080-acceptance`, pane `%0`. CUDA runs are held
  until the game closes and a fresh idle check passes. CPU-only checkout,
  environment, and model snapshot preparation may proceed meanwhile.
- The profiler worktree was clean, and its two owned files matched `main`
  exactly after cherry-pick integration. The temporary worktree and branch
  were removed; the published implementation remains at `bfce552`.
- CPU-only remote preparation now has a clean HTTPS clone of pushed project
  commit `daad157` under the configured workdir. The supervised
  `cuda-envsetup-20260924` run completed `uv sync --locked --group w1a1` with
  Python 3.11.15 and PyTorch 2.14.0+cu130 (CUDA build 13.0). The supervised
  `cuda-runtime-deps-20260924` run installed Transformers 4.57.6 and AngelSlim
  at the config-pinned revision; resolved Hugging Face Hub 0.36.2 and
  Accelerate 1.15.0. A first environment-report command exited 1 after a
  package metadata-name typo; a corrected import/package check later passed.
  No CUDA command has run while the GPU is occupied.
- Corrected package-version evidence is in remote supervised
  `runs/cuda-envfreeze-20260924/stdout.log`. The CPU-only supervised
  `cuda-model-snapshots-20260924` job downloaded and hashed the pinned model
  snapshots under the remote workdir. No GPU-backed command has started.
- `cuda-model-snapshots-20260924` finished with exit 0. The remote target has
  13 files / 8,060,926,626 bytes, and the drafter has 5 files / 436,987,505
  bytes; all file SHA256 values match the prior local manifest. Remote model
  manifest SHA256 is
  `2db1c860f059bd8702ca6bb523e64f0c7d064c7a95f59919a001fc88168a91e3`;
  resolved config SHA256 is
  `41e4509a5a444357b9c6c3079569ea9649a6ce9a34752b0d5908913d9062d6a2`.
  Prompt SHA256 remains `0d6a698d6816592c6ff435fed2fea4cdafe9f5248393d2a5ac091e1551919476`.
  Import and package metadata checks pass; CUDA-backed model loading has not
  been attempted. A fresh Windows-side idle check is next.
- Fresh Windows-side samples at 20:51:55–20:52:05 stayed at 43% utilization
  and 5,587 MiB, with Fortnite and its overlays still running. The local
  ignored record is `results/preflight-5080-20260924T034000Z/host-preflight.txt`
  (SHA256 `11904b197bb51c43ac1bd3e752bc9eb4ca03407c7a77b0625d23bb7e23d0f495`).
  It contains exact commands, remote run IDs, hashes, and process checks.
  All CPU setup/download supervisor jobs are terminal; the remote checkout is
  clean at `daad157`, no project process group remains, and both SSH sessions
  were closed. CUDA parity, acceptance, and profiling have not started.
- A final independent tmux MCP check at 20:56:12 again found the Fortnite
  process and overlays, 43% utilization, and 5,587 MiB allocated. That SSH
  connection and its local tmux session were closed. The GPU remains in use.
- Next: establish that Fortnite is gone and the GPU is idle using a fresh
  tmux MCP host check. Then assign one Luna GPU owner for the supervised CUDA
  parity smoke, full held-out sweep, and layer profile. No QAT begins here.
- The user asked to proceed again on 2026-09-24. The same Luna operator now
  owns the fresh idle gate and, if it passes, the supervised CUDA parity,
  acceptance, and profiling runs. Stop at a parity blocker; no QAT or 2080 Ti.
- Fortnite is gone. Three Windows-side idle samples at 00:14:58–00:15:08
  showed 0–1% use with 12,872 MiB free; idle EOS overlay processes remained.
  No other project job was active. The supervised one-prompt
  `cuda-parity-smoke-20260924` run exited 1 before model loading or CUDA
  allocation because the pinned AngelSlim import required missing `datasets`.
  This is an environment dependency failure, not a parity outcome. The
  operator is checking pinned package metadata and will retry under a new run
  ID after installing the minimal required dependency.
- AngelSlim's pinned wheel metadata required `datasets`; supervised CPU setup
  installed it (resolved version 5.0.1) plus two import-path requirements,
  `shortuuid` and Pillow. The pinned Torch 2.14.0, Transformers 4.57.6, and
  AngelSlim commit remained intact; CPU import probe passed. A fresh idle gate
  showed Fortnite absent and 0–1% GPU use before retry.
- Supervised `cuda-parity-smoke3-20260924` loaded all three target shards but
  exited 1 at the strict ordinary-EAGLE versus target-only greedy check. The
  disabled W1A1 wrapper and quantized variants did not run. Initial run peak
  sampling left about 3,358 MiB free; after supervisor exit Windows GPU memory
  returned to the 3,107 MiB idle baseline. Raw token IDs are in remote
  `results/cuda-parity-smoke3-20260924/greedy-parity.json`; the operator is
  capturing a bounded verifier-logit trace. The full held-out sweep and layer
  profile are stopped until the baseline mismatch is diagnosed.
- The first CUDA mismatch on `prose-01` is generated index 3 (fourth token):
  target-only ID 272 versus ordinary EAGLE ID 11, after shared IDs
  `[785, 7406, 41506]`. This is earlier than the Metal BF16 divergence at
  index 20. The raw parity file SHA256 is
  `592278d6ca4d1f23c3890f50cecd0cd0473de0f985d07c75f7e28a9c2c8c53a4`.
  Both target and draft loaded in BF16. The token source and full-prefix versus
  tree-verifier logits were traced next.
- The BF16 trace confirms the divergent token was a target-selected seed after
  two accepted draft positions. At absolute position 40, tree verification
  tied IDs 11, 272, and 7578 at logit 21.0 and selected lowest ID 11; the
  full-prefix target pass on the same 41-token prefix had 272 at 21.0 and 11
  and 7578 at 20.875, selecting 272. The immediate token-choice mechanism is
  known; the source of the tree/cache logit shift remains unresolved. Remote
  `results/cuda-parity-diagnostic-20260924/trace.json` SHA256 is
  `05563bd35b5f502c77a0a4485779e02bf74e8e07bda25d6f0a17341ac09a65ac`.
  Its diagnostic supervisor finished exit 0 and GPU memory returned to idle.
  `experiments/pytorch-w1a1-cuda-acceptance.md` records the current evidence.
- The separate ordinary drafter profile
  `cuda-drafter-profile-20260924` finished exit 0 on `prose-01`: two warmups,
  five measured repetitions, 70 draft invocations, 628.69 ms total draft-event
  time. The nine candidate linears accounted for 240.95 ms (38.32%): fusion
  1.03%, attention 10.08%, FFN 14.29%, head 12.93%; residual 61.68% includes
  other work and instrumentation. Peak PyTorch allocation was 9,781,691,904
  bytes. Remote profile SHA256 is
  `1677a3dd0b0cce78cec1e7296c6b18a30ed51e5e780fcd53509b8faed473c15f`.
  This diagnostic timing is not a binary or end-to-end speedup result.
- `ae47119` adds an explicit exploratory CUDA mode to the acceptance runner:
  strict parity remains default, while opt-in runs preserve per-prompt target
  mismatch records and mark the run as development evidence. `make check`
  passes all 22 tests. The operator will validate that mode on one prompt
  before considering a full held-out exploratory sweep.
- The operator fetched exact remote code commit `ae471198911fe7e093a4e26ff95655105843b725`
  with a clean tree. A fresh idle check found no Fortnite or project process
  and 0–1% GPU use. Supervised
  `cuda-exploratory-ordinary-fusion-20260924` is running one prompt with the
  ordinary and fusion variants and `--allow-greedy-mismatch`. Its output is
  exploratory verifier acceptance only; full-suite work awaits this check.

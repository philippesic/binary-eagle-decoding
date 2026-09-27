# Binary scale fitting on RTX 5080

Status: active, 2026-09-27 UTC. User-approved bounded experiment.

## Objective and scope

Measure four all-nine-linear fixed-sign binary candidates with A16 inputs:
A row mean-absolute, B group-128 mean-absolute, C fitted row, D fitted group-128.
Use F32 reference scales, fixed original source signs/layout, a valid nonnegative
constrained least-squares solve, and actual post-cast training inputs without
centering. Freeze solver and capture budgets before fitting. Do not train signs,
a readout, or the full model. Rounded exports are separate audit endpoints.

Use the existing 96 training prompts and 24 development prompts. Evaluate one
deterministic pass per development prompt at D=5, p_min=0, greedy, on the same
RTX 5080 native chain target/verifier, alongside FP16 and Q4_0 EAGLE controls
and target-only raw-token correctness. Preserve all 24 reserved-final prompts.
Report acceptance counts/denominators, first proposal, depth-two/three survival,
per-prompt IDs/divergences, hashes, scale changes and reconstruction errors.
Select using development acceptance. Publish the result and recommend promotion
or body/readout diagnosis; do not automatically start follow-up research.

## Ownership and resources

Independent orchestrator chat: 01a0e221-39c7-7352-ad48-f864146fbed1.
No dependency or reporting to originating chat.
Parent checkpoint: f6eb3f544ccd5defcce8a4a18b2d80124bb5a683.
Pinned llama.cpp starting revision: feba1569848e74996651a42a9751f8afa5958b1d.
Isolated checkout: /Users/pippo/.codex/worktrees/binary-scale-fitting/binary-eagle-decoding.
RTX 5080 is the only authorized GPU. Registry read from
~/.config/binary-eagle-decoding/hosts.toml; never contact RTX 2080 Ti.
Orchestrator owns all GPU work through tmux MCP session scale-fitting-5080
(session $11, pane %26), using scripts/remote_job.py for every experiment.
No remote experiment currently running.
Bounded Sol runtime worker /root/runtime is inspecting native reference/capture
implementation, without GPU access. Parent owns this checkpoint and reporting.

## Checks and evidence

2026-09-27 02:11 UTC preflight: RTX 5080, 16,303 MiB total / 1,940 MiB used,
0% utilization, no reported compute processes, driver 615.71.08. Local pause
request false. SSH via tmux MCP and wsl.exe -e /usr/lib/wsl/lib/nvidia-smi.
Read evaluation protocol, operations, next-step consolidation and scale-fitting
research section. Existing historical studies remain sealed.

## Next action

Finish runtime/capture design; freeze practical calibration/solver budget;
implement and test bounded reference extensions; capture training activations,
fit and evaluate the four variants plus matched controls. Publish coherent
progress regularly, including the submodule before its parent gitlink.

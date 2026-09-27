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

2026-09-27 09:11 UTC preflight (nvidia-smi displayed local02:11): RTX 5080, 16,303 MiB total / 1,940 MiB used,
0% utilization, no reported compute processes, driver 615.71.08. Local pause
request false. SSH via tmux MCP and wsl.exe -e /usr/lib/wsl/lib/nvidia-smi.
Read evaluation protocol, operations, next-step consolidation and scale-fitting
research section. Existing historical studies remain sealed.

## Next action

Finish runtime/capture design; freeze practical calibration/solver budget;
implement and test bounded reference extensions; capture training activations,
fit and evaluate the four variants plus matched controls. Publish coherent
progress regularly, including the submodule before its parent gitlink.

## Frozen calibration and solver protocol (before any fit)

Capture ordinary native drafter with explicit F32→F16→F32 boundaries at all
nine selected linears, greedy D=5/p_min=0, 32 generated tokens on every one of
96 train prompts. Use at most32 evenly spaced rows over each layer/prompt's
complete capture stream (prefill and recurrent), at most3072 rows/layer.
Capture values remain uncentered. Compare source BF16→F32 weights in canonical
GGUF Q/K row order; audit BF16→F16 source cast errors and sign differences.

F32 scales throughout. Row NNLS uses exact scalar constrained ridge solution.
Group128 uses feasible projected accelerated gradient, output-row batches32,
maximum512 iterations, relative projected-gradient tolerance1e-6. Per-row ridge
lambda=1e-4 times mean diagonal of ZᵀZ/N, centered at mean-absolute anchor.
No sweep. Record convergence residuals and iteration counts. Each fitted row
falls back to its baseline if direct unregularized reconstruction SSE worsens.
Freeze signs, inputs, source, grouping, regularizer and budget for all variants.
Native reference arithmetic: F16-cast X promoted F32, ordered F32 signed sums,
F32 scale multiplication, and F32 sum across groups. This is a quality reference,
not an optimized kernel or throughput claim. Audit fitting-versus-native
reduction-order error explicitly.

Development uses unchanged128 maximum output tokens (EOS honored), one pass
on all24 development prompts, D5/p_min0, temperature0, seed42, no thinking,
context2048, concurrency1, target/draft F16 KV. Same pinned runtime and target
for target-only, FP16, Q4_0, A, B, C, D. Capture/trace diagnostics have no timing
interpretation. Include zero-, partial-, full-accept transitions in parity gates.

## Implementation checkpoint

Remote isolated checkout: /home/philip/binary-eagle-decoding/scale-fitting-20260927.
Source/target files are shared read-only via models symlinks to the host project.
Training/development manifests regenerated from the pinned deterministic generator;
train SHA25680e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74,
development SHA256a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885.
Final manifest generated but not evaluated or used for selection.
Remote environment: torch2.14.0+cu130, CUDA compiler13.1, compute targetSM120.
Baseline build run scale-build-base-20260927 exited2 (cmake absent from default
PATH); retry scale-build-base2-20260927 uses existing .venv tool PATH.
No GPU inference has started. A second tmux pane%27 is for status inspection.
Source worker owns only submodule; solver worker owns fitter/export tests;
Luna worker owns screen-runner tests. Astra review requires post-cast/source
checks and native boundary-state assertions; implementation includes them.

Runtime published: llama.cpp2e8d2e8dac6354798037e4e9aca455cd898bf06f,
branch scale-reference-a16. CPU-only local build and112/112 W1A1_MUL_MAT
operator cases passed (88 legacy,24 grouped). CUDA validation pending.
Version2 row/group scales preserve existing version1 default. Dense capture
GGML_EAGLE_DENSE_A16=1 and GGML_W1AX_CAPTURE_DIR; EAGLE_STATE_TRACE_JSONL
records boundary feature hash, accepted-row position and cache truncation extent.
This trace does not claim full cached K/V numerical parity.
Remote baseline build3 running with verified existing CUDA/glibc header patch,
SHA25613256b220d400a5665cde8bc87c21b0188944390c6a945a7fffa2827254b305a.
Build2 failed the known rsqrt exception declaration conflict, before any GPU run.

Fitter/exporter0258aac independently reviewed and8CPU tests passed. Original
source dtype must beBF16, F16 canonical GGUF weights must match exact cast,
exported scales finite/nonnegative. Legacy CPUtorch row mean reduction preserved.
Runner/analyzer15fa0e4 plus b984a4e passed8CPU tests, including raw IDs, terminal
acceptance, per-prompt state slices, missing capture failures and child cleanup.
Runtime CUDA build: runs/scale-build-reference-20260927, started09:22UTC,
currently running. Baseline build3 finished successfully09:21:32UTC.
Parent main published through0258aac. No GPU inference/fitting yet.

09:26UTC: RTX5080 CUDA gate112/112 passed, backendCUDA0; run
scale-cuda-operators-20260927 finished exit0 and supervisor cleaned its group.
Native graph cast diagnostic running: scale-cast-gate-20260927,
results/scale-cast-gate,3 historical prompts (prose-01/code-01/reasoning-01),
128tokens each, ordinary versus A16cast-only, both capture all9 layers/state.
This is diagnostic data only, never calibration fitting. Live owner parent,
pane%26; inspect via%27. Next verify state/cast gate then96train capture.

09:28UTC cast gate complete: ordinary and cast_only exact output IDs, proposals,
accept counts and state-event records on all3 prompts. Each path accepted199
in182 rounds. All6 per-prompt state audits passed; each path completed75zero,
100partial and3full accept→seed transitions, plus3 initial seeds. Saved local
compact copy results/binary-scale-fitting-5080/cast-analysis.json in main checkout.
GPU released after gate; refreshed preflight0%/no compute processes,2115MiB
noncompute use, pause false. Training capture scale-train-capture-20260927 is
now running under pane%26, results/scale-train-capture. CPU-only replay build
scale-replay-build-20260927 under%27. Replay extension8b81c53 published on main,
CPU fixture checks441outputs exact scalar parity, independent review passed.

09:29:58UTC training capture96/96 finished exit0, all9 layers/request checked,
2.4GB raw preserved. Refreshed GPU/pause check before fitting: no compute jobs,
0% utilization,2115MiB noncompute use, pausefalse. scale-fit-20260927 running
under%26: existing frozen fitter on CUDA, outputs results/scale-fit. No solver
or hyperparameter changes. CPU-only pureQ4_0 control conversion running under
%27 (CUDA_VISIBLE_DEVICES=-1), scale-q4-control-20260927; target unchanged.
Native library hashes and environment saved results/scale-native-hashes.txt
and results/scale-environment.json. Replay executable built successfully.

# QAT optimization integration checkpoint

All five requested areas are implemented and CPU-tested; published main eb66093. Actual CUDA readiness
remains unverified: both GPUs are paused by the human, and this goal performed
no remote/GPU execution, real-data optimizer updates, real correction fitting or
sealed-final reads. The goal remains open.

## Implemented controls

| Area | Implemented and tested | Remaining evidence |
| --- | --- | --- |
| Computation | No-gradient A1 shortcut, opt-in single-forward autograd, shared quantization, duplicate depth CE removed | Full-model CUDA forward/backward and synchronized timing |
| Cache and batching | Chunked K/V-only context, per-chain stacked heads, masks/cache casts/ancestry checks, independent B2/B4 probe harness | Actual memory/timing; larger optimizer batches are not implemented or claimed |
| Binary optimization | Initialization, separate learning rates, AdamW/SGD, gradient rules, clipping, persistent sign-flip diagnostics | Real training comparison with equal budgets |
| Activation quantization | Learned A4/A8 relative clipping and A1 thresholds, six shared boundaries, versioned state/export/native packing | CUDA compilation, exact packing and real native decisions |
| Curricula | Direct A1, A8→A1 and A8→A4→A1 schedules, depth weighting, fresh optimizer transfer, budgets, refresh ancestry and exact resume | Every-stage actual CUDA receipt and fresh eligible provider |
| Added human scope | Raw-input rank1/rank4 fusion correction with optional bias; per-row affine binary midpoints for fusion/all | Nonzero native CUDA operators and actual-model decisions; real fitting operands |

Default cache/head and binary gradient behavior stay opt-in. New recipe receipts
bind critical source, full native revision, runtime, hardware, effective packed
deployment state and every scheduled precision. Optimized CUDA launches reject
missing or stale evidence. CPU-only planning loads no models and queries no GPU.
Measured collectors derive checks from raw data rather than accepting handwritten
pass flags; old frozen source/runtime receipts cannot admit new recipes.

## Verification

Final integrated Python suite: **952 tests, four skips**, 104.553 seconds on
Apple M3 Max, Python3.11.15, Torch2.14.0 CPU-only and NumPy2.4.6. Raw log:
`runs/qat-optimization-readiness/cpu-final-suite.log`. After final source-inventory
and formatting changes, the whole suite passed again:952tests/fourskips in
104.517seconds, recorded in
`runs/qat-optimization-readiness/cpu-final-formatted-suite.log`. Diff-scoped Ruff
and whitespace checks pass. An additional 35 readiness-tool tests passed after
the collector import fix.

Published native revision **8025a07773b7828bdeb4f3e0b834c8b54cb65c66** on the
user's llama.cpp fork, branch `feature/learned-w1ax`: 220/220 CPU operator cases;
57 exact packed-byte pairs, 36 projection cases, 114 loader cases, 59 encoder
graphs and 218 audited arithmetic nodes. The actual JSON report is
`/private/tmp/eagle-native-json-cpu-6.json` (also preserved under ignored
`runs/qat-optimization-readiness/native-cpu/`); it identifies Apple M3 Max CPU and
is correctly refused as CUDA evidence. CUDA source is uncompiled and unrun.
The detailed [native report](06-native-quantizers.md) records raw logs.

Tests cover hard/native arithmetic order, zeros/subnormals, scale revival,
quantizer and affine gradients, later-state/K/V reachability, shared parameter
identity, frozen operand ownership, schemas2–5, malformed metadata, strict
admission, exact paired/curriculum resume, zero optimizer moments and source/data
ancestry. Baseline fixture repairs preserve the original assertions and prevent
mocked process IDs from reaching actual OS signal calls.

A16 affine sums use FP16 boundary values accumulated in F32, not integer codes;
Torch/native accumulation order may differ. Use the recorded cancellation-aware
sum and output RMS gates, with real drafter-decision checks. Exact integer-code
sum claims apply to A1/A4/A8 under actual model width bounds.

## Outstanding endpoint

The GPU pause takes precedence over the requested completion endpoint. After a
new human resume instruction, one operator must build the published native
source in a separate project directory, collect actual CUDA packing/nonzero
correction/midpoint and train-ancestry native decisions, then run the zero-update
paired or complete-schedule full-model forward/backward/memory/timing producer.
Preserve failed evidence. The existing corpus preparation is incomplete
(10,000 train, 224/1,002 development, 327 retained manifests, zero QAT updates).
No acceptance, speed or quality improvement is inferred from these CPU checks.
Q4_0 EAGLE remains the primary success comparison.

The [runbook](../../docs/QAT_OPTIMIZATION_RUNBOOK.md) specifies the launch gates;
[goal state](../../docs/goals/qat-optimization-readiness.md) records integration
and retained worktrees. Reports01–10 contain bounded worker findings; this
checkpoint supersedes their earlier interim test counts and smoke limitations.

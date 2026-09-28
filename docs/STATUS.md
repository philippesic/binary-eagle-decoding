# Current project status

**Comparison target:** Q4_0 EAGLE is the baseline to beat for acceptance,
latency and total throughput. FP16 EAGLE is secondary diagnostic context.
The target/verifier model precision remains as frozen for each experiment.

**Active goal:** [joint binary EAGLE body and head](goals/recurrent-binary-body-head.md), CPU preparation under the user's explicit no-GPU restriction. **Latest completed goal:** [mixed precision rescue and frozen-body head adaptation on RTX 5080](goals/binary-rescue-head-5080.md), completed 2026-09-28 UTC. The preceding [binary scale fitting goal](goals/binary-scale-fitting-5080.md) completed 2026-09-27 UTC.

**Current goal milestone (2026-09-28 UTC):** the [joint-training protocol](../experiments/recurrent-binary-qat-plan.md) records the candidate-D W1A16 representation, exact-prefix recurrent supervision, proposed bounded trial and stop gates. The CPU hard-binary core, nine-linear installer, exact-prefix trace validator, masked recurrent loss, differentiable proposal-chain interface, optimizer ownership/checkpoint step and learned-scale GGUF serializer are integrated. The focused CPU gates pass, including a training-checkpoint-to-GGUF synthetic roundtrip and a two-step causal-cache gradient test. A new scalar CPU replay matched 430/430 archived candidate-D native samples across all nine projections. This is arithmetic evidence on old training captures, not a trained-model quality result. Existing cached head states cannot train the body; a real feature/cache/verifier capture, efficient training forward and native trained-scale GGUF loader amendment remain necessary. No accelerator, remote host, model training or final-set prompt has been used for this goal. Q4_0 remains the primary future acceptance, latency and throughput gate; no new quality or speed claim exists.

**Current milestone:** the [final rescue/readout report](../experiments/binary-rescue-head-5080.md) records a negative result against Q4_0 EAGLE. Q4_0 reached 1.042 accepted drafts/round and 135.1 full-request tokens/s; the admitted attention+fusion Q8_0 rescue reached 0.696 and 83.3 (0.616×), and the fitted frozen-D-body FP16 head reached 0.555 and 60.8 (0.450×). All 120 primary measured raw outputs per path matched Q4_0 and target-only. The 264-request speculative round calibration, 92 graph-verified server blocks across final primary/diagnostic runs, and separate six-prompt longer-context comparison are complete. The longer-context client rates were 127.0 Q4_0, 68.5 combined rescue, 47.1 fitted head and 93.8 target-only tokens/s. Neither endpoint beat target-only. Full raw results, logs, binaries and model artifacts are indexed in the remote archive named in the report. Final RTX 5080 check found no owned process or compute app, 1,916 MiB whole-device use and 0% utilization. The user owns any future body-aware QAT or final-set decision; none was started.

The approved four-way A16 screen completed all **168 requests** (7 paths × 24 development prompts). A/B/C/D accepted **0.119 / 0.161 / 0.290 / 0.425 drafts per round**, versus **1.037 FP16** and **1.042 Q4_0**. Fitted group scales improved 3.585× over row-mean A, closing 33.20% of the Q4_0 gap. D beat all other binary candidates on every prompt but trailed both controls on every prompt. All seven paths matched target-only raw IDs on all 24 prompts. See the [report](../experiments/binary-scale-fitting-5080.md) for counts, depth survival, calibration, artifacts and limitations.

The scale-screen recommendation at that time was to retain D for a common-history body/readout diagnostic; the completed goal above supplied it. The [all-layer W1Ax suite](goals/w1ax-activation-precision-suite.md) and [RTX 2080 Ti quantization suite](goals/rtx2080ti-quantization-suite.md) remain completed and sealed; the prior [W1A1 research goal](goals/full-w1a1-eagle-project.md) remains checkpointed. The next scope/budget choice is user-owned.

**Final W1Ax result:** the twelve-cell development policy grid completed and
validated all **11,520 requests**, alongside the sealed historical/development,
operator, round, profiling, streaming and 720-request context measurements.
Every W1Ax path had lower pooled decode and full-request throughput than both
same-cell anchors. The largest observed decode ratios were 0.707× FP16 and
0.684× Q4_0. The [main report](../experiments/w1ax-activation-precision-results.md)
and [complete policy appendix](../experiments/w1ax-policy-grid-results.md) retain
fixed versus development-selected policies, raw-output differences, timing-control
variation and measurement limits. The interrupted 120-request attempt remains
preserved and excluded from completed-cell rates. Final verification found no
owned jobs or GPU compute apps; the 2080 Ti was idle. The goal file records exact
artifacts, hashes, checks and completion evidence.

Ubuntu 24.04 WSL2 and SSH are reachable through the shared host registry. The
RTX 2080 Ti (SM75) passed five native W1A1 CUDA backend
cases, a standalone 21-case/880-dot binary-MMA probe, and both integrated
portable/MMA eight-case backend gates. All five W1A1 draft GGUFs passed
source-row audits; the FP16 target track fits in 11,264 MiB VRAM.

The full [nine-variant Turing comparison](../experiments/rtx2080ti-quantization-suite.md)
completed 540/540 matched requests across 12 prompts and five repetitions.
Ordinary EAGLE decoded at 82.49 tokens/s. Native head W1A1 reached 74.81
(0.907× ordinary), and all-group W1A1 46.31 (0.561×). Q4_0 and Q8_0 draft
controls reached 91.51 and 87.73 (1.109× and 1.064×). Fusion, attention,
and FFN W1A1 also trailed ordinary. All eight speculative variants produced
identical text on all 60 paired requests; each differed from target-only on
one prompt. Ratios to target-only are timing observations, not strict
lossless speedups. Q4_0/Q8_0 stored types are verified, and a later isolated
trace confirmed their Q8_1 activation conversion and MMVQ/MMQ dispatch.

A separate 240-request four-path comparison found integrated binary-MMA head
W1A1 at 74.72 decode tok/s versus portable head W1A1 at 74.59, a 1.0017×
ratio with paired 95% interval 0.9973–1.0062. Both paths emitted identical
text on all 60 paired requests; no end-to-end MMA gain was resolved. The
genuine native W8A8/W4A4 300-request vector comparison has also finished
with verified CUDA dispatch. W8A8 decoded at 80.18 tok/s versus ordinary
EAGLE's 80.99 (0.990×; paired 95% interval 0.975–1.005), so no difference
was resolved. W4A4 decoded at 41.40 tok/s (0.511×), with accepted drafts
collapsing to 0.092/round versus ordinary's 1.168. The seven-path Tensor
Core comparison completed 420/420 requests with identical text and acceptance within
each default/MMA pair. W8A8 MMA decoded at 0.821× its default DP4A path
(paired 95% interval 0.817–0.824); W4A4 MMA at 0.880× its default vector
(0.876–0.884). Specialized matrix instructions were slower for this EAGLE
workload. A separate executed-path trace confirmed Q4_0/Q8_0 use Q8_1
activation quantization with MMVQ during one/two-token work and MMQ during an
observed 38-token operation. The goal file has exact commits, owners, and raw
hashes. The [final synthesis](../experiments/rtx2080ti-synthesis.md) and
packing-inclusive CUDA traces are preserved; final cleanup confirmed all
supervisors stopped and the GPU released.

## RTX 5080 result

A dedicated packed W1A1 operation, EAGLE output-head GGUF exporter/loader,
and CUDA dispatch were integrated in the 5080-tested llama.cpp revision
`92bc706`. The expanded five-setting revision is `8d2b18a`.
CUDA backend correctness passed 5/5 cases, all 32,000 packed head rows were
audited against the published BF16 source, and every packed benchmark server
logged actual CUDA XOR/POPCOUNT dispatch. See the
[integration report](../experiments/ggml-w1a1-cuda-5080.md).
A separate [captured-input parity run](../experiments/real-head-parity-5080.md)
checked 8 real drafter inputs against all 32,000 packed head rows on the
5080: exact sign packing and 256,000 integer dots, with no scaled-output
tolerance failures.

The [matched five-repetition comparison](../experiments/native-end-to-end-5080.md)
completed 180 requests on 12 fixed prompts with the same FP16 target and
server settings:

| Variant | Request tokens/s | Decode tokens/s |
| --- | ---: | ---: |
| Target-only | 96.67 | 99.39 |
| Ordinary EAGLE | 123.96 | 132.88 |
| Packed-head W1A1 EAGLE | 115.38 | 122.94 |

Packed-head W1A1 achieved **0.931× ordinary request throughput** and
**0.925× ordinary decode throughput**. Draft generation became faster per
round (4.386→3.515 ms), but accepted draft tokens fell (1.161→0.892 per
round), requiring 480 extra verification rounds. All five repetitions and
all three prompt categories favored ordinary EAGLE. The packed draft used
148 MiB less GPU memory while loaded. Both speculative paths exceeded
target-only throughput, but their outputs differed from target-only on two
prompts, so that ratio is not a clean lossless speedup claim.
An integrated-kernel Nsight Compute attempt was limited by
`ERR_NVGPUCTRPERM`; separate standalone packing-inclusive CUDA-event timings
are preserved, but no integrated per-kernel trace is claimed.

Ordinary and packed EAGLE decoded texts matched on all 60 paired requests.
A separate raw-token check found identical ordinary/packed IDs on the two
target-only mismatch prompts. An isolated [raw verifier-logit
trace](../experiments/native-verifier-trace-5080.md) reproduced both: at the
emitted rows, the target verifier itself ranked the speculative output first,
by 0.008074 and 0.000729 raw-logit units. The drafts were rejected, so these
were not wrongly accepted tokens. Target-only ranked the opposite IDs first
by 0.000963 and 0.016508 nats. Numerical sensitivity is plausible, but the
precise native baseline mismatch cause remains unproven. The earlier BF16
PyTorch verifier trace separately identified a tree-versus-incremental target
logit tie; see the [acceptance
report](../experiments/pytorch-w1a1-cuda-acceptance.md).

## Other completed gates

The [bounded head-only QAT pilot](../experiments/qat-head-pilot-results.md)
improved validation KL but reduced fixed held-out W1A1 acceptance to 1.565
drafts/round from the untrained 1.677. That recipe was stopped without
held-out tuning. The user's requested [W4A4/W8A8 accepted-per-round
comparison](../experiments/pytorch-int4-int8-cuda-acceptance.md) measured
0.2882 and 2.1816 respectively under the BF16 PyTorch verifier; those are
numerical simulations, not native INT4/INT8 timing.

A standalone SM75 binary-MMA probe cross-compiled to `BMMA.88128.XOR.POPC`
and later passed 21 cases/880 exact integer dots on the RTX 2080 Ti; see the
[2080 Ti suite](../experiments/rtx2080ti-quantization-suite.md). A focused
[related-work note](../experiments/related-work-note.md) keeps novelty claims
narrow: quantized EAGLE and native QAT already exist.
An opt-in [integrated binary-MMA
candidate](../experiments/integrated-binary-mma-5080.md) also passed 8/8
scalar-reference backend cases on the 5080, matched all 85 packed-draft
tokens in one model request, and compiled to SM75 SASS containing the exact
binary-MMA instruction. It subsequently passed the real SM75 backend gate and
tied portable W1A1 in the matched head comparison above.
The paired benchmark runner and analysis now support an opt-in fourth MMA
variant with separate selector/dispatch records and same-device MMA/portable
speed ratios. Local fake-server and analysis checks passed 15/15; the
[2080 Ti runbook](RTX2080TI_RUNBOOK.md) specifies the required run.

## Next research gate

The [post-suite consolidation](../experiments/one-bit-next-steps-2026-09-27.md)
updates the earlier agent brainstorms using the completed policy and round
measurements. It recommends recovering useful binary weights with wider
activations first: finish missing graph/state parity checks, then a train-only
fixed-sign row/group scale-fitting screen, with readout/body adaptation
conditional on its result. W1A1 remains the research endpoint. The subsequently approved scale-fitting screen is now complete (report above).
Body/readout adaptation and any expanded training budget remain user-owned;
reserved-final evaluation has not begun.

The [all-layer W1Ax study](../experiments/w1ax-activation-precision-results.md)
is complete, including the predeclared policy grid. Its report separates
acceptance, identical-input operator cost, complete round timing and serving
rates, with explicit measurement limits. Use its quality evidence to guide the [QAT revisit
plan](../experiments/qat-revisit-plan.md): audit
drafter-state/target-verifier alignment and target probability mass outside the
draft vocabulary before training, then test one bounded target-aligned recipe
on the frozen new development/final prompts. If acceptance improves, run native
same-device end-to-end comparisons against **both** FP16 EAGLE and Q4_0 EAGLE.
Later, screen non-EAGLE drafters such as block-parallel DFlash/DSpark before
investing in their W1A1 kernels. The completed 2080 Ti and 5080 experiments
are sealed, and the GPUs were released after their runs.

## Local research review (2026-09-25)

Seven Astra-high local-only analyses are collected in the
[ranked synthesis](../experiments/research-review-2026-09-25.md). They identify
target-aligned QAT capture, unused native cache-catch-up graph work, genuine
early draft caps, shared activation packing, structured scales, and a later
DFlash/DSpark quality screen as bounded opportunities. These are advisory
findings, not new performance results or changes to the active W1Ax protocol.
The initial review performed no GPU work or web search. A subsequent
[primary-source cross-reference](../experiments/research-cross-reference-2026-09-25.md)
revised the seven reports: DSpark becomes the lead later architecture candidate
with matched DFlash control; hard CE and structured scales remain hypotheses;
native sample-and-match is distinguished from probability-ratio verification.
The current W1Ax goal, one-run QAT budget and sealed final set are unchanged.
This literature review produced no new model or GPU result.

## PrismML / quantization research (2026-09-25)

The user-requested [PrismML research report](../experiments/prism-quantization-research-2026-09-25.md)
separates ternary task-score claims from true binary-weight execution and our
W1A1 acceptance objective. A full selected-weight geometry audit and a bounded
local CPU Q1 control support investigating representation fitting and
head/body adaptation before new kernel work. Existing Q1_0 support was verified;
the naive grouped control is not an optimized Prism checkpoint.

The active SM75 suite, its GPU owner, frozen protocol and reserved final prompts
are unchanged. New fitting/QAT budgets and any practical weight-only branch
remain proposals for the user, not additional experiments started by this review.

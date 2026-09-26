# Current project status

**Active goal:** [all-layer W1Ax activation-precision suite on RTX 2080 Ti](goals/w1ax-activation-precision-suite.md),
opened 2026-09-25. The [RTX 2080 Ti quantization
suite](goals/rtx2080ti-quantization-suite.md) completed on 2026-09-25.
The prior [W1A1 EAGLE research goal](goals/full-w1a1-eagle-project.md) is
checkpointed, and its Turing measurements are complete. **Current research
priority:** run the W1Ax activation-precision study, then revisit W1A1
quantization-aware training (QAT) to recover held-out draft acceptance. Every
future native W1A1 throughput comparison must include
both ordinary FP16 EAGLE and Q4_0 EAGLE drafts under the same target/verifier;
alternative drafter architectures are a later research direction. No new
QAT run has begun.

**Concurrent suite progress:** the latest [goal checkpoint](goals/w1ax-activation-precision-suite.md)
records W1Ax implementation, 88-case SM75 gates, real-input replay and the
completed historical (480 requests) and development (960 requests) eight-path matrices. Matched operator replay and the 480-request round trace are also complete. Full-server profiling and streaming diagnostics are complete. The 720-request context matrix is complete; the twelve-setting development policy sweep is underway. The goal file’s successor handoff records the live job, completed worker transfer, exact next actions and stop procedure. Consult that checkpoint for current
run ownership, paths and stop instructions. These records were preserved during
integration of the literature review; its authors did not operate or revalidate
that GPU run.

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

First establish the [all-layer W1Ax activation-precision
matrix](../experiments/w1ax-activation-precision-plan.md) on the 2080 Ti,
including W1A16/W1A8/W1A4 and matched FP16, Q8_0, Q4_0, W1A1 anchors. The
protocol separates acceptance, identical-input operator cost, complete EAGLE
round cost, and end-to-end serving rate. Implementation and focused SM75 checks have since progressed in the
linked goal checkpoint; the primary serving comparisons are sealed and broader diagnostics
remain to be completed by the suite owner. Use its quality evidence to guide the [QAT revisit
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

# Current project status

**Active goal:** [benchmark the RTX 2080 Ti quantization suite](goals/rtx2080ti-quantization-suite.md).
The prior [W1A1 EAGLE research goal](goals/full-w1a1-eagle-project.md) is
checkpointed; its remaining Turing measurements are now tracked in the active
benchmark goal. Ubuntu 24.04 WSL2 and SSH are reachable through the shared
host registry. The RTX 2080 Ti (SM75) passed five native W1A1 CUDA backend
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
lossless speedups. Q4_0/Q8_0 stored types are verified, while their exact
decode kernel path remains source-inferred pending an opt-in trace.

A separate 240-request four-path comparison found integrated binary-MMA head
W1A1 at 74.72 decode tok/s versus portable head W1A1 at 74.59, a 1.0017×
ratio with paired 95% interval 0.9973–1.0062. Both paths emitted identical
text on all 60 paired requests; no end-to-end MMA gain was resolved. The
remaining active gates are SM75 correctness and timing for genuine W8A8/W4A4
draft operators, an executed Q4_0/Q8_0 kernel trace, and a compact final
synthesis. The goal file has exact commits, owners, raw hashes, and next action.

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
and passed 21 cases/880 exact integer dots in an explicitly labeled SM120
proxy run. It has **not** executed on the RTX 2080 Ti; see the
[probe record](../experiments/sm75-binary-mma-plan.md). A focused
[related-work note](../experiments/related-work-note.md) keeps novelty claims
narrow: quantized EAGLE and native QAT already exist.
An opt-in [integrated binary-MMA
candidate](../experiments/integrated-binary-mma-5080.md) also passed 8/8
scalar-reference backend cases on the 5080, matched all 85 packed-draft
tokens in one model request, and compiled to SM75 SASS containing the exact
binary-MMA instruction. It remains on a published experimental branch; no
SM75 binary was run and no MMA speed result is claimed.
The paired benchmark runner and analysis now support an opt-in fourth MMA
variant with separate selector/dispatch records and same-device MMA/portable
speed ratios. Local fake-server and analysis checks passed 15/15; the
[2080 Ti runbook](RTX2080TI_RUNBOOK.md) specifies the required run.

## Next gate

The nine-variant SM75 matrix and full four-path binary-MMA comparison are
sealed with pinned source and raw hashes. The next gates are isolated
W8A8/W4A4 SM75 build and execution checks and a short Q4_0/Q8_0 kernel
trace. The [runbook](RTX2080TI_RUNBOOK.md) governs those runs.
The 5080 experiments are complete and sealed. Its remote checkout was restored
to pinned `92bc706`; all supervised jobs exited, no project process remained,
and repeated GPU samples showed 0% utilization. The repository checks pass
(78 tests); the expanded W1A1 branch tests passed 8/8 on CPU and CUDA.

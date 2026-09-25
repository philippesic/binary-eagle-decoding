# Goal: benchmark the RTX 2080 Ti quantization suite

**Opened:** 2026-09-24
**State:** complete 2026-09-25
**GPU owner:** none; RTX 2080 Ti released

## Objective and scope

Benchmark all five native W1A1 EAGLE coverage settings and the other relevant
Turing operand precisions on the RTX 2080 Ti. Keep the Qwen3-4B target, prompts,
verification settings, and hardware matched within each comparison. Distinguish
true weight-and-activation INT8/INT4 arithmetic from Q4_0/Q8_0 GGUF storage
formats. Compare portable one-bit execution with binary Tensor Core MMA and
compare genuine W8A8/W4A4 default paths with opt-in INT8/INT4 MMA.

The full checkpoint chronology, workers, commits, and intermediate run IDs
are preserved in the [archived goal record](../../experiments/rtx2080ti-goal-checkpoints.md).
The [experiment report](../../experiments/rtx2080ti-quantization-suite.md)
has exact commands, environment, model/binary/config hashes, raw artifact
locations, acceptance counts, paired intervals, and cleanup. The
[synthesis](../../experiments/rtx2080ti-synthesis.md) states the research result.

## Completed evidence

- Ubuntu 24.04 WSL2 on RTX 2080 Ti, compute capability 7.5, used a user-space
  CUDA 12.8.93/GNU 13.4 toolchain. All 18 pinned source files were hash
  verified. All five W1A1 GGUFs passed source-row audits, including zero sign
  mismatches across 65,280 all-group rows. W8A8/W4A4 all-nine exports matched
  every source-derived code and F32 scale byte.
- Real SM75 backend gates passed: production W1A1 5/5, integrated portable
  and binary MMA 8/8 each, W8A8 default/MMA 5/5 each, and W4A4 default/MMA
  4/4 each. A binary-MMA probe matched 880 exact integer dots; signed INT8
  and INT4 MMA probes matched 63 cases and 308 outputs. Executed CUDA/SASS
  evidence includes BMMA.88128.XOR.POPC, IMMA.8816.S8.S8, and
  IMMA.8832.S4.S4.SAT.
- The frozen 12-prompt, five-repetition matrices completed 540 production
  requests, 240 portable-versus-binary-MMA requests, 300 genuine W8A8/W4A4
  default requests, and 420 INT8/INT4 MMA-versus-default requests. One server
  ran at a time; all required loader, selector, and CUDA dispatch markers
  were checked. Paired bootstrap intervals used 2,000 prompt/repetition
  resamples.

| Same-device paired comparison | Decode-rate ratio | Paired 95% interval |
|---|---:|---:|
| W1A1 fusion / ordinary EAGLE | 0.598 | 0.545–0.660 |
| W1A1 attention / ordinary EAGLE | 0.596 | 0.552–0.646 |
| W1A1 FFN / ordinary EAGLE | 0.716 | 0.677–0.756 |
| W1A1 head / ordinary EAGLE | 0.907 | 0.869–0.946 |
| W1A1 all nine / ordinary EAGLE | 0.561 | 0.511–0.617 |
| Q4_0 draft / ordinary EAGLE | 1.109 | 1.081–1.135 |
| Q8_0 draft / ordinary EAGLE | 1.064 | 1.053–1.075 |
| Genuine W8A8 default / ordinary EAGLE | 0.990 | 0.975–1.005 |
| Genuine W4A4 default / ordinary EAGLE | 0.511 | 0.466–0.561 |
| Binary Tensor Core head / portable W1A1 head | 1.002 | 0.997–1.006 |
| W8A8 INT8 MMA / W8A8 DP4A | 0.821 | 0.817–0.824 |
| W4A4 INT4 MMA / W4A4 vector | 0.880 | 0.876–0.884 |

- Acceptance explains much of the W1A1 and W4A4 loss. In the production
  matrix, ordinary EAGLE accepted 1.168 drafts/round versus 0.860 for
  head W1A1 and 0.055 for all-group W1A1. In the native low-bit matrix,
  W8A8 accepted 1.127/round and W4A4 only 0.092/round versus ordinary 1.168.
  Same-format default/MMA pairs had identical text and acceptance counts,
  so their measured rate differences are execution cost.
- A separate five-repetition Q4_0/Q8_0 CUDA trace confirmed both formats
  quantize live F32 activations to Q8_1. Observed one/two-token calls used
  MMVQ and a 38-token call used MMQ; no cuBLAS branch appeared. MMVQ byte
  DP4A is source-backed, not separately disassembled in that diagnostic.
  These block-scale paths are not the genuine W4A4/W8A8 contracts.
- Nsight Systems CUDA-software traces measured actual EAGLE pack and dot
  kernels on the 2080 Ti. At N=1, head packing-inclusive medians were
  31.4 microseconds for portable W1A1, 149.4 versus 434.7 microseconds for
  W8A8 DP4A versus MMA, and 152.7 versus 362.4 microseconds for W4A4 vector
  versus MMA. Head and FFN shapes, N=37/38 prefill, pair counts, same-stream
  method, and raw trace/SQLite hashes are in the experiment report. These
  short profiled runs explain cost; sealed request rates remain the
  end-to-end conclusion.

## Correctness limits and working state

Every speculative variant within each matched matrix emitted identical text
on its paired requests. Target-only differed on one stable reasoning-02 heading
capitalization prompt, so target-only rate ratios are timing observations
rather than strict lossless speedups. The server omitted generated token IDs.
Nonstreaming requests had no client time-to-first-token; host draft-call spans
do not isolate full target verification latency. The findings apply to this
single-sequence EAGLE workload, not all batch sizes or future quantizers.

The parent main gitlink remains the published, SM75-tested production llama.cpp
commit 34e21b7. Experimental low-bit and MMA commits remain published on the
user's llama.cpp fork; no parent gitlink points to an unpublished commit.
Weights, raw runs, and profiler captures remain ignored and outside Git.
The local parent main checkout is clean and pushed. The final WSL audit found
all 79 supervised jobs finished, no project server/benchmark/profiler/build
process and no compute app. GPU was idle at 0% with 855 MiB used and 10,173
MiB free. The remote production checkout and committed gitlink both matched
34e21b7; the operator closed its SSH panes. A pre-existing detached tmux
session remains untouched, with no persistent SSH connection. The remote
clone was clean at parent 11a25f5 during audit; later local commits were
documentation-only and pushed to main.

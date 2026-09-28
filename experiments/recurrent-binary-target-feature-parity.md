# Independent target-feature comparison across CPU and RTX 5080

**2026-09-28 UTC.** This diagnostic compares native raw target layer-input
taps `[2,18,33]` against an independent Hugging Face Qwen3-4B forward on
exact frozen training-prompt prefixes. Each tap is 2,560 F32 values; the
native ledger joins 7,680 values per token with token ancestry and position.
The Hugging Face source weights were rounded to F16, and sampled FFN gate
matrices plus every requested prompt embedding row matched the pinned FP16
GGUF bitwise. This is a selective source check; it does not certify every
target tensor or backend operation.

The Apple M3 Max checks used PyTorch 2.14.0, Transformers 4.57.1, eager
attention and F32 computation from F16-rounded source weights. The earlier
prose comparison is in the [CPU capture report](recurrent-binary-cpu-capture-smoke.md).
Two new checks used the existing code and reasoning CPU captures under
Flash Attention `auto`:

| Native CPU prompt | Median relative row L2 at tap 2 / 18 / 33 | Largest relative row L2 |
| --- | --- | ---: |
| Code: streaming text 01 | 0.542% / 0.397% / 0.279% | 1.359% (tap 33) |
| Reasoning: rate and work 01 | 0.501% / 0.569% / 0.490% | 2.088% (tap 33) |

The first token in each prompt again had a larger absolute difference
(`64.785` in the layer-18/33 rows with native RMS above 335) but about
0.397% relative row error. The new ignored code/reasoning reports have
SHA256
`ebb65efac81f58c6abda95f0d21ee46411267e527eab470b996674b59a5ad0a2`
and `e3f735ece2764025910a2633e932ccc44f4d1cc89c353e4ff4072814ea5f16b6`.
The CUDA-mode checker preserved the code-prompt CPU metrics exactly before
the CUDA run; four focused tests and Ruff passed in parent commit `8e8f1a2`.

One bounded **same-device** RTX 5080 comparison then reused the earlier
eight-token CUDA native capture of the prose training prompt. The independent
forward used the local F16-rounded Hugging Face weights in CUDA F16 eager
computation (`torch 2.14.0+cu130`, Transformers 4.57.6). The native CUDA
target used the pinned FP16 GGUF with Flash Attention enabled. All 32
prompt embedding rows matched bitwise. The measured feature differences
were:

| Tap layer | Median relative row L2 | Largest relative row L2 | Full-prompt RMS |
| --- | ---: | ---: | ---: |
| 2 | 0.311% | 0.789% | 0.002698 |
| 18 | 0.426% | 1.367% | 0.307267 |
| 33 | 0.385% | 1.753% | 0.313354 |

At the exceptional first row, the native layer-18/33 RMS was about 335;
the 66–74 absolute maxima corresponded to roughly 0.52% relative row L2
error there.

The GPU comparison used a full-prefix `use_cache=False` Hugging Face pass,
while the native server may split prefill into microbatches and uses a
different attention/reduction path. The F16 compute choice fits the 5080
and is recorded explicitly; it is not an F32 same-backend proof. These
nonzero differences keep target-feature numerical parity **open**. No
state/logit tolerance has been declared from them, and no training or
Q4_0 performance conclusion follows.

A second supervised CUDA/F16 forward changed only Hugging Face attention
from eager to PyTorch SDPA (commit `95b924d`, four focused tests). Median
relative row L2 errors became **0.306%, 0.434%, 0.382%** at taps 2, 18 and
33, versus 0.311%, 0.426%, 0.385% under eager. Largest rows were 0.789%,
1.388% and 1.775%, versus 0.789%, 1.367% and 1.753%. The small and mixed
changes do not resolve the native gap or isolate one cause. The SDPA report
SHA256 is `fb333b9e76238a46ebddb875bc2a351ae66f589dae8a433ba5314745197ddb60`;
its supervised process exited zero and the GPU returned idle.

The CUDA report is preserved outside Git at
`checkouts/recurrent-gpu-capture-20260928/runs/recurrent-target-feature-cuda-20260928/comparison.json`
on the registered RTX 5080 host, SHA256
`b2ee0c273b7b7a3a66abb61541ac8ee619728b43a04aabdf471fd0ad3b9c8a29`.
It records target/draft/Hugging Face shard and native capture hashes. The
supervised diagnostic exited zero and the process group stopped; final GPU
check showed 0% utilization, 1,372 MiB whole-device use and no compute app.
No development or final prompt was used.
The SDPA counterpart is preserved beside it under
`runs/recurrent-target-feature-sdpa-cuda-20260928/comparison.json`.

## Complete frozen training-prefix distribution

The full-capture diagnostic in parent commit `4327679` validated source file
hashes, all 96 frozen task/prompt owners, streamed target-feature row ancestry,
microbatch row identities and exact first-round prefixes. It selected 3,112
prefill rows (22–61 tokens per prompt) from the previously sealed 96-request
capture. The capture manifest SHA256 is
`2b2f49861c010214d2e424ad49053c829acdd6c6dafccbad721406390ed888fd`;
raw feature values SHA256 is
`d2b4602a3c6749d9bd27bea8635a5113ea66bfca469525707d09c5b5714e1c7a`.
The diagnostic used the same pinned target GGUF, D draft and local HF source
as the one-prompt CUDA check. All selected prompt embedding rows and sampled
gate weights matched the F16 GGUF. Six synthetic checks and Ruff passed.

Two supervised RTX 5080 forwards used PyTorch 2.14.0+cu130 and Transformers
4.57.6 with F16-rounded source weights and full-prefix `use_cache=False`
computation. The independent attention implementations were eager and SDPA;
the native capture used its own CUDA Flash Attention path. Values below are
relative row L2 errors against the native F32 feature rows, across all 3,112
rows per tap:

| Tap | Eager median / p99 / max | SDPA median / p99 / max |
| --- | --- | --- |
| 2 | 0.293% / 0.830% / 2.433% | 0.288% / 0.797% / 2.544% |
| 18 | 0.538% / 1.387% / 10.014% | 0.531% / 1.270% / 12.135% |
| 33 | 0.488% / 1.687% / 2.779% | 0.480% / 1.641% / 2.908% |

The tap-18 maximum in both runs is frozen training prompt
`qat-revisit-train-code-data-validation-03` at prefill position 3 (raw feature
row 26,465). Its native tap-18 row has L2 norm 58.85; the relative outlier
is not solely a near-zero denominator. This diagnostic cannot attribute the
outlier to one native operator. The source hashes in the two reports are
identical. Their SHA256 values are
`49d38214b1b59ad1597ee3111075f87fa05be9efa8ad24d91314c8f2374e93dd`
(eager) and
`09a51dfb61361b2c3f7f6cb24aa2656d21a6673fc754090808353b0314cdceca`
(SDPA). The ignored reports remain on the registered 5080 host at
`checkouts/recurrent-gpu-capture-20260928/runs/recurrent-target-feature-full96-20260928/report.json`
and the corresponding `recurrent-target-feature-full96-sdpa-20260928` run.
Both supervisors finished with exit zero, process groups stopped, and the
device returned to 0% utilization, 1,372 MiB whole-device use and no compute
app. No training or development/final prompts were used.

A read-only cross-request check grouped the same sealed native prefill rows
by exact token prefix. Among 427 pairs sharing a prefix, 103 raw feature
rows were not bitwise identical across requests; all first-token pairs were
identical, while differences began at positions 1–5. The largest relative
row L2 difference within a shared prefix was 0.473/0.801/0.821% at taps
2/18/33. This is an observed dependence on the request execution context,
not evidence that future tokens should causally affect earlier positions.
Batch geometry, numeric dispatch and capture indexing remain possible causes.
The 10–12% tap-18 outlier has no duplicate four-token prefix in this capture,
so this check does not explain it.

A bounded repeat of that one frozen training prompt used the **same native
CUDA server binary SHA256**
`0e49bbfa0514bf037dbf1c9f5dadf959e87f5c793ebd79db5bc753f97d1df8f9`
and the same pinned target, D draft and training split. All 29 prefill token
IDs, positions and prefix ancestries matched the full capture. All
**222,720 F32 feature values** (29 × 7,680), including the outlier at
position 3, matched the full capture bitwise. A one-token first attempt
captured identical prefill bytes but failed the runner's final audit because
the one-token response created no speculative draft round; its logs remain
preserved. The eight-token repeat passed its continuity and response audits,
and its manifest SHA256 is
`b5d4f5fae6b59c752742b9c45e6e770436c28fbbf1f48ba5fa3cc1cd148f6611`.
The repeat's raw feature file SHA256 is
`80b3cbea77d579c01d4ff61aa62a733892816f67c72f18baf771cea38ef85749`.
Its files remain outside Git under
`checkouts/recurrent-gpu-capture-20260928/runs/recurrent-outlier-recapture8-20260928/`.
The supervised repeat exited zero; the native server returned zero and
stopped, with no project process or compute app remaining. This rules out a
one-off native capture of the outlier under the same backend and prompt, but
does not identify which operator creates the independent-forward difference.

## Layer-0 ggml CPU operator probe

Parent commit `c798fd3` adds a standalone ggml graph for the pinned 32-token
prose prefix on Apple M3 Max CPU, one thread, with CUDA/Metal/Accelerate/BLAS
disabled. It checks bitwise HF-to-GGUF identity for the prompt embeddings,
layer-0 norm and complete Q/K/V weights. The graph measures native RMS norm
and pre-RoPE Q/K/V projections before any attention. Against a Torch F32
reference, norm RMS error was `2.65e-9` (61,283/81,920 values exact).

| Pre-RoPE projection | Native versus F32 RMS | Median relative row L2 | RMS after F16-casting normalized F32 input |
| --- | ---: | ---: | ---: |
| Q | 8.31e-5 | 0.086% | 8.27e-5 |
| K | 9.16e-5 | 0.087% | 9.11e-5 |
| V | 5.92e-5 | 0.142% | 5.88e-5 |

Two native-fixture checks, Ruff and format checks passed. The ignored
[operator report](../results/recurrent-layer0-projection-final-20260928.json)
has SHA256
`d7bd1dd9b2b9beeca550682b67b142fdf8a29f92b1a5c82b60143edd696b2d62`.
It records target, capture, HF shard, helper and CPU library hashes. The
small input-cast effect leaves an independent projection arithmetic difference
before attention. This is a CPU operand graph, not the native target's entire
CUDA graph, and it does not explain the tap-18 outlier or define a training
tolerance.

## Native tap-2 input intervention on the outlier

Parent commit `f5825e8` adds a bounded independent-forward intervention. It
reuses the sealed 96-request ownership, hash and prefill checks, then runs the
29-token outlier prompt twice on RTX 5080 in CUDA F16 eager computation. The
second forward replaces the complete input to Hugging Face decoder layer 2
with the captured native F32 tap-2 rows **cast to F16**; a scoped pre-hook is
removed after the forward. The same pinned GGUF/source embedding and sampled
weight checks pass. Four synthetic tests, Ruff and format checks passed.

| Tap and position 3 | Baseline relative row L2 | With native tap-2 input |
| --- | ---: | ---: |
| 2 | 0.324% | 0.020% |
| 18 | 10.014% | 11.659% |
| 33 | 0.835% | 0.948% |

Across the 29 rows, median tap-18 error changed from 0.611% to 0.573%.
Replacing the earlier input nearly eliminates the tap-2 input difference,
but does **not** close the position-3 tap-18 gap. The remaining F16 cast and
distinct downstream arithmetic prevent a claim of exact native causality;
the result directs the next check to the target layers after tap 2.

The ignored report remains on the registered 5080 host at
`checkouts/recurrent-gpu-capture-20260928/runs/recurrent-tap2-intervention-20260928/comparison.json`,
SHA256 `cebef815e90de17a8a5d4b597758422ef84f5bfe705e65a0032ea6a87c227a45`.
It records the capture manifest and raw feature hashes, prompt row IDs,
hardware and software versions, and per-row baseline/intervention errors.
The supervised run exited zero, its process group stopped, and the 5080
returned to 0% utilization, 1,372 MiB whole-device use and no compute app.
No training or development/final prompt ran.

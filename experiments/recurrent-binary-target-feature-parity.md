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

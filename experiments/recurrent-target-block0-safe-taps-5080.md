# Safe target block-0 operator comparison on RTX 5080

This bounded CUDA diagnostic uses the frozen 29-token **code/data-validation
training prompt**, pinned Qwen3-4B F16 target GGUF and candidate D. A
standalone llama.cpp context evaluates the same token prefix with a
block-0 graph callback. Its captured `l_out-0` is compared against the
sealed native layer-1 input from the earlier target ladder (manifest
SHA256 `72410d35fae0b1561fca0546e8e3b6e58a30506af75fd0802d10da865db6151d`).
The native helper runs on the RTX 5080 (SM120), with CUDA 13.1 and the
same pinned ggml target graph source as the prior server capture. It
does not run an optimizer or use development/final prompts.

## Capture validity

The callback itself can change CUDA execution arithmetic. The
output-only callback reproduced all **74,240/74,240** sealed layer-1 F32
values bitwise. Capturing attention norm, K normalization, V projection,
or the FFN-side tensors separately also preserved all 74,240 values.
Capturing Q normalization alone, or all taps together, yielded only
**72,329/74,240** exact values (maximum difference `0.000244140625`).
Consequently Q-normalization tensors from those runs are excluded from
server-path attribution. The four output-preserving captures are the
native comparators below; each has its own sealed file and helper hash.

## HF CUDA/F16 versus output-preserving native taps

The independent Hugging Face Qwen3Model eager forward used the identical
29-token prefix and F16 source checkpoint, with sampled GGUF weight
identity checked. Prompt embeddings match the
sealed native layer-0 input bitwise. Qwen3 target RoPE is NeoX
half-split, so K-normalized channels are compared in their original row
order (the EAGLE draft's interleaved permutation does not apply).

| Block-0 boundary | Position-3 relative row L2 | Median across 29 rows |
| --- | ---: | ---: |
| Attention RMS norm | 0.0325% | 0.0298% |
| K projection plus head RMS norm | 0.0922% | 0.0743% |
| V projection | 0.0994% | 0.1076% |
| Attention output plus residual (`ffn_inp`) | 0.2547% | 0.2269% |
| FFN RMS norm | 0.2964% | 0.3002% |
| FFN output | 0.3496% | 0.3117% |
| Complete block output | 0.2721% | 0.2401% |

For a same-input ablation, the captured native attention-norm F32 tensor
was cast to F16 and supplied directly to HF K/V projections. At position
3, K-relative error changed only **0.0922% → 0.0916%**, and V changed
**0.0994% → 0.0934%**. The native-norm-to-F16 cast itself has 0.0199%
relative row error there. Thus the ordinary HF norm-input difference
does not explain the K/V discrepancy under this F16 same-input test;
projection/backend arithmetic or its input cast remains material. The
growth from K/V to the attention residual is an observation, not an
attribution to a specific Q or Flash Attention instruction, because a
faithful Q-normalization tap was unavailable.

The supervised native capture runs and HF comparison exited or stopped
with process groups released; final RTX 5080 utilization was 0% with
1,372 MiB whole-device baseline use and no project compute process.
The main ignored HF machine report is at
`checkouts/target-block0-operator-20260928/runs/target-block0-safe-hf-c-20260928/comparison.json`
on the registered WSL host, SHA256
`e2012372501f4f9e0595465e4e69e7656330917bdf8c0547c22bd521bb1b9b43`.
Output-preserving native report SHA256 values are
`ec5055512864a8b7093070dbe097a904ba995c3608f98d8c9f185e00db9e6e88`
(attention norm), `5348621e9178b6fbe1256adbe85a25440eb7568f55c434c9351f2962b6e24160`
(K norm), `f8e3fe0622b7cbb574e6c75d32cb018af82cd53e9686c97ff6fde7fa8f4bbe7e`
(V), and `14dd0443a1559c03d307ec4058e2316d463cbb2c911ddc9c8e4ebc245cebb95c`
(FFN). The intrusive Q control is retained with SHA256
`01b54d3748c3bea39b15a527d78a30346f3381e65bf69cad69fffaa43a011e7d`.
The reports record model, helper, CUDA library, compatibility-header
build, source and capture hashes; the failed setup attempts remain
preserved and excluded from the numerical comparison.

This is one frozen training prefix on RTX 5080 CUDA/F16, not SM75
performance, general target-feature parity, a numerical training gate,
or Q4_0 acceptance/latency/throughput evidence. The earlier 96-request
capture remains training-ineligible; exact versus predeclared tolerance
and any all-body optimizer budget remain user-owned.

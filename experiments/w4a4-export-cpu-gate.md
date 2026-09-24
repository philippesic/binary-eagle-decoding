# W4A4 EAGLE export and CPU gate

**Status:** Nine-linear converter, GGUF readback, and numerical CPU gate passed locally on 2026-09-24. This is a format and reference result; there is no W4A4 runtime loader, CUDA operator, SM75 dispatch, or performance measurement.

**Code:** published llama.cpp fork branch `feat/w4a4-eagle-export`, commit `f0cf0cfc198fff6f5a9a26c490e932ed4691d7ab`, based on W8A8 CPU/loader commit `492818599`. The parent repository gitlink remains unchanged for this report.

## Format and arithmetic

`--w4a4-eagle` exports all nine EAGLE-3 draft linears: fusion, Q/K/V/O attention, gate/up/down FFN, and output head. It applies the existing source-to-GGUF mapping, including Q/K RoPE row permutation, before quantization. Each BF16 row is converted to F32, divided by its F32 absmax/7 scale, rounded to nearest even, and clamped to signed codes `[-7,7]`. A zero row gets zero codes and a zero scale. Two two's-complement signed nibbles share each byte: even logical K in the low nibble, odd K in the high nibble. An odd final K has a zero high padding nibble; code `-8` is forbidden. `*.w4a4_packed` has GGUF raw type I8 and physical shape `[output rows, ceil(logical K/2)]`; `*.w4a4_scale` has F32 shape `[output rows]`. No dense `*.weight` shadow is stored for these linears.

Version 1 metadata under `eagle3.w4a4.*` declares all four groups and nine canonical weight names; signed packed I4 weight and activation codes; `[-7,7]`, nibble order, encoding, padding, F32 row/token absmax/7 scales, nearest-even rounding, zero codes, I32 accumulation, and F32 dot → weight scale → activation scale → optional bias. Each tensor has explicit logical K and packed/scale tensor names. The converter rejects non-EAGLE or incomplete one-layer checkpoints, big-endian GGUF, and simultaneous W1A1/W8A8 flags.

The CPU reference packs activations per token using the same quantizer, unpacks both operands, forms an exact I32 dot, applies F32 scales in the declared order, adds optional F32 bias, and casts to the input dtype. This matches the prior [PyTorch W4A4 simulation](pytorch-int4-int8-cuda-acceptance.md) quantization rule. The prior simulation uses FP32 `F.linear` on integer-valued floats, so long sums need a stated model-level tolerance rather than a general bitwise-equality assumption.

## Local evidence

- Focused tests from the isolated llama.cpp checkout: `results/convert-env/bin/python -m unittest tests/test_w4a4_eagle_conversion.py tests/test_w8a8_eagle_conversion.py tests/test_w1a1_eagle_conversion.py`: **12/12 passed**. W4A4 cases cover nine names, signed nibbles, odd-K zero padding, forbidden `-8`, zero rows/tokens, half-way ties, GGUF packed I8/F32 roundtrip, invalid inputs, and a small exact parity case against the FP32 code simulation.
- `ruff check conversion/w4a4.py tests/test_w4a4_eagle_conversion.py`, Python `compileall` on changed Python files, and `git diff --cached --check`: passed. Broader preexisting `conversion/llama.py` style findings were left outside this stage.
- Full local export used the AngelSlim EAGLE safetensors file SHA256 `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e` and the pinned Qwen3-4B target config/tokenizer. The ignored GGUF is `models/gguf/Qwen3-4B-eagle3-w4a4-export-smoke.gguf`, **115,613,408 bytes**, SHA256 `0471dd2a1ac7628ae97018dad5d24aaf08cbfc6a258758a975d4e60b0d40beed`. Readback found 23 tensors: nine packed I8 matrices, nine F32 row-scale arrays, four other F32 tensors, and I64 `d2t`. All nine packed matrices and scales matched a fresh quantization of source safetensors exactly, including Q/K permutation. No dense shadow was present. For each matrix, a real-weight two-row CPU output with a full-width BF16 input matched the prior FP32 code simulation exactly after BF16 cast. Source reduction widths were 2,560, 4,096, 5,120, 7,680, and 9,728. These checks ran on an Apple M3 Max CPU with PyTorch 2.11.0; they are not SM75 checks.

The conversion command, run from the isolated llama.cpp checkout, was:

```sh
/Users/pippo/github/binary-eagle-decoding/results/convert-env/bin/python convert_hf_to_gguf.py /Users/pippo/github/binary-eagle-decoding/models/hf/Qwen3-4B_eagle3 --target-model-dir /Users/pippo/github/binary-eagle-decoding/models/hf/Qwen3-4B --outtype f16 --w4a4-eagle --outfile /Users/pippo/github/binary-eagle-decoding/models/gguf/Qwen3-4B-eagle3-w4a4-export-smoke.gguf
```

The local ignored audit script is `results/w4a4_export_audit.py`; it verifies every version-1 metadata value, tensor inventory, shape, logical K, packed bytes, scales, and sampled numerical output.

## Remaining runtime gate

Implement a strict EAGLE loader/graph bridge for `eagle3.w4a4` and a CPU operator, then a CUDA path whose actual SM75 instructions and operands are checked. Verify all nine outputs and draft logits against the reference, accepted counts and generated text against the frozen prompt suite, actual dispatch/SASS, and packing-inclusive timings before filling the native W4A4 benchmark row. The prior RTX 5080 simulation retained only **0.2882 accepted drafts/round** versus **2.3166** for ordinary BF16 EAGLE; it does not establish a native speed result.

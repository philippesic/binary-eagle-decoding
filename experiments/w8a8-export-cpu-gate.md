# W8A8 EAGLE export and CPU gate

**Status:** converter and CPU numerical gate passed locally on 2026-09-24. No runtime loader, GGML operator, CUDA execution, or RTX 2080 Ti timing is claimed.

**Code:** llama.cpp fork branch `feat/w8a8-eagle-export`, commit `bad469841`; the parent repository gitlink remains at `8d2b18a9c9b3a42927404a91799a823c758c09b6` pending a reviewed runtime integration.

## Format and arithmetic

The opt-in `--w8a8-eagle` converter flag writes all nine EAGLE draft linears. Each original BF16 weight row becomes signed I8 codes in `[-127,127]` and one F32 absmax/127 row scale. The converter runs the existing EAGLE source-to-GGUF mapping first, including Q/K RoPE row permutation. `*.w8a8_codes` has GGUF type I8 and logical matrix shape `[output rows, input K]`; `*.w8a8_scale` has GGUF type F32 and one value per output row. The ordinary `*.weight` shadow is omitted. Metadata under `eagle3.w8a8.*` versions the format, enumerates the nine canonical weight names and tensor names, and declares signed I8 operands, F32 row/token scales, nearest-even rounding, zero-vector codes, I32 accumulation, and output scale order.

The CPU reference converts each input vector to F32, uses one F32 absmax/127 scale, rounds to nearest even, and maps a zero vector to zero codes. It computes exact I32 dots, converts each to F32, multiplies the weight scale and then activation scale, adds optional F32 bias, and casts to the original input dtype. This preserves the prior PyTorch [W8A8 simulation](pytorch-int4-int8-cuda-acceptance.md) quantization and scaling rule. The prior FP32 `F.linear` accumulation can differ from exact I32 dots when a long sum loses integer precision in F32; future model-level parity must use a stated tolerance and compare accepted tokens, rather than assume bitwise equality.

This is a new opt-in GGUF format. The pinned llama.cpp loader has no W8A8 EAGLE graph/operator for it; the artifact is **not loadable for inference yet**. Existing W1A1 and Q8_0 GGUF semantics remain unchanged. The converter rejects non-EAGLE checkpoints, missing linears, big-endian output, or simultaneous W1A1 flags.

## Local evidence

- Focused CPU tests: `results/convert-env/bin/python -m unittest tests/test_w8a8_eagle_conversion.py tests/test_w1a1_eagle_conversion.py` from the llama.cpp checkout: **7/7 passed**. These cover nine-name enumeration, signed I8/F32 GGUF roundtrip, zero vectors, half-way ties, numerical agreement with the earlier FP32 fake-quant formula on a small exact case, and invalid input rejection.
- `ruff check conversion/w8a8.py tests/test_w8a8_eagle_conversion.py`: passed. `compileall` on the converter, reference, and test files: passed. `git diff --check`: passed. Existing broader `conversion/llama.py` style findings predate this change and were not rewritten.
- Full local conversion from `models/hf/Qwen3-4B_eagle3` and the pinned Qwen3-4B target config completed with `--outtype f16 --w8a8-eagle`. The ignored artifact is `models/gguf/Qwen3-4B-eagle3-w8a8-export-smoke.gguf`, 218.8 MB as reported by the writer, SHA256 `48d8c517253ee24278412efc18eaf38340d6e9eede4fc819f64ab268dab590d8`. GGUF readback found 23 tensors: nine I8 code tensors, nine F32 scale tensors, four other F32 tensors, and the I64 `d2t` map. All nine code and scale arrays exactly matched a fresh CPU quantization of the safetensors weights after the Q/K row permutation; no dense shadow was present.

The equivalent conversion command, run from the parent repository root with the feature checkout at `third_party/llama.cpp`, is:

```sh
results/convert-env/bin/python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B_eagle3 --target-model-dir models/hf/Qwen3-4B --outtype f16 --w8a8-eagle --outfile models/gguf/Qwen3-4B-eagle3-w8a8-export-smoke.gguf
```

## Handoff and remaining gate

The next feature stage is a strict EAGLE loader/graph bridge for the versioned W8A8 tensors, then a CPU or CUDA implementation of dynamic activation quantization and signed I8 dot with F32 scale application. The repeated one-sequence draft step is one-token matrix-vector execution, so the first operator must handle that shape well and must log its actual dispatch. Before any timing claim, compare all nine outputs and logits against the CPU reference, check greedy token/acceptance behavior, and verify SM75 integer instructions and packing-inclusive cost. Keep the native W8A8 benchmark row unmeasured until those gates pass.

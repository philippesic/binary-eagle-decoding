# Native Q4_0 and Q8_0 EAGLE draft artifact preparation

**Status:** local conversion and GGUF tensor-type audit only. No RTX 2080 Ti
load, CUDA dispatch, acceptance, or timing result exists. These are llama.cpp
**weight-quantized** draft formats, not the earlier simulated W4A4/W8A8
operand quantizers. Activation format and CUDA kernels must be verified on the
benchmark host before precision or speed claims.

The source is the pinned ordinary EAGLE-3 FP16 GGUF generated from
`AngelSlim/Qwen3-4B_eagle3@fd331e59626c8e95c392381a16ee59d518727fbb`.
The llama.cpp checkout was `92bc70602e13214d6db94c007894261b51f5f36c`.
The CPU `llama-quantize` binary SHA256 was
`6b9a42487e253bf240ec39059baa2e6f485272ed7556d693d54f7c9e0570e70e`.
Exact commands from the project root:

```sh
cmake --build build/llama-cpu --parallel 8 --target llama-quantize
build/llama-cpu/bin/llama-quantize --pure models/gguf/Qwen3-4B-eagle3-f16.gguf models/gguf/Qwen3-4B-eagle3-q4_0.gguf Q4_0 8
build/llama-cpu/bin/llama-quantize --pure models/gguf/Qwen3-4B-eagle3-f16.gguf models/gguf/Qwen3-4B-eagle3-q8_0.gguf Q8_0 8
```

| Draft GGUF | Bytes | SHA256 | Tensor-type audit |
| --- | ---: | --- | --- |
| Source FP16 | 442,700,800 | `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1` | 9 F16 linears, 4 F32 other tensors, 1 I64 mapping |
| Q4_0 | 128,988,160 | `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280` | 9 Q4_0 linears, 4 F32 other tensors, 1 I64 mapping |
| Q8_0 | 238,105,600 | `29ee91f09971555458cf420461fdfeeeabc9236f8a2c20662b7333a980a49507` | 9 Q8_0 linears, 4 F32 other tensors, 1 I64 mapping |

The nine quantized linears are `fc.weight`, Q/K/V/attention output, FFN
gate/up/down, and `output.weight`. The 4 F32 tensors are normalization weights;
`d2t` remains I64. The `gguf.GGUFReader` enumeration audit confirmed actual
tensor types rather than inferring them from filenames. The ignored local
artifacts are under `models/gguf/` and
`results/quantization-prep-20260924/`. The Q4_0 and Q8_0 conversion logs have
SHA256 values `66d1abbc1e057fcaa23afe85ebd79addbeadf93a88606513fd21029b21b8c39c`
and `3520b681a537cb50e9d2ba79e6fb830bef88dda35926840aa3a6694e7c624469`.

A two-token CPU `llama-cli` smoke with the pinned FP16 target,
`--spec-type draft-eagle3`, context 512, and each quantized draft exited zero.
This only checks that the local model pair loads and runs; it does not validate
CUDA dispatch, output parity, acceptance, or speed. The exact command used for
each draft was:

```sh
build/llama-cpu/bin/llama-cli -m models/gguf/Qwen3-4B-f16.gguf -md models/gguf/Qwen3-4B-eagle3-q4_0.gguf --spec-type draft-eagle3 --spec-draft-n-max 2 --ctx-size 512 --n-gpu-layers 0 --spec-draft-ngl 0 --no-warmup --single-turn -p 'Say hello.' -n 2 -lv 2
```

The Q8_0 run differed only in its draft GGUF path. CPU smoke log SHA256 values
were `06c97c3d4a64bd0f554828514c5ac83854b356af2094cd221a298c5aa326d776`
and `86d84835d64a561a029773d8943e15ad77bc2fea649c4533bd9eaf1daebe7d55`.

For the 2080 Ti comparison, copy or reproduce these artifacts under its
project workdir; hash again there. Use the same target GGUF and server settings
as every other draft variant in that track. Run a loader/correctness smoke,
record actual CUDA kernel and activation precision, then five alternating
repetitions with acceptance and end-to-end timing. Report these rows as
`Q4_0 weight-only` and `Q8_0 weight-only`, separately from genuine W4A4/W8A8
if those kernels are later implemented.

# Native packed W1A1 path: staged implementation

This design targets the pinned llama.cpp submodule revision
`6e60f35608ec6918b44a9839c0c433687165f086`. It is an implementation
plan, not a native result. The first scope is the EAGLE-3 drafter vocabulary
head; later attention/FFN groups depend on acceptance and packing-inclusive
timing. `kernels/binary_reference.py` is the CPU bit-order and integer-dot
contract. The Python BF16 fake-binary wrapper uses different rounding points,
so native acceptance must be remeasured under a matching numerical contract.

## Packed representation and arithmetic

- Map nonnegative values, including both signed zeros, to bit 1; negative
  values to bit 0. Reject nonfinite weights at export. Store consecutive K
  features in little-bit-order uint32 words, with bit `k % 32` of word
  `k // 32`. The packed weight tensor has GGML dimensions
  `[ceil(K/32), output_rows]`; store it in GGUF as ordinary I32 bit patterns.
- Store one F32 mean-absolute weight scale per entire logical output row in a
  companion GGUF tensor. The pinned exporter casts BF16 weights to F32 and
  uses PyTorch's F32 row mean, then writes the F32 result. Record format
  version, selected tensor names,
  logical K, bit order, sign rule, scale precision, and output arithmetic in
  metadata. Keep K separate from padded word count.
- For each activation token vector, accumulate absolute F32 magnitudes in
  F64, divide by logical K, round once to an F32 mean-absolute scale, and pack
  its signs once. The standalone prototype's earlier F32 reduction can differ
  near rounding boundaries, so it is a performance prototype rather than the
  exact GGML numerical contract. Integer dot is exactly
  `K - 2*sum(popcount((weight_word XOR activation_word) & valid_bits))`.
  Padding bits cannot contribute to the dot or scale. Apply row and token
  scales in an explicit order using F32 arithmetic and return F32 logits to
  the surrounding GGML graph. Preserve any bias separately.
- An all-zero row has zero scale and therefore zero pre-bias output, while its
  signs still follow `sign(0)=+1`. Test both +0 and -0. Native kernel results
  must agree with the packed CPU reference for integer dots exactly and for
  scaled outputs within a documented floating tolerance.

## Minimal GGML and EAGLE integration

Add a dedicated operation resembling
`ggml_w1a1_mul_mat(ctx, packed_weight, weight_scales, activations, logical_k)`
with F32 activation input and F32 output. Keep the packed representation in
ordinary GGUF I32/F32 tensors; do not overload `GGML_OP_MUL_MAT` or reuse
`GGML_TYPE_Q1_0`. The latter has 128-element block scales and ordinary
activation dispatch, so it does not implement this W1A1 rule.

| Area | Narrow edit surface |
| --- | --- |
| GGML op | `ggml/include/ggml.h`, `ggml/src/ggml.c`: append op, constructor, name/symbol/count, explicit unsupported backward handling. |
| CPU fallback | `ggml/src/ggml-cpu/ggml-cpu.c` and backend support predicate: correct packed reference execution. |
| CUDA | New `ggml/src/ggml-cuda/w1a1.cu/.cuh`; register forward dispatch and `supports_op` in `ggml-cuda.cu`. Use backend scratch/stream APIs for activation packing and the XOR/POPCOUNT kernel. |
| EAGLE graph | `src/models/models.h` and `src/models/eagle3.cpp`: load packed head/scales and branch at the drafter-owned head projection. Preserve `d2t` handling and target verifier. |
| GGUF conversion | `conversion/llama.py` plus narrow option plumbing: transform layouts first, then export I32 packed words and F32 scales explicitly with metadata. |
| Loader | `src/llama-arch.h/.cpp` and model loader support probing: count companion tensors and probe them against the new operation. |
| Tests | `tests/test-backend-ops.cpp`, converter round-trip checks, and captured real EAGLE activations. |

The converter's ordinary path converts non-F16/F32 tensors to F32. Packed I32
words must bypass that path or high bits may be lost. Every companion tensor
must be named, loaded, and counted; marking it `GGML_OP_NONE` may cause the
loader to skip it as unused. `create_tensor()` buffer selection probes the
operation associated with a tensor, so packed weights must not be probed as
ordinary `GGML_OP_MUL_MAT` weights. Reject missing companions when metadata
selects the binary head; do not silently fall back to the target head. Reject
LoRA on the selected projection initially so the binary branch cannot discard
an adapter.

Fusion and Q/K/V have direct `build_lora_mm` call sites in `eagle3.cpp`.
Attention output and FFN projections sit inside shared graph helpers. Add
scoped EAGLE dispatch there only after head correctness and timing; avoid
changing other architectures' normal projections. When multiple projections
share one activation, reuse its sign/scale packing rather than paying for it
once per matrix.

## Stopping boundaries and measurements

1. **Numerical contract:** packed CPU and CUDA dots agree for K tails, dirty
   padding, signed zeros, all-zero vectors, scales, layouts, and sampled rows
   from every measured EAGLE K. Compare with the existing BF16 fake-binary
   simulation and explicitly quantify rounding differences.
2. **Kernel prototype:** benchmark both already-packed and packing-inclusive
   latency on the observed token counts and nine layer shapes. Include launch,
   scratch allocation, scale reduction, and CUDA graph replay effects. Stop
   optimizing a shape when packing-inclusive savings disappear.
3. **Head integration:** GGUF round trip preserves packed bit patterns and
   scales; no dense shadow head remains resident; `d2t` and target outputs
   remain correct; captured head inputs produce matched native/reference
   outputs. Then remeasure draft acceptance under the native numerical path.
4. **End-to-end comparison:** run target-only, ordinary EAGLE, and native W1A1
   with matched target/runtime settings. Report acceptance, emitted tokens,
   draft/verification time, decode and request throughput, memory, and raw
   artifacts. A result on RTX 5080 is a 5080 result. RTX 2080 Ti execution is
   required for SM75 binary speed claims; its host address remains missing.

Portable CUDA `__popc` can establish a correct packed path without relying on
binary tensor-core instructions. A later SM75 instruction probe must confirm
the exact binary MMA form and emitted code on the Turing host. Compilation
for SM75 or execution on RTX 5080 is not SM75 performance evidence.

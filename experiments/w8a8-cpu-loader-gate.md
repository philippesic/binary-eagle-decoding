# W8A8 EAGLE loader and CPU execution gate

**Status:** passed on local Apple M3 Max CPU, 2026-09-24. No CUDA/SM75 result or speed comparison is claimed.

**Code:** published llama.cpp fork branch `feat/w8a8-eagle-export`, commit `492818599` atop converter commit `bad469841`. The parent repository gitlink is deliberately unchanged at `8d2b18a`; this report is a separate parent commit.

## Implemented path

The EAGLE loader accepts the versioned `eagle3.w8a8` format only when its four declared groups equal fusion, attention, FFN, and head, and its nine tensor names exactly match the eligible linears. It checks each I8 code tensor and F32 row-scale tensor for presence, shape, type, per-tensor logical K/name metadata, and absence of dense or W1A1 shadows. Missing metadata with W8A8 tensors, incomplete coverage, or simultaneous W1A1 metadata is rejected. The graph routes all nine linears to `GGML_OP_W8A8_MUL_MAT`; it rejects draft LoRA for these operators and preserves ordinary and W1A1 routes.

The GGML CPU operator accepts I8 codes `[K,M]`, F32 row scales `[M]`, and F32 activations `[K,N]`. It computes one F32 absmax/127 activation scale per token, nearest-even signed I8 codes in `[-127,127]`, zero codes for a zero vector, an I32 dot, then F32 weight-scale and activation-scale multiplication. A strided activation view is supported. This matches the export contract's quantization and scaling order; the prior PyTorch simulation's FP32 floating matmul can still differ from an exact I32 accumulation on long rows. The backend test compares to an independent scalar oracle, including ties, zeros, multiple tokens, and a strided input.

## Checks

- CPU-only CMake configured and built `test-backend-ops`, `test-w8a8-eagle-load`, and `llama-cli` with `GGML_METAL=OFF`, `GGML_CUDA=OFF`, `LLAMA_BUILD_TESTS=ON`. Hardware was Apple M3 Max. `git diff --cached --check` passed before the submodule commit.
- `build/w8a8-cpu/bin/test-backend-ops -b CPU -o 'W8A8.*' -p 'K=8'`: **1/1 passed**.
- `test-w8a8-eagle-load --accept` loaded the complete [W8A8 export](w8a8-export-cpu-gate.md), including all nine pairs. The same binary loaded ordinary FP16 and all-nine W1A1 drafts. Two one-byte GGUF mutations were rejected: `nearest_even` changed to an unsupported rounding rule, and `fc.w8a8_codes` changed to a missing tensor name. The loader reported respectively “unsupported or incomplete all-linear coverage” and “metadata and all nine code/scale tensor pairs must agree.”
- CPU `llama-cli` with the pinned FP16 Qwen3-4B target and W8A8 draft, `--spec-type draft-eagle3 --spec-draft-n-max 5 --spec-draft-p-min 0 --predict 8 --single-turn --fit off --n-gpu-layers 0 --spec-draft-ngl 0`, exited 0. The EAGLE statistics recorded four draft-generation calls, 16 proposed draft tokens, and two accepted tokens. Ordinary FP16 and all-nine W1A1 drafts also exited 0 in the same eight-token smoke setup. These are execution checks, not comparative performance measurements. Ignored logs are under local `results/w8a8-loader-negative/`.

## Next gate

CUDA currently has no `GGML_OP_W8A8_MUL_MAT` implementation. A CUDA build may route this op to CPU, but that behavior has not been checked and would **not** constitute a native SM75 W8A8 benchmark. The next stage needs a CUDA activation pack and integer dot for the one-token EAGLE decode shape, plus multi-token encoder/reconciliation shapes, with explicit GPU dispatch evidence and numerical comparison against the CPU reference. Then validate actual SM75 execution, logits/acceptance, target text parity, and packing-inclusive timing under the frozen benchmark protocol. Keep the native W8A8 suite row unmeasured until those checks pass.

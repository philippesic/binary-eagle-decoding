# W4A4 EAGLE loader and CPU execution gate

**Status:** strict all-nine loader, graph routing, and CPU numerical operator passed local checks on 2026-09-24. This is an Apple M3 Max CPU result, not a CUDA/SM75 or performance result.

**Code:** published llama.cpp fork branch `feat/w4a4-eagle-export`, commit `edd3615561519f7b1537b4b215d5c58eb822e7f5` atop the [W4A4 converter and export gate](w4a4-export-cpu-gate.md) commit `f0cf0cfc198fff6f5a9a26c490e932ed4691d7ab`. This report makes no parent gitlink update.

## Implemented path

The EAGLE loader accepts `eagle3.w4a4.version = 1` only with the four declared groups and exact nine canonical linears. It validates the signed packed I4/F32 scale metadata, per-tensor logical K and tensor names, physical I8 `[ceil(K/2), M]` and F32 `[M]` shapes, and absence of dense, W1A1, or W8A8 shadows. Missing metadata, incomplete coverage, or mixed quantization metadata is rejected. Before accepting the model it scans every packed weight byte for forbidden `-8` in either nibble and nonzero high padding when K is odd; it also checks row scales for nonfinite or negative values. All nine EAGLE graph sites route to `GGML_OP_W4A4_MUL_MAT`; draft LoRA is rejected for this operator.

The GGML CPU operator dynamically packs F32 activation values into signed I4 codes `[-7,7]`, using one F32 absmax/7 scale per token, nearest-even rounding, and zero codes for a zero token. It consumes packed weight and activation bytes, computes exact I32 dots, converts to F32, and applies weight then activation scale. The graph retains its existing optional bias path. A strided activation view is supported. As in the export gate, this matches the prior PyTorch quantizer; long FP32 `F.linear` reductions in the simulation need a stated tolerance because the native dot accumulates exactly in I32.

## Checks

- CPU-only CMake configured and built `test-backend-ops`, `test-w4a4-eagle-load`, and `llama-cli` with `GGML_METAL=OFF`, `GGML_CUDA=OFF`, `LLAMA_BUILD_TESTS=ON`. Hardware: Apple M3 Max. `git diff --cached --check` passed before commit.
- `build/w4a4-cpu/bin/test-backend-ops -b CPU -o 'W4A4.*'`: **2/2 passed**. The independent scalar oracle checks odd K=9, ties, both signs, zero token, multiple tokens, and a strided view; a second case checks the maximal EAGLE reduction K=9,728 with two tokens and exact I32 accumulation. The W8A8 backend regression case also passed 1/1.
- `test-w4a4-eagle-load --accept` loaded the full 23-tensor [W4A4 export](w4a4-export-cpu-gate.md), including all nine packed/scale pairs. The same binary loaded ordinary FP16, all-nine W1A1, and W8A8 EAGLE drafts. Five one-byte or same-length GGUF mutations were rejected at load: unsupported rounding, an altered coverage name, a missing packed tensor name, and forbidden `-8` in the low or high nibble. The error messages identified the corresponding contract failures. The source model's nine K values are even; the odd-K padding rule was checked by the focused Python packing test and native odd-K backend case, but not by a mutated full model.
- Eight-token CPU `llama-cli` smoke with the pinned FP16 Qwen3-4B target, W4A4 draft, `--spec-type draft-eagle3 --spec-draft-n-max 5 --spec-draft-p-min 0 --single-turn --fit off --n-gpu-layers 0 --spec-draft-ngl 0`, exited 0. Verbose logs confirm all-nine W4A4 load, draft-eagle3 initialization, six draft generations, 20 proposed draft tokens, and zero accepted. This is an execution check on one short prompt, not a quality or speed estimate. Ignored logs and mutated GGUFs are under local `results/w4a4-loader-negative/`.
- Focused W4A4 Python converter/reference tests passed 5/5 after the runtime change; `git diff --check` passed.

## Next gate

Implement and validate a CUDA operator for packed signed I4 weight and activation codes on SM75. Confirm actual GPU dispatch and executed SASS for the one-token decode and multi-token shapes, compare all nine outputs/logits and accepted drafts against the CPU reference under the frozen prompt suite, then measure packing-inclusive and end-to-end timing. The existing Q4_0/Q8_1 control executes different quantization and arithmetic and cannot stand in for this W4A4 contract. Keep native W4A4 benchmark cells unmeasured until those gates pass. This branch adds a GGML op and CPU implementation; integrating it with concurrent W8A8 CUDA changes may require a small merge in shared GGML/loader files.

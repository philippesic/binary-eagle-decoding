# Native packed EAGLE head: GGUF and local runtime check

**Date:** 2026-09-24. **Status:** conversion/CPU-path check on Apple M3 Max;
GGML CUDA execution and end-to-end performance are unmeasured.

The published AngelSlim EAGLE-3 BF16 draft revision is
`fd331e59626c8e95c392381a16ee59d518727fbb`. llama.cpp submodule commit
`d9ab59ce19bc5206436a07789d6ca66dbc79ddad` (pushed to the user's fork
branch `w1a1-integrated`) contains the dedicated CPU/CUDA GGML operation,
strict EAGLE packed-head loader, and opt-in exporter. The exporter command was:

```sh
results/convert-env/bin/python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B_eagle3 --target-model-dir models/hf/Qwen3-4B --outtype f16 --w1a1-eagle-head --outfile models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf
```

The resulting ignored file has 289,229,312 bytes and SHA256
`6250363f5fdb70fcb3113be90cca8755e916ac0da533a76e340335aa418c16ca`.
Conversion log SHA256 is
`c4e8e9c4afdaa725467f7c129c362c967756e132b83d5c7652fec477236c9ff3`.
The GGUF contains I32 `output.w1a1_packed` with shape `[80, 32000]` in GGML
dimension order and F32 `output.w1a1_scale` `[32000]`; the ordinary dense
`output.weight` is absent. It retains `d2t` and the other draft tensors.
Versioned metadata records logical K=2560, little-bit-order signs, zero as
positive, and F32 mean-absolute scaling/arithmetic.

An independent full-head readback compared **all 32,000 rows** with the pinned
BF16 `lm_head.weight`: zero packed-sign mismatches and zero F32 scale
differences. The converter's two focused tests pass. The integrated Metal
`llama-cli` build completed; a local mixed Metal target/CPU drafter smoke with
`--spec-type draft-eagle3 --spec-draft-n-max 5 --spec-draft-ngl 0` loaded the
packed file and generated eight greedy tokens. Its raw log SHA256 is
`31da7fb7d725459df28138218d12dcf864e84699b2602d1bff997289e4b9a363`.
The worker's CPU-only speculative smoke also logged explicit packed-head
selection and exited successfully.

The integrated submodule's `test-backend-ops test -b CPU -o W1A1_MUL_MAT`
passed all five cases on M3 Max after both bridge and CUDA-source commits were
combined: K=31/32/33/2560 plus a strided K=33 input. Raw test-log SHA256 is
`baa8484d4d04ec810a94cd68d5c41142a079ff0a19060362eead8b6afc2bfb22`.

The packed head uses an integer XOR/popcount dot and F32 scales, whereas the
earlier PyTorch fake-binary acceptance sweep rounded at BF16 intermediate
stages. That sweep's 1.677 accepted drafts/round must not be assigned to this
native path. Next tests are GGML CUDA compilation/backend correctness on the
5080, native-contract acceptance, and paired full request/decode timing with
the ordinary FP16 GGUF anchor.

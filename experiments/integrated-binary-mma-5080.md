# Integrated W1A1 binary-MMA candidate: 5080 proxy validation

**Date:** 2026-09-24. **Scope:** an opt-in candidate on the user's llama.cpp
fork branch `feat/w1a1-sm75-mma` at
`9bb01a682ed4ba5e506870c8a38338589830b164`, based on the pinned
production revision `92bc70602e13214d6db94c007894261b51f5f36c`.
The parent repository's production gitlink and the measured end-to-end
benchmark binary were unchanged. This is numerical and compile evidence for
the binary Tensor Core alternative, not a Turing runtime or speed result.

The candidate retains the original packed activation/sign/scale kernel and
portable XOR/`__popc` path as the default. Setting
`GGML_CUDA_W1A1_MMA=1` selects a one-warp
`m8n8k128` XOR/POPCOUNT matrix instruction after activation packing. The
kernel masks both operands' final K word, converts the integer mismatch count
using logical K, then applies the existing F32 row and activation scales in
the same order and stores token-major output. Unsupported builds/devices and
invalid selector values fail explicitly. A separate test commit adds full
8×8, partial and multi-tile 9×10 with a dirty K=129 tail, and 320×1 with
the real head's K=2560 to the independent scalar-reference backend suite.

The isolated CUDA 13.1 SM120a build on RTX 5080 completed 358/358 steps,
producing `llama-server` and `test-backend-ops`. The default portable and
opt-in MMA runs each passed **8/8** scalar-reference W1A1 cases. Their logs
confirmed distinct portable XOR/POPCOUNT and binary-MMA dispatch. A separate
packed-head server request with MMA selected reproduced all **85** generated
token IDs and the `stop` reason of the sealed production portable
`prose-04` response. This one-request result is a functional smoke, not a
timed comparison or broad output-equivalence claim.

An isolated SM75 CUDA library build completed 157/157 steps and linked with
`w1a1.cu.o` compiled for `compute_75/sm_75`. The SM75 image was inspected
without executing it on the 5080. SASS in the candidate
`libggml-cuda.so.0.25.1` contains the exact
`BMMA.88128.XOR.POPC` mnemonic. The full disassembly and a compact excerpt
are preserved in remote raw artifacts. No SM75 binary was executed.

Six supervised build/test/smoke/inspection jobs all exited zero. The ignored
remote artifact directory is
`/home/philip/binary-eagle-decoding/results/binary-mma-sm120-20260924/`;
its `artifact-manifest.json` SHA256 is
`f0a578fc6bb4324f7a37387a202a70bdceed2dbd16497fa944dc1868a6ae62e7`.
The manifest preserves 25 file hashes, exact commands and environment, raw
backend/server responses, supervisor states, the full SASS dump and excerpt,
and a cleanup snapshot (SHA256
`d409d2ead8821b4dd25db6a883f90004309eb2530eb2858973b1acd5d2c0beb2`).
The full SM75 SASS dump is
`results/binary-mma-sm75-20260924/sm75-sass-full.txt`, SHA256
`d86b73b9507d28c7ff1ae270189b5d761e3ada08237b1745c392a2fa8c495b37`.
The remote llama.cpp checkout was restored clean to production `92bc706`.
Repeated final RTX 5080 samples showed 3,040 MiB used / 12,938 MiB free /
0% utilization, with no WSL project process. Both SSH connections and the
tmux session were closed.

**Limit:** RTX 5080 execution validates the candidate's native path on
SM120a only. The compile-only SM75 image cannot establish instruction runtime
correctness, packing-inclusive timing, or end-to-end benefit on the RTX 2080
Ti. The current 2080 Ti host address is absent from the shared host registry.

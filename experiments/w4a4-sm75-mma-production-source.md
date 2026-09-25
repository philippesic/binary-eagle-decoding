# Opt-in W4A4 SM75 Tensor Core source gate

**Status:** opt-in production CUDA source published on 2026-09-24. It has not been compiled or run on a GPU yet; no live EAGLE Tensor Core correctness or timing claim is made.

**Code:** llama.cpp fork branch `feat/w4a4-sm75-mma`, commit `eaa7fb18d42332dddaa0fbe0950f92d298c06e13`, based on the published combined W8A8/W4A4 revision `d0724427b61f6ff4733b502d0ce3d800a2f4cd86`. This report does not update the parent gitlink.

## Selector and arithmetic

The default W4A4 CUDA path remains the packed signed-I4 vector dot. Setting `GGML_CUDA_W4A4_MMA=1` selects a separate SM75 `m8n8k32` signed-I4 Tensor Core candidate; an unsupported device fails explicitly rather than silently using the vector path. The first decode (`N=1`) and multi-token (`N>1`) dispatch in each path logs a distinct marker. Both paths reuse one per-token activation pack: F32 absmax/7 and activation/scale with explicit round-nearest F32 division, nearest-even signed codes `[-7,7]`, zero-token codes, and strided input rows.

The candidate kernel uses the per-lane A/B/D mapping already checked by the standalone [SM75 probe](w4a4-cuda-source-gate.md): one 32-thread warp computes an 8-row × 8-token tile, loops over K by 32 signed nibbles, and accumulates in I32 through `mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32`. Every lane participates for masked M/N/K tails. Safe byte assembly handles odd K without unaligned or out-of-bounds word reads. Each output applies explicit ordered round-nearest F32 operations, `((float) dot * row_scale) * token_scale`; bias stays in the EAGLE graph. This source is scoped to SM75 even though the PTX instruction may exist on later NVIDIA GPUs.

The **standalone** probe previously passed six cases and 308 exact I32 outputs on the RTX 2080 Ti; executed SASS contained `IMMA.8832.S4.S4.SAT`. That establishes the instruction fragment mapping. It does **not** validate the newly integrated production kernel, activation pack, graph routing, or speed.

## Checks and handoff

- The isolated worktree configured and built CPU-only `test-backend-ops` on Apple M3 Max. `test-backend-ops -b CPU -o 'W4A4.*'` passed **4/4**: K=9/N=3 ties and zero token; K=9,728/N=2 exact long reduction; new K=33/M=9/N=1 decode and K=33/M=9/N=9 multi-token cases with odd-K, row/token tails, zero token, and strided activations. These same backend cases are registered for CUDA. `git diff --cached --check` passed before commit.
- No local `nvcc` is available, so this new production candidate has **not** been CUDA compiled. No GPU was used by this source task.
- The GPU operator should build the published commit on SM75, run `GGML_CUDA_W4A4_MMA=1 test-backend-ops -b CUDA -o 'W4A4.*'`, and compare its exact backend outputs with the default vector path and CPU oracle. Then verify the `SM75 signed-I4 Tensor Core MMA m8n8k32 candidate` marker and executed kernel SASS, load the audited all-nine W4A4 draft, compare logits/accepted counts/text under the frozen prompt suite, and only then measure packing-inclusive and end-to-end timing against the vector path and ordinary EAGLE. Keep this candidate unlabeled as a live Tensor Core result until those checks pass.

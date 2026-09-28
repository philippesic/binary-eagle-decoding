# Same-input Torch Q path against safe native Q on RTX 5080

This bounded RTX 5080 (SM120) probe compares Torch CUDA/F16 Qwen3
source modules with the output-preserving native block-0 post-RoPE Q
capture on the frozen 29-token code/data-validation training prefix.
Both paths receive the same captured native F32 attention-norm rows;
the Torch input is explicitly cast to F16. The F16 Q matrix and F32
head-norm weights are audited against the pinned GGUF and HF source.
Torch uses the installed Qwen3 RMS norm, rotary embedding and
`apply_rotary_pos_emb` modules, with the HF config and implementation
hashed in the machine report. No full target model forward is run in
this operator comparison.

| Stage on identical input | Exact F32 / 118,784 | Position-3 relative row L2 |
| --- | ---: | ---: |
| Torch F16 raw Q projection vs ggml raw Q | 14,513 | 0.2880% |
| Torch F16 normalized Q vs retained ggml normalized Q | 3 | 0.2337% |
| Torch F16 post-RoPE Q vs safe native Q | 2 | 0.2347% |
| Exact ggml CUDA post-RoPE Q vs safe native Q | **118,784** | 0 |

The Torch post-RoPE maximum absolute gap is `0.08107709884643555`,
with 0.2306% median relative row L2 across the 29 positions. The
discrepancy is present at raw projection and persists after norm and
RoPE; these stage errors are not additive. Explicit F16 casting of the
ggml input changes no raw or post-RoPE Q value.

The ggml exact replay cannot expose its intermediate norm tensor after
the graph finishes: that buffer is reused. Marking the raw and normed
tensors as outputs preserves their readback, but alters post-RoPE Q by
16,383 F32 values, maximum `9.5367431640625e-7`. The retained graph's
raw projection is still **118,784/118,784** identical to the exact
graph, so the Torch raw comparison is a faithful same-input boundary.
Its normalized stage comparison is diagnostic only; the final Torch
comparison uses the output-preserving native Q directly. The first
unretained intermediate readback and the retained-only run were kept
as ignored diagnostics and excluded from the fidelity claim.

The final supervised run is
`checkouts/target-block0-operator-20260928/runs/target-q-cuda-d-20260928`
on the registered WSL host. Its ignored `comparison.json` SHA256 is
`849b6c26b40261062f2664783c2c0b7672dca681510cebfd7687364f8d0e5474`.
It records all stage metrics, operand/capture hashes, source module and
helper hashes, software versions and CUDA library identity. The
supervisor exited zero and released its process group; the GPU returned
to 0% utilization and 1,372 MiB whole-device baseline use. Local C++
syntax and Ruff lint/format passed. No optimizer, final prompt or Q4_0
serving evaluation ran. This one SM120 prefix does not establish SM75
performance, global target-feature parity or a training tolerance.
Attention/residual attribution, later blocks, the exact-versus-numeric
training policy and all-body budget remain open.

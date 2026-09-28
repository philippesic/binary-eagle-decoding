# Native CPU draft-attention oracle from stored cache bytes

**2026-09-28 UTC.** This CPU-only diagnostic reconstructs each EAGLE draft
decoder attention operation from the archived Apple M3 Max captures. It uses
the actual post-write F16 physical K/V cache rows, the captured F16 attention
mask, native RoPE-applied F32 Q, and the same execution's native F32
`kqv_out-0` as the reference. No model forward, new inference, GPU run,
training step or performance measurement occurred.

Parent commit `22bf7ce` adds a bounded ggml CPU helper and a strict replay
loader. The loader re-audits each cache/graph file, joins `group_execution`
to cache `execution`, applies physical writes in order (including later slot
rewrites), and rejects any mask-visible slot without known K/V bytes. It
preserves the native query batch shape and 256 K/V slots, including masked
padding. The helper uses Qwen3/EAGLE head geometry `32×128` query and
`8×128` K/V, scale `1/sqrt(128)`, zero ALiBi/softcap, no sinks,
`n_kv_max=0` and F32 accumulator precision as in the pinned native graph.
The captured server and helper both used 10 CPU threads with Flash Attention
enabled. Six focused tests passed, including compiled-helper fixtures;
Ruff and formatting passed.

| Frozen training capture | Decoder executions | Query rows | F32 attention elements bitwise equal | Maximum / RMS error |
| --- | ---: | ---: | ---: | ---: |
| Prose: urban waterways 01 | 26 | 69 | 282,624 / 282,624 | 0 / 0 |
| Reasoning: rate and work 01 | 20 | 76 | 311,296 / 311,296 | 0 / 0 |
| **Total** | **46** | **145** | **593,920 / 593,920** | **0 / 0** |

Every per-execution, per-head relative L2 error was also zero. The sweep
covers the first seed, multirow context and later executions with cache
position rewrites. The ignored reports are
`results/recurrent-native-attention-oracle-20260928/prose.json` (SHA256
`b7ea8ca93f3e294770eb65d992ab2b3ce9b47735e2b13eb6fe264d89e2364fa4`)
and `results/recurrent-native-attention-oracle-20260928/reasoning.json`
(SHA256
`b4cc1c95c17ec0326a17f23c56e8c159138fd2bbd182f6e1cfd7e021b02c47ee`).
They record pinned FP16 target/D hashes, all raw capture and source hashes,
helper/build hashes, row counts, cache slots and per-head results. The local
helper binary SHA256 is
`f63177876148da8e285afc39c021a9d28b9cc75bd020d92471a3d58bcdc3487f`;
CPU ggml library SHA256 is
`8b20e9dc5c61371f5fddd974b73075bcdde4dccb63790ac610e70e69e718bf7f`.
The helper was rebuilt from pinned fork commit
`87cdf11fb6fbfe5d35ab297ecae163c718a9593f` on Apple M3 Max arm64.

This establishes a byte-exact **native CPU attention oracle from the
captured operands**. It does not make the Python differentiable adapter's
F32 attention numerically identical to ggml, prove other drafter operations
exact, or validate CUDA/SM75 arithmetic. The next bounded check is to use
this oracle against the same student Q/K/V and inspect attention,
pre-norm state, logits and gradients before considering any training
arithmetic change.

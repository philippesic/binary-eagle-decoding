# Exact ggml CPU key RoPE replay on the reasoning prefix

The pinned ggml CPU RoPE graph was replayed on the **same native raw F32
key vectors** from all 46 context positions of the sealed reasoning
first-round capture. The replay used EAGLE's normal-pair RoPE mode,
128 dimensions, frequency base 1,000,000, scale 1, original context
40,960, and no YaRN extension. The rebuilt graph output matched captured
`Kcur_rope-0` **47,104/47,104 F32 values bitwise**. Casting its output to
F16 matched **47,104/47,104 actual native stored context key values**.

The prior [cache-boundary intervention](recurrent-binary-reasoning-cache-boundary.md)
already matched all 46 raw K/V projection rows bitwise after ordered FC
and binary K/V arithmetic, and all context stored values. The seed row at
position 46 was already byte-identical. Composing those checked
boundaries with this ggml RoPE replay accounts for the complete
47-position prefix: **48,128/48,128 key** and **48,128/48,128 value**
F16 stored elements. The Python adapter's own RoPE arithmetic had left
one F16 key threshold at each of positions 15, 20 and 40. This replay
isolates those three bits to that arithmetic choice on the captured raw K
input. It does not change the default student forward.

The helper links the same Apple CPU `libggml-cpu` build used in the
earlier attention operand ablation (library SHA256
`8b20e9dc5c61371f5fddd974b73075bcdde4dccb63790ac610e70e69e718bf7f`),
whose RoPE implementation source SHA256 is
`3ad6c159f0e9c2e8cef589be2f41f8ca7221a4e465488ad13ae5bff55669158c`.
The helper binary SHA256 is
`d44d710f4debd138ec4f790ab7317626650d9ff69e9770aeec19c478a718def1`.
The ignored [machine report] is
`results/recurrent-rope-oracle-20260928/rope-parity.json` in the main
checkout, SHA256
`cf99b7bf89847a692800e8dc1c6870bd7e862476abeaa57fa3d9c1d5334a9f88`.
It verifies the sealed manifest, prior ordered-raw-K result, CPU library
and source hashes, native stored-cache audit, every F32 graph output and
every F16 key write. The helper and comparator both ran successfully;
Ruff lint and format checks passed.

This is an Apple M3 Max CPU diagnostic on one frozen first-round prefix.
No CUDA/SM75 result, training tolerance, optimizer, final prompt or Q4_0
evaluation follows from it. First-depth state/logit parity remains open.

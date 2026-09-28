# Reasoning draft-cache projection boundary on Apple CPU

The sealed first-round reasoning capture and the archived grouped-matmul
student cache were joined at the same 47 visible positions. The original
student cache differs from native storage at **70/48,128 F16 key** and
**66/48,128 F16 value** elements. Of those differences, positions 26 and
45 contain 33+26 key and 34+20 value elements respectively; the seed at
position 46 matches both caches bitwise.

The probe rebuilt the 46 context positions from the pinned raw target
features, target embedding, candidate-D weights and original grouped
student. Its rebuilt context cache matched the archived student cache
bitwise. Across these positions, grouped FC arithmetic left 14 F16-cast
fused-input differences on 13 rows. At positions 0, 26 and 45, replacing
only the two/one/one differing F16 input coordinates and using ordered
binary K/V projection made both raw K and V outputs **1,024/1,024 F32
exact** per row. The ordered FC projection followed by the existing RMS
norm matched the captured `g_norm-0` and fused input **bitwise at every
one of 46 positions**; ordered K and V then each matched all **47,104
raw F32 projection values**. This isolates the material context K/V gap
to FC grouped reduction crossing a few fused-input F16 thresholds, rather
than different K/V weights or stored-cache write behavior.

After ordered FC and K/V projection, the adapter's RoPE calculation
reproduced **47,101/47,104** native stored F16 keys and **47,104/47,104**
native stored F16 values across the context. The three remaining key
differences are one F16 bit at each of positions 15, 20 and 40. Replaying
the same RoPE formula on the original grouped K reproduced the archived
student key cache **47,104/47,104**; the residual is therefore at the
native-versus-adapter key rotation arithmetic boundary. The existing seed
cache at position 46 matches natively, so the complete 47-position
intervention yields 48,125/48,128 keys and 48,128/48,128 values.

The [earlier broader CPU diagnostic](recurrent-binary-cpu-broader-diagnostic.md#native-style-norm-and-rope-replay-across-later-rounds)
also observed key thresholds at positions 15, 20 and 40 on a later
reasoning round. This check adds a sealed first-round stored-cache join,
reproduction of the grouped student cache, and an ordered-FC intervention
across the whole context. The ignored machine report is
`results/recurrent-attention-operands-20260928/reasoning-cache-projection-boundary.json`
in the main checkout, SHA256
`28aa44ed15f1798271f338338ea3cdb4821ea4f816aded38bc499a59955ca7f2`.
It records every position's counts, three detailed raw projection
interventions, the exact F16 threshold indices, file hashes and versions.
The sealed capture manifest SHA256 is
`99b9003698bcdf5505421a1667dac4d21a8545471d91330c04181c9f2a931d40`;
candidate D SHA256 is
`10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`.

The probe ran on Apple M3 Max CPU with NumPy 2.4.6 and Torch 2.14.0.
Ruff lint and format checks passed. No GPU, model server, optimizer,
reserved-final prompt or Q4_0 evaluation ran. The result is one frozen
reasoning prefix, not general recurrent or CUDA/SM75 parity. The next
arithmetic check is the three residual native key RoPE F16 thresholds,
then first-depth attention/state comparison using corrected cache operands.

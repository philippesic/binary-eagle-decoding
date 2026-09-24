# Binary kernel work

`binary_reference.py` is a CPU-only correctness reference for the native W1A1
path. It packs a row-major weight matrix `(output_features, K)` and a row-major
activation matrix `(tokens, K)`. Each logical row has `ceil(K/32)` unsigned
32-bit words. Feature `k` occupies bit `k % 32` of word `k // 32`, so bit zero
corresponds to the first feature. A bit is `1` for a nonnegative value,
including `+0.0` and `-0.0`, and `0` for a negative value. Unused high bits in
the last word are set to zero by the packers and masked by the dot operation.
Its integer result is exactly `K - 2*popcount(XOR)` over logical features.

Default weight scales are per output row and activation scales are per token;
both are mean absolute values. The `rms` and `unit` alternatives match the
simulation's scale rules. A scalar explicit scale broadcasts over all rows;
a scale sequence must have one element per row. Scales and output arithmetic
round to IEEE float32. The reference accepts finite source values, flattens no
batch dimensions implicitly, and returns `(tokens, output_features)` float32
values. A caller must flatten and restore any leading batch/time dimensions.
The reference does not promise bitwise agreement with BF16 PyTorch simulation,
which may round at different stages.

The audited AngelSlim EAGLE-3 checkpoint has reduction widths `K` of 2,560,
4,096, 5,120, 7,680, and 9,728; they occupy 80, 128, 160, 240, and 304 words
per row. For example, the feature-fusion weight is `(2560, 7680)` and the
vocabulary-head weight is `(32000, 2560)`. These dimensions come from
`experiments/eagle3-graph-audit.md`; tests exercise each audited width with a
sample row without downloading or allocating the complete matrices.

This Python code is a numerical reference, not a GPU kernel or throughput
claim. The SM75 XOR/POPCOUNT kernel, measured-shape correctness comparison,
packing-inclusive timing, and llama.cpp dispatch remain separate work.

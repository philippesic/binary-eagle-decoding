# Bounded SM75 binary-MMA check

**Status:** planned for the RTX 2080 Ti; its address is absent from the shared
host registry. The validated portable `__popc` path is true packed W1A1
execution, but it is not a Tensor Core path, and 5080 timing cannot establish
Turing performance.

NVIDIA's [PTX ISA binary MMA instruction](https://docs.nvidia.com/cuda/archive/12.6.0/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
supports the Turing-compatible form:

```ptx
mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc
  {d0, d1}, {a}, {b}, {c0, c1};
```

The official [m8n8k128 fragment
layout](https://docs.nvidia.com/cuda/archive/12.6.0/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-88128)
gives each lane one 32-bit packed register from A and B and two integer
accumulators. With warp lane `l`, `g=l>>2`, `t=l&3`, A uses weight row `g` and
K word `4*tile+t`; B uses activation column `g` and the same word. Each lane
holds output row `g` and columns `2*t`/`2*t+1`. All lanes participate in every
instruction, including partial row/token tiles. Canonicalize padding bits of
both inputs to zero and convert the final mismatch count using logical
`K - 2*count` before the established F32 scales.

The small gate is one 8×8×128 tile with independently known bits, then K=31,
32, 33, 2560 and the other audited EAGLE widths, signed zeros, dirty tails,
and the head's actual 1/10-token shapes. Compare exact integer dots and
scaled outputs against the CPU packed reference and portable `__popc` path.
Build an SM75 cubin, inspect disassembly for binary MMA, execute on the 2080
Ti, and time packing-inclusive calls against the portable path. At one token,
seven of eight MMA columns are idle, so speedup is uncertain. If the portable
end-to-end run is negative, this bounded probe is needed before claiming that
binary Tensor Cores cannot change that conclusion.

Compile-only evidence on the 5080 host can check syntax and emitted SM75
instructions, but no SM75 runtime or speed conclusion follows until the 2080
Ti is reachable. No Turing or binary-MMA result exists yet.

# Bounded SM75 binary-MMA check

**Status:** standalone SM75 probe implemented and cross-compiled; runtime
correctness and timing await the RTX 2080 Ti, whose address is absent from
the shared host registry. The validated portable `__popc` path is true packed W1A1
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

The standalone probe was integrated at parent commit `dee6556`, with full and
partial tile cases, audited K widths, dirty tails, and an independent dense
CPU sign reference. On the RTX 5080 host's CUDA 13.1.115 toolkit, an SM75
compile-only supervised run exited zero in 1.50 seconds. PTX contained the
specified `mma.sync...xor.popc` form and `cuobjdump --dump-sass` showed six
static `BMMA.88128.XOR.POPC` sites in the SM75 image. The binary was **not
executed** on the 5080; the probe rejects non-SM75 devices. This establishes
that the toolchain can emit the intended Turing instruction, not that the
fragment mapping is numerically correct or faster on a 2080 Ti. The exact
compile command was:

```sh
nvcc -std=c++17 -O2 -arch=sm_75 -I/home/philip/binary-eagle-decoding/results/cuda-glibc-compat/include kernels/sm75_mma_probe.cu -o results/sm75-mma-probe-20260924/sm75_mma_probe
```

The ignored host artifact manifest under
`results/sm75-mma-probe-20260924/` has SHA256
`fd7303c951015d0cce86abce85a66da1b888aa11e5776d1c4f3beae7232ffdcf`.
Source SHA256 was
`fe589a26800b6e9e605b4647900ddc1178ce9df1e1532185f47e51bb3a00a880`,
binary `c72c9b2c84e26eafc73b98a1baabebd0bac936f1bbee1f24deab18b485bf9cd5`,
SASS `d20e62ffef2a0c5ff780ceb42db5ade3c7141bf3139f99c1f3ec2f1c8b16f99e`,
and PTX `9c56ea151c7ef6604218a36fb367846b50c22a263c7c9e32ba9a5792dcc71148`.
Both compile and inspection supervisors exited zero; the 5080 remained idle.

Parent commit `a595ee8` added an explicit `--proxy-correctness` option to the
same standalone program. Default execution still rejects non-SM75 devices;
the proxy option permits only SM120 and labels its output as insufficient for
an SM75 runtime or speed claim. The CUDA 13.1 toolchain can include both
SM75 SASS and compute_75 PTX in one binary, allowing the 5080 to JIT the
same virtual instruction form for a **numerical mapping check only**. That
bounded 5080 proxy run has now completed; it cannot replace a Turing run.

The bounded proxy run subsequently passed on the RTX 5080: a single
supervised `--proxy-correctness` invocation exited zero with **21 PASS cases
and 880 exact integer-dot outputs** against the independent dense CPU sign
reference. It used an executable containing SM75 SASS plus compute_75 PTX;
the 5080 JIT ran the virtual binary-MMA form. Full and partial tiles, dirty
tails, and all audited K widths passed. The program printed its proxy-only
label and explicit warning against SM75 runtime/performance inference.
Compiler, binary, stdout, disassembly, and final GPU-state hashes are
sealed in ignored host artifacts. The numerical mapping is validated on the
5080 proxy path; a real 2080 Ti run remains required. The exact compile
command was:

```sh
nvcc -std=c++17 -O2 --gpu-architecture=compute_75 --gpu-code=sm_75,compute_75 -I/home/philip/binary-eagle-decoding/results/cuda-glibc-compat/include kernels/sm75_mma_probe.cu -o results/sm75-mma-proxy-20260924/sm75_mma_probe
```

The ignored `results/sm75-mma-proxy-20260924/` manifest SHA256 is
`e6d9dadaf116dcd92f52b1fba3b187044c8c4c5f2eb56a50b0d95e777158250a`.
The source SHA256 is
`c89695021e077248cee5700888c74030351ee58ac2bb1828ed293acca4aaebfd`,
binary `2f31b3fdb004d2df8d47a302db09c9dfb5002cd4152611b73d8ac3bd5b4b7267`,
SASS `293cd29ffe6e53fdf010687bc63106cc27198b7464d6a7dd2693c14302c3c072`,
PTX `aa83cae54dbe3a7363cba7296beaf4ea1fa544916ab800485fdc628410c0d2b8`,
and run stdout `bd7b0cfb923a90b17fa0db1f078abb4b4d81277f4511052e752a4b51b080bb97`.
The single proxy run exited zero; post-run RTX 5080 samples were back at the
3,047 MiB/0–1% idle level, with no project processes or SSH/tmux session.

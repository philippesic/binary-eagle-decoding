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
claim. SM75 instruction probing and llama.cpp dispatch remain separate work.

## Standalone CUDA prototype

`w1a1_cuda.cu` and `w1a1_cuda.cuh` implement activation sign/mean-absolute
packing and a portable XOR/`__popc` linear. The companion
`w1a1_cuda_bench.cu` checks numerical behavior and records CUDA-event timings.
They are **not** wired into llama.cpp. This is a prototype of the F32 native
arithmetic contract, not a tensor-core kernel or an end-to-end speed result.

On a CUDA 13 host with an architecture supported by the installed toolkit,
build the 5080 executable after confirming the device architecture:

```sh
mkdir -p build
nvcc -std=c++17 -O3 -arch=sm_120 -U_GNU_SOURCE -D_DEFAULT_SOURCE \
  -Xcompiler=-Wall,-Wextra \
  -o build/w1a1_cuda_bench kernels/w1a1_cuda.cu kernels/w1a1_cuda_bench.cu
build/w1a1_cuda_bench --correctness-only > results/w1a1-cuda-correctness.json
build/w1a1_cuda_bench --warmups 10 --samples 30 > results/w1a1-cuda-bench.json
```

`sm_120` is the intended 5080 build target; check the actual device and
`nvcc --list-gpu-arch` first. On the tested CUDA 13.1 / glibc 2.43 WSL host,
`-U_GNU_SOURCE -D_DEFAULT_SOURCE` avoids conflicting `rsqrt` declarations
while retaining declarations required by libstdc++; record the flags as part
of the build manifest. The runner prints one JSON object and returns
nonzero on a failed numerical check. Capture compiler command/version, GPU,
driver, clocks, project commit, and file hashes alongside the JSON according
to `docs/EVALUATION.md`. The commands above do not create the ignored
`results/` directory; the experiment runner must create its unique run
directory first.

The correctness check covers K=31/32/33 and all five audited EAGLE K widths
(2560/4096/5120/7680/9728), with seven sampled weight rows and three token
rows per width. It compares every integer dot to direct CPU source-value signs,
activation words to an independent CPU pack, and scales/output to double-sum
CPU mean-absolute scales. Cases include both signed zeros, all-zero rows,
negative subnormals, and deliberately dirty padding bits. Dot equality is
exact. F32 scale tolerance is `2e-7 + 5e-5*abs(expected)` and output tolerance
is `2e-4 + 5e-5*abs(expected)`; the F32 CUDA reduction tree differs from the
CPU double sum. No `--use_fast_math` is used.

The benchmark covers all nine audited matrix shapes at one and ten token
rows. Ten matches common calls in the CUDA drafter profile. It alternates
prepacked linear and pack-plus-linear measurements after warmup and emits raw
CUDA-event samples, min, median, and p95. Buffers, packed weights, and dense
activations are already on device; the inclusive span contains activation
sign packing, scale reduction, and binary linear. It excludes allocation,
host-to-device transfers, graph replay, bias, and surrounding draft work.
Packed weights are synthetic for timing. Use captured real activations,
CUDA-graph replay, matched ordinary baselines, and the full verifier for any
subsequent performance claim. Nothing here measures the RTX 2080 Ti or SM75
binary instructions.

## SM75 binary-MMA probe

`sm75_mma_probe.cu` is a separate one-warp `m8n8k128` XOR/popcount correctness
probe for an RTX 2080 Ti. Build an SM75 cubin plus `compute_75` PTX so the
same executable can JIT on a later architecture:

```sh
nvcc -std=c++17 -O2 --gpu-architecture=compute_75 \
  --gpu-code=sm_75,compute_75 -o sm75_mma_probe kernels/sm75_mma_probe.cu
./sm75_mma_probe
cuobjdump --dump-sass sm75_mma_probe | grep -E 'BMMA|MMA'
```

It checks every integer output against a dense CPU sign reference for a full
8-by-8 tile, partial row/token tiles, K tails with dirty padding bits, and the
five audited EAGLE widths. The default run rejects non-SM75 devices. On an
SM120 RTX 5080, `./sm75_mma_probe --proxy-correctness` runs those same exact
integer-dot checks through the JIT-compiled PTX. It reports a proxy result
and does not establish SM75 runtime correctness or performance. Compilation
or disassembly alone also does not establish either result; this probe never
measures a complete EAGLE draft call.

## Real packed-head parity fixture

`scripts/create_real_head_fixture.py` reads BF16 pre-head activations from the
QAT capture and the actual I32/F32 packed head tensors in the GGUF. Its ignored,
versioned binary fixture contains eight deterministic captured rows, **all**
32,000 packed output rows, the source SHA256 hashes, expected sign words, F64
sum then F32 activation scales, exact integer dots, and ordered F32 outputs.
Its JSON sidecar records the capture, GGUF, fixture and code hashes, selected
source row indices, project and llama.cpp revisions, the GGUF contract version,
package versions, and reference arithmetic policy. The script rejects an
existing output file or sidecar; keep each run in its own directory.

On a checkout with the captured inputs and packed GGUF present, create it using
the existing GGUF conversion environment (which includes `gguf-py`):

```sh
results/convert-env/bin/python scripts/create_real_head_fixture.py \
  --capture results/qat-head-capture-20260924/train.pt \
  --gguf models/gguf/Qwen3-4B-eagle3-head-w1a1.gguf \
  --output results/real-head-fixture/w1a1-real-head.bin
```

Build and run the standalone CUDA checker on the RTX 5080 after GPU ownership
is available:

```sh
nvcc -std=c++17 -O3 -arch=sm_120 -U_GNU_SOURCE -D_DEFAULT_SOURCE \
  -o build/w1a1_real_head_check \
  kernels/w1a1_cuda.cu kernels/w1a1_real_head_check.cu
build/w1a1_real_head_check results/real-head-fixture/w1a1-real-head.bin
```

It validates the fixture version, dimensions, and exact file length; compares
every activation sign word and integer dot exactly; and checks scale/output
against the F64-sum reference with the same F32 reduction tolerances used by
the standalone synthetic correctness benchmark. It prints one JSON result and
exits nonzero on mismatch. This is numerical parity for the portable CUDA
prototype. It does not measure latency, invoke the SM75 binary-MMA kernel, or
establish a 2080 Ti result.

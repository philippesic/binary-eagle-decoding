# RTX 2080 Ti execution gate

**Pending input:** the current RTX 2080 Ti host/IP. The username is `philip`.
Do not infer an address from the 5080 or an old session. Record the supplied
address only in the shared local `hosts.toml` via `scripts/agent_env.py`, not
in Git. All SSH to this WSL GPU must use tmux MCP and all experiments must
use `scripts/remote_job.py` with unique run IDs.

The [active project goal](goals/full-w1a1-eagle-project.md) and
[evaluation protocol](EVALUATION.md) govern this gate. The 5080
[native end-to-end comparison](../experiments/native-end-to-end-5080.md)
found head-only packed W1A1 slower than ordinary EAGLE, but 5080 behavior
cannot establish SM75 correctness, utilization, or speed. A standalone
[binary-MMA probe](../experiments/sm75-binary-mma-plan.md) cross-compiled to
SM75 and passed numerical checks only on an SM120 proxy path.

## Access and preflight

1. Register the user-supplied IP with `python scripts/agent_env.py set-host
   rtx2080ti <ip> philip --port 22`; verify the pause flag and recorded
   workdir. Use tmux MCP for the SSH session. Check actual GPU name, compute
   capability 7.5, VRAM, driver, toolkit, compiler, glibc, free disk, and
   other active processes. Give the 2080 Ti one experiment owner.
2. Fetch pinned parent `main` and its llama.cpp submodule. Hash the exact
   source, model snapshots, config and prompts. Start every compile, test,
   conversion, and inference under a supervised unique run directory. Keep
   raw artifacts outside Git.
3. Use this host's own toolkit and headers. The 5080 needed a private CUDA
   13.1/glibc 2.43 header patch; apply it only if the 2080 Ti has the same
   demonstrated conflict. Never modify the system CUDA header in place.

## Correctness before timing

1. Build llama.cpp CUDA for `sm_75` with backend tests enabled, recording
   CMake cache, nvcc command/version and binary hashes. Run
   `test-backend-ops test -b CUDA0 -o W1A1_MUL_MAT`; require all five
   scalar-reference cases to pass.
2. Compile the standalone `kernels/sm75_mma_probe.cu` for SM75, inspect
   `cuobjdump` for `BMMA.88128.XOR.POPC`, and execute the **default** mode on
   the real 2080 Ti. Require all 21 cases/880 exact integer dots to pass.
   The 5080 proxy pass is not a substitute. If this fails, stop the MMA claim
   and keep the portable `__popc` path separate.
3. Convert the pinned target and ordinary/packed EAGLE draft on this host,
   auditing all 32,000 packed head rows and hashing the three GGUFs. A short
   native server smoke must show the packed-head loader and explicit CUDA
   XOR/POPCOUNT dispatch before a timed run.

## Matched hardware comparison

The 2080 Ti has a different memory budget. Try the same FP16 target/draft,
context and KV settings as the 5080 comparison only after checking actual
free VRAM. If they do not fit, define a **separate quantized-target track**
and use the identical target GGUF, KV precision, placement and context for
target-only, ordinary EAGLE and packed-head W1A1. Record conversion command,
target quantization and resulting hash; do not compare its throughput ratio
against the 5080 FP16 track.

Run the paired harness with the frozen 12 prompt IDs, greedy settings, two
warmups/server, at least five alternating repetitions, and one variant/server
at a time. Preserve per-request wall/prefill/decode time, completion lengths,
accepted/proposed/round counts, draft-time stats, GPU memory snapshots,
text/token parity evidence, and explicit W1A1 CUDA dispatch. Analyze pooled
token/time ratios and paired prompt/repetition spread. Profile packing and
the portable binary kernel separately; do not add overlapping GPU spans as
though serial. A negative result is valid if its bottleneck and limitations
are quantified. If the portable path is negative, only claim that path is
negative; compare an actual integrated MMA path before ruling out Turing
binary Tensor Cores.
The current activation packer accumulates magnitudes in F64 before writing
an F32 scale; measure that reduction separately on Turing rather than
carrying over the 5080 pack cost. The 5080 integrated-kernel NCU attempt was
denied GPU performance counters, so its standalone CUDA-event result is the
available component baseline, not an integrated per-kernel profile.

After each run, stop the supervised process group, check its state and GPU
process list, record final memory/utilization, close SSH/tmux, and checkpoint
the goal/status with exact commit, artifact paths, and hashes.

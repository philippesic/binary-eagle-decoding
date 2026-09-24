# Goal: complete W1A1 EAGLE research program

**Opened:** 2026-09-24  
**State:** active local preparation; RTX 2080 Ti measurements await Ubuntu WSL and SSH access

**Orchestrator:** current Codex task  
**GPU owner:** none; RTX 5080 is idle

**First autonomous work window:** 2026-09-24 09:08–19:08 UTC; the objective
continues beyond that window if required.

## Objective

Work through the remaining stages of `docs/PROJECT_OVERVIEW.md` to determine
whether true packed W1A1 EAGLE drafting improves end-to-end inference. Deliver
a pinned and correct native path with same-device target-only and ordinary
EAGLE anchors, or a reproducible negative result that quantifies why it fails.
The user has delegated research choices for this work window and granted full
RTX 5080 access. Make bounded decisions from evidence, checkpoint them in
`docs/DECISIONS.md`, and continue independent work when one path is blocked.

## Completion evidence

- Resolve the ordinary BF16 target/EAGLE greedy divergence enough to state
  verifier correctness and the limits of acceptance measurements, or document
  a precise remaining blocker with traces. Keep target weights and sampling
  semantics fixed within comparisons.
- Choose selected W1A1 layer coverage and/or a bounded drafter-only QAT recipe
  using the held-out acceptance and measured layer costs. Keep training data
  separate from `configs/acceptance_prompts.jsonl`; report resulting held-out
  acceptance and precision coverage.
- Implement a numerical binary-dot reference, weight/activation packers, and
  an SM75 XOR/POPCOUNT kernel for actual measured drafter shapes. Test tail
  masking, zero sign, scales, layouts, and numerical agreement. Benchmark
  packing-inclusive latency and state actual GPU/compiler/dispatch.
- Integrate selected packed binary draft operations into the pinned llama.cpp
  path. Preserve submodule changes as reviewable commits, publish its commit
  to the user's fork before updating the parent gitlink, and verify unchanged
  target/verifier behavior.
- Compare target-only, ordinary FP16 EAGLE, and native W1A1 end-to-end under
  matched settings on each relevant GPU. The RTX 2080 Ti is required before an
  SM75 speedup claim; its host address is not currently in the local registry.
  Include request/decode timing, accepted/proposed/emitted tokens, draft and
  verifier cost, memory, prompt spread, and raw artifacts/hashes. Report a
  repeatable gain or a quantified negative result, without substituting
  simulations for native binary timing.

## Boundaries and working decisions

- Keep the published Qwen3-4B target and AngelSlim EAGLE-3 checkpoint as the
  starting pair. Existing pinned CUDA acceptance reports in `experiments/`
  show ordinary BF16 2.3166 accepted drafts/round, all-group W1A1 0.2022,
  W4A4 0.2882, and W8A8 2.1816 under the AngelSlim verifier. Strict
  target-only/ordinary parity fails at generated token 4 on RTX 5080 BF16;
  do not treat those counts as target-equivalent.
- The RTX 5080 is the sole routine GPU experiment device. Give it one owner at
  a time; all WSL SSH must use tmux MCP and all remote experiments must use
  `scripts/remote_job.py`. A user pause request stops new GPU work first.
- Do not commit model weights, datasets, captures, or raw runs. Use ignored
  local/remote `results/` and `runs/`, with hashes in compact reports.
- No QAT, native low-bit throughput, or SM75 performance result exists yet.
  CPU/Metal checks cannot substantiate SM75 performance.

## First independent work units

1. **Luna high GPU operator:** exclusively inspect the RTX 5080 and run a
   bounded BF16 verifier-parity investigation, then prepare matched llama.cpp
   CUDA conversion/runtime anchors if the diagnostic leaves time. Own remote
   sessions/jobs/raw artifacts only; stop at an external host or resource
   blocker, and report exact commands, process status, and hashes.
2. **Sol high kernel owner:** in an isolated worktree, own a pure numerical
   binary-dot reference, packer, and focused CPU correctness tests under
   `kernels/` and `tests/`. No GPU use or llama.cpp edits. Deliver a reviewable
   commit and stop at that boundary.
3. **Astra medium advisor:** read-only focused advice on the BF16
   tree-verifier versus full-prefix target logit shift and the minimal checks
   that distinguish rounding from cache/mask/position errors. No GPU use.
4. **Orchestrator:** review/integrate worker commits, decide the next bounded
   QAT/coverage and native-kernel steps from evidence, plan llama.cpp changes,
   own this goal file and `docs/STATUS.md`, and push coherent milestones.

## Current checkpoint

- Starting project commit `811fbbd` on clean pushed `main`. The completed
  [RTX 5080 W1A1 acceptance report](../../experiments/pytorch-w1a1-cuda-acceptance.md)
  and [W4A4/W8A8 report](../../experiments/pytorch-int4-int8-cuda-acceptance.md)
  preserve prompt/model hashes, code revisions, raw artifact locations,
  acceptance counts, and verifier limitations. All prior GPU supervisors ended
  and SSH sessions closed; no live project experiment is recorded.
- The RTX 5080 registry entry is populated and unpaused; the RTX 2080 Ti entry
  has a username but no host address. Verify actual current host/resource state
  before any new run. No new GPU job has been launched yet.
- Opening checkpoint `254eaba` was pushed before worker dispatch. Active
  assignments: `/root/cuda_acceptance_operator` alone owns the 5080 parity
  investigation; `/root/binary_reference` owns an isolated CPU kernel-reference
  worktree; `/root/parity_advice` is read-only. The orchestrator owns docs,
  integration, and research decisions. Workers must not edit each other's
  files or share the GPU.
- The thread heartbeat `continue-w1a1-eagle-project` is active every 30
  minutes for the roughly 10-hour away window, with instructions to avoid
  duplicate work while a task/worker is active and to report meaningful
  progress only. It will be paused at the window end.
- CPU packed-binary reference/packer was reviewed and integrated on `main` at
  `f8f3209`; the isolated worktree/branch were removed after verifying its
  three owned files matched the pushed cherry-pick. `make check` passes 38
  tests. The reference uses row-major little-bit-order uint32 words, sign(0)
  = +1, logical-K tail masking, per-row/per-token FP32 scales, and
  `K - 2*popcount(XOR)`. No GPU kernel or native speed claim exists yet.
- A numerical-contract issue needs explicit handling: the prior PyTorch
  W1A1 wrapper performs sign GEMM and successive scaling in BF16, while the
  new packed reference uses integer dot with FP32 scales/output. The native
  path must pin its rounding contract and either match/re-evaluate held-out
  acceptance or reproduce BF16 stages for a parity diagnostic. Do not project
  previous accepted/round values directly onto a native implementation.
- The pinned llama.cpp graph/loader audit selected a narrow native route:
  one dedicated W1A1 matrix operation with I32 packed-weight and F32 scale
  companion tensors, initially at the EAGLE head. The staged file/interface
  plan and loader/converter pitfalls are in `docs/native-w1a1-path.md`.
  Existing GGML Q1_0 is weight-only block quantization and does not implement
  this row-scaled W1A1 contract.
- A bounded head-only QAT pilot was chosen from acceptance and layer-cost
  evidence, after the current GPU parity diagnostic. Its data split,
  forward-equivalence checks, 500-step/45-minute stop, export audit, and
  held-out gate are fixed in `experiments/qat-head-pilot-plan.md`. No QAT
  process has started and the 5080 remains owned by the parity operator.
- `5cb3ea4` integrated a trainable head-only W1A1 arithmetic core from an
  isolated Sol worktree, which was cleaned after the cherry-pick. Its FP32
  latent weights use a documented clipped weight STE, while the BF16 forward
  is bitwise equal to `fake_binary_linear` on tested inputs; BF16 export is
  explicit. `make check` now passes 43 tests. No capture, optimizer, or
  held-out QAT run has begun. Sol worker `/root/qat_capture` owns the separate
  prompt-generation and feature-capture scripts; the orchestrator will own
  training/evaluation integration after its artifact schema is fixed.
- `f88eaa9` integrated deterministic QAT prompt generation and pinned-model
  drafter-head input capture; the Sol worker's isolated worktree/branch were
  removed after verification. `make check` passes 48 tests. A local generator
  run produced ignored, disjoint prompt manifests: 96 train (SHA256
  `4e44fd2f5806cd7edd56de56a87a211efcc1293b418a5e99cdfc2c1a3cce7a8a`)
  and 24 validation (SHA256
  `2dcec4dd9c954506415b63fe13cd165f6395eab942da61562d82a284d89cc531`),
  balanced across three categories and bound to the held-out prompt hash.
  Remote capture and official runtime integration have not been validated yet.
- A bounded 5080 BF16 parity replay finished under supervisor
  `fullw1a1-parity-matrix-20260924-r1`; the GPU returned to its 3,050 MiB
  baseline and 0% utilization before handoff. At absolute target position 40,
  the EAGLE tree verifier has logits 21.0 for token IDs 11/272/7578 and
  selects lowest ID 11; incremental and full-prefix target evaluation select
  272, with token 11 at 20.875. Mutating every off-path sibling in a replay
  leaves the selected hidden state/logits bitwise identical. This bounds the
  discrepancy to BF16 tree-versus-incremental arithmetic/tie sensitivity at
  this row rather than sibling contamination. The 12-prompt acceptance counts
  remain verifier-relative, not strict target-equivalent. The parity operator
  has relinquished the GPU and closed SSH/tmux. Remote code revision
  `92c91a4cecdd0123a2e1d7bf7c6d394fa673dabe`, run path
  `~/binary-eagle-decoding/runs/fullw1a1-parity-matrix-20260924-r1/`, raw
  trace path `results/full-w1a1-parity-20260924/`, run manifest SHA256
  `39630fc737efcfe9ad9a710b4007188c29613398cfdafb35336847ffa654e421`,
  trace SHA256
  `17bd8f7626cf42b1919add214c095b5b614c4e711f63f097b59f9a622e59df52`,
  script SHA256
  `5a401cbd84ceccbcc3b3b45aecc04236f5c3608c6aff22719dae09257cb8b35b`.
- `fa58da2` integrated bounded head QAT training and `2ddc513` added a
  derived BF16 full-drafter checkpoint exporter and fixed-config held-out
  evaluator manifest. `4b7a365` integrated a standalone CUDA activation
  pack/XOR-popcount prototype. `make check` passes 56 tests. The CUDA source
  still needs real CUDA compilation/correctness/timing; its worker now alone
  owns the 5080. QAT capture/training remain unrun on the GPU.
- llama.cpp submodule commit `257e2c670e892bd9ca9c404167b2919020b969f8`
  adds a dedicated GGML packed W1A1 CPU operation. Its five independent
  scalar-reference backend tests passed on M3 Max, including K tails and
  strided activations. This submodule commit was pushed to the user's fork
  branch `w1a1-cpu-op` before the parent gitlink update. The orchestrator is
  integrating that gitlink; separate Sol owners are building the GGML CUDA
  dispatch and EAGLE packed-head loader/export bridge without GPU access.
- Remaining immediate sequence: finish QAT capture/training and held-out
  evaluation, validate GGML CUDA dispatch after GPU handoff, then run paired
  native comparisons. The RTX 2080 Ti
  address is still absent from the local host registry, so SM75 results await
  access even while 5080 work proceeds.
- The standalone CUDA prototype compiled and passed exact integer-dot
  correctness on the 5080 at eight K widths, plus F32 scales/output
  tolerances. Two supervised nine-shape CUDA-event runs preserve raw samples.
  In the repeat, the 10-token head medians were 0.070016 ms prepacked and
  0.079136 ms packing-inclusive. These omit draft, verifier, H2D, allocation,
  and graph overhead; the one-token samples were variable. Full hashes,
  compiler flags, and table are in
  [the CUDA prototype report](../../experiments/native-w1a1-cuda-prototype.md).
  Branch `feat/cuda-w1a1-prototype` at `8950822` was pushed; fixes `26484b7`
  and `37ebbc6` were cherry-picked into `main`. The GPU was cleanly released.
- The RTX 5080 was handed exclusively to `/root/cuda_acceptance_operator` for
  the QAT pilot. Remote checkout `855fea5` produced the expected disjoint
  96/24 prompt hashes. Capture supervisor `qat-head-capture-20260924` is
  complete, with per-prompt cap 384 and global cap 32768. The later training
  run is documented below. The parity/CUDA prototype supervisors and SSH
  sessions are closed.
- Pinned target and ordinary EAGLE draft converted to FP16 GGUF locally; CPU
  and Metal runtime builds succeeded, and Metal target-only/ordinary EAGLE
  smoke generation loaded both. Source/output hashes, commands, log hashes,
  and limitations are in
  [the conversion report](../../experiments/gguf-baseline-conversion.md).
- Dedicated GGML CUDA dispatch `96e11b5` and strict packed EAGLE-head
  loader/export bridge `d9ab59c` were integrated on submodule branch
  `w1a1-integrated` after the CPU op. The combined submodule commit
  `d9ab59ce19bc5206436a07789d6ca66dbc79ddad` was pushed to the user's
  fork before parent gitlink update. Mac Metal build, two converter tests,
  original-head packed GGUF export, and local mixed Metal target/CPU packed
  drafter smoke passed; CUDA compilation/backend tests are still pending GPU
  handoff. Packed GGUF SHA256 is
  `6250363f5fdb70fcb3113be90cca8755e916ac0da533a76e340335aa418c16ca`.
- The integrated submodule's CPU backend tests passed 5/5 after all changes;
  an all-32,000-row GGUF readback had zero sign/scale mismatches and no dense
  shadow head. See [native GGUF check](../../experiments/native-w1a1-gguf-check.md).
  Submodule commit `92bc706` adds a one-time explicit CUDA W1A1 dispatch log
  for the later server benchmark, was pushed to the fork before parent gitlink
  update `102a062`, and remains uncompiled on CUDA.
- QAT capture `qat-head-capture-20260924` finished with 32,768 train and 9,216
  validation BF16 rows, disjoint 96/24 prompts, and balanced ordinary/W1A1
  trajectories. A CUDA dry run validated artifact hashes and forward parity.
  Bounded head-only training `qat-head-training-20260924` completed 500 steps
  in 39.31 seconds; best step 500 reduced validation KL from 2.50634 to
  1.01696 and improved teacher top-1 agreement from 0.47667 to 0.54167.
  The GPU supervisor exited and memory returned to baseline. Raw manifests,
  exact hashes, config, and limits are in
  [the QAT pilot report](../../experiments/qat-head-pilot-results.md).
  `/root/cuda_acceptance_operator` retained GPU ownership for the one frozen
  exported-checkpoint held-out acceptance run; no tuning used held-out data.
- `235a359` integrated a PyTorch head-only simulation of GGML's integer sign
  dot, F32 exported weight scales, F64-summed/F32 activation scale, ordered
  F32 products and F32 logits. Its ordinary variant stays BF16, and the
  config explicitly labels the full PyTorch BF16 versus GGUF F16 upstream
  difference. `make check` passes 63 tests; no 5080 acceptance run yet.
- `6906771` integrated a same-device, sequential target-only/ordinary/packed
  llama-server comparison harness, with five repetitions, exact prompt/config
  and artifact hashes, raw responses/metrics, process cleanup, and explicit
  CUDA dispatch evidence. Its four fake-server tests pass; no real benchmark
  has run. The RTX 2080 Ti address remains absent.
- The one held-out QAT export/evaluation gate finished cleanly on code
  `855fea5`. The derived model differed only at `lm_head.weight`; the new
  safetensors SHA256 is
  `450e1d3e27244a46faa2dc28b937b507d490c7a1c66d1ab2763c4c6fc677f5a1`.
  Trained W1A1 accepted 936 drafts / 598 rounds = 1.56522, below untrained
  W1A1 1.67668; the trained head at ordinary BF16 accepted 981 / 541 =
  1.81331, below original ordinary 2.31659. Exact target-greedy streams
  matched 6/12 and 4/12 respectively. The bounded recipe is stopped without
  held-out tuning. Full totals, mismatch positions, raw hashes and limitations
  are in [QAT pilot results](../../experiments/qat-head-pilot-results.md).
  Supervisor exited zero; GPU returned to its 3,050 MiB/0% baseline and SSH
  and tmux closed. The GPU is ready for GGML CUDA validation.
- A real one-request Metal `llama-server` API smoke for target-only, ordinary
  EAGLE, and packed-head EAGLE returned timing and speculative counters; it
  also found the default logger omits the packed loader message. Benchmark
  config pins `-lv 4`, which emitted it locally, and requires the explicit
  CUDA op marker. This is schema/load evidence, not a performance result.
- Native GGML CUDA build attempt `runs/native-ggml-cuda-build-20260924/`
  on RTX 5080/CUDA 13.1.115/GCC 15.2/glibc 2.43 stopped at 27/368 before
  W1A1 source validation: suppressing `_GNU_SOURCE` to avoid CUDA's rsqrt
  declaration conflict hid pthread clockwait/clocklock prototypes needed by
  libstdc++ `<mutex>`. The supervisor exited 1, no GPU job remained. A bounded
  private-header probe and successful retry are documented below. Do not infer
  a kernel defect from this host-header failure.
- A private CUDA include-tree copy with only two `rsqrt`/`rsqrtf` declaration
  exception-specification fixes passed a bounded nvcc probe with normal GNU
  macros; the system header remained unchanged. The full patched SM120a
  llama.cpp CUDA build then completed 341/341 steps under supervisor
  `native-ggml-cuda-build-patched-20260924`. Supervised CUDA0
  `test-backend-ops test -b CUDA0 -o W1A1_MUL_MAT` passed all five cases
  (K=31/32/33/2560 and strided K=33) on the RTX 5080, including the explicit
  CUDA XOR/POPCOUNT dispatch log. Both supervisors exited and GPU returned to
  its 3,050 MiB/0% baseline. The pinned target and ordinary draft F16 GGUF
  are byte-identical to local conversions. The packed draft has identical
  sign bits but F32 scale reductions differ by at most three ULP across Mac
  and WSL; all 32,000 rows match the corresponding host source. Exact 5080
  GGUF hashes are in [the integration report](../../experiments/ggml-w1a1-cuda-5080.md).
  A one-prompt CUDA server smoke returned HTTP 200, confirmed packed-head
  loader and CUDA XOR/POPCOUNT dispatch, and recorded 50 proposed / 2 accepted
  / 12 rounds. The server and supervisor exited, GPU returned to baseline.
  The sealed native validation manifest at remote
  `results/native-ggml-cuda-validation-20260924/run-manifest.json` has SHA256
  `8707fc2e774508a8945211168841edeafea15774ddd8b4ddd4c4ba9b572bab3d`.
  The next gate is the five-repetition matched native benchmark; the dry run
  passed and its supervised run is documented below.
- The benchmark dry run passed on remote parent `ef2f47b`/submodule
  `92bc706`, checking all three GGUFs, 12 prompt IDs, five alternating
  orders, two warmups/server, F16 target/draft KV, greedy 128-token limit,
  CUDA binary and hashes. The one supervised paired run
  `runs/native-eagle-5080-20260924/` ran under PID/PGID 6950,
  with `/root/cuda_acceptance_operator` as sole GPU owner. Initial GPU sample
  during repetition 0 target-only warmup was 11,481 MiB used / 4,497 MiB
  free at 82% utilization. It later completed all 180 requests; supervisor
  and all server children exited, and the GPU returned to baseline. The
  operator retained ownership for a separate two-prompt token-ID diagnostic,
  now complete.
- `dee6556` integrated a bounded standalone SM75 binary-MMA correctness
  probe from an isolated Sol worktree, which was then removed. The probe uses
  `mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc`, compares full
  and partial 8×8 tiles and audited K widths to dense CPU signs, and rejects
  non-SM75 runtime devices. Local formatting/diff/CPU lane-layout emulation
  checks passed. A later supervised CUDA 13.1.115 compile-only run on the
  5080 host emitted an SM75 image with six static
  `BMMA.88128.XOR.POPC` SASS sites and the matching PTX instruction; it
  exited zero in 1.50 seconds. The binary was not run because the 2080 Ti
  address is absent; no Tensor Core numerical or speed claim follows.
- The paired 5080 run progressed through all of repetition 0 and 20 rows of
  repetition 1 (56/180 measured rows total) without OOM/error; one expected
  server held 12,487 MiB at 84% GPU utilization. Final results are below.
- The complete 5080 comparison found packed-head W1A1 at 115.3805 request
  and 122.9426 decode tokens/s versus ordinary EAGLE 123.9626 and 132.8794,
  or **0.931×/0.925× ordinary**. Target-only was 96.6686/99.3866, so both
  speculative paths beat it. Packed draft-generation time fell 4.386→3.515
  ms/round, but accepted drafts/round fell 1.161→0.892 and verification
  rounds rose 3420→3900. All five per-repetition speed ratios favored
  ordinary; paired descriptive 95% intervals were [0.891,0.973] request and
  [0.884,0.970] decode. Every packed server logged actual CUDA dispatch.
  Ordinary and packed decoded texts matched 60/60; both mismatched target-only
  on the same two prompts (50/60 matches). Raw manifest/report/records hashes,
  memory and category spread are in
  [the native end-to-end report](../../experiments/native-end-to-end-5080.md).
  A separate six-request token-ID diagnostic ran without altering the timed
  data. Ordinary and packed token arrays were identical on both mismatching
  prompts; relative to target-only, first differences were index 33 (target
  ID 438, speculative 264) and index 123 (target 362, speculative 5737).
  Diagnostic summary SHA256
  `1806e2f230d0a07f4957bce7a3ae8e7ba6a79ee4b6b982fe24dec5c585043bf6`,
  50-file artifact seal SHA256
  `bcbf739dfbd4409059ff333bf36977f2a70d02d217193a568fe028c68165ad68`.
  Its supervisor exited and final three GPU samples were 3,050 MiB used,
  12,928 MiB free, 0%; no project process or SSH/tmux session remained.
  The RTX 2080 Ti address is still absent.
- A final four-request `n_probs:5` diagnostic reproduced the same token IDs
  at the two native divergence positions. In target-only logits the
  speculative IDs ranked second, 0.0009633 nats below ID 438 at `prose-04`
  and 0.0165080 nats below ID 362 at `reasoning-04`. The ordinary EAGLE API
  exposes only placeholder/empty candidate probabilities for accepted draft
  tokens, so the actual verifier logits at those positions are unavailable.
  Numerical batch sensitivity is plausible, but the native baseline
  target-equivalence cause remains unproven. Summary SHA256
  `cba59b6090bfacdf5713482f1b134ace97b2583ab195d521031008912696e0fe`,
  artifact-manifest SHA256
  `693415914997891220807069260c9f6df38e8d48f3a984ea7bb0661b5d27b723`.
  No benchmark data was changed. Final three RTX 5080 samples were 3,047 MiB
  used / 12,931 MiB free / 0%; no project processes, SSH, or tmux remained.
- `a595ee8` added a guarded `--proxy-correctness` mode to the SM75 MMA probe.
  A new combined SM75-SASS/compute_75-PTX binary compiled on CUDA 13.1 and
  ran the virtual instruction on the RTX 5080 once under supervision:
  21 cases and 880 integer dots exactly matched an independent dense CPU
  sign reference across full/partial tiles, dirty tails, and audited K widths.
  This is **5080 proxy numerical evidence only**; the program's default mode
  still rejects non-SM75 devices, and the RTX 2080 Ti address remains absent.
  Remote `results/sm75-mma-proxy-20260924/` manifest SHA256
  `e6d9dadaf116dcd92f52b1fba3b187044c8c4c5f2eb56a50b0d95e777158250a`
  preserves compiler/source/header/binary/SASS/PTX/stdout/state hashes. The
  one supervised proxy run exited zero; final GPU samples were 3,047 MiB
  used / 12,931 MiB free / 0–1%, and all project processes/SSH/tmux ended.
- One final bounded integrated-kernel profile attempted Nsight Compute
  2025.4.1 on a 16-token packed-draft request. The request executed and logged
  CUDA W1A1 dispatch, but the profiler returned `ERR_NVGPUCTRPERM`; no kernel
  counts/durations were produced. Partial artifacts are sealed under remote
  `results/packed-w1a1-ncu-profile-20260924/`, manifest SHA256
  `1f3268c4dce267609fdc7cec9b355dcd14d382cab780e1539c81e36def2cd031`.
  The operator stopped after this one attempt. Three GPU samples returned to
  3,047 MiB used / 12,931 MiB free / 0%, no project processes remained, and
  SSH/tmux closed. No additional 5080 GPU job is planned while the 2080 Ti
  address is pending.
- The remaining hardware gate is documented in
  [the RTX 2080 Ti runbook](../RTX2080TI_RUNBOOK.md). Its current address is
  absent from the shared host registry; the username `philip` is known and
  the user was asked asynchronously for the IP. No 2080 Ti run has started.
- A bounded native baseline verifier diagnostic was completed independently
  of the sealed 5080 benchmark. Pinned llama.cpp `common_sampler` samples
  target verifier rows at `tools/server/server-context.cpp:3897-3920`; a
  temporary opt-in trace commit
  `b06892b697cba17a1e47ec3fc770716a91a8e699` on fork branch
  `feat/verify-logit-trace` logs raw target top-two logits and emitted/draft
  status only at generated positions 33/123 under
  `W1A1_TRACE_VERIFY_LOGITS=1`. The parent gitlink remains `92bc706`;
  isolated submodule worktree `/private/tmp/llama-verify-logit-trace` owns
  the diagnostic patch, with a local CPU server build/diff check passed.
  `/root/cuda_acceptance_operator` owned the 5080 for one two-prompt
  ordinary-EAGLE trace using a separate CUDA build and supervised run.
- The remote diagnostic branch was fetched over HTTPS after the submodule's
  SSH `origin` fetch failed; no build or GPU request had started at that point.
  The separate SM120a trace-server build ran under supervised ID
  `verify-logit-trace-build-20260924`, PID/PGID 626/626, completed 356
  build steps and exited zero. The isolated two-prompt run
  `verify-logit-trace-run-20260924` (PID/PGID 5245/5245) also exited zero.
  Both ordinary output ID arrays reproduced the sealed baseline. Eight
  opt-in verifier rows were logged at positions 33/123 with no null logits.
  The two emitted rows showed the verifier ranked the speculative token first
  by 0.008074 (`prose-04`) and 0.000729 (`reasoning-04`) raw-logit units;
  both drafts were rejected. Thus the immediate mismatch is the batched
  target verifier choosing a different argmax than target-only decoding,
  with numerical sensitivity plausible but exact cause unproven. See
  [the trace report](../../experiments/native-verifier-trace-5080.md).
  Remote `results/verify-logit-trace-20260924/artifact-manifest.json` SHA256
  `9e64ede2a31c50417b843d5245ab45aff206453f1df61eee36bd001835aac898`
  seals responses, logs, summary, commands, environment, and file hashes.
  Remote submodule was restored clean to `92bc706`; all project processes
  and tmux sessions ended. Repeated final GPU samples were 3,046 MiB used /
  12,932 MiB free / 0%. Do not treat the temporary diagnostic build as timed
  evidence.
- A compile-only integrated SM75 check ran on the 5080 host under
  supervised ID `integrated-sm75-cuda-build-20260924`, owned by
  `/root/cuda_acceptance_operator`. It uses a separate ignored
  `build/llama-cuda-sm75` tree (leaving the measured SM120 binary intact),
  CUDA 13.1/GCC 15, `CMAKE_CUDA_ARCHITECTURES=75`, and the previously audited
  private CUDA header copy. No SM75 binary will be executed on the 5080.
  The build completed 358/358 steps (supervisor exit 0), producing the CUDA
  backend test/server binaries and shared library without executing them.
  Supervised disassembly also exited zero: the integrated W1A1 CUDA PTX has
  `xor.b32`/`popc.b32`, and its SM75 SASS contains `POPC`/LOP3 in the XOR
  kernel. The sealed remote
  `results/integrated-sm75-backend-20260924/artifact-manifest.json` SHA256 is
  `05fddb64d3b47f1b5c5086a02fc6a257ccb35944bc65be04668f8f1466c7d76c`.
  No SM75 binary was executed; the 2080 Ti runtime gate
  remains pending.
- `1e4e76a` integrated a versioned real-head parity fixture generator and
  standalone CUDA checker from an isolated Sol worktree (cleaned after
  integration). It uses eight deterministic rows from the ignored BF16 QAT
  capture and **all 32,000** packed GGUF head rows, with exact sign/dot
  reference and F32 output tolerance. Local synthetic/full-size CPU fixture
  checks and `make check` passed 78 tests; the remote real capture fixture and
  CUDA execution followed after the integrated SM75 build was sealed. The
  supervised real-head checker passed on RTX 5080: exact activation sign words
  and **256,000/256,000 integer dots** across eight captured vectors and all
  32,000 output rows; zero scale/output tolerance failures. Max absolute
  activation-scale/output errors were 1.1920929e-7/9.53674316e-7. This is
  standalone CUDA arithmetic parity, not an integrated GGML real-logit or
  SM75 runtime result. All three fixture/compile/checker supervisors exited
  zero. The sealed remote
  `results/real-head-fixture/artifact-manifest.json` SHA256 is
  `dd3b732e5f9188589b12700f3e059275a6bc1b7e81e407655cc066854b7a6d98`;
  it records capture/GGUF/fixture/binary/stdout/state/code hashes and selected
  row indices. Three final 5080 samples returned to 3,046 MiB used /
  12,932 MiB free / 0%; no project process or SSH/tmux remained.

## Next action after the 5080 verifier trace

The 5080 evidence is sealed and GPU ownership is released. Keep the
diagnostic llama.cpp branch `feat/verify-logit-trace` published on the user's
fork and its clean local worktree at `/private/tmp/llama-verify-logit-trace`
until the trace artifact has been reviewed; the production parent gitlink
remains `92bc706`. The remaining hardware gate is actual SM75 execution and
same-device timing on the RTX 2080 Ti. Its address is blank in the shared host
registry, so wait for the user's current address, then follow
[the runbook](../RTX2080TI_RUNBOOK.md) with a single GPU owner. Do not infer
Turing performance from SM120 proxy correctness or compile-only evidence.

The native report now includes the zero-draft screening bound from sealed
totals: holding packed acceptance and the measured non-draft residual fixed,
removing all 13.707 s of draft time would yield 158.79 decode tokens/s,
1.195× ordinary on this 5080 run. Reaching ordinary instead requires packed
draft time no higher than 2.351 ms/round, 33.1% below its measured value.
The main comparison, overview, and decision log now link the raw verifier
trace and preserve the target-equivalence limitation. This is analysis of
existing artifacts; no new GPU run or target/verifier code change occurred.

## Integrated binary-MMA candidate in progress

Because a portable-kernel negative result on SM75 would leave the Tensor Core
alternative unmeasured, `/root/sm75_mma_probe_impl` owns a bounded, isolated
llama.cpp `w1a1.cu` implementation based on production `92bc706`. It will
keep portable XOR/POPCOUNT as default and expose an explicit opt-in MMA path;
no GPU is assigned to the implementation worker. In a separate worktree
`/private/tmp/llama-w1a1-mma-tests`, branch `feat/w1a1-mma-tests` commit
`f52dbc7` (published to the user's fork) expands the independent scalar
reference checks to full 8×8, partial/multi-tile 9×10, and head-like
320×1 shapes. The Mac CPU backend passed 8/8 cases, including dirty K tails.
The implementation commits `e3026d3` and `6d18bde` add an opt-in
`GGML_CUDA_W1A1_MMA=1` path and cache the selector/device capability without
altering the portable default. The numerical kernel passed host emulation for
8×8, 9×10, and 320×1 shapes; no CUDA run has yet occurred. The test commit
was cherry-picked into the implementation branch, now published as
`feat/w1a1-sm75-mma` at `9bb01a6`. Sole 5080 owner
`/root/cuda_acceptance_operator` is assigned a supervised SM120a proxy
build/correctness check plus compile-only SM75 SASS inspection, restoring the
remote production checkout afterward. Actual SM75 runtime and performance
still require the RTX 2080 Ti address.
The remote candidate fetch verified full SHA
`9bb01a682ed4ba5e506870c8a38338589830b164`; parent/production gitlink
remain unchanged. The isolated SM120a CUDA build supervisor
`binary-mma-sm120-build-20260924` (PID/PGID 655) completed 358/358 steps
and exited zero in `build/llama-cuda-bmma-sm120-20260924`, using CUDA 13.1
and the audited private include root. The portable-default supervisor
`binary-mma-sm120-portable-20260924` and opt-in supervisor
`binary-mma-sm120-mma-20260924` both exited zero with 8/8 scalar-reference
W1A1 backend cases; logs distinguished portable XOR/POPCOUNT from binary
MMA dispatch on compute capability 1200. The model-level supervised smoke
`binary-mma-server-smoke-20260924` (PID/PGID 5346) exited zero: candidate
MMA logged dispatch and matched all 85 generated `prose-04` token IDs and
`stop` reason from the sealed production portable response. These are 5080
proxy correctness checks, not SM75 runtime or speed evidence. GPU returned
to roughly 3,040 MiB used / 12,938 MiB free / 0% after the request.
A separate supervised SM75 compile-only job
`binary-mma-sm75-build-20260924` completed 157/157 steps and exited zero.
The inspected candidate library's SM75 SASS contains
`BMMA.88128.XOR.POPC`; this image was never executed on the 5080. All six
supervised jobs exited zero. The sealed remote manifest at
`results/binary-mma-sm120-20260924/artifact-manifest.json` has SHA256
`f0a578fc6bb4324f7a37387a202a70bdceed2dbd16497fa944dc1868a6ae62e7`
and preserves 25 file hashes, exact commands, environment, raw outputs,
supervisor states, and cleanup. Full SM75 SASS dump SHA256 is
`d86b73b9507d28c7ff1ae270189b5d761e3ada08237b1745c392a2fa8c495b37`.
Remote parent and production submodule were restored clean to `92bc706`;
all project processes stopped and both SSH/tmux connections closed. Final
GPU samples were 3,040 MiB used / 12,938 MiB free / 0%. The candidate
remains a published experimental branch, not the production gitlink. See
[the integrated MMA report](../../experiments/integrated-binary-mma-5080.md).
The next meaningful hardware action is to obtain the current RTX 2080 Ti
address and run default/opt-in correctness plus same-device timing there.

## Four-variant 2080 Ti comparison preparation

The paired runner was originally fixed to target-only, ordinary EAGLE, and
portable packed-head W1A1, so it could not directly alternate the new MMA
variant on one GPU. `/root/paired_benchmark_harness` owns an optional fourth
`packed_head_w1a1_mma` variant in the runner and its tests, with explicit
per-server `GGML_CUDA_W1A1_MMA` selection and raw dispatch evidence; no GPU
is assigned. Analysis support for three or four complete paired variants was
committed/pushed at parent `ab45e49`, with dedicated MMA/portable pooled
ratios and bootstrap intervals. The runner from isolated worker `ac2d447`
was cherry-picked into main as `473e9d2`; `[evaluation] binary_mma = true`
selects all four variants, otherwise the three-variant behavior remains.
The runner saves per-variant selector environments, checks distinct dispatch
markers, and reports MMA/portable pooled speedup and text parity. A local
fake-server test needed a test-only bypass for slow reverse DNS in Python
3.14's `HTTPServer.server_bind`; the production runner did not change for
that environment issue. The combined runner/analysis suite now passes 15/15
under Python 3.14, and the worker independently passed 15/15 under the
project's locked Python 3.11. Focused Ruff checks pass. No GPU was used.
The [2080 Ti runbook](../RTX2080TI_RUNBOOK.md) now names the opt-in flag,
required dispatch fields, and MMA/portable analysis. The current 2080 Ti
address is still absent; that actual runtime and timing gate remains open.
The runner manifest now also captures the actual llama.cpp checkout commit,
status, and diff hash separately from the parent gitlink, since the 2080 Ti
candidate trial will run from published `9bb01a6` while production remains
pinned at `92bc706`. The combined focused suite passed 15/15 again under
the locked Python 3.11 environment; Ruff checks and `git diff --check` pass.

## 2080 Ti access and expanded coverage request

The user supplied the current 2080 Ti address on 2026-09-24. It was registered
only in the shared machine-local hosts file with user `philip`; it is not in Git.
A tmux-MCP SSH attempt to port 22 timed out (exit 255). The user reports WSL
installed but no Ubuntu distribution. No remote command, build, model transfer,
or GPU process has started. The access tmux session was closed. The user was
asked asynchronously to complete Ubuntu/SSH setup or provide a different port.

After reviewing the 5080 evidence, the user requested native tests on the
2080 Ti for all five previously simulated coverage settings: fusion-only,
attention-only, FFN-only, head-only, and all nine linears, with ordinary EAGLE
and target-only anchors. The native pinned branch `92bc706` implements only
head-only. `/root/full_w1a1_bridge` is preparing selectable packed conversion
and EAGLE graph routing in an isolated submodule worktree, without GPU use or
parent gitlink changes. The benchmark harness will need variant expansion and
per-group dispatch evidence before a full five-setting run. The separate
binary-MMA candidate remains experimental until real SM75 correctness.

The user also requested a QAT revisit. The earlier head-only teacher-KL pilot
trained 500 steps and improved validation KL but lowered held-out acceptance
from 1.677 to 1.565 drafts/round. The next QAT protocol must specify a new
training objective, which groups are trainable, and a fresh untouched held-out
suite before training; do not reuse the original 12 prompts for selection. A
[bounded target-aligned head-first protocol](../../experiments/qat-revisit-plan.md)
records the alignment, split, selection, stop, and wider-group gates. It is a
plan, not a new measured result.

Next: complete and review the selectable native bridge and benchmark controls;
prepare a bounded QAT plan and model/precision manifest locally. Once Ubuntu
and SSH are reachable, perform the runbook preflight, correctness gates, and
same-device measurements with one GPU owner. No 2080 Ti result exists yet.

The pinned FP16 EAGLE draft was locally quantized with `--pure` to Q4_0 and
Q8_0 GGUFs at parent `098042e`/llama.cpp `92bc706`. A GGUF readback counted
nine linears at the requested weight type in each, with four F32 normalizer
tensors and one I64 mapping. Source, output and tool SHA256 values, exact
commands, bytes, and log hashes are in the [artifact prep report](../../experiments/native-weight-only-draft-prep.md).
These are weight-only candidates, not native W4A4/W8A8 timing. Actual 2080 Ti
loader, activation/kernel precision and end-to-end measurements remain pending.

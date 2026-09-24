# Goal: complete W1A1 EAGLE research program

**Opened:** 2026-09-24  
**State:** active  
**Orchestrator:** current Codex task  
**GPU owner:** `/root/cuda_acceptance_operator` (Luna high), bounded QAT capture/training
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
  `/root/cuda_acceptance_operator` remains the GPU owner for one frozen
  exported-checkpoint held-out acceptance run; no tuning against held-out.
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

# Goal: complete W1A1 EAGLE research program

**Opened:** 2026-09-24  
**State:** active  
**Orchestrator:** current Codex task  
**GPU owner:** unassigned until the first RTX 5080 host/resource check  
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
  before any new run. No new worker or GPU job has been launched yet.
- Opening checkpoint must be committed and pushed before parallel work starts.

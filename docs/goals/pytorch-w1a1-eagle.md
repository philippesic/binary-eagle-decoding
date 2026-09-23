# Goal: PyTorch W1A1 EAGLE drafter

**Opened:** 2026-09-23  
**State:** active  
**Orchestrator:** current Codex task  
**GPU owner:** unassigned; no remote jobs are running

## Objective

Produce a reproducible PyTorch simulation of W1A1 on selected linear layers of
the published `AngelSlim/Qwen3-4B_eagle3` drafter paired with
`Qwen/Qwen3-4B`. Keep target weights and verification behavior unchanged. This
goal is about binary-operand quality and acceptance, not native speed.

## Completion evidence

- Pin model snapshot revisions and hashes; document the exact loading path,
  target feature inputs, drafter tensor names/shapes, and precision of each
  eligible linear group, including the vocabulary head.
- Provide a runnable PyTorch drafter path with selectable W1A1 layer groups.
  Its forward pass applies explicit weight and activation sign rules and scales;
  unchanged operations remain at their original precision.
- Check fake-binary linear outputs against an independent small numerical
  reference, including zero-sign behavior and scale broadcasting. Verify that
  disabling quantization recovers the ordinary drafter path within documented
  tolerance.
- Run a fixed held-out prompt suite through the unchanged target/verifier and
  record proposed and accepted draft tokens by layer-group configuration.
  Preserve raw counts, resolved config, model revisions, hardware, and commands
  in ignored `results/`; summarize evidence in `experiments/`.
- State any model-loading or hardware blocker precisely if the full acceptance
  run cannot be completed. Do not substitute synthetic values for acceptance.

## Boundaries and decisions

- Begin with post-training fake quantization in PyTorch. Quantization-aware
  training is a later research decision if measured acceptance is poor.
- Do not build GGUF W1 tensors, CUDA kernels, or end-to-end speedup claims in
  this goal. FP16 and target-only throughput anchors belong to the later direct
  comparison with native W1A1.
- The initial sign-at-zero and scaling granularity will be explicit, reversible
  experiment settings rather than assumed properties of the checkpoint.
- Do not alter the target model or shared sampling/verification semantics.

## First independent work units

1. **Graph and loading audit (read-only):** identify EAGLE-3 module names,
   shapes, target hidden-state plumbing, and the least invasive PyTorch loading
   path. Stop with a short technical finding; no code edits or GPU use.
2. **Quantizer core (isolated worktree):** implement configurable fake-binary
   linear math and focused numerical tests under `src/w1a1_eagle/` and `tests/`.
   Stop at a reviewable commit; no model downloads or GPU use.
3. **Orchestrator integration:** resolve checkpoint revisions, build the EAGLE
   adapter/evaluation path, review and integrate worker output, and own this goal
   file and `docs/STATUS.md`.
4. **GPU experiment operator (after host access):** exclusively own RTX 5080
   model downloads, environment capture, and acceptance runs through tmux MCP.
   Save raw artifacts under the configured remote project directory and report
   exact commands and results. No concurrent GPU experiment owner.

## Current checkpoint

- Goal opened and pushed on `main` at `a9e1082`. No W1A1 implementation or
  model files exist yet.
- Hugging Face API reports both checkpoints public and ungated. Snapshot pins:
  target `1cfa9a7208912126459214e8b04321603b3df60c`; drafter
  `fd331e59626c8e95c392381a16ee59d518727fbb`. Model file hashes remain
  pending until download.
- `eagle_audit` is reviewing the graph/loading path without edits. `quant_core`
  owns `src/w1a1_eagle/` and focused tests in the temporary worktree
  `/Users/pippo/.codex/worktrees/w1a1-quant-core` on
  `feature/w1a1-quant-core`; it has no GPU ownership.
- The orchestrator prepared 12 self-authored held-out prompts across prose,
  code, and reasoning in `configs/acceptance_prompts.jsonl`. They will not be
  used for training or calibration.
- The RTX 5080 host/user details are pending from the user. The local host
  registry is blank. RTX 2080 Ti is outside this goal.
- Next: review audit findings, integrate the quantizer core, build the PyTorch
  EAGLE adapter and acceptance runner, then execute the pinned run on RTX 5080.

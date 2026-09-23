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
- `eagle_audit` finished a read-only structural audit, summarized in
  `experiments/eagle3-graph-audit.md`. The pinned drafter has one decoder layer,
  uses target block-input taps `[2, 18, 33]`, owns a 32,000-row output head,
  and borrows target token embeddings. Forward parity is not yet verified.
- `quant_core` implemented the inference-only fake-binary linear core and five
  passing numerical tests. It was reviewed and integrated into `main` at
  `88c41a3`; its temporary worktree and branch were removed after push. Tests
  pass with Frameworks Python 3.11 and PyTorch 2.8.0. The project uv environment
  does not yet include PyTorch; lock the simulation environment before full runs.
- `adapter_advice` found official AngelSlim EAGLE-3 PyTorch inference code and
  recommended it as the forward implementation/oracle. Source pin:
  `0358da9c651e6a7d7ccafea26ced4b9c98d11681`. This avoids reimplementing
  head dimension 128, Q/K/V cache semantics, recurrent prenorm state, and
  offset-form `d2t` for the first candidate.
- `eagle_adapter` implemented selective wrappers for feature fusion, attention,
  FFN, and the drafter-owned head. It was reviewed, formatted, and integrated
  into `main` at `3a98c7c`; 13 local tests passed with Frameworks Python 3.11
  and PyTorch 2.8.0. Its temporary worktree and branch were removed after push.
- Source audit of the pinned AngelSlim greedy verifier found that
  `accept_length_list` excludes the target-selected seed; each tree proposes
  `total_token - 1` draft nodes. See
  `experiments/eagle3-acceptance-metric-audit.md`.
- The orchestrator added the pinned experiment config, snapshot download and
  hash script, strict official-code loader, and held-out acceptance runner at
  `6e22fae`. The runner dry run sees 12 prompts and five variants. The target
  tokenizer-only snapshot was fetched locally; non-thinking prompt lengths are
  32–56 tokens. No target or draft weight payloads have been downloaded.
- Isolated local import of the pinned AngelSlim source succeeds with Python
  3.11, PyTorch 2.14, and Transformers 5.6. A meta-device instance of the
  actual drafter accepted all nine eligible wrappers. Recent Transformers
  synthesizes a default RoPE dictionary when the raw draft config says null;
  the local loader restores `None` for this pinned unscaled checkpoint. Full
  target/drafter loading and forward parity remain untested.
- `acceptance_audit` is independently reviewing the committed loader/runner;
  it owns no files or GPU.
- The orchestrator prepared 12 self-authored held-out prompts across prose,
  code, and reasoning in `configs/acceptance_prompts.jsonl`. They will not be
  used for training or calibration.
- The RTX 5080 host/user details are pending from the user. The local host
  registry is blank. RTX 2080 Ti is outside this goal.
- Next: incorporate the runner review, lock a CUDA environment on RTX 5080,
  download and hash both pinned snapshots, then verify full-checkpoint forward
  behavior and a short deterministic target-only/speculative continuation.
  Before the held-out sweep, check hidden-state taps `[2, 18, 33]`, token/feature
  shift, absolute positions, cache rollback, and offset-form `d2t`. Produce
  per-prompt raw counts and a compact acceptance report. The RTX 5080 host/user
  details remain the only external input needed; no remote jobs are running.

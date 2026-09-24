# W1A1 QAT revisit after the RTX 5080 head pilot

**State:** proposed bounded protocol; no new training or held-out result.
The user requested a QAT revisit and native tests of every previously simulated
W1A1 coverage setting on the RTX 2080 Ti. Native post-training comparisons and
QAT are separate experiments. See the [first pilot result](qat-head-pilot-results.md)
and the [2080 Ti runbook](../docs/RTX2080TI_RUNBOOK.md).

## Why change the recipe

The first pilot optimized the binary head against the original drafter head on
cached pre-head vectors and selected the checkpoint by cached validation KL.
It improved KL from 2.506 to 1.017 but reduced online held-out accepted draft
tokens per round from 1.677 to 1.565. Even the trained head used at ordinary
precision reduced acceptance from 2.317 to 1.813. The target verifier, rather
than the original drafter head, determines which proposed tokens are accepted.
Cached-state KL also misses changes in future states and tree proposals.
These are possible explanations, not a proven causal decomposition.

## Freeze data and alignment before training

1. Create new 96-train, 24-development, and 24-final prompt manifests, balanced
   across prose, code, and reasoning. Freeze and hash them before capture or
   optimization. Split by prompt and topic/template family, not by captured
   activation row. Audit IDs and content hashes against one another, the
   original 12 held-out prompts, and the first QAT pilot's prompts. Use the
   original 12 only for historical regression reporting, never recipe or
   checkpoint selection.
2. On the pinned AngelSlim target/drafter, trace target next-token labels and
   logits alongside drafter inputs, absolute positions, recurrent depth,
   trajectory source, and `d2t`/`t2d` mappings. Before optimization, prove the
   label for each captured state is the token scored by the verifier at that
   state. Record target probability mass outside the 32,000-token draft
   vocabulary and the exact treatment of unmapped targets. A failed mapping or
   off-by-one audit blocks training; do not silently drop rows.
3. Match the training fake-binary forward, export, and native packed arithmetic
   closely enough to state their rounding differences. The first pilot used a
   BF16 fake-binary forward; the native llama.cpp comparison uses FP16 model
   tensors, F32 scales, and integer packed dots. Run a fixed-input parity audit
   before interpreting native results from a trained checkpoint.

## Bounded first experiment

- Train the head's latent weights only, leaving target weights, other drafter
  groups, vocabulary maps, and normalization frozen. Start from the original
  head, not the previously trained pilot head. Use a target-aligned next-token
  loss on mapped draft vocabulary labels, with one prespecified small KL term
  toward the original drafter distribution if alignment and coverage checks
  pass. Record each term and its weight. No objective/hyperparameter search on
  the final set.
- Keep the prior explicit sign/scale forward rule and STE unless a separate
  forward-parity failure forces a documented change. Bound the run to 500
  optimizer steps or 45 minutes. Save steps 0, 100, 250, and 500 and stop for
  nonfinite gradients, invalid target alignment, OOM, or export mismatch.
- Export each saved checkpoint and evaluate its **online** accepted drafts per
  round on the development prompts with unchanged target/verifier/settings.
  Select one checkpoint by pooled accepted/round, breaking a tie by target
  token loss. Report proposed and accepted counts, rounds, emitted tokens,
  per-prompt and per-depth spread, target-greedy mismatches, and cached KL as a
  diagnostic. If no checkpoint improves on the original untrained binary
  head, stop this recipe.
- Evaluate the selected export once on the untouched final prompts against
  original ordinary EAGLE, original untrained binary head, and the earlier
  trained pilot head. A final acceptance improvement over the unchanged binary
  path is required before spending effort on wider QAT. Do not infer native
  throughput from acceptance alone.

## Wider coverage decision

The requested native 2080 Ti matrix includes fusion-only, attention-only,
FFN-only, head-only, and all-group W1A1 **without QAT** first. Use its measured
acceptance, draft cost, and precision coverage to choose a second QAT group.
Training FFN, attention, fusion, or all groups changes recurrent states and
requires activation STE plus the pinned AngelSlim recurrent/alignment loss;
the cached-head trainer cannot be reused for that purpose. Before a wider run,
freeze a single candidate, objective, data split, time/step cap, and final gate
in an amendment here. Preserve the target-only and ordinary anchors on the
same 2080 Ti track.

## 2080 Ti execution and reporting

The Windows host currently lacks Ubuntu WSL and reachable SSH, so no 2080 Ti
QAT or inference result exists. Once accessible, first inventory VRAM and
check whether the FP16 target/draft pair fits. If not, use a separately named
quantized-target track with the same target GGUF for every compared draft.
For every trained or untrained native variant, run the correctness/dispatch
gate and at least five alternating timed repetitions with the same target,
prompts, context, KV type, sampling, and speculative settings. Preserve model,
code, build, compiler, driver, GPU, clocks/power, memory, exact commands,
prompt and output hashes, acceptance counts, and per-request timings under a
run-specific ignored results directory. Compare portable and SM75 binary-MMA
dispatch only after both pass correctness on the actual 2080 Ti.

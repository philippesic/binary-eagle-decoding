# W1A1 QAT revisit after the RTX 5080 head pilot

**State:** proposed bounded protocol; no new training or held-out result.
The user requested a QAT revisit and native tests of every previously simulated
W1A1 coverage setting on the RTX 2080 Ti. Native post-training comparisons and
QAT are separate experiments. See the [first pilot result](qat-head-pilot-results.md)
and the [2080 Ti runbook](../docs/RTX2080TI_RUNBOOK.md).

**2026-09-25 direction:** the native five-setting 2080 Ti matrix is complete.
QAT acceptance recovery is now the research priority. A trained native W1A1
candidate must be timed against **both** ordinary FP16 EAGLE and standard
Q4_0 EAGLE drafts with the same target and verifier. The Q4_0 control is a
block-scaled weight format, not the separate W4A4 quantizer. Alternative
non-EAGLE drafters are a later direction, not part of this QAT gate.

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

   This manifest gate is now frozen locally by
   `scripts/generate_qat_revisit_prompts.py` at parent commit `ded3234`.
   Ignored files under `data/qat-revisit/` have SHA256 values: train
   `80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74`,
   development `a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885`,
   and final `67575d0466b03758723b5e6be6d78237385fd3ce45b2caf67ea15c4bc83f4c62`.
   Generation audited all three sets against the original 12 and the prior
   96/24 pilot manifests by ID and exact content hash, with whole topic/template
   families assigned to one split. The generator SHA256 is
   `92d7b1bc86b559d35088c004b04c1acd3a1c66530b13e837d70761512057daf7`.
   The full project `make check` passed 85 tests after integration. No model was trained or evaluated on
   these new prompts yet.
2. On the pinned AngelSlim target/drafter, trace target next-token labels and
   logits alongside drafter inputs, absolute positions, recurrent depth,
   trajectory source, and `d2t`/`t2d` mappings. Before optimization, prove the
   label for each captured state is the token scored by the verifier at that
   state. Record target probability mass outside the 32,000-token draft
   vocabulary and the exact treatment of unmapped targets. A failed mapping or
   off-by-one audit blocks training; do not silently drop rows.

   A source audit found that `d2t[i]` is an **offset**: target token ID is
   `i + d2t[i]`. The current capture records a head-call ordinal and row, but
   not the parent tree node/path or corresponding target verifier row. The
   first head call predicts after the target-selected seed; each later call
   predicts after its draft parent node. The next capture must record these
   node IDs/paths and match each row to the verifier's next-token logits,
   including padded, pruned, EOS, and unmapped rows. A direct read of the
   pinned safetensors checkpoint (SHA256
   `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`)
   found 32,000 unique, strictly increasing absolute `d2t` IDs in a 151,936
   target-token vocabulary. `t2d` is a boolean mask with exactly 32,000 true
   entries at those IDs; its inverse consistency passed. The all-group GGUF's
   I64 `d2t` tensor exactly matched the absolute IDs. This checks the mapping
   table, not the target probability mass that falls outside it.
   The loss treatment of target labels outside draft support remains a
   research decision to fix after measuring their frequency and probability
   mass, before training.
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
  throughput or a win over the Q4_0 draft from acceptance alone.

## Wider coverage decision

The requested native 2080 Ti matrix tested fusion-only, attention-only,
FFN-only, head-only, and all-group W1A1 **without QAT**. Every setting lost to
ordinary EAGLE; head-only retained the most acceptance (0.860 accepted
drafts/round versus ordinary's 1.168), while all-group fell to 0.055. Use
these measured quality/cost results to choose any later QAT group.
Training FFN, attention, fusion, or all groups changes recurrent states and
requires activation STE plus the pinned AngelSlim recurrent/alignment loss;
the cached-head trainer cannot be reused for that purpose. Before a wider run,
freeze a single candidate, objective, data split, time/step cap, and final gate
in an amendment here. Preserve the target-only and ordinary anchors on the
same 2080 Ti track, and include Q4_0 as the second draft throughput anchor.

## Native execution and reporting

The 2080 Ti now has Ubuntu WSL, reachable SSH, and a completed FP16-target
comparison suite; no new QAT model has been trained or run natively. For a
trained native candidate, run the correctness/dispatch gate and at least five
alternating timed repetitions with the same target,
prompts, context, KV type, sampling, and speculative settings. Preserve model,
code, build, compiler, driver, GPU, clocks/power, memory, exact commands,
prompt and output hashes, acceptance counts, and per-request timings under a
run-specific ignored results directory. Include target-only for correctness
context and both ordinary FP16 EAGLE and Q4_0 EAGLE draft anchors for throughput.
The Q4_0 anchor must use the same FP16 target and verifier as the trained
candidate. Compare portable and SM75 binary-MMA dispatch only after both pass
correctness on the actual 2080 Ti.

## Primary-source clarification (2026-09-25)

The [literature cross-reference](research-cross-reference-2026-09-25.md) retains
this single 500-step/45-minute pilot. The original-draft KL objective was a
legitimate compression proxy that failed the observed acceptance gate; its
causal contribution is unmeasured. Target-argmax CE plus a fixed regularizer
is a deliberately simple greedy diagnostic, not an established best loss.
Capture target probabilities/support mass for diagnostics without adding a
loss search. A future extra hard-versus-soft target arm or training-state
refresh requires its own scoped decision; advisory alternatives do not expand
this budget.

For any later body QAT, EAGLE-3 motivates predicted-state recurrence with
matching masks/positions, not mandatory target-feature regression. Do not
interpret the earlier phrase “recurrent/alignment loss” as requiring target
hidden-state MSE. The actual pinned training path needs an implementation audit.
Keep the forward rule, STE and exporter fixed in the first supervision test;
clipped gradients and recomputed scales are not established causes of failure.

## PrismML study follow-up (2026-09-25)

The [extended research report](prism-quantization-research-2026-09-25.md) adds
representation/cast checks and conditional scale/readout diagnostics before
broader QAT. Fitting these parameters is training and must use the designated
training split, not historical/development captures merely because those files
already exist. New exposure budgets and larger trainable scope are proposed
research decisions; they do not amend this frozen 500-step/45-minute pilot.
A trained latent head's ordinary-precision regression is diagnostic only; QAT
optimizes its quantized forward, whose online acceptance remains the gate.

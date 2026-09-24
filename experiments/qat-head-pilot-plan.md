# Bounded W1A1 drafter-head QAT pilot

**Status:** capture and bounded training completed; fixed held-out acceptance
is pending. See [pilot results](qat-head-pilot-results.md). This is one bounded
quality recovery experiment, not a foundation-model training program or a
speed claim.

## Why this pilot

On the RTX 5080 held-out suite, ordinary BF16 EAGLE accepted 2.317 draft
tokens/round; post-training W1A1 on only the drafter vocabulary head accepted
1.677. The head accounted for 12.93% of one-prompt instrumented draft time.
All-group post-training W1A1 accepted only 0.202/round. Start with the isolated
head to establish a correct, exportable QAT path before attempting FFN or broad
coverage. Fusion is low priority: it occupied 1.03% of measured draft time and
had poor post-training acceptance.

## Data and fixed model path

- Create 96 training and 24 validation prompts balanced across prose, code,
  and reasoning. Keep topics and exact content disjoint from the 12 held-out
  prompts in `configs/acceptance_prompts.jsonl`. Freeze prompt IDs/hashes
  before feature capture. Held-out prompts are evaluation-only.
- With the pinned target/drafter loaded, capture the input to the drafter's
  own `lm_head` via a forward pre-hook during ordinary and untrained head-only
  W1A1 `eagle_generate` trajectories, approximately half from each. These
  vectors are already normed; do not apply another norm. Retain prompt,
  trajectory, token/tree-depth identifiers and reservoir-cap at roughly
  32,768 training rows so low-acceptance prompts cannot dominate.
- Unload the target and full drafter before optimization. Keep frozen original
  BF16 head weights to compute teacher logits on the same cached `[N,2560]`
  vectors. Preserve the published draft vocabulary and `d2t`/`t2d` maps; do
  not rebuild vocabulary mapping from the small pilot corpus.

## Trainable arithmetic and bounded training

- Train only the head's FP32 latent weights. Its forward must be bitwise equal
  on cached BF16 inputs to the current `fake_binary_linear`: `sign(0)=+1`,
  BF16 weight view, BF16 per-row mean-absolute weight scale, BF16 per-token
  input scale, BF16 sign GEMM, and the same order of scale multiplications.
  Inference `W1A1Linear` detaches cached signs and is unsuitable for QAT.
- Implement an explicit sign straight-through estimator for weight gradients
  with a clipped, scale-normalized surrogate and detached scales. State the
  backward rule and clip threshold in the report. Inputs are constants for
  this head-only pilot; activation STE is unnecessary here.
- Optimize temperature-1 KL/cross entropy between frozen ordinary-head and
  binary-head logits over the same 32,000 draft vocabulary. Compute
  softmax/log-softmax in FP32; also record teacher top-1 agreement and top-10
  overlap. Proposed initial optimizer: AdamW, lr `1e-4`, zero weight decay,
  gradient norm clip 1, batch 64–128, 20-step warmup. These are starting
  parameters, not an established optimum.
- Stop at 500 optimizer steps or 45 minutes, whichever occurs first.
  Validate every 50 steps; retain step 0 and the best validation checkpoint.
  Stop early on nonfinite values, failed forward equality, export mismatch, or
  three consecutive validations without improvement. At most one prespecified
  `3e-4` retry is allowed if gradients are finite but progress is negligible.

## Export, held-out gate, and limits

Save FP32 latent weights plus optimizer/RNG state in ignored results for
resumption. Export the selected head as BF16 into a fresh copy of the pinned
drafter checkpoint, replacing only `lm_head.weight`. Audit every other tensor,
including mappings and the intentional absence of `embed_tokens.weight`, and
hash both training and inference artifacts. Evaluate **the exported checkpoint**
once on the fixed 12 held-out prompts with unchanged target/verifier/settings.
Compare accepted/proposed nodes and accepted/round to ordinary and untrained
head-only W1A1 under the same BF16 verifier, retaining target-greedy mismatch
flags. A recovery toward roughly 1.9 accepted/round is a provisional quality
gate for considering native head execution; it is not an end-to-end speedup
threshold. If the bounded pilot fails, report that result and stop this recipe
instead of expanding a hyperparameter search.

This first pilot reconstructs the original drafter head on captured inputs. It
does not train the recurrent EAGLE body. Later FFN/all-group QAT would need
the pinned AngelSlim offline training alignment and recurrent loss; do not
infer that from a successful head pilot.

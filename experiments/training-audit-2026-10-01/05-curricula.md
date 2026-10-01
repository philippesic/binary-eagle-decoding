# Precision, recurrence curricula and training budget

Read-only audit dated 2026-10-01. Local source is pinned to parent
`d725fc88d52fc49841c2b20128d176c1e7f486d8`; the live capture/training experiment
was neither changed nor exercised. Paper versions inspected: ParetoQ v2,
EAGLE-3 v3, BiT arXiv:2205.13016, BiBERT arXiv:2203.06390, HASS v2 and DSQ
arXiv:1908.05033. Official EAGLE-3 repository links below were inspected on
2026-10-01 at current `main`; they are explanatory sources, not pinned project
dependencies. No model, large capture payload, GPU, remote host or sealed final
was accessed.

## Verdict and ranking

Keep direct hard-forward A1 as the reference. The best precision-curriculum
candidate is **one short A8 warm-start followed by a substantial A1 phase**,
tested against direct A1 at equal total GPU time. Do not adopt an A8→A4→A1
ladder by default. A warm-start is plausible initialization assistance, not an
established acceptance improvement or a substitute for enough A1 exposure.

| Rank | Candidate | Recommendation |
| --- | --- | --- |
| 1 | Direct hard A1, full captured horizon, measured learning curve | Preserve the current control and establish actual learning before changing it. |
| 2 | Short A8→A1 with all nine binary-weight projections retained | Prepare an isolated two-arm comparison; keep most compute for A1. |
| 3 | Full-horizon CE with mild early-depth weighting | Cheap preparation; test separately if native root acceptance is poor or deeper losses compete with it. |
| 4 | Bounded refreshed student trajectories | Strong mechanism for token-distribution mismatch, but incurs new native capture/audit and changes corpus ancestry. |
| 5 | Progressive projection quantization or state distillation | Defer until a projection/depth failure is evidenced. |
| 6 | A8→A4→A1 and soft-forward annealing | Defer: added schedules and transition risks without project evidence. |

## What the current trainer actually learns

`continuous_qat.py` presents each audited round to independent A8 and A1
models/optimizers, serially interleaved on the same device. It is not a warm-start
or mutual-distillation system. `recurrent_qat.py:270-278` initializes latent
sign parameters to **±0.5**, using signs from the original dense F16 weights,
and mean-absolute row scales. Dense magnitudes are discarded from the latent
initialization. Candidate-D uses a separate group-128/A16 representation;
capturing D trajectories does not initialize either student from D.

All nine selected linears use hard binary weights. A8 uses signed absmax,
nearest-even activation quantization; A1 uses signs and mean-absolute activation
scales. Activation-scale statistics are detached. Dense F32 matrix operations
simulate the quantized forward. Norms, target/verifier, borrowed embeddings and
vocabulary map stay frozen. AdamW uses sign/scale LR 1e-3/1e-5, β=.9/.999,
zero decay, norm clipping 1 and 100 paired warmup updates, then constant LR.

The crucial recurrence distinction is explicit in
`recurrent_rollout.py:72-84,141-158`: accepted-prefix cache reconstruction uses
current parameters under `no_grad`; proposal states and K/V are attached, and
the next feature is the student's `result.pre_norm`. However, the next token is
`row['proposed_token_id']` from the capture. Student argmax never chooses the
training token chain. This is **current-state recurrence on forced captured
tokens**, with truncated context gradients, on an off-policy candidate-D token
distribution. It is not one-step frozen-state fitting, fully differentiable
context reconstruction, or current-student own-token rollout.

The trace audit checks exact token/prefix ancestry, positions, masks and mapping.
Mean CE supervises supported native target labels. Unsupported labels remain in
quality denominators; an unsupported earlier row still carries state gradients
to supported later rows. The capture horizon is at most five. The label-only
continuous corpus has no full target probabilities. Native development uses
the actual exported A8/A1 choices; its CE diagnostic reuses D-prefix rounds.
Those metrics measure different distributions and must remain separately named.

Local outcomes warn against inferring acceptance from loss. The row-A16
100-step calibration covered just two prose prompts/495 labels, had all 18
gradients finite/nonzero and **zero sign flips**; its 0.923-second mean is not
an A8/A1 completion estimate. The earlier head-only KL pilot improved validation
KL 2.506→1.017 while held-out accepted drafts/round fell 1.677→1.565. Neither
result proves a curriculum ineffective, but both reject loss-only success gates.
Sources: [continuous preparation](../continuous-w1ax-preparation.md),
[calibration](../w1ax-row-a16-calibration-5080.md),
[head pilot](../qat-head-pilot-results.md).

## Primary evidence and transfer limits

| Source | Documented result | What it supports / does not establish |
| --- | --- | --- |
| [BiT, §§4–5.5, Appendix A.1](https://arxiv.org/pdf/2205.13016) | BERT/GLUE: direct binary distillation average 71.0; W32A32→W1A2→W1A1 reaches 73.5 without augmentation. Progressive initialization with the FP teacher scores 73.4. A three-stage A4→A2→A1 route did not improve the two-stage A2→A1 route; a closer low-precision intermediate can beat a more accurate high-precision intermediate. | Good evidence for staged initialization; weak evidence that intermediate-teacher replacement is essential. It is task-specific BERT distillation with different quantizers, not equal-GPU-hour EAGLE hard-CE or evidence that A4 is the best project intermediate. |
| [ParetoQ v2, §2](https://arxiv.org/html/2502.02631v2) | MobileLLM-125M fixed 100B-token allocation peaks near 90% FP/10% QAT. Across models, lower-bit QAT requires more adaptation: about 30B-token saturation for ≤2-bit versus 10B for 3/4-bit, with greater weight movement. | Supports pretrained initialization and substantial final low-bit adaptation. This is mainly weight quantization with embedding/output exceptions, not W1A1 recurrent EAGLE. Its 90/10 pretraining split is not an A8/A1 fine-tuning prescription, and absolute token counts do not transfer. |
| [BiBERT, §3.3, Table 2](https://arxiv.org/pdf/2203.06390) | Fully binary BERT GLUE average 63.2 versus baseline 50.4. Direction-matching distillation uses normalized Q/K/V similarity patterns; direct attention-score matching can misdirect optimization. | Supports selective, scale-aware distillation rather than indiscriminate tensor MSE. Changes attention architecture and quantizes operands this project leaves wider; no curriculum-only or drafter-acceptance attribution. |
| [EAGLE-3 v3, §§2–3](https://arxiv.org/html/2503.01840v3) | Removing feature prediction improves first-token acceptance, but without recurrent training second-token acceptance suffers. Training-time test feeds draft states back, enabling data scaling. | Strongly supports the attached student-state unroll already present. Also argues against restoring a mandatory target-feature regression constraint to an unconstrained EAGLE-3 state. |
| [Official EAGLE-3 model](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/traineagle3/cnets.py), [trainer](https://raw.githubusercontent.com/SafeAILab/EAGLE/main/eagle/traineagle3/main.py) | Model lines 835,849–852 retain its output states while shifting existing input tokens, targets and masks. Trainer lines 278–279 weight step loss by 0.8^i. | Primary code confirms own-state exposure does not imply own-token sampling. The weighting is a documented implementation choice, not an isolated causal ablation. |
| [HASS v2, Table 4](https://arxiv.org/html/2408.15766v2) | LLaMA2-Chat-7B, mean acceptance length at T=0: Top-K EAGLE-2 4.78, Align-2 5.11, Align-3 5.15, Align-4 5.16, Align-5 5.11. | Supports recurrent exposure but shows deeper is not always better. Different model/objective; alignment steps are not the local chain's exact depths. |
| [DSQ, Table 3](https://arxiv.org/pdf/1908.05033) | ResNet-20/CIFAR-10 W1A1: basic binary 82.46%, binary DSQ 83.80%, piecewise DSQ 84.11%. | Soft quantizer continuation can help CNN optimization; no language/recurrent/acceptance evidence, and project hard-native forward must still be assessed. |

## Mechanisms and practical choices

**Precision staging.** A8 can adapt binary signs/scales while preserving richer
activation information; transferring that solution may reduce the simultaneous
weight-and-activation damage faced by direct A1. Conversely, A8 can specialize
to amplitudes that A1 removes, consuming the scarce budget before the hard task.
Keep weights binary throughout; an FP warm-start would test a different question.
An A4 bridge is available in `W1AxContract`, but the continuous paired driver is
A8/A1-specific, and A4 adds export/readiness/transition work. Two days should
prepare A8→A1 first, not a ladder search.

Transfer latent signs **with their magnitudes**, effective scales, optimizer
state and exposure counters; never reconstruct latent parameters as fresh
±0.5 from exported signs. Distance to zero controls future sign flips and is
part of the learned initialization. Preserving Adam moments is the simplest
single-policy comparison; β2=.999 can retain A8 gradient-scale history for
roughly 1,000 updates. Record pre/post-switch gradients, effective steps and
sign flips. Resetting moments may help after the distribution jump, but requires
a matched reset control or a separately labeled later experiment. Do not reset
warmup/counters silently. No new hysteresis band is warranted merely to suppress
transition flips: it changes binary optimization, and is a separate hypothesis.
At ±0.5 initialization and 1e-3 LR, a nominal unit-normalized, same-direction
update needs about 500 full-LR steps to cross zero. This is an intuition, not an
Adam bound; clipping, changing gradients and warmup alter it. A short calibration
therefore cannot adjudicate binary sign learning.

**Recurrence curriculum.** Prefer retaining all five available steps with mild
early weighting before implementing 1→3→5 truncation. For a simple chain with
conditional acceptance probabilities a_j, expected accepted length is
Σ_k Π_{j≤k} a_j: early improvements affect multiple survival terms. This is a
planning approximation, not an exact objective for the native tree. A normalized
0.8^depth-weighted CE is a reasonable single alternative; preserve unsupported
denominators and report effective weighted exposures. Root-only training loses
later-state robustness. Shorter unrolls also leave prefix reconstruction cost
unchanged, so savings must be measured rather than inferred from horizon ratios.

**Feature/state distillation.** Existing raw target features are conditioning
inputs, not valid targets for the student's unconstrained recurrent state. Native
D `heads.f32` offers aligned normalized pre-head states in source captures, but
the continuous label provider does not expose them as loss targets. A D-state
auxiliary needs an exact execution-row/prefix join and explicit provider changes;
it would anchor A1 to a different group-scaled binary drafter. A frozen A8/FP
draft teacher can be replayed on unchanged captured tokens without target
recapture, but costs another teacher forward and needs teacher-state/cache
contracts. Q/K/V similarity distillation needs corresponding teacher operands,
not merely existing target features or head states. Defer this complexity unless
plain CE reveals a specific state instability.

**Trajectory refresh.** Changed student tokens require fresh native target
features/labels on those exact prefixes. The existing stopped-training refresh
path provides explicit checkpoint/drafter bindings and audited ancestry; use a
new corpus contract, never append silently to exact resume. Reserve bounded
compute for this if offline loss improves while native acceptance stalls.
Naive scheduled sampling with old D labels is invalid.

## Smallest decisive later experiment

Prepare, but do not alter the current run: two independent arms from identical
row initialization and frozen provider order, (D) direct A1 for B GPU hours,
(C) A8 for 0.2B then A1 for 0.8B. The 20% choice is a bounded proposal, not a
literature optimum. Use the same hard CE, full horizon, optimizer policy and
global warmup; transfer all state at the switch. This needs a new explicit
experiment/transition manifest and A1 export gate; changing precision is not an
exact resume of the immutable existing paired run. Do not combine depth weighting,
distillation, optimizer changes or refreshed data into this first test.

Measure B after actual A8/A1 throughput calibration. Match total supervised
training GPU time; separately report capture/audit/export/dev GPU time and total
project cost. Charge pre-existing A8 training to the curriculum's cumulative
cost even if its marginal reuse cost is small. Record unique prompts/rows,
repeated presentations, per-depth labels and precision-phase token exposures.
Frozen shard order is balanced by prompts, not necessarily by rows or GPU time;
check each precision phase's domain/length exposure before attributing a gain.
Equal hours can yield unequal tokens; show both time-matched and common-token
learning-curve points. The 10k-prompt/100k-row admission floor is not a budget
sufficiency guarantee, and paired steps are two optimizer updates, not one.

Use predeclared checkpoints and the fixed 24 native dev prompts for screening.
Confirm an apparent improvement on a predeclared balanced larger unsealed dev
subset (e.g. 96 prompts), with aggregate accepted/proposed/round counts,
per-depth conditional acceptance and domains. Require native export/math gates,
unchanged verifier behavior, no new greedy stream discrepancy, and positive
paired prompt-level uncertainty against direct A1 before calling the curriculum
better. Beating Q4_0 acceptance is the primary project quality gate; curriculum
improvement below it is evidence for this recipe only. Measure latency and total
throughput separately with the evaluation protocol's repeated, order-balanced
native runs. RTX 5080 results cannot establish SM75 gains. Keep finals sealed;
a small inconclusive result ends this bounded comparison rather than launching
a generic schedule grid.

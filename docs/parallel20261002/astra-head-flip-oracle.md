# Exact fixed-state binary-head flip diagnostic

Fifth-slate independent advisor research, October2. Proposal only; no source,
recipe, model or data changed. Intended follow-up to the completed sign-inertia
study, which found that flip counts and lower CE do not establish persistent
useful decisions. Root owns any team dispatch. No GPU/SM75 claim.

For one fixed current-student head input, retain integer activation codes a,
fixed positive token scale beta, binary head row signs s, row scale alpha,
integer dot d=sum(s_j*a_j), and the fixed bias/affine contribution. Flipping
coordinates F in that row changes its dot exactly to

    d_new = d - 2 * sum(s_j * a_j for j in F).

Recompute the changed row's logit using the declared native multiplication and
addition order. Do not simply add an algebraic delta to the old F32 logit:
rounding can differ. A1/A4/A8 codes fit this identity; A16 is excluded from the
integer certificate. Row scales, activation parameters and states stay fixed.
Affine midpoint sum and ordinary bias stay fixed; head has no FC correction.

For old/new row logit z_r and z'_r, define delta=z'_r-z_r and old softmax q_r.
Exact real-arithmetic hard-label CE change is

    delta_CE = log(1 + q_r * (exp(delta) - 1)) - 1[y=r] * delta.

Use stable logsumexp/log1p branches at extremes. Independently recomputed
old/new logsumexp is the oracle, not this formula's potentially unstable
floating evaluation. Several flipped bits in the same row still produce one
changed logit. Multiple affected rows require summing their denominator
changes, with a stable implementation; single-row candidate evaluation is a
useful bounded initial contract.

Top1/rank-margin change needs only the new row logit and the best competing
row, with the actual vocabulary map/tie convention. A full head GEMM is not
necessary for each nominated coordinate. Existing gradient information can
nominate at most32 candidate coordinates; nomination is not an update policy.
The diagnostic evaluates candidates without modifying parameters or optimizer.

Independent scalar JavaScript/F64 algebra check executed31,104 cases: all16
four-bit sign states,81 ternary code rows,4 single-bit flips, row scales
0.1/1/3 and labels0/1 with two fixed competitor logits. Direct CE difference
and the formula agreed to maximum3.233e-13. This validates the algebra only;
it is not a PyTorch/native F32 gate or a model quality result.

## Proposed bounded implementation gate

Use actual RowBinaryLinear synthetic inputs and export quantizer rules to
retain raw integer dots/codes. Compare candidate projected rows against full
head reevaluation, including zero scale, exact ties, nonidentity d2t map,
learned threshold/clip and fixed affine midpoint. Require correct CE/margin
classification and native-order F32 comparison; clarify errors around tiny
rounding margins instead of claiming exact real-arithmetic ordering survives.
One independent validator should reconstruct dense flipped rows without using
the delta implementation. Report work saved as avoided candidate GEMMs, not
GPU latency. Source-bound implementation stays under an isolated research
path if root launches it.

## Interpretation limits

The current body/state is held fixed and only head signs change. The resulting
decision certificate applies at this prefix. A changed native token changes
future embeddings/private states and may improve or damage the complete
trajectory; no suffix or full-session acceptance prediction follows. CE can
improve without crossing a rank boundary. Unsupported target labels carry no
mapped hard CE and must retain the separate acceptance denominator.

This is useful as training telemetry before proposing a new binary optimizer:
it can distinguish latent movement, sign flips and immediate head-decision
changes. Any policy that actually chooses flips is a new user-owned recipe.

## Constructive overshoot case

Independent scalar calculation: one variable head row has logit2*s, competing
row logit0, s=+1, four identical code1 inputs with three labels on the variable
row and one on the competitor. Mean CE is0.6269280110; derivative with respect
to relaxed s is+0.2615941560. Its linear prediction for flipping s by−2 is
−0.5231883119, but actual flipped mean CE is1.6269280110: loss increases by1
and top1 accuracy drops75%→25%. This is finite-step overshoot of a convex
one-dimensional loss, not a wrong autograd derivative. The exact candidate
oracle detects it. No optimizer/training run was performed for this example.

For one changed row per token, a cheaper sufficient bound is also available:
let delta_i be the row logit change and A=mean((q_ir−1[y_i=r])*delta_i).
Convexity and the coordinate Hessian bound q*(1−q)<=1/4 give

    A <= mean_delta_CE <= A + mean(delta_i**2)/8.

Use the same supported-row weights as the declared loss. A>=0 rules out CE
improvement; an upper bound<0 certifies improvement. The gap can be wide for
binary flips, which is why exact logsumexp evaluation remains useful. This
bound applies to fixed states and row-only logit changes; no recurrence or
native acceptance certificate follows.

## Native confidence boundary, if a later nonzero p_min is selected

The native EAGLE sampler ranks a top10 list. For top1-versus-runner-up logit
margin m>=0 and ordinary untempered softmax on that list, its top confidence
lies between 1/(1+9*exp(−m)) and1/(1+exp(−m)), assuming at least two finite
retained candidates. The lower bound uses at most9 competing terms, each no
larger than exp(−m); the upper uses the runner-up term alone. With one finite
candidate the confidence is1. These bounds describe proposal stopping only,
never target acceptance.

At p_min0 every finite case passes, so this cannot improve the current frozen
zero-threshold quality policy. For future nonzero thresholds, exact top10
confidence requires its actual retained scores and tie behavior; a rank-only
head-flip oracle should not claim policy preservation from unchanged top1
alone. Any logit processors must be applied before computing this margin.
This is a source-semantic observation, not a new policy recommendation or test
extension requested from the completed head-flip team.

# Robust finite-update loss certificate

Independent fifth-slate advisor derivation. No live optimizer/loss changes,
models, captured rows or GPU work. This extends completed directional teacher
uncertainty to the actual finite change made by a binary candidate.

Hold exact target prefix/state, compact teacher metadata and row masks fixed.
Student logits may change nonlinearly, including recomputed body states under
the same forced captured token prefix. Let q_old and q_new be their softmaxes,
known teacher masses a, unknown mapped tail allocation t with sumT, outsideo,
and M=1-o. Then the represented source objective changes by

    delta_L(t) = [sum_known a_i*log(q_old_i/q_new_i)
                  + sum_tail t_i*log(q_old_i/q_new_i)] / M.

This is exact at the represented coefficients and independent of the STE
Jacobian. It remains valid when stored S=sum(a)+T differs from M; it does not
silently renormalize. Source finite arithmetic still needs a numerical margin.

The existing simplex/capped-tail extremizer applies to log-probability ratios.
Return certain improvement, certain regression, or unidentified update, with
attaining teacher completions. The two endpoints describe one scalar loss
difference and are jointly valid witnesses, unlike assembling independent
coordinate extrema. Batch rows use the unchanged actual objective mask and
reduction. Different teacher prefix/state or missing current-policy trajectory
cannot be repaired by this certificate.

## Concrete separation from gradient certainty

Teacher: knownp0=.375, unknown mapped tailT=.625, cap.375 on each of two tail
tokens. Thus p1 lies in[.25,.375]. Student q_old=(.3,.4,.3). Change only row1
logit by−4, yielding q_new approximately
(.4939684350,.0120631300,.4939684350).

Executed scalar F64 calculation gives directional derivative along this change
in[−.6,−.1] for every feasible teacher, yet exact finite mean CE difference in
[+.5013108562,+1.0013108562] for every feasible teacher. The entire uncertain
teacher set agrees on infinitesimal improvement and finite-step regression.
Binary flips are finite moves; gradient-sign certainty alone is insufficient.
This is a synthetic mathematical example, not evidence of a live optimizer bug.

## Bounded production-shaped diagnostic proposal

Reuse the completed teacher uncertainty extremizer, add old/new logits input,
and independently enumerate the tiny capped-polytope vertices. Verify actual
compact loss differences lie in nominal intervals with a declared CPU numeric
margin. Realize the example with a RowBinaryLinear sign flip and appropriate
fixed bias/scale; compare source hard forwards without taking a real-data update.
Test one improvement, one regression and one ambiguous case, plus the existing
tiny-M rejection policy. Do not repeat the full certifier sweep.

This supports offline checkpoint comparisons or a shadow update diagnostic.
Automatically rejecting/choosing optimizer steps would define a new training
recipe, require its compute charged and need human approval. It cannot certify
native acceptance, throughput, or counterfactual future prefixes.

## Independent implementation review

Reviewed the owner's production-shaped offline certifier, proof and actual tiny
head/body forward examples. The implementation correctly computes M using the
source F32 subtraction, preserves accepted S!=M, and aggregates repeated exact
teacher identities before taking extrema. This matters: opposite old→new and
new→old occurrences of the same teacher should cancel exactly, rather than
artificially receiving independent adversarial tails.

Found one admission ambiguity: a mapped-top-k caller could supply an arbitrary
tail cap smaller than the cutoff proved by stored masses. That stronger cap
narrows the teacher set without evidence and could make an unjustified certain
classification. Owner agreed to reject such overrides and accept only the
derived cutoff plus declared rounding allowance; separately attested full-vocab
cutoffs remain explicitly caller-owned. One focused negative control was added.
The prior constructive example uses the justified cap and remains unchanged.

# Fifth slate: deployment objectives and teacher uncertainty

Direct human steering resumes sustained CPU research. This overrides the earlier
recommendation to stop replenishing. Root already owns activation-reuse and
snapshot-copy production-shaped teams; this advisor does not duplicate them.
Latest root usage88% used/12% remaining; local control last observed87% used,
stop=false, reset1791049896 unchanged. Check control before each substantial
chunk; stop on research_stop, <=1% remaining or actual original reset. No
GPU/SSH, real optimizer work, live source edits, or final/held-out payloads.

## Packet 1: full-session reward and stopping objective

Owner Sol high, independent Luna high. Paths:
`research/parallel20261002/session_objective/{reference,validation}/` and
`experiments/parallel20261002/session_objective/`.

The overnight objective derived accepted-prefix reward at frozen roots and
explicitly left changing round-root occupancy and full-session runtime outside
its estimator. Existing break-even arithmetic freezes measured rounds; it
does not certify a new policy whose accepted length changes subsequent roots.
This is the new question, not another local CE-versus-prefix counterexample.

Implement an exactly enumerable finite-session reference matching native
greedy draft/target sample-and-match semantics, correction/bonus emission,
terminal EOS/cap behavior and dynamic next-round roots. Use tiny synthetic
transition tables with declared draft/verify/catch-up costs. Compare exact
total emitted/time with frozen-root accepted-prefix and local ratio rankings.
Produce a constructive ranking reversal, or prove the proposed reversal
impossible under its stated constraints. Do not choose arbitrary huge costs
and call them measured: include constant-cost controls and realistic restricted
cost forms, clearly separate synthetic counterexamples from historical timing.

Derive a ratio-of-expectations or finite-session objective with reward
emitted minus rho times cost, stating which root occupancy and continuation
terms are needed. Show that a one-round reward without those terms cannot
generally substitute for full-session optimization. For stochastic gradients,
state required behavior support/current-policy labels and distinguish them
from deterministic argmax's locally constant objective. No new training
algorithm is enabled. Deliver one practical reporting/selection diagnostic
and the minimal future capture fields, rather than a general RL framework.

Gate: brute-force enumeration equals independent dynamic programming for
counts, EOS, corrections, costs and proposed policy comparisons; reward-ratio
root/fixed-point calculation agrees with direct totals. A special-case no-win
result is useful. Explicitly demonstrate when historical fixed-root evidence
cannot identify the new session ranking. No additional data capture requested.

Primary-source starting point:
[SpecDec++](https://arxiv.org/html/2405.19715v2) formulates adaptive length via
an MDP, but uses ratio/residual verification and a constant forward-cost model.
Our native match verifier and measured nonconstant shape costs require a fresh
applicability check, not reuse of its threshold theorem unchanged.
[Sutton et al.](https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)
provides policy-gradient background, not permission to infer missing support
or omit state occupancy. Read relevant primary derivations before claiming
an exact learning gradient.

## Packet 2: compact-teacher loss and gradient uncertainty certificate

Owner Sol high, independent Luna high. Paths:
`research/parallel20261002/teacher_uncertainty/{reference,validation}/` and
`experiments/parallel20261002/teacher_uncertainty/`.

`recurrent_qat.compact_probability_loss` assigns unknown mapped tail uniformly
and conditions by1-outside_mass. The overnight capture-bounds reference bounded
acceptance dot products, not CE or sign-gradient reliability. Current primary
hard_ce route avoids this particular approximation; report that clearly. The
compact mode is an implemented conditional objective whose future admission
needs an actionable uncertainty diagnostic.

Derive tight CE and directional-gradient intervals for all teacher tails
consistent with stored top-k masses, mapped tail total and outside mass. Begin
the exact stated unconstrained tail simplex. Add a per-token top-k ordering cap
only if actual producer semantics establish it; mapped top-k and full-vocab
top-k are different. For fixed student logits, teacher CE is linear in teacher
mass; directional derivatives through the attached surrogate Jacobian also
admit extremal linear bounds. Do not claim each coordinate's extremum is
simultaneously attainable as one teacher distribution.

Implement the certifier around actual compact_probability_loss and a tiny
RowBinaryLinear reference. Supply exact feasible teacher distributions with
identical stored metadata where uniform-tail gradient direction is reliable,
unidentified, or opposite a feasible full-teacher gradient. Show a binary sign
crossing counterexample only if the actual optimizer/update realizes it;
otherwise report gradient ambiguity without inflating it to a decision claim.
No hyperparameter sweep or trained model required. Include zero tail,
one-element tail, near-zero mapped mass and masked rows.

Gate: independently enumerate simplex vertices for tiny vocabulary and match
analytic loss/directional bounds; uniform approximation lies inside the
interval; exact top-k constraints, when declared, tighten rather than loosen
bounds. Compare true autograd directional VJP to the formula for the declared
surrogate; do not finite-difference a hard quantizer. Return an eligibility
report: certain direction, ambiguous direction, invalid metadata, or missing
producer contract. Do not replace the live loss or infer native target
probabilities after processors from raw softmax summaries.

## Advisor continuation

After packets dispatch, independently challenge the full-session theorem
assumptions and the teacher interval geometry. In particular, distinguish
equal pointwise acceptance under constant costs from a changed root/cost
distribution; and distinguish per-coordinate uncertainty from a jointly
realizable update. Check source-bound decision implications, not additional
coverage counts. Record refinements here and send them to root/owners.

No CPU gate here establishes acceptance or throughput against Q4_0. The
deliverables specify what future evidence would make such a comparison sound.

## Independent advisor findings while packets dispatch

### Compact-mass tolerance changes the exact derivative

The API accepts total masses within absolute1e-4 of one, then normalizes by
M=1-outside_mass. Let S=sum(topk_probs)+tail. For accepted metadata S need not
equal M exactly. The exact logit derivative of the implemented weighted CE is
(S*q-p)/M, not q-p/M. A certifier using normalized probability algebra must
either enforce S=M in its declared input contract or report this distinction.
Tiny mapped mass amplifies tolerance/cancellation; M<=0 is masked. This is a
source/API issue in optional compact training, not evidence against hard_ce.

### Entropy alone does not certify this native acceptance

[AdaEDL AppendixB](https://arxiv.org/html/2410.18351v1) derives a ratio-verifier
bound using target/draft divergence, then substitutes an empirical relation
between cross-entropy and draft entropy. The substitution is an approximation,
not a distribution-free inequality. It also concerns sampled ratio/residual
verification, whereas current EAGLE emits its top candidate for target matching.

Independent scalar CPU/JavaScript calculation, synthetic distributions:
q=(.999,.001), p=(.001,.999). Draft entropy0.0079072551; proxy
1-sqrt(.5*H)=0.9371221219. Actual deterministic-draft native match probability
is0.001; ratio overlap is0.002. Thus neither acceptance quantity has that
unconditional entropy lower bound. In the point-mass limit, draft entropy is
zero while a disjoint target accepts nothing. No model or dataset was used.

The pinned native top10 candidate probabilities are also renormalized truncated
scores; they do not expose full-vocabulary draft entropy. Positive global logit
scaling changes this confidence/entropy without changing greedy top1. At frozen
p_min=0 that scaling cannot improve native acceptance through early stopping;
at nonzero p_min it changes the proposal policy and must be evaluated as such.
This rules out importing an entropy certificate into a W1A1 promotion gate
without target-alignment evidence. It does not reject entropy as an empirical
feature in a separately evaluated stopping predictor.

### Full-session source interpretation

SpecDec++ theorem3.1 supplies a sufficient stop condition with an unknown
policy-dependent continuation constant, not a ready-made universal threshold.
For repeated sessions use total emitted/total time (ratio of expectations),
not mean per-request ratios. For a fixed greedy output length, emitted tokens
are fixed and total cost alone determines the ranking. EOS/cap rules must be
defined before claiming this reduction. Sent these constraints to root for
the session-objective owner.

### Session theorem review completed

Read the team's complete report while its validator finished. Its clean
counterexample needs no arbitrary state surcharge: a six-token greedy target,
fixed draft length2, and two globally prefix-consistent drafts yield initial
accepted counts2 versus1 but complete rounds4 versus3. The rival roots expose
different future failures. An all-correct alternative is identical at the
observed first root yet wins globally, establishing partial identification.

The team's no-win control is sound under its explicit stronger premises:
pointwise correctness-set inclusion, stateless globally consistent drafts,
fixedK, no early draftEOS/confidence, constant nonnegative draft/verification
cost coefficients and no context/catch-up cost. Its next-root function is
monotone; induction orders roots and both round/draft counts. I checked the
terminal/cap argument and advised preserving executed verifier rows separately
from bonus emission. No theorem blocker found. For exact target-law preserving
stochastic policies, expected total emitted count is also invariant with fixed
prompt/EOS/cap rules; the team was asked to state this explicitly.

### Additional concrete diagnostic prepared

[Exact head-flip oracle](astra-head-flip-oracle.md) derives a fixed-state
candidate-flip CE/rank diagnostic without another full head GEMM. An executed
scalar overshoot case shows the surrogate gradient predicting an improving
binary flip while actual finite-flip CE worsens by1 and accuracy falls75%→25%.
This is a new bounded implementation packet for root consideration, not a
selected optimizer. Its strict limitation is local head-state attribution;
changed native tokens alter later states.

## Advisory checkpoint at95% used

Control remains stop=false and original reset unchanged. Continue only original
ordinary allowance until the shared stop policy fires. No GPU/remote job owned.

- Session objective completed; exact reversal and restricted no-win proof
  reviewed. Native costs remain synthetic and no policy is selected.
- Teacher uncertainty completed; source-produced tiny-M masking and actual toy
  gradient/sign reversal are explicit. Main hard_ce route is unaffected.
- Head-flip oracle completed and independently checked; advisor required the
  same native baseline for CE deltas, separate legacy A4/A8 drift, exact dot
  range, and declared tie limitations.
- Finite-update certifier completed. Advisor caught an unjustified stronger
  mapped-top-k cap and owner rejected it before finalization. See
  [finite update note](astra-finite-update-certificate.md).
- Native session trace completed. Advisor reviewed root/cache/terminal/replay
  and shared timing semantics, then caught the adapter's replayed-versus-new
  proposal counter mismatch; owner corrected it. See
  [native trace note](astra-native-trace-review.md).

Next concrete CPU admission packet sent to root: compile the full native server
with the exact trace extension in an isolated CPU-only source/build directory.
The real emitter already compiles in a stubbed fixture, but callback insertions
have not passed full-server typechecking. Reuse the owner and independent build
operator, record exact compiler/source/patch hashes, use bounded threads, and
do not load models, run a server or infer CUDA performance. QAT integration
and live-source ownership remain separate.

## Stopped on direct human instruction

Shared control observed97% weekly used, research_stop=true,
reset_observed=false, reason: Direct human: stop all research and research teams;
leave QAT running. Advisor stopped immediately after this observation. No
research command/session/subagent/GPU/remote job is owned by this advisor.
All advisory findings are saved in this file and linked advisor notes. The
full native server compile was pending/in progress under its separate owner;
advisor makes no completion claim. Root/monitor notified. No further research
will resume without new human authorization. QAT ownership/work is unaffected.

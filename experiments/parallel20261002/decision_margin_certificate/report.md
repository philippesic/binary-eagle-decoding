# Conditional decision certificate

Deliverable: dependency-free F64 executable in
`research/parallel20261002/decision_margin_certificate/reference/certificate.py`.
Acceptance: independent exhaustive tiny-box vertices reproduce analytic
probability extrema and ordered greedy stability; malformed contracts and
nonfinite values fail closed. This is a numerical deployment gate prototype;
production source, recipes and GPU execution are unchanged.

For reference logits l and EXTERNALLY supplied maximum coordinate errors e,
form lower/upper endpoints l-e and l+e, expanded by the caller's separate
absolute logit arithmetic allowance. The ordered reference winner w is stable
iff its lower endpoint exceeds every earlier competitor's upper endpoint and
is at least every later competitor's upper endpoint. Otherwise return ambiguous.
Endpoint arithmetic is expanded one binary64 representable step for nonzero
radii; zero-radius exact ties retain their order. Mapping must be injective and
fixed. Input/state identity, vector and mapping SHA256, error provenance and
arithmetic allowance provenance appear in the result.

At positive fixed temperature T, p_i decreases with each other logit and
increases with its own. Its tight real-arithmetic minimum sets coordinate i
low and all other coordinates high; its maximum reverses those endpoints.
Compute centered exponentials BEFORE division by T and expand the endpoints
by the caller's absolute probability arithmetic allowance. Each coordinate has
its own attaining vertex: collecting upper bounds does not create a probability
vector. F64/libm rounding is not proved here; a safety claim is conditional on
the externally justified arithmetic allowance enclosing implementation error.
Numerically collapsed/overflowed finite box endpoints fail closed. Common
1e300 offsets and opposite-sign 1e308 logits are covered by owner checks.

The contract requires the entire fixed processed normalization domain, stable
processor state and ordered token mapping, processor identity, first-in-order
tie rule, and finite positive temperature. Missing or raw top-k domains return
processor_contract. No probabilities are invented from raw top-k telemetry.
Temperature zero (greedy native temperature mode) is outside the combined
softmax API: it returns invalid rather than claiming a positive-temperature
probability distribution. Greedy results here describe ordered processed
logits, not stochastic sample-and-match acceptance or cache-state equivalence.

Read-only native source observation: llama.cpp HEAD
`9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`,
`src/llama-sampler.cpp:1053` selects index zero and updates only on strict `>`;
the nonpositive-temperature implementation at line270 follows the same ordered
tie rule. `common/sampling.cpp:344-411` builds the configured chain of filters,
penalties, temperature and final distribution sampler. Filters can reorder or
change the support. Native CPU tie scope is therefore current candidate order,
not globally lowest token ID; backend argmax parity is unverified. The existing
native-round adapter declares private cache/feature/processor state unresolved,
so its event schema alone cannot satisfy this certificate's processor contract.

Constructive RMS failure: with10000 coordinates, logits `[1e-5,0,-1,...]`,
increase only coordinate1 by2e-5. The greedy winner changes from0 to1 while
RMS error is2e-7, fifty times smaller than the original margin1e-5. Maximum
coordinate error2e-5 exposes the flip. RMS must not be passed as e. Empirical
CPU errors cannot establish universal CUDA coordinate bounds.

Fixed-J directional expectation is deferred; it is optional and unnecessary
for this bounded acceptance check. No unknown true-J claim is made.

Integration needs: review and integrate this isolated research prototype and
validator/report only. A future native admission must supply complete processed
logits, fixed domain/order/state ancestry, externally justified coordinatewise
maxima and separate numeric allowances. No CUDA gate or hardware performance
result follows from these synthetic CPU checks.

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
Radius addition and endpoint arithmetic are each expanded one binary64
representable step for nonzero radii; zero-radius exact ties retain their order. Mapping must be injective and
fixed. Input/state identity, vector and mapping SHA256, error provenance and
arithmetic allowance provenance appear in the result.

At positive fixed temperature T, p_i decreases with each other logit and
increases with its own. Its tight real-arithmetic minimum sets coordinate i
low and all other coordinates high; its maximum reverses those endpoints.
Compute centered exponentials BEFORE division by T and expand the endpoints
by the caller's absolute probability arithmetic allowance. Each coordinate has
its own attaining vertex: collecting upper bounds does not create a probability
vector. Prefix/suffix exclusion normalizers and a second-largest anchor for
removing the largest coordinate evaluate all extrema in O(vocabulary size),
without subtraction of nearly equal mass totals. F64/libm rounding is not proved here; a safety claim is conditional on
the externally justified arithmetic allowance enclosing implementation error.
Numerically collapsed/overflowed finite box endpoints fail closed. Common
1e300 offsets and opposite-sign 1e308 logits are covered by owner checks,
including temperature 1e308 (finite normalized separation despite overflowing
unscaled subtraction). A half-difference fallback handles that case.

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

Constructive RMS failure: with 10000 coordinates, logits `[1e-5,0,-1,...]`,
increase only coordinate 1 by 2e-5. The greedy winner changes from 0 to 1 while
RMS error is 2e-7, fifty times smaller than the original margin 1e-5. Maximum
coordinate error 2e-5 exposes the flip. RMS must not be passed as e. Empirical
CPU errors cannot establish universal CUDA coordinate bounds.

Fixed-J directional expectation is deferred; it is optional and unnecessary
for this bounded acceptance check. No unknown true-J claim is made.

Integration needs: review and integrate this isolated research prototype and
validator/report only. A future native admission must supply complete processed
logits, fixed domain/order/state ancestry, externally justified coordinatewise
maxima and separate numeric allowances. No CUDA gate or hardware performance
result follows from these synthetic CPU checks.

Owner verification: 9/9 unittest checks pass on Python 3.11.3, Darwin arm64,
CPU only, F64 floats; Ruff lint/format pass. Independent validation finally
passes 10/10 checks; its fixture census is in validation/.
Owner commits: `c615bc8` implementation and `920d43c` arithmetic hardening.
No native backend invocation or device measurement was performed.

Final implementation `359765e` adds linear-time normalization; independent
validator now passes 10/10 checks in 0.002 seconds. A separate complete 32000
coordinate synthetic smoke returns stable and 32000 intervals in 0.0533 seconds
on the same Darwin arm64 CPU/Python binary64 environment. This is a local
prototype usability check, not native inference throughput or CUDA evidence.

Independent validator commit `c3e027d` covers 18 tiny boxes / 168 exhaustive
vertices and final certificate SHA256
`dd06cc138cfed102712b3e72165889acb74ebb55dc077c138e38e61e5355a8c5`.
Its CPU checks establish the formulas on those fixtures, not a universal
arithmetic error bound. Final control `research_stop=true`; all research has
stopped with no persistent process or device allocation. Only checkpoint/push
preservation and orchestrator integration remain.

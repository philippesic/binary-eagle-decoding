# Binary scale fitting on RTX 5080

**Status:** complete, 2026-09-27 UTC. The approved experiment is finished; no follow-up research was started.

## Result and next research choice

**Fitted scales recover substantial acceptance, but this recipe remains well below both controls.** D reaches 0.424965 accepted drafts/round, 3.585× A, closing 33.35% of the A→FP16 gap while retaining 40.96% of FP16 acceptance. D beats A, B and C on each of the 24 prompts, and trails FP16 and Q4_0 on each of the 24. This is descriptive evidence from a development-selected screen, not a final-set generalization claim.

| Path | Accepted drafts / round | First proposal | Prefix depth 2 | Prefix depth 3 |
| --- | ---: | ---: | ---: | ---: |
| FP16 EAGLE | 1.0374 (1,552/1,496) | 57.98% (861/1,485) | 28.79% (425/1,476) | 12.35% (181/1,466) |
| Q4_0 EAGLE | 1.0415 (1,555/1,493) | 58.37% (865/1,482) | 29.24% (431/1,474) | 12.23% (179/1,464) |
| A: row mean | 0.1185 (323/2,725) | 11.65% (315/2,703) | 0.30% (8/2,682) | 0.00% (0/2,660) |
| B: group mean | 0.1611 (423/2,625) | 15.50% (404/2,606) | 0.74% (19/2,585) | 0.00% (0/2,564) |
| C: fitted row | 0.2899 (685/2,363) | 25.13% (589/2,344) | 3.66% (85/2,325) | 0.35% (8/2,307) |
| D: fitted group | 0.4250 (909/2,139) | 34.23% (727/2,124) | 7.54% (159/2,108) | 1.00% (21/2,090) |

Each table cell preserves numerator/denominator. Accepted drafts/round pools verifier-accepted drafts over all completed non-replay native rounds, including rounds with no proposal. Prefix survival at depth k is `count(accepted >= k) / count(proposed >= k)`; it is not conditional on accepting the preceding depth. The extra target token is never an accepted draft. No accepted token was omitted at terminal stopping in this screen. No-proposal round counts were FP16 11, Q4_0 11, A 22, B 19, C 19, D 15.

**Correctness:** all 168 requests completed. Every path emitted 128 raw IDs per prompt (3,072/path), matching target-only exactly on all 24 prompts. All 144 speculative per-prompt boundary-state audits passed. This is empirical agreement on this workload, not universal losslessness. The differing round counts reflect accepted chunks completing the same output in fewer rounds.

**Recommendation:** retain D as the stronger binary reference and propose a bounded common-history body/readout diagnostic before performance promotion. First cross ordinary/D bodies and output heads on identical forced token histories; then, if approved, fit one regularized readout on frozen D-body training states. The present own-history acceptance results do not locate the remaining error in body versus head. D’s depth-three survival is only 1.00% versus FP16’s 12.35%; scale fitting recovered roughly one-third of the control gap, not control-level quality.

Fusion’s fixed-budget approximate solve limits any claim that scale-only recovery has been exhausted. No additional solver sweep, readout fit, QAT, A1 change, architecture pivot or kernel optimization was started. The next scope/budget decision remains user-owned. No throughput or speedup is claimed.

## Frozen experiment

The approved screen isolates scale representation and fitting across all nine draft linears. All four candidates retain the same original BF16 checkpoint signs, canonical GGUF Q/K row permutation, A16 boundary inputs, F32 scales, target and native greedy chain. Only scales were fitted; signs and non-scale parameters stayed fixed. No readout fitting or full-model QAT was performed.

| Candidate | Scale granularity | Scale choice |
| --- | --- | --- |
| A | One per output row | Original mean absolute magnitude |
| B | One per 128 weights | Mean absolute magnitude |
| C | One per output row | Activation-output-fitted |
| D | One per 128 weights | Activation-output-fitted |

The 218,234,880 selected weights use 65,280 row scales or 1,704,960 group scales. The grouped F32 reference has 1.25 bits/selected weight including scales; this is a controlled quality implementation, not standard Q1_0 or a throughput result. All nonselected norm tensors and draft-to-target mapping are unchanged. Target embeddings and the native attention, normalization, residual and cache machinery remain in their existing precision.

All seven development paths use the same FP16 Qwen3-4B target and native target-greedy sample-and-match verifier on RTX 5080. The six speculative paths use D=5 and p_min=0. Shared settings are temperature=0, seed=42, thinking disabled, context=2048, concurrency=1, target/draft KV=F16, no prompt-cache reuse, maximum 128 output tokens with EOS honored. One deterministic pass per 24 development prompts; no throughput repetitions. CUDA graphs are disabled and round/state diagnostics enabled for every path. The 24 reserved-final prompts are untouched.

FP16 and pure Q4_0 EAGLE are same-device controls. All nine Q4_0 matrix types were audited; Q4_0 is the standard weight-only CUDA control, not an A16 or W4A4 candidate. The four binary candidates use explicit F32→F16→F32 input values, sequential F32 signed sums, F32 scale products, and (for grouped scales) an ordered F32 sum over groups. No binary weights are materialized in F16.

## Calibration and solver

All 96 training prompts generated 32 tokens each using ordinary EAGLE with explicit A16 boundaries. Actual inputs to all nine linears were captured. A fixed selection of 32 evenly spaced rows per layer per prompt yields 3,072 examples/layer, spanning prefill and recurrent calls. Inputs are uncentered. The reference output is original BF16 weights promoted to F32, in canonical GGUF layout, applied to the post-cast inputs.

For each output row, `Z[t,g] = sum(sign(w_i) * x[t,i])` over group g. Fit `Zs` to the original output, with `s >= 0`. The row solution is the exact scalar nonnegative ridge formula. The group solver uses feasible monotone restarted accelerated projected gradient, with 32 output rows/batch, at most 512 iterations and a relative projected-gradient tolerance of 1e-6. Ridge strength is 1e-4 times the mean diagonal of ZᵀZ/N, centered at the mean-absolute anchor. These choices were frozen before fitting. Every fitted row had a direct unregularized SSE fallback to its anchor; no row needed it. Fitting uses F32 BLAS with TF32 disabled and F64 residual sums. This is a numerical least-squares surrogate, audited separately against native reduction order.

All 65,280 C rows met tolerance. D met tolerance on 62,720/65,280 rows. All 2,560 fusion rows reached the 512-iteration cap; their maximum relative projected-gradient residual was 1.76374e-5. D is therefore a feasible bounded approximate fit, not a claim of fully converged NNLS. Zero group scales are permitted by the nonnegative constraint. No post-result solver extension was performed.

The dense source-cast audit found maximum BF16→F16 error 2.98023224e-8 and 141 sign-at-zero changes from tiny negative weights rounding to negative zero. All four binary artifacts retain original BF16 signs, not signs derived from that rounded dense control. Exported signs/scales exactly match the fitted arrays, and every nonselected tensor is unchanged.

Normalized calibration squared error is Σ‖prediction−reference‖² / Σ‖reference‖²:

| Layer | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| fc | 0.9740 | 0.4630 | 0.2898 | 0.1589 |
| output | 0.1685 | 0.1665 | 0.0725 | 0.0548 |
| blk.0.attn_q | 0.2748 | 0.2625 | 0.2276 | 0.1569 |
| blk.0.attn_k | 0.2853 | 0.2067 | 0.2075 | 0.0948 |
| blk.0.attn_v | 0.2685 | 0.2103 | 0.2176 | 0.1095 |
| blk.0.attn_output | 0.3004 | 0.2347 | 0.2470 | 0.1481 |
| blk.0.ffn_gate | 0.1778 | 0.1758 | 0.1021 | 0.0795 |
| blk.0.ffn_down | 0.2541 | 0.2517 | 0.2238 | 0.1584 |
| blk.0.ffn_up | 0.2035 | 0.2015 | 0.1468 | 0.1123 |

These errors use fixed ordinary training states. Development acceptance, including changed recurrent states, determines selection.

## Validation

- CPU and RTX 5080 CUDA operator gates each passed 112/112 cases, including group boundaries, strided inputs and non-F16-exact source values. These checks make no SM75 claim.
- Ordinary versus explicit A16-cast native execution matched output IDs, all proposal/acceptance rounds and boundary-state records on three historical prompts. Each path covered 75 completed zero-accept, 100 partial-accept and 3 full-accept transitions, plus 3 initial seeds.
- Legacy-format A and version 2 A matched all output IDs, proposal/acceptance rounds and boundary-state records on those same three historical prompts. Their tensor payloads are identical.
- All four exported artifacts passed real-input native/scalar replay on 16 fixed captures across all nine layers, 430 sampled outputs each (1,720 total), with zero absolute or relative error.
- Separate native-versus-F32-BLAS sampled discrepancies were finite, with maximum |Δ|/(1+|native|) 1.37680e-5. This is a bounded, shape-dependent arithmetic audit, not full fitting-batch equivalence.
- Boundary traces verify retained feature copies, positions and draft-cache truncation extent. They do not constitute bytewise comparison of cached K/V tensors. The existing native recurrence is used throughout; no Python tree evaluator is substituted.

## Per-prompt development acceptance

Values are accepted drafts/round. Full counts, raw IDs and per-round/state records are preserved in the artifact index below. IDs omit the common `qat-revisit-development-` prefix.

| Prompt | FP16 | Q4_0 | A | B | C | D |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prose-household-energy-01 | 0.984 | 0.924 | 0.050 | 0.124 | 0.296 | 0.396 |
| prose-household-energy-02 | 0.954 | 0.984 | 0.134 | 0.104 | 0.296 | 0.366 |
| prose-pollinators-01 | 1.082 | 1.117 | 0.165 | 0.176 | 0.323 | 0.549 |
| prose-pollinators-02 | 1.048 | 1.082 | 0.085 | 0.095 | 0.270 | 0.396 |
| prose-pollinators-03 | 0.984 | 0.924 | 0.095 | 0.144 | 0.296 | 0.337 |
| prose-maps-and-navigation-01 | 0.814 | 0.814 | 0.104 | 0.176 | 0.283 | 0.396 |
| prose-maps-and-navigation-02 | 0.716 | 0.693 | 0.067 | 0.124 | 0.233 | 0.309 |
| prose-maps-and-navigation-03 | 0.924 | 0.841 | 0.076 | 0.114 | 0.283 | 0.411 |
| code-numeric-arrays-01 | 0.984 | 1.048 | 0.104 | 0.104 | 0.245 | 0.396 |
| code-numeric-arrays-02 | 1.048 | 1.016 | 0.144 | 0.187 | 0.221 | 0.396 |
| code-numeric-arrays-03 | 1.117 | 1.082 | 0.221 | 0.233 | 0.337 | 0.494 |
| code-resource-management-01 | 1.117 | 1.117 | 0.210 | 0.233 | 0.337 | 0.351 |
| code-resource-management-02 | 0.954 | 0.984 | 0.085 | 0.134 | 0.283 | 0.366 |
| code-resource-management-03 | 0.896 | 0.868 | 0.144 | 0.165 | 0.165 | 0.337 |
| code-search-and-order-01 | 1.309 | 1.190 | 0.104 | 0.144 | 0.460 | 0.512 |
| code-search-and-order-02 | 0.984 | 1.048 | 0.114 | 0.176 | 0.198 | 0.380 |
| reasoning-comparative-evidence-01 | 1.016 | 1.117 | 0.104 | 0.165 | 0.296 | 0.443 |
| reasoning-comparative-evidence-02 | 0.671 | 0.693 | 0.067 | 0.124 | 0.245 | 0.283 |
| reasoning-comparative-evidence-03 | 1.190 | 1.228 | 0.165 | 0.270 | 0.337 | 0.568 |
| reasoning-deductive-classification-01 | 1.646 | 1.646 | 0.095 | 0.176 | 0.283 | 0.512 |
| reasoning-deductive-classification-02 | 1.117 | 1.190 | 0.114 | 0.165 | 0.309 | 0.512 |
| reasoning-deductive-classification-03 | 1.016 | 1.082 | 0.144 | 0.210 | 0.296 | 0.628 |
| reasoning-optimization-01 | 1.309 | 1.228 | 0.134 | 0.198 | 0.380 | 0.568 |
| reasoning-optimization-02 | 1.540 | 1.646 | 0.155 | 0.165 | 0.351 | 0.427 |

## Artifact identity

Native runtime: `2e8d2e8dac6354798037e4e9aca455cd898bf06f`; fitter: parent `0258aac`; replay extension: parent `8b81c53`. Runtime and parent changes were published before evaluation. RTX 5080, CUDA compiler 13.1.115, driver 615.71.08, PyTorch 2.14.0+cu130. The existing private CUDA/glibc compatibility header was verified and reused; system headers were not changed.

| Artifact | SHA256 |
| --- | --- |
| FP16 target GGUF | `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6` |
| BF16 draft source | `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e` |
| FP16 draft GGUF | `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1` |
| Q4_0 draft GGUF | `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280` |
| A GGUF | `b454ecd93ba70b02ad57147ee5ee136344997a51cec699520e0b0f237a58e597` |
| B GGUF | `0d499be1d89c1faba6f0015a16337d3902d85538638b0be2487de66f5fc41041` |
| C GGUF | `6c57bff661615a9230d2f02ffb712ade4a3c435579ee78753cd0762a9855e34e` |
| D GGUF | `10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf` |

## Preserved evidence and reproduction

Full remote archive: RTX 5080 host, `/home/philip/binary-eagle-decoding/runs/binary-scale-fitting-5080-artifacts-20260927/`. It retains all training/cast captures, source/output hashes, fitted models and per-row scale/error/convergence arrays, Q4 control, every request/response/raw ID, round/state traces, build and supervisor logs, native binaries and environment. Source/target models remain in the host project’s `models/` directory. The completed temporary remote checkout was removed after preservation.

Local copy: `/Users/pippo/github/binary-eagle-decoding/results/binary-scale-fitting-5080/`. Its `artifact-index.json` records hashes and path relocations. The local copy includes every development request and response, full raw output IDs, fitted GGUF/NPZ artifacts and all sampled replay evidence. The larger complete calibration captures remain in the remote archive. Original absolute execution paths in raw manifests are intentionally retained; use the index’s relocations.

The fixed development prompt SHA256 is `a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885`; training is `80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74`. No final prompt was evaluated.

| Evidence file, relative to archive `results/` | SHA256 |
| --- | --- |
| `scale-fit/report.json` | `a06268592d9eb0e5980e22fdaa1b4996d4fe6763d6af763baa75f7a8164d74d9` |
| `scale-development/summary.json` | `6c47d86eaccdc049879c9ac19fd62e855534263dcb2e6fd9fe39e4c549f233e2` |
| `scale-development/analysis.json` | `a2d00ed2874a6f13bc5295fd88f9db964f3d5f4afad6bc345380abc7e77d49bf` |
| `scale-validation/replay-summary.json` | `21eab895ae7e2ebc705e82bc464ba6550cf3226195169431a23e2b0c5cabb6d8` |
| `scale-native-hashes.txt` | `076d20dd8e18519b89740754d5b426c9b0cab1c51b18d53e78d1205954024dc7` |

Reproduction uses [capture/screen runner](../scripts/run_binary_scale_screen.py), [fitter/exporter](../scripts/fit_binary_scales.py), [native replay](../kernels/w1ax-replay/README.md), [analysis](../scripts/analyze_binary_scale_screen.py), and the frozen `configs/binary_scale_*.json` files. Exact commands are in each archived supervisor `state.json` and per-variant manifest. The executed development parent revision was `08eaa79`; subsequent report/formatting commits do not alter the measured arithmetic.

Final independent review recomputed the totals from the per-request files and confirmed 168/168 completed requests, no replay rows, no accepted-but-unemitted drafts, all raw-ID matches and all boundary-state checks. The focused Python suite passed 16 tests, and changed Python files passed Ruff. Every owned supervisor and server stopped. Final RTX 5080 checks showed no compute processes and 0% utilization; no RTX 2080 Ti connection or run was made.

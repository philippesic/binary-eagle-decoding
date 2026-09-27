# W1Ax development policy grid: complete results

All 12 predeclared cells completed: **11,520 measured requests**, comprising
24 development prompts × five repetitions × eight paths × 12 cells. No reserved
QAT-final prompts or QAT training were used. This appendix accompanies the
[main results](w1ax-activation-precision-results.md) and
[frozen protocol](w1ax-activation-precision-plan.md). No cell is dropped for an
unfavorable rate or output mismatch. The interrupted D2/p0.0 attempt a1 is
excluded; its complete replacement a2 is used. Every other cell uses a1.

The hardware is **RTX 2080 Ti, SM75, 11,264 MiB VRAM, Ubuntu 24.04 WSL2**.
The target/verifier is FP16 Qwen3-4B, with FP16 KV, context 2048, concurrency one,
greedy target sampling, seed 42, thinking disabled, 128-token output cap, two
warmups, and five measured repetitions. Variants rotate with one server at a
time. FP16/Q8_0/Q4_0 EAGLE use corresponding draft weight formats; Q8_0/Q4_0
execute with CUDA Q8_1 activation conversion, not research W8A8/W4A4. W1A16,
W1A8, W1A4, and W1A1 share the same packed-sign weights/F32 row scales across all
nine selected draft linears. Their activation paths are FP16 SIGNADD, INT8,
four-plane bit-serial, and XOR/POPCOUNT respectively; ordinary-precision
exceptions and operator validation are detailed in the main report.

All four W1Ax paths remain below both FP16 and Q4_0 anchors in the fixed-policy
comparison and in the comparison of independently selected cells. Selection here
means the **highest observed pooled development decode rate**, not an optimal
policy or demonstrated policy benefit.

## All cells: pooled rates

Rates are output tokens divided by summed server-reported decode seconds, or by
summed full request-wall seconds including prefill. They are not averages of
per-request ratios or maximum online serving capacity. Each entry pools 120
requests; every path/cell emitted 15,360 tokens. A16/A8/A4/A1 abbreviate the four
W1Ax paths. Displayed values are rounded to three decimals from the source JSON.

Decode tokens/s:

| Cell | Target | FP16 | Q8_0 | Q4_0 | A16 | A8 | A4 | A1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1/p0.0 | 60.440 | 78.647 | 79.944 | 81.105 | 48.862 | 55.506 | 53.452 | 54.132 |
| D1/p0.1 | 60.464 | 78.452 | 79.689 | 81.412 | 48.855 | 55.444 | 53.482 | 54.183 |
| D1/p0.3 | 60.396 | 77.698 | 78.487 | 79.894 | 47.906 | 53.651 | 52.055 | 53.146 |
| D2/p0.0 | 65.882 | 82.327 | 84.511 | 85.304 | 41.708 | 49.224 | 48.274 | 48.732 |
| D2/p0.1 | 65.718 | 81.826 | 84.037 | 84.597 | 41.453 | 48.947 | 48.087 | 48.445 |
| D2/p0.3 | 65.868 | 81.164 | 83.087 | 83.104 | 43.454 | 50.018 | 53.621 | 54.556 |
| D3/p0.0 | 65.928 | 92.184 | 95.946 | 98.068 | 41.026 | 52.888 | 52.925 | 53.747 |
| D3/p0.1 | 65.803 | 91.801 | 95.746 | 97.967 | 40.903 | 52.759 | 52.773 | 53.795 |
| D3/p0.3 | 65.831 | 87.158 | 90.108 | 91.495 | 42.483 | 50.781 | 54.087 | 55.315 |
| D5/p0.0 | 60.489 | 76.829 | 82.183 | 84.583 | 29.985 | 42.680 | 44.847 | 45.500 |
| D5/p0.1 | 60.545 | 77.076 | 81.889 | 84.397 | 30.001 | 42.580 | 44.893 | 45.503 |
| D5/p0.3 | 60.444 | 75.602 | 78.924 | 79.656 | 35.570 | 44.530 | 49.333 | 50.175 |

Request tokens/s:

| Cell | Target | FP16 | Q8_0 | Q4_0 | A16 | A8 | A4 | A1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1/p0.0 | 58.548 | 74.286 | 75.462 | 76.551 | 47.120 | 53.311 | 51.433 | 52.066 |
| D1/p0.1 | 58.549 | 74.115 | 75.279 | 76.748 | 47.136 | 53.250 | 51.483 | 52.115 |
| D1/p0.3 | 58.542 | 73.513 | 74.218 | 75.464 | 46.260 | 51.625 | 50.143 | 51.165 |
| D2/p0.0 | 63.852 | 77.964 | 79.913 | 80.697 | 40.552 | 47.638 | 46.764 | 47.179 |
| D2/p0.1 | 63.704 | 77.532 | 79.503 | 80.021 | 40.310 | 47.356 | 46.564 | 46.891 |
| D2/p0.3 | 63.867 | 76.924 | 78.623 | 78.677 | 42.189 | 48.379 | 51.747 | 52.617 |
| D3/p0.0 | 63.963 | 86.836 | 90.157 | 92.062 | 39.924 | 51.082 | 51.135 | 51.900 |
| D3/p0.1 | 63.849 | 86.603 | 90.056 | 91.993 | 39.833 | 50.982 | 51.008 | 51.966 |
| D3/p0.3 | 63.900 | 82.455 | 85.059 | 86.282 | 41.334 | 49.144 | 52.228 | 53.377 |
| D5/p0.0 | 58.685 | 72.812 | 77.546 | 79.743 | 29.343 | 41.411 | 43.462 | 44.084 |
| D5/p0.1 | 58.747 | 73.026 | 77.346 | 79.568 | 29.365 | 41.335 | 43.526 | 44.091 |
| D5/p0.3 | 58.636 | 71.629 | 74.653 | 75.308 | 34.657 | 43.113 | 47.615 | 48.412 |

## Fixed grid reference and development selection

The fixed reference below is the **development-grid D5/p0.0 run**, not the
separate historical/primary matrix. Every paired value is **decode / request**.
The two fixed-anchor columns compare paths measured in that same D5/p0.0 cell.

| Path | Fixed tok/s | Fixed / FP16 | Fixed / Q4_0 |
| --- | --- | --- | --- |
| Target only | 60.489 / 58.685 | 0.787 / 0.806 | 0.715 / 0.736 |
| FP16 EAGLE | 76.829 / 72.812 | 1.000 / 1.000 | 0.908 / 0.913 |
| Q8_0 EAGLE | 82.183 / 77.546 | 1.070 / 1.065 | 0.972 / 0.972 |
| Q4_0 EAGLE | 84.583 / 79.743 | 1.101 / 1.095 | 1.000 / 1.000 |
| W1A16 | 29.985 / 29.343 | 0.390 / 0.403 | 0.355 / 0.368 |
| W1A8 | 42.680 / 41.411 | 0.556 / 0.569 | 0.505 / 0.519 |
| W1A4 | 44.847 / 43.462 | 0.584 / 0.597 | 0.530 / 0.545 |
| W1A1 | 45.500 / 44.084 | 0.592 / 0.605 | 0.538 / 0.553 |

For each path, the next table keeps its highest-observed-decode cell fixed and
compares it with the independently selected FP16/Q4_0 cells, both D3/p0.0, and
with its own fixed D5/p0.0 result. Request rates use that same decode-selected
cell; there is no separate request-rate selection. Target-only has no operative
draft policy: its selected row is merely the fastest repeated control cell.

| Path | Selected cell | Selected tok/s | / selected FP16 | / selected Q4_0 | / own fixed |
| --- | --- | --- | --- | --- | --- |
| Target only | D3/p0.0 | 65.928 / 63.963 | 0.715 / 0.737 | 0.672 / 0.695 | 1.090 / 1.090 |
| FP16 EAGLE | D3/p0.0 | 92.184 / 86.836 | 1.000 / 1.000 | 0.940 / 0.943 | 1.200 / 1.193 |
| Q8_0 EAGLE | D3/p0.0 | 95.946 / 90.157 | 1.041 / 1.038 | 0.978 / 0.979 | 1.167 / 1.163 |
| Q4_0 EAGLE | D3/p0.0 | 98.068 / 92.062 | 1.064 / 1.060 | 1.000 / 1.000 | 1.159 / 1.154 |
| W1A16 | D1/p0.0 | 48.862 / 47.120 | 0.530 / 0.543 | 0.498 / 0.512 | 1.630 / 1.606 |
| W1A8 | D1/p0.0 | 55.506 / 53.311 | 0.602 / 0.614 | 0.566 / 0.579 | 1.301 / 1.287 |
| W1A4 | D3/p0.3 | 54.087 / 52.228 | 0.587 / 0.601 | 0.552 / 0.567 | 1.206 / 1.202 |
| W1A1 | D3/p0.3 | 55.315 / 53.377 | 0.600 / 0.615 | 0.564 / 0.580 | 1.216 / 1.211 |

## Descriptive uncertainty and control variation

The report uses 2,000 deterministic crossed prompt/repetition resamples, seed
42. Prompts and repetition labels are independently drawn with replacement; the
same Cartesian-product draw is applied to both comparison sources, then tokens
and time are pooled. Five repetitions are timing repeats of 24 prompts, not
120 independent quality examples. Full per-prompt, category, and repetition
pooled summaries are retained under each cell's `grouped_summaries`.

The central 95% percentile ranges below are **descriptive, conditional on the
fixed development-selected winners**. Winners are not reselected in each draw.
These are not holdout estimates, selection-corrected or simultaneous intervals,
or confidence bounds for policy benefit. Across different cells, matching
repetition labels imposes covariance that was not experimentally established;
arbitrary pairing can change range width. Systematic timing variation is
unmodeled. Selected/selected and selected/fixed ratios cannot isolate a policy
effect, even when their resampling range excludes one.

Target-only decode rates range from 60.396 to 65.928 tokens/s. D1 and D5 controls
are near 60.4–60.5, while D2/D3 controls are near 65.7–65.9. The audited depth
cells have identical target-only commands, requests, model/binary/harness hashes,
and raw outputs; the only environment difference is D1's temporary directory.
Depth in target-only records is metadata: no draft-depth or confidence flag is
sent to its server or requests. D5 returned to the lower range after resumption,
so a simple persistent pre/post-pause explanation is unsupported. The cause is
not established; no hardware explanation or policy effect is inferred.

W1Ax ranges are shown for both anchors at fixed D5/p0.0, then for independently
selected anchors and the same path's fixed reference. Each bracket is a ratio
range, not a rate; the preceding tables give the observed pooled ratios.

| Path | Comparison | Decode range | Request range |
| --- | --- | --- | --- |
| W1A16 | fixed / fixed FP16 | [0.375, 0.404] | [0.390, 0.415] |
| W1A16 | fixed / fixed Q4_0 | [0.342, 0.367] | [0.356, 0.379] |
| W1A16 | selected / selected FP16 | [0.512, 0.547] | [0.526, 0.558] |
| W1A16 | selected / selected Q4_0 | [0.480, 0.516] | [0.496, 0.527] |
| W1A16 | selected / own fixed | [1.623, 1.636] | [1.599, 1.613] |
| W1A8 | fixed / fixed FP16 | [0.535, 0.574] | [0.550, 0.586] |
| W1A8 | fixed / fixed Q4_0 | [0.486, 0.521] | [0.502, 0.535] |
| W1A8 | selected / selected FP16 | [0.581, 0.622] | [0.595, 0.632] |
| W1A8 | selected / selected Q4_0 | [0.545, 0.586] | [0.561, 0.597] |
| W1A8 | selected / own fixed | [1.293, 1.307] | [1.280, 1.294] |
| W1A4 | fixed / fixed FP16 | [0.558, 0.608] | [0.574, 0.620] |
| W1A4 | fixed / fixed Q4_0 | [0.507, 0.552] | [0.524, 0.565] |
| W1A4 | selected / selected FP16 | [0.564, 0.608] | [0.581, 0.621] |
| W1A4 | selected / selected Q4_0 | [0.530, 0.573] | [0.548, 0.587] |
| W1A4 | selected / own fixed | [1.193, 1.218] | [1.190, 1.213] |
| W1A1 | fixed / fixed FP16 | [0.570, 0.612] | [0.585, 0.624] |
| W1A1 | fixed / fixed Q4_0 | [0.518, 0.557] | [0.535, 0.570] |
| W1A1 | selected / selected FP16 | [0.577, 0.622] | [0.593, 0.635] |
| W1A1 | selected / selected Q4_0 | [0.542, 0.586] | [0.559, 0.600] |
| W1A1 | selected / own fixed | [1.203, 1.228] | [1.198, 1.223] |

## Raw-token comparisons

Each entry below is mismatched **requests / 120**, expressed as a triple against
**FP16 / Q4_0 / target-only** in the same cell. Zero denotes observed sequence
agreement only; it does not establish general losslessness. Counts across
columns/cells overlap and must not be summed as independent failures. The JSON
retains mismatched prompt counts, raw-ID hashes, and first-divergence evidence.
Any ratio involving differing outputs is a timing observation, not a strict
lossless speedup; no numerical or other cause is assigned here.

| Cell | Target | FP16 | Q8_0 | Q4_0 | A16 | A8 | A4 | A1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D1/p0.0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| D1/p0.1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| D1/p0.3 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 10 / 10 / 10 | 15 / 15 / 15 | 0 / 0 / 0 | 0 / 0 / 0 |
| D2/p0.0 | 5 / 5 / 0 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 |
| D2/p0.1 | 5 / 5 / 0 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 | 0 / 0 / 5 |
| D2/p0.3 | 10 / 10 / 0 | 0 / 10 / 10 | 5 / 5 / 5 | 10 / 0 / 10 | 5 / 5 / 5 | 5 / 5 / 5 | 15 / 5 / 5 | 10 / 10 / 0 |
| D3/p0.0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| D3/p0.1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| D3/p0.3 | 5 / 0 / 0 | 0 / 5 / 5 | 5 / 5 / 5 | 5 / 0 / 0 | 15 / 10 / 10 | 10 / 5 / 5 | 5 / 0 / 0 | 10 / 5 / 5 |
| D5/p0.0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| D5/p0.1 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| D5/p0.3 | 0 / 5 / 0 | 0 / 5 / 0 | 0 / 5 / 0 | 5 / 0 / 5 | 5 / 10 / 5 | 5 / 10 / 5 | 0 / 5 / 0 | 5 / 10 / 5 |

Selected comparisons below also have 120 paired requests each. The W1A1 selected
cell differs on five requests (one prompt repeated five times) in each of these
comparisons. Other selected paths show zero mismatches in these comparisons.

| Selected path | vs selected FP16 | vs selected Q4_0 | vs own fixed |
| --- | --- | --- | --- |
| Target only | 0 | 0 | 0 |
| FP16 EAGLE | 0 | 0 | 0 |
| Q8_0 EAGLE | 0 | 0 | 0 |
| Q4_0 EAGLE | 0 | 0 | 0 |
| W1A16 | 0 | 0 | 0 |
| W1A8 | 0 | 0 | 0 |
| W1A4 | 0 | 0 | 0 |
| W1A1 | 5 | 5 | 5 |

At every depth, p0.0 and p0.1 have identical raw IDs and per-request speculative
counter dictionaries across all 960 pairs. The frozen EAGLE3 sampler retains
top-k=10, normalizes those candidates, then takes the maximum and stops only if
its probability is strictly below `p_min`. For finite valid logits that maximum
is at least 0.1. This is **top-10-normalized confidence, not full-vocabulary
probability**; target temperature zero does not configure this separate draft
sampler. Floor 0.3 remains distinct and exhibits output differences. Timing
variation between p0.0 and p0.1 is not evidence of a confidence-policy benefit.
Source: frozen runtime `common/speculative.cpp:493–506,820–850`,
`common/sampling.cpp:394–399`, and `src/llama-sampler.cpp:1150–1209`.

## Provenance and reproduction

The measured runner is project
`2f6ab1469fa8efe34d15e0028d1e3980e67132f6`, clean runtime
`feba1569848e74996651a42a9751f8afa5958b1d`. Analysis uses separately shipped helper
commit `c64e260`; it does not change the measured runner. The supervisor records
successful analysis (exit 0) at 2026-09-27 00:56:18 UTC.

SHA-256 identities:

| Artifact | SHA-256 |
| --- | --- |
| Final policy report | `9d742f355c41ea5d23d12db39fadca24843fcab03fa42a229ad6857634c7c499` |
| Frozen suite.json | `86e8d272d34aa96c219cd7a340a994c1aa02421a62e0cc850be688a266f46c88` |
| Development prompts | `a3b97d942a99f1bddd5bb97216c32a9920aaa50788baa5bdb92354842547e885` |
| Analysis helper | `f628b5babcfe875f3a62a1b87ab8e5d280d91043b44dae070acc5fe7403ba9db` |
| Frozen benchmark harness | `1ca8eb8aaa4b958982cb1f8488ae41e4949b81f3e78d7f3d0e19be01b35f6675` |
| Server binary | `b3b33b79ff0eb0150586763410cc4565af1aa510ea5d0ad9441502dbd49c4132` |
| Final progress snapshot | `344365061b682d9c3798cc2fa738252c5b87c8c29eb6991d5635ec1fed7be67d` |
| Analysis state snapshot | `75785dd03044209c4f48fd0baee371261b0621355d9fbba0f5796628c3814660` |

Source suite:
`/home/philip/binary-eagle-decoding/runs/w1ax-project-src/runs/w1ax-diagnostic-suite-20260926/suite.json`.
Raw source root:
`/home/philip/binary-eagle-decoding/runs/w1ax-project-src/results/`.
Each run directory is `w1ax-diagnostic-suite-20260926-development-` plus the
suffix below. The final JSON additionally preserves every config, manifest,
per-cell report, and records SHA-256 and its absolute source path.

| Cell | Run suffix | records.json SHA-256 |
| --- | --- | --- |
| D1/p0.0 | d1-pmin-0p0-a1 | `16ebc05fae6d83f36531bb42a13ff07178a894c962f3c483250eb25e9fed4394` |
| D1/p0.1 | d1-pmin-0p1-a1 | `82834ef60cf2f444870e16565b20b7cca3c057df9feef81ddc6a78141499172e` |
| D1/p0.3 | d1-pmin-0p3-a1 | `8e5b18ab0a3ef66a53ae27ce4788ec83dd068ccada193559f7e70c96465226c8` |
| D2/p0.0 | d2-pmin-0p0-a2 | `2a125f417f0d50a00265cca54670e572bf0017056ad40dfa8f4def7322f3461d` |
| D2/p0.1 | d2-pmin-0p1-a1 | `b56fd3ba367e38bdf57d5ad8df471b178001a81d60aa1851e78a2dc9feb34f5d` |
| D2/p0.3 | d2-pmin-0p3-a1 | `58a8802c57065d9b20aa8f0a7a5990a9547ac87067e4abb11cbb82abf4bdf107` |
| D3/p0.0 | d3-pmin-0p0-a1 | `f7080eded9f9cca4694cf47cb0c276ccfe89728fb28cbe4128c9cac664e8d29c` |
| D3/p0.1 | d3-pmin-0p1-a1 | `2d00eaea2fdf3fb4eee02fa0dab49c5d8780b15ea84047a98a83b0cffcec922e` |
| D3/p0.3 | d3-pmin-0p3-a1 | `042d29583645cdad233e3e884262b40e65cdce89330959835006a0154b2375f5` |
| D5/p0.0 | d5-pmin-0p0-a1 | `38dad0c9f9d8e9b592380b931c00ba5599fd7f90bb05600760057f5b0f50ac61` |
| D5/p0.1 | d5-pmin-0p1-a1 | `873e384c34b8aa1543851f26e65af14bf6df37e3c30aa95ac5f80b566722d12b` |
| D5/p0.3 | d5-pmin-0p3-a1 | `cc864b8e51cc55ce204e994b40e881c2fa51c46f7f80bce6a2e8a89b13d7d481` |

Remote final report:
`/home/philip/binary-eagle-decoding/runs/w1ax-policy-grid-analysis-20260927/report.json`.
Local verified mirror:
`/Users/pippo/github/binary-eagle-decoding/runs/w1ax-analysis-mirror/final-policy/report.json`.
Large records and raw outputs remain outside Git. To reproduce on the original
Linux artifact layout, use the separately shipped helper through tmux MCP and
a fresh supervisor. Choose an unused run ID and output directory; the example
below preserves the sealed report:

```sh
cd /home/philip/binary-eagle-decoding
python3 scripts/remote_job.py w1ax-policy-grid-reproduce-20260927 -- \
  python3 runs/w1ax-analysis-src/policy-final-c64e260/analyze_w1ax_policy_grid.py \
  /home/philip/binary-eagle-decoding/runs/w1ax-project-src/runs/w1ax-diagnostic-suite-20260926/suite.json \
  --results-root /home/philip/binary-eagle-decoding/runs/w1ax-project-src/results \
  --bootstrap-samples 2000 --seed 42 \
  --output /home/philip/binary-eagle-decoding/runs/w1ax-policy-grid-reproduce-20260927/report.json
```

The analyzer validates the successful frozen attempt per cell, complete paired
records, model/config/prompt hashes, and dispatch evidence. It retains all
raw-output mismatches. Original source paths matter: the progress file binds
the suite path. A relocated mirror requires explicit provenance handling rather
than silently rewriting this recorded source identity.

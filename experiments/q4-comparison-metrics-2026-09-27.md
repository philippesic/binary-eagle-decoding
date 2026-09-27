# Q4_0 comparison metrics — 2026-09-27

Q4_0 EAGLE is the primary comparison target, per the user’s explicit decision. FP16 remains diagnostic context. This does not change the main target/verifier precision. No new GPU experiment was run for this summary.

## Original W1Ax scales: RTX 2080 Ti

Fixed development D=5, p_min=0, 24 prompts × five repetitions; 120 requests and 15,360 output tokens per path. These are the completed policy-grid measurements, not the earlier instrumented historical diagnostics.

| Draft | Accepted/proposed | Accepted/round | Decode ms/token | Decode tok/s | Full request tok/s | Mean full request s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_0 | 20.991% (7,730/36,825) | 1.0362 (7,730/7,460) | 11.82 | 84.583 | 79.743 | 1.605 |
| W1A16 | 2.385% (1,590/66,655) | 0.1174 (1,590/13,540) | 33.35 | 29.985 | 29.343 | 4.362 |
| W1A8 | 2.260% (1,515/67,025) | 0.1113 (1,515/13,615) | 23.43 | 42.680 | 41.411 | 3.091 |
| W1A4 | 0.949% (675/71,135) | 0.0467 (675/14,450) | 22.30 | 44.847 | 43.462 | 2.945 |
| W1A1 | 1.112% (785/70,570) | 0.0547 (785/14,340) | 21.98 | 45.500 | 44.084 | 2.904 |

Acceptance is accepted/proposed draft tokens; accepted/round is a distinct metric. Decode throughput follows the project convention: completion tokens divided by summed server decode seconds. Decode ms/token is its reciprocal. Request throughput and request latency use measured client wall time including prefill. A matched per-round draft latency was not captured in this cell.

Source: [complete policy grid](w1ax-policy-grid-results.md), fixed cell `d5-pmin-0.0`, run `w1ax-diagnostic-suite-20260926-development-d5-pmin-0p0-a1`. The independently checked raw `records.json` SHA256 is `38dad0c9f9d8e9b592380b931c00ba5599fd7f90bb05600760057f5b0f50ac61`.

## New fitted scales: RTX 5080

The [scale-fitting report](binary-scale-fitting-5080.md#recorded-latency-and-throughput-post-run-extraction) now records acceptance and timing observations for Q4_0 and all four W1A16 scale candidates. Only W1A16 was refitted; no fitted A8/A4/A1 measurement exists. Do not compare raw speeds across these two GPUs.

The new screen was a single pass per prompt with diagnostic logging and CUDA graphs disabled. Its recorded timings are useful observations, not a repeated performance benchmark. Fitted-group D had the highest binary acceptance (8.699% of all proposals), while fitted-row C had the highest observed binary decode throughput (61.455 tok/s). D reached 51.303 tok/s; the same-run Q4_0 reference reached 123.070 tok/s.

The distinction matters: improved draft acceptance does not guarantee improved total throughput when the draft implementation costs more per round. No timing optimization or follow-up study has been started.

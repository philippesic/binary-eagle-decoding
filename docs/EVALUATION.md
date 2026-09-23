# Evaluation protocol

## Freeze the experiment

Use the same RTX 2080 Ti and target model for all comparisons. Record project
and llama.cpp commits (plus any dirty diffs), model snapshot revisions and hashes,
conversion commands, CUDA/driver/compiler versions, GPU memory/clock/power state,
build flags, and actual backend dispatch. Save an environment manifest alongside
each raw run. `configs/baseline.toml` proposes initial settings; unresolved values
must be filled before any result is treated as reproducible.

Freeze prompt IDs/content hashes, chat template, thinking mode, tokenizer,
context lengths, batch/microbatch sizes, KV precision, placement, concurrency,
sampling settings, seed, stopping conditions, and speculative parameters. Keep
calibration/QAT data separate from held-out evaluation. Start with a small fixed
prompt suite, then include representative code, reasoning, and prose workloads.

Use warmups and at least five measured repetitions initially. Record actual
output lengths and EOS behavior. Alternate comparison order and investigate
variance before claiming gains. Report aggregate token/time ratios and the
distribution across prompts/runs, rather than averaging speedup ratios alone.

## Baselines and precision labels

| Variant | Requirement |
| --- | --- |
| No speculation | Same target and runtime settings; mandatory reference. |
| FP16 EAGLE | Existing draft checkpoint converted to FP16; mandatory reference. |
| W8A8 / W4A4 EAGLE | Desired comparisons; identify whether kernels actually execute at that operand precision. |
| W1A1 simulation | Accuracy/acceptance experiment; floating-point simulation is not binary acceleration. |
| W1A1 native | Packed binary operands and XOR/POPCOUNT computation; identify normal-precision exceptions. |

For every variant, list precision and execution kernel for each linear group,
including the output head. Do not relabel a weight-only quantized format as a
fully quantized activation/weight implementation. Mark unavailable comparisons
explicitly rather than silently replacing them.

## Definitions and timing

- Acceptance rate: accepted draft tokens / proposed draft tokens. Preserve both
  counts. Do not include the extra target token in the accepted-draft numerator.
- Mean accepted draft tokens per round: total accepted draft tokens / speculative
  rounds. Also report total emitted tokens per round, including target tokens.
- Decode throughput: generated output tokens / decode wall time; define the
  precise prefill/decode boundary and token counting once and use it everywhere.
- Request throughput: generated output tokens / full request wall time, including
  prefill. Report time to first token separately when available.
- Speedup over normal EAGLE: W1A1 decode tokens/sec / FP16 EAGLE decode tokens/sec.
- Speedup over no speculation: W1A1 decode tokens/sec / target-only decode tokens/sec.

Record per-round draft latency, verification latency, other overhead, and
accepted/proposed counts. Decompose draft cost into packing (including sign and
scale computation), binary matrix operations, output head, and remaining graph
work. Keep enough raw data to recompute metrics.

Use CUDA events or a profiler for GPU components, with appropriate stream
synchronization and warmups. Measure host wall time for end-to-end performance.
Profile separately when instrumentation changes timing; overlapping GPU spans
must not be summed as though they were serial. Report both kernel-only and
packing-inclusive timings on actual EAGLE matrix shapes. Include memory use,
launch overhead, and graph capture settings.

## Correctness and decision criteria

Before performance claims, validate packed dots against a numerical reference,
including non-word-aligned reductions, sign at zero, scales, and layouts. Check
the simulated and native binary paths agree within a documented tolerance.
Check greedy outputs against the unchanged target and investigate divergences;
for stochastic sampling, validate the verifier's required proposal semantics
and distributional behavior instead of requiring identical random sequences.

Evaluate acceptance on held-out prompts before kernel investment. Estimate an
optimistic upper bound using measured emitted tokens per round and verification
cost with draft time reduced toward zero. If that bound cannot beat FP16 EAGLE,
prioritize diagnosing acceptance and stop broad kernel development.

Publish a positive result only when repeated end-to-end measurements support it
and variability is disclosed. Otherwise quantify the bottleneck and which
assumptions failed. A correct, well-instrumented negative result meets the goal.

## Run artifacts

Use `results/<run-id>/` for raw logs, a copy of the resolved config, environment
manifest, exact commands, prompt manifest, counts, and timing data. These files
are ignored by Git. Put compact reports and links to preserved raw artifacts in
`experiments/`; record storage locations and hashes when moving artifacts.
Never substitute missing measurements with zero or fabricated estimates.

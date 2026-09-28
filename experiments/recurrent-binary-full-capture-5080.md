# Frozen 96-prompt recurrent D capture on RTX 5080

**2026-09-28 UTC.** The user-authorized RTX 5080 (SM120a) ran one
instrumented, non-training candidate-D own-history capture on the frozen
96-prompt training split, capped at 128 generated tokens per request. The
target remained pinned FP16 Qwen3-4B. Q4_0 EAGLE remains the primary future
acceptance, latency and throughput comparison; this capture makes no quality
or speed claim. No development or reserved-final prompt was used.

The fork was `0abe6e586`; the parent source checkout was `dbdca38`, with
explicit 65,536 raw target-logit and 131,072 target-feature row limits.
The pinned target, D draft, 96-prompt file and canonical D map had SHA256 or
raw-map digests
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`,
`10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`,
`80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74`
and `03d2f0e3420175955c14a43a1c1dda360cde26c8a03318ffb1a013bb06d0fe0a`.
Both models were fully offloaded to CUDA0 with F16 K/V caches and A16 draft
activations. The supervised run took 07:42:44–07:50:00 UTC and exited zero.
Its process group stopped; a subsequent check found no project process or
GPU compute app, 0% utilization and 1,372 MiB whole-device use.

| Captured or audited item | Count |
| --- | ---: |
| Completed requests / unique train task owners | 96 / 96 |
| Native EAGLE rounds | 8,295 |
| Head states and raw target-verifier logit rows | 40,815 each |
| Raw target-feature rows | 52,297 |
| Selected accepted-prefix feature rows | 15,042 |
| Valid and supported verifier labels | 40,815 / 39,705 |
| Unsupported valid labels | 1,110 |
| Labels reached by the live verifier | 12,061 |

All 40,815 verifier logits were captured within the limit; their raw F32
file is 24,805,071,360 bytes. The raw feature file is 1,606,563,840 bytes.
The native row and feature preparers checked frozen split, request ownership,
round/proposal ancestry, verifier provenance, vocabulary mapping and file
hashes. Internal accepted-prefix continuity passed all 8,295 rounds. The
preparation bundle passed after its audit was changed to stream raw logits
in bounded batches. Its mapped target probability mass on these captured
rows averaged **0.9719726884**. This is a diagnostic of draft-vocabulary
coverage on candidate-D histories, not an acceptance estimate.

The first bundle attempt exited 137 while the former audit materialized
all 40,815 × 151,936 logit values as Python float lists. The raw capture,
manifests and preparer outputs remained intact. Commit `51317fc` replaced
that path with bounded F64 batches; 11 focused capture-audit tests passed,
including chunk-boundary and nonfinite-row checks. A second supervised
CPU-only bundle audit exited zero. Its manifest is marked
`training_eligible: false` and `readiness: preparation_only`.

The raw capture is preserved outside Git under
`checkouts/recurrent-gpu-capture-20260928/runs/recurrent-full96-20260928/`
on the registered RTX 5080 host. SHA256s: capture manifest
`2b2f49861c010214d2e424ad49053c829acdd6c6dafccbad721406390ed888fd`,
cell manifest
`f2f65984d6fc6857bc1857d9f484a0a2dbe371ea4c47a477a77687bdc621607b`,
continuity `d420cfac091e3911cd90f559329bdead4f07a3789d892baa8dcd31107fa3dbd1`,
prepared rows `d7dec4e9857494b406a5c31d7d183720440d79a4a30601e0430cf8e5f2d15a15`,
prepared features `ae68af92dad00337407ac213a43d64ba579d9b889b65ed0a65cb3f8d42f2cb83`,
bundle manifest `8919cd05614f952f945bf6e1bf97a6bb8c30a13f47f5d02f3c7b169f93e8e31c`
and bundle audit `9a43b4a708664bb3f42287411595365b3b955fc4f83a67fab20010331909caff`.

At the first bundle checkpoint, per-request raw response-emission audit
had not yet completed.
Initial target sampling, terminal sampler parity, numerical target-feature
parity, whole-drafter state/attention parity, trained GGUF quality and native
Q4_0 timing remain open. No all-body optimizer budget was approved or used.

## Complete raw-response audit

The per-request auditor subsequently joined **96/96 saved raw responses** to
their own native round emissions: 12,251 output tokens, 8,295 rounds and
3,786 accepted drafts on these training histories. Ninety-five requests
ended at the output cap. One reasoning request stopped after accepted EOS
token `151645`; its canonical verifier batch also recorded one later token
that the live trace and response correctly did not emit. The strict auditor
admits only that final-round EOS prefix clipping and counts one instance.
All other 8,294 round traces matched their canonical verifier emissions
directly. The earlier all-request audit failed on this case and was
preserved; commit `bca41b1` added the narrow stop rule with a focused
positive/negative fixture. The successful supervised CPU audit exited zero;
its report SHA256 is
`8ebba18b41a58a886a71914298c139433832083bdac10a83001072f23afd4051`.

This closes request ownership, emission and terminal-stop consistency for the
captured 96 requests. It does not independently verify the initial target
sample, the terminal sampler arithmetic, or target-feature numerical values.
The bundle remains preparation-only and training-ineligible.

# CPU audit of archived W1 latency and capture evidence

**Purpose:** establish what existing logs already say about latency scopes and
the 96-prompt capture, and set a CPU-only verification plan for Phase 1A. This
is an offline report, not a new benchmark or a change to archived metadata.

## Audit record

- Audit host: macOS 27.0, arm64 Apple M3 Max; Python 3.11.3. No model was
  loaded, no inference or accelerator API was called, and no GPU status was
  queried.
- Source checkout: `/Users/pippo/github/binary-eagle-decoding`, commit
  `8bd1b91ad8475d899b1edc91da6a0ae945af14fc`.
- Report checkout: managed worktree branch `verify-archived-evidence`, based
  on the same commit.
- Archived artifacts were read in place under the ignored
  `results/binary-rescue-head-5080/` directory. No raw data or reports were
  copied into Git. No temporary run directory or output files were created;
  cleanup was not applicable.
- Commands used included `uname -a`, `python3 --version`, `sw_vers`, and
  `sysctl -n machdep.cpu.brand_string`; the `sed -n` reads listed under
  [sources](#archived-timing-evidence-and-honest-scopes); and this read-only
  metadata query from the source checkout:

  ```sh
  python3 - <<'PY'
  import json
  from pathlib import Path
  obj = json.loads(Path('results/binary-rescue-head-5080/round-analysis.json').read_text())
  print({'schema': obj['schema'], 'workload': obj['workload'],
         'variant_count': len(obj['variants']),
         'analysis_sources_sha256': obj['analysis_sources_sha256'],
         'round_calibration_status': obj['round_denominator_calibration']['status'],
         'validated_requests': obj['round_denominator_calibration']['validated_requests'],
         'limitations': obj['limitations']})
  PY
  sha256sum results/binary-rescue-head-5080/round-analysis.json \
    results/binary-rescue-head-5080/timed-primary-manifest.json \
    experiments/binary-rescue-head-5080.md \
    experiments/w1ax-activation-precision-results.md
  ```

  The actual metadata query printed these fields plus the path-specific source
  hashes; no per-request prompt fields were read. Hashes are recorded below.
  `git diff --check` was also run in the report worktree.
- No tests or benchmarks were run for this accounting task. Existing results
  below are historical evidence, not a new validation result.

## Archived timing evidence and honest scopes

| Evidence | Archived scope and result | What it does not measure |
| --- | --- | --- |
| RTX 2080 Ti W1Ax historical trace | Reported mean **CPU-wall** round spans: Q4_0 24.542 ms (4.435 `draft()`, 18.152 target decode plus synchronization, 1.047 `process()`); W1A1 23.441 ms (3.815, 18.061, 1.034 ms respectively). FP16 was 26.827 ms (6.730, 18.138, 1.076 ms). Trace summary covers 3 prompts × 8 paths × 5 measured repetitions (120 measured requests) with warmups reported separately. | CPU wall stages can include waiting for GPU work. `target decode + sync` is not a CUDA kernel duration or a pure verifier span. Begin/seed and request prefill are outside traced rounds. Other proposal, accept, checkpoint, repair and host work remains; spans from different levels are not additive. These values are not the repeated, uninstrumented request timings. |
| RTX 2080 Ti matched W1Ax timing cell | 24 development prompts × 5 repetitions = 120 requests and 15,360 output tokens/path. Q4_0 measured 84.583 decode tok/s and 79.743 full-request tok/s; W1A16/A8/A4/A1 measured 29.985/42.680/44.847/45.500 decode tok/s. | No matched per-round `draft()` latency in this cell. Decode uses server completion tokens / summed decode time; full-request rate includes prefill/client wall. Do not infer a component cost from these aggregate rates. |
| RTX 5080 fitted-scale screen | One pass over 24 development prompts, fixed variant order, no timing repetitions, CUDA graphs disabled, diagnostic logging enabled. Mean instrumented CPU-wall `draft()` was Q4_0 3.49 ms/round, fitted-row C 7.79, fitted-group D 14.49. Whole-system server decode was Q4_0 123.07, C 61.45, D 51.30 tok/s. | A single diagnostic pass, not a performance benchmark. No client request wall in that runner; not comparable to RTX 2080 Ti rates. `draft()` spans are not device-only times. |
| RTX 5080 rescue timing | Clean uninstrumented timing: 12 paths × 24 development prompts × 5 measured repetitions = 1,440 requests, plus 120 warmups. Q4_0 reached 135.09 client tok/s / 142.13 server decode tok/s; target-only 97.74 / 100.00. This is the cleanest archived request-level primary comparison. | No per-projection or per-decoder-step attribution. Keep client rate (including prefill) separate from server decode rate. It is SM120 evidence and says nothing about SM75 speed. |
| RTX 5080 rescue stage analysis | Separate instrumented run: 12 paths × 24 measured requests plus 2 warmups/path = 312 requests. The offline report validates complete-round denominator calibration on **264 requests across 11 speculative variants**. Round costs retain disjoint CPU-wall categories, shared/overlapping spans, and unattributed remainder. | This is a diagnostic trace, not the clean request timing above. Inclusive stage spans and nested scalar durations must not be summed into the disjoint partition. Fixed-trajectory span-removal calculations are counterfactuals. |
| RTX 5080 Nsight lifetime profile | Historical Nsight server-lifetime sums: Q4_0 3.785 s and D 11.044 s of GPU kernel time; D's custom A16 subset summed 5.916 s. | Different round trajectories, startup/warmups and whole server lifetimes; not request, round, or paired latency. It cannot be subtracted from request wall or combined with CPU spans. |

The W1Ax report documents 0 invalid, nested, overlapping, out-of-bounds or
clamped-residual rows in its archived span audit. It also explicitly warns
that `accept_ms` is only the acceptance-hook span, not isolated target
verification. The rescue report's newer accounting retains unassigned round
time and explicitly keeps overlapping/inclusive spans out of its disjoint
partition.

Archived hashes: ignored `round-analysis.json` SHA256
`0f73d410289dc577a543771cc8649a1e4b1e198bbdc51b678b40dcd0d2156e83`, and
`timed-primary-manifest.json` SHA256
`98914b5b569ffc9ca7d5f921181b16d6af7cc3f93bbc0f5cff097adfc713ebd3`. Tracked
reports: [W1Ax precision results](w1ax-activation-precision-results.md),
[Q4_0 comparison metrics](q4-comparison-metrics-2026-09-27.md),
[fitted-scale results](binary-scale-fitting-5080.md), and
[binary rescue results](binary-rescue-head-5080.md). The rescue report's
recorded SHA256 is `1404679c3444c7d018306cb313ea54a4c60fb0fbe2907fe211b47ebb1ec94ed6`;
the W1Ax results SHA256 is
`c80c2571e0015d19f0cdf1869ffb1743f3e4aa24b2e0346c174210b69c8bee96`.

### Missing attribution for the upcoming matched A/B

Existing evidence does not provide packing/scale, dot-product, and output
times for identical inputs per projection; component time by recurrent decoder
step; a request/round to CUDA-event/kernel join; or an isolated target
verification duration. `process()` and `draft()` are CPU-wall spans, so they
may include device waits. The same-build Q4_0 and W1Ax fixed-context comparison
must keep client full-request wall, server prefill/decode, draft-call wall,
`process()` wall, target decode/synchronization, accepted/proposed/emitted
counts, and any profiler/kernel metrics as distinct fields. Preserve an
unassigned remainder and do not subtract operator medians from a separately
measured whole-request rate.

The runtime workstream is adding the explicit `eagle_process_stage_v1` and
`eagle_draft_stage_v1` CPU-wall tags plus whole-call counters/analyzer. The
archived rescue trace provides a schema precedent for nonoverlapping
accounting, but it does not fill the per-projection, per-step or device-event
gaps above. This report does not claim a source change or a speedup.

## CPU-only validation plan

After the owners expose their CLI contracts, verify only with explicit CPU
devices and accelerator paths disabled:

1. Run `python3 scripts/account_eagle_latency.py --analysis
   <round-analysis.json> --output <report.json>` on an archived trace
   copy/read-only input. Check stage totals, round/request counts,
   missing-stage reporting, and that the disjoint partition plus unassigned
   time reconciles to the stated round scope. Keep inclusive spans and
   CUDA-kernel records outside that sum.
2. Run the joint trainer's tiny CPU fixture with explicit `device=cpu` for
   A16/A8/A4/A1. Check hard-quantized forward dispatch, gradients through
   recurrent earlier states/cache, finite loss/gradients, and trainable body
   plus head ownership. These are contract/gradient checks, not convergence or
   training-throughput evidence.
3. Run data-manifest tools on synthetic or authorized local fixtures with
   network and model loading disabled. Check deterministic deduplication,
   topic/conversation split isolation, source/provenance retention, and
   token/mask/EOS/truncation accounting. Keep raw datasets and reports under
   ignored run directories.
4. Never use CPU checks to claim SM75 kernel correctness/performance or CUDA
   convergence. Actual target capture, native acceptance and timing remain
   behind restored GPU access.

This audit did not execute those checks because the corresponding implementation
work is still active with its owners.

## Capture readiness in place of the old eligibility gate

The inherited 96-prompt bundle remains marked `training_eligible: false` and
`readiness: preparation_only`; this report does not edit its manifest. Its
structural evidence is useful and specific: 96/96 response audits passed;
40,815 head/verifier-logit rows, 52,297 raw feature rows and 15,042 selected
accepted-prefix feature rows were joined; 39,705/40,815 valid labels map into
the draft vocabulary, leaving 1,110 unsupported labels (2.72%); mean mapped
target probability mass was 0.9719726884 on captured candidate-D histories.
Native prefix continuity, vocabulary/ancestry audits and the all-request
emission audit passed. The raw F32 target-logit file alone is 24,805,071,360
bytes. None of these counts establishes broad source coverage or current
student-prefix labels after its proposals change.

Use this practical replacement classification:

- **Structurally audited for reuse and smoke/regression fixtures:** yes, on its
  captured native D prefixes, with unsupported-label rates and response joins
  visible.
- **Substantive training corpus for joint QAT:** not yet. The 96 prompts are a
  regression capture, while Phase 1A requires larger, diverse, deduplicated,
  provenance- and split-audited data; captures still need to use the frozen
  native target on the exact prefixes being trained. The old full-vocabulary
  logits are too large to scale blindly; use the data owner's explicit compact
  teacher or streaming contract.
- **Native acceptance/throughput evidence:** none for a newly trained joint
  checkpoint. The inherited capture was not training and cannot support a
  Q4_0 promotion claim.

Therefore, keep the old flag unchanged and annotate readiness in new experiment
reports/manifests only: `audited_smoke_fixture`, with `substantive_qat_corpus`
and `native_deployment_evidence` still pending. This preserves ancestry and
metadata while replacing the vague gate with concrete scope.

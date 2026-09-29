# First bounded W1Ax training-data capture on RTX 5080

On 2026-09-29 UTC, the frozen `plan-003/shard-0000` training shard was captured
with the native F16 target/verifier and candidate D group-128/A16 EAGLE draft
on NVIDIA GeForce RTX 5080 (SM120). This is a data-preparation and provenance
check, not joint QAT or a held-out acceptance result. Q4_0 EAGLE remains the
primary performance baseline. No new or legacy final prompt text was opened.

## Frozen inputs and supervised run

- Ordered shard: 31 train prompts; shard manifest SHA256
  `17b8c65c47b335449e7573e42ec644f0682a7b5dfb97bb1874f1bc8eb05cd8b6`;
  prompt JSONL SHA256
  `968ffbb21b23f934912862ef6f7add7d03bf0cfbbb3910492f2f0f50075fc18a`.
  Its metadata has 12 prose, 10 reasoning and nine code prompts. This is a
  small calibration shard, not the full 2,000-prompt training tier.
- Target GGUF SHA256
  `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`;
  candidate D GGUF SHA256
  `10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`;
  absolute I64 d2t map SHA256
  `6dcd8cadd270000775cb04278c5f994647cc1ab2e2e762982c5ff7f31efc78f4`.
- Remote supervised capture `runs/w1-shard0000-capture-20260929/` exited zero
  at 07:59:33 UTC. Its 31 requests generated 12,610 raw verifier logit rows,
  below the 16,384 hard cap, and 7,663,651,840 raw logit bytes. Raw logit
  SHA256 is
  `76fc50d438e79005d184de8a3d4accd34b50ad76e02dd99a6cb7c47514967a24`.
  The cell manifest SHA256 is
  `e8bdbd3eb8f0e754a58defdf138ce6197282c4634120a5b4565113ad88875090`.
  The supervisor and server process group stopped; afterward the RTX 5080
  returned to 0% utilization and about 2,900 MiB whole-device baseline use.

## Preparation and audit

The independent shard `observe` check verified the raw hash, bytes, row cap
and request ownership. Native row preparation yielded 2,562 rounds and
12,610 rows; feature preparation selected 7,796 accepted-prefix target-feature
rows. Bundle `runs/w1-shard0000-bundle-20260929/bundle/` has manifest SHA256
`3ee7a8f4526f1dbcca1b6e0ea0756e42eff81333d7a88137afebe7213d3c1947`.
Its audit counted 12,610 valid rows, 12,250 draft-supported rows, 360
unsupported rows and 3,776 verifier-reached rows. Sampled mapped target
probability mass averaged 0.97013. The independently rerun audit was
byte-identical to the builder audit (SHA256
`832325813eefea67dc97dc0d251b1e37b3a7b4a7349a4e26eb3aa9663198508b`).

The teacher row adapter emitted all 12,610 rows. Four top-64 compact teacher
shards, with 4,096/4,096/4,096/322 rows, were built; their manifest SHA256 is
`2d4b39867e2199c39116d98491baf5b46c706c17709b13d0efdebf910255fc52`.
The compact verifier completed with zero errors and checked source hashes,
row ranges, prefix hashes, shapes and probability mass. The source and bundle
copies of the full raw F32 logits remain retained; the compact shards do not
yet replace either raw copy under the v1 re-audit contract.

The audited rows span all three frozen train domains. A CPU-only count from
the hashed bundle's prompt metadata and `rows.jsonl` is preserved at
`runs/w1-shard0000-domain-coverage-20260929/stdout.log`, SHA256
`8c532fb32466609d902984bb992863b090f6bde6056990affab95609bd245151`:

| Domain | Prompts | Rows | Supported labels | Verifier-reached rows | Distinct target labels |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 12 | 4,815 | 4,659 | 1,382 | 1,235 |
| Reasoning | 10 | 4,020 | 3,926 | 1,260 | 510 |
| Code | 9 | 3,775 | 3,665 | 1,134 | 659 |

These sums reproduce the audit's 12,610 total, 12,250 supported and 3,776
verifier-reached rows. They show broad-domain coverage in this small shard,
not dataset sufficiency for body QAT or a learned-quality claim.

An internal native round-continuity audit verified 2,562 rounds, 4,065 prefill
inputs, 15,172 speculative inputs and 11,390 rejected suffix inputs. Its
report SHA256 is
`d31851548a0f14563fb8cb0f757860790f76e92692f3c85b2972004d946a60ee`.
It does not by itself prove initial sampling, terminal emission, request
completeness or numeric feature parity. A shard-aware all-request emission
audit (`4db0440`) then checked every request/response file hash and joined all
31 native responses to their raw ID emissions: 3,836 response tokens,
2,562 rounds and 1,220 accepted drafts, with 29 length stops and two stop
tokens. Its report SHA256 is
`9f6c120d21d06ed8c1fdbcb44fc2e8c4af9a4259cf94cbfef343cbed539bdc1f`.
The audit still marks initial sampler parity and terminal sampler parity
unverified; it proves the observed emissions and request coverage.

## Readiness

The bundle declares `readiness: preparation_only` and
`training_eligible: false`. It retains four unverified gates: native model
execution identity, target-feature numeric parity, full drafter
mask/position/KV parity, and cross-round acceptance ancestry. The internal
continuity and all-request audits narrow the last gate but do not silently
rewrite the manifest. The concrete provider refuses this bundle before loading models.
Thus no captured-data joint training, quality recovery or throughput conclusion
follows from this capture. Separately, the model-independent row-scale A4
synthetic CUDA fixture ran 100 steps on RTX 5080 under
`runs/w1-joint-cuda-fixture-a4-20260929/` and exited zero in 19.5 seconds.
`training_run.json` SHA256 is
`4515af9a865a3dab6607b310de68d735ffeb855e04acf0e3d455b1154c2dcda8`.
All numeric metrics were finite and every step recorded 18 gradient tensors;
loss was 1.40130 initially and 1.40134 finally, with zero sign flips. This
only validates the tiny device/optimizer path, not a real model's training
rate, memory, convergence or deployment quality. The job's process group
stopped and the device returned to 0%, about 2,900 MiB baseline.

The captured cell pins native server binary SHA256
`a57f9e784eb2528d2de954d12ebcb9ff57c1ee24cd8b11e3df55e526125e52f0`,
which still matches the built server. It records CUDA device 0, all GPU
layers, F16 K/V cache, graph-disabled capture, A16 candidate D and the
target/D GGUF hashes above. The archived pinned Hugging Face target/draft
snapshot manifest SHA256 is
`2db1c860f059bd8702ca6bb523e64f0c7d064c7a95f59919a001fc88168a91e3`;
every listed file was rehashed and both snapshots passed the manifest check.
The prepared provider manifest SHA256 is
`4ce8a76f41f1aeecd7f953951dbe3782d9ae2c27b6794234c7c28d39dabc7d47`.
It binds the 31 prompts, capture, d2t, model snapshots and compact teacher,
and correctly preserves `training_eligible:false`. A one-step CPU CLI probe
exited with `ValueError: training manifest is not eligible` before loading
models; no training artifact was produced.

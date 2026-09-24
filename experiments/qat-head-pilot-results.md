# Bounded head-only W1A1 QAT pilot

**Device:** RTX 5080, BF16 PyTorch AngelSlim drafter/target. **Date:**
2026-09-24. **State:** bounded pilot complete; the single fixed held-out
export/evaluation gate failed to improve acceptance. This is a quality experiment, not native
W1A1 speed evidence.

The training/capture code stayed at parent commit
`855fea51f00b05678faa8d971093b7a78c797e25`. The pinned Qwen3-4B target,
AngelSlim EAGLE-3 draft, and original BF16 head are unchanged. The 96 training
and 24 validation prompts were deterministically generated with 32/8 per
prose, code, and reasoning category and content-disjoint from the 12 held-out
prompts. Their SHA256 values are
`4e44fd2f5806cd7edd56de56a87a211efcc1293b418a5e99cdfc2c1a3cce7a8a`
and `2dcec4dd9c954506415b63fe13cd165f6395eab942da61562d82a284d89cc531`.
The held-out prompt hash
`0d6a698d6816592c6ff435fed2fea4cdafe9f5248393d2a5ac091e1551919476`
was used only to check disjointness, not for training or checkpoint selection.

The supervised capture ran ordinary and untrained head-W1A1 trajectories
without gradient updates, then saved BF16 pre-head vectors and a frozen BF16
teacher head. The train tensor has 32,768 × 2,560 rows, split 16,391 ordinary
and 16,377 W1A1 trajectory rows; validation has 9,216 rows, exactly 4,608
per trajectory. Manifest path:
`results/qat-head-capture-20260924/capture-manifest.json`, SHA256
`599064290f1c4662fc9f8e382b39cb29727838996d35643934b7dfa95ef9008e`.
The train/validation/teacher tensor SHA256 values are respectively
`62565c9b675e19bdddc3fb0a963128c98df7539640f6ba6c5b4361a7131ef2e9`,
`84e34a8e2456a5f0a6ef9119a37994ee8330ff5c8d17aadb8814749205be106f`,
and `8da3582069e5bb3b9339aed07f31d4647bae6f448d17b7223a4dd26c63bc2f61`.

The CUDA dry run checked hashes, prompt isolation, and bitwise training-head
forward equivalence before optimization. The bounded training then ran all
500 steps in 39.31 seconds, optimizing only the head's FP32 latent weights
against the frozen BF16 teacher with temperature-1 full-vocabulary KL,
AdamW (`lr=1e-4`, zero weight decay), batch 64, 20-step warmup, and norm-1
gradient clipping. The custom weight STE passes a scale-normalized surrogate
inside its documented clipping window; forward remains the BF16 fake-binary
operation. Validation was separate from training and checked every 50 steps.
The best checkpoint was at step 500:

| Validation metric | Step 0 | Best step 500 |
| --- | ---: | ---: |
| KL to original BF16 head | 2.50634 | 1.01696 |
| Teacher top-1 agreement | 0.47667 | 0.54167 |
| Teacher top-10 overlap | 0.52530 | 0.60256 |

The trainer reported no nonfinite gradients, OOM, or export-forward mismatch.
Its summary is `results/qat-head-training-20260924/training-summary.json`,
SHA256 `aa96e1c40adb5393d9d8cd0bf1019a9fec0d09d4c6cbe3b2ca918ea49c4bf233`;
the exact capture, dry-run, training commands, 21 artifact hashes and GPU
environment are in `results/qat-head-training-20260924/run-manifest.json`,
SHA256 `d818b3000f12f755cc0bc3b40da16bda0bbc2a38ecedffc15123e30deb1400ba`.
The BF16 export-ready best head is
`results/qat-head-training-20260924/best-head.pt`; its full hash is recorded
in the summary/run manifest. All three supervised jobs terminated and the
GPU returned to its Windows baseline of 3,050 MiB used and 0% utilization.

## Single held-out gate

The exporter placed only `best-head.pt`'s BF16 weight into a fresh copy of the
pinned full drafter checkpoint. It verified all other tensors and the key set
unchanged, including `d2t`/`t2d` and the missing `embed_tokens.weight`. The
five source/derived draft files match by path; only `model.safetensors` changed.
Derived checkpoint SHA256 is
`450e1d3e27244a46faa2dc28b937b507d490c7a1c66d1ab2763c4c6fc677f5a1`,
derived model manifest SHA256 is
`6e980e788855d850a16e60e81cab6bac9ea5e4d2292b73b1225e54adcc7c2f16`,
and tensor/file audit SHA256 is
`a693025bbe9984498e63846d377e2b115d4bba6729684e5cbd6fd5b3eb8257c5`.

The exporter/evaluator then ran **once** on the frozen 12 held-out prompts,
with the same pinned BF16 target, AngelSlim verifier, greedy decoding, prompt
manifest, and acceptance settings as the original 5080 sweep. The two derived
variants were the trained head in ordinary full-precision execution and the
trained head with W1A1 fake-binary execution:

| Draft variant | Accepted | Proposed | Rounds | Accepted/round | Exact target-greedy streams |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original ordinary BF16 (earlier fixed sweep) | 1,061 | 27,022 | 458 | 2.31659 | 5/12 |
| Original untrained W1A1 head (earlier fixed sweep) | 949 | 33,394 | 566 | 1.67668 | 4/12 |
| Trained head, ordinary BF16 | 981 | 31,919 | 541 | 1.81331 | 4/12 |
| Trained head, W1A1 BF16 | 936 | 35,282 | 598 | 1.56522 | 6/12 |

The derived ordinary and trained-W1A1 paths respectively matched 1,117 and
1,014 of 1,506 checked target positions through each stream's first
divergence. The earliest mismatches recur at generated position 3 on
`prose-01` and `prose-03`; the run manifest records all mismatch indices.
Raw artifacts remain under `results/qat-head-heldout-20260924/` on the 5080
host: `run-manifest.json` SHA256
`8e99e0ed25b55150824d93c535cc0397d942d61353602719ed8ec7644071de9c`,
`acceptance.jsonl` SHA256
`78556cbe20d9b0b6a65a226b122bc211a639006bdff1d305524e41bedc522020`,
`summary.json` SHA256
`c2502bcc5c0df710df5584a62bb726363886e17d1e15fa2de9ea8ad0f0c90455`,
and `greedy-mismatches.jsonl` SHA256
`c360f29276dd3bbecf94cfb9497cc511d754f6ad1baa3540cdfa49a5310989e9`.
The evaluator supervisor exited zero and Windows GPU samples returned to its
3,050 MiB/0% baseline with no project process; SSH/tmux sessions were closed.

The held-out result rejects this bounded head-reconstruction recipe: lower
validation KL did not recover W1A1 draft acceptance and the exported head also
hurt ordinary execution. The accepted/round counts are verifier-relative;
generated-stream mismatches remain, so they do not establish strict
target-equivalent decoding. No further tuning against this held-out set is
planned. Native packed execution with the original head still needs direct
correctness and same-device throughput measurement.

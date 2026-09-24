# Bounded head-only W1A1 QAT pilot

**Device:** RTX 5080, BF16 PyTorch AngelSlim drafter/target. **Date:**
2026-09-24. **State:** capture/training complete; the single fixed held-out
export/evaluation gate is pending. This is a quality experiment, not native
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

The validation improvement does not establish speculative acceptance.
The next gate exports only the best BF16 head into a fresh full drafter
checkpoint, audits unchanged tensors/maps, then evaluates that checkpoint
once on the fixed held-out prompts against the original verifier-relative
ordinary/untrained-head references. If acceptance fails to recover, stop this
head-reconstruction recipe rather than tuning against held-out data.

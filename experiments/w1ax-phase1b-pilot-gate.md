# Frozen gate for the first real-model W1Ax calibration

Frozen before the row-A16 native diagnostic on 2026-09-29 UTC. The user chose
the short practical pilot. This gate can authorize only 100 hard-CE row-A16
steps on the already audited shard-0000. The original preparation bundle and
raw logits must remain unchanged and retained. Q4_0 EAGLE remains the primary
performance baseline. No final prompt is used.

## Inputs and sample

- First shard bundle manifest SHA256:
  `3ee7a8f4526f1dbcca1b6e0ea0756e42eff81333d7a88137afebe7213d3c1947`;
  prompt SHA256:
  `968ffbb21b23f934912862ef6f7add7d03bf0cfbbb3910492f2f0f50075fc18a`.
- Candidate-D native capture and independent audit remain the teacher ancestry.
  They are not evidence that the row-A16 student has the same proposals.
- Row-A16 checkpoint-zero SHA256:
  `5b371f79831c4c8da0ffc4bc5a0a4b9a6817a1fdf7c2bddfeaec6b001c4f6b78`;
  exported GGUF SHA256:
  `fa9406b72fb6ef19e2eca0bc0891bc20099dc318283721af3f540e056ebd3f45`.
- The exact native diagnostic prompt file is the **first train prompt in
  shard order** for prose, reasoning and code: `dolly:line-005896`,
  `gsm8k:train-000315`, `mbpp:task-496`. Its ignored JSONL SHA256 is
  `b3f3570cf2e45c570678b98f135257a609de319b01232ce4353e4286eb595703`.
  It is a sample, not a replacement training split. The diagnostic config is
  [w1_phase1b_pilot_student_diagnostic.json](../configs/w1_phase1b_pilot_student_diagnostic.json).

For each prompt, compare the first exact-prefix proposal root. Then, in
ascending output-position order, compare the first later root whose complete
accepted-prefix token IDs appear in both the candidate-D capture and native
row-A16 trace. Prefer one after an accepted continuation and one after a
verifier rejection if both exist. Record all unmatched candidates rather than
substituting a conveniently matching prompt. The deterministic rule is frozen
here before the row-A16 diagnostic, so later root selection cannot tune the
gate to a favorable outcome.

## Pass/fail checks

1. Reverify the prompt, source cell, bundle, snapshot, target/D/row-A16 model,
   d2t and server hashes. Repeat exact-prefix and response ancestry audits.
   The seed is conditioning context, and terminal-only emissions are excluded
   from supervised rows. Initial/terminal sampler arithmetic is outside this
   calibration gate and remains explicitly unverified.
2. On the selected roots, require exact token IDs, absolute d2t mapping,
   decoder positions, causal visibility and cache lengths. Require finite
   target features, student logits and gradients; F16 K/V storage and the
   frozen native/borrowed embedding rows must agree on the selected tokens.
   Recheck exported hard sign bits and row scales against the checkpoint.
   Any ancestry, mapping, mask, cache or operand-identity mismatch blocks.
3. Compare Torch CUDA row-A16 root proposal IDs to native row-A16 proposals.
   Count every disagreement and the Torch top-two raw-logit margin. A changed
   top choice with margin above **0.02 raw logit units** blocks. A disagreement
   at or below 0.02 is a recorded near tie and requires the bounded native
   verifier trace to show no wrong acceptance and the matched Q4_0 response
   IDs to remain exact. Measure logit/state error where native head states are
   captured; do not assert bitwise backend parity from this gate.
4. Before training, enumerate eligible provider rounds, supported labels and
   exact-prefix joins under a new calibration-only readiness contract. The
   hard-CE provider manifest must have no compact-teacher attachment. A
   versioned gate report must hash every input and record each result. Any
   failed or missing check keeps the provider ineligible. Do not clear the
   original bundle's four unverified gates or mark it full-body eligible.

If all checks pass, run exactly one supervised 100-step row-A16 CUDA hard-CE
calibration. Record per-step time, peak GPU/process memory, finite loss and
gradient coverage, output checkpoint hashes and any early stop. The run is a
compute and execution calibration; no Q4_0 quality or throughput claim follows
from training loss. No full-tier capture, four-width sweep or final evaluation
is authorized by this gate.

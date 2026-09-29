# Joint row-A16 calibration on RTX 5080

The authorized real-model calibration completed **100/100 optimizer steps** on
2026-09-29 UTC, with exit code zero. This measures training execution and
resources. It does not establish learned acceptance, native latency or a gain
against the primary Q4_0 EAGLE baseline.

## Gate and execution

The frozen [pilot gate](w1ax-phase1b-pilot-gate.md) passed all five versioned
checks. Nine selected roots cover prose, reasoning and code, including roots
following accepted continuations and verifier rejections. Torch/native state
and logit relative RMS maxima were `3.551677e-5` and `7.771191e-5`, with zero
proposal disagreements. Exact chronological positions, token IDs, prefix
lengths, projected-to-stored F16 K/V and causal masks passed. The selected
prenorm reconstruction bridge had maximum relative RMS `8.242505e-8` and
maximum absolute error `9.846686e-7`; it is a reconstruction, not an observed
exact normalized callback. Three measured row-A16 response ID sequences matched
Q4_0. The separate backward check found all 18 gradients finite on each root.

Readiness report:
`runs/w1-pilot-readiness-final-20260929/readiness.json`, SHA256
`103396b1f25ca127da8ab2b13326a3db085e1629dcdedb479bc100698fbf1dcd`.
Provider overlay SHA256:
`e11ebb8806c8fb65caeb526da54da45c9ab7e99b5282d16b1f100985051adb94`.
Scope is only `row_a16_hard_ce_100_steps`; original bundle/provider and
full-body training eligibility remain false. No compact teacher is attached.

The sole coordinator supervised
`runs/w1-row-a16-hardce-calibration-20260929/` through tmux MCP session `$35`,
pane `%57`, with process group `22238`. Start/end were
`2026-09-29T21:18:16.322395+00:00` and
`2026-09-29T21:20:51.900143+00:00` (155.58 seconds including loading,
validation, checkpoint serialization and other overhead). Parent trainer source
was `82fb9cc`; native cache evidence used published gitlink `83a070ec`.
No other project GPU experiment ran concurrently.

Hardware was NVIDIA GeForce RTX 5080, CUDA `cuda:0`, compute capability 12.0
(SM120), under WSL. All nine body/head linears use hard binary weights,
nonnegative F32 row scales and an F16 activation-boundary cast with F32 dense
training arithmetic. Latent weights/scales are optimized with the declared STE;
this is not a packed native training kernel. Captured native target features
and labels condition the unroll; prefix reconstruction is under `no_grad`,
while proposal-state/K/V recurrence remains differentiable. Frozen target,
embedding, norms and d2t operands remain unchanged.

Command:

```sh
python3 scripts/remote_job.py w1-row-a16-hardce-calibration-20260929 -- \
  .venv/bin/python scripts/train_joint_w1ax.py \
  --activation-bits 16 --scale-layout row --objective hard_ce \
  --device cuda:0 --allow-accelerator --steps 100 --require-complete \
  --provider w1ax_capture_provider:create_provider \
  --provider-manifest runs/w1-pilot-readiness-final-20260929/provider.json \
  --output-dir runs/w1-row-a16-hardce-calibration-20260929/checkpoint
```

## Measured calibration

| Measurement | Result |
| --- | ---: |
| Completed optimizer steps | 100 |
| Active gradient tensors per step | 18 of 18 |
| Finite losses, gradients and updated parameters | All steps |
| Loss first / last | 6.127609 / 6.774653 |
| Loss minimum / maximum | 3.785579 / 11.090801 |
| Sign flips summed over steps | 0 |
| Scale L1 movement summed over steps | 23.751972 |
| Latent weights outside clipping interval | 0 |
| Synchronized step time, total | 92.255111 s |
| Step time, mean / median | 0.922551 / 0.737187 s |
| Step time, minimum / maximum | 0.354521 / 16.701416 s |
| CUDA allocator peak allocated | 9,119,919,104 bytes (8.494 GiB) |
| CUDA allocator peak reserved | 10,789,847,040 bytes (10.049 GiB) |
| Process lifetime peak RSS | 12,459,868,160 bytes (11.604 GiB) |

The maximum step time was the first step and includes startup effects within
the timed region. Timing synchronizes CUDA immediately before feature transfer
and after the optimizer, covering transfer, prefix reconstruction, forward,
backward and optimizer work. CUDA memory is the PyTorch allocator peak since
per-step reset, not whole-device use. RSS is the process-lifetime high-water
mark, including model loading; Linux KiB was converted to bytes.

The deterministic first 100 captured rounds contain 495 supported labels and
9,393 exact prefix joins. They cover only `dolly:line-000446` and
`dolly:line-001397`; the underlying 31-prompt shard and separate numerical gate
span all three domains. Different rounds have different losses, so the first
and last loss are not a matched convergence comparison. Zero sign flips and
nonzero scale movement do not prove useful quality improvement.

## Artifact ancestry and terminal state

All raw data, models and checkpoints remain outside Git under the remote
project directory. Exact output hashes:

| Artifact | SHA256 |
| --- | --- |
| `checkpoint/joint.npz` | `2179b29fde4b9c435e02758621b563f4f4099ac799862c4822f2ace1c0c89825` |
| `checkpoint/joint.json` | `66447942b17cc6b53ec61a5adc3f80854092cd1c7fcfde8df0953d5760026b4d` |
| `checkpoint/training_run.json` | `ff6a31e923f1b6152111ed90db01e44e6f2793b2df43bcd912a6783b3e0c7273` |
| `state.json` | `cdacdb076aad06a6a0efbe46ca19c6e27fb3c352590fa3a92f326c15fdc8d26d` |
| `summary.json` | `650d65de202f9062821882f73a8b4fe3b84a34c9767eb24fbf801d38b5f8c376` |

Input bundle manifest SHA256:
`3ee7a8f4526f1dbcca1b6e0ea0756e42eff81333d7a88137afebe7213d3c1947`;
ordered prompt SHA256:
`968ffbb21b23f934912862ef6f7add7d03bf0cfbbb3910492f2f0f50075fc18a`;
model snapshot manifest SHA256:
`2db1c860f059bd8702ca6bb523e64f0c7d064c7a95f59919a001fc88168a91e3`;
base draft GGUF SHA256:
`c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.
The readiness report binds target/D/export/checkpoint-zero/d2t, executable,
trace and audit hashes. See the [source capture report](w1ax-shard0000-capture-5080.md)
for original raw-logit and target/D hashes.

The supervisor is terminal, and process group `22238` has no remaining member.
RTX 5080 returned to 0% utilization and 2,843 MiB whole-device baseline use.
No OOM or memory workaround was required. The saved checkpoint declares
`row_w1ax_requires_native_validation`; no trained export or acceptance test has
run. Reserved-final data remains sealed. Full-tier capture, four-width training
and final evaluation require the user's research decision. Authorized runtime,
storage and refresh engineering remains in the active project queue.

# Continuous W1A8/W1A1 experiment preparation

Prepared in task `01a0f01d-65c6-7af0-9660-99c07e95cacd`, continuing the existing
joint body/head goal. This report records CPU preparation, not CUDA fit,
convergence, trained acceptance or throughput. Final CPU audit/integration is
recorded below after it completes. Native operations and training are USER-start.
Q4_0 is the primary comparison; native target/verifier F16 and frozen
embeddings/norms/mapping remain unchanged. No final set is used for selection.

## Delivered artifacts

| Requirement | Implementation / evidence | Boundary |
| --- | --- | --- |
| Diverse, grouped scalable data | `prepare_continuous_w1ax_data.py`, locked public revisions/licenses, frozen metadata, opaque leakage checks, CPU tokenization | Prompts only; sufficiency and semantic cross-legacy contamination unproven |
| Deployable input packet | `pack_continuous_w1ax_inputs.py`; deterministic verified tar, no sealed/reserve prompt payload | Manual transfer through tmux MCP |
| Audited teacher/provider | `w1ax_continuous_stages.py`, v2 `NativeCaptureProvider` / `StreamingNativeProvider` | Fresh native label/features only; no CPU teacher substitute |
| A8/A1 eligibility | `check_continuous_w1ax_readiness.py`: independent export/cache/numeric/backward reports, six train roots per width | Actual CUDA gates USER-start; A16 calibration cannot authorize them |
| Concurrent training | `continuous_qat.py`, `train_continuous_w1ax.py`, `configs/continuous_w1ax.json` | Two independent models/optimizers, identical rounds, interleaved minibatches |
| Continuous operation / resume | Atomic checkpoint directory publication, exact CPU state/RNG/cursor tests, source/runtime checks, orphan recovery, stop/signals | Unbounded by default; caps explicit; code/data/runtime changes rejected |
| Resource ownership | Shared GPU owner lock, RAM/disk/CUDA admission, bounded logs/checkpoints, controlled native child sessions | No fallback precision/hardware, no unmanaged second trainer |
| Development | Fixed 24 new prompts, Q4_0/native A8/A1 acceptance + declared teacher-forced loss rows, serialized offload, deadlines/retention | No timing claim, no sealed access, no automatic final evaluation |
| Hourly health | `check_continuous_w1ax_health.py`, saved PAUSED heartbeat and supported setup notes | No active monitor; Luna-only heartbeat override unavailable in tool |
| Manual operation | `docs/CONTINUOUS_W1AX_RUNBOOK.md` | Training survives coordinator exit; monitor cannot tune/restart/terminate |

The forward quantizes all nine selected linears with row-scale one-bit weights;
A8 signed absmax/nearest-even activations and A1 signs/mean-absolute activations
follow their independent deployment contracts. Dense F32 matmul performs the
training simulation; it is not native binary acceleration. Frozen exceptions
remain ordinary arithmetic. Context cache reconstruction uses current parameters
under a declared no-gradient boundary; proposal-unroll states and K/V remain
attached, with later-only state AND K AND V gradient tests.

Default optimizer: AdamW, sign LR1e-3 /scale LR1e-5, .9/.999 betas, epsilon1e-8,
no decay, clip1, foreach false; 100 paired warmup updates then constant LR.
Both variants begin from original dense signs/mean-absolute row scales, not
candidate-D group scales. Seeds8101/1101 have separate state. Sign flips, latent
near-zero/bound fractions, scale movement, clipping, later gradients and step
times are exposed. Zero flips after2000 pairs warns for manual inspection only.
A1 saturation is N/A. No learning-efficacy or finite-budget sufficiency claim.

## Frozen data and actual preparation

[Data report](continuous-w1ax-data-preparation.md) includes exact revisions,
licenses, categories, groups, lengths, tokenization and all shard hashes.
Primary ignored manifest: `data/continuous-w1ax/freeze-002/manifest.json`;
[tracked exact metadata](continuous-w1ax-freeze-002.manifest.json), SHA256
`9fc8caa80f2c29785eeaeff01f3875c27fee46853350de9ceebe682f8d12b8dc`.

10,000 train prompts (3,334 prose /3,333 code /3,333 arithmetic reasoning),
1,002 dev, 1,002 sealed test, 24,507 independent reserve. Train input coverage
is1,502,567 chat tokens; dev161,458. These count input positions once per unique
prompt, not vocabulary diversity or supervised predictions. Train p50/p95/max
is87/393/1301 tokens; cap1792, no truncation. This chiefly covers short/medium
contexts. There are eight Dolly categories and nine code languages. Reserve is
unbalanced and not silently included. Source answers/seeds are not teacher labels.

Source/script SHA checks match the original manifest. Actual opaque split audit
passed IDs/groups/topics/message-hash disjointness; all index hashes plus allowed
train/dev/reserve payload hashes passed. Sealed text was not reopened. Legacy
final indexes lack semantic shingles, so cross-legacy semantic/near leakage is
unproven. Intermediate freeze-001 and historical final sets remain preserved.
Source+two freezes occupy~301MiB; weights/raw captures remain outside Git.

Actual tar packet: `data/continuous-w1ax/launch-inputs.tar`, 22,016,000 bytes,
61 members, SHA256
`b5ce086e87dbdc57e7a0543607f5834ee201fb0c585572fcb63bc9379cb3dc31`.
Its original manifest/source lock, train/dev files and opaque indexes were verified;
zero sealed/reserve payload members. No transfer or service was started.

## Native stages and numeric policy

The CPU resolver reads retained source records and the actual pinned train/dev
inventory; it creates≤32-prompt native shards, three fixed train probes and
24 fixed independent dev prompts (8/domain, input≤512). It does not open seals.
Manual start performs separate width gates BEFORE large capture, then label-only
native capture/audit, provider coverage audit and dual-model CUDA smoke before
optimization. The common offline corpus uses candidate-D/A16 native exact
prefixes for comparable teacher-forced presentations to both students. It is not
current-student on-policy data. Old labels cannot be attached to changed prefixes.

The fresh label-only schema audits native sampler labels, request/response
continuity, task/round/prefix joins, masks/EOS, accepted features and map hashes.
It disables full target logits; old raw streams/eligibility flags remain intact.
V2 eligibility is a new hashed external readiness binding, not mutation of a
preparation bundle. The streaming provider retains its first model-loading shard
plus one active shard, with explicit epochs/repeated versus unique coverage.
Default substantive tier requires10,000 observed train prompts/100,000 supervised
rows; this is an admission floor, not enough-data evidence.

Each precision gate has at least six audited training roots (two per domain),
exact exported bit/scale/frozen-operand checks, native stored F16 cache/masks,
Torch cache/positions, 18 finite/nonzero sign/scale gradients and later-only
state/K/V paths. Practical state/logit relative RMS threshold0.10, no changed
native choice with margin>0.02; lower-margin changes are recorded. These bounded
checks address deployment/training risks, not HF/native target bit parity.
Native cache instrumentation is bounded at1024 executions/1GiB per width.
No numeric gate, capture, native inference or optimizer stage ran in preparation.

Retained native runtime is b4e366d4, frozen manifest SHA256
`a199cfdabd81b5ba7414509e31007eab338c125b881a9276a78696cd9208fd5e`,
server SHA256
`b5093749d67888bc2cafdb6a65c479f4c182f0a904820f1dae4870b6ae66d41c`.
Actual binary/shared-library inventories and mapped libllama/libggml verification
are required at user start. Q4_0 SHA256 is
`2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`;
stored-type audit checks all nine Q4_0 projections. It is not labeled W4A4.
Parent native gitlink stays published b4e366d4; no new native commit was needed.

## Estimates and safeguards

CPU arithmetic:218,234,880 latent weights +65,280 scales per lane. Shared frozen
embedding has151936×2560 F16 elements. Dual model/Adam persistent state~5.61GiB
with shared embedding, one active gradient~0.813GiB, assumed graph3GiB gives
~9.42GiB peak. This assumption excludes uncertain fragmentation/external use;
actual CUDA admission decides fit. Historical A16 one-model8.494GiB allocated/
10.049GiB reserved is context, not new A8/A1 evidence. Hard-sign tensors are
reused once per unroll while preserving autograd; gradients are cleared per lane.

Paired resume state~4.88GiB; three retained generations plus one atomic staging
generation and NPZ exports~26.0GiB, before coverage metadata/logs/capture artifacts.
GPU floor≥1GiB free, reserved peak≤12GiB. Linux host floor2GiB with additional
12GiB conservative model-loading admission, measured copy/offload storage and
checkpoint transfer/export buffers. Host availability estimates are not memory
reservations. Resource/nonfinite checks occur in-process, not only hourly.

From actual11,002 train/dev indexes (1,664,025 input tokens), a128-token output
cap gives~87.90GiB accepted-feature projection if all outputs reach the cap.
Native+accepted feature upper estimate~377.25GiB, normalized draft-head states
~67.15GiB, metadata allowance~135.69GiB: **580.09GiB conservative total**, plus
10GiB free floor; checkpoints/gate artifacts add more. This is source arithmetic,
not measured teacher storage. Full-vocabulary target logits are0 planned bytes.
Native counts/EOS, file overhead and actual feature events change real size.
Per-shard caps/floors fail safely; no silent shrink/precision/hardware substitution.

Development runs every1000 pairs after a checkpoint and GPU offload of BOTH
models/moments. Native A8/A1/Q4_0 rollouts use fixed24 prompt IDs, one repetition,
128 output cap; losses use up to two supported candidate-D teacher-forced rounds
per same prompt, recording IDs/rounds/prefix hashes. These are different prefix
distributions, explicitly distinct metrics. Deadline1200s; last3 owned attempts
and last256 compact summaries retained, including partial attempts. The full1002
independent dev prompt pool is captured once, allowing future wider validation;
no automatic wider/final evaluation or throughput claim. Exact resume pins all
sources, math code and Python/Torch/NumPy/declared CUDA runtime precision settings.

## Manual refresh and recovery

Start/status/stop/resume/extraction/CPU-resolver commands are in the runbook.
All SSH uses the shared host registry and tmux MCP. Each GPU run is wrapped in
`remote_job.py` with a unique ID and300-second stop grace. Native server sessions
record owner PID/PGID, cleanup on signal/STOP/deadline, and defer Python signal
exceptions through spawn/assignment so detached servers are not orphaned.
CPU recovery refuses live owners/unknown groups and quarantines owned partial
captures without deleting raw data; complete cells can rebuild labels on CPU.
No automatic infinite retry/restart occurs.

To prepare a current-student trajectory refresh, STOP training first. Select a
checkpoint's `A8` or `A1` subfolder and a declared unsealed training microshard
from the frozen stages config. Use its exact recorded prompt SHA256:

```sh
python3 scripts/remote_job.py --stop-grace-seconds 300 continuous-refresh-a8-001 -- \
  .venv/bin/python scripts/w1ax_continuous_stages.py refresh --allow-cuda \
  --stages-config runs/continuous-preparation/stages.json \
  --checkpoint-dir <absolute-checkpoint-dir>/A8 --activation-bits 8 \
  --prompts <declared-training-microshard.jsonl> \
  --prompts-sha256 <recorded-full-sha256> --output <new-absolute-refresh-dir>
```

For A1 use its subfolder, bit1 and a new run/output ID. The refresh exports the
actual current checkpoint, acquires the same GPU lock, enforces RTX5080/SM120,
recaptures native exact-prefix labels/features and writes hashed receipt/ancestry.
The new bundle remains preparation-only until a new explicit readiness/provider
binding. No automatic append to an exact-resume corpus or reuse of changed-prefix
labels. The v2 provider now accepts an explicit captured-drafter binding containing the
actual exported GGUF/audit, NPZ/manifest and refresh receipt. Candidate-D remains
the frozen norm/map reference, not a substituted current teacher. Missing/wrong
actor binding fails before model loading. CPU binding tests preserve the raw
capture's false eligibility flag and qualify both precision consumers only with
new explicit readiness that includes the refreshed capture SHA. Publish that
provider on CPU after supplying the new readiness evidence:

```sh
.venv/bin/python scripts/w1ax_continuous_stages.py make-refresh-provider \
  --stages-config runs/continuous-preparation/stages.json \
  --refresh-receipt <new-absolute-refresh-dir>/refresh.json \
  --readiness <new-hashed-readiness.json> --output <new-provider.json>
```

The command audits both consumers before publication; it never creates eligibility
reports or appends data to an exact-resume experiment. Refresh inputs must match
a declared TRAIN microshard path and hash in the full hashed stages config before
payload access. Renamed/custom/mixed/sealed files are rejected. The decision to
start a new expanded/refresh experiment remains user-owned.

## CPU validation and preparation boundary

Local hardware:Darwin/arm64 CPU, Python3.11.15, Torch2.14.0. Tests cover real
source/shard/mask/schema/serialization/cancellation/resume contracts with small
CPU graphs and synthetic native evidence; they do not prove CUDA execution.
Final guarded suite counts/log hashes are recorded below after integration.

Boundary incident: earlier CPU regressions called legacy `tiny_joint_fixture`,
which used global `torch.manual_seed`. That calls CUDA/MPS/XPU/MTIA RNG APIs even
when tensors stay on CPU. Earlier wording that no accelerator API was touched
was too strong. The helper now uses the CPU generator only (`3f1cd89`); new
continuous fixtures already use CPU-specific generators. No native GPU capture,
model inference/training, WSL connection or recurring monitor execution occurred.
Torch2.14 CPU AdamW also consults the accelerator registry. Final CPU audit
stubs that registry to None and guards backend seeding/discovery/memory/stream
operations before test imports, rather than relying on tensor device labels.
These checks validate CPU fixtures; they do not certify unguarded third-party
library internals as free of accelerator API side effects.

Hourly heartbeat `a8-a1-health-check-enable-after-manual-start` is saved PAUSED,
attached to this preparation chat. Its tool exposes no model/reasoning override;
current chat Sol does not guarantee Luna-only future heartbeat execution.
[Monitor setup](../docs/CONTINUOUS_W1AX_MONITOR.md) records supported Luna/high
configuration and the exact limitation; leave paused until verified after user
start. A standalone model-pinned schedule requires the user's explicit choice;
it was not silently substituted. No Sol/Astra coordinator is required for the
long-lived training process, and the monitor cannot tune/capture/evaluate/restart
or terminate it. Sealed finals remain unopened; acceptance/convergence/storage/
real CUDA fit are unverified USER-start gates.

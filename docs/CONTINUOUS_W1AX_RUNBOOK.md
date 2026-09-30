# Continuous W1A8 / W1A1 joint QAT

CPU preparation is complete after the final audit. Native capture, CUDA gates,
training and monitor activation are **manual**. Public prompts are not training
tensors. Q4_0 is the primary native comparison; target/verifier FP16 precision,
borrowed embeddings, norms and mapping stay frozen. Final sets remain sealed.
See the [preparation report](../experiments/continuous-w1ax-preparation.md) for
hashes, tests, estimates and limitations, including the corrected legacy RNG call.

## What runs

One long-lived process keeps two independent models, Adam optimizers and RNG
states. It advances A8 then A1 on identical audited rounds, with one autograd
graph live at a time. This interleaves minibatches; it does not run two complete
experiments sequentially. Frozen resources may be shared. The accepted-prefix
cache is recomputed with current weights under a declared no-gradient boundary;
within each proposal unroll, later losses reach earlier student states AND K/V.
No target inference is substituted with CPU/Hugging Face labels.

Defaults: original dense signs/mean-absolute row scales; all nine body/head
linears; hard native-style W1A8/W1A1 forward and STE; hard sampled-target CE.
AdamW sign LR1e-3, scale LR1e-5, betas .9/.999, epsilon1e-8, no decay,
clip norm1, foreach disabled, 100-pair warmup then constant LR. These are
unvalidated learning defaults. Sign flips, scales, clipping and recurrent
gradients are logged; A1 saturation is N/A. No automatic tuning, rescue loop,
architecture switch or stop for slow learning.

## Local package and host setup

Prepared packet: `data/continuous-w1ax/launch-inputs.tar`, 22,016,000 bytes,
SHA256 `b5ce086e87dbdc57e7a0543607f5834ee201fb0c585572fcb63bc9379cb3dc31`.
It has 10,000 train/1,002 dev prompts, original manifests and opaque indexes;
sealed/reserve prompt payloads are omitted. The local full freeze also preserves
1,002 sealed tests and 24,507 reserve prompts. Source revisions, licenses and
reproduction/expansion commands are in the [data report](../experiments/continuous-w1ax-data-preparation.md).
No finite tier is promised sufficient; repeated presentations are counted.

Read the registered RTX5080 host/workdir via `python3 scripts/agent_env.py status`
and `~/.config/binary-eagle-decoding/hosts.toml`. All agent SSH/transfers must use
tmux MCP. Transfer the packet through that session; no transfer occurred in
preparation. On the registered artifact host, inside its project directory:

```sh
git pull --ff-only
git submodule update --init
tar -xf <transferred-launch-inputs.tar>
.venv/bin/python scripts/w1ax_continuous_stages.py prepare-config \
  --corpus-manifest data/continuous-w1ax/freeze-002/manifest.json \
  --output runs/continuous-preparation/stages.json
.venv/bin/python scripts/train_continuous_w1ax.py --estimate
```

`prepare-config` is CPU only. It defaults to retained hard-CE provider
`runs/w1-shard0000-provider-hardce-20260929/provider.json`, frozen b4 runtime
`runs/w1-final-runtime-timing-freeze-20260929/runtime/llama-server`, and
`models/gguf/Qwen3-4B-eagle3-q4_0.gguf`. If retained paths differ, pass
`--legacy-provider`, `--binary`, `--q4-draft`; hashes must still match. It verifies
binary/shared libraries and Q4_0, writes 32-prompt shards, three training gate
probes and a fixed balanced 24-prompt development subset. It never opens seals.
Use the existing pinned `.venv` (Torch2.14); no new service is needed.

## Manual start

Choose an absolute experiment directory under the registered project and a
unique supervisor ID. Start inside a persistent tmux MCP session:

```sh
python3 scripts/remote_job.py --stop-grace-seconds 300 continuous-a8-a1-001 -- \
  .venv/bin/python scripts/train_continuous_w1ax.py --start --allow-cuda \
  --stages-manifest runs/continuous-preparation/stages.json \
  --run-dir <absolute-experiment-dir>
```

The launcher checks RTX5080/SM120, then independent A8/A1 checkpoint-zero export,
cache/mask, numeric and gradient gates before large native label-only capture.
It audits the frozen train/dev teacher shards, requires the configured 10,000
train prompts/100,000 unique supervised rows, and runs a full-size dual-model
CUDA memory/backward smoke before optimization. Calibration-only A16 records
cannot qualify A8/A1. Any failed gate stops; no precision/hardware fallback.
No full-vocabulary target logits are captured; historical raw data stays intact.

Default run has no step/token/time/epoch cap. Optional caps are in
`configs/continuous_w1ax.json`; copy it to a resolved run config and pass
`--config` to change caps/settings deliberately. Status reports stage progress,
separate model metrics, unique coverage versus repeated presentations and
checkpoint/runtime provenance. Training survives a coordinator/chat exit.

## Status, stop, resume

```sh
.venv/bin/python scripts/train_continuous_w1ax.py --status --run-dir <absolute-experiment-dir>
.venv/bin/python scripts/train_continuous_w1ax.py --stop --run-dir <absolute-experiment-dir>
```

Wait for stopped status and supervisor exit. SIGINT/SIGTERM/SIGHUP also stop
owned native sessions and request a paired training boundary. The future GPU
owner checks its process groups before reporting the GPU free. The supervisor
allows 300 seconds to write large checkpoints and bounds stdout logs.

Resume with a NEW supervisor ID, identical source/config/runtime and `--resume`:

```sh
python3 scripts/remote_job.py --stop-grace-seconds 300 continuous-a8-a1-002 -- \
  .venv/bin/python scripts/train_continuous_w1ax.py --start --allow-cuda --resume \
  --stages-manifest runs/continuous-preparation/stages.json \
  --run-dir <absolute-experiment-dir>
```

It restores both optimizers, per-lane/global RNG, counters and cursor. A crash
mid-pair resumes the previous durable pair; a fully published checkpoint whose
`latest.json` write failed is verified and recovered. Runtime/math changes or
corrupt state fail safely. Only caps may change during exact resume.

Interrupted capture before optimization may resume without a model checkpoint.
If an incomplete native folder remains, preserve it with the CPU recovery tool
before the explicit retry (it refuses live owners/unknown process groups):

```sh
.venv/bin/python scripts/w1ax_continuous_stages.py recover-partial \
  --run-dir <absolute-experiment-dir> \
  --stages-config runs/continuous-preparation/stages.json
```

Complete native cells can rebuild audited labels on CPU; incomplete cells are
quarantined, never deleted. No automatic retry or rescue loop runs.

## Development, resources and monitoring

Every 1,000 paired steps, checkpointed development evaluates A8/A1 versus the
frozen Q4_0 on 24 new prompts (8/domain, one repetition), plus up to two audited
label rounds per same prompt. Native acceptance is distinct from teacher-forced
loss; these instrumented checks are not throughput claims. Both training models
and moments move to CPU before serialized native/Torch validation. Default wall
cap is20 minutes, with three owned attempts/256 compact summaries retained.
The larger dev pool is captured once for independent future validation, not for
sealed selection. Full acceptance/timing/final experiments require later user work.

CPU arithmetic estimates: ~9.42GiB CUDA peak with assumed3GiB graph budget,
~4.88GiB paired resume state and ~26.0GiB checkpoint/export retention plus staging.
Actual fit is unknown. GPU requires≥1GiB free and reserved peak≤12GiB; host floor
is2GiB, with an additional12GiB admission before model load and measured storage
checks before copies/offload/recovery/exports. Capture forecast for11,002 prompts
is conservative~580.09GiB plus10GiB free floor, before checkpoints/gate artifacts.
Actual teacher storage must be measured at manual start. Disk/memory/nonfinite
safeguards run immediately in-process; logs/checkpoints have bounded retention.

[Hourly monitor setup](CONTINUOUS_W1AX_MONITOR.md) is **PAUSED**. After manual
start, configure this chat for verified GPT-6 Luna/High and record actual run
paths, then enable the existing monitor. Its tool has no per-heartbeat model
override; leave paused if Luna-only execution cannot be confirmed. No Sol/Astra
coordinator is needed to keep training alive. The monitor only checks health,
stays quiet while healthy and never tunes, restarts, evaluates or terminates training.

Manual trajectory refresh is a separate supervised, locked USER-start capture:
see the preparation report for its typed checkpoint/source binding and commands.
Changed-prefix labels are always recaptured natively; no silent data-source change
is allowed during exact resume. Sealed tests are never automatically evaluated.

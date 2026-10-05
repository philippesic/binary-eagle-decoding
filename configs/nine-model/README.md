The source declarations in this directory are preparation artifacts. They do
not allocate GPU time or grant campaign readiness. `profiles.json` lists direct
fixed A8, direct A1, and explicit A8-to-A1 reset options; an option is usable
only when its real family launcher supports it. Optional probes remain off.
`protocol.json` is a proposed ten-repeat development protocol, with a separate
native diagnostic pass. Its evaluation allowance requires human selection.

The frozen builder consumes an external `nine_model_bundle_inputs_v1` JSON.
Every artifact locator contains exactly `path` (absolute canonical regular file)
and `sha256`. Required fields are:

| Field | Required content |
| --- | --- |
| `inputs` | Immutable target, native binary and runtime libraries, protocol, prompt locator, and actual SM120 admission plan |
| `controls` | `eagle`, `dspark`, `dflash`: original Q4 model locator, `frozen_original: true`, explicit deployment coverage/exceptions |
| `candidates` | Six `<family>_a8/a1` records: actual training config, base model, initial calibrated GGUF and serialization audit, fusion calibration, completed data admission, profile, deployment coverage, actual loader/CUDA marker contract |
| `qa_ledger` | Independent `nine_model_qa_ledger_v1` locator; all portable requirements PASS with production evidence, current critical `source_files` hashes; fresh SM120 remains separately PENDING |
| `budget` | `nine_model_selected_budget_v1` locator with `human_selected: true`, six explicit training limits and train/export wall caps, admission/evaluation wall caps |
| `resource_policy` | Host/GPU free floors and return tolerances, all expressed in bytes |
| `gpu_uuid`, `gpu_control_path` | Actual selected device identity and shared machine-local pause registry path |

`--inspect-draft` reports missing preparation dependencies without loading a
model or touching a GPU. It cannot publish a production bundle:

```sh
python3 scripts/prepare_nine_model_bundle.py --inputs resolved-inputs.json --materialize-configs selected-configs
python3 scripts/prepare_nine_model_bundle.py --inputs selected-configs/resolved-inputs.json --materialize-admission-plan selected-configs/sm120-plan.json
python3 scripts/prepare_nine_model_bundle.py --inputs selected-configs/sm120-plan.json-inputs/resolved-inputs.json --inspect-draft
python3 scripts/prepare_nine_model_bundle.py --inputs selected-configs/sm120-plan.json-inputs/resolved-inputs.json --output frozen-bundle.json
```

The production builder calls the actual training config parser and completed
block-data admission API. It requires materialized model/data/calibration
artifacts; it does not silently capture data or repeat an entire corpus audit.
Admission-plan inputs additionally pin the actual backend-op test binary,
block graph test binary, native teacher binary/source revision and three bounded
original TRAIN golden manifests. EAGLE goldens use the three native taps and
remain distinct from its original speculative/tree corpus. The plan generator
invokes the real native/EAGLE/block/zero-update/portability CLIs; it constructs
no model and accesses no GPU. Its block source bindings omit the eventual
bundle hash to avoid a circular plan/bundle identity; admission later injects
the external bundle hash. Emitted plans are execution requests, never readiness
or training-admission receipts. Missing materialized source/data/model/golden
inputs remain PENDING.

It rejects unbound stage hooks. Training/export/evaluation commands invoke the
implemented source APIs and are pinned with current source bytes.

After human availability and recipe/budget selection, the operator uses the
shared host registry and tmux MCP to connect to the authorized RTX5080. The
prepared local-host command below must run inside a detached Linux tmux session
through `remote_job.py`. Replace the uppercase arguments with the frozen
artifact paths/hash and fresh exclusive availability lease; they are artifact
arguments, not source edits:

```sh
python3 scripts/remote_job.py RUN_ID --stop-grace-seconds 90 -- python3 scripts/run_nine_model_campaign.py --start --bundle BUNDLE_JSON --bundle-sha256 BUNDLE_SHA256 --availability LEASE_JSON --run-dir runs/RUN_ID/campaign --supervisor-state runs/RUN_ID/state.json
```

The lease schema is `nine_model_gpu_lease_v1`, with host `rtx5080`, actual
`gpu_uuid`, `user_announced_available: true`, `sole_owner: true`,
`pause_requested: false`, bundle hash, and a startup freshness interval no
longer than 300 seconds. It records existing human authorization. Freshness is
checked at initial admission; the unchanged ownership and current shared pause
flag are checked throughout the job. An expired startup interval does not
require another human approval for already-authorized automatic chaining.

The lifecycle runs fresh admission, candidate training, verifies committed
checkpoint and resource/context return, exports through the family serializer,
checks resource return, then evaluates all nine models plus target-only in a
fresh process. Runtime graph admission is separate from serialization proof.
Primary report ratios use each family's original Q4; EAGLE Q4 is an additional
same-device anchor. Detailed native round/dispatch traces and bounded memory sampling run outside
clean timing. Memory reports give whole-device sampled maxima and separate
per-process RSS maxima with cadence/lower-bound scope; they do not sum shared
RSS mappings or claim true allocator peaks. Actual Torch training allocated/
reserved peak measurements are separate.

STOP, pause, source/artifact mismatch, failed gates, wall caps and resource
failure preserve existing checkpoints and per-attempt receipts. Evaluation
requires all successful committed endpoints. Explicit `--resume` reuses exact
trainer resume state and recovers an already committed receipt after an
orchestrator crash. It refuses surviving owned kernel identities; an operator
must first stop those exact owned processes and verify resources. It never
terminates another team's processes. Linux `/proc`, CUDA PID census,
`/dev/dxg` holder identities where applicable, and host/GPU memory return jointly
gate release. Missing observers fail closed. Final prompt bytes remain sealed
through preparation/training; their locator is checked without opening bytes
until the authorized fresh final evaluator after the model freeze.

CPU fixtures prove lifecycle software paths. They cannot certify SM75/SM120
execution, real-model convergence, memory headroom or throughput. macOS cannot
validate Linux process-group cleanup; Linux-only checks remain separately
identified until the coordinated operator runs them.

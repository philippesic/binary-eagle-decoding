# Benchmarks

Native runs measure acceptance/speed; loss/top1 diagnose learning. Routine training
checks execute only the candidate against saved target/baseline/leader records. The
complete pause must fit 120 seconds every 1800–3600 optimizer seconds; qualify the actual
pause before production. Request scores and whole application tasks stay separate.

```sh
uv run --locked --no-sync --offline python -m benchmarks COMMAND INPUT [OUTPUT]
```

Writing commands default to the matching folder under ignored `artifacts/` and reject
destinations outside it. [Storage](storage.md#new-output-locations) defines relative names,
explicit paths and the shared saved-reference registry.

| Commands | Purpose |
| --- | --- |
| `plan`, `verify-inputs` | Native comparison units/warmups; optional model/runtime hashing |
| `native-build`, `native-helpers` | Authenticated runtime/helper builds |
| `curate`, `admit-bank`, `select-supplement` | Role-safe prompt preparation/selection |
| `prepare-project-workflows`, `prepare-replays`, `prepare-panels` | Project controls, measured histories, live/harness inputs |
| `capture`, `evaluate` | Teacher features; matched teacher-forced DEV loss |
| `compare`, `live` | Isolated native requests; executing tool episodes |
| `campaign-preflight`, `campaign`, `campaign-report` | CPU admission; training; read-only health |
| `prepare-checks`, `record-reference` | Frozen lean cohorts; authenticated reusable references |
| `prepare-initialization` | Reusable authenticated model basis; no training or capture |
| `suite` | Broader panels and optional state selection |
| `verify-greedy`, `verify-sampler`, `profile-native` | Independent decisions, distributions, kernel activation |
| `reduce-trace` | Structural round accounting, not an independent verifier |
| `calibrate-depth` | Serving draft depth for one target, drafter, runtime and GPU |
| `profile-rounds`, `round-profile-report` | Where a drafter's time goes inside one drafting round |

Execution commands freeze source and use [the numerical environment](development.md)
where needed. Preserve inputs, logs, ownership and failures. Comparison outputs must be
new; capture resumes verified records. Resume campaigns/suites with their recorded source
and specification. Checkout edits do not alter active runs. CPU preflight is not runtime
qualification. Run [local checks](development.md); there is no hosted CI.

## Routine DEV cohorts

| Panel | Units per candidate/target | Execution |
| --- | ---: | --- |
| Training screen | 60 requests, 128-token caps | One candidate pass; saved controls |
| Broader requests | 120 requests, existing longer caps | One pass |
| Prefix sensitivity | 6 groups × 3 prefixes = 18 requests | One pass |
| Live | 12 tasks, two per family | One round; failures stay denominated |

Screen categories: 15 code, 12 documents, 10 reasoning, 10 conversation, 8 structured/tool,
5 writing. Input bands: 30 below 512 tokens, 18 at 512–2047, 9 at 2048–8191, 3 at 8192–16384.
The early-answer cap makes this a progress indicator; broader requests retain longer
answers. Category estimates are noisy. Broader categories 30/25/20/20/15/10 and input
bands 60/36/18/6 preserve breadth. Prefix groups split 3 code/3 structured, each under
minimal/coding/tools. Eight measured application histories remain in request coverage.

`prepare-checks` accepts schema 1 with authenticated `requests`, `harness`, `live`
comparison specifications, producing workloads and multi-arm `*-establish.json` specs.
Use them to establish missing controls, then bind references for candidate-only execution.
One-time multi-arm control establishment is timed separately; the 120-second screen
deadline applies to routine candidate-only execution and complete training interruptions.
The [bank definition](../benchmarks/definitions/prompt-bank-v1.json) owns source identities
and the 6K TRAIN bank; smaller evals do not change training data. Targets remain Qwen3.5
4B Q4KM/NVFP4, with separate reference records.

Legacy short120/full600/harness90/live×3/sample120×3 evidence keeps its original meaning;
these are supported readers, not routine defaults. No default sampled sweep or repeats.
Additional seeds/repetitions answer explicit uncertainty, not an automatic checklist.

These are reused DEV cohorts. Independent confirmation requires fresh curation and
access receipts; a new label supplies no unseen data. Never consume heldout24 aliases.
Public dialogue is partly synthetic, planning dominates non-arithmetic reasoning, and
long inputs are mostly papers. Length is not difficulty; this is not measured traffic.
One-project trials cannot establish universal transfer. Workflow regressions cannot be
rescued by request scores. Broader candidate runtime aims for 15–20 minutes/precision;
measure it rather than treating that estimate as qualification.

### What each panel resembles

Measured on the Q4KM target-only arm; the training column is the frozen 6K TRAIN bank.
The current 220,095-update state was trained on the earlier 15,647-capture corpus.

| Property | 6K TRAIN bank | Training screen | DEV requests | Live tasks |
| --- | ---: | ---: | ---: | ---: |
| Mean input tokens | 625 | 1,556 | 1,366 | 4,747 context, 644 new per call |
| Inputs of 2,048 tokens or more | 8% | 20% | 20% | 87% of calls |
| Output per request | to EOS or 256–2,048 cap | 110 mean; 47 of 60 hit 128 | 400 mean | 92 per call, 1,024 cap |
| Project `tools` harness prefix | 200 of 6,000 groups | none | none; harness panel separate | every call |

The screen sees only answer openings, and inputs there are longer than in training. Live
calls reuse a cached context and evaluate about seven new prompt tokens per generated
token, so drafter prompt cost matters most there. Function-calling prompts (Glaive) count
as structured requests without the project harness. None of these is measured user traffic.

## Reusable references

`record-reference REGISTRATION REGISTRY` accepts schema 1 with authenticated `comparison`
(the desired cohort) and completed `result`, plus source `arm` and zero-based `rep`.
It authenticates whole source-panel completeness, selected raw responses/counters and
exact unchanged task definitions. Projection from a larger complete panel is explicit;
partial/invalid panels cannot establish references. Changed caps require new evidence.

The `artifacts/benchmarks/references` registry keys records by protocol and model content identity. Its first
canonical observation is reused, never overwritten. Separate observations remain under
`observations/`; deliberate repeats cannot silently replace the reference. Raw result,
specification and response/transcript receipts remain attached. Model identities persist
even if obsolete weights are later retired with receipts; executing weights still requires
full authentication.

Candidate-only comparison specs contain one `candidate` arm and `references` mapping
`target`, `baseline` and optionally `prior-leader` to authenticated registry records.
Preflight refuses missing, damaged, incomplete or incompatible evidence before training.
Protocol binds runtime/server/libraries, target precision, context/depth, role, exact
messages/tools/caps and generation policy. Cached target signatures validate fresh greedy
outputs/stopping. Hardware/driver identity is checked at execution. Reports distinguish
fresh rows from historical summaries and report acceptance differences with source-group
uncertainty. Historical speed ratios are descriptive, without paired timing intervals.
Changed runtime/hardware needs new controls; consequential close speed decisions need
fresh matched trials. Filesystem/tool tasks retain whole-episode failure denominators.

### Current protocol

Request policy v3 (`native-greedy-request-v3`, `native-live-v3`) serves with runtime-v6
(`artifacts/native/runtime-v6`), exact verification, depth 3, `--cache-ram 0`, F16 KV and
32,768 context. Target and Baseline references exist for both targets on all four cohorts,
and S4 is the prior leader on the screen, panel and prefix harness
([protocol switch](../results/benchmark-protocol-v3/result.md)). Everything they depend on
is in `artifacts/benchmarks/references/runtime-v6-depth3/`:

| Folder | Contents |
| --- | --- |
| `inputs/` | The multi-arm request/harness/live specifications given to `prepare-checks` |
| `lean-q4km/`, `lean-nvfp4/` | Lean workloads and `*-establish.json` specifications |
| `runs/` | The source panels the references authenticate; never delete while referenced |
| `establish/` | Target and Baseline live establishment, derived from `live-establish.json` (see [live](#live-application-and-instructions)) |
| `registrations/` | `record-reference` inputs, including refused ones |
| `specs/` | Candidate-only specifications with target and Baseline references and S4 as a placeholder candidate (`*-candidate.json`; `q4km-screen-candidate.json` is the campaign screen), `*-leader.json` that add the S4 prior leader where it has a reference, and `suite.json` |
| `executed-sources/` | The frozen code that ran the panels |

Swap the `candidate` drafter in a copy of a `specs/` file to check a new state. Records
of earlier protocols (runtime-v5, depth 2, policies v1/v2) stay valid for their own
protocol. A comparison refuses every reference whose protocol differs from its own, so
those controls cannot serve v3 runs, and current code cannot run the old protocol.
Acceptance and speed ratios from depth-2 or runtime-v5 panels are not comparable with v3.

## Prompts and captures

`curate` authenticates pinned sources and native token counts. `admit-bank` rechecks
selected fingerprints and known/opaque exclusions. Labels follow requests rather than
incidental context words. Tool stages are explicit; translations name a destination.
Optional `development_replays` retains eight first-file-read histories from measured
project trials, including failures, once per source group. Actual tools/messages and
transcript identities remain attached. Optional `harnesses` adds the intended `tools`
prefix to 100 TRAIN code-edit and 100 TRAIN tool groups. Preserve their ancestry/roles;
the alternate `coding` prefix stays outside TRAIN.

Selection keeps category/subtype/length marginals and required examples. It can choose
shorter complete records within cells to fit 399 decimal GB of maximum raw FP16 features,
leaving about 1 GB for metadata under 400 GB. FP32 raw features are twice that. No padding/
truncation manufactures coverage. Exact inputs/caps set the maximum, EOS the actual size.
These are dataset planning ceilings, not global disk limits. [Data](data.md) owns formats,
partial tails, mixtures and ancestry.

Capture resumes the exact specification in its existing output: authenticate completed
`progress.json` records and skip their IDs. Keep incomplete attempt directories and
failure receipts. Before restarting, inspect `attempt-N/owner.json` against process
birth/boot and actual GPU clients; a missing controller or tool session does not prove
the native process ended. Preserve healthy owned work. A changed bank, selection, target,
precision or policy needs a new specification/output rather than editing the old one.
Use the recorded source when recovering an interrupted execution; checkout edits do
not replace its provenance.

Unattended execution needs a persistent host session or detached process with a recorded
controller identity and owned log. Verify the native owner and advancing validated
records before handoff. Read-only inspection does not require launching another teacher.
Completion requires the terminal result, exact requested IDs and product validation;
an existing data directory or a live process alone is insufficient. A user cancellation
ends the allocation; old specifications do not authorize restarting it.

[`capture`](../benchmarks/capture.py) binds teacher/runtime/tokenizer, bank/IDs, dtype,
context, byte ceiling, exclusions and supervision policy. Every new record passes the
bounded product loader before publication: hashes, geometry, finite values, token/label
arrays and EOG. Validation receipts and partial attempts stay with ignored raw data under `artifacts/`.
[FP16 adoption](../results/fp16-storage-adoption/result.md) qualifies HE4/Q4KM storage;
training converts it to FP32 on the device. It does not establish lifetime/other-family equivalence.

`plan COMPARISON.json` reports native units/warmups; `verify-inputs` hashes models/runtime.
The older schema 1 [metadata example](../tests/fixtures/benchmark-plan.json) remains supported.
Training CPU preflight is in [execution](execution.md). Neither grants authority.

## Native comparison

[`native-comparison-v1`](../benchmarks/comparison.py) binds build/server/libs, target,
arms, workload, repeats/warmups, context/depth, role and tier. References use `path`,
`sha256`, `bytes`, relative to the specification or absolute. Fresh control runs start with `target`; cached runs execute speculative arms only.
Keep external baselines and candidates distinct. Server/libs must match their build.

Greedy arms must match target token IDs, text and stopping; actual returned generation
settings are checked. Native servers set `GGML_CUDA_DECODE_CONSISTENT=32` so verification
batches of up to 32 tokens use one-token arithmetic; without it greedy outputs differ after
near-ties. Its cost and depth limit are [measured](../results/cuda-decode-consistency-cost/result.md). Accepted/proposed counters reconcile with independent native logs;
fallback-only arms are refused. Correction/bonus tokens are not accepted drafts. Zero
proposals make acceptance unavailable.

Request time covers pretokenized HTTP completion including prefill/decode, excluding
rendering/tokenization/load/warmup. Live time includes harness/tools. Models are loaded
and warmed, thinking disabled, KV F16 and draft policy fixed. Independent requests/tasks
start with empty caches; live sessions reuse prefixes after their first call. `gpu.txt`,
commands, loaded libraries and ownership receipts identify execution. The lease preserves
other tasks' jobs.

Reports use total tokens/total time, acceptance totals, EOS/cap counts and category/length/
harness breakdowns. Panels draft at depth 3 for both targets (`--spec-draft-n-max`),
so a round proposes at most three tokens; 53% acceptance means about 1.6 accepted per round.
Depth 3 is Baseline's best exact depth and within about 1% of each measured drafter's own
best ([depth fairness](../results/draft-depth-fairness/result.md)); the earlier depth 2
overstated HE4/Baseline speed ratios by about 8%. Summaries therefore add `derived_rounds` (emitted − accepted − requests: each round emits
its accepted drafts plus one target token, and the first token comes from prompt processing)
with proposed, accepted and emitted tokens per round. On the 1,200-request Q4KM panel this
matched the native round counters within 0.1%. Requests ending at EOS match exactly; a
request stopped by its token cap can count one round more than the runtime's draft-round
counter, consistent with a final capped token emitted without a draft. `speed_vs_target` and `speed_vs_baseline` divide each arm's tokens/s by
that control's; against historical controls they are descriptive, not paired timing.
`server_prompt_seconds` and `server_prediction_seconds` sum the native server's own
prompt-processing and token-prediction partitions (null when a historical row predates
them). Drafters add prompt work, so keep the split visible: in the live panel it cancelled
the decoding gain ([standing](../results/he4broad-final-standing/result.md)). That extra time is
runtime overhead shared by every DFlash-style drafter: host-side feature copies and larger
context checkpoints, not drafter compute ([prefill overhead](../results/speculative-prefill-overhead/result.md)).
Runtime-v6 removes most of it with outputs identical to runtime-v5
([fixes](../results/drafter-prompt-overhead-fixes/result.md)), and panels use it.
Servers run with `--cache-ram 0`: the RAM prompt cache saved the previous request's state
before each new one (about 90 ms per request, outside `server_prompt_seconds`) and nothing
reused it. Live sessions reuse their prefix from the slot's own cache, which stays on.
Paired group bootstraps preserve category composition and keep repeats/
variants together. Sparse strata make intervals unavailable; multiple candidates also
receive adjusted intervals. Wide intervals are inconclusive. Compare matched arms within
targets, without pooling away regressions or mixing absolute scores across changed eras.
Sampled mode declares temperature/top-p/top-k/min-p, three seeds and target/runtime-bound
qualification. Unequal paths get descriptive robustness, not greedy equality or checkpoint
ranking by throughput.

## Depth calibration

`calibrate-depth SPEC [OUTPUT]` picks the draft depth to serve a trained drafter with on a
given target, runtime and GPU, in two to four minutes. It is separate from training and
from the fixed benchmark depth. Schema `depth-calibration-v1` names an authenticated
greedy DEV `comparison` (runtime build, server, libraries, target, workload, context),
the `drafter`, `tasks` and `cost_tasks` counts, `survival_cap` and `cost_cap` output
caps, `max_depth` (2–15) and `exact` (exact verification, `GGML_CUDA_DECODE_CONSISTENT=32`;
false sets 0). Example: [panel calibration](../results/draft-depth-calibration/spec-example.json).
Servers take the benchmark arguments, including `--cache-ram 0`; calibrations recorded
before that change also counted the RAM prompt-cache save in their request overhead.

It runs under the GPU lease:

1. A survival run drafts at `max_depth` on `tasks` requests spread over categories and
   input lengths. The server's per-position counters give the share of rounds whose first
   *i* drafts were all accepted.
2. Short cost sessions at every depth from 0 (target only) to `max_depth` run `cost_tasks`
   requests chosen across input lengths. Each gives decode milliseconds per round, split
   into the drafter's own forward (server `dur` counter) and the rest (target verification
   and round overhead). The target cannot verify a batch without a drafter in this
   runtime, so the target part is measured through the drafting path.
3. The model predicts tokens per round at each depth by splitting each deep round's
   accepted run into rounds of the shallower depth. Summing per-position shares instead
   undercounts accepted drafts per round at shallow depths by up to 14%. It combines this with the round cost
   into decode and complete-request tokens/s. Complete time keeps the survival run's
   per-request prompt and request overhead, scaled by the measured ratio for target only.

`calibration.json` in the output records identities (build, server, libraries, target,
drafter, workload, GPU, code), settings, the cost curve, survival shares, predictions,
the best depth, depths within 2% of it, the gain over target only and per-category best
depths. Other per-request rows and server logs stay in the session folders. Cite the
record by its path and hash. Measured accuracy against panels is in
[the result](../results/draft-depth-calibration/result.md). Predictions are for the
calibration's own capped requests; outputs do not check exactness, which remains the
benchmarks' job. Category recommendations rest on few rounds.

## Round profile

`profile-rounds SPEC [OUTPUT]` splits one drafting round into target verification, drafter
work and the rest, and splits drafter work by component, for each drafter, depth and prompt
length band. Re-run it on every new candidate. Schema `round-profile-v1` names a clean greedy
DEV `comparison` (runtime, server, libraries, target, workload, context), a `diagnostic_build`
(the `native-build` result of the same runtime with `round_profile_nvtx_v1` appended to its
`source_edits`; any other difference is refused), the Nsight Systems `profiler` binary, named
`drafters`, `depths`, `lengths` (bands of `minimum`–`maximum` input tokens with a task count)
and `output_cap` and `exact` as in calibration. The current inputs are in
`artifacts/benchmarks/round-profile-specs/` and the diagnostic runtime is
`artifacts/native/runtime-v6-round-profile`.

Each drafter and depth runs three server sessions under the GPU lease, on the same requests:

| Clock | Runtime | What it measures |
| --- | --- | --- |
| `clean` | Unmodified | Server counters: decode ms per derived round and the drafter's generate counter (`draft_ms`) |
| `phases` | Diagnostic, Nsight, CUDA graphs as served | Named host ranges per round and the GPU work launched inside each |
| `nodes` | Diagnostic, Nsight, CUDA graphs off, one range per graph node | GPU kernel time per graph node, grouped into components |

The diagnostic runtime marks drafting (`srv.draft`), the target's checkpoint copies, the
target verification batch, the feature handoff into the drafter's context cache
(`srv.process`), sampling and acceptance (`srv.verify`), and inside each graph build, input
upload, launch, synchronisation and the drafter's token selection. With
`BITDRAFT_NVTX_NODES` set, every graph node, and every node folded into a fused kernel,
carries its operation, output shape and source tensors. A round runs from one `srv.draft` to
the next with no prompt batch between them.

Components come from those names, not from a fixed model layout: weight tensor names
(`blk.N.attn_q`, `ffn_down`, `fc`, `output`, …) first, then operations (flash attention,
rope, cache writes, top-k), and the drafter phase they run in. The tables `WEIGHT_ROLES`,
`OP_ROLES` and `GROUPS` in [round_profile.py](../benchmarks/round_profile.py) own that
mapping; extend them for a new architecture. Each component reports GPU ms per round, weight
bytes read per round and multiply-adds per slot (one row of its input), from node shapes.

`round-profile.json` holds every clock under its own name, averaged per traced round, with a
`reconciliation`: the clean drafter counter against the traced draft wall, drafter kernel
time on the phases and nodes clocks, and the share of nodes-clock drafter time attributed to
graph nodes. `round-profile.txt` is the short table. Traces stay in the session folders;
`round-profile-report SPEC OUTPUT` rebuilds both files from them without the GPU.

Limits: Nsight's API tracing lengthens the phases round (measure the gap in each run, never
pool it with clean time). The nodes clock exists for kernel attribution; its walls are slower
because graphs are off. The server's `draft_ms` covers only drafting; the feature handoff
(`srv.process`) is drafter work outside that counter. Clean runs cannot split target
verification from the handoff and sampling.

## Live application and instructions

`prepare-project-workflows` freezes source/recipes and verifies broken, repaired and
followup controls in real project modules. `prepare-panels` creates live inputs and 30
fixed-history groups; `prepare-checks` selects 6 of those groups under `minimal`, `coding`, `tools`. Record instruction origins;
identify authored alternatives. Treatments replace system messages, retaining task facts.

`live` executes list/read/replace/write/test tools in isolated copies. Bubblewrap protects
checks and excludes network, credentials and host source. Reads support ranges; edits
require unique matches; truncation is explicit. Reset each task; count both initial and
followup phases. Limits: 16 turns/900 seconds per task, 1024 tokens per call. Retain warmups
outside primary totals and reconcile their counters. All failures/timeouts stay denominated.
Report completion, whole-task/model/tool/overhead time and greedy trajectory/filesystem
agreement, including shared failures. Faster incomplete work is not a win. Deployment
claims need trials in the intended application/configuration.

Live greedy agreement is checked, not guaranteed. When a drafter's accepted drafts end a
call with the end-of-turn token, that token stays in the slot cache. The next call then
reuses one more cached token and splits its new prompt into different batches. Prompt
batches over 32 tokens use fused arithmetic, so a near tie can flip. Under protocol v3,
S4 diverged this way on 1 of 12 Q4KM and 2 of 12 NVFP4 tasks; Baseline never differed.
A diverged arm makes its panel unusable for `record-reference`, so live controls are
established from target and Baseline alone, and a drafter that diverges has no live reference.

## Training and selection

[`training-campaign-v2`](../benchmarks/campaign.py) binds intent, candidate-only screen,
initial drafter, optimizer-time cadence, failure policy and `maximum_pause_seconds` ≤120.
`campaign-preflight` validates roles, model/data/source and saved controls on CPU. V1 is
legacy-pilot-only. V2 pilots accelerate cadence for tests while retaining all 60 prompts.

Boundary/final checks keep the same trainer and Adam/RNG state alive. Completed gradients
are released; trainable parameters and optimizer moments use a CPU mapping of that
worker's just-published checkpoint while frozen model buffers remain on GPU. File-backed
pages can be reclaimed instead of retaining another anonymous-memory model copy.
The candidate runs under the same owned GPU lease,
then trainable state returns before the next update. This avoids reconstructing the trainer.
GPU memory and host paging still require measured qualification for a new model/hardware.
A new acceptance leader
publishes a reusable reference; later checks compare with that record without rerunning
its model. The candidate must beat that leader by more than the paired source-group
interval in `historical_comparisons["candidate/prior-leader"]`; otherwise the earlier
state stays leader ([selection](domain.md#selection)). Invalid/partial evaluations never select.
Adam/RNG/cursor restore exactly; changed recipes/data require explicit continuation.
`training-boundary` and next `training-ready` journal events measure save/export, evaluation,
selection bookkeeping and restoration through next sample readiness.
The next sample must be materialized before readiness is
recorded; its cold data-loading cost counts in the pause. Capture caches and background
loads are released before native evaluation. A proposed leader stays pending until the full pause passes.
Boot/controller changes make
that interval unavailable. The campaign records interruption budgets and refuses continued
budget qualification after an overrun; final evaluation has no subsequent resume interval.
Initial startup is separate. Preserve all costs and incomplete attempts.
An optional authenticated `initial_reference` reuses the exact initial drafter's compatible
screen; it does not require another initial inference pass. Recovery retries a whole
incomplete candidate screen in a new output directory, retaining the failed attempt.
Normal repeated checks use one training worker. Journal capacity may rotate it into a
new supervised worker while preserving global budgets and durable recovery.
Recovery of a final check at the update guard reuses its completed evaluation and does
not launch another identical screen.
`initial_drafter` must match a completed producer bound to the durable parent, or a
zero-update export derived from that parent. An arbitrary same-name GGUF cannot seed a
campaign. V2's compatibility field `provisional_fast_leader` records an acceptance score;
it is not a throughput verdict. Pilot V1 campaigns measure every arm live and select on
tokens/s, requiring the lower bound of the paired candidate/prior-leader speed-ratio
interval to exceed one. Completed campaigns return idempotently without new runs.

For production, bind an authorized training intent, matching initial export, saved
60-request comparison and optional initial reference. Omit `pilot`, set cadence to 1800
optimizer seconds, `maximum_pause_seconds` to 120 and `evaluation_failure` to `stop`.
An intent's optional `model_update_seconds` sets a separate cumulative time allowance:
7200 requests two hours of optimizer work. `model-update-budget` is a complete campaign
stop with a final screen; the update-count guard still applies if reached first.
Run `campaign-preflight SPEC` before `campaign SPEC NEW_OUTPUT`; inspect with
`campaign-report OUTPUT`. Qualification recipes deliberately stop after a few updates;
copying their budgets does not create a production allocation. A new target precision
also needs matching teacher data and its own qualified controls.

Latest recovery, provisional fast leader and full-DEV incumbent are distinct.
[`suite`](../benchmarks/suite.py) prevalidates matching runtime/settings/role/target/arms.
`full-suite-v2` uses the lean one-pass panels and saved controls. `completed_panels`
reuses complete matching specifications, never partial blocks.
`candidate_bindings` references state/drafter/execution journals; completed producer or
boundary-publication receipts must bind the state to the evaluated export. Highest full-request score selects the experimental incumbent,
with earlier-update ties. Workflow/harness evidence stays separate; no deployment promotion.
Bindings may include a saved `prior-leader` as well as newly executed candidates. Each
must match that panel's authenticated model identity and retain its producer journal.
A ranking containing historical candidates is explicitly descriptive; close speed decisions
require fresh matched trials. Preserving a prior reference does not rerun its model.
The V2 suite's `targets` maps each precision to authenticated `requests`, `harness` and
`live` specifications, with optional `candidate_bindings` and `completed_panels`.
Use one greedy pass and the saved controls for every panel. Inspect individual outcomes
before interpreting an aggregate or recording an incumbent.

Diagnostics preserve RNG and report category loss/top1 on 120 fixed role-safe DEV windows,
plus first-position top-1/prefix survival ([execution](execution.md#sampling-and-clocks)).
They are not acceptance. Measure the complete training interruption against the short
wall budget; do not stretch the training interval to accommodate an oversized check.
Keep model/data/evaluation/checkpoint clocks separate.
`campaign-report OUTPUT` needs no GPU: liveness, pointers, journals, learning/native scores,
costs/exposure. Attempts include retries; unique positions use actual label widths;
lifetime history stays unknown. Failed native attempts and controller failures remain
visible after recovery; a retained `failure.json` alone does not describe current health.
Inspect status, actual ownership and terminal journals before restarting. After code
changes, read `OUTPUT/binding.json` and run from its code snapshot under
`OUTPUT.parent/executed-sources/IDENTITY`, using the configured numerical interpreter:

```sh
BINARY_NUMERICAL_PYTHON=/absolute/numerical/bin/python /absolute/numerical/bin/python -B -m benchmarks --frozen campaign /absolute/SPEC /absolute/OUTPUT
```

Preserve incomplete evaluation blocks; their failure policy governs continuation. Per-run
retention bounds states/exports; global retirement preserves recovery/selected states
with receipts after two later segments.

## Qualification and closeout

`native-build` uses pinned source/headers/compiler, without old objects; helpers rebuild
rendering/tokenization/capture/verifier tools. Source includes the measured sampled-EOG
repair. An optional `source_edits` list names edit sets in `benchmarks/runtime_edits.py`;
each edit must match the pinned source exactly once, and `runtime_repairs` records them. Trace builds are diagnostic, excluded from timing. `verify-greedy` independently
checks target argmax, prefixes, correction/bonus/EOG and counters. `verify-sampler` tests
both actual native algorithms on probabilities, zero support and EOG. CPU distributions
and neural GPU requests are separate evidence. `reduce-trace` only checks supplied structure.

`profile-native` owns/verifies/stops/reaps Nsight and checks expected/forbidden kernels and
layer counts. Integer/packing activation needs positive/negative controls; profiler time
is excluded. Follow [results/retention](results.md) and the pinned
[close-results procedure](../skills/close-results/SKILL.md). Extract and verify compact
measurements, figure inputs, failure denominators and executed-source identities before
retiring obsolete raw outputs. Update FINDINGS; record artifact disposition and lost
replay/resume. Reusable references keep the request/source records their authenticated
readers require. Current data/model inputs and recovery/selected states retain their
dependency closure. An ignored path or finished run does not establish disposal readiness.

[Measured qualification](../results/reusable-acceptance-checks/result.md) covers complete
training pauses, exact recovery, both target precisions and reused-reference comparisons.

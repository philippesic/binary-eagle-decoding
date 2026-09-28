# One-bit drafter research plan

Updated 2026-09-28 following the user's project audit and request to launch a
fresh team. This is the overarching plan; durable execution state remains in
[STATUS.md](STATUS.md) and the existing
[active goal](goals/recurrent-binary-body-head.md). Keep one active goal.

## Objective and present authorization

Establish whether jointly trained one-bit EAGLE body and output-head models can
beat Q4_0 EAGLE in native draft acceptance, draft latency, and complete-request
throughput. Compare W1A16, W1A8, W1A4 and W1A1 with honest representation and
kernel labels. FP16 EAGLE is diagnostic context; target-only is a required
end-to-end control. Keep the Qwen3-4B target/verifier precision frozen within
each experiment.

The user authorized the first EAGLE phase and a fresh team, with **no GPU
access**. Start CPU-only implementation, data preparation, analysis and tests
now. Do not use CUDA, Metal, MPS, remote GPUs, cloud accelerators, or remote
model inference. Both local GPU pause flags are set. GPU access must be
explicitly restored before the later experiment stage. CPU results cannot
establish CUDA correctness, speed or training convergence.

DFlash/DSpark are a later roadmap phase, not active work. Do not start a second
architecture, new model training from scratch, or a broad unrelated search.

## Handoff and existing assets

Continue from parent `d111335` (speed-first AGENTS rule), following predecessor
checkpoint `5110257`; native llama.cpp gitlink is
`21f617d4ef3f5dc383d3ab8dc619daaa87db7ff8`. The predecessor chat is
`01a0e7b2-e75d-7c00-8ad8-3101a73d8409`, **Continue joint binary EAGLE parity**.
The audit/planning chat is `01a0e9aa-3619-7012-9b4a-e4002a36901a`.

Reuse these assets instead of rebuilding or re-proving them:

- Native packed kernels and EAGLE conversion/loader support, matched Q4_0
  controls, operator tests, export audits and benchmark runners.
- Candidate D: one-bit signs and fitted nonnegative F32 group-128 scales,
  A16 inputs. Nine trainable matrices include fusion, Q/K/V/O, FFN gate/up/down,
  and the output head. Target, borrowed embedding, norms and vocabulary map
  remain frozen unless a later experiment explicitly changes that contract.
- CPU recurrent training interface, hard-sign surrogate, differentiable state
  and cache tests, learned-scale exporter, and native CPU diagnostic oracles.
  The present joint trainer is a CPU reference, not a validated fast GPU trainer.
- Audited native capture from 96 training prompts: 40,815 verifier rows,
  39,705 supported labels, 15,042 retained feature rows, 12,251 emitted tokens.
  The 2.72% unsupported-label rate remains visible in quality denominators.
- Existing ignored raw artifacts and manifests, especially
  `results/binary-rescue-head-5080/round-analysis.json` and the artifact indices
  linked from the capture and performance reports. Read local data in place;
  temporary worktrees do not automatically contain ignored datasets or models.

The old capture remains a small smoke/regression dataset. Its legacy
`training_eligible: false` metadata is not silently changed by this plan;
replace the former gate with a documented practical readiness assessment.
The old 24 development prompts are heavily reused. Preserve the old 24 final
prompts without inspecting their contents or using them for selection.

The predecessor reported its last supervised GPU job stopped, process group
417 absent, and tmux session `$33` closed. This is historical handoff evidence,
not a current GPU availability claim. No remote job ownership transfers here.

## Research policy after the audit

Favor useful experiments over numerical archaeology. The old next action to
trace target block-14 FFN operations is superseded. Independent Hugging Face
target arithmetic need not reproduce every native floating-point value when
the teacher inputs and labels come from captured native execution. Do not
resume that investigation without a concrete effect on the active experiment.

Keep token/feature ancestry, masks, positions, vocabulary mapping, frozen
parameter ownership and verifier semantics correct. Declare practical numerical
checks at the trainable drafter/export boundary, use representative cases and
native proposal/trajectory evaluation, and record discrepancies. A changing
reduction order is not an automatic research blocker. The CPU native oracle is
a diagnostic, not a required production training implementation.

Efficient QAT may use dense floating-point matrix multiplication on quantized
values. The forward must apply the chosen hard weight/activation quantizer;
full-precision latent parameters must not bypass it. This is training
simulation, not evidence of native binary acceleration. Adopt a documented
surrogate gradient rather than trying to differentiate exact CPU instruction
sequences.

Timebox individual diagnostics to the concrete question they can resolve.
Record a stop condition before starting another probe. Favor one integrated
experiment over a growing chain of prerequisite micro-experiments. Keep routine
engineering autonomous; report major representation/objective choices with
evidence in DECISIONS.md instead of repeatedly asking about implementation details.

## Phase 1A — EAGLE preparation and implementation, CPU only (active)

Workstreams can proceed in parallel with explicit file ownership.

### A. Account for latency and remove evidenced waste

Implement a reproducible drafter timing report with separate scopes:

1. Individual quantized projection: activation packing/scales, dot, output.
2. One decoder step: projections, attention, norms, residual/activation work,
   vocabulary selection, state transfers and synchronization.
3. Whole `draft()` call: seed plus recurrent steps, actual proposals computed
   and returned, amortized time per proposed token.
4. `process()` feature fusion/cache catch-up, outside `draft()`.
5. Full request, prefill/decode, accepted/proposed/emitted tokens and target work.

Use local archived traces to establish what can already be attributed. Add
stage tags and offline analysis needed for a later fixed-input, fixed-context
Q4_0/W1Ax A/B run. Preserve unassigned time. CPU waits may contain GPU work;
do not add overlapping spans or subtract isolated operator medians from an
unrelated whole-model benchmark. CUDA graphs were already enabled in the
historical server profile; missing per-node attribution is a measurement gap.

Prioritize source-supported changes, keeping one native runtime writer:

- Share packed activations/scales across Q/K/V and FFN gate/up; consider joint
  projection calls. This can eliminate three repeated packs per decoder step.
- Avoid expanding the 32,000-entry draft vocabulary to 151,936 entries just
  to select candidates. Preserve mapped IDs, ties, probabilities, confidence
  thresholds and fallback behavior.
- Audit the repeated GPU-to-host-to-GPU recurrent-state path. Prepare a bounded
  device-state change if the later profile can test its value; do not begin an
  unbounded context API rewrite.
- Evaluate binary-kernel reduction/launch cleanup where source evidence is
  clear. CPU builds/tests establish contracts only; CUDA performance stays open.
- Prune unused head outputs in eligible cache-update calls; investigate K/V-only
  catch-up separately with its single-layer/no-consumed-output preconditions.
  These changes affect `process()`, not the previously reported draft-only time.

Implement bounded changes behind opt-in paths where useful. Record per-change
validation and a prepared A/B command. Apply shared runtime improvements to Q4_0
and FP16 too. Do not claim any speedup from source review or CPU timing.

### B. Finish reusable joint QAT machinery

Build on `src/w1a1_eagle/recurrent_*` and the existing native-step adapter.
Provide an efficient device-configurable training path with small CPU fixtures
and an explicit GPU execution guard. Keep all eight body linears and the head
jointly trainable; verify later-position loss reaches earlier states/cache.

Make activation precision configurable: A16 boundary casts, A8/A4 per-token
quantization/scales matching deployment, and A1 signs/scales. Specify weight
and activation surrogate gradients, clipping, scale handling, optimizer,
learning rates, batch/sequence schedule, seed and checkpoint selection.
Monitor actual sign flips, gradient coverage, scale movement and saturation:
nonzero gradients alone do not prove effective binary-weight learning.

Resolve the representation compatibility explicitly: native group-128 scales
currently support A16 only. Existing row-scale W1Ax kernels support all four
activation widths. Prepare a comparison of (a) a common row-scale four-format
sweep with D-group/A16 as a separate reference, and (b) extending group scales
to lower activation widths. Preserve D; do not silently equate differing scale
representations or make unsupported exports. Implement shared interfaces while
the bounded representation choice is documented.

Provide hard target-label CE as the current reference objective and a clearly
specified target-probability distillation option. A small predeclared objective
comparison may be useful later; avoid a Cartesian hyperparameter search. The
earlier failed head-only KL fit used different teacher/state assumptions and
does not establish that target distillation fails.

The old 500-step/45-minute proposal becomes a smoke budget, not evidence of
adequate full-body training. Prepare token/example-based budgets and a learning
curve. A future 100-step GPU calibration must measure throughput and memory
before estimating completion time or freezing a larger run budget.

### C. Expand and assess data

Prepare a reproducible ingestion, filtering, split and manifest pipeline now.
Use diverse realistic prose, code and reasoning sources, with source revisions,
licenses/provenance, deduplication and topic/conversation grouping. Public data
inspection/download and CPU manifest work are allowed; target inference waits
for GPU access. Keep datasets and model artifacts outside Git.

Initial planning scales are roughly 2,000 training prompts, then up to 10,000
if the learning curve warrants it, targeting hundreds of thousands to millions
of useful target tokens. These are configurable study sizes, not a claim that
any fixed prompt count is sufficient. Track unique prompts/conversations,
lengths, domains and token counts separately from repeated training presentations
and overlapping unrolls. Prepare a substantially larger independent development
set and a sealed final set (e.g. 192 prompts each, balanced by domain), while
retaining the legacy final set unopened. Freeze the new manifests before use.

Use the frozen target as teacher. Native captured features/logits are reusable
on their exact prefixes. Saved D-prefix labels cannot be relabeled as current
student-prefix labels after tokens change. Support teacher-forced offline
training and a later bounded trajectory refresh strategy; evaluate actual
native rollout acceptance to expose distribution mismatch.

Avoid scaling the old full-vocabulary F32-logit dump blindly: 96 prompts already
produced 24.8 GB of logits. Prepare compact teacher probabilities with an
explicit top-k/tail-mass/normalization contract, or streamed/on-demand teaching.
Top-k distillation is an approximation and must be labeled accordingly. Keep
unsupported labels and mass outside the draft vocabulary visible.

Data quality checks must cover source diversity, exact/near duplication,
split leakage, masks/EOS/truncation, label alignment and vocabulary coverage.
Reusing a frozen target makes target-matching labels appropriate; it does not
prove broad data coverage. Classify the archived same-prefix feature variation
from existing metadata if helpful, but do not revive backend parity archaeology.

### Phase 1A completion gate

- Latency accounting tools and a finite later GPU profile/A/B protocol exist;
  missing attribution is explicit.
- Reviewed CPU-testable runtime changes are implemented or have concrete,
  bounded patches ready for CUDA validation; no speedup claim is made.
- One integrated joint-training entry point handles declared W1Ax contracts,
  checkpoint/export boundaries and no-GPU safeguards; tiny CPU training tests pass.
- Larger-data source/split manifests or reproducible preparers are available,
  including coverage, deduplication and compact-teacher storage decisions.
- Representation and objective options are documented, with actionable defaults
  and unresolved major choices separated from ordinary implementation work.
- The next GPU experiment can start from pinned commands once access and its
  measured budget are approved. Checkpoint and stop at that boundary rather
  than manufacturing more parity tasks.

## Phase 1B — EAGLE measurements and training (GPU access required)

After explicit availability is restored, use one supervised GPU owner and the
host registry/tmux procedures in AGENT_OPERATIONS.md.

1. Validate CUDA deployment math and obtain a warmed fixed-input drafter profile
   for Q4_0 and W1Ax. Measure each optimization A/B and retain useful changes.
   Report absolute draft ms/token and full draft calls; separate accepted tokens
   and complete-system throughput. Use a fair common-optimization baseline.
2. Capture the first larger-data tier; run the 100-step trainer calibration and
   small native checkpoint-zero/export check. Freeze the actual compute budget,
   precision/scale contract, objective and selection rule from these measurements.
3. Train independent W1A16/A8/A4/A1 checkpoints using shared infrastructure and
   comparable declared data/token budgets. Evaluate checkpoints by native
   acceptance, including per-depth and per-domain results, not training loss alone.
4. Expand data/refresh trajectories only when a predeclared learning-curve gate
   supports it. A small undertrained negative result rejects that recipe/budget,
   not the possibility of binary EAGLE.
5. Benchmark finalists against Q4_0, FP16 and target-only with matched settings,
   context lengths and hardware. Lead with Q4_0. Keep drafter-only latency and
   full-request metrics distinct. Recheck selected candidates on RTX 2080 Ti
   for SM75 claims; 5080 numbers are not transferable performance evidence.
6. Freeze candidate/policy before opening final evaluation. Record a useful
   negative result if practical gains are absent; do not continue indefinite
   kernel or QAT rescue cycles.

## Phase 2 — DFlash/DSpark transfer (future user decision)

Start only after the EAGLE decision point and an explicit architecture choice.
First verify a compatible pretrained Qwen3-4B checkpoint and measure its normal
native precision baseline. Our pinned runtime has DFlash/DSpark inference paths;
this does not mean their binary export or QAT support is already implemented.

Reuse quantizers, surrogate-gradient modules, optimizer/checkpoint infrastructure,
data split tools, teacher storage and benchmark reporting. Replace EAGLE-specific
rollouts with the released architecture's block masks, conditioning and losses.
Inspect required target feature taps before reusing any captured features.

DFlash predicts a block in one backbone pass; DSpark adds a small sequential
head. Their larger token batches might use kernels better, but neither has a
local binary quality/speed result. Identify shared versus private output heads:
never quantize the target head in place for a drafter; use a separate draft copy
or explicitly retain a wider head. Keep this compatibility layer small rather
than building a general framework before the first EAGLE result.

## Fresh team and ownership

Launch one fresh project chat as the Sol-high coordinator for the existing active
goal. It owns STATUS.md, the active goal file, decisions, integration and user
updates. It should immediately assign bounded parallel work:

| Role | Initial ownership |
| --- | --- |
| Sol feature owner — runtime | Native EAGLE graph/kernel changes and latency instrumentation; sole native-submodule writer |
| Sol feature owner — QAT | Recurrent quantizer/training/export Python modules and training runner |
| Sol feature owner — data | Dataset ingestion/splits/manifests and compact-teacher preparation; no reserved-final contents |
| Luna high — validation | Archived-log accounting and focused CPU tests; coordinate test-file ownership with feature owners |
| Astra medium — focused advisor | One bounded review of scale-contract compatibility and trainability risks; no open-ended parity investigation |

Use temporary worktrees and explicit file partitions. Workers are not alone and
must preserve others' changes. Integrate reviewed, tested work into main and
push coherent checkpoints; publish native commits before parent gitlinks. Never
archive a worktree still needed by an active worker. Record child IDs, owned
files/worktrees and next actions in the existing goal file before handoff.

The previous agent's artifacts are inputs, not active assignments. Do not message
or reactivate old workers. Once CPU deliverables are complete, checkpoint the
GPU-only boundary and wait for the user's availability update.

## Planning estimates and evidence

Working estimate from the audit: roughly 3–7 focused development days for the
first shared setup, followed by several GPU hours to a few GPU days for data,
four initial training runs and evaluation. This is a planning range, not measured
all-body training throughput, an authorized compute cap, or a promised result.
A first alternative architecture might add 3–7 development days after reusable
EAGLE infrastructure exists; larger structural changes can exceed that range.
Replace estimates with measured step, capture and benchmark costs as soon as
GPU access returns.

Relevant local evidence:

- [Existing recurrent protocol](../experiments/recurrent-binary-qat-plan.md)
- [Full native training capture](../experiments/recurrent-binary-full-capture-5080.md)
- [5080 rescue and full-round timing](../experiments/binary-rescue-head-5080.md)
- [SM75 W1Ax study](../experiments/w1ax-activation-precision-results.md)
- [Pipeline source audit](../experiments/research-review-2026-09-25/04-pipeline.md)
- [Architecture source audit](../experiments/research-review-2026-09-25/02-architectures.md)

Primary research context: [Gemma 3 technical report](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf)
describes about 5,000 QAT steps and probability distillation, but does not
disclose the QAT dataset count; its pretraining trillions are not QAT tokens.
[EAGLE-3](https://arxiv.org/abs/2503.01840) motivates recurrent training and data
scaling. [DFlash](https://arxiv.org/abs/2602.06036) and
[DSpark](https://arxiv.org/abs/2607.05147) motivate a later block-drafting screen,
not an assumed local one-bit speedup.

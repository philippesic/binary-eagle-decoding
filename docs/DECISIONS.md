# Decision log

## Selected: A8 integration and controlled QAT comparison — October 3, 2026

The human selected the proposed QAT repair team and monitored comparison,
explicitly deferring A1. The active [A8 goal](goals/a8-qat-recovery-and-comparison.md)
defines a two-hour-per-arm initial budget, shared validated execution
optimizations, and a candidate with learned A8 quantizers, all-nine affine
midpoints and the existing lower-inertia AdamW recipe. Direct-bit Bop and
architecture/refresh/curriculum changes are deferred. This closes the prior
pending fork for this bounded experiment; it does not select a quality winner.
Actual effective configuration, parameter movement, step-zero controls and
native checkpoint evaluation are required. The recovering monitor may repair
execution defects and resume exact state within the recorded budget; it may
not silently drop features or change the scientific comparison. Old paired
step1000 and raw failures remain historical.

## Active user direction: EAGLE W1 preparation (2026-09-28)

The user requested the [overarching plan](W1_RESEARCH_PLAN.md) and a fresh team
to start its EAGLE phase from the previous checkpoint, with **no GPU access**.
This continues the existing body/head goal and authorizes CPU-only runtime
implementation, latency accounting, reusable W1Ax QAT preparation and expanded
data tooling. Both host pause flags are set. DFlash/DSpark remain future work.

The user explicitly favors development speed over bit-exact cross-backend
accuracy; this is recorded in AGENTS.md at `d111335`. Replace the old target
block-14 investigation as a prerequisite with practical train/export numerical
checks and later native acceptance measurements. Preserve masks, token ancestry,
mapping and verifier correctness. Use native teacher data without requiring a
second backend to reproduce every target value.

The 96 training prompts are smoke/regression data, not a sufficient serious
all-body QAT corpus. Prepare larger, diverse train/dev/final splits and compact
teacher storage. Proposed sizes and estimates in the plan are configurable
planning targets, not verified sufficiency or approved GPU execution budgets.
Finalize the row/group scale comparison, objective and actual compute budget
before GPU experiments; make independent CPU implementation progress meanwhile.

The audit also distinguished one-bit W1A1 popcount from the current slow
W1A16 sign-add implementation. Recorded amortized draft-only latency is not
an isolated neural-forward time or a hardware lower bound. Runtime opportunities
(shared packing, compact-vocabulary selection, device-state recurrence and
eligible cache-update pruning) are source-supported but have no measured A/B
gain yet; changes shared with Q4_0 must benefit that comparison path too.

## Phase 1A implementation defaults and pending research choices

**Phase 1B first-shard readiness choice (2026-09-29):** the frozen
31-prompt RTX 5080 shard has 12,610 audited verifier rows and 7,796 selected
native target-feature rows. Its independent bundle audit is byte-identical to
the builder audit; compact-teacher, internal-continuity and 31/31 full-request
emission audits pass. The versioned bundle still declares
`training_eligible:false`, with model execution identity, numeric feature
parity, full drafter mask/position/KV parity and cross-round ancestry
unverified. The provider correctly rejects it before model loading. A
model-independent row-A4 CUDA fixture completed 100 finite steps, but does not
measure real-model rate or memory. See the [capture report](../experiments/w1ax-shard0000-capture-5080.md).

Recommended decision: permit a **bounded 100-step calibration on this shard**
only after a versioned readiness record verifies pinned native model/server
execution identity, closes initial/terminal response ancestry as needed for
training rows, and passes a frozen small-sample student checkpoint-zero
operand/trajectory gate (exact prefixes, masks, mapping, quantized signs and
finite logits; count proposal disagreements and judge near ties by native
acceptance). Use captured native target features/logits as teacher, retain the
original preparation manifest and raw logits, and label the resulting
eligibility specifically for this calibration. This follows the project's
practical numeric policy: independent Hugging Face versus native F16 drift
alone does not block when it does not change labels, student decisions,
gradients or conclusions. A strict alternative requires broad backend
feature/FFN numeric parity first, delaying calibration without an identified
training failure. Neither option authorizes the full 2k capture or a final
quality claim. The user owns this readiness decision; no flag has been
promoted and no captured-data QAT has run.

**New checkpoint-zero evidence:** The pinned dense drafter produced identical
untrained row-scale checkpoint bytes for A4 and A16. Both GGUFs passed native
load/CUDA graph checks and matched Q4_0 raw outputs on all 24 old development
prompts, but accepted only 131/2,917 (A4) and 323/2,725 (A16)
drafts/rounds versus Q4_0's 1,555/1,493. See the
[native checkpoint-zero report](../experiments/w1ax-checkpoint-zero-native-5080.md).
The stronger A16 result supports using A16 for the first real-model 100-step
**rate and memory calibration** if the practical readiness gate is selected;
it does not justify substantive training or final quality evaluation by
itself. The small accepted/round values also support an alternative research
fork: revise the initial row representation or scale fitting before allocating
the full four-width QAT budget. Candidate D's fitted group scales are not an
identical-initialization control. This choice remains user-owned and does not
change the capture's ineligible metadata.

**User choice, 2026-09-29 18:17 UTC:** proceed with the short practical pilot.
Run the focused native ancestry/state/student-trajectory checks first. If the
predeclared gate passes, publish a separately versioned eligibility record for
only shard-0000, retain the original preparation bundle and raw logits, and
run one supervised 100-step row-A16 hard-CE calibration on RTX 5080. Measure
real-model step time, peak memory, finite loss/gradients and checkpoint
integrity. A failure in the focused gate stops the real-data run and is
reported without relaxing it. This choice does not authorize a full 2k data
capture, a four-width training sweep, final-set use or a quality/speed claim.

The CPU implementation uses a **row-scale W1Ax interface** for a comparable
A16/A8/A4/A1 sweep; candidate D's group-128/A16 remains a separate reference.
Collapsing D group scales to one row scale would be a new, lossy initialization,
not the same D weights. Native row-scale kernels cover all four activation
formats, but the current learned-checkpoint metadata loader gate rejects
row-scale formats below A16. The bounded loader fix and CPU acceptance/rejection
matrix are implementation work, not proof of CUDA performance. Lower-width
exports stay explicitly unsupported until that gate and metadata tests pass.

The first training objective is hard target-label CE on valid exact-prefix
native capture rows, including reached teacher-forced rows after a live
rejection as already clarified below. Compact target-probability distillation
is a declared secondary option with top-k mass, mapped-tail mass and
outside-draft mass retained separately; it is an approximation, not a new
full-vocabulary label. The earlier failed head-only KL fit had a different
state/teacher contract and does not decide this comparison. The first later
GPU calibration, not the inherited 500-step smoke budget, will establish
training rate and a token/example budget.

**Pending user-owned representation choice:** compare (a) independent row-scale
W1Ax checkpoints with D-group/A16 as a separately labeled reference, or (b)
extend group-128 native scale support to A8/A4/A1. The actionable Phase 1A
default is (a), because all four row kernels already exist and (b) requires
additional native arithmetic and cost validation. No acceptance or speed
conclusion is implied. Revisit after fixed-shape GPU profiling and checkpoint
zero native export checks.

**Provisional practical numeric gate for Phase 1B:** preselect a small non-final
train/dev prefix sample and compare exported student operands and bounded native
proposal trajectories, including exact ancestry, masks, mapping, hard quantized
codes, finite values, logit error and proposal disagreements. Margin-separated
top-1 changes, cache/mapping differences or systematic proposal changes block;
near-tie differences are counted and resolved by native acceptance. The sample
and tolerances must be frozen before the GPU run. This replaces open-ended
independent HF target arithmetic parity as a training prerequisite; actual
native acceptance versus Q4_0 is the decisive quality test.

**Pending full-tier teacher-storage choice:** the CPU [shard plan](../experiments/w1ax-sharded-capture-plan.md)
splits the candidate 2k training prompts into 65 bounded captures. The old
96-prompt raw full-vocabulary logit file was 24.8 GB; the current conservative
2k plan estimates 633.26 GB of raw logits, and v1 bundles duplicate that raw
file before re-audit. The first small capture/calibration can retain raw bytes
within a 12-GiB-per-shard cap. Before capturing the full tier, choose an
auditable label-only bundle for hard CE, a compact top-k/tail bundle with a new
re-audit contract, or explicitly provision raw storage. The actionable default
is to design a re-auditable compact/label-only path rather than silently
accumulating more than 1 TB of raw copies. Do not delete v1 raw files: their
current audit depends on them. Capture commands remain paused until GPU access
is explicitly restored; the storage representation is a user-owned fork before
the full 2k capture, not a reason to delay CPU interfaces or the first bounded
GPU calibration.

Record research or infrastructure forks when evidence could lead to different
next steps. Keep the current decision and the reason; revisit when new data
changes the tradeoff. Routine implementation choices belong in commits.

| Date | Decision | Basis | Revisit trigger |
| --- | --- | --- | --- |
| 2026-09-23 | Use Qwen3-4B + AngelSlim EAGLE-3 on a pinned llama.cpp submodule as the starting pair. | Pinned upstream documents this conversion and runtime path. | Baseline incompatibility or memory limit. |
| 2026-09-23 | Use RTX 5080 for available experiment work; reserve RTX 2080 Ti for native SM75 binary timing. | User hardware access and Turing hypothesis. | Hardware access changes. |
| 2026-09-23 | Defer target-only and FP16 EAGLE measurements until direct comparison with native W1A1. | The published EAGLE baseline is the functional starting point; the user wants matched measurements alongside binary execution rather than a separate baseline milestone. | A setup issue blocks W1A1 development, or paired comparisons require an earlier diagnostic. |
| 2026-09-23 | Start the PyTorch W1A1 drafter from the official AngelSlim EAGLE-3 inference code pinned at `0358da9c651e6a7d7ccafea26ced4b9c98d11681`, wrapping selected linear modules. | The checkpoint has no modeling Python file; official code already handles Qwen3 head dimensions, recurrent state, caches, and offset-form `d2t`. A fresh adapter would duplicate those semantics. | Pinned code cannot run with a reproducible environment, or parity checks reveal a defect. |
| 2026-09-23 | Provisionally use Transformers 4.57.6 for the pinned AngelSlim PyTorch runtime. | Local import and meta-device target/drafter construction work with 4.57.6; Transformers 5.6/5.17 lack the `default` RoPE initializer expected by AngelSlim's Qwen3 target, despite AngelSlim's package metadata requiring 5.6+. | Full-checkpoint forward fails, or an upstream-compatible fix makes a supported 5.x version preferable. |
| 2026-09-23 | Superseded preliminary fork: selective W1A1 layers or bounded QAT for broad coverage. | The Metal development run measured 1.683 accepted drafts/round for head-only and 0.202 for all listed groups; no ordinary Metal acceptance baseline was recorded. | Superseded by the full RTX 5080 exploratory evidence below. |
| 2026-09-24 | Under the user's delegated research choices, investigate BF16 parity and run a bounded head-only QAT pilot while building the native packed-head path. | On the 5080 held-out suite, ordinary EAGLE accepted 2.317 drafts/round, head-only W1A1 1.677, FFN-only 1.067, and all groups 0.202. Candidate linears occupied 38.32% of one-prompt instrumented draft time; the head occupied 12.93%. Head-only is the narrowest useful coverage test. | Held-out QAT fails to recover acceptance, packed execution loses after packing, or native output differs materially. |
| 2026-09-24 | Treat the BF16 target/EAGLE mismatch as a measured tree-verifier tie sensitivity; preserve verifier-relative acceptance labels and do not assert strict target equivalence. | At target position 40, the BF16 EAGLE tree row ties IDs 11/272/7578 at 21.0 and chooses 11; incremental and full-prefix target evaluation choose 272. Replacing every off-path sibling leaves the selected logits and hidden state bitwise unchanged. The raw 5080 trace and hashes are in the active goal file. | A broader prompt sweep reveals a distinct cache/mask bug, or a paired native run requires exact generated-stream equivalence. |
| 2026-09-24 | Pending native low-bit comparison choice: W8A8 is the stronger post-training acceptance candidate under the current all-linear scheme; W4A4 needs quality recovery or narrower coverage before a speed claim. | Matched RTX 5080 accepted drafts/round: ordinary BF16 2.3166, W4A4 0.2882 (12.4% of ordinary), W8A8 2.1816 (94.2%). This is FP32 numerical simulation of symmetric signed W4A4/W8A8, not hardware INT execution; the BF16 verifier parity issue remains. | When choosing native kernels, match the quantizer exactly and run same-device end-to-end timing. Weight-only INT4/INT8 requires separate acceptance evidence. |
| 2026-09-24 | Run one bounded head-only W1A1 QAT pilot after the current parity diagnostic; defer all-group training. | Head-only post-training W1A1 retained 1.677 of ordinary 2.317 accepted drafts/round and is isolated enough to train from captured pre-head vectors after unloading the target. All-group W1A1 fell to 0.202/round. A 500-step/45-minute head reconstruction pilot is feasible within 5080 memory and has a clear export/held-out gate. | If exact forward/export checks fail, validation stalls, or held-out acceptance does not recover, stop the recipe and record the negative result before considering FFN/broader QAT. |
| 2026-09-24 | Use portable packed XOR/`__popc` as the first native operation; reserve a bounded SM75 binary-MMA probe before concluding that binary Tensor Cores cannot help. | The 5080 prototype passed packed correctness and component timing but does not use binary MMA. The project gate allows an SM75 XOR/POPCOUNT kernel, while the hardware motivation specifically names Turing binary Tensor Cores. SM75 `mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc` is a separate implementation candidate. | Obtain the RTX 2080 Ti address and compare packing-inclusive portable versus MMA timing on it, especially the one-token head shape. |
| 2026-09-24 | Stop the bounded BF16 head-reconstruction QAT recipe after one held-out check; continue native correctness and paired timing with the original packed head. | Validation KL improved from 2.506 to 1.017, but the exported trained W1A1 head accepted only 1.565 drafts/round on the fixed 12-prompt 5080 suite versus 1.677 for the prior untrained head. The trained head used at full precision also reduced ordinary acceptance to 1.813 versus the original 2.317. The target greedy stream matched on only 6/12 and 4/12 prompts respectively. These are verifier-relative counts; no speed conclusion follows. | A separately justified training objective and new untouched evaluation set could motivate another pilot, but do not tune this recipe on these held-out prompts. |
| 2026-09-24 | Record a negative head-only native W1A1 result versus ordinary EAGLE on the RTX 5080; do not optimize this head path further before the remaining correctness/SM75 gates. | Five matched repetitions over 12 prompts: packed W1A1 reached 115.38 request / 122.94 decode tokens/s versus ordinary EAGLE 123.96 / 132.88, or 0.931× / 0.925×. Draft time/round fell 4.386→3.515 ms, but accepted drafts/round fell 1.161→0.892, requiring 480 extra rounds. Both speculative variants exceeded target-only throughput. All packed runs logged actual CUDA dispatch. | A new coverage/quality recipe with a justified fresh evaluation set, or real SM75 binary-MMA timing that changes the cost side. Do not infer SM75 behavior from 5080. |
| 2026-09-24 | Treat native target-only speed ratios as timing observations until target equivalence is proven; retain the packed-versus-ordinary negative result. | An opt-in two-prompt raw-logit trace reproduced the sealed ordinary outputs. At both emitted divergence rows, the target verifier ranked the speculative token first by a narrow margin and rejected the draft. The immediate difference is between target-verifier and target-only argmax, not an incorrectly accepted draft; the exact numerical/cache cause is unresolved. Ordinary and packed decoded text matched on all 60 paired requests. | A controlled target-logit parity experiment identifies and resolves the cause, or a new runtime changes the target/verifier execution path. |
| 2026-09-24 | Keep binary MMA as an opt-in experimental llama.cpp branch pending actual SM75 execution; compare it with portable W1A1 on the 2080 Ti before any Turing Tensor Core conclusion. | The integrated branch passed 8/8 backend cases for both dispatches on RTX 5080, matched 85/85 production portable tokens in one packed-draft request, and emitted `BMMA.88128.XOR.POPC` in compile-only SM75 SASS. This is proxy correctness and instruction evidence, not Turing runtime or latency. | User supplies current 2080 Ti address, then real SM75 correctness and same-device portable/MMA end-to-end measurements. |
| 2026-09-24 | Expand the 2080 Ti comparison to all five previously simulated W1A1 coverage settings and revisit QAT with a new held-out gate. | The user explicitly requested fusion-only, attention-only, FFN-only, head-only, and all-group tests on Turing after reviewing the 5080 results. The only native integrated setting so far is head-only. The prior 500-step head teacher-KL QAT pilot lowered held-out acceptance despite improved validation KL. | Native group conversion/runtime correctness and SM75 access; a new QAT recipe must be judged on untouched held-out prompts rather than the original 12. |
| 2026-09-24 | Benchmark Turing's FP16, binary, INT8, and INT4 paths with execution-accurate labels; implement true W8A8 before true W4A4 while measuring the already prepared Q8_0/Q4_0 controls separately. | NVIDIA's Turing guide lists all four hardware operand widths. The static pinned-GGML audit predicts one-token MMVQ: Q8_0 with Q8_1 activation uses byte `DP4A`, while Q4_0 with Q8_1 activation extracts nibbles for byte `DP4A`. Those block-scale paths are distinct from the earlier whole-row/whole-token W8A8/W4A4 simulations. W8A8 previously retained 2.182 accepted drafts/round versus ordinary 2.317; W4A4 retained only 0.288. Actual SM75 dispatch and genuine new-format correctness are still pending. | A measured Q4_0/Q8_0 dispatch changes the labels, or W8A8/W4A4 implementation checks expose an incompatible contract. |
| 2026-09-24 | Keep the FP16 target and 2,048-context comparison track on the RTX 2080 Ti. | A fixed-binary short four-path smoke loaded target-only, ordinary EAGLE, portable head W1A1 and MMA head W1A1. Peak observed loaded use was 10,110 MiB of 11,264 MiB, with 1,154 MiB minimum free; all 20 short requests completed, and the GPU returned idle. | A longer five-W1A1/Q4/Q8 run exceeds the memory budget or shows unstable allocation; if so, define a separately matched quantized-target track. |
| 2026-09-25 | Record a negative portable W1A1 result versus ordinary EAGLE for every tested coverage setting on the RTX 2080 Ti; preserve the separate binary-MMA and true W8A8/W4A4 gates. | The sealed 540-request matrix gave ordinary 82.49 decode tok/s; fusion 49.32, attention 49.17, FFN 59.05, head 74.81, and all-group 46.31. Their accepted drafts/round fell from ordinary 1.168 to 0.271, 0.238, 0.472, 0.860, and 0.055. Paired head/ordinary decode ratio was 0.907 (95% interval 0.869–0.946); all-group was 0.561 (0.511–0.617). All eight speculative variants emitted identical text on all paired requests. | Real SM75 integrated binary-MMA end-to-end timing or a new acceptance-improving quantization/training recipe changes the cost-quality tradeoff. |
| 2026-09-25 | Treat the integrated SM75 binary-MMA head path as performance-tied with portable W1A1 for this workload; continue true W8A8/W4A4 gates. | In a separate 240-request paired run, MMA/portable request and decode ratios were 1.0019 and 1.0017, with 95% intervals 0.9966–1.0068 and 0.9973–1.0062. Both paths had identical acceptance counts and completion text on all 60 pairs, and distinct CUDA dispatch markers. | A different batch/shape or kernel tuning changes packing-inclusive cost; do not generalize the tie beyond this EAGLE head workload. |
| 2026-09-25 | Treat default native W8A8 as tied with ordinary EAGLE and default native W4A4 as a measured negative result on the 2080 Ti; test opt-in INT8/INT4 Tensor Core candidates separately. | The sealed 300-request all-nine-linear vector run gave ordinary 80.99 decode tok/s, W8A8 80.18 (0.990×; paired 95% interval 0.975–1.005), and W4A4 41.40 (0.511×; 0.466–0.561). Accepted drafts/round were 1.168, 1.127, and 0.092. Both low-bit CUDA dispatches were confirmed and all speculative outputs matched text across 60 pairs. | A live Tensor Core implementation changes the cost side, or a separately trained quantizer materially recovers W4A4 acceptance on untouched prompts. |
| 2026-09-25 | Record opt-in live SM75 INT8 and INT4 Tensor Core paths as slower than their respective default CUDA paths for this single-sequence EAGLE workload. | A full 420-request paired run gave W8A8 MMA/default decode ratio 0.8205 (95% interval 0.8172–0.8236) and W4A4 MMA/default 0.8801 (0.8759–0.8844). Each pair used the same GGUF, matched text on all 60 paired requests, and had identical accepted/proposed/round totals; distinct MMA dispatch and no default marker were logged. | A different batch shape, fused packing/layout, or materially revised kernel changes the cost side; do not infer a general Tensor Core disadvantage. |
| 2026-09-25 | Use both ordinary FP16 EAGLE and Q4_0 EAGLE drafts as throughput comparison anchors for future W1A1 work, with the same target and verifier. Refocus next on QAT acceptance recovery; evaluate alternative drafter architectures after that. | On the 2080 Ti, ordinary EAGLE decoded at 82.49 tok/s, Q4_0 at 91.51, and all-nine W1A1 at 46.31. W1A1 shortened the measured draft call but accepted only 0.055 drafts/round versus Q4_0's 1.182. The first head-only teacher-KL QAT pilot also lowered held-out acceptance, so a new target-aligned objective and untouched final set are needed. Q4_0 names a block-scaled draft weight format, not the separate W4A4 quantizer. | A new QAT recipe improves held-out acceptance and native end-to-end rate, or a measured non-EAGLE drafter offers a better quality/cost tradeoff. |
| 2026-09-24 | Keep the target-aligned QAT treatment of out-of-draft-vocabulary labels pending until a mapped-mass audit. | Pinned AngelSlim uses offset-form `d2t`: draft ID `i` maps to target ID `i + d2t[i]`. A direct checkpoint audit verified 32,000 unique absolute IDs and a matching 32,000-true `t2d` mask over 151,936 target IDs; the GGUF mapping matched exactly. The current cached-head capture still lacks parent tree-node identity and matching verifier logits. A 32,000-class head cannot directly represent every target-vocabulary label. | Capture aligned verifier rows and measure unmapped-label frequency and target probability mass, then freeze a documented loss rule before any new QAT run. |
| 2026-09-23 | Rotate a long Codex task after its second compaction, using a written checkpoint. | Context drift in previous long runs; hooks can inject post-compact instructions but cannot launch a successor. | Handoff fails in practice or hook behavior changes. |
| 2026-09-25 | Before new QAT training, establish an all-nine-linear W1A16/W1A8/W1A4 activation-precision sweep on the 2080 Ti, rerunning FP16, Q8_0, Q4_0, and W1A1 in the same timing matrix. Measure quality, identical-input operator cost, complete round cost, and serving throughput separately. | The user requested these formats and a durable measurement list. Existing FP16/Q8_0/Q4_0/W1A1 `draft()` times are known, but they omit EAGLE `process()` and isolated verification; matched FP16/Q4_0 component traces are missing. Binary weights must be held identical across W1Ax to identify the activation-precision effect. See the [study plan](../experiments/w1ax-activation-precision-plan.md). | Numerical/dispatch gates fail, memory prevents the matched matrix, or the development-prompt acceptance screen rules out a format before native timing. |

## Pending options from the local research review (2026-09-25)

The user requested seven independent Astra-high analyses. The
[review synthesis](../experiments/research-review-2026-09-25.md) preserves all
reports and the evidence for these options; no new research direction has been
selected and the active W1Ax goal remains unchanged.

- **Native work elimination:** stage-tag and test unused-head removal during
  cache catch-up, then separately evaluate K/V-only reconstruction, genuine
  per-sequence draft caps and accepted-prefix deferral. Local source supports
  these opportunities; full-round savings and cache correctness are unmeasured.
  Apply shared optimizations fairly to FP16 and Q4_0 anchors.
- **QAT supervision and support:** first audit state/parent/verifier joins,
  native-chain versus PyTorch-tree sampling and unsupported target labels.
  Preserve the bounded target-aligned head pilot before considering an extra
  reconstruction-control arm, refreshed states or wider recurrent training.
  The optional arms in the advisory report do not amend the frozen budget.
- **Precision versus architecture:** use common-weight W1Ax evidence to choose
  whether one structured-scale/mixed-precision follow-up has merit. DFlash/DSpark
  is the strongest later non-EAGLE screen in the review, with no local W1A1
  result yet; an architecture pivot remains the user's decision.
- **Policy diagnostic budget:** audit top-10 confidence normalization before
  running potentially redundant `p_min=0.1` trials. The full Cartesian policy
  grid is costly; any scope or grid amendment must be recorded before selection,
  not silently substituted for the requested primary matrix.

## Literature adjustments (2026-09-25)

The [primary-source cross-reference](../experiments/research-cross-reference-2026-09-25.md)
updates the advice above without opening a new goal or changing the frozen
protocol. Favor paired released **DSpark with DFlash control** if a later
architecture screen is chosen. Treat target-greedy CE as a bounded hypothesis,
not a proven optimum; the failed draft-KL pilot does not establish its cause.
Structured scales and learned thresholds remain alternative representation
candidates whose kernel costs must be measured. Preserve the single 500-step /
45-minute QAT run; optional objective-control or refresh arms remain unapproved
additional scope, not an existing training budget.

Full-round instrumentation, alignment/support audits, same-device FP16/Q4_0
anchors and source-supported graph/cap experiments retain priority. Any later
policy must account for context and batch cost, and any stochastic theory must
match the actual native target-sample-and-match verifier. Literature supports
these checks; it supplies no local W1A1 speedup or missing timing measurements.

## Pending quantization recovery choice after PrismML review (2026-09-25)

The [45-minute research report](../experiments/prism-quantization-research-2026-09-25.md)
recommends an ordinary/cast-only/dense-binary/packed-binary graph bridge, followed
by a small row/group × mean-absolute/output-fitted scale comparison on designated
training captures. A fitted readout is a diagnostic for body/head incompatibility
before escalating to full recurrent QAT. Existing Q1_0 export/runtime support
removes an immediate kernel-port prerequisite.

The checked 98.2% external claim is ternary aggregate benchmark retention, not
W1A1 or token agreement. Released binary codes differ from naive signs in a
bounded artifact sample, but the checkpoint-producing recipe remains undisclosed.
Neither literature nor the local CPU diagnostics establishes a native SM75 win.

User choices still pending: which representation and training scope to advance;
any replacement for the existing one-run QAT budget; whether practical one-bit
weights are a separate deployment track before returning to A1. The original
W1A1 objective, current suite and FP16/Q4_0 comparison anchors remain intact.

## Post-suite recovery recommendation (2026-09-27)

The user requested a consolidation of next steps after the completed W1Ax
suite. The [consolidated recommendation](../experiments/one-bit-next-steps-2026-09-27.md)
supersedes earlier advice to finish the suite or try the same policy grid.
Every tested W1Ax policy loses to both same-cell anchors, and restoring A16
still leaves development acceptance at 0.117 versus FP16's 1.047 drafts/round.
The historical fixed-trajectory removal of draft() spans is insufficient to
reach either anchor. Representation/quality recovery therefore takes priority
over another kernel or policy sweep of the unchanged model.

Recommended next bounded experiment: reuse completed operator gates, finish
missing graph/state parity checks, and test row/group-128 × mean-absolute/
output-fitted scales with fixed signs and A16 on designated training captures.
Use online development acceptance, not reconstruction error, as the gate.
A fitted readout on frozen binary-body states is a conditional diagnostic
before choosing head/interface or wider recurrent training.

Pending user-owned choice: adopt useful W1 weights with wider activations as
an intermediate milestone toward W1A1, or prioritize immediate joint W1A1
adaptation; separately choose any training-scope/budget amendment. The existing
500-step/45-minute head pilot is a narrower proposal, not authorization for
all-body QAT. Mixed precision and a DSpark/DFlash architecture screen remain
later alternatives. No new goal, GPU run, training or final-set use began.


## After the approved scale screen (2026-09-27)

The user approved and the team completed the bounded four-way fixed-sign A16
scale experiment on RTX 5080. [Results](../experiments/binary-scale-fitting-5080.md):
row mean 0.119, group mean 0.161, fitted row 0.290, fitted group D 0.425 accepted
drafts/round, versus FP16 1.037 and Q4_0 1.042. D wins over A/B/C on all 24
prompts but loses to both controls on all 24. All output IDs match target-only.
Final prompts remain sealed. This completes that authorization.

Recommended next choice: use D as the binary reference for a bounded forced
common-history body/head factorial, then a training-only regularized readout
on frozen D-body states if the user approves. This would diagnose whether a
head/interface correction can use the remaining body information. Current
own-history acceptance does not locate the defect. Do not promote this quality
reference into kernel/performance work as if control-level acceptance recovered.

Alternative: separately authorize more scale-only fitting, acknowledging that
fusion's 2,560 group rows exhausted the frozen 512-iteration budget (maximum
relative projected-gradient residual 1.76e-5, tolerance 1e-6). The present result
is not a proof that all scale solutions are exhausted. No budget extension,
sign training, readout fitting, broader QAT, A1 change or architecture pivot is
automatically authorized. The next scope and budget remain user-owned.


## Primary comparison baseline (user decision, 2026-09-27)

The user explicitly selected **Q4_0 EAGLE, not FP16 EAGLE**, as the comparison
target. Use Q4_0 as the primary denominator and success gate for acceptance,
latency and total model throughput. FP16 remains a secondary diagnostic control.
This does not authorize changing the target/verifier model precision, reopening
completed goals, or starting a new GPU experiment. Historical measured values
remain unchanged; future summaries should lead with Q4_0.


## Approved rescue and readout follow-up (2026-09-27)

The user approved BOTH four independent Q8_0 subset rescues of fitted-group D and one bounded true-verifier-aligned FP16 head fit on frozen D body, followed by graph-enabled repeated native performance measurements on RTX 5080 only. The [completed protocol](goals/binary-rescue-head-5080.md) froze comparison, combination admission, data boundaries and budgets. Q4_0 remains primary; original FP16-head swap was an essential control. Final prompts stay sealed. The completed results and recommendation are recorded below.

Pending larger decisions remain user-owned: head rebinarization, body QAT, changed signs/scales/A1, expanded training/search budgets, architecture change, deployment promotion and final-set evaluation. This exploratory goal recommends one next choice without starting it automatically.


## Completed rescue/readout assessment (2026-09-28)

The approved rescue and head-fit experiments completed, and the clean
five-repetition RTX 5080 primary timing matrix is available. Q4_0 remains the
lead comparison: 1.042 accepted drafts/round and 135.1 full-request tokens/s.
The admitted attention+fusion Q8_0 rescue reaches 0.696 accepted/round and
83.3 tokens/s (0.616× Q4_0). The single bounded fitted FP16 head reaches
0.555 accepted/round and 60.8 tokens/s (0.450× Q4_0). Every measured output
matches the Q4_0 and target-only raw token IDs. Neither candidate beats even
the 97.7 tokens/s target-only control. The fully rerun 24-prompt round trace
matched all five timing repetitions and validated complete-round denominators.
The separate six-prompt, longer-context diagnostic also favored Q4_0: 127.0
client tokens/s, versus 68.5 (0.539× Q4_0) for the combined rescue, 47.1
(0.370×) for the fitted head and 93.8 for target-only. These are
development-workload findings; no final-set data was used.

Recommendation: do not promote either endpoint as a Q4_0
replacement. If the user chooses another bounded recovery goal, prioritize
target-aligned recurrent body-plus-head adaptation under an explicitly fixed
binary/A16 deployment format, with native Q4_0 quality and throughput gates.
The forced common-history factorial assigns a much larger first-position
agreement loss to the D body (19.8 points under Q4 head) than to the D head
(5.9 points under Q4 body); the frozen-body dense head gains only 8.6% over
its untrained control. In the fully instrumented fixed-trajectory calculation,
removing every exclusive draft CPU span gives the combined rescue 1.088× the
instrumented Q4_0 round-only rate but only 0.993× Q4_0's separately timed
decode rate; these scopes cannot be equated to measured speedup. The fitted
head remains below the instrumented Q4_0 round rate even under that
counterfactual. These observations motivate a body-aware fit but do not prove
that it will recover quality or overcome current A16 runtime cost.
This is a recommendation, not authorization: no broader QAT, budget increase,
head rebinarization, final-set evaluation or new GPU run begins from it.

## Joint binary body/head preparation (2026-09-28)

The user approved a new jointly trained binary body/head research goal and
initially prohibited **all GPU and accelerator use**. CPU preparation began
under the [active goal](goals/recurrent-binary-body-head.md)
and [trial protocol](../experiments/recurrent-binary-qat-plan.md). The earlier
head-only 500-step/45-minute budget is not silently transferred to all-body
training. Q4_0 remains the primary later native gate; no new outcome has been
measured.

Pending user-owned research choices, with independent CPU work available:

| Choice | Options and present evidence | Gate |
| --- | --- | --- |
| Scale boundary | Keep D's `scale >= 0` including exact zero, or change to strictly positive scales with a documented initialization difference. D's NNLS fit explicitly allowed zeros; deployment already accepts nonnegative F32 scales. CPU implementation should retain D exactly. | Freeze before training/export. |
| Training prefix distribution | Teacher-force bounded native D prefixes with exact labels and differentiable student recurrence, or refresh exact-prefix labels as student proposals change. The former is reproducible and CPU-testable but off-policy after sign changes; the latter requires new real-model target capture and extra compute. Neither guarantees live acceptance. | Capture/support audit and a fixed budget before real training. |
| First all-body budget | Proposed one 500-step / 45-optimizer-minute run, at most 8,160 starts of up to five steps on the frozen 96 training prompts; select on 24 development prompts, leave 24 final sealed. This is a proposed cap, not authorization for training. | Explicit future user budget decision; the GPU access restriction has been lifted. |
| Unsupported labels | The full frozen 96-prompt D capture has 1,110 unsupported labels among 40,815 valid rows (2.72%); mean mapped target probability mass is 0.972 on those captured prefixes. The proposed CE still applies only to valid supported labels while keeping unsupported rows in denominators. A new vocabulary-support objective is a separate user choice. | Reject any capture with wrong offset/inverse map or prefix/verifier join; do not infer live acceptance from mapped mass. |
| Learned-scale metadata | Resolved for the loader declaration: fork commit `7f23c89b3` accepts `f32_learned_nonnegative` under v2/v3, and eight CPU-only native loader fixtures pass. The exporter retains that truthful rule rather than relabeling learned scales as NNLS. | A real trained GGUF still needs full native load and numerical parity before deployment. |
| Training reduction order | Use the exact sequential F32 CPU reference, or explicitly choose grouped F32 matmul within each 128-input group for a practical trainer. The grouped path still runs hard signs/A16/scales and ordered group accumulation, but 35/430 archived D outputs differed from native by up to `0.0001220703125` absolute (`7.422315e-7` scaled). The grouped student matched **44/44** native mapped top IDs over 11 recorded rounds from three training prompts; native-order first-round checks matched 15/15. Normalized state drift reached `0.004617`, dominated by attention. | Freeze one path before training and require broader native argmax/trajectory parity before interpretation. |
| Full-drafter numeric gate | Require exact backend-matched attention arithmetic, or predeclare a bounded state/logit tolerance together with native argmax and accepted-trajectory checks. Two CPU requests now have exact stored native F16 K/V writes and prefix masks; the corrected adapter matches 123,900/123,904 projected F16 keys and all values across three later-round joins. Modeling native F16 attention arithmetic cut first-seed maximum differences from roughly `0.010/0.0057/0.0099` to `0.00178/0.00000095/0.000847` for prose/code/reasoning, but residual state drift remains. Matching 44 top IDs does not establish a safe tolerance. | Keep whole-drafter state/logit and target-feature parity unresolved; no training-quality or Q4_0 promotion follows. |

Post-acceptance CPU joins, stored-cache byte captures and mask checks now
constrain the first material gap to attention arithmetic in the tested
single-sequence geometry. The source-guided F16 replay substantially reduces
that gap but does not choose the tolerance option above.

Independent target-feature forwards now also cover code/reasoning on Apple
M3 Max CPU and one prose prefix in CUDA/F16 on the RTX 5080. The median
relative row L2 discrepancy at taps 2/18/33 is about 0.3–0.6%; the
largest checked CPU reasoning row reaches 2.088%, and the CUDA prose
maximum reaches 1.753%. Source prompt embeddings and sampled gate weights
match the pinned FP16 GGUF, but the independent eager and native attention
paths differ. The [target-feature note](../experiments/recurrent-binary-target-feature-parity.md)
records exact metrics and limits. This evidence does not by itself select
an acceptable training tolerance or an alternate exact backend path.
Changing only the independent 5080 F16 attention from eager to PyTorch SDPA
left median errors at 0.306/0.434/0.382%, a small mixed change from
0.311/0.426/0.385%; attention implementation alone did not close the gap.
The complete 96-prompt training-prefix check found median errors of
0.293/0.538/0.488% across taps 2/18/33, but also one **10.014%** tap-18
row (12.135% with SDPA) on the same prompt/position. An isolated ggml CPU
layer-0 probe found projection differences before attention despite nearly
exact norm and bitwise source operands. These results strengthen the case for
investigating exact operator behavior before setting a tolerance; they do
not themselves authorize a threshold, optimization budget or final-set use.
In the native full capture, 103/427 pairs of rows with the same token prefix
also differ across requests after the first token (up to 0.821% relative
row L2 at tap 33); the execution/capture cause is unresolved.
A same-binary CUDA recapture of the outlier prompt reproduced all 222,720
native prefill values bitwise; the independent-forward gap is therefore
reproducible on that native path, without identifying its operator cause.
A bounded F16 intervention that substituted native tap-2 input into the
independent forward reduced the outlier's tap-2 error from 0.324% to 0.020%
but increased its tap-18 error from 10.014% to 11.659%. This favors
investigating the intervening target blocks before setting any tolerance;
the intervention's F16 cast and backend difference preclude an exact causal
attribution. The pending exact-parity versus predeclared-tolerance decision
and all-body budget remain with the user.
A new native layer-input ladder reproduced the sealed old taps exactly and
located the outlier's largest adjacent relative-error rise across target
block 14 (1.209%→4.148%; absolute RMS 0.01202→0.04218). This narrows the
next exact-parity probe to that block but does not yet identify its attention,
FFN or residual arithmetic. It does not justify setting a numeric threshold.
A same-input F16 intervention then reduced block-14 output error from 4.148%
to 0.149% (RMS 0.04218→0.001516), showing that this block mainly amplifies
an upstream difference under the independent forward. Screen local blocks
from their own native inputs before selecting a tolerance or blaming a single
block operator. The research choice remains open.
The full same-input screen across blocks 0–17 found at most 0.272% local
output error at the outlier position, compared with 10.014% accumulated
error by layer 18. This favors a distributed-backend-arithmetic plus
amplification explanation, not a single large local failure. Other positions
have different absolute-error rankings, and the 29-token screen cannot set
a global tolerance. The choice of exact backend arithmetic versus a
predeclared numeric/trajectory gate remains user-owned; captured native
features and verifier logits can be reused while independent drafter parity
work continues.
See the [feature report](../experiments/recurrent-binary-target-feature-parity.md).

The drafter now has a **bitwise native CPU attention oracle** for the archived
prose/reasoning captures: 593,920/593,920 F32 outputs match from stored F16
cache bytes and masks, including later rewrites. A real-size later-only CE
probe also proved gradient reaches the earlier proposal's pre-norm state
and appended K/V without an optimizer step. These close operand/oracle and
causal-gradient preparation gaps. The default Python student still uses F32
attention. An optional native-forward/F32-surrogate-backward CPU diagnostic
now closes prose first-seed attention bitwise after Q/K row conversion and
reduces reasoning error, while preserving later-only gradients. Its
derivative is explicitly a surrogate, and residual state/cache differences
remain. The user must choose whether this arithmetic becomes part of a
training recipe or whether a predeclared numeric/trajectory tolerance is
acceptable. These findings do not transfer the earlier head-only optimizer
budget to all-body training.
The same-input FFN check then matched the prose native output exactly with
ordered F32 binary arithmetic; grouped matmul differed by at most
`1.4305e-6`. Both modes remained `6.1035e-5` from native on reasoning,
far above their `1.9073e-6` mutual difference. Thus switching the whole
trainer to native-order reduction alone would not close that residual.
The training arithmetic and numeric gate remain user-owned decisions;
the next exact-parity probe is the missing native FFN intermediates.
The subsequent first-seed reasoning recapture found ordered gate and up
bitwise exact at all 9,728 values each. The first gap is native versus Torch
SiLU (3,171 differing values, maximum `4.7684e-7`); the down output's
maximum gap is `6.1035e-5`. This supports a bounded elementwise SiLU
arithmetic probe before choosing whether to carry exact native CPU arithmetic
into the student. One CPU row does not settle the exact-versus-predeclared-
tolerance policy or authorize all-body optimization. See the
[stage report](../experiments/recurrent-binary-ffn-stage-parity.md).

**CPU forward-parity evidence (2026-09-28):** an opt-in, no-grad Apple M3
Max CPU student now uses pinned ggml RoPE, Flash Attention and vector
SiLU with ordered W1/A16 candidate-D projections. Its ordinary context
rebuild and draft-step calls match all checked native K/V bytes, graph
states and captured head-logit probes across prose and reasoning first
rounds plus one reasoning accepted-draft catch-up (13 depths total).
This closes those bounded CPU forward comparisons, including the earlier
SiLU gap. It does not validate CUDA/SM75 arithmetic, independent target
feature reproduction, all 96 training trajectories, full mapped logits
or the backward derivative. The user-owned choice remains either to
establish an exact training-backend path or to predeclare a bounded
numeric/state and accepted-trajectory tolerance before optimization.
Neither option nor the proposed all-body budget is approved by these
CPU diagnostics. See the [integrated report](../experiments/recurrent-binary-integrated-cpu-diagnostic.md).

**Target-feature CUDA gate update (2026-09-28):** four output-preserving
RTX 5080 block-0 callbacks on the frozen 29-token training outlier
showed nonzero HF CUDA/F16 differences at attention norm, K/V projection,
attention residual and FFN. At position 3, relative row errors were
0.0325%, 0.0922/0.0994%, 0.2547% and 0.2721% at those successive
boundaries. Replacing the HF K/V input with native norm cast to F16 left
0.0916/0.0934% K/V error. Q normalization cannot be used as native-path
evidence because requesting that callback changed 1,911 block-output
values; output-only, K, V, norm and FFN callbacks preserved the sealed
block output bitwise. These measurements favor further exact operator
work before setting a global tolerance, but do not compel the user to
choose that path. The [report](../experiments/recurrent-target-block0-safe-taps-5080.md)
adds no all-body budget or final-set authorization.

**Same-input V operator evidence (2026-09-28):** an isolated ggml CUDA
V projection on the RTX 5080 matched the output-preserving native server
tap in all 29,696 F32 values on the identical captured normalized input
and F16 weight. Explicitly casting the ggml input to F16 changed no
output values; Torch CUDA/F16 on the same operands matched only 5,351.
This makes backend matmul arithmetic/dispatch, rather than that input
cast, the immediate V mismatch under the tested geometry. K, intrusive
Q, attention and later target blocks remain unproven. The result
strengthens the exact-backend option but does not select it, set a
tolerance or authorize optimization. See the [operator
report](../experiments/recurrent-target-v-cuda-projection.md).

**Same-input K operator evidence (2026-09-28):** an isolated ggml CUDA
K projection and per-head RMS norm on the RTX 5080 matched all 29,696
output-preserving native server values bitwise, with either F32 or
explicitly F16-cast native norm input. Torch CUDA/F16 differed already
at raw K projection (6,530/29,696 exact), and its Qwen3 norm output
reproduced the earlier 0.0916% position-3 native gap. Feeding the Torch
norm identical ggml raw F32 K cut that row gap to zero and gave
24,456/29,696 exact values; casting the raw K to F16 introduced 0.0338%
row error. Backend projection arithmetic and F16 intermediate precision
both matter under this geometry. This supports continued operator
attribution but does not choose an exact training backend versus a
predeclared numerical/trajectory gate, or authorize optimization. The
native Q tap remains intrusive. See the [K report](../experiments/recurrent-target-k-cuda-projection.md).

**Output-preserving Q evidence (2026-09-28):** a deferred post-RoPE Q
read at the already safe K-norm callback preserved all 74,240 block-0
output values and the safe K payload. A standalone RTX 5080 ggml CUDA
Q projection, 32-head norm and NeoX RoPE replay then matched all
118,784 captured Q values bitwise; explicit F16 input casting changed
none. The old direct Q callback differed by at most `9.5367431640625e-7`
in Q and changed 1,911 later output values. Its Q data remain excluded
from native server-path attribution. This closes one same-input Q
operator fidelity gate, but Torch Q, attention/residual, later target
blocks and accepted trajectories remain open. The finding does not
select the user-owned exact-backend versus numeric/trajectory policy or
authorize training. See the [Q report](../experiments/recurrent-target-q-deferred-cuda.md).

**Same-input Torch Q evidence (2026-09-28):** with the safe post-RoPE
native Q boundary available, a source-module Torch CUDA/F16 Qwen3
control on the identical native norm input matches 14,513/118,784
raw ggml Q F32 values and only 2/118,784 safe native post-RoPE Q F32
values. Position-3 relative row errors are 0.2880% and 0.2347%,
respectively. The exact ggml replay still matches all native Q values.
Retaining ggml intermediate tensors changes its final Q by tiny F32
amounts, so only the unretained graph is used as the native fidelity
path. Projection backend arithmetic is already a material source of
Q difference under this geometry; attention/residual amplification
and later target blocks remain to be attributed. These observations
do not choose a training numeric/trajectory policy or budget. See
the [Torch Q report](../experiments/recurrent-target-q-torch-control.md).

**Post-RoPE K capture evidence (2026-09-28):** a direct K-RoPE callback
preserved all 74,240 sealed block-output values on the frozen 29-token
RTX 5080 training prefix and yielded one 29,696-value F32 K tensor.
Output-preserving Q, K, V and FFN-input boundaries are now available
for same-input attention attribution. The result does not resolve
Flash Attention/output-projection arithmetic or justify either a
training tolerance or optimizer budget. See the [safe K-RoPE
report](../experiments/recurrent-target-k-rope-safe.md).

**Same-input attention evidence (2026-09-28):** a source-Qwen3
CUDA/F16 eager replay matches its full-model block-0 residual
74,240/74,240 F16 values bitwise. Replacing source Q/K/V with
output-preserving native post-RoPE Q/K and V cast to F16 reduces the
frozen position-3 residual error from 0.2547% to 0.2048%, leaving a
material gap. The remaining difference cannot be assigned solely to
Flash Attention: Q casting, F16 output projection and residual
arithmetic also differ from native ggml. The loaded HF model's rotary
frequency buffer is F16; the earlier isolated Q control used F32,
and both variants are reproduced. Continue exact same-input attention
and output-projection attribution before selecting a target-feature
numeric/trajectory gate or requesting an all-body budget. See the
[attention report](../experiments/recurrent-target-attention-intervention.md).

**Exact Flash Attention residual evidence (2026-09-28):** a pinned
standalone ggml CUDA graph on output-preserving native Q/K/V,
256-slot F16 cache and F16 causal mask matches all 74,240 native
block-0 attention-residual F32 values bitwise, including F16 O
projection and F32 residual add. Rounding Q through F16 and back to
the kernel-required F32 changes no value in this geometry. This
closes the combined same-input native operator fidelity gate; the
remaining Torch path difference must be decomposed across attention,
O projection and residual stages before selecting a training
numeric/trajectory policy. It does not authorize an all-body budget.
See the [Flash Attention report](../experiments/recurrent-target-flash-attention-cuda.md).

**Pre-O attention boundary (2026-09-28):** a native RTX 5080
`kqv_out-0` callback captured 118,784 F32 values and preserved all
74,240 sealed block-output values. The safe tensor separates Flash
Attention from O projection for the next same-input comparison. It
does not yet assign the remaining Torch residual error to either
stage or change the user-owned numeric policy and training budget.
See the [pre-O report](../experiments/recurrent-target-attn-output-safe.md).

**Attention/O stage attribution (2026-09-28):** standalone ggml CUDA
Flash Attention and O-plus-residual stages each match their
output-preserving native block-0 taps bitwise on the frozen RTX 5080
training prefix. Torch eager attention on native Q/K/V has 0.1058%
position-3 pre-O error, while Torch F16 O projection on the identical
native pre-O tensor differs from ggml by 0.2071%. F16-versus-F32
residual addition is smaller (0.0212% mutual position-3 row error).
These controlled differences explain why native Q/K/V substitution
alone did not close the Torch residual; they do not combine as a
single additive error budget or establish later-layer/trajectory
parity. Continue later-block attribution before choosing a training
numeric policy or all-body budget. See the [stage-split
report](../experiments/recurrent-target-attention-stage-split.md).

**Block-14 capture evidence (2026-09-28):** six native attention/FFN
stage tensors on the frozen RTX 5080 training prefix preserve all
74,240 sealed block-14 output values. They permit a stage-specific
test of the earlier position-3 error jump without using the
intrusive or approximate block-0 callbacks as proxies. The capture
alone does not assign the amplification to attention or FFN, set a
numeric tolerance or authorize all-body training. See the
[block-14 report](../experiments/recurrent-target-block14-safe-stages.md).

**Block-14 stage intervention and GPU availability (2026-09-28):**
with accumulated HF input, the position-3 error is 1.2157% at FFN
input, 12.1527% at FFN branch output and 4.1478% at block output.
Replacing only the full block input with native rows cast to F16
leaves 0.1015%, 0.5045% and 0.1491% at those boundaries. This
localizes the observed amplification mainly to the FFN path on that
row, while leaving the causal FFN operation, other rows and global
target-feature policy unresolved. The prior block-14 intervention
metrics and eleven source weights reproduce. The user has now made
the RTX 5080 unavailable; the local host pause blocks new runs and
the completed supervisor's process group is absent. No training
budget or numeric/trajectory gate is selected by this evidence.
See the [stage report](../experiments/recurrent-target-block14-stage-intervention.md).

**Exact-prefix row clarification (2026-09-28):** the native verifier computes
target logits for every proposed prefix in a speculative batch, and the cloned
sampler advances along each proposed token even after an earlier live
rejection. A valid, supported row beyond that rejection can contribute
teacher-forced later-position CE. Keep `verifier_reached` as a separate live
acceptance diagnostic and exclude rows with no actual target label or
invalid/padded ancestry. This clarification preserves the proposed recurrent
objective; it does not turn off-policy CE into a live acceptance estimate.
For `A` accepted drafts, native target **input features** from seed and
accepted draft rows `j=0..A` join retained history; `j>A` are rejected
suffix. The verifier label at `j=A` is still a reached label. The CPU feature
preparer checks this offset explicitly.

**RTX 5080 authorization (2026-09-28):** after the earlier availability
notice, the user explicitly said the GPU is free and to use it when needed.
The RTX 5080 is available for this goal's parity and capture work under
the host, tmux and one-owner protocol. This does not authorize the proposed
all-body training budget or final-set evaluation. The subsequently completed
96-prompt capture and GPU release are recorded in the active goal file;
no optimization was started.

## Versioned compact capture storage preparation

The CPU implementation provides `recurrent_binary_capture_v2` with the
`recurrent_capture_storage_policy_v2` hard-CE label-only policy. This is an
engineering option for the existing goal, not selection of the full-tier
storage budget. It omits a second raw full-vocabulary logit copy while keeping
the original v1 bundle and native capture unchanged. Conversion requires one
successful retained-raw v1 audit before publishing a new immutable v2 bundle.
For future captures, `build_recurrent_capture_v2.py` directly accepts the native
raw stream plus prepared rows/features: it creates an owned transient v1 audit
shadow using a same-filesystem hardlink, converts to v2, then removes only that
shadow's metadata/link. This avoids creating a duplicated-raw v1 prerequisite.
The original raw inode, byte count and link count are verified after cleanup;
original hashes are checked by the v1 and v2 ancestry audits. Cross-device
hardlink failure aborts with no copy fallback. The legacy builder's default
remains an independent raw copy. This is not a raw-free source capture.
`raw_retirement_allowed` is always false; neither conversion nor audit deletes
raw artifacts. Existing v1 audit and provider behavior stays unchanged.

The v2 audit rederives rows from retained native cloned-verifier-sampler head
metadata and canonical rounds, checks exact proposal prefixes, absolute/offset
d2t identity, unsupported-label masks, live verifier reach, source hashes,
accepted-prefix feature coverage and every request's response emissions. It
retains the historical raw-logit row/source joins as ancestry. Labels are the
captured sampler labels, including valid labels beyond an earlier live
rejection; they are never reconstructed by an argmax assumption. Original
prepared feature bytes and source/preparer/v1 audit manifests remain hashed.

Label-only storage cannot recompute target probabilities, mapped probability
mass or the original sampling arithmetic after raw bytes become unavailable.
The audit states those limits and leaves initial/terminal sampler parity and
native feature/KV numerical parity unverified. Historical native cell manifests
omit the timing round-trace hash; v2 records that trace's conversion hash and
rechecks all its emissions against canonical rounds and raw response IDs. A
cell trace hash, when present, must also match. This preserves response
continuity without claiming an older manifest authenticated an omitted file.

The converter and independent audit can be run on train-only artifacts (write
the new audit outside the immutable bundle):

```sh
python3 scripts/build_recurrent_capture_v2.py \
  --rows-dir <prepared-rows> --features-dir <prepared-features> \
  --target-logits <native-raw-logits.f32> --capture-root <native-capture> \
  --train-prompts <train_prompts.jsonl> --output <new-v2-bundle> \
  --expected-prompt-sha256 <frozen-train-shard-sha256> --expected-prompt-count <count> \
  --expected-target-sha256 <target-gguf-sha256> \
  --expected-draft-sha256 <draft-gguf-sha256> --expected-map-raw-sha256 <d2t-raw-sha256>
# Alternatively, convert an existing audited v1 bundle:
python3 scripts/convert_recurrent_capture_v2.py \
  --source-manifest <v1-bundle>/manifest.json \
  --capture-root <native-capture> --output <new-v2-bundle> \
  --expected-prompt-sha256 <frozen-train-shard-sha256> \
  --expected-prompt-count <frozen-train-shard-count>
python3 scripts/audit_recurrent_capture_v2.py \
  --manifest <new-v2-bundle>/manifest.json --output <new-audit.json> \
  --expected-prompt-sha256 <frozen-train-shard-sha256> \
  --expected-prompt-count <frozen-train-shard-count>
```

`load_audited_capture_v2` exposes accepted-prefix round inputs for an explicit
future hard-CE provider adapter. V2 remains `training_eligible: false` and
`readiness: preparation_only`; a separately reviewed versioned eligibility
contract and provider integration are required before training uses it. The
pilot's v1 calibration eligibility does not carry over automatically.

The full-tier choice remains among label-only hard CE, the existing versioned
compact top-k/tail teacher with an additional probability re-audit contract,
and provisioned retained raw storage. Top-k/tail probabilities remain an
approximation, with outside-draft mass visible and no implicit draft
renormalization. This implementation chooses no full-tier capture, raw
retirement, final-data access, training budget or new objective. The next
engineering integration is the explicit hard-CE v2 provider path; the next
storage decision belongs to the user before the full 2k capture.

## Continuous A8/A1 host capacity path (pending)

User-authorized launch on2026-09-30UTC passed the bounded W1A8 native/CUDA gate
on RTX5080/SM120 with frozen target FP16, but A1 initialization stopped on the
unchanged12GiB additional+2GiB floor host admission. WSL exposes15.21GiB total;
available RAM afterGC+glibc trim was12.94GiB,1.06GiB below the14GiB guard.
No optimizer/full corpus capture began. Both supervisors exited and evidence
is preserved; no further retry/guard relaxation is authorized by this checkpoint.

Options presented to user:
- Prepare a WSL memory increase for review after measuring Windows capacity
  and current settings. Applying it requires a WSL restart; it changes host
  resource allocation, not research math/model precision. No change applied.
- Keep the cap and engineer sequential isolated readiness processes to release
  model/runtime residency between stages. This needs process/stop/provenance
  tests and actual admission validation; sufficient headroom is not guaranteed.

The WSL memory setting and restart behavior are documented in
[Microsoft's configuration reference](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).
Root awaits user choice, continues only read-only capacity evidence, and keeps
the hourly pinned-Luna monitor paused while there is no live experiment.

Capacity inventory is complete: Windows total33,477,398,528B and free
18,851,332,096B at observation; current user's `.wslconfig` absent;
WSL2.7.11.0/kernel6.18.33.2-2. Its default50%cap is consistent with observed
VM total (inference). A raised ceiling is not full immediate allocation;
actual concurrent physical headroom still requires measurement after an
approved change. No setting/restart/change was applied.

User chose the memory path by confirming “Should be good, go for it” after
20GB/restart instructions. Luna is authorized to verify actual new capacity and
resume under unchanged guards. The agent has not changed Windows settings; the
user reports the prerequisite complete. Actual measured result is pending.

Verified after restart: config`memory=20GB`, WSL total20,971,151,360B,
available20,237,811,712B and Windows free physical21,653,372,928B. Resource path
is resolved for the next measured admission; actual stage fit remains gated.
The agent did not change or restart Windows/WSL.

## Training optimization audit recommendations — 2026-10-01

The user requested five GPT-6.1 Sol/high source-and-literature audits of custom
QAT/fusion, batching/cache construction, binary optimization, learned activation
quantizers and precision/recurrence curricula. The
[ranked synthesis](../experiments/training-optimization-audit-2026-10-01.md)
and its five reports are complete. These recommendations do not select a new
training recipe or authorize modifying the live exact-resume experiment.

Recommended engineering choice: prepare chunked K/V-only Torch context,
stacked per-chain head and A1 no-gradient native-only forward. The architecture
supports removing 77.88% of context linear MACs without changing round-level
Adam cadence; wall-time benefit remains unmeasured. CPU contract gates and
later same-device phase timing precede an adoption claim. Full custom backward,
ragged minibatching and a packed training stack are lower priority.

Quality choices remain user-owned and conditional on the baseline learning
curve. The narrowest optimizer test changes latent initialization inertia
from the current ±0.5 while preserving checkpoint-zero signs/scales; the
strongest causal-LLM precedent is ParetoQ-inspired weight/scale gradients,
with different activation/head/budget assumptions. Learned A4 relative clipping
or A1 thresholds require a new deployment quantizer; positive A1 amplitude
alone adds redundant capacity. A short A8-to-A1 warm-start and mild full-horizon
early-depth CE weighting are separate candidates, not a combined schedule.
Charge all warm-start/teacher/search compute, preserve data ancestry, compare
native acceptance at equal GPU hours and report common-token learning curves.
No current-student token may reuse a changed-prefix capture label. Q4_0 remains
primary; final prompts stay sealed. No new experiment was launched by the audit.

## Broader CPU-first research options — 2026-10-01

Ten GPT-6.1 Sol/high audits requested by the user are complete; the
[ranked shortlist](../experiments/broader-project-audit-2026-10-01.md)
links all ten reports. This is a broader preparation proposal within the active
goal, not an architecture pivot or authorization to alter the live preparation
run. The preparation-only boundary recorded in `553244e` remains in force;
no optimizer updates are authorized by this audit.

Recommended quality track: a bounded fusion-only sign/scale fitter plus one
cheap affine/bias/residual control on the same eligible local training operands.
Direct fitting retains the existing sign/scale format; corrections have distinct
metadata/precision costs and need separate native gates. Full calibration
captures are largely remote; diagnostic samples do not substitute for them.

Independent alternatives are a learned Q1_0 export/activation oracle, coupled
FFN slicing, fixed-support head factorization/shortlist preparation, whole-round
data selection, DSpark/DFlash compatibility fixtures, adaptive stopping and a
two-bit control. Preserve exact data ancestry, frozen target/verifier, honest
bits and activation precision. Existing teachers/features cannot silently be
reused at changed prefixes or missing target taps. Compare real accepted
proposals and complete decoding cost against Q4_0 before promotion.

No option is selected by publishing the audit. New training, capture, architecture
or deployment experiments remain user-owned. The user requests simpler,
high-impact explanations from now on; detailed evidence remains in the reports.

## Preparation resume audit cost — 2026-10-02

The preparation owner inspected frozen source `7547d253`, without contacting or
changing the running supervisor08. At07:41:20.302347UTC the saved CPU observation
was healthy at audit ordinal200/353, with327 completed manifests retaining
10,000train and224of1,002development prompts. All optimizer counters remain zero.
The remaining778development prompts need native response/feature capture and
label construction; the current ordinal instead describes repeat audit progress.

The frozen capture loop skips native capture for an existing labels manifest,
but always calls `audit_native_labels`. Every resume starts this loop at zero.
The CPU audit hashes all manifest-owned files, re-derives labels/maps/prefix joins,
rebuilds accepted-prefix features in a temporary directory and compares their
hashes, then validates traces, feature ledgers and request/response ancestry.
Provider manifests are assembled after the entire capture/audit loop. This is
substantial CPU/file work, not regeneration of the already retained prompts.

Five existing saved observations advanced109→200 between06:42:04.636798 and
07:41:20.302347UTC:91shards in3,555.666seconds, about39.1seconds/shard or92shards/hour.
Individual intervals were approximately38–40seconds/shard. At that observed rate,
roughly127retained shards still ahead imply another83minutes of repeat audit;
this is a conditional estimate, not an ETA for full preparation. Native capture,
readiness/coverage assembly, paired CUDA smoke and checkpoint-zero save follow.
No timing of those remaining stages is established by these CPU observations.
Recorded free disk at07:41 was424,596,348,928bytes (395.4GiB); the registered
622,868,961,328byte forecast is a conservative full-corpus upper bound, not actual
occupied storage. No CPU-versus-disk bottleneck profile or device throughput was
measured, so neither disk capacity nor GPU speed is identified as the cause.

**Proposal, not selected or deployed:** persist an atomic successful per-shard
audit receipt bound to the audit implementation/schema, frozen runtime/config,
prompt bytes/count/split, capture manifest and every owned payload hash, including
request/response provenance. On a later explicitly coordinated resume, rehash the
current bytes and verify exact inventory/ancestry against that receipt before
reusing the completed deterministic derivation result. Missing/changed/unreadable
evidence must fall back to the full audit; changed gates or semantics invalidate
receipts. Do not merely trust manifest existence or a cursor. CPU fixtures must
prove unchanged results and fail-closed handling of altered payloads, identities,
inventory and interrupted receipt publication; a bounded same-data timing check
must establish whether avoiding reconstruction actually saves time after hashing.
This requires review of the validation-equivalence tradeoff before adoption.

The current frozen preparation continues unchanged, with soleRTX5080ownership
and `--prepare-only`. No restart, audit bypass, source deployment, new experiment
or optimizer update is authorized by this proposal. The validation/training owner
still waits for full preparation and verified GPU release.


## Provider construction repeats full audits — 2026-10-02

Full capture is measured:353 manifests/10,000train/1,002development. At
12:59:39UTC the frozen checker flagged a stale readiness_complete heartbeat;
13:07:09UTC original child676/startticks85132 advanced198CPUticks and259MB of
rchar in2seconds while reading captured features. No failure/restart or GPU
release is established. Frozen7547d253 source plus the approved preparation-only
launcher82f0185a explains substantial additional work after capture assembly.

`StreamingNativeProvider.__init__` constructs/audits each of320 train children.
A8 and A1 provider construction repeats that twice. The launcher's coverage
iteration audits319 more children, reusing the first:320+320+319=959 additional
full audits. `NativeCaptureProvider` loads labels through `audit_native_labels`,
which hashes payloads, reconstructs accepted-prefix features and validates
response ancestry. No stage heartbeat is emitted during those loops. At the
previous observed39.1seconds/shard this projects10.39hours; it is not measured
provider-phase timing, remaining duration or a promised endpoint.

Extend the audit-receipt proposal above to reuse a verified immutable audit
result across provider construction and coverage within the same process.
Receipts must bind implementation/schema, exact inventory/payload bytes and
source/runtime/config/prompt/response ancestry; changed or incomplete evidence
falls back to a full audit. First establish CPU equivalence and fail-closed
changed-byte/identity/inventory tests, then measure actual same-data savings.
A later reviewed implementation should also publish honest bounded progress
heartbeats during full audits. No live source/gate bypass, timeout change,
restart, new training recipe or optimizer update is selected here. Current
frozen preparation continues unchanged; the user owns adoption of this proposal.

## Selected: matched A8 recovery comparison after the first 1,000 paired steps — October 3, 2026

The bounded readiness/training handoff is complete for the admitted fixed A8/A1
recipe. The [actual report](../experiments/qat-optimization-readiness/first-1000-paired-steps-2026-10-03.md)
shows accepted drafts/round Q4_0=1.306255, A8=0.127287, A1=0.069031 on24 unsealed
development prompts; both binary arms matched Q4_0 response token IDs on all24.
No matched step-zero development control establishes the training effect.
Training covered13 prompts/4,846 rows from10,000 TRAIN/3,899,930 supervised rows.

The following options were pending after the completed pilot:

| Option | Evidence and work needed | Limit |
|---|---|---|
| Broaden training with the fixed recipe | Retain the intact step1000 checkpoint; define more data exposure and an explicit budget, positive-checkpoint resume and resource-safe evaluation lifecycle. | This small pilot cannot predict acceptance recovery; in-process evaluation RAM admission currently fails after CPU offload. |
| Qualify and compare an optimization/quantizer/refresh recipe | Use implemented controls with fresh exact full-source/provider/effective-model/native/backward/memory/timing admission for every enabled family. | Current reference receipts and native synthetic feature tests cannot admit optional full-model deployments or choose a quality winner. |

Q4_0 remains primary. Target/verifier precision, frozen data ancestry, cache/mask
semantics and sealed finals stay unchanged. RTX2080Ti and supporting research
remain paused. No new goal, calibration fit, trajectory recapture or optimizer
restart is implied by this decision record.

**Later human selection, October 3:** run a fresh matched A8-only reference and
combined learned-A8/all-nine-midpoint/lower-inertia-AdamW candidate comparison,
7200 cumulative trainer seconds per arm, with step-zero/scheduled/final native
development evaluations (1200 seconds each), resource-safe standalone evaluation,
exact resume and a bounded recovering monitor. A1, RTX2080Ti, Bop, curriculum,
trajectory refresh and architecture changes remain deferred. The historical
paired step1000 checkpoint is preserved, not reused as a new recipe's resume.
The standalone owner and executed state are in
[A8 QAT recovery and comparison](goals/a8-qat-recovery-and-comparison.md).
Target/verifier precision and sealed finals remain frozen. This selection
supersedes the preceding pending-budget statement for this bounded comparison.

## Pending: genuine W1A8/W1A1 DSpark/DFlash coverage — October 4, 2026

The human resumed the independent RTX2080Ti study and requested Q4, A8 and A1.
A8/A1 mean binary weights with real native activation packing. Existing DSpark/
DFlash has no W1 exporter, loader or FFN graph path; setting an activation flag
alone leaves its weights floating-point. The released-reference and supported
FFN-only Q4 measurements proceed within the shared7200s timing allowance.

The concrete proposed extension covers all15 FFN gate/up/down matrices across
five layers, with I32 packed signs and F32 row scales, existing CUDA W1A8 INT8
or W1A1 XOR/POPCOUNT operations. Embedding, private full head, attention, feature
fusion, Markov factors, confidence/norms and FP16 target remain at their admitted
precision. It needs exporter metadata, loader and graph integration, but no new
kernel or training project. Fresh sign/mean-absolute-scale conversion may lose
acceptance; actual native dispatch and frozen development measurements must
establish behavior. No speed or quality claim follows from source support.

The user owns the choice between this bounded FFN extension and completing only
released-reference/Q4 for now. The question is pending in the study chat; no
response or implementation authorization is inferred from elapsed time.
Details: [precision admission](../experiments/dspark-sm75-20261003/precision-admission.md).


The independent supported screen is now complete ([results](../experiments/dspark-sm75-20261003/results.md)).
Released DFlash7/DSpark7 outperform primary Q4 EAGLE in every repeat and match
all144 primary outputs each. FFN-Q4 saves about512MiB and also beats its paired
anchor; cross-phase rate variation limits causal precision-speed conclusions.
The above W1 integration choice remains pending; no binary-weight benchmark,
training, new goal or extra budget was silently selected.

## Clarified: within-DSpark precision comparison — October 4, 2026

The human's requested comparison is DSpark Q4 versus DSpark W1A8 versus
DSpark W1A1, rather than leading with DSpark versus EAGLE. Completed Q4
coverage is only fifteen FFN gate/up/down matrices across five layers;
W1A8/W1A1 cells remain unimplemented and unmeasured. No existing BF16/Q4
result supplies those missing binary cells.

The recap chat asked whether to hold that fifteen-FFN coverage fixed across
all three formats (recommended controlled scope) or include all eligible
drafter matrices, including fusion, attention and the full head. That scope
answer is pending; no new goal, remote operation, training or budget started
by this clarification. Current project-wide Q4 EAGLE comparison policy and
previously frozen measurements are preserved.

## Proposed research selection for nine trained and baseline drafters — October 4, 2026

The human confirmed separate EAGLE and DSpark/DFlash training tracks, each with
frozen Q4 and fusion-calibrated long-QAT W1A8/W1A1 candidates: nine final models.
They requested a deep backlog selection reviewed through peer-review MCP with
Opus 5.5 rather than Fable. Two focused calls with claude-opus-5-5-high completed.
The first broad max-reasoning call timed out and contributes no feedback.

[Reviewed recommendation](../experiments/nine-model-qat-research-plan-2026-10-04.md)
records evidence, selected development probes, conditional optimizations, deferred
branches and qualifications to reviewer suggestions. Recommend architecture-specific
native binary deployment/captures, all-domain fusion initialization at actual
A8/A1 arithmetic, meaningful balanced training coverage, direct-A1 versus A8-to-A1,
learned A1 thresholds, depth weighting, one own-prefix refresh probe and isolated
optimizer controls. Confirm the best combination against the best singleton and
fixed control. Floating fusion residuals/affine centers and head quantization are
conditional deployment experiments with explicit coverage and cost.

Use common token-exposure learning curves plus total lineage/capture/search
compute, expanded selection and independent confirmation prompts, seed-variance
checks and native request timing. The previous fixed-A8 development curve flattened
around 25k–35k updates; more hours alone are not demonstrated to recover Q4 quality.
Original Q4 controls remain frozen. Additional matched-coverage/domain-adaptation
controls are development diagnostics outside the final nine. No architecture or
precision attribution follows solely from a best-model deployment table.

The human owns final quantized tensor coverage, floating exceptions and total
compute allocation. No new active goal, schedule, model implementation, GPU/remote
action or training budget was started by this recommendation. The completed
A8 goal and original experiment reports are unchanged.

## Additional nine model research reviewed — October 4, 2026

The human requested exactly eight agents: four primary-literature researchers
and four reasoning-first advisors using intermittent primary-source validation.
All eight completed, with reporting-only writes to uniquely assigned files.
Five successful peer-review MCP calls requested claude-opus-5-5-high (Opus 5.5):
four audits covered all complete reports, then one reviewed root reconciliation.
[Reports and synthesis](../experiments/nine-model-research-slate-2026-10-04/synthesis.md)
and the [updated plan](../experiments/nine-model-qat-research-plan-2026-10-04.md)
record exact evidence, applicability, costs and every disposition.

Recommend small training bookkeeping work, paired-only immutable upload reuse,
profile-gated status cadence/fused FP32 AdamW/grouped attention, static DSpark
confidence suppression with Q4 controls, and cheap output assembly only if
measured. Add zero-scale row orientation rescue and post-norm/raw scoring to
the existing fusion screen. Saved authenticated A8 metadata confirms all 383
zero-scale control rows have negative fitting correlation; this is not proof
of permanently dead QAT rows or improved native acceptance. Fresh initializer
rescue needs no existing optimizer moment migration.

Additional gated learning options are localized A8 norm gains and a short
attached EAGLE context tail. Refine existing refresh with separately identified
old/fresh whole-round replay, loss denominator and Markov predecessor/censoring
contracts. Data filtering, EWGS/dampening, full VAT and floating-format dispatch
remain conditional/deferred rather than new default long runs. Preserve the
existing bounded exploration allowance and nine final artifacts.

Peer qualifications are explicit: recipe-audit writes are probe-gated; live
status is per update. Scale offsets can receive gradient at zero. Small EWGS
multipliers are not mathematically inert. Reviewer timing estimates are not
measurements. Per-row validation selection requires separate confirmation.
Training-only optimizations have no frozen-Q4 training counterpart; eligible
native runtime changes need matching Q4 evaluation.

No model implementation, training, GPU/remote action, monitoring or new goal
was started. Final tensor scope and compute allocation remain human decisions.

## Selected: independent nine model preparation team — October 4, 2026

The human requested a separate independent Codex task/agent team to prepare and
test the reviewed nine-model program, ready for prompt QAT launch and automatic
post-training evaluation. RTX2080Ti is permitted for development/CUDA preflight
when opened; RTX5080 is unavailable and reserved for training/evaluation later.
Parent locally paused new RTX5080 runs without contacting the host.

The [preparation objective](goals/nine-model-qat-preparation.md) defines ownership,
work packages, captures/fusion/trainer/native pipeline integration, independent
QA, hardware-specific evidence and the launch boundary. Native target/teacher
ancestry, frozen Q4 controls and sealed finals remain protected. Software and
SM75 readiness must be distinguished from fresh SM120 admission; no claim of
literal 100% or bypass of source/resource checks is authorized. Major final
coverage/recipe/budget choices remain human-owned while preparation proceeds.

## Pending: nine-model prepared coverage and compute — October 4, 2026

Coordinator `01a10903-1c7a-71b1-abb1-0de3ecc046b8` is implementing the authorized
foundation. No real-model optimizer updates, quality/performance evaluation or
RTX5080 action are started. The preparation team can exercise CPU fixtures and,
once opened, bounded RTX2080Ti CUDA/real-model zero-update checks and TRAIN data
capture. The human still selects final coverage, recipes and campaign budget.

Concrete coverage options being prepared:

- EAGLE: existing all-nine binary projection profile with deployed A8/A1
  arithmetic; explicit original floating exceptions/target bindings.
- DSpark/DFlash: exactly fifteen FFN binary projections, retaining original
  fusion, private head/embedding, attention and Markov/confidence; alternatively
  fifteen FFN plus binary fusion with calibrated A8/A1 arithmetic. The latter
  adds a precision change beyond historical FFN-only Q4 controls and must be
  labeled as a deployment comparison. Historical controls are never redefined.
- Full block attention, private head/embedding and Markov quantization remain
  excluded from initial implementation unless separately admitted.

Fixed A8 reference remains the default software profile. Direct A1 and A8→A1
with explicit optimizer state/reset policy are preparation profiles, not a
selected winning long-run recipe. Learned thresholds/other probes remain
explicitly off by default until independently admitted. The unsuccessful
learned-A8/midpoint/low-inertia combination is not inherited as the default.

Cost evidence currently supports only EAGLE projections: prior fixed-A8 arm
processed206163 rows in7200 accounted seconds (~28.6rows/s); a3,899,930row
pass extrapolates to37.8accounted hours, conditional on the old workload and
crash charging. Block training/capture rates and new balanced schedule cost
are unmeasured and will be costed using bounded preparation observations, not
borrowed EAGLE rates. Exact target-only five-tap/full-vocabulary block TRAIN
captures are missing; capture path/materialization and producer hardware
portability are readiness dependencies.

Both released block models use the frozen author-anchor-first driver
(`draft-dspark`, first prediction slot0, seven bidirectional noise rows), even
rank-zero DFlash. Family name must not infer generic DFlash slot1 semantics.
Static confidence/runtime options need matching controls if measured later.

Next decision presentation will bind actual implemented profiles, capture
size/cost and balanced row/epoch milestones with wall/evaluation allowances.
No unbounded budget, automatic recipe promotion or sealed-final tuning.


## Nine-model preparation evidence for the next decision

Current hardware boundary is Mac only following **“pause all 5080 usage
continue mac only.”** The briefly authorized RTX5080 interval completed locked
dependency setup; no compiler probe/model/capture/training/evaluation ran. All
remote work stays paused. No long recipe or allocation is selected by the
following evidence.

Implemented block coverage alternatives are exactly `ffn15` and
`ffn15_fusion` (sixteen matrices including FC), with full private head/embedding,
attention, norms and DSpark Markov retained. Actual FC16 CPU native graphs and
zero-update Torch forward/backward now pass for both families and A8/A1;
FFN15-only full-source native checks are being completed separately. These use
untrained original-weight prototypes and synthetic operands. They do not select
a winning coverage or predict CUDA memory/acceptance. Historical Q4 controls
remain unchanged and are still unavailable locally for the two block families.

Calibration default preserves actual reference magnitudes: EAGLE±0.5; block
original source magnitudes, with fitted signs/scales overlaid explicitly.
Unit±1 is an off-default probe because it changes distance to sign flips under
AdamW. Row-orientation rescue improved saved EAGLE raw fit but worsened postnorm
validation for both precisions, so it is off by default. Actual original block
FC/gamma/F32-epsilon references are now available; authentic five-tap TRAIN
capture is the remaining fitting dependency. Historical EAGLE validation has
no code-domain rows and cannot establish balanced final calibration.

The bounded preparation pilot is concrete:9 prompt-disjoint original TRAIN
records,3 domains×3 roles(train/calibration-fit/calibration-validation), plus
6 separate golden captures, native prompt≤512/new tokens≤32/chain≤544.
Full-vocabulary F32 teacher upper bound is3,226,189,824 bytes for the9 block
chains plus137,339,904 bytes for6 goldens, with total retained cap8GiB. This is
a development artifact budget only. It is not enough evidence to call a serious
long-QAT campaign prepared. Actual native capture and updates-per-second rates
are unmeasured; the previous EAGLE28.6rows/s extrapolation above remains merely
conditional. The7.29–8.47second Mac synthetic forward/backward observations
cannot be used as a real CUDA training-cost estimate.

Human decisions to resolve after artifact/hardware preparation:

| Decision | Prepared alternatives | Evidence still needed |
|---|---|---|
| Block deployment coverage | FFN15 with original fusion, or FFN15+binary calibrated FC | Native calibrated trajectory checks and a fair within-family original-Q4 comparison |
| A1 training path | Direct fixed A1, or charged A8 warm phase followed by explicit reset and positive A1 exposure | Short approved real-QAT comparison; neither source profile is selected as superior |
| Objective/storage | Captured-prefix hard CE; DSpark full-vocabulary CE+probability L1 with exact-prefix teacher support | Authentic capture rates, storage and bounded native prefix mismatch diagnostics |
| Exposure/accounting | Equal supervised-token/whole-chain milestones with domain-balanced cursors; record actual source tokens and per-stage updates | User-selected tokens/epochs, optimizer wall caps and evaluation allowance for all6 candidates |

No budget is inferred from successful tests. Optional fused AdamW/logging cadence
are implemented and gated, with no CUDA speed claim. Evaluation protocol remains
proposed until its allowance is selected. The prepared launcher automatically
sequences admitted training/export/resource-return/fresh evaluation once the
human chooses those inputs and reopens the host; no further source integration
or redundant permission is intended at that boundary.


## Delegated RTX5080 overnight execution — October4,23:41PDT

Human explicitly reopened5080, authorized relevantQAT afterpreflight and
requested30-minute monitoring/repair withhealthyQATcontinuingthroughmorning.
This replaces preparation-only/Mac-only limits. Coordinator records operational
choices under this delegation: calibratedfixedA8/directA1, blockffn15_fusion,
probesOFF/reference/private precision retained; firstproductionadmittedA8lane
preferredEAGLE, thenremaininglanes, serializedGPU. Firstlane24h cumulative
trainerallocation86400seconds, no smaller exposurestop, checkpoint250updates
initially withactualintervalcheck. Numbers are agent-selectedoperationalsettings,
not directhumanexact numericchoices or evidenceofwinningrecipe. Fullobjective
remains6candidates+3originalQ4; budgetmetadata must preserveauthorizationprovenance.

Stagedlaneadmission preserves allseven source/resource/kernel/model/backward/
memory/portability gates; unavailablelaterfamilies remainPENDING. Finalquality
results/trainedexports do not gatefirstlaunch. Seriouseligibledata/calibration
andfreshhardwareproof remaintrue prerequisites; oldlocalEAGLEdiagnostic and
blocknine-chainpilot do not becomeproductionready by goahead. Seeactivegoal.

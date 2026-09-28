# Decision log

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

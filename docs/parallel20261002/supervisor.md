# Parallel QAT support supervisor

Human-authorized launch on October 2, 2026, at about 18:31 UTC. This supports
the existing QAT optimization readiness goal; it does not start a new goal.
Supervisor chat: `01a0fddd-d4fd-7e70-8720-528a4e23006e`.

## Usage policy and persistence

Initial verified Codex weekly window: 78% used / 22% remaining,
`windowDurationMins=10080`, `resetsAt=1791049896`. The next observation at
18:33:24 UTC was 79% used / 21% remaining, ordinary usage allowed, same reset.
The reset currently corresponds to October 3, 10:51:36 a.m. Pacific.
Credits were available; no credit purchase or reset redemption is authorized.

Luna monitor `/root/usage_monitor` checks every five minutes. The app confirmed
creation of the ACTIVE five-minute heartbeat `parallel-research-usage-control`,
targeting the supervisor chat. The old `overnight-research-usage-control` was
deleted; an attempted update explicitly returned nonexistent. This is the sole
research heartbeat for this launch. Runtime state and actual observations are
in ignored `runs/parallel20261002/control.json` and `usage.jsonl`.

At remaining <=1%, disallowed ordinary allowance, or the known original-window
reset boundary, latch research stop before stopping/checkpointing every research
leader and descendant. Preserve all work and clean only owned CPU jobs. While
above 1% in the original window, after all teams finish, prompt Astra for a new
evidence-backed slate and launch one bounded team per task. Missing usage is
unknown, not zero. A reset-time correction alone does not establish a reset;
a clear usage drop or rollover with refreshed allowance does. On verified reset,
stop research and its usage monitor and pause the research heartbeat permanently.

QAT, its required preparation/GPU owners, and protected QAT supervision continue
through the research stop and reset, including existing paid credits if needed.
An agent cannot guarantee uninterrupted GPU computation or credit-funded restart;
the protected owner must verify stalls, gates, and actual continuation. Research
must never consume paid credits or the refreshed weekly allowance.

## Agents and ownership

QAT coordination: `/root/qat_priority`, Sol high. Its checkpoint is
[qat-priority.md](qat-priority.md). Existing owners were archived and their
heartbeats deleted. The agent unarchived/woke QAT owner
`01a0fc3d-bbe1-7e93-a19b-a9200dfa186c`, preparation successor
`01a0fdd6-8e11-7393-9aed-5c99bd08e428`, and diagnosis owner
`01a0fb34-e010-7b11-ac9a-f72cf2367c6c`. Replacement ACTIVE 15-minute
heartbeats were verified: `qat-validation-and-training-handoff` and
`a8-a1-luna-health-and-recovery`, targeting their respective owners. Preparation
owns its registration update. No new GPU operator was launched by this supervisor.
The LOCAL198 transport diagnosis was explicitly closed and handed off at
18:35:29 UTC; raw proof SHA256
`6711fe18e9de9992a28113f4ea135f07bae9dd033c6de20aff98322d29198c8f`
was locally verified, with lease diagnosis/operator inactive. No successful SSH
or remote checker followed from the diagnosis. The sole preparation operator
now owns a fresh bounded connection/check before any guarded transaction. Historical full captures and zero updates are not fresh
remote health or GPU release evidence. CUDA retest/training remains unverified.

Research advisor: `/root/astra_research`, Astra medium. Its four initial packets
are in [astra-research.md](astra-research.md). Each Sol high leader owns its
isolated temporary worktree and a bounded independent Luna validator. File
ownership is limited to matching `research/parallel20261002/<slug>/` and
`experiments/parallel20261002/<slug>/` trees; no live source or recipe mutation.

| Leader | Worktree | Branch | Direct contribution |
|---|---|---|---|
| `/root/lsq_batching` | `/private/tmp/eagle-parallel-20261002/lsq-batching` | `research/20261002-lsq-batching` | Prove learned clip/threshold gradient equivalence across actual head batching paths |
| `/root/curriculum_transition` | `/private/tmp/eagle-parallel-20261002/curriculum-transition` | `research/20261002-curriculum-transition` | Establish precision-transition reset and crash-resume state contract |
| `/root/recurrent_vjp` | `/private/tmp/eagle-parallel-20261002/recurrent-vjp` | `research/20261002-recurrent-vjp` | Independently verify composed cache/provider gradients and mask/truncation semantics |
| `/root/sign_inertia` | `/private/tmp/eagle-parallel-20261002/sign-inertia` | `research/20261002-sign-inertia` | Distinguish sign inertia/chatter and representational limits using exact synthetic fixtures |

The monitor registry records descendants and their exact ownership. All four
teams and validators were registered active. CPU synthetic evidence must not be
reported as native CUDA acceptance or throughput; Q4_0 EAGLE remains the primary
comparison baseline. No SSH, GPU/Metal, sealed finals, or real-data optimizer
steps for research. Root reviews/tests/integrates coherent branches into main,
pushes, and removes only fully merged preserved worktrees. Never drop unmerged
work at a budget stop.

## Next actions

1. Protected QAT owner resolves transport ownership and advances real preflight
   and training only when existing gates permit.
2. Teams produce independently validated, source-bound actionable reports.
3. Monitor enforces usage policy; root integrates completed work and launches
   subsequent Astra packets only while original allowance permits.
4. On a heartbeat, inspect actual agents and durable records before recreating
   anything; idle between healthy owner turns is not a failed GPU process.


## First research findings — October 2, 18:39 UTC

Fresh root usage:80% used/20% remaining, original weekly reset unchanged,
ordinary usage permitted. Four leaders remain active; no refill is due. LSQ
independent validator completed362935f; other independent validation is pending.
Astra is doing a focused challenge of LSQ normalization/remedy semantics, rather
than starting another task slate. Monitor retains control/log ownership.

Preliminary owner reports, not integrated production changes:

- LSQ team reproduces the real head path across72 synthetic cases: logits/loss
  match, but serial:batched learned head clip/threshold gradients differ by
  sqrt(valid depth); synthetic SGD step differs up to0.002749. Shared chain
  normalization restores equivalence while changing serial semantics; a serial
  learned-head fallback is a candidate preserving those semantics. QAT owner
  was informed to assess actual pending recipe exposure before training.
- Recurrent team reports216 composed CPU VJP comparisons passing, two synthetic
  AdamW steps/resume matching serial, and a stale-cache negative control shifting
  logits0.03090015. No production defect found so far; independent tests pending.
- Curriculum team confirms activation reset/fresh optimizer is explicit policy,
  retained binary/scale/correction/midpoint state is exact, and resumes accept
  malformed Adam payloads in synthetic probes. Authentic transition crash replay
  and trajectory consequences remain under independent validation.
- Sign team distinguishes reachable sign patterns from infeasible XOR: existing
  control variants can stall64steps at master magnitude.5; magnitude.02 reaches
  a useful flip around20steps. Infeasible XOR can flip/chatter without useful
  decision improvement. These are tiny CPU fixtures, not a selected live recipe.

All teams keep source and live recipes untouched. Final source-bound reports,
independent checks and commits precede root integration. CUDA readiness, actual
acceptance and Q4_0 native throughput still require their existing GPU gates.


## First slate integrated; second slate launched — October 2, 18:46 UTC

All first-slate owners/validators completed. Root reviewed findings, integrated
all four source-bound CPU reports/fixtures, and ran the combined new suite:
30/30 tests passed on Apple arm64 CPU/PyTorch2.14; diff checks pass. Worker
commits map to main: LSQe61cf9e→0576fd5 and349ad51→949b037, curriculum77d90fe
→4904f80, recurrent4fa1357→d9c1e55, sign ee01400→7af7630. LSQ's goal append
conflict was resolved retaining current QAT-owner notes and all incoming evidence.
QAT doc40253ab→c694dce,9afc496→ca02272.

The current QAT owner's model_gate_plan is the sole provider/source correction
owner. Protected coordinator qat_priority owns only explicitly partitioned LSQ
regression adaptation/review. No competing provider fix is allowed. Frozen
preparation config remains fixed-activation/head batching disabled and unaffected.
Pending learned-activations/combined profiles need corrected reference-equivalence
and honest effective fallback metadata before admission.

Latest actual bounded preflight connection at18:40:51UTC: SSH255, no remote
checker/staging/hold/CUDA; remote health UNKNOWN. Raw proof/cleanup in
[protected checkpoint](qat-priority.md). Transport and keeper closed, lease
operator inactive, conditional fixtureGOinactive. Root requested current5080
address/user/port asynchronously; no guess or duplicate retry. Protected
QAT/preparation heartbeats stayACTIVE.

First-slate ignored run folders were copied before cleanup to
`runs/parallel20261002/archived-worktrees/<worktree-slug>/`; sign/LSQ primary
raw copies also remain at their existing main-run paths. Each clean worker
branch's owned file contents were verified identical to integrated main.
Integration/cleanup receipt: ignored `first-slate-integration.json`. Retire
only after successful push; no unmerged work or raw artifacts may be lost.

Second slate: `/root/auxiliary_vjp`, `/root/export_function`,
`/root/refresh_contract`, each Sol high plus independent Luna validator, in
`/private/tmp/eagle-parallel-20261002/<auxiliary-vjp|export-function|refresh-contract>`
branches `research/20261002-<slug>`. Exact packets in updated Astra slate. Tasks
close combined auxiliary gradients, serialized GGUF composed function, and
refresh metric/admission/budget handoff gaps. Ownership follows matching
research/experiments topic directories and unique tests; no live source edits.
Fresh18:44:20usage81% used/19% remaining, same original reset; monitor registered
new teams. Refill only after all three teams/descendants finish and budget permits.


First-slate cleanup completed after verified push69c253c: all four clean
worktrees and branches retired, with owned contents verified equal to main and
ignored runs archived. Receipt records original tips/paths/retirement timestamps.
Second-slate active worktrees and protected QAT regression/source work remain
untouched. No experiment processes were active in retired worktrees.


## Second slate integration — October 2, 18:58 UTC

All second-slate leaders/validators completed. Root reviewed reports and
cherry-picked evidence: auxiliary51df26e/f99f03f/26f4e65→9b14ac7/78617ca/b606b50;
exported-function ed8d93b/446485b/e7afc14/b53da23→ac8451d/3eee242/a6f241f/92f6e10;
refresh6a373b9/8273e25→cfbf9dd/9e39ff1. Combined second-slate20/20 tests
pass on project Torch2.14/macOSarm64 CPU, diff checks pass. No core changes.
Full results and limitations remain in each report.

48 all-auxiliary gradient comparisons and three clipped steps pass;81 serialized
projection outputs exact in two independent decoders; refresh metric already
means accepted/proposed, no automatic learning-curve producer found. Proposed
count receipt/cumulative ledger stays a synthetic proposal granting no training.
Native and full-provider readiness remain separate.

Fresh18:54:58weekly82% used/18% remaining, same original reset and ordinary
allowance allowed. Astra now generates third slate: isolated curriculum resume
hardening prototype, source-bound training memory ledger, actual development
evaluator recipe closure. Launch after exact packets arrive; no repeated sweeps.

Protected learned-head correction67fe388 plus regression adaptationaf82c93
passed final immutable82/82 CPU tests; full hashes/readiness metadata reviewed.
QAT coordinator requests sole-root main integration acknowledgment from current
QAT owner before cherry-picking to prevent concurrent edits. No source adoption
or CUDA-ready claim yet. Pending host info remains unchanged; protected prep
heartbeat suppresses duplicate unchanged SSH while awaiting that reply.

All three clean second-slate worktrees owned content matches main; ignored
run folders copied to main archived-worktrees/<slug>. Receipt
second-slate-integration.json records original tips/paths before retirement.
Retire only after successful push; protected QAT worktrees are separate.


## Protected correction and third slate — October 2, 19:01 UTC

Current QAT owner explicitly deferred main integration to this root. Source
correction67fe388→59ef557, adapted auditaf82c93→f9a21f3, immutable validation
9f8b6db→45c41aa, protected reportea782f2→5b67aab integrated. All four final
source/test SHA256 values exactly match accepted82/82 immutable CPU evidence;
QAT owner separately confirms82/82 on its isolated combined branch. Main native
step source did not change during research integration; a6f241f modified only
the research GGUF decoder/report. No unchanged-source retest is needed.

Trainable learned head under gradients preserves serial invocation normalization;
fixed/frozen/no-grad supported batching remains enabled. Requested/effective
execution and saturation scope are explicit in readiness metadata. Historical
LSQ report/raw evidence/privatepatch unchanged; current tests assert preserved
gradients/updates. No frozen remote source/config or model job was changed.
New current-source CUDA/model/recipe/readiness receipts remain mandatory before
training; CPU passes are not native performance evidence.

Second-slate20/20 tests also pass against the corrected main provider, closing
new-report dependency integration. Pending host-address reply is now durably
registered by preparation owner, whose heartbeat remainsACTIVE/local observation
only with no duplicate unchanged SSH until that gate resolves.

Third slate launched all three: `/root/eval_recipe`, `/root/resume_validation`,
`/root/memory_ledger`, each Sol high+independent Luna. Worktrees under
/private/tmp/eagle-parallel-20261002/<eval-recipe|resume-validation|memory-ledger>,
branchesresearch/20261002-<slug>; exact packets in astra-third-slate.md. Owned
research/experiments topic dirs+unique tests only. Evaluator recipe construction,
staged semantic resume prototype, and source-bound peak-memory ledger target
actual unresolved gates; no live source modifications by research teams.

Monitor18:59:32usage83%used/17%remaining, original reset unchanged; no next slate
until all three teams/descendants complete. QAT and all descendants protected.


Public push3121623 verified. Second-slate clean worktrees/branches retired after
owned content equality and ignored-run preservation; second-slate receipt records
actual retirement. Protected QAT worktrees are retained for their owners' cleanup.
QAT owner acknowledges no production native_step change from researcha6f241f.

Preliminary third-slate risks, independent closure pending: evaluator default
fixed recipe rejects six modern named profiles across A8/A1 after native capture;
existing checkpoint_joint_config plus actual loader passes14 syntheticcases.
Memory ledger identifies host save transfer-temp overlap not charged in
finalpayload+16MiB allowance (largest F32head327,680,000B); exact peak/storage
calibration pending. Both findings relayed protected QAT owner, no unreviewed
source/floor/recipe changes or GPU proof.


At19:03UTC protected qat_priority retired its own two clean, fully integrated
report/regression worktrees after patch-equivalence/exact-eight-file checks and
raw preservation. Root verifies both worktrees/branches absent and before-fix
log present; receiptqat-worktree-cleanup.json. The protected agent continues
mainread-only coordination, peer QAT source/integration worktrees untouched.
Root usage19:03:23UTC84%used/16%remaining, original reset unchanged; third
three teams/validators allactive, no refill/stop.

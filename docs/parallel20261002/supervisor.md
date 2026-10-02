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
The LOCAL198 transport diagnosis requires explicit raw completion/handoff before
new remote dispatch. Historical full captures and zero updates are not fresh
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

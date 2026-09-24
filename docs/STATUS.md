# Current project status

**Active goal:** none. The completed [RTX 5080 W1A1 draft acceptance
goal](goals/rtx5080-draft-acceptance.md) and
`experiments/pytorch-w1a1-cuda-acceptance.md` contain the pinned CUDA setup,
strict BF16 parity diagnostic, full 12-prompt exploratory acceptance sweep,
and drafter layer-cost audit. Ordinary EAGLE accepted 2.317 drafts/round,
head-only W1A1 1.677, and all listed W1A1 groups 0.202 under the same
verifier. Strict target-only/ordinary parity failed at generated token 4 due
a target-selected BF16 verifier tie; the deeper logit shift is unresolved.
Candidate linears occupied 38.32% of one-prompt instrumented draft time. These
are verifier-relative acceptance and diagnostic timing results, not native
binary or end-to-end speed claims. All supervised runs ended, the GPU returned
to idle, and tmux SSH sessions were closed.

**Next decision:** the user chooses the next goal: investigate BF16 verifier
parity, narrow W1A1 coverage, or bounded drafter QAT. Evidence and tradeoffs
are in `docs/DECISIONS.md`. No QAT or native binary work is active.
The user also requested a follow-on INT4/INT8 accepted-per-round comparison;
its [measurement plan](../experiments/int4-int8-acceptance-plan.md) fixes the
same held-out CUDA setup and keeps the operand-precision interpretation
explicit. Implementation and measurement are underway without a new native
binary or throughput claim.
The prior [PyTorch W1A1 EAGLE
goal](goals/pytorch-w1a1-eagle.md) and
`experiments/pytorch-w1a1-metal-acceptance.md` contain the Metal development
evidence and its BF16 greedy-parity limitation.

**Repository:** initial llama.cpp scaffold and agent infrastructure are in place.
The published target/draft pair is the starting point; local conversion, remote
CUDA builds, and GPU throughput are not yet validated. See
`docs/PROJECT_OVERVIEW.md` for the overall research gates.

When a goal is active, link its `docs/goals/<slug>.md` here and summarize the
current stage, owner tasks, live remote jobs, next actions, and user decisions.

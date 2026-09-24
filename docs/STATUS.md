# Current project status

**Active goal:** [RTX 5080 W1A1 draft acceptance](goals/rtx5080-draft-acceptance.md).
The goal is paused at the user's request. The phase will confirm the
fixed held-out PyTorch acceptance sweep on CUDA,
check target/EAGLE greedy parity, and audit drafter layer cost before the user
chooses selective W1A1 coverage or bounded QAT. The current RTX 5080 address
and SSH username are recorded in the machine-local host registry, and the
shared pause flag blocks new 5080 runs.
The profiling tool is integrated on `main` and its CPU checks pass. The GPU
operator must wait for the user to resume GPU work before connecting; no
experiment job was started in this goal. The prior [PyTorch W1A1 EAGLE
goal](goals/pytorch-w1a1-eagle.md) and
`experiments/pytorch-w1a1-metal-acceptance.md` contain the Metal development
evidence and its BF16 greedy-parity limitation.

**Repository:** initial llama.cpp scaffold and agent infrastructure are in place.
The published target/draft pair is the starting point; local conversion, remote
CUDA builds, and GPU throughput are not yet validated. See
`docs/PROJECT_OVERVIEW.md` for the overall research gates.

When a goal is active, link its `docs/goals/<slug>.md` here and summarize the
current stage, owner tasks, live remote jobs, next actions, and user decisions.

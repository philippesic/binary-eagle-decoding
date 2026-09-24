# Current project status

**Active goal:** [RTX 5080 W1A1 draft acceptance](goals/rtx5080-draft-acceptance.md).
The user resumed the RTX 5080 work. The phase will confirm the
fixed held-out PyTorch acceptance sweep on CUDA,
check target/EAGLE greedy parity, and audit drafter layer cost before the user
chooses selective W1A1 coverage or bounded QAT. The current RTX 5080 address
and SSH username are recorded in the machine-local host registry, and the
shared pause flag is clear.
The profiling tool is integrated on `main` and its CPU checks pass. The Luna
operator completed CPU-only setup: the pinned CUDA Python environment and both
model snapshots are prepared and hash verified. All remote setup jobs stopped
and SSH sessions closed. Fortnite occupied the GPU at the last check, so no
CUDA experiment has started. The Luna operator is checking GPU idleness now;
it will run parity first if the device is free.
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

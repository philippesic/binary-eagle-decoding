# Current project status

**Active goal:** none. The completed [PyTorch W1A1 EAGLE
goal](goals/pytorch-w1a1-eagle.md) delivered the fake-binary drafter, selective
layer adapter, model loader, and held-out acceptance report. The 12-prompt,
five-configuration Metal development sweep is in
`experiments/pytorch-w1a1-metal-acceptance.md`: combined W1A1 accepted 0.202
draft nodes/round, versus 1.683 for head-only. Ordinary BF16 EAGLE and
target-only greedy generation diverged at token 21 on Metal; FP32 Metal matched
through 33 tokens. The locked `make check` gate passes all 15 tests. No
experiment jobs are running.

**Repository:** initial llama.cpp scaffold and agent infrastructure are in place.
The published target/draft pair is the starting point; local conversion, remote
CUDA builds, and GPU throughput are not yet validated. See
`docs/PROJECT_OVERVIEW.md` for the overall research gates.

**Proposed next goal:** RTX 5080 acceptance confirmation and drafter layer-cost
audit, followed by the user decision between selective W1A1 coverage and bounded
QAT. Starting it requires the current RTX 5080 IP or hostname, SSH
username/port, and working directory. Store them in the machine-local host
registry, not this file. Start a new goal only when the user requests it.

When a goal is active, link its `docs/goals/<slug>.md` here and summarize the
current stage, owner tasks, live remote jobs, next actions, and user decisions.

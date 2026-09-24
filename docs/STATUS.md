# Current project status

**Active goal:** [PyTorch W1A1 EAGLE drafter](goals/pytorch-w1a1-eagle.md).
The graph audit, fake-binary quantizer, selective adapter, and acceptance runner
are committed on `main`. The 12-prompt, five-configuration Metal development
sweep completed; see `experiments/pytorch-w1a1-metal-acceptance.md`. Combined
W1A1 accepted 0.202 draft nodes/round, versus 1.683 for head-only. Ordinary
BF16 EAGLE and target-only greedy generation diverged at token 21 on Metal;
FP32 Metal matched through 33 tokens. The next stage is RTX 5080 confirmation.
No remote jobs are running; RTX 5080 access details are pending. The target
stays unchanged.

**Repository:** initial llama.cpp scaffold and agent infrastructure are in place.
The published target/draft pair is the starting point; local conversion, remote
CUDA builds, and GPU throughput are not yet validated. See
`docs/PROJECT_OVERVIEW.md` for the overall research gates.

**Pending inputs:** current RTX 5080 IP or hostname, SSH username/port, and
working directory for model and acceptance runs. Store them in the machine-local
host registry, not this file.

When a goal is active, link its `docs/goals/<slug>.md` here and summarize the
current stage, owner tasks, live remote jobs, next actions, and user decisions.

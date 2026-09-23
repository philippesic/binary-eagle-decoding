# Current project status

**Active goal:** [PyTorch W1A1 EAGLE drafter](goals/pytorch-w1a1-eagle.md).
The current stage is graph/loading audit and fake-binary quantizer construction.
No remote jobs are running; RTX 5080 access details are pending. The next
actions are to dispatch independent audit/implementation work, pin model
snapshots, and build the held-out acceptance path. The target stays unchanged.

**Repository:** initial llama.cpp scaffold and agent infrastructure are in place.
The published target/draft pair is the starting point; local conversion, remote
CUDA builds, and GPU throughput are not yet validated. See
`docs/PROJECT_OVERVIEW.md` for the overall research gates.

**Pending inputs:** current RTX 5080 IP or hostname, SSH username/port, and
working directory for model and acceptance runs. Store them in the machine-local
host registry, not this file.

When a goal is active, link its `docs/goals/<slug>.md` here and summarize the
current stage, owner tasks, live remote jobs, next actions, and user decisions.

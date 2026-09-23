# Current project status

**Active goal:** none. Use `$start-goal-team` in a new Codex task to open the next
goal. The next research milestone proposed by the overview is the target-only
and FP16 EAGLE baseline on RTX 5080, followed by SM75-specific measurements on
RTX 2080 Ti when available. The user decides the next official goal.

**Repository:** initial llama.cpp scaffold and agent infrastructure are in place.
The target/draft pair, remote CUDA builds, and GPU throughput are not yet
validated. See `docs/PROJECT_OVERVIEW.md` for the overall research gates.

**Pending inputs:** current IP, SSH username/port, and working-directory details
for each WSL host when GPU work begins. Store them in the machine-local host
registry, not this file.

When a goal is active, link its `docs/goals/<slug>.md` here and summarize the
current stage, owner tasks, live remote jobs, next actions, and user decisions.

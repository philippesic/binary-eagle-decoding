# User and agent collaboration lessons

Record observed contention, cooperation difficulties, confusion, and concrete
user mistakes here so future agents can offer useful reminders and communicate
more clearly. Include agent mistakes or behavior that contributed to the issue.
Use specific evidence rather than assumptions about the user's personality or
intent. A disagreement alone does not establish a mistake.

Consult relevant entries when needed. These lessons provide context; the user's
current instructions take precedence. Correct or retire entries when new
evidence changes the lesson.

## Entry format

- Date and context:
- Observation and evidence:
- Status: confirmed mistake, disagreement, or unresolved uncertainty.
- Agent contribution:
- Practical lesson or reminder:
- Correction or resolution:

## Lessons

### 2026-10-02: Report execution phases and test producer contracts

- Observation and evidence: The user twice asked whether QAT was running. Optimizer updates remained zero while agents completed data validation and runtime work. An authorized handoff then stopped preparation09, but the controller rejected its brief intentional-stop status as “Recorded optimizer progress.” Actual classification showed an interrupted supervisor with exit_code=0, empty models, absent process groups and no new run.
- Status: Confirmed agent producer/consumer schema mismatch; no user mistake or observed optimizer activity.
- Agent contribution: The controller expected optimizer fields that the pinned pre-trainer signal handler omits. An earlier collector also expected returncode while remote_job writes exit_code. Mocked schemas did not cover these actual producer branches. Short admission windows also expired across agent/tool round trips.
- Practical lesson: State separately whether CPU validation, GPU model preparation or real optimizer training has executed. Test actual supervisor and signal-stop schemas, plus the real timeout/decoder ancestry, before changing process state. Mint fresh permission adjacent to execution after verified guards rather than spending it on coordination messages.
- Correction or resolution: Preserve original false/refusal receipts and raw evidence; accept only a narrowly joined producer-specific contract. Continue from the verified terminal boundary without restarting or re-signalling the old process. Actual optimizer progress remains required for a training claim.


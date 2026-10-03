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

### 2026-10-02: Explain audit progress and remove repeated work

- Date and context: The user asked whether QAT preflight was running and for an ETA after preparation recovery09 resumed.
- Observation and evidence: Reporting “2/353” prompted “The old agent was at like a couple hundred”; the user then asked what auditing meant and why reading files took so long. All 353 captures were retained. The counter described a restarted semantic verification pass. A fresh observation measured 33 shards in1289.174 seconds, about 39.07 seconds per shard, with 959 further semantic rereads in the unchanged provider path.
- Status: Confirmed agent reporting and implementation problems; no confirmed user mistake. The user's interpretation was understandable because the phase and retained capture count were omitted.
- Agent contribution: Recovery did not reuse completed audit results. The audits reconstructed labels/features from native traces and compared them, while progress reporting sounded like a simple file read. Coordination and launch-wrapper/schema errors added delays. Early answers used vague multi-hour estimates before measuring the rate and repeatedly described pending checks without a concrete resolution.
- Practical lesson or reminder: Report saved captures, current verification progress, GPU model preparation and actual optimizer updates separately. Explain the specific computation and its measured cost. Distinguish necessary first validation from avoidable repeated work; prioritize a bounded implementation that reuses authenticated successful results. Label extrapolations as workload estimates and identify unmeasured stages. User-facing updates should emphasize an actual result or evidenced blocker.
- Correction or resolution: Source/input-bound audit receipts, retained-corpus import and authenticated historical-pass adoption were implemented and tested. The ordered 353 metadata joins and later actual payload/configuration checks passed. These results permit reuse of the completed pass while preserving current integrity checks; they do not establish model readiness or optimizer training. Current execution status belongs in STATUS and the active goal, not this historical lesson.

### 2026-10-02: Finish bounded probes without assuming live cleanup targets

- Date and context: RTX5080 pipeline resumed after a WSL boot change; agents ran the existing 240-second CPU disconnect probe to verify survival before a long preparation job.
- Observation and evidence: The same processes survived the disconnected interval and heartbeats advanced44→206. Separate after-snapshot and cleanup tool steps crossed the probe's natural deadline. Cleanup attempted to inspect /proc/567 before a signal and found it absent. Final read-only state proved finishedexit2/time_limit, both groups empty and session absent; no signal was sent.
- Status: Confirmed agent lifecycle/coordination issue; no user mistake and no model or numerical failure.
- Agent contribution: The reused cleanup template assumed a still-live supervisor, while tool/report round trips consumed the remaining bounded lifetime. The root accepted that template without covering natural completion.
- Practical lesson or reminder: Combine the after-snapshot and exact-owned cleanup in one bounded operation when possible. Handle already-terminal natural completion by checking recorded state, summary, identities, groups and session; do not restart the probe or signal a reused PID. Preserve real exit codes and distinguish fixture limits from model success gates.
- Correction or resolution: A bounded read-only verification established complete return. The continuation joins the actual before/after/final receipts, accepts only the authenticated CPU probe time-limit contract, and still requires exit0 plus full readiness for preparation/model success.

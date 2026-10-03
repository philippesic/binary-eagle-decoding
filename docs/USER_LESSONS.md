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

### 2026-10-03: Use one accountable team and distinguish stage status from model progress

- Date and context: The human asked whether QAT was going after prolonged preflight, then directed one team to own preflight and QAT monitoring.
- Observation and evidence: Optimizer updates were zero. Multiple supervisors relayed status questions and performed overlapping source investigation. The live program reported models{} through provider/coverage and initial model loading; a readiness_complete stage marker did not mean final model readiness. Fresh output metadata established353 provider manifests completed.
- Status: Confirmed coordination and reporting confusion; no user mistake.
- Agent contribution: External supervisor intervention and this lead's repeated coordination added complexity. This lead initially treated an empty model list as proof construction had not begun; exact source showed it was a stale stage field until trainer smoke status.
- Practical lesson or reminder: Keep one QAT lead and its existing execution team. Report actual optimizer updates, dated process evidence, completed output markers and remaining gates. Do not infer an exact live call site or ETA from a stale stage field or CPU/I/O counters.
- Correction or resolution: External routine coordination stopped by human instruction. The single team verified the shared-GGUF manifest loop completed and prioritized bounded coverage/smoke markers before choosing any controlled operational change; current source and captures remain protected.

### 2026-10-03: Avoid false-positive transport guards during bounded diagnostics

- Date and context: The user requested a concrete change after prolonged preflight. The single team attempted a bounded existing-only profiler check to locate remaining startup work.
- Observation and evidence: Local guards first rejected a payload with one extra terminal newline, then searched for full path/PID text literals even though the wrapper constructed them from separate values. No remote command had been sent. After exact byte/parsed-value correction, the actual availability check found the profiler absent and stopped without a sample.
- Status: Confirmed agent guard/coordination mistakes; no user mistake, model failure or measured live hot path.
- Agent contribution: Ad hoc wrapper construction and textual assertions added local delay to a short diagnostic. Earlier runtime inspection also named copied metadata under a directory that was never staged.
- Practical lesson or reminder: Generate payload and command from the same saved bytes, validate parsed structured values, and verify copied references against actual staged placement. Preserve exact identity and content gates, but avoid testing incidental source formatting. Reuse the existing operator transport machinery and collect bounded failure logs with the original operation.
- Correction or resolution: Original failed guards and runtime descriptors are preserved. The diagnostic completed within its restricted remote scope and closed transport; Static05 corrects only three metadata references and carries its own bounded inner failure evidence. Missing profiler leaves the bottleneck unknown and does not justify a job restart.
- Follow-up evidence: When combining the next ordinary health check with three owned metadata reads, the prep supervisor changed the latest ordering flag to health-first but left the older lease scope saying metadata-first. The same operator rejected the contradiction before SSH. This lead's separate ordering clarifications also contributed to multiple authority representations. Propagate an ordering change atomically to every durable authority field, then reread it once before dispatch; keep one structured execution order instead of conflicting prose fields. No human error, remote action or live-job change occurred before correction.
- Root correction: This lead assumed the static runtime manifest record included a byte-size field and raised a local KeyError. The actual static producer emits path/SHA only; the transport separately records sizes. The same assumption prompted an unnecessary planner schema adjustment. Mandatory path/SHA checks remain intact and the optional-size check is harmless, but future checks must read the exact producer before asserting extra fields. No remote action or failed runtime gate followed from this local mistake.

### 2026-10-03: Separate submitters from wrappers and management supervisors

- Observation and evidence: The held-slot prototype treated GNU timeout as the CUDA submitter and required both preparation supervisor806 and child807 to be stopped. The owner caught the role mismatch before any signal; the supervisor must stay live for cancellation and cleanup. A timeboxed correction records the fixture process through same-PID exec and holds only GPU-capable producers. Root review also found that expiry/CANCEL or a missing operation file could block read-only cleanup after termination.
- Status: Confirmed agent implementation/coordination mistakes; no human error or actual GPU/hold action from the prototypes.
- Agent contribution: This lead's broad whole-producer wording did not distinguish management and GPU execution roles sufficiently. The worker reused wrapper identity and successful-operation assumptions instead of the actual executor and cancellation producer branches.
- Practical lesson: Specify process roles and exact owned groups up front. Record real kernel identity across exec, test natural fast completion and interrupted cleanup, preserve a live management supervisor, and separate expired execution authority from read-only cleanup evidence.
- Correction or resolution: Original hard-disabled prototypes are preserved. Seven focused CPU checks cover same-PID execution, live-manager/held-child roles, failure/missing-operation collection and expired/CANCEL cleanup. Actual hold/quiescence/context-return proof remains required before a native test.

### 2026-10-03: Keep terminal detection on the critical path

- Observation and evidence: Preparation ended successfully at08:57:52 UTC, but the next collected health observation detected it at09:39:55. During that interval agents developed a held-slot protocol that became unnecessary once the job finished. The human reiterated that QAT should run as soon as possible.
- Status: Confirmed supervision/priority delay; no user mistake and no failure of the completed preparation.
- Agent contribution: This lead added local ownership/controller work while following indirect owner snapshots. The preparation owner also spent time on that local work; the usual15-minute observation cadence did not detect the terminal transition promptly.
- Practical lesson: Preserve a short normal health/terminal observation at its cadence while local implementation proceeds. A successful terminal transition cancels obsolete hold work immediately and moves the same operator to release verification, remaining model gates and training.
- Correction or resolution: Actual success is checkpointed; held-slot work is stopped and preserved without any hold signal. The sole operator is redirected to final release and the reviewed released-mode native fixture, with no new preparation or redundant audit.

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

### 2026-10-04: Name exact manifest artifacts

- Context: Candidate A8 checkpoint15,000 was archived and natively evaluated.
- Evidence: The verified checkpoint-wide `manifest.json` hash is bdef1786..., while
  `A8/joint.json` is5f1f5c8a... and the resume checkpoint is50e0764c.... The operator
  briefly called an unlocated1e8c9381... value the manifest. Direct file checks
  found no matching step15,000 manifest, and the operator retracted that label.
- Agent contribution: The operator used an abbreviated artifact name without an
  exact path. Root initially considered different manifest scopes plausible, then
  kept that interpretation unproven and requested exact path/hash joins. Actual
  checkpoint/archive identity remained stable; no model corruption is shown.
- Practical lesson: Pair every integrity hash with its exact artifact path/type.
  Different hashes need an identified file or a correction, rather than an assumed
  serialization/scope explanation. Keep unlocated values out of evidence claims.

### 2026-10-04: Keep live observations monotonic

- Context: The sole A8 GPU operator supervised a reference boundary and evaluation.
- Evidence: Messages describing 08:29 preparation and 08:30 updates arrived after
  the 08:43 step15,000 boundary. The operator then called the old 08:30 snapshot
  latest. Its ledger phase/update/observation regressed to 08:29 while a nested
  training observation remained at 08:41. A fresh packet confirmed actual exit0
  at 08:43:32, checkpoint73429bd0 and budget2767.995; evaluation launched08:47:10.
- Agent contribution: The operator mixed older observations with current state;
  cached ledger updates may have contributed, but the write mechanism is not
  established. Root compared timestamps and exact process/checkpoint identities,
  requested a fresh packet and retained the existing pipeline. No duplicate
  training launch, source change or user mistake is shown.
- Later clarification at09:45: The operator now stores current state in top-level
  `latest_operator_observation`; legacy `training.status` and `last_remote_status_observation`
  may retain old snapshots. Root selected those legacy fields and had not yet
  advanced its own monitor registration, contributing to another stale-state
  report. Fresh process observations and the newer canonical field were current.
  Schema ambiguity, not a demonstrated concurrent GPU job, explains this episode.
- Practical lesson: Use observed_at and actual start/end times separately; delivery
  order does not establish freshness. Re-read the ledger before patching it and
  reject older snapshots as current state. On conflicting status, inspect the same
  kernel handles and authoritative supervisor/checkpoint/budget files before acting.

### 2026-10-04: Compare actual SSH key fingerprints

- Context: The sole RTX5080 operator reconnected to observe resumed A8 training.
- Evidence: SSH presented ED25519 fingerprint `SHA256:CFMXul7DWzihyD1Qr3ENOSZ7CUGJHzRh1tFjN0j4ZFI`.
  The Mac's existing trusted `known_hosts` entry has that exact fingerprint,
  verified by `ssh-keygen -F` followed by `ssh-keygen -lf`. The operator had
  compared it with a hashed host-label salt from the WSL client's old file.
- Agent contribution: The operator misread the host-label field and reported an
  identity mismatch. Root authenticated the existing public key locally; the
  operator acknowledged the error. No user mistake or host-key rotation is shown.
- Resolution: The same trusted public key is authorized for the actual tmux MCP
  client's known-host file, retaining strict host-key checking. Last observed
  training state stays stale until a fresh remote observation succeeds.
- Practical lesson: Identify the SSH client environment and compare fingerprints
  computed from key bytes. A hashed host-label field is not a key fingerprint;
  check existing trusted local anchors before requesting human clarification.

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

## October3: released validation delayed by local orchestration checks

After actual final preparation release09:46UTC, constructing the native wrapper
and two incorrect local evidence-path guards delayed validation. Parent ownership
resolved the guards; the first actual Linux attempt then refused an unreadable
/proc/367 FD table before CUDA launch. Root contributed by supplying a reviewed
fixture controller without a complete concrete transport/evidence wrapper and
by not testing the real process-permission boundary early. This is an agent
coordination/implementation issue, not a user mistake or numeric failure. Keep
one operator, give it one complete reviewed transaction, test permission/ancestry
census with actual evidence, and preserve unknown-actor denial without inventing
additional proof protocols or redoing completed audit work.

### 2026-10-03: Make bounded remote dispatch one-shot across inherited state

- Observation and evidence: The coordinator authorized one readonly root FD census. The sole operator executed the same command twice after overlooking the first completed query in inherited state. The second output replaced the first stdout while the first parsed record remained; different reader births and timestamps exposed the mismatch. Both actual results reported no contexts or unknown actors.
- Status: Confirmed agent repetition/provenance mistake; no human error or MCP defect established. No model, CUDA or signal action occurred.
- Agent contribution: The execution wrapper lacked an exclusive local dispatch claim; repeated tool invocation reused its output paths. Coordination initially relayed only the first result's hash.
- Practical lesson: Separate dry prechecks from a one-shot dispatch claim, use unique request/output names, and inspect saved command completion before reissuing a bounded operation. Never overwrite original raw evidence during parsing or retries.
- Correction or resolution: First bytes were recovered using the actual producer's exact serializer and matched the coordinator's earlier independent SHA; explicitly label recovery. Second original bytes/parse and both pane captures are separate. A local one-shot claim and fresh per-nonce readonly census outputs are required for the next controller transaction.

### 2026-10-03: Exercise the pinned loader with real argument types

- Observation and evidence: After all five actual full-source model gates passed, the training CPU-stage wrapper called exact6f load_config with a string. The producer calls Path.read_text; CPU staging failed before creating training output or GPU/optimizer execution.
- Status: Confirmed agent integration/test mistake; no user mistake, data problem or failed model gate.
- Agent contribution: Feature implementation passed the plan's string directly. Local16tests mocked the source loader contract, and root whole-code review missed this caller/producer mismatch despite reading the producer earlier.
- Practical lesson: Add one genuine pinned-API probe using the exact config/path types before execution; test successful real producer calls, not only invented/mocked schemas. For a narrow typed-API failure, preserve old packet/claim/raws and create a corrected immutable packet without repeating passed model/corpus work.
- Correction or resolution: Source/math/config/caps are unchanged; packet02 coerces config to Path and adds a genuine loader test. No extra human confirmation or numerical diagnostic is required.

### 2026-10-03: Preserve the agreed Q4 development boundary

- Context and evidence: The human requested QAT as soon as the GPU was free. After actual model gates passed, coordination briefly treated Q4 development as a required prelaunch report. The controlling training handoff instead retained Q4 development as existing training supervision and explicitly removed invented prelaunch report schemas.
- Status: Agent interpretation disagreement, resolved against the controlling handoff; no user mistake. TRAIN readiness probes do not establish development quality.
- Agent contribution: Older goal wording still referred to Q4 development launch gates, and the preparation owner repeated that ambiguous wording. The coordinator initially had to reconcile the stale wording with the newer explicit handoff.
- Practical lesson: Read the current agreed measurement boundary before adding a launch prerequisite. Authenticate the frozen development pool and baseline before launch, then collect the existing scheduled or serialized checkpoint evaluation. Preserve actual model/data/ownership gates without inventing a new proof schema or asking for redundant human confirmation.
- Correction or resolution: The same monitors were clarified, training ran for 1,000 paired updates, and the original development RAM-preflight failure remains recorded. Separate evaluation of its intact checkpoint is being prepared with unchanged memory and ownership checks.

### 2026-10-03: File extensions do not distinguish metadata from raw tensors

- Evidence: After the saved-checkpoint evaluation entered its terminal collection branch, the wrapper recursively included every JSON and JSONL file. A1 native heads.jsonl contains raw tensor data and exceeded the 16 MiB per-file collection cap, so collection exited before returning the actual report and terminal receipts.
- Status: Confirmed agent collector/review mistake; producer success or failure still needs its own original evidence. No user mistake or need to repeat evaluation follows.
- Agent contribution: The operator used an extension-based recursive collector, and root accepted it during whole-code review without restricting raw native tensor files.
- Practical lesson: Collect an explicit list of small status, report, identity and manifest files. Keep tensor/logit dumps remote, record their producer hashes where required, and label log tails as partial. A collector failure does not establish producer failure.
- Resolution: Original failed bytes and claim are preserved. The same operator is authorized to retrieve bounded explicit metadata and fresh resource-return evidence without rerunning the GPU evaluation.

### 2026-10-03: Separate implemented QAT features from the executed recipe

- Context: In the retrospective, the human asked whether the 1,000-step run used sign flipping instead of gradient descent, midpoint adjustment, and all implemented optimizations.
- Evidence: The original step1000 manifest records reference A1 computation, fixed activations, null binary_optimization/affine_weights/fusion_correction, and disabled cache/head optimization and persistent sign diagnostics. The baseline uses AdamW on floating latent signs with a hard-sign forward and surrogate backward; 398 A8 and 54 A1 cumulative sign flips do not establish a direct bit-flip optimizer. The recipes report explicitly deferred Bop.
- Status: Confirmed gap between implemented features and the executed recipe, with user/agent confusion about that distinction; no user mistake established. The fixed-reference numerical checks passed, but they do not validate the requested combined optimized recipe.
- Agent contribution: Describing the bounded reference handoff as goal completion made the broader implementation scope easy to confuse with actual training coverage. Reporting flip counts without naming the optimizer also left the mechanism unclear.
- Practical lesson: Before a training launch, state the effective enabled recipe in plain language, including latent-gradient versus direct-bit optimization, midpoint coverage, and performance switches. Report implementation completion, recipe validation, and training results separately. A reference pilot must not imply the combined optimized recipe was exercised.
- Retrospective timing: First paired update to step1000 was approximately 13 minutes 29 seconds; launch to terminal was approximately 15 minutes 20 seconds on RTX5080/SM120. Only 13 prompts and 4,846 supervised rows were consumed. No new run or budget is authorized by this clarification.

### 2026-10-03: Keep the requested QAT execution in a standalone chat

- Context: The human requested a QAT team and recovering monitor, then clarified
  that it must run as a separate Codex task with no cross involvement.
- Status: Confirmed coordination mismatch; no user mistake. This chat had
  interpreted the team request as inline subagents and attached a monitor here.
- Agent contribution: The coordinator started implementation inside the advisory
  chat despite the project preference for long work in an inspectable Codex task.
- Practical lesson: Once standalone ownership is requested, checkpoint and stop
  inline workers/monitor, dispatch a single independent chat, and cease observing
  or messaging it. Preserve unfinished work so the new owner can review it.
- Resolution: Four workers interrupted, original heartbeat paused, partial
  worktrees and readiness commit retained, standalone handoff recorded.

### 2026-10-03: Validate an execution flag on the selected precision

- Context: The standalone A8 comparison requested learned quantizers, all-nine
  midpoints and lower-inertia AdamW, while A1 remained held out.
- Evidence: The draft shared config used `a1_computation=single_forward`.
  Sole-operator source review found this also selects `_LearnedSingleForward`
  for a learned A8 module, whereas fixed A8 ignores that selector. Its name
  therefore hid a candidate-only gradient implementation difference.
- Status: Confirmed agent configuration/integration mistake, not a user error.
  The recipe owner corrected both arms to `a1_computation=reference`; requested
  quantizers, midpoint coverage, inertia and cache/head optimizations remain.
- Practical lesson: Audit the executed precision branch before calling a flag
  irrelevant. Preserve agreed gradient rules in matched comparisons and record
  deliberate effective-path exceptions explicitly.

## October 3: released block metadata does not establish runtime noise length

- Context: The human authorized a released DSpark/DeepSpec DFlash screen with short and trained-maximum proposals on RTX2080Ti.
- Evidence: Agents initially configured native proposal length three as the short reference. Focused Astra review found that the native driver also shrank the bidirectional noise block from seven slots to three, changing the conditioning of the first three predictions despite checkpoint metadata still declaring block seven.
- Status: Confirmed agent interpretation error, corrected before model execution; no user mistake or measured quality failure.
- Agent contribution: Root froze the initial short length before auditing its graph meaning; the native owner and advisor then exposed the distinction. A guarded reference path now computes all seven trained noise slots and proposes only the first three, with actual runtime counts to be checked.
- Practical lesson: Audit runtime input slots, masks and read slots for a short proposal arm. Checkpoint block-size metadata alone does not establish the released reference's conditioning. Preserve default runtime behavior outside the explicit reference mode and require same-prefix native proposal checks.


## October 3: shared pause record lacks request provenance

- Context: The human authorized an independent released-architecture study on RTX2080Ti and explicitly resumed that host for the study. Root resumed its shared flag at startup; the operator initially observed it unpaused.
- Evidence: Immediately before first model load, the flag was true with update time22:52:45UTC. The control schema stores only a boolean and timestamp, so the study cannot identify its writer or whether it reflects a newer human pause. No model or measurement was launched; completed CPU preparation is preserved and owned GPU contexts are absent.
- Status: Confirmed shared-state conflict; pause provenance and human intent remain uncertain. No user mistake or agent fault for the pause is asserted.
- Agent contribution: The team relied on initial resume during lengthy CPU preparation, then performed the required adjacent launch check. Root held the launch and asked the human for clarification without contacting the independent QAT/fusion owners.
- Clarification from the standalone A8 owner: its operator restored the RTX2080Ti flag after observing it false, under this task’s instruction that RTX2080Ti and unrelated research remain paused. This was a standing A8 handoff instruction, not a new human pause request. The separate study’s resume authorization was unknown to the A8 operator. The A8 team will stay off RTX2080Ti and avoid further flag writes based solely on its deferral, while a later direct human host decision controls global state.
- Practical lesson: Store host-control writer, reason and authorizing request when ownership changes. Recheck immediately before a GPU launch and honor an unexplained newer pause while clarifying its scope; preserve completed preparation so a resume does not repeat audits/builds.

### 2026-10-03: Validate the actual frozen prompt contract

- Context: The A8 comparison explicitly reuses prepared unsealed development data.
- Evidence: The first step-zero evaluation used the correct 24 prepared prompts
  (SHA131a3db...ba081), including original Magicoder/Dolly/GSM8K IDs, for acceptance.
  The new timing helper then rejected them because it required an unrelated
  `qat-revisit-development-*` prefix. No optimizer update occurred.
- Agent contribution: The helper introduced an unsupported naming assumption.
  Its synthetic fixtures shared that assumption; root reviewed timing/deadline/
  precision behavior but missed the mismatch with the actual frozen manifest.
- Resolution: Helper-only repair3bd4837 pins the exact prepared content hash and
  validates unique unsealed opaque IDs. Root16regressions pass; failed raw data,
  native receipts and checkpoint-zero are retained for bounded recovery.
- Practical lesson: Use the actual declared manifest/content identity when
  integrating an evaluator. Exercise its real ID styles in a contract fixture;
  naming conventions must not replace dataset-role and provenance checks.

## October 4: token ancestry does not establish identical numerical draft inputs

- Context: The independent RTX2080Ti study required seven computed noise rows
  for both three- and seven-token proposals. Its native checker also required
  equal first-three proposals at every matching token prefix.
- Evidence: Released BF16 passed. FFN-Q4 completed all25 native probe requests,
  with valid masks/cache/immutable target and complete outputs matching admitted
  paths, but four cross-length proposal joins differed. The recorded injection
  batches had different spans/counts (for example55..58 versus55..62) and only
  aggregate feature hashes. Token equality does not establish identical native
  features or fused draft KV across those different numerical histories.
- Agent contribution: Root approved the token-only key without separating this
  input assumption from the seven-row layout requirement. The implementation
  followed that rule. Luna preserved the failure and held timing; Astra advised
  a bounded numerical-history comparison before diagnosing a layout bug.
- Status: Confirmed insufficient comparison key, not a user mistake. The cause
  and size of later draft decision changes are not established as harmless
  rounding. Corrected admission passed with all five first-block histories and
  first-three proposals matching per architecture. Later unequal-history joins
  had2 DSpark/6 DFlash changes; equal-history disagreement remains a hard failure
  requiring a focused same-cache diagnostic. All original guards and raw failures remain preserved.
- Practical lesson: Compare recorded numerical input/cache ancestry when testing
  deterministic draft invariance. Check the actual seven-row graph and read slots
  independently. Count unequal-history decision changes explicitly; do not turn
  symbolic-prefix agreement into a bit-parity requirement or waive a genuine
  equal-input discrepancy. Use native acceptance/throughput to judge the variant.

## October 4: compare frozen protocol bytes separately from a serialized copy

- Context: Final source-bound CPU audit of a completed native diagnostic.
- Evidence: The frozen protocol's hash matched both run configurations. The
  benchmark writer saved equivalent JSON with sorted keys, producing a different
  byte hash. The audit incorrectly required the copied artifact hash to equal
  the frozen source hash and failed before checking raw logits.
- Agent contribution: The helper fixtures wrote identical serialization for
  source/copy, so they missed the actual shared writer's sort_keys behavior.
  Luna preserved the failure; root/source owner added an explicit frozen-source
  input, raw source hash check and complete semantic equality with the saved
  artifact. A fixture now uses the actual differing serializations.
- Resolution: Final CPU audit passes against original untouched artifacts;
  source/copy hashes both remain recorded. No GPU rerun or numeric threshold
  change. This was an agent provenance-check mistake, not a user error.
- Practical lesson: Bind immutable source bytes and bind copied artifact bytes
  independently. When a workflow intentionally serializes data, compare the
  full parsed values too and exercise the real writer in a contract fixture.

## October 3: separate CPU permission from another chat's GPU resume

- Context: Following the RTX5080 CPU/GPU pause, the human requested CPU work to
  resume in the independent fusion chat. The A8 owner separately recorded the
  human's GPU-resume instruction at 23:21 PDT.
- Evidence: Shared `gpu-control.json` was unpaused and the latest STATUS/goal
  checkpoint explicitly attributed it to “resume gpu usage”; this chat's new
  request said “resume cpu work”. Fusion calibration was already complete.
- Agent contribution: The fusion agent initially said GPU work would remain
  paused before checking the latest shared records, relying on its earlier
  chat-local pause state. It corrected the statement before any flag mutation.
- Resolution: CPU permission was recorded separately. Valid A8 GPU authorization,
  GPU-control flags, monitor, goal and checkpoint were left unchanged. No new
  experiment or GPU allocation was launched by the fusion team.
- Practical lesson: Check current shared authorization before describing host
  state. CPU-only permission does not grant this team GPU use or revoke a
  separately authorized GPU resume. Preserve other owners' current permissions.

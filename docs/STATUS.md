# Current project status

**Comparison target:** Q4_0 EAGLE is the baseline to beat for acceptance,
latency and total throughput. FP16 EAGLE is secondary diagnostic context.
The target/verifier model precision remains as frozen for each experiment.

## Human pause: owned RTX5080 usage stopped — October 3, 18:34 PDT

The human requested “pause5080gpuusage”. Candidate training stopped gracefully,
exit0, at2081updates/cursor2084; exact optimizer/RNG/cursor checkpoint and budget
are archived. Candidate used441.282of7200seconds; reference preserved/evaluated
5000updates/764.725seconds. All owned train/native/monitor groups and remote tmux
sessions are absent. Host residual3537MiB/21% is separate from the stopped A8 jobs.
SSH transport closure is being finalized. LocalRTX5080newruns are blocked;
existingheartbeat and Goal PAUSED. No new GPU/eval/recovery or remotework beyond
closure until explicithumanresume. Comparison remains incomplete;2080untouched.

## Active: A8 QAT recovery and comparison — October 3, 2026

**Standalone owner:** chat `01a103da-0980-7332-a041-3f95aca6a3f5` has claimed
exclusive ownership. The originating chat and its interrupted workers are
uninvolved. The SAME `a8-qat-recovery-monitor` is retargeted here and ACTIVE.
Three new bounded workers own trainer/evaluator/resume, recipe/movement audits,
and independent checks plus sole GPU operation; root reviews readiness and
integrates. See the [ownership checkpoint](goals/a8-qat-recovery-and-comparison.md#standalone-ownership-claimed--october-3-2026).
Integrated fixes are published (frozen execution source583480c; main8583b68).
BOTH actual A8 arms have optimizer updates and exact positive-step resume proof.
Reference is evaluated at5000updates/764.725seconds; candidate learned A8/all-nine
midpoints/latent0.1 AdamW is running at1200updates/245.81seconds. All33candidate
tensors show finite/nonzero gradients and actualmovement; cache/head execution
and learned-head serial exception are observed. Arms alternate at scheduled
5000boundaries toward independent7200second caps. Reference interim acceptance
7.7% and requestTPS60%ofQ4_0 remain below the primary baseline.
 The only active goal is
[A8 QAT recovery and comparison](goals/a8-qat-recovery-and-comparison.md).

RTX5080 is resumed for this A8 work; A1 is held out. The human separately resumed
RTX2080Ti for DSpark/DFlash; this team stays off that host and changes no shared
pause flag. Separate fusion CPU work is excluded from this comparison.
Reuse completed data and preserve the historical paired checkpoint. Initial
training budget is two cumulative hours per A8 control/candidate arm, plus
separately bounded step-zero/checkpoint/final evaluation. Candidate enables
learned A8 quantizers, all-nine midpoints and existing lower-inertia AdamW; both
arms request validated cache/head optimizations, with the deliberate learned-head
serial-training exception disclosed. A1/Bop/fusion/curriculum/refresh are deferred.

One 15-minute heartbeat supervises bounded recovery from committed checkpoints
without silent recipe/budget changes. Sole operator verified fresh idle RTX5080
and located retained preparation; no new actual model or training start yet.
Historical complete/stopped sections below describe the previous goal.

## All project agents stopped and archived — October 3, 18:18 UTC

The user requested: “Go ahead and stop and archive all agents.” Both active
QAT/preparation heartbeats were paused; supporting research monitoring was already
paused. Eight remaining project chats were archived, with the terminal preparation
chat to archive after this checkpoint. No other project agent is registered live.
Both GPU hosts are locally paused against new work. No remote query or job was
started: the last verified experiment groups, native contexts and operator
transport were already terminal and closed.

The QAT goal remains COMPLETE. Preserve the original 1,000-update checkpoint,
training exit1 after the RAM preflight, separate evaluation exit0 and below-Q4_0
result. Unmerged research worktrees and all experiment evidence remain intact.
No further monitoring or agent work is scheduled. See the
[shutdown checkpoint](goals/qat-optimization-readiness.md#user-stop-and-archive-all-agents--october-3-1818-utc).

## QAT optimization-readiness goal complete — October 3

The admitted fixed A8/A1 recipe has current RTX5080/SM120 native/model/backward/
memory/five-repeat and full-source save/resume evidence. Both models completed
1,000 optimizer updates and saved an intact paired checkpoint. Separate original
Q4_0development evaluation completedexit0: accepted drafts/roundQ4_0=1.306255,
A8=0.127287,A1=0.069031 on24 unsealed development prompts, all24 response token-ID
sequences matcheachbinaryarm toQ4. Thischeckpoint isfarbelow theprimarybaseline;
no accuracy/speed/serving-throughput win orglobaloptimizedrecipe readinessclaim.

Originalautomaticdevelopment hostRAMfailure afterstep1000 and rawtensorcollection
capfailure remainpreserved. Freshall-ownedgroups/nativecontexts/source/resource
return andoperator244/71024closureverified; no ownedexperiment remains. Actual
trainingused13prompts/4846rows from10000TRAIN/3899930rows. Optionalrecipes/A4/
curricula/calibration/refresh requiretheir owncurrentadmissions; longertraining
andresource-safe evaluator lifecycle aretheuser's nextresearchdecision. RTX2080Ti
and supportingresearchremainpaused; no new goal/budget/recipe selected.

See the [final report](../experiments/qat-optimization-readiness/first-1000-paired-steps-2026-10-03.md),
[completion checkpoint](goals/qat-optimization-readiness.md#final-verified-result-and-completion-record--october-3-1352-utc),
and pendingdecision inDECISIONS. Theboundedgoal iscomplete for theadmittedrecipe;
same monitorsretainterminalfacts withoutnewruns/deadqueries ornewbudget selection.

**14:01 UTC terminal supervision checkpoint:** completion audit/report publication
`f8668fa` verified locally; all owned jobs and the sole Luna operator remain
terminal, no remote action this tick. The preparation owner is checkpointing its
safe context rotation; current QAT coordinator remains `01a1014d-9673-7a31-8292-72f8748501f6`.
See the [terminal supervision handoff](goals/qat-optimization-readiness.md#terminal-supervision-rotation-checkpoint--october-3-1401-utc).
The SAME monitors retain quiet terminal facts; no new goal/run/budget is selected.

## RTX5080 QAT pipeline resumed — October 3, 06:11 UTC

The human said **“Gpu is free resume now.”** RTX5080 is resumed for the existing
QAT/preparation pipeline and necessary remote work. RTX2080Ti remains paused;
supporting research remains stopped. The human assigned all preflight/QAT
coordination to this single team; external supervisor intervention is stopped.
Latest direct human priority is **“Gpu is free I expect qat to run asap.”**
Start the already-authorized new training run as soon as current data/model/native
and ownership gates pass; no redundant confirmation or gate waiver.
The same QAT monitor is ACTIVE and bound to
acknowledged owner `01a1014d-9673-7a31-8292-72f8748501f6`; the same preparation
owner/operator retain sole remote control. No duplicate schedule or operator.

**Preparation completed successfully at08:57:52 UTC, observed09:39:55.**
Supervisor exited0;806/807 are absent. The checker reports healthy/terminal,
complete coverage of10,000 TRAIN prompts/3,899,930 supervised rows, paired A8/A1
step-zero models and checkpoint. Ready receipt SHA bdfa56f8 and checkpoint
e9d01984 are recorded in the goal. Optimizer updates remain zero. Exact terminal
collection and fresh final GPU-release proof passed at09:46 UTC through the sole
operator: both groups empty, no project workers/compute apps, matching source/
checkpoint and resource floors. The held-slot plan is obsolete and preserved;
no hold occurred.
Root issued conditional GO for one reviewed released-mode copied-fixture CUDA
test after verified release, then the remaining actual-model/native/backward/
memory/timing/save-resume gates and already-authorized new training run.

**10:24 UTC coordinator checkpoint:** the copied CUDA fixture has not launched.
Its released-mode guard refused unreadable /proc/367/fd/0 before slot/launch;
transport closed with exact evidence. Same preparation owner/sole Luna owns
bounded readonly diagnosis and narrow census correction. Original ready/smoke/
coverage/checkpoint metadata joins independently passed; actual paired backward
and memory evidence is retained. A local prepared-checkpoint launch helper is
being checkpointed for review, to avoid repeating the completed full audit.
After two compactions this coordinator is rotating at this safe boundary;
see [durable handoff](goals/qat-optimization-readiness.md#coordinator-rotation-checkpoint--october-3-1024-utc).
Training still has zero updates; native/current model/timing/save-resume gates
remain pending. No new user decision or confirmation is required.

Local launch-path worker finished: draftdd19f09,12focused CPU tests passed;
no actual-model integration or independent review yet. Its clean feature worktree
is preserved for the successor; original frozen preparation is untouched.

**Ownership transfer verified10:28UTC:** QAT successor
`01a1014d-9673-7a31-8292-72f8748501f6` acknowledged the durable checkpoint and
now owns registration plus the SAME existing ACTIVE15minute heartbeat. Prior
coordinator retires; same preparation owner and sole operator continue. No new
goal/operator/schedule, GPU interruption or human confirmation.

**Historical preparation progress:** the unique job launched at06:43UTC, supervisor806/
child807, after verified current-boot durability and fresh launch guards. Prep
owner `01a10084-101e-7311-94e9-9658f9dc648f` acknowledged the same job and active
heartbeat. Checks at06:51 and06:55 verified live kernel identities, retained353
manifests/10,000TRAIN/1,002development, active CPU/I/O, models{} and zero optimizer
updates. At07:21 the data-stage marker advanced to readiness_complete; metadata
at07:26 confirms all353 provider manifests and indexes exist. The repeated shared
GGUF manifest hash loop is finished. Model/coverage markers are being checked;
empty model records alone do not establish whether model loading has begun.
Model smoke/checkpoint-zero and full readiness remain pending. At08:24 the
checker reports only a stale-heartbeat failure; exact child807 remains active
with positive CPU/I/O progress. This warning is preserved, not waived. Both
active precision gates are legacy v1, so the modern NPZ validation branch is
inactive. The bounded diagnostic at08:01 found no installed profiler and
stopped before sampling; the exact remaining startup hot path is unmeasured.

The terminal continuation passed46 root/independent local tests. CPU static04
returned exit1 with all owned groups gone: its descriptor referenced three
metadata files under a never-staged directory. Static05 fixes only those paths
and passed35 local guards. Actual Static05 finished at08:13 UTC with exit0;
the08:14 collector verifies passed inventory, unchanged protected files and
empty helper groups. Local transport/keeper closure and all41 evidence file pins
are verified. All three exact inventory records are now read back and hash-joined
to the successful operation. This is CPU static proof; current-package CUDA and actual-model
gates remain pending. A bounded copied-fixture CUDA command plan is locally
reviewed and its actual inventory inputs are verified; fresh exclusive GPU
ownership is still required before dispatch. The08:42 eligibility snapshot found
all four completion markers absent and child807 holding two /dev/dxg handles.
Zero utilization and an empty WSL compute-app list do not prove context absence;
the native test remains inactive. The held-slot fixture controller is now
code-reviewed after seven focused CPU checks. Final preparation completion now
supersedes that hold plan; released-mode physical proof is required before native
dispatch. Final release is verified; the same operator stages the reviewed
fixture before fresh adjacent native admission. Standard prepared-run training
would repeat provider/coverage startup; a safe reuse path is being resolved.
A local source audit found repeated common GGUF hashing per shard;
a small cache fix for FUTURE starts passed80 independent CPU tests and31 root
checks. Current source6f/live preparation is unchanged. Old09 remains terminal,
its recovery budget is exhausted, and original evidence is preserved. RTX2080Ti
and research stay paused. The existing goal is incomplete; see the
[resume checkpoint](goals/qat-optimization-readiness.md#human-resume-rtx5080-qat-pipeline--october-3-0611-utc).

**Actual current native CUDA fixture PASSED at11:10 UTC on RTX5080/SM120.**
All22raw artifacts and the unchanged Python report validator pass;57pack/
114loader/59graph/218arithmetic/36projection cases. Owned groups/submitter/
contexts returned; protected source/config/runtime unchanged. LOCAL241/keeper
10523 are closed; same sole operator awaits a NEW source-bound model transaction.
This is synthetic scope; optimizer updates remain ZERO. Prepared adapter is
public/tested `d211275` (17guards plus genuine tiny CPU restore/real Git imports).
Actual model CPUstage now passed: original full10000TRAIN/3899930rows and zero
checkpoint are authenticated, generated binding4f794 pins clean helperd211/source6f.
Final13guard model packetac7f is frozen/reviewed; SAME Luna has NEW242 transport.
ACTUAL ALL5 model gate phases PASSED11:55:55 exit0; all164originals/currentreceipt
validator/group-context return checked.30forward30back/five repeats, finite/later
grads; native12caseszero changed choices, reserved9.11GB/minfree6.48GB. This is
TRAIN/readiness evidence, not heldoutacceptance or convergence. ACTUAL QAT reached1000pairedsteps and published checkpoint62f88/manifestb48c.
Originalautomaticdevelopment then FAILED hostRAMpreflight15.03GB; supervisor
finished12:41:37exit1/no signal, preservedstatusfailed and no Q4report. Bothlanes
finite18gradtensors, signflipsA8=398/A1=54; noquality/convergenceclaim. Exact
readonlyreconnect/all5originals verifies checkpoint/exports/source/runtime and
terminalgroupempty, jobprogressafterconnectionclosure306s (no sameLIVEbirth
postreconnectclaim). Actualfullrelease PASSED12:54:25: bothgroups/contexts gone,
host20.293GB/GPU13533MiBfree, all7originals+logtail provenance verified;
SAME soleLuna244/273/71024 retained for evalhandoff.
NEWserializedevalpacket02 fd5439 passedrootwholecode/20CPUtests/all14pins;
conditionalCPUstage sourceapproval issued for intact1000checkpoint+specificRAMfailure. No trainingrestart/source/caps/memorygate waiver. ACTUAL serializedevalCPUstage PASSED13:03:41: bothoriginals265285B, stagec1e3/
actualrunlatest98e7/source1002dev/failedterminal/final1000checkpoint joined.
FINAL EVAL GPU GO13:12 forreviewedcommandd5dc/wrappera8ef; ACTUAL separate
evaluation joblaunched by SAMEsoleLuna244, ownerGOb8ce33. Existing1200bound/
source6f/Q4baseline/math/memory gates unchanged. EvalCOLLECTORfailed16MiBcap on
rawheads.jsonl afterterminal-return branch; actualproducerexit/report/currentreturn
stillunverified. SAMEsoleLuna244 doesONE<=4MiB explicit READONLY metadata/fresh
return collection; no evalrerun/optimizer/newoperator. Goal incomplete.
No repeat corpus audit or unit fixture, source changes or new user confirmation.
See the [native milestone](goals/qat-optimization-readiness.md#actual-copied-native-cuda-fixture-passed--october-3-1110-utc).

## Human stop: local Mac only — October 3, 00:10 UTC

The human ordered: **“Stop gpu work. Only local Mac work allowed until I say so.
Tell all agents.”** This supersedes every earlier remote permission and approval.
Both host pause flags are true; QAT and preparation heartbeats are PAUSED.
All root agents were notified, and the main supervisor relayed the stop to its
teams. The sole existing preparation operator is restricted to shutdown and
verification, then disconnect. No new remote CPU/GPU job, transfer, polling,
recovery or SSH is permitted. The sole operator reports MCP223 and keeper55194
closed. Its final verification wrapper failed locally before SSH; no remote
query occurred and no retry is permitted. Last remote evidence confirms old09
stopped and no replacement launched; current GPU availability is unverified.
The local closure proof is saved and its SHA independently verified; the sole
operator lease is closed. See the linked goal for its path, hash and limitations.

Optimizer updates remain zero. Old09 was already stopped; the proposed new
preparation continuation was never approved and is cancelled. Local unfinished
work and all original evidence are preserved. The goal remains incomplete;
only local Mac work may continue. Acknowledged successor chat
`01a0ff1c-1007-75e3-a772-eaeb078179b3` owns the local checkpoint and has recorded
the existing operator’s final shutdown result. See the [local-only handoff](goals/qat-optimization-readiness.md#local-only-stop-and-successor-handoff--october-3-0010-utc).

## Old preparation stopped; continuation repair — October 2, 23:59 UTC

The authorized transfer stopped old09 cleanly at 23:51 UTC. Read-only
classification verifies its supervisor and child absent, both groups empty,
intentional STOP status, unchanged source/runtime/configuration and no new run.
The stopped-status producer omits optimizer fields; the transfer guard rejected
that brief status as progress. Root verified the exact producer branch and the
pre-stop zero state. There is no evidence of optimizer activity.

A terminal-specific continuation packet is being prepared. It will not signal,
restart or resume old09; it will require the preserved stop evidence, source/data
checks, fresh GPU/context/resource checks and the unique new preparation-only
launch. The exact timeout/decoder ancestry fix is tested. GPU release and new
model start remain unverified. [Transfer checkpoint](goals/qat-optimization-readiness.md#controlled-stop-classified--october-2-2359-utc).

## Runtime copy passed; GPU transfer cleared — October 2, 23:48 UTC

The isolated ten-file runtime copy passed. Static inspection failed only while
decoding non-UTF-8 bytes from a readelf metadata dump; the new package and all
original files are preserved. A lossless, metadata-only decoder fix passed 43
CPU tests and is under final review.

The GPU transfer packet passed all 21 tests under root and independent review.
Root authorized the sole operator to stop the exact old preparation process,
verify complete GPU release, and launch the new `6f1444b` preparation-only run
with the validated configuration. Actual transfer and new model start remain
pending. This does not permit optimizer updates. [Execution checkpoint](goals/qat-optimization-readiness.md#runtime-copy-success-and-gpu-transfer-go--october-2-2348-utc).

## Clean runtime fix public — October 2, 23:30 UTC

The loader packaging fix is integrated and pushed as `973d10c`, after 39 CPU
tests under root and independent review. It copies the complete ten-file runtime
into a new directory and changes only the verified search-path string slots to
`$ORIGIN`. Every other byte and original file remains unchanged. Strict source,
dependency, ELF and provenance checks remain enabled.

The six-guard-tested CPU copy/static packet is cleared for the sole operator.
Actual packaging and fresh packaged CUDA validation remain pending. The separate
GPU preparation transfer packet is undergoing focused review; new-source CPU
payload/configuration acceptance is complete. Optimizer updates remain zero.
[Runtime deployment checkpoint](goals/qat-optimization-readiness.md#clean-runtime-fix-integrated--october-2-2330-utc).

## CPU payload/configuration validation accepted — October 2, 23:14 UTC

The actual new-source CPU validation passed and its process groups returned.
Root independently verified the successful operation, supervisor `exit_code=0`,
all 28 source hashes, exact validated configuration and source preservation.
The original collector's false summary is preserved; it used the wrong exit-code
field. A separate bound acceptance records the actual success.

The new source and retained payloads are validated for preparation. Optimizer
QAT has not started. Next is controlled transfer from recovery09 to the new
preparation-only model run; its executable transfer packet is being completed.
Native runtime packaging is being fixed independently: seven ELF files have a
trailing empty search-path entry, and original binaries remain untouched.
[Accepted configuration and next execution](goals/qat-optimization-readiness.md#cpu-configuration-accepted--october-2-2314-utc).

## CPU payload validation running — October 2, 23:02 UTC

The new-source CPU job actually started at 22:59 UTC. Fetch, detached checkout,
vendor initialization and CPU runtime probe passed; payload/configuration
validation is running under child4221, with supervisor4108 and group4109.
Old recovery09 remains untouched. This verifies data and configuration, with
no model, GPU or optimizer execution.

Server04 exited1 because its first ELF artifact has an empty runtime search-path
entry. Collection verified its groups are gone and all 7 runtime / 14 frozen
files are unchanged. A bounded read-only ELF census is cleared to identify the
minimum packaging correction. Optimizer QAT remains unstarted, with zero updates.
[Actual validation checkpoint](goals/qat-optimization-readiness.md#cpu-configuration-bootstrap-and-payload-validation-running--october-2-2302-utc).

## CPU server inspection actually started — October 2, 22:54 UTC

Static-only server04 started at 22:53:07 UTC: supervisor3911, child3912 and
runner3914, under the reviewed 180-second inspection / 240-second supervisor
bounds. Result and complete process-group return remain pending. This is a CPU
inspection, with no model, GPU or optimizer execution.

Fresh recovery09 observation at 22:51 UTC verified audit153/353, zero updates,
source/runtime identities and about 19.47 GB host memory. The separately reviewed
new-source CPU validation packet is accepted next, after server04 return and fresh
coordination. Optimizer QAT has not started. [Actual start checkpoint](goals/qat-optimization-readiness.md#cpu-static04-actual-start--october-2-2254-utc).

## New-source CPU validation packet accepted — October 2, 22:47 UTC

The executable new-source staging/configuration packet passed 18 tests under
both root and independent review. Its 11 file hashes match. The sole operator
is authorized to run it after static server04 returns, with fresh resource and
ownership checks. It validates actual retained payloads on CPU before any old09
teardown; it does not construct models or permit optimizer updates.

Fresh server04 preflight at 22:42 UTC verified recovery09 at audit139/353, all
retained captures, zero updates, exact process/source/runtime identities and
19.44 GB available host memory. A new five-minute CPU admission was issued;
actual server04 start/result remains pending. QAT optimizer training has not
started. [Execution packet checkpoint](goals/qat-optimization-readiness.md#new-source-cpu-validation-packet-accepted--october-2-2247-utc).

## Historical full audit pass verified — October 2, 22:26 UTC

The actual read-only inventory matches all 353 manifests to the earlier completed
semantic pass, including both provider indexes and every audit sidecar. All
10,000 TRAIN and 1,002 development prompts remain intact. Root approved the
reviewed historical-audit path and generated source-bound inputs for a new
`6f1444b` preparation-only run. Current payload integrity checks remain required;
this does not establish final model readiness or permit optimizer updates.

The sole operator has completed and cleaned the inventory transport. It next
runs the accepted static-only server inspection, then stages and validates the
new source/configuration before controlled recovery09 teardown and transfer.
Frozen recovery09 remains untouched, last observed auditing 109/353 with zero
updates. Research remains stopped. [Actual evidence and execution](goals/qat-optimization-readiness.md#historical-full-pass-verified-and-new-inputs-bound--october-2-2226-utc).

## Audit reuse and runtime fix public — October 2, 22:17 UTC

Historical full-pass recognition is integrated as `6f1444b`, after 49 focused
CPU tests and independent review. It requires the exact archived supervisor08
source/launch/completion records, complete ordered readiness/index/sidecar joins,
unchanged semantic audit code, and current payload integrity verification before
publishing durable receipts. Live metadata must still match before enabling it.
The server timestamp fix is public `a345f87`, with 23 tests and independent review;
its six-guard-tested static-only rerun packet is cleared for the sole operator.

The first metadata query failed at the Windows-to-WSL command boundary before
Linux execution. A minimal transport wrapper correction is underway; this is not
a corpus or audit failure. Frozen recovery09 remains untouched, supporting research
stays stopped, and optimizer updates remain zero. [Execution checkpoint](goals/qat-optimization-readiness.md#audit-reuse-and-runtime-fix-public--october-2-2217-utc).

## Build succeeded; audit provenance check dispatched — October 2, 22:08 UTC

CPU server03 actually ran: configure/build succeeded, static inspection rejected
`compile_commands.json` freshness against the existing W1Ax object. Both owned
process groups are gone, all 7 runtime and 14 frozen files are unchanged. A narrow
proof-bound runtime inspection fix is in progress; no rebuild or timestamp edit.

Historical supervisor08 pre-launch records bind the clean frozen audit source,
and its health snapshot reached completed stage readiness. The sole operator has
conditional GO for one bounded metadata inventory to compare the saved readiness
record's ordered hashes with all 353 retained manifests. This may support reuse
of that completed semantic pass after current payload integrity verification.
No historical shortcut is enabled. The reviewed new-source prepare-only packet
is being completed; frozen recovery09 continues untouched and updates remain zero.
[Evidence checkpoint](goals/qat-optimization-readiness.md#actual-server-build-and-historical-audit-evidence--october-2-2208-utc).

## Retained-capture adoption integrated — October 2, 22:03 UTC

Audit receipts are public `c0f3d85`; retained native capture import is public
`01015ac`. Root reviewed the import and independently passed 53 importer,
receipt and legacy stage CPU tests. The new path references every saved capture
and original A8/A1 gate by explicit SHA, writes external durable audit receipts,
and runs ordinary provider, coverage, model smoke and checkpoint-zero readiness
gates in a separate new-source run. It does not require the unfinished old run
to be declared ready. Historical sidecars are not assumed trusted; absent new
receipts require one full semantic audit per shard.

Frozen recovery09 remains untouched, observed at 71/353 at 21:56 UTC with all
10,000 TRAIN/1,002 development prompts retained and zero optimizer updates.
The corrected CPU server03 packet passed fresh resource/source admission;
actual build PID/result remains pending with the sole preparation operator.
A concrete new-source adoption packet is being prepared before ownership transfer.
Supporting research remains stopped. [Current checkpoint](goals/qat-optimization-readiness.md#retained-capture-adoption-integrated--october-2-2203-utc).

## Preparation recovery running — October 2, 21:19 UTC

Actual09 prep-only recovery is verified live after the current-boot disconnect
proof. Supervisor2713/child2714 retain all353manifests/10,000train/1,002development
prompts; reconnect is healthy at audit2/353,zero updates/no final receipt. Existing
monitor is rebound to exact09 identities. Native9e2 CUDA synthetic repair is passed
and returned; necessary CPUserver03 admission follows alongside CPU audit. Current
actor/model/resource/final corpus gates and optimizer start remain pending.
[Live recovery](goals/qat-optimization-readiness.md#preparation-recovery09-verified-running--october-2-2119-utc).

## Actual native pass; preparation recovery next — October 2, 20:59 UTC

Native9e2 FTZ repair passed ACTUAL RTX5080/SM120 CUDA fixtures:57pack,114loader,
59graph,218arithmetic and36projection cases. Root reran unchanged raw validator;
byte-exact report and fresh complete teardown/return are verified. Research and
its monitor are stopped by human; QAT/prep and necessary support remain active.

SAME prep Luna passed fresh09 recovery/source/corpus/resource preflight and
current WSLconfig memory20GB/instanceIdleTimeout=-1. New90second CPUdisconnect
proof precedes exact prep-only resume; actual09 start remains pending. Current
model/data/resource gates and first optimizer update remain unverified. New-run
authenticated completed-corpus reuse is tested/public1cb2999; frozen prep unchanged.
[Actual result and next work](goals/qat-optimization-readiness.md#actual-cuda-repair-pass-and-protected-continuation--october-2-2059-utc).

## Native retry after exact wrapper failure — October 2, 20:36 UTC

FreshRTX5080 census passes approved idle/context/resource/source guards. First
launcher stage failed before admission on /proc/366/cwd permission; no native
job started. Minimalv2 preserves process identity, classifies transport ancestors
first and retains project/unknown-ownership denial. Independent25guardtests pass;
both owners accept, SAME sole operator retries with a new helper directory.
Actual native start/result/recovery/training remain unverified. First supported
training path is fixed reference A8/A1 with Q4_0 development evaluation.
[Exact failure and retry](goals/qat-optimization-readiness.md#exact-stage-failure-and-minimal-v2--october-2-2036-utc).

## Executable packet accepted — October 2, 20:21 UTC

QAT successor reviewed the minimal postboot fixture packet;19 local guard tests
pass. SAME prep Luna performs fresh exclusive admission and ONE pinned120s
native9e2 fixture, then proves teardown before admitted exact prep-only recovery.
Actual GPU grant/native start/recovery/training remain unverified. The bounded
collector sources metadata builder is public7588a87; actual model/data/native
byte verification still required. Human directs simplest passing recipe first.
[Accepted packet](goals/qat-optimization-readiness.md#postboot-fixture-packet-accepted--october-2-2021-utc).

## Successor execution preparation — October 2, 20:15 UTC

QAT ownership and SAME ACTIVE15minute heartbeat are verified under01a0fe3f.
Prep01a0fdd6 retains the single remote operator. Protected local work now builds
the fresh-boot exclusive fixture launcher and the later bounded sources metadata
builder in disjoint partitions. Old676 hold/return is revoked; no actual fresh
GPU admission, CUDA start/pass, recovery or training is verified yet.
[Current work](goals/qat-optimization-readiness.md#successor-local-execution-preparation--october-2-2015-utc).

## Current execution checkpoint — October 2, 20:10 UTC

Active goal remains [QAT optimization readiness](goals/qat-optimization-readiness.md).
Fresh CPU classification proves a changed WSL boot and absent original
preparation processes/tmux; frozen source/runtime and stored full captures match,
but final readiness receipt is absent. Sole operator has completed and cleaned
its transport. No GPU availability, validation start, recovery or training is
verified. Connection to human-corrected192.168.4.24 now works with the trusted key.

Human reiterates actual research/QAT execution. Main supervisor is directed to
resume useful isolated CPU research teams; preparation owner retains the SAME
sole operator for fresh exclusive GPU proof/native repair retest, then serialized
admitted preparation recovery. Old676 hold/return guard is obsolete. Training
gates remain mandatory. QAT successor01a0fe3f is acknowledged and SAME ACTIVE15m heartbeat target is
independently verified; predecessor retires with no owned remote job or lease.
[Executable checkpoint](goals/qat-optimization-readiness.md#executable-qat-rotation-checkpoint--october-2-2010-utc).

## Current coordination — human research stop

Active goal remains [QAT optimization readiness](goals/qat-optimization-readiness.md).
QAT successor01a0fe3f and sole prep/GPU support remain active. Native CUDA fixture
ran under832/834 and exited0; raw validation pending, no optimizer-training or
final readiness claim.

Direct human STOP overrides research replenishment. All research teams, Astra,
usage monitor and their CPU jobs are stopped; SAME research heartbeat PAUSED.
All unmerged worktrees/branches/raw artifacts preserved. Research must not
resume from queued old heartbeats without new human instruction. QAT and its
necessary support/monitors/available-credit protection are unchanged.
[Stop verification](parallel20261002/supervisor.md#human-stop--october-2-2044-utc).

Earlier preparation and research observations follow.

**Current QAT/preparation state — October2 18:40 UTC:** successor
01a0fdd6-8e11-7393-9aed-5c99bd08e428 is acknowledged. Both restored15minute
monitors are ACTIVE; preparation uses a8-a1-luna-health-and-recovery. The one
fresh connection attempt verified a persistent local execution keeper but SSH
still timed out before its CPU checker. No staging/hold/CUDA test; all owned
local transport/keeper cleanup verified. Remote health remains UNKNOWN; historical
14:06 full10,000train/1,002dev/zero-update observation is not fresh release proof.
Current source9e2 runtime and20-pass controller are ready; fixture GO is inactive
pending fresh coordination/availability. RTX2080Ti paused.

Learned activation plus batched head changes clip/threshold update semantics;
a reference-preserving serial learned-head fallback is being implemented/tested
in isolated new source before admitting those profiles. Frozen preparation uses
fixed activations/serial head and is untouched. No selected recipe or optimizer
updates. [Checkpoint](goals/qat-optimization-readiness.md#successor-preflight-and-learned-head-gate--october2-1840-utc).

## Overnight research supervisor handoff — 2026-10-02 15:05 UTC

The supporting CPU research teams and all descendants are complete, with no
active research process. The Codex weekly bucket is still the original window:
77% used / 23% remaining at 15:05:23 UTC, `resetsAt=1791049896`; ordinary
usage is allowed and no reset credits were used. The reset estimate is unchanged,
there is no fixed cutoff, and the research stop latch is not triggered. Continue
five-minute checks and stop before an actual reset or at the existing <=1%
threshold. Do not spend a fresh allowance.

QAT owner `01a0fc3d-bbe1-7e93-a19b-a9200dfa186c` and preparation owner
`01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed` are still the acknowledged owners;
their existing 15-minute heartbeats remain ACTIVE and target the same owners.
At 15:08 UTC the QAT owner is active reviewing a local controller-test harness
mismatch and process-safety code; no GPU work started. The preparation owner is
active on local controller failure tests. Latest durable state still has full 10,000/1,002
train/development captures, but no readiness receipt, validation lease, CUDA
retest, training start, or optimizer update is verified. RTX5080 remains under
the preparation job's ownership; RTX2080Ti remains paused. Preserve the active
goal and exact pending gates in
[the QAT checkpoint](goals/qat-optimization-readiness.md#supervisor-rotation-checkpoint--october2-1505-utc).

At 15:13 UTC the QAT owner completed checkpoint `86bd7ca`; controller fixes
remain pending and no GPU job or lease is active. The prep owner is still in
local controller failure tests with staging/GPU execution blocked pending review.
The original weekly allowance remains 77% used / 23% remaining at
`resetsAt=1791049896`.

At 15:18 UTC QAT reports three controller fixes implemented locally, with
expanded tests and immutable review still pending. The preparation owner remains
in local failure tests; staging and GPU execution are still blocked pending
review. Both 15-minute owner heartbeats remain ACTIVE; no lease or GPU activity.

At 15:38 UTC the expanded QAT controller safety tests still fail, so CUDA
execution remains blocked. Preparation found a PID-reuse identity edge case when
the prior process group appears empty; the draft now checks PID identity and the
corrected harness is being rerun. No lease, CUDA test, training, optimizer update,
or GPU release is active.

At 15:53 UTC QAT's focused review says the three controller blockers are fixed
and PID-reuse checks fail closed; the final identity test and exact launch-packet
review remain pending. Preparation is rerunning the corrected harness. No GPU
lease or CUDA execution has started.

At 15:58 UTC QAT reports all 20 local controller-safety tests passing. It is
verifying hashes, the final change, and the exact command before deciding whether
to authorize the bounded fixture. Preparation is sending the same scripts,
hashes, and one-shot command for both-owner review. Nothing is staged remotely;
no lease, CUDA fixture, training, or optimizer update has started.

At 16:03 UTC both owners report review complete and conditionally authorize one
fixture-only transaction. The preparation operator still must pass fresh
ownership/resource preflight before any hold or CUDA start; its active turn is
pending that preflight. No lease, hold, CUDA fixture, or training is active.

At 16:08 UTC the operator reports staging is in progress, but the fresh preflight
has not established an exclusive hold: lease record says GPU not reserved and no
CUDA result is verified. QAT is between healthy turns. The operator remains the
single owner of the conditional transaction; no training/update is active.

At 16:13 UTC the QAT owner reports SSH timed out before staging; no hold or CUDA
test occurred, local cleanup is verified, and remote health is unknown. The lease
record is `atomic05_transport_unknown_no_remote_execution` with no active
operator/reservation. Do not infer GPU free or training failure from the timeout;
both existing owner monitors remain active.

At 16:18 UTC preparation published its timeout checkpoint and keeps the existing
monitor active. Remote health is still unknown; no lease or CUDA result is
verified. Do not infer GPU availability from the stale lease record.

At 16:38 UTC the registered host is still unreachable (`No route to host`) on
the preparation owner's observation-only check. QAT reports no validation lease
or test. Remote health remains unknown; do not claim the GPU is free or the
preparation job failed.

The checkpoint is prepared for a fresh Codex task, but task creation was
rejected by the app's automatic approval gate (`approval policy is never`). No
successor task was created. The existing single research heartbeat remains
attached to the current supervisor; keep it as the only five-minute monitor.


## Overnight support research — 2026-10-02

The three authorized CPU research topics are complete; all research leaders and
descendants report finished, with no persistent experiment processes. Results
and limits are in `experiments/overnight20261002/`. The supervisor continues
monitoring the original Codex weekly window (72% used / 28% remaining at
11:00:31 UTC; resetsAt 1791049896); there is no fixed morning cutoff. Stop
research before the latest verified original-window reset and never use a fresh
allowance for research.

QAT ownership is verified under successor `01a0fc3d-bbe1-7e93-a19b-a9200dfa186c`;
its existing handoff heartbeat is ACTIVE and targets that successor. Preparation
owner `01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed` retains GPU ownership. The latest
saved healthy preparation checkpoint has 10,000 train / 288 development prompts
and zero optimizer updates; no validation lease, CUDA retest, or training start
is verified.

**Latest validation checkpoint, October2 16:00 UTC:** immutable V5 controller
passes20/20 mocked local safety tests. Source re-review confirms prior blockers
fixed; final PID-equality-only delta verified by reconstructing V4's exact hash.
Both owners accept ONE conditional fixture-only atomic05 transaction, performed
entirely by the SAME preparation Luna. Fresh source/ownership/resource/exclusive
hold proof is still required before CUDA execution. Max600s/return reserve180s,
prebuilt fixture120s; backendops deferred. Root CUDA operator remains offline.
Actual grant, fixture start/result and same-process return are not yet verified;
training admission and final preparation endpoint remain separate gates.
[Checkpoint](goals/qat-optimization-readiness.md#atomic05-conditional-fixture-authorization--october2-1600-utc).

**Active goal:** [QAT optimization readiness](goals/qat-optimization-readiness.md).
All five requested controls plus raw fusion correction and affine binary weights
are implemented;952CPUtests/fourskips and native CPU fixtures pass. Actual CUDA
build succeeded on RTX5080/SM120, but packing failed: learnedA1,k33,token2,
minimum F32 subnormals, expectedbeta2^-149/native0. Exact raw case is preserved.
Native9e2c7a900 now disables FTZ only in W1Ax CUDA code; unchanged CPU fixtures
pass, actual CUDA retest is pending. No numerical gate was weakened.

The human assigned this chat validation and later training. Early GPU tests are
allowed under exclusive bounded reservations during CPU audit; training still
requires complete verified data/currentrecipe/model/native/memory gates. Both
completed reservations were returned. Latest actual SIGCONT09:57:37UTC resumed
original676/startticks85132,audit272→274,post45sCPUhealthhealthy/zerooptimizer;
no restart or GPU overlap. The short hold exceeded its deadline2m17 because a
return guard copied a65character hash; it was corrected from actual64byte-text
receipt SHA, preserving gates and exactlyone SIGCONT. Preparation again owns
RTX5080;2080Ti paused. The corrected CPU rebuild is complete. Sourceb32/native9e2, CUDAlibrarySHA
29e41b5e…acf97d8a, full runtime/DLL/RUNPATH and FTZflag-order proof are saved.
Fresh10:37:08UTC retest reservation was denied WITHOUTSIGNAL: ordinal327's
labelmanifest is missing at the first unfinished development capture transition.
All runtime/source/resource guards passed; no GPU reservation or test was granted.
Preparation can now advance new dev capture. Root waits a verified safe CPU
boundary or full release; CUDA repair retest remains pending. [Handoff](QAT_TRAINING_HANDOFF.md).
QATheartbeat remainsACTIVE every15minutes; no CUDA-ready or training-start claim.

**Successor retest deferred, October2 11:01 UTC:** a NEW short request was
denied before any GPU query or signal because current development ordinal334
has no completed labels manifest. The one CPU boundary observation confirms
334 completed manifests /10,000train /448dev, zerooptimizer and original674/676
live;554dev remain. This is boundary evidence, not a fresh full-health checker
or GPU release. The sole Luna's prebuilt120s/conditional180s retest plan is
frozen and ready; no new validation job or hold. Twelve current-native profile
plans pass locally. CPU preparation now addresses a separate current-server
manifest helper; old frozen runtime/labels cannot be relabeled as new readiness.
[Checkpoint](goals/qat-optimization-readiness.md#successor-retest-boundary-denied--october2-1101-utc).

**Current-runtime CPU preparation published:** helper e5dcbb9 passes15 focused
CPU guards; three frozen TRAIN gate prompts have exact selection ancestry.
No CUDA readiness or provider eligibility follows. Latest owner CPU health
11:23UTC is healthy340/353 /10,000train /640dev, zerooptimizer/no receipt.
The sole Luna prepares a bounded CPU-only server target/static inspection,
protecting all seven ready operator artifacts; no remote launch yet. Retest and
training still await their separate gates. [Checkpoint](goals/qat-optimization-readiness.md#current-runtime-cpu-helper-published--october2-1130-utc).

**Latest QAT checkpoint, October2 12:08 UTC:** v2 runtime-binding guard eaa3929
passes53 relevant CPU tests; authentic v1 fixed A8/A1 full-corpus reuse remains
supported with separate current actor gates. Optional/A4 full-corpus admission
needs an explicit ancestry bridge. New reservation03 failed transport; one
CPU safety proof confirms original preparation running/no hold receipt/no grant.
Normal11:58CPUhealth shows10,000train/992dev, zerooptimizer/no readyreceipt.
Capture nearing completion does not release GPU; final preparation endpoint and
all new CUDA/model/recipe gates remain pending. CPU server job is still local,
unlaunched. [Checkpoint](goals/qat-optimization-readiness.md#runtime-admission-and-reservation-safety--october2-1208-utc).

**Full capture measured, October2 12:13 UTC:** same preparation job now has
353 completed label manifests /10,000train /1,002dev exactly. CPU checker is
healthy/nonterminal at teacher_shard_complete, zerooptimizer/no readyreceipt.
Final readiness/coverage/paired smoke/checkpointzero/terminal and fresh GPU
release proof remain required. The sole Luna has CPU-only GO for a bounded
server target/static inspection protecting7operator artifacts; dispatch is not
job-start proof. No GPU lease, CUDA retest or training. [Checkpoint](goals/qat-optimization-readiness.md#full-capture-measured--october2-1213-utc).

**CPU server attempts closed:** two bounded source/RAM/artifact preflights
stopped before building: first a progress-count parser edge, then a real UI asset
custom dependency outside the narrow approved scope. Both failures preserved;
all7operator artifacts unchanged, owned groups/transports gone. No third CPU
retry. Native9e2 CUDA fixture stays ready and untested, queued after verified
preparation release. Normal12:45CPUhealth is full capture/readiness assembly,
zerooptimizer/no finalreceipt/nonterminal. [Checkpoint](goals/qat-optimization-readiness.md#bounded-cpu-server-attempts-closed--october2-1253-utc).

**New monitor alert classified, October2 13:07 UTC:** the unchanged preparation
checker flagged a stale heartbeat at12:59. One bounded CPU sample confirms the
original job actively reading captured features (+198CPUticks/+259MBrchar over2s),
with full corpus/zerooptimizer/no finalreceipt. Source shows959additional full
provider audits;10.39h is a projection, not an ETA. No restart/recovery/GPUrelease.
Same owner monitor will retain stale classification plus CPU counter deltas.
Local phase-appropriate early-validation hold planning resumes under existing
human authorization, with fresh occupancy/resource/posthold grant still required.
[Checkpoint](goals/qat-optimization-readiness.md#stale-heartbeat-classified-as-active-cpu-audits--october2-1307-utc).

**Reservation04 closed, October2 13:56 UTC:** original preparation child676
resumed at13:56:16.319110,29seconds before its hard deadline; same supervisor
and process identities, +199CPUticks after2seconds. No validation test was
launched: coordination/receipt processing consumed the available fixture window,
so its grant was revoked and cancellation acknowledged before fresh guarded
return. Full353/10,000train/1,002dev and zero updates remain; one CPU checker
still flags the same stale provider-audit heartbeat, with active work verified.
GPU belongs to preparation again; SAME15minute monitorACTIVE, no recovery/restart.
[Checkpoint](goals/qat-optimization-readiness.md#reservation04-returned-before-deadline--october2-1356-utc).

**Validation task rotation at a safe boundary:** after two compactions, the
validation/training owner transferred to acknowledged successor
01a0fc3d-bbe1-7e93-a19b-a9200dfa186c (Continue QAT validation and training). No owned remote
jobs or exclusive lease are active. Fresh preparation health10:43:57 UTC is
healthy329/353, completed10,000train/288dev (714dev remain), zerooptimizer.
The built native9e2 repair awaits an exclusive actual CUDA retest. Acknowledgment and SAME active15min heartbeat target were verified from local
registration and automation readback. This rotation does not release the
preparation GPU. [Executable handoff](goals/qat-optimization-readiness.md#validation-ownership-rotation--october2-1043-utc-checkpoint).

**Overnight CPU research authorized (October1):** three teams investigate W1A1
representation geometry, native accepted-prefix objectives and block-parallel
drafter design, with one usage supervisor. QAT task and its monitor are protected
and may use credits. Latest human correction: stop research at <=1% remaining or the actual monitored
reset; the former09:55PDT cutoff is removed. Do not burn the new allowance. No new active goal,
GPU ownership or frozen-training change. [Assignments and control](OVERNIGHT_RESEARCH.md).

**Preparation supervision rotation (2026-10-02 06:42 UTC):** latest single CPU
check healthy, re-audit109/353;327completed manifests retain10,000train/224dev,
zerooptimizerupdates,no readyreceipt. SAME supervisor08 remains soleRTX5080
preparation owner with --prepare-only;2080Ti paused. All tick agents/transports
finished. Acknowledged successor `01a0fb62-cbdb-72f0-8e86-4055b2ccb4ed` owns the SAME ACTIVE15min monitor; GPU
job unchanged; [exact handoff](goals/qat-optimization-readiness.md#preparation-owner-rotation--2026-10-02-0642-utc)
records source identity,job ownership,stopprocedure,tests and nextactions.
Separate validation/training owner still waits for verified fullprep/GPUrelease.

**Existing corpus preparation resumed and verified:** SAME --prepare-only run under supervisor08. CPU health **2026-10-02 05:31:39 UTC** (October1,10:31p.m.PDT), independent ownership **05:32:52UTC** confirms server673/supervisor674/child676 and live prep-only flag afterdisconnect/reconnect. Retained327manifests /10,000train /224dev,zeroQATsteps; re-audit4/353 then6/353 is not lostdata. Existing15min monitorACTIVE;2080Ti paused. No newcaptures yet: everyresume restarts the audit loop over completed data before unfinished dev capture. Same fullgates and automatic stop-before-QAT endpoint; new optimization-feature GPU checks still await ownership release.

**Historical existing corpus preparation pause:** Verified **2026-10-01 23:15:28 UTC** (4:15 p.m. PDT): supervisor07 interrupted/exit0,owned groups1204/1205 absent,no project processes or GPU compute apps. Retained327completed manifests /10,000train /224of1,002dev,zeroQATsteps; preparation remains incomplete. Data/partials/stop-before-QAT boundary preserved. No auto-resume until new human resume instruction.
**Historical preparation observation before pause:** [joint binary EAGLE body and head](goals/recurrent-binary-body-head.md), Phase 1 of the [one-bit research plan](W1_RESEARCH_PLAN.md). **Preparation-only RTX5080 run healthy; QAT optimizer updates disabled.** User endpoint is complete data/QAT prep, then stop. At **2026-10-01 20:44:55 UTC** (1:44 p.m. PDT), supervisor07 passed post-disconnect CPU health and live cmdline includes --prepare-only; server1203/supervisor1204/child1205 verified. Retained327manifests /10,000train /224dev,zerooptimizer; re-audit ordinal1/353 is not lost data. Same capture/audit/readiness/coverage/paired CUDA smoke and initial zero-update save precede automatic exit before optimizer loop. Existing15min monitorACTIVE with prep-only prompt; budget1of2 unchanged. Only launcher stopping control changed; math/native/runtime/config/data/precision/nullcaps and finals remain frozen. RTX2080Ti paused. See latest scope checkpoint.

**Broader CPU-first research audit complete (2026-10-01):** ten user-requested
GPT-6.1 Sol/high agents researched model compression, binary representations,
corrections, direct fitting, low-bit formats, draft policies, output heads,
native execution, data selection and alternative architectures. The
[plain-language ranked list](../experiments/broader-project-audit-2026-10-01.md)
recommends a bounded fusion sign/scale fitter plus a cheap correction control;
learned Q1_0 export and coupled FFN pruning are independent preparation options.
All ten reports are complete. This changes documentation only; no code,
model/test/GPU/remote run or sealed-final action occurred. No new experiment
or architecture was selected, and no fresh live health observation is claimed.
The user requests simpler explanations with high-impact summaries from now on.
See the [checkpoint](goals/recurrent-binary-body-head.md#ten-agent-broader-research-audit-completed).

**Training optimization audit complete (2026-10-01):** five user-requested
GPT-6.1 Sol/high agents reviewed source and primary research. The strongest
implementation candidates are bulk K/V-only Torch prefix construction, a
stacked per-round head and an A1 native-only no-gradient shortcut. Source
arithmetic permits removing 77.88% of prefix-stage linear MACs; no training
speedup or acceptance gain was measured. Binary optimizer, learned quantizer
and curriculum changes remain bounded proposals. Reports and the
[ranked synthesis](../experiments/training-optimization-audit-2026-10-01.md)
are published; all workers completed. No code, remote/GPU, live experiment,
monitor or sealed-final action occurred. See the
[audit checkpoint](goals/recurrent-binary-body-head.md#five-agent-training-optimization-audit-completed).

**Bounded GPU Phase 1B complete (2026-09-29):** fixed four-variant
quality/timing A/B, CUDA deployment checks, first-shard capture/audit and
100-step real row-A16 calibration are verified. All additional bounded
selector quality/timing and tracing gates completed; integrated worker
worktrees are archived, and the GPU is idle. Binary acceptance still trails
Q4_0; no full-body eligibility or final evaluation is claimed. The broader
project continues with the user-authorized CPU preparation below. See the
[completion audit](../experiments/w1-phase1b-completion-audit.md) and
[goal checkpoint](goals/recurrent-binary-body-head.md#bounded-gpu-phase-1b-complete).

## Continuous A8/A1 preparation complete (2026-09-29)

CPU preparation is implemented and independently audited in task
`01a0f01d-65c6-7af0-9660-99c07e95cacd`, continuing the same project goal.
The initial tier has10,000 train/1,002 dev/1,002 sealed-test prompts and24,507
reserve; all sealed payloads remain unopened. A verified22MB launch packet,
continuous dual-model engine, exact CPU resume/recovery, label-only v2 native
stages, refreshed-source binding, fixed Q4_0 dev evaluation and manual runbook
are published. **117 guarded CPU tests pass**; actual CUDA/training/acceptance
and teacher tensor volume remain unverified USER-start gates.

The hourly heartbeat is **PAUSED**, with the per-heartbeat Luna model override
limitation recorded. No native inference/capture/training or WSL GPU action was
launched. A legacy CPU fixture's accelerator RNG API calls were identified and
corrected; the final audit guards library backend APIs. See the
[preparation report](../experiments/continuous-w1ax-preparation.md),
[manual runbook](CONTINUOUS_W1AX_RUNBOOK.md), and
[goal checkpoint](goals/recurrent-binary-body-head.md#continuous-a8a1-cpu-preparation).
The user has now authorized Luna to start the prepared native gates, capture and
continuous paired W1A8/W1A1 training on RTX5080, then monitor with durable spaced
checks. This supersedes the preparation-only boundary; RTX2080Ti remains paused.
One Luna operator owns the GPU. No tight polling/sleep loop is authorized.
See the [launch checkpoint](goals/recurrent-binary-body-head.md#luna-launch-authorization).

Luna operator `/root/luna_launch` connected through tmux MCP, verified no
competing project process and RTX5080 compute capability12.0. Initial baseline
was2,617MiB/2%utilization and764GiB free disk. Packet/frozen artifact hashes match.
A Unicode JSONL delimiter bug was fixed/tested/pushed as`41e230d`, preserving
all data bytes; host config resolves10,000 train/1,002 dev with622,868,961,328B
capture forecast. First supervisor stopped on the unchanged14GiB host-RAM guard.
Corrective validation`2b87b5b` passed the bounded W1A8 native/CUDA gate, then
stopped at A1 initialization: available12.94GiB after reclamation,1.06GiB below
admission; WSL exposes15.21GiB total. No optimizer/full corpus capture ran.
Both attempts/evidence are preserved and no project process remains. Hourly
heartbeat is **PAUSED**; it dispatches pinned Luna/high once a live run is
verified. User-reported memory change now verified: `[wsl2] memory=20GB`,
WSL total20,971,151,360B/available20,237,811,712B at post-restart preflight.
Supervisor03/04 passed A1 memory admission but failed deployment-relevant state/
decision checks. Retained head replay exactly matches native output; CPU tests
confirm scaled-sum cancellation can flip the next A1 sign. Corrected native-order
A1 forward/reference code is integrated as`f0566aa`/`119e418`, retaining local STE
gradients and unchanged other widths/criteria.144 guarded CPU checks pass. Luna
completed both fresh gates (seven checks/six roots each) in new experiment
`luna-continuous-a8-a1-native-order-20260930`, preserving old failures/captured
bytes. The first hourly check at10:23UTC found supervisor interrupted/exit0 at
09:12UTC and trainer stopped on signal15. The stop predates monitor activation;
the earlier live handoff report relied on stale startup evidence. Monitor is
PAUSED, logs/data preserved, no restart. Signal origin and partial capture
recovery require investigation before resume. See [monitor setup](CONTINUOUS_W1AX_MONITOR.md)
and the historical interruption checkpoint. The repair now uses WSL
`instanceIdleTimeout=-1` plus host-side tmux`binary-eagle-runtime`, proved over
90s with all clients closed. Complete32promptshard retained, incomplete611MB
capture quarantined intact. Supervisor02 is running under host tmux847 with
supervisor848/trainer855, verified after disconnect at19:35:12UTC. Capture is
healthy; no optimizer steps. See the
[current checkpoint](goals/recurrent-binary-body-head.md#post-disconnect-real-resume-verified-live).

## Historical checkpoints

**Final selector timing live (2026-09-29):** packed shared/warp quality
pairs each passed 6/6, completing the bounded selector quality gates. The
validated resident/shared/warp runtime is now in an immutable order-balanced
timing queue. Ratios and cleanup remain pending; the goal stays active. See
the [checkpoint](goals/recurrent-binary-body-head.md#final-selector-quality-passed-and-frozen-timing-live).

**Resident and event/warp CUDA gates passed (2026-09-29):** the fixed
resident path matched 6/6 real quality pairs with accepted CUDA0 copies and
no transfer fallback. Warp integer assertions passed 133 operator/three
fanout cases. Event fixtures and actual-model tracing audited without reference
or cap errors. Remaining packed-selector quality/timing and cleanup are
recorded in the [checkpoint](goals/recurrent-binary-body-head.md#resident-fix-and-eventwarp-cuda-gates-passed).

**Runtime timing complete (2026-09-29):** all 120 paired requests and
80 CUDA-graph blocks passed. On the bounded three-prompt workload, Q4_0
order-balanced decode ratios were compact **1.04028** and K/V-only **1.00817**;
row-A16 was 1.02666/1.08174. A combined CUDA build is now live for the
resident-copy fix and event/warp gates. See the [checkpoint](goals/recurrent-binary-body-head.md#runtime-timing-finished-and-combined-cuda-build-started).

**Resident fix integrated (2026-09-29):** the guarded scheduler-copy fix
passed 19 CPU allocator/routing cases and is integrated with default-off event
and warp diagnostics. CUDA proof is queued after the live immutable timing
run. Compact timing has two behavior-matched blocks; K/V-only timing continues.
See the [checkpoint](goals/recurrent-binary-body-head.md#scheduler-copy-candidate-and-combined-runtime-integrated).

**Passing-selector timing underway (2026-09-29):** compact sampling and
K/V-only catch-up are in a supervised order-balanced, five-repetition timing
queue on the frozen three-domain sample and immutable runtime/libraries.
Resident state remains off while its scheduler-copy fix is CPU-tested.
See the [checkpoint](goals/recurrent-binary-body-head.md#scheduler-evidence-and-frozen-timing-queue).

**Resident diagnostic (2026-09-29):** restoring original input placement
restored all six request pairs exactly, isolating forced placement as the
trigger. The passing path still uses host traffic and is not a speed claim.
One bounded scheduler-log probe is live to audit device-only admission; the
resident selector remains gated. See the [checkpoint](goals/recurrent-binary-body-head.md#resident-placement-isolated).

**Native quality gate (2026-09-29):** compact sampling and K/V-only
catch-up each matched 6/6 request pairs exactly. Resident state preserved
outputs but changed row-A16 proposals/acceptance and failed the gate; it
remains off pending a focused placement diagnostic. The GPU is idle. See
the [checkpoint](goals/recurrent-binary-body-head.md#real-selector-quality-results-and-resident-divergence).

**CUDA runtime fixtures passed (2026-09-29):** corrected dense and all six
packed A1/A4/A8 sharing off/on fixtures passed, alongside 133 operator and
three fanout cases. Matched real-model compact/cache-only/resident quality
pairs are running sequentially on the frozen three-domain sample. The native
refresh bridge is integrated with 20 CPU tests; capture/training permissions
remain false. See the [checkpoint](goals/recurrent-binary-body-head.md#corrected-cuda-fixtures-and-native-quality-queue).

**Stable CUDA checks (2026-09-29):** dense runtime fixture, 133/133 binary
operator cases and three shared-pack fanout graphs passed on RTX 5080. Packed
fixture reruns await a tested encoder batch-metadata initialization fix;
no numerical or cache assertion is being relaxed. Real selector quality and
event/warp validation remain pending. See the [checkpoint](goals/recurrent-binary-body-head.md#stable-cuda-operator-and-dense-fixture-milestone).

**Stable runtime validation started (2026-09-29):** shared packing and
bounded resident state are integrated as native `8fd9b399a` / parent `0261d1b`
after CPU checks. The supervised CUDA build is live; actual CUDA fixtures and
selector comparisons are next. See the [checkpoint](goals/recurrent-binary-body-head.md#stable-runtime-integration-after-calibration).

**Real-model calibration complete (2026-09-29):** the gated RTX 5080 run
finished 100/100 row-A16 hard-CE steps with finite loss and all 18 gradients.
Mean/median synchronized step time was 0.923/0.737 s; CUDA allocator peaks
were 8.494 GiB allocated and 10.049 GiB reserved. The GPU is idle with no
remaining run process. Original/full-body eligibility remains false; no
trained native quality claim has been made. The authorized engineering backlog
remains active. See the [report](../experiments/w1ax-row-a16-calibration-5080.md)
and [checkpoint](goals/recurrent-binary-body-head.md#real-model-calibration-completed).

**Calibration started (2026-09-29):** all five readiness gates passed,
including nine ordered cache/head bridges. The sole GPU owner launched the
authorized supervised 100-step row-A16 hard-CE calibration on RTX 5080.
Completion and timing/memory measurements remain pending. Original data
and full-body training eligibility remain false. See the
[launch checkpoint](goals/recurrent-binary-body-head.md#calibration-readiness-passed-and-optimizer-launched).

**Readiness adapter integrated (2026-09-29):** the selected-head prenorm
observer adapter passed 13 CPU tests and is pushed as `fd8e1a1`. The supervised
CPU readiness retry is running against the existing capture; no optimizer step
has started. See the [checkpoint](goals/recurrent-binary-body-head.md#selected-head-observer-adapter-integrated).

**Pilot cache/backward gates passed (2026-09-29):** nine roots have finite
all-nine gradients and exact frozen operands; the CUDA audit matched
5,533,696 key and value elements each and 5,404 causal masks. Q4_0 and
row-A16 response IDs match on the three prompts. Versioned readiness
assembly is completing the ordered sequence/task/cache ancestry join before
any optimizer run;
original eligibility remains false. Real label-only v2 conversion/audit
passed on the 31-prompt shard without copying 7.66 GB of raw logits. See the
[latest evidence](goals/recurrent-binary-body-head.md#phase-1b-completed-backward-and-native-cache-evidence)
and [storage/refresh checkpoint](goals/recurrent-binary-body-head.md#compact-storage-and-refresh-engineering-checkpoint).

**Parallel engineering resumed (2026-09-29):** the user's companion-chat
authorization starts the remaining CPU runtime, compact-storage and refresh
work within the same goal. The bounded pilot remains first: numeric checks
passed, while backward-only gradient and reduced CUDA cache capture checks
are still in progress. A full graph capture reached its 512-MiB cap and is
incomplete; it does not authorize training. Calibration-only evidence and
measurement guards are pushed; the original bundle stays ineligible. See the
[assignments and current job](goals/recurrent-binary-body-head.md#parallel-engineering-authorization-and-pilot-continuation).

**Torch CUDA pilot gate (2026-09-29):** nine selected roots passed the
frozen numerical check with zero proposal disagreements. Maximum state/logit
relative RMS was `3.55e-5`/`7.77e-5`, below `0.10`; the three Q4_0 response
ID pairs matched exactly. The original bundle remains ineligible until
cache/mask, gradient and versioned provider gates pass. The sole coordinator
is building the opt-in CUDA cache diagnostic after preserving the original
runtime. See the
[checkpoint](goals/recurrent-binary-body-head.md#phase-1b-torch-cuda-numerical-gate).

**Pilot revalidation (2026-09-29 18:45 UTC):** fresh first-shard re-audit
passed byte-identically, all pinned provider/model snapshot hashes matched,
and all nine exported row-A16 sign-bit/scale pairs matched checkpoint zero.
The GPU was idle at the fresh check. The Torch/native numeric check and
selected-root CUDA cache/mask evidence remain required before calibration;
the original bundle is still ineligible. See the
[checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-pilot-revalidation-and-implementation).

**Pilot resumed (2026-09-29 18:17 UTC):** the user chose the short practical
native gate followed by one 100-step row-A16 hard-CE calibration if it passes.
The original capture remains ineligible until a versioned gate record proves
the focused checks. RTX 5080 access remains resumed; a fresh check found 0%
utilization, about 2.9 GiB baseline use and no project process. The
[decision](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices)
and [goal checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-short-pilot-resume)
record the limited scope and stop condition.
The three-domain native row-A16 diagnostic has now completed with exact
response-ID matches to the candidate-D capture and 199 shared proposal roots
across the three prompts. The focused Torch-versus-native check is next; no
eligibility flag has changed.

**Prior decision boundary:** the native GPU Phase 1B task has completed its
fixed A/B, CUDA, first-shard capture/audit, synthetic device gate and untrained
checkpoint-zero quality checks. The first-shard provider still rejects
captured-data QAT because its readiness flag is false. The user owns the
[practical-native versus strict-parity and initialization choice](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices)
before a real-model 100-step calibration. The native task is blocked on that
choice after three consecutive goal turns; the project goal and RTX 5080
access flag are unchanged. No GPU process is active.

**RTX 5080 resume (2026-09-29 06:54 UTC):** a fresh tmux MCP session found 0% utilization, about 2.9 GiB baseline memory and no project process. The local RTX 5080 pause flag was resumed. Two full timed pairs in opposite orders completed, with 480 behavior-matched requests and 20 verified graph blocks per condition. The [timing report](../experiments/eagle-prune-timing-5080.md) gives order-balanced on/off server decode ratios of Q4_0 1.0085×, D 1.0513× and FP16 1.0112×; target-only moved 1.0013×. D remains far slower than Q4_0 overall. The first bounded 31-prompt native capture finished: 12,610 raw verifier-logit rows, 7,796 selected feature rows, byte-identical independent bundle audit, compact teacher and 31/31 response audit. A 100-step model-independent row-A4 CUDA trainer fixture exited zero with finite metrics and 18 gradient tensors per step. The [capture report](../experiments/w1ax-shard0000-capture-5080.md) records hashes and limits. Its manifest remains preparation-only and training-ineligible, so real QAT still requires a documented readiness decision. The GPU has returned to baseline with no project process. The [goal checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-first-shard-audit) records run IDs, owner and next gate. The prior pause checkpoint below remains historical.

**Row checkpoint-zero gate (2026-09-29):** pinned dense weights produced one
untrained nine-linear row checkpoint; A4 and A16 exports passed serialization
and native CUDA graph execution. On 24 old development prompts, A4 accepted
131 drafts over 2,917 rounds and A16 accepted 323 over 2,725; matched Q4_0
accepted 1,555 over 1,493. Both row variants emitted the same raw IDs as Q4_0
on all 24 pairs. The [native report](../experiments/w1ax-checkpoint-zero-native-5080.md)
records hashes and limits. This is low checkpoint-zero acceptance, not a QAT
or timing result. The first-shard provider remains ineligible; the
[readiness/initialization choice](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices)
is pending. No project GPU process is running and RTX 5080 is at baseline.
The first-shard CPU domain check confirmed all 12,610 audited rows across
prose (4,815), reasoning (4,020) and code (3,775), with 12,250 supported
labels in total; see the capture report. This does not change eligibility.

**GPU pause checkpoint (2026-09-29 05:55 UTC):** an earlier reply misread “No pause gpu work actually” as a direction to continue. The coordinator corrected that interpretation, set the RTX 5080 pause flag, verified every Phase 1B supervisor terminal and no project `llama-server`, trainer or supervisor process on WSL, and closed tmux MCP session `$34`. The GPU still has Windows game/display workload; no project GPU process remains. The completed CUDA build, 112/112 operator check and exact 96-pair quality comparison are preserved. Timed A/B, first-shard capture and training calibration did not run. See the [goal checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-pause-checkpoint).

**GPU Phase 1B startup (2026-09-28):** the coordinator resumed the local RTX 5080 flag and owns the sole GPU experiment through tmux MCP. Pinned target/FP16/Q4_0/D/config hashes matched; verified links resolve archived remote artifacts. The supervised SM120 CUDA rebuild completed, and 112/112 W1A1 matrix-operation cases passed. The remote 2k candidate freeze and 65-shard plan now match their original hashes exactly. The paired off/on native quality check matched all 96 requests in generated IDs, speculative counters and checked round semantics; the [report](../experiments/eagle-prune-quality-5080.md) records hashes and limits. The first-shard runner preflight passed without inference. With all project processes stopped, sustained non-project GPU load persists above 90% utilization and 4.8 GiB use; uncontended timing and the memory-heavy capture await resource clarification. The RTX 5080 pause flag remains resumed. The [goal checkpoint](goals/recurrent-binary-body-head.md#gpu-phase-1b-startup) records session, runs and limits.

**CPU Phase 1A complete (2026-09-28):** the fresh team's latency analyzer,
opt-in shared runtime patch, four-format row-scale joint QAT with a guarded
Torch device path, eligible-capture/multi-shard provider, pinned candidate
2k/192/192 data freeze and bounded 65-shard capture plan are reviewed,
CPU-tested, committed and pushed on main through `d9f3ae9`; native gitlink
`14c188e` is published in the user's fork. The
[completion checkpoint](goals/recurrent-binary-body-head.md#cpu-phase-1a-completion-and-gpu-only-boundary)
records hashes, limits and the first supervised GPU commands. All temporary
team worktrees were archived after preserving ignored data. The same project
goal remains active; native acceptance, CUDA timing/training and SM75 claims
wait for explicit restored access. Full-tier raw-logit storage and final
representation/objective budgets are user-owned choices before their runs.

**CPU Phase 1A team setup (historical):** fresh task
`01a0e9d0-1273-70a1-972e-8d1381f72701` resumed the same goal with separate
runtime, joint QAT, data and CPU verification workers. Their file ownership
and worktrees are recorded in the [goal checkpoint](goals/recurrent-binary-body-head.md#fresh-team-execution-cpu-phase-1a).
Row-scale W1Ax is the actionable common-format implementation default while
candidate D group-128/A16 stays separately labeled; the native learned-row
loader gate now passes CPU metadata tests for all four widths. The options and
provisional practical numeric gate are in [DECISIONS.md](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices).

The archived-latency and capture-readiness report, larger-data preparers and
compact-teacher schema, and synthetic joint W1Ax QAT path are now integrated
and pushed (`e5bd3dc`, `0b5fce7`, `6db186f`). The corrected native runtime
patch and latency analyzer were integrated as `112693f`, with native gitlink
`14c188e` published in the user's fork. Subsequent data/capture/provider
commits and final CPU checks are summarized in the completion checkpoint.

**Candidate data freeze (2026-09-28):** pinned Dolly/GSM8K/MBPP source files
and a hashed local manifest now supply 2,000 candidate train prompts plus
192 independent development and 192 sealed new final prompts, balanced across
three broad domains (`c4f6764`). Raw prompts remain ignored under `data/`; the
new final text was not opened. This is not yet tokenized, captured or approved
as sufficient training coverage. The [source report](../experiments/w1a-public-source-freeze.md)
records hashes, terms and limitations. A Luna worker accidentally queried
local CUDA/MPS availability once during environment discovery; no accelerator
operation or model inference ran, and verification resumed on explicit CPU.

**Trainer integration (2026-09-28):** the joint QAT CLI now accepts an audited
native-prefix provider (`29f0e96`). It checks eligibility before model loading,
rechecks trace and exact-prefix compact-teacher ancestry, and retains recurrent
state/cache gradients in CPU fixtures. The concrete future-capture factory is
integrated as `83e72d6`; it still awaits eligible larger captures. A guarded
device-aware Torch rollout is integrated as `4d0684d`; bounded raw-logit
sharding is integrated as `08c70b0` (65 capped train shards), and the
capture-tool prompt-contract extension is integrated as `a18e5a9`. The v1 bundle still
requires raw logits for re-audit; the full-tier compact/label-only storage
choice is recorded in [DECISIONS.md](DECISIONS.md#phase-1a-implementation-defaults-and-pending-research-choices).
Native training quality and throughput remain unmeasured.

**Pre-team handoff (historical):** the user requested an overarching plan and a fresh team to continue the existing EAGLE work with no GPU access. Parent starting point was `d111335`, following the predecessor's `5110257`; native gitlink then was `21f617d4ef3f5dc383d3ab8dc619daaa87db7ff8`. Both host pause flags were set. The plan superseded the old next action to continue target block-14 parity. The old 96 prompts are smoke/regression data, not an adequate full-body QAT corpus. The [handoff](goals/recurrent-binary-body-head.md#fresh-team-handoff-eagle-w1-cpu-phase) and [completion checkpoint](goals/recurrent-binary-body-head.md#cpu-phase-1a-completion-and-gpu-only-boundary) preserve the state across tasks.

**Latest completed goal:** [mixed precision rescue and frozen-body head adaptation on RTX 5080](goals/binary-rescue-head-5080.md), completed 2026-09-28 UTC. The preceding [binary scale fitting goal](goals/binary-scale-fitting-5080.md) completed 2026-09-27 UTC.

**Pre-handoff checkpoint (2026-09-28 UTC, historical):** the output-preserving RTX 5080
block-14 stage capture was compared with source HF CUDA/F16 eager on
the frozen 29-token training prefix. At position 3, accumulated HF
error grows from 1.2157% at FFN input to 12.1527% at FFN branch
output and 4.1478% at complete block output. Replacing the full
block-14 input with native rows cast to F16 leaves 0.1015%, 0.5045%
and 0.1491% at those boundaries; the earlier intervention controls
reproduce exactly. This locates the observed amplification chiefly
in the FFN path on that row, without identifying the causal FFN
operation. The [stage report](../experiments/recurrent-target-block14-stage-intervention.md),
[safe capture](../experiments/recurrent-target-block14-safe-stages.md)
and [goal checkpoint](goals/recurrent-binary-body-head.md#forty-seventh-goal-turn-block-14-stage-attribution-gpu-paused)
record hashes and limits. Our supervised job finished and process
group 417 is absent. New RTX 5080 runs are locally blocked; a later
read showed 95% GPU utilization and 8,011 MiB whole-device use from
a workload outside this project run. The Apple CPU drafter forward
has 18 exact diagnostic depths. Full target-feature parity, a
training numeric policy and all-body budget remain open. Training,
final-set and Q4_0 evaluation remain gated.

**Prior CPU arithmetic checkpoint (2026-09-28 UTC):** corrected CPU RMS norm and RoPE
frequency arithmetic matched all 619,520 F16 fused-input and 123,904 raw
F32 K/V projection elements across three post-acceptance joins. F16 projected
value writes matched 123,904/123,904; projected key writes matched
123,900/123,904. Native stored cache bytes and attention arithmetic were
still unverified at that checkpoint. The [goal file](goals/recurrent-binary-body-head.md#twelfth-goal-turn-native-style-cpu-norm-and-rope)
records the checks, report hashes, 5080 access change and next gate.

**Stored-cache checkpoint (2026-09-28 UTC):** an opt-in native CPU capture
verified actual F16 draft-cache writes and exact-prefix mask inputs on two
frozen training requests. Key and value bytes each matched all 148,480
captured F16 elements, including two reserve rows per request; all 145
decoder mask rows allowed exactly slots through their query position. The
[cache report](../experiments/recurrent-binary-cpu-cache-parity.md) and
[active goal checkpoint](goals/recurrent-binary-body-head.md#thirteenth-goal-turn-actual-stored-draft-cache-and-mask)
record hashes, commits and limits. Attention arithmetic, general cache
behavior, full 96-prompt capture, training and Q4_0 quality/speed gates
remain open. No GPU run had started at that checkpoint.

**RTX 5080 capture checkpoint (2026-09-28 UTC):** the authorized eight-token
CUDA smoke on one frozen training prompt captured raw target logits and
features, passed continuity/response audits and reproduced the CPU raw IDs.
Its supervised run stopped and the GPU returned idle. The full 96-prompt
capture is prepared with expanded explicit row limits and an isolated
remote checkout; it has not started. The [goal checkpoint](goals/recurrent-binary-body-head.md#fourteenth-goal-turn-cuda-capture-path-and-full-run-setup)
records the owner, tmux session, remote directory, commits, source hashes
and stop procedure. No training budget or final-set use has been approved.

**Full frozen-training capture (2026-09-28 UTC):** the supervised RTX 5080
run finished all 96 training requests with 40,815 joined head/verifier-logit
rows, 52,297 raw target-feature rows and 8,295 native rounds. Its process
group stopped and the GPU is free. Internal continuity and native row/feature
preparers passed; the bundle and all-request response audits remain in
progress. The [active goal checkpoint](goals/recurrent-binary-body-head.md#fifteenth-goal-turn-frozen-96-prompt-cuda-capture-audit-pending)
records counts, hashes, limitations and the next CPU checks. The raw capture
is not yet training-eligible; no optimization or final-set evaluation ran.

**Final capture preparation checkpoint (2026-09-28 UTC):** the frozen
96-prompt bundle and all-request audit now pass. All 40,815 raw verifier
logit rows and 15,042 retained feature rows are joined; 96/96 responses
match 12,251 native emissions, including one final EOS stop with an
un-emitted canonical suffix. Mean mapped target probability mass is 0.972
on captured candidate-D histories. The bundle remains explicitly
`training_eligible: false`. A CPU attention arithmetic ablation explains
most first-seed drift but leaves residual numerical differences. The
[active goal checkpoint](goals/recurrent-binary-body-head.md#sixteenth-goal-turn-full-capture-audited-training-still-gated)
and [full capture report](../experiments/recurrent-binary-full-capture-5080.md)
record hashes, checks, failed first audit attempts and remaining gates.
The RTX 5080 is free; no training or final-set use occurred.

**Target-feature checkpoint (2026-09-28 UTC):** independent Hugging Face
forwards on Apple M3 Max CPU and, separately, RTX 5080 CUDA/F16 used frozen
training prefixes and checked source embeddings. Their raw target-feature
rows still differed from native execution by median relative row L2 errors
of roughly 0.3–0.6% across taps 2, 18 and 33; the largest checked row
reached 2.088% in the CPU reasoning capture. The
[target-feature report](../experiments/recurrent-binary-target-feature-parity.md)
records per-tap measurements, hardware, versions and source hashes. Switching
the independent CUDA forward from eager to SDPA attention barely changed
the differences. Both CUDA diagnostics stopped and the GPU is free. Exact
target-feature and
whole-drafter state/logit parity remain open; the capture bundle is still
training-ineligible pending a user-owned numeric gate and training budget.

**Full-prefix feature checkpoint (2026-09-28 UTC):** the sealed 96-request
capture supplied 3,112 complete training prefill rows. On RTX 5080 CUDA/F16,
independent eager forwards differed from native tap-2/18/33 features by
median relative row L2 of 0.293/0.538/0.488%. One tap-18 row reached
10.014%; the same row reached 12.135% with independent SDPA attention.
An Apple M3 Max ggml CPU layer-0 operator probe found near-exact RMS norm
but nonzero pre-attention Q/K/V projection differences from F32 references.
The native capture also has 103/427 cross-request prefill-row pairs with
identical token prefixes but different feature bytes, beginning after the
first token; the cause needs investigation. A bounded recapture of the
tap-18 outlier prompt matched all 222,720 native prefill F32 values bitwise,
so that row is reproducible under the same native CUDA binary.
The [goal checkpoint](goals/recurrent-binary-body-head.md#eighteenth-goal-turn-full-training-prefix-feature-distribution-and-layer-0-probe)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md)
record hashes, hardware and limits. Both supervised GPU comparisons exited
zero and released the 5080. Exact target-feature and whole-drafter parity,
numeric gate, training budget and Q4_0 evaluation remain open.

**Tap-2 intervention checkpoint (2026-09-28 UTC):** on the reproducible
29-token outlier training prompt, substituting captured native tap-2 input
into an independent RTX 5080 F16 forward cut position-3 tap-2 error from
0.324% to 0.020%, while tap-18 error rose from 10.014% to 11.659%.
Thus the early tap-2 mismatch alone does not explain the later outlier.
The [goal checkpoint](goals/recurrent-binary-body-head.md#nineteenth-goal-turn-native-tap-2-input-intervention)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md#native-tap-2-input-intervention-on-the-outlier)
record bounds, source hash and limits. The supervised GPU run stopped;
target layer-by-layer parity, the numeric gate, training and Q4_0 evaluation
remain open.

**Target ladder checkpoint (2026-09-28 UTC):** a bounded native RTX 5080
capture on the 29-token outlier prompt recorded target layer inputs
`0–18,33`. Existing taps 2/18/33 matched the same-run and sealed full96
feature bytes exactly. The independent F16 forward's position-3 error had
its largest adjacent rise across target block 14: 1.209% at layer-14 input
to 4.148% at layer-15 input, with absolute RMS error 0.01202→0.04218.
The [active goal checkpoint](goals/recurrent-binary-body-head.md#twentieth-goal-turn-target-layer-input-ladder-localizes-the-outlier)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md#native-target-layer-input-ladder-on-the-outlier)
record source hashes, hardware and limits. Both supervised runs stopped and
the GPU is free. The responsible block operation, exact target parity,
training gate and Q4_0 evaluation remain open.

**Block-14 intervention checkpoint (2026-09-28 UTC):** substituting captured
native layer-14 input into the independent RTX 5080 F16 forward reduced
the outlier's layer-15 relative error from 4.148% to 0.149% and absolute
RMS error from 0.04218 to 0.001516. Block 14 amplifies earlier drift in
this comparison; its same-input operator mismatch is much smaller. The
[active goal](goals/recurrent-binary-body-head.md#twenty-first-goal-turn-block-14-amplifies-upstream-drift)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md#same-input-block-14-intervention)
record the source hash and limits. The supervised run stopped and the GPU
is free. The first material local operator difference and training gate
remain open.

**Local target-block screen (2026-09-28 UTC):** on the same frozen 29-token
prefix, independent RTX 5080 F16 forwards supplied each block `0–17` its
own captured native input. At the position-3 outlier, every same-input
block-output error was at most 0.272% relative row L2, compared with the
accumulated 10.014% layer-18 error. Small local backend differences are
amplified through later blocks; the all-row ranking differs and no numeric
gate follows from one prompt. The [goal checkpoint](goals/recurrent-binary-body-head.md#twenty-second-goal-turn-local-target-blocks-and-amplified-state-drift)
and [feature report](../experiments/recurrent-binary-target-feature-parity.md#same-input-local-block-screen)
record source hashes and limits. The supervised run stopped and the GPU is
free. Full-drafter parity, training budget and Q4_0 evaluation remain open.

**CPU drafter checkpoint (2026-09-28 UTC):** a standalone ggml replay using
actual stored F16 draft K/V bytes, captured masks and native queries matched
all **593,920** F32 attention output elements bitwise across 46 prose and
reasoning decoder executions on Apple M3 Max. A separate no-optimizer
real-size two-step D probe applied CE only at the later proposal and found
nonzero gradients in the earlier pre-norm state and appended K/V rows, with
zero gradient on the earlier logits. The [goal checkpoint](goals/recurrent-binary-body-head.md#twenty-third-goal-turn-native-cpu-attention-oracle-and-real-causal-gradient),
[attention report](../experiments/recurrent-binary-cpu-attention-oracle.md) and
[gradient report](../experiments/recurrent-binary-real-later-gradient.md)
record hashes, hardware and limits. The Python student still uses a different
F32 attention forward, and exact whole-drafter parity, training budget and
Q4_0 evaluation remain open. No GPU was used for these two checks.

**Optional native student-attention checkpoint (2026-09-28 UTC):** the CPU
student can now use the pinned ggml attention result in an explicit
diagnostic forward while retaining an F32 surrogate backward. Correcting
Python/native Q/K row order made the prose first-seed attention output
4,096/4,096 F32 elements bitwise equal; reasoning's remaining maximum
attention error is 0.002172 from stored K/V operand differences. First
normalized-state maximum error fell to 0.000184 prose and 0.000511
reasoning, and a real-size later-only loss still reached earlier state/K/V
through the surrogate without an optimizer step. The [goal checkpoint](goals/recurrent-binary-body-head.md#twenty-fourth-goal-turn-optional-native-forward-student-attention)
and [diagnostic report](../experiments/recurrent-binary-native-attention-student.md)
record exact hashes and limits. This mode is not a chosen training recipe;
full-drafter parity, all-body budget and Q4_0 evaluation remain open.

**Same-input FFN checkpoint (2026-09-28 UTC):** on a captured native FFN
input, candidate D's native-order CPU arithmetic matched all 2,560 prose
output values bitwise; grouped matmul differed by at most 1.4305e-6.
On reasoning, both modes remained 6.1035e-5 from native while differing
from each other by at most 1.9073e-6. The [goal checkpoint](goals/recurrent-binary-body-head.md#twenty-fifth-goal-turn-same-input-binary-ffn-boundary)
and [FFN report](../experiments/recurrent-binary-ffn-same-input.md) record
hashes and limits. This narrows the remaining reasoning FFN gap beyond
grouped reduction order; exact whole-drafter parity, training budget and
Q4_0 evaluation remain open. No GPU was used.

**Handoff checkpoint (2026-09-28 06:35 UTC):** the active goal file records
the current objective, pushed commits, CPU tests, projected K/V write
comparison, pending 5080 clarification and exact next actions. Code round 2
matched 37,681/37,888 F16-rounded key operands and 37,723/37,888 value
operands across 37 reconstructed context positions; native stored K/V bytes
remain unread. No agent, model server, local experiment or remote job is
running. The RTX 5080's availability has not changed the explicit CPU-only
restriction. Continue from [the active goal handoff](goals/recurrent-binary-body-head.md#rotation-handoff-2026-09-28-0635-utc).

**Current goal milestone (2026-09-28 UTC):** the [joint-training protocol](../experiments/recurrent-binary-qat-plan.md) records the candidate-D W1A16 representation, exact-prefix recurrent supervision, proposed bounded trial and stop gates. The CPU hard-binary core, nine-linear installer, exact-prefix trace validator, masked recurrent loss, differentiable proposal-chain interface, optimizer ownership/checkpoint step and learned-scale GGUF serializer are integrated. The focused CPU gates pass, including a training-checkpoint-to-GGUF synthetic roundtrip and a two-step causal-cache gradient test. A new scalar CPU replay matched 430/430 archived candidate-D native samples across all nine projections. This is arithmetic evidence on old training captures, not a trained-model quality result. Existing cached head states cannot train the body; a real feature/cache/verifier capture, full-drafter numeric parity and trained-export native validation remain necessary. No accelerator, remote host, model training or final-set prompt has been used for this goal. Q4_0 remains the primary future acceptance, latency and throughput gate; no new quality or speed claim exists.

**Second CPU milestone:** native decoder memory position is one behind the
shifted input-token index; the rollout and tests now use that position.
Accepted-prefix cache rebuilding uses raw target features with a fresh cache
and an explicit truncated-gradient boundary. The trace rejects draft-head
logits mislabeled as raw target verifier logits. The CPU capture gate now
requires a hashed raw-feature ledger joined to every accepted-prefix anchor,
with 7,680-wide F32 rows and frozen target tap order. An explicit grouped-F32-matmul
training option matched 395/430 archived native D outputs exactly; its maximum
absolute difference was `0.0001220703125`, so real-model argmax and cache
parity remain gates. Training checkpoint manifests record the arithmetic mode.
No current accelerator operation was performed.

**Third CPU milestone:** the forked llama.cpp loader now recognizes truthful
`f32_learned_nonnegative` metadata at published submodule commit `7f23c89b3`.
A CPU-only `libllama` build and eight native loader fixtures passed. This
checks metadata and tensor coverage, not numerical parity of a trained GGUF.
The pinned AngelSlim forward's floating-weight dtype read and F32 K/V cache
were identified as incompatible with the proposed hard-binary/F16-KV path.
An explicit CPU decoder-step adapter now runs all nine binary linears through
two synthetic proposal steps and a masked optimizer update, including F16
K/V cache writes. A frozen-D GGUF initialization audit passed all nine
tensor pairs and found 17,005 exact-zero scales. Native full-drafter numeric
parity remains unverified. A read-only CPU GGUF view verified the pinned FP16
target and D draft hashes, memory-mapped the frozen target embedding and
extracted one F16 row plus four F32 draft norms. The adapter can consume these
operands without copying the full target embedding or changing target weights.

**Fourth CPU milestone:** fork commit `c282087a9` adds an opt-in 32-row-capped
file of raw target verifier logits, copied before sampler processing and
separate from the existing mapped draft-head logit file. A CPU-only
`llama-server` build passed with GPU and optional Accelerate/BLAS backends
disabled; runtime capture was not exercised. The CPU capture audit checks
that file's hash, target-vocabulary width, source label and unique exact-prefix
row joins. Live `verifier_reached` and teacher-forced valid/support masks are
now recorded separately, so later-position loss is not censored merely by
an earlier live rejection. Accepted-prefix raw target-feature capture and
real-model parity remain open.

**Fifth CPU milestone:** fork commit `ddcf2a608` adds opt-in bounded raw
target-feature rows and one retention disposition per decoded row. The native
source captures ordered target layer-input taps before EAGLE fusion, records
exact token ancestry and marks speculative input rows `j<=A` retained after
`A` accepted drafts. A CPU-only server build passed with optional accelerator
and BLAS backends disabled. Parent CPU preparers now validate native
head/round/label/map joins and select accepted-prefix feature rows; 59
recurrent tests pass. No real-model feature capture or microbatch row-order
parity has run, and Q4_0 quality/throughput gates remain untouched.

**Sixth CPU milestone:** a pinned `recurrent-train` capture mode now prepares
the frozen 96-prompt D/D own-history run with explicit raw target-logit and
feature budgets, request ownership/ranges, and hashes for each raw stream.
It has only been exercised with mocked CPU server output. A separate CPU
continuity audit checks complete captured prefill, every speculative input,
the accepted-prefix retention rule and consecutive round prefixes/seeds;
initial sampling, terminal emission and request completeness remain
unverified. A bundle builder joins both native preparers and the final
capture audit, checks pinned target/draft hashes and the D map digest, and
always marks output `training_eligible: false`. The focused 73 recurrent
and 16 capture-runner CPU tests pass. No real-model
capture, training, accelerator or Q4_0 quality/throughput gate ran.

**Seventh CPU milestone:** the first actual pinned FP16-target/candidate-D
model request ran on an accelerator-disabled Apple M3 Max CPU server. A
missing A16 activation setting initially stopped draft loading; the runner
now sets it explicitly. The eight-token frozen-training-prompt diagnostic
captured four native rounds, 16 head and raw target-logit rows, and 53 raw
target-feature rows. Internal continuity, both preparers and the final
preparation-only bundle audit passed. A response audit joined all eight
output IDs to the seed, four round emissions and the terminal no-proposal
trace; the final sample lacks an independent logit check. See the
[CPU capture report](../experiments/recurrent-binary-cpu-capture-smoke.md)
for hashes and limits. A one-row microbatch repeat matched all captured
feature, head-state and target-logit values bitwise. The focused 77 recurrent,
16 capture-runner and nine native-adapter CPU tests pass. No training,
GPU/accelerator or Q4_0 comparison ran.
The real-model CPU adapter now accepts the pinned 2,560-hidden/4,096-Q
attention geometry and replays five first-round proposals. Both exact-order
and grouped-matmul paths matched all five native mapped top IDs; normalized
state drift reached about 0.0034 in one element, so numerical parity remains
unverified. A separate diagnostic two-position CPU SGD step produced finite
nonzero sign and scale gradients in all nine linears, exported a learned-scale
GGUF, and loaded it in the CPU native server for one matching eight-token
request. This verifies the joint gradient/checkpoint/export/loader path at
real dimensions, not training quality or Q4_0 performance.

**Eighth CPU milestone:** the same pinned one-prompt run with Flash Attention
disabled preserved all output IDs but changed native raw target features.
Replaying each capture's own features reduced first/fifth-depth state drift
under the non-flash path, without closing numeric parity. An independent
local Hugging Face CPU forward checked all 32 prompt embedding rows and
sampled target FFN weights bitwise against GGUF, then compared ordered layer
2/18/33 inputs with the native feature stream. Median relative row L2
differences against the non-flash stream were 0.579% / 0.382% / 0.280%.
The [CPU capture report](../experiments/recurrent-binary-cpu-capture-smoke.md)
records the per-row limits and hashes. These are alignment diagnostics,
not exact target-feature parity or SM75 results. The focused 80 recurrent,
16 capture-runner and nine adapter CPU tests pass. No GPU/accelerator ran.

**Ninth CPU milestone:** the forked native server now has a bounded opt-in
draft graph callback, published first in fork commit `b4df1b547`. On one
CPU request its completed trace left existing output IDs, head states,
target features and verifier logits byte-identical. A numeric join to the
first proposal shows decoder inputs, fusion norm and Q/K/V bitwise exact
after Q/K row conversion; RoPE differs by at most `3.55e-6`. The first
material gap is attention output: max `0.009986` with native Flash
Attention auto, `0.005002` with it off. Replaying attention from native
Q/K/V reproduces that gap. The [CPU capture report](../experiments/recurrent-binary-cpu-capture-smoke.md)
has source hashes and the mapped draft-logit comparison. This localizes
the drift but does not certify full parity or Q4_0 quality/throughput. The
focused 86 recurrent, 16 capture-runner and ten adapter CPU tests pass.

**Tenth CPU milestone:** a pinned accelerator-disabled diagnostic runner
captured one code and one reasoning training prompt alongside the earlier
prose prompt. Exact-prefix CPU cache rebuilding and grouped-matmul proposal
unrolls matched all **44/44** native mapped top IDs across **11 rounds**;
the largest normalized state-element difference was `0.004617`. Native
graph traces again put the first material gap at attention. The
[three-prompt report](../experiments/recurrent-binary-cpu-broader-diagnostic.md)
and ignored hashed summary retain per-category counts and limits. This
does not establish the 96-prompt capture, training quality or Q4_0
acceptance/throughput. No GPU or accelerator was used.
The focused 90 recurrent, 16 capture-runner and ten adapter CPU tests pass.

**Eleventh CPU milestone:** a later post-acceptance round from each of the
three training categories was joined to its native decoder graph. Rebuilt
CPU prefix inputs and unrotated Q/K/V matched native bitwise or within
`2.4e-7` at the fused-input boundary. The first material mismatch again
appeared at attention (maximum `0.00984`–`0.01175`); mapped seed top IDs
matched in all three joins. The [broader report](../experiments/recurrent-binary-cpu-broader-diagnostic.md)
records the individual rounds and hashes. Native stored K/V bytes remain
unread, so cache parity is not established. The recurrent CPU suite now
passes 91 tests. No accelerator was used.

**Current milestone:** the [final rescue/readout report](../experiments/binary-rescue-head-5080.md) records a negative result against Q4_0 EAGLE. Q4_0 reached 1.042 accepted drafts/round and 135.1 full-request tokens/s; the admitted attention+fusion Q8_0 rescue reached 0.696 and 83.3 (0.616×), and the fitted frozen-D-body FP16 head reached 0.555 and 60.8 (0.450×). All 120 primary measured raw outputs per path matched Q4_0 and target-only. The 264-request speculative round calibration, 92 graph-verified server blocks across final primary/diagnostic runs, and separate six-prompt longer-context comparison are complete. The longer-context client rates were 127.0 Q4_0, 68.5 combined rescue, 47.1 fitted head and 93.8 target-only tokens/s. Neither endpoint beat target-only. Full raw results, logs, binaries and model artifacts are indexed in the remote archive named in the report. Final RTX 5080 check found no owned process or compute app, 1,916 MiB whole-device use and 0% utilization. The user owns any future body-aware QAT or final-set decision; none was started.

The approved four-way A16 screen completed all **168 requests** (7 paths × 24 development prompts). A/B/C/D accepted **0.119 / 0.161 / 0.290 / 0.425 drafts per round**, versus **1.037 FP16** and **1.042 Q4_0**. Fitted group scales improved 3.585× over row-mean A, closing 33.20% of the Q4_0 gap. D beat all other binary candidates on every prompt but trailed both controls on every prompt. All seven paths matched target-only raw IDs on all 24 prompts. See the [report](../experiments/binary-scale-fitting-5080.md) for counts, depth survival, calibration, artifacts and limitations.

The scale-screen recommendation at that time was to retain D for a common-history body/readout diagnostic; the completed goal above supplied it. The [all-layer W1Ax suite](goals/w1ax-activation-precision-suite.md) and [RTX 2080 Ti quantization suite](goals/rtx2080ti-quantization-suite.md) remain completed and sealed; the prior [W1A1 research goal](goals/full-w1a1-eagle-project.md) remains checkpointed. The next scope/budget choice is user-owned.

**Final W1Ax result:** the twelve-cell development policy grid completed and
validated all **11,520 requests**, alongside the sealed historical/development,
operator, round, profiling, streaming and 720-request context measurements.
Every W1Ax path had lower pooled decode and full-request throughput than both
same-cell anchors. The largest observed decode ratios were 0.707× FP16 and
0.684× Q4_0. The [main report](../experiments/w1ax-activation-precision-results.md)
and [complete policy appendix](../experiments/w1ax-policy-grid-results.md) retain
fixed versus development-selected policies, raw-output differences, timing-control
variation and measurement limits. The interrupted 120-request attempt remains
preserved and excluded from completed-cell rates. Final verification found no
owned jobs or GPU compute apps; the 2080 Ti was idle. The goal file records exact
artifacts, hashes, checks and completion evidence.

Ubuntu 24.04 WSL2 and SSH are reachable through the shared host registry. The
RTX 2080 Ti (SM75) passed five native W1A1 CUDA backend
cases, a standalone 21-case/880-dot binary-MMA probe, and both integrated
portable/MMA eight-case backend gates. All five W1A1 draft GGUFs passed
source-row audits; the FP16 target track fits in 11,264 MiB VRAM.

The full [nine-variant Turing comparison](../experiments/rtx2080ti-quantization-suite.md)
completed 540/540 matched requests across 12 prompts and five repetitions.
Ordinary EAGLE decoded at 82.49 tokens/s. Native head W1A1 reached 74.81
(0.907× ordinary), and all-group W1A1 46.31 (0.561×). Q4_0 and Q8_0 draft
controls reached 91.51 and 87.73 (1.109× and 1.064×). Fusion, attention,
and FFN W1A1 also trailed ordinary. All eight speculative variants produced
identical text on all 60 paired requests; each differed from target-only on
one prompt. Ratios to target-only are timing observations, not strict
lossless speedups. Q4_0/Q8_0 stored types are verified, and a later isolated
trace confirmed their Q8_1 activation conversion and MMVQ/MMQ dispatch.

A separate 240-request four-path comparison found integrated binary-MMA head
W1A1 at 74.72 decode tok/s versus portable head W1A1 at 74.59, a 1.0017×
ratio with paired 95% interval 0.9973–1.0062. Both paths emitted identical
text on all 60 paired requests; no end-to-end MMA gain was resolved. The
genuine native W8A8/W4A4 300-request vector comparison has also finished
with verified CUDA dispatch. W8A8 decoded at 80.18 tok/s versus ordinary
EAGLE's 80.99 (0.990×; paired 95% interval 0.975–1.005), so no difference
was resolved. W4A4 decoded at 41.40 tok/s (0.511×), with accepted drafts
collapsing to 0.092/round versus ordinary's 1.168. The seven-path Tensor
Core comparison completed 420/420 requests with identical text and acceptance within
each default/MMA pair. W8A8 MMA decoded at 0.821× its default DP4A path
(paired 95% interval 0.817–0.824); W4A4 MMA at 0.880× its default vector
(0.876–0.884). Specialized matrix instructions were slower for this EAGLE
workload. A separate executed-path trace confirmed Q4_0/Q8_0 use Q8_1
activation quantization with MMVQ during one/two-token work and MMQ during an
observed 38-token operation. The goal file has exact commits, owners, and raw
hashes. The [final synthesis](../experiments/rtx2080ti-synthesis.md) and
packing-inclusive CUDA traces are preserved; final cleanup confirmed all
supervisors stopped and the GPU released.

## RTX 5080 result

A dedicated packed W1A1 operation, EAGLE output-head GGUF exporter/loader,
and CUDA dispatch were integrated in the 5080-tested llama.cpp revision
`92bc706`. The expanded five-setting revision is `8d2b18a`.
CUDA backend correctness passed 5/5 cases, all 32,000 packed head rows were
audited against the published BF16 source, and every packed benchmark server
logged actual CUDA XOR/POPCOUNT dispatch. See the
[integration report](../experiments/ggml-w1a1-cuda-5080.md).
A separate [captured-input parity run](../experiments/real-head-parity-5080.md)
checked 8 real drafter inputs against all 32,000 packed head rows on the
5080: exact sign packing and 256,000 integer dots, with no scaled-output
tolerance failures.

The [matched five-repetition comparison](../experiments/native-end-to-end-5080.md)
completed 180 requests on 12 fixed prompts with the same FP16 target and
server settings:

| Variant | Request tokens/s | Decode tokens/s |
| --- | ---: | ---: |
| Target-only | 96.67 | 99.39 |
| Ordinary EAGLE | 123.96 | 132.88 |
| Packed-head W1A1 EAGLE | 115.38 | 122.94 |

Packed-head W1A1 achieved **0.931× ordinary request throughput** and
**0.925× ordinary decode throughput**. Draft generation became faster per
round (4.386→3.515 ms), but accepted draft tokens fell (1.161→0.892 per
round), requiring 480 extra verification rounds. All five repetitions and
all three prompt categories favored ordinary EAGLE. The packed draft used
148 MiB less GPU memory while loaded. Both speculative paths exceeded
target-only throughput, but their outputs differed from target-only on two
prompts, so that ratio is not a clean lossless speedup claim.
An integrated-kernel Nsight Compute attempt was limited by
`ERR_NVGPUCTRPERM`; separate standalone packing-inclusive CUDA-event timings
are preserved, but no integrated per-kernel trace is claimed.

Ordinary and packed EAGLE decoded texts matched on all 60 paired requests.
A separate raw-token check found identical ordinary/packed IDs on the two
target-only mismatch prompts. An isolated [raw verifier-logit
trace](../experiments/native-verifier-trace-5080.md) reproduced both: at the
emitted rows, the target verifier itself ranked the speculative output first,
by 0.008074 and 0.000729 raw-logit units. The drafts were rejected, so these
were not wrongly accepted tokens. Target-only ranked the opposite IDs first
by 0.000963 and 0.016508 nats. Numerical sensitivity is plausible, but the
precise native baseline mismatch cause remains unproven. The earlier BF16
PyTorch verifier trace separately identified a tree-versus-incremental target
logit tie; see the [acceptance
report](../experiments/pytorch-w1a1-cuda-acceptance.md).

## Other completed gates

The [bounded head-only QAT pilot](../experiments/qat-head-pilot-results.md)
improved validation KL but reduced fixed held-out W1A1 acceptance to 1.565
drafts/round from the untrained 1.677. That recipe was stopped without
held-out tuning. The user's requested [W4A4/W8A8 accepted-per-round
comparison](../experiments/pytorch-int4-int8-cuda-acceptance.md) measured
0.2882 and 2.1816 respectively under the BF16 PyTorch verifier; those are
numerical simulations, not native INT4/INT8 timing.

A standalone SM75 binary-MMA probe cross-compiled to `BMMA.88128.XOR.POPC`
and later passed 21 cases/880 exact integer dots on the RTX 2080 Ti; see the
[2080 Ti suite](../experiments/rtx2080ti-quantization-suite.md). A focused
[related-work note](../experiments/related-work-note.md) keeps novelty claims
narrow: quantized EAGLE and native QAT already exist.
An opt-in [integrated binary-MMA
candidate](../experiments/integrated-binary-mma-5080.md) also passed 8/8
scalar-reference backend cases on the 5080, matched all 85 packed-draft
tokens in one model request, and compiled to SM75 SASS containing the exact
binary-MMA instruction. It subsequently passed the real SM75 backend gate and
tied portable W1A1 in the matched head comparison above.
The paired benchmark runner and analysis now support an opt-in fourth MMA
variant with separate selector/dispatch records and same-device MMA/portable
speed ratios. Local fake-server and analysis checks passed 15/15; the
[2080 Ti runbook](RTX2080TI_RUNBOOK.md) specifies the required run.

## Next research gate

The [post-suite consolidation](../experiments/one-bit-next-steps-2026-09-27.md)
updates the earlier agent brainstorms using the completed policy and round
measurements. It recommends recovering useful binary weights with wider
activations first: finish missing graph/state parity checks, then a train-only
fixed-sign row/group scale-fitting screen, with readout/body adaptation
conditional on its result. W1A1 remains the research endpoint. The subsequently approved scale-fitting screen is now complete (report above).
Body/readout adaptation and any expanded training budget remain user-owned;
reserved-final evaluation has not begun.

The [all-layer W1Ax study](../experiments/w1ax-activation-precision-results.md)
is complete, including the predeclared policy grid. Its report separates
acceptance, identical-input operator cost, complete round timing and serving
rates, with explicit measurement limits. Use its quality evidence to guide the [QAT revisit
plan](../experiments/qat-revisit-plan.md): audit
drafter-state/target-verifier alignment and target probability mass outside the
draft vocabulary before training, then test one bounded target-aligned recipe
on the frozen new development/final prompts. If acceptance improves, run native
same-device end-to-end comparisons against **both** FP16 EAGLE and Q4_0 EAGLE.
Later, screen non-EAGLE drafters such as block-parallel DFlash/DSpark before
investing in their W1A1 kernels. The completed 2080 Ti and 5080 experiments
are sealed, and the GPUs were released after their runs.

## Local research review (2026-09-25)

Seven Astra-high local-only analyses are collected in the
[ranked synthesis](../experiments/research-review-2026-09-25.md). They identify
target-aligned QAT capture, unused native cache-catch-up graph work, genuine
early draft caps, shared activation packing, structured scales, and a later
DFlash/DSpark quality screen as bounded opportunities. These are advisory
findings, not new performance results or changes to the active W1Ax protocol.
The initial review performed no GPU work or web search. A subsequent
[primary-source cross-reference](../experiments/research-cross-reference-2026-09-25.md)
revised the seven reports: DSpark becomes the lead later architecture candidate
with matched DFlash control; hard CE and structured scales remain hypotheses;
native sample-and-match is distinguished from probability-ratio verification.
The current W1Ax goal, one-run QAT budget and sealed final set are unchanged.
This literature review produced no new model or GPU result.

## PrismML / quantization research (2026-09-25)

The user-requested [PrismML research report](../experiments/prism-quantization-research-2026-09-25.md)
separates ternary task-score claims from true binary-weight execution and our
W1A1 acceptance objective. A full selected-weight geometry audit and a bounded
local CPU Q1 control support investigating representation fitting and
head/body adaptation before new kernel work. Existing Q1_0 support was verified;
the naive grouped control is not an optimized Prism checkpoint.

The active SM75 suite, its GPU owner, frozen protocol and reserved final prompts
are unchanged. New fitting/QAT budgets and any practical weight-only branch
remain proposals for the user, not additional experiments started by this review.


## Parallel support launch — October 2, 18:35 UTC

Four isolated CPU research teams and independent Luna validators are active,
coordinated by Astra medium, with protected QAT/preflight ownership restored.
Five-minute usage heartbeat `parallel-research-usage-control` is ACTIVE; initial
window now21% remaining, resetsAt1791049896. CPU research stops at<=1%;
research and its monitor stop on reset while QAT continues including existing
credits. Actual new CUDA/training start is not verified. See
[supervisor](parallel20261002/supervisor.md) and
[active-goal checkpoint](goals/qat-optimization-readiness.md#parallel-support-launch--october-2-1835-utc).


**Parallel support18:46UTC:** first four CPU audits integrated,30/30 combined
new tests pass; three next teams launched (auxiliary VJPs/exported function/refresh
contract) at19% original weekly remaining. QAT's sole source worker prepares
learned-head serial-training preservation; frozen prep unaffected. Latest remote
check18:40:51SSH255/no CUDA, healthunknown; host details requested. Protected
QAT supervision ACTIVE. [Checkpoint](parallel20261002/supervisor.md#first-slate-integrated-second-slate-launched--october-2-1846-utc).


**Parallel support18:58UTC:** second CPU slate complete/integrated,20/20 combined
tests pass; auxiliary gradients/exported function gates pass; refresh rate unit
confirmed, receipt remains proposal.18% original weekly remaining; Astra selects
third batch. Protected learned-head correction final82/82 immutable CPU tests
pass, main integration coordination pending. GPU connectivity stillunknown/host
replypending; protected supervisioncontinues.
[Checkpoint](parallel20261002/supervisor.md#second-slate-integration--october-2-1858-utc).


**Parallel support19:01UTC:** learned-head correction59ef557 and adaptation
f9a21f3 integrated with exact immutable82-test hashes; second-slate20tests pass
on corrected main. No remote runtime/training adoption. Three third-slate CPU
teams active (evaluator recipe/resume validation/memory ledger),17%original
weeklyremaining. Prep pending-host-info gate durable; protected monitorsACTIVE
quiet/local-only until resolution.
[Checkpoint](parallel20261002/supervisor.md#protected-correction-and-third-slate--october-2-1901-utc).


**Parallel support19:11UTC:** third CPU slate integrated as reports/prototypes,
26 root checks pass. Confirmed296MiBsave-host undercount and modern evaluator
recipe construction defect; isolated resume validation prototype also passes.
QAT owner assesses adoption/fresh identity, frozen jobs unchanged.16% original
weeklyremaining; Astra prepares fourth2optimization tasks, no GPU/SSHwhilehost
info pending. [Checkpoint](parallel20261002/supervisor.md#third-slate-integrated--october-2-1911-utc).

## Preparation launch and supervision rotation — October 3, 06:43 UTC

The unique current-source preparation-only job actually started. Supervisor806 /
child807 run in detached host tmux, with zero optimizer updates authorized.
Launch receipt, fresh resource/source checks and the sole operator's complete
local transport closure are verified. Full model readiness and GPU release
remain pending; old09 is historical and cannot be resumed or restarted.
The saved one-check CPU monitor query now binds the actual new job and current
source, and counts retained captures from their original authenticated paths.
The preparation owner is rotating after two compactions without interrupting
the detached job; explicit successor acknowledgment and SAME heartbeat transfer
are recorded in the [preparation rotation checkpoint](goals/qat-optimization-readiness.md#preparation-supervision-rotation--october-3-0643-utc).

Preparation transfer is complete: acknowledged successor
`01a10084-101e-7311-94e9-9658f9dc648f` owns the unchanged detached job and the
SAME ACTIVE15minute heartbeat. Predecessor and its sole operator are retired;
no active transport or duplicate schedule. Exact transfer proof and remaining
gates are in the linked rotation checkpoint. Full preparation readiness,
GPU release and optimizer training remain pending.

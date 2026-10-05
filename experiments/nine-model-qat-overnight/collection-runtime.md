# Independent collection runtime source adapter

Bounded deliverable: a source-only freeze/inspect/controller/stage adapter for the
independently hashed six-candidate collection, reusing the existing native argv
builder and request loop. It preserves six original lane/train/export identities,
three original Q4 model/provenance records, target/F16 verifier/KV, native sampler,
head/cache/math, fixed ordering and authenticated protocol. The same-bundle entry
point retains its original bundle/train/export equality checks and receipt schema.
No historical provenance is rewritten; no replacement Campaign receipt is made.

`scripts/run_nine_model_endpoint_collection.py` freezes schema
`nine_model_collection_runtime_plan_v1`. The explicit inputs are a validated
collection locator, six original upstream owner records and separate supervised
producer job records. CLI source preparation uses `--freeze-output`, `--collection`,
`--collection-sha256`, `--upstream-owners`, `--upstream-jobs`. Preparation opens no
prompt messages, queries no GPU and starts no process. Existing output is rejected.
Inspection uses `--plan`/`--plan-sha256` without `--start`.

Each owner entry is exactly `controller`, `supervisor`, `identity_evidence`.
Identities have original PID/start_ticks/boot_id, appear in existing hashed
producer/monitor evidence and join the unchanged terminal remote_job PID fields
and original lane resource-baseline boot. The extra job map requires `run_dir`,
`supervisor_state`, those same identities and evidence. Terminal jobs must be
natural exit0/no signal, and their original command must include that exact run
directory. Every original export receipt must be inside a declared producer run;
include separate endpoint watchers and all other original upstream producers.
Original process.json/process-lineage.json inventories are frozen and rechecked.
Dedicated controller/supervisor PGIDs must equal their respective leader PIDs;
this guarantees the existing LinuxResources PGID check has a birth identity.
Terminal metadata does not establish actual process absence.

The new source inventory explicitly pins the **new dispatcher and evaluator**,
the current Python executable and local transitive scripts/src Python files.
Original evaluation-source and training-lane source inventories are independently
preserved. Their historical evaluator pin cannot prove this new route. Relocated
or edited source requires a new runtime-plan freeze; old source must remain at
its original authenticated paths. Evaluation environment/resource policy comes
from the authenticated evaluation source, rather than assuming six historical
training checkouts have identical library paths. It must preserve or strengthen
every original lane's floors and resource-return tolerances. Live host pause uses
the original common host-control path; no policy, protocol or allowance is chosen.

Execution requires `--start`, a new run directory, current plan-bound
nine_model_gpu_lease_v1 availability and detached remote_job state. Controller
admission verifies fresh <=300s sole-owner availability, current live pause,
actual remote_job supervisor kernel cmdline/group/session and live birth identity,
Linux tmux and sufficient stop grace. It acquires the existing shared
rtx5080-campaign.lock with nonblocking flock, writes its actual birth identity,
and verifies that another descriptor cannot acquire it. Only then does the
LinuxResources observer establish current UUID/SM120/boot, whole-device contexts,
all original process/group absence and resource floors/return.

The controller invokes the pinned stage through unchanged SubprocessRunner. Its
fresh typed continuation binds collection and runtime-plan identity, immutable
startup lease, live parent/lock owner and actual upstream release/resource proof.
The child rechecks actual supervisor identity and lock. A fresh typed per-cell
continuation follows actual release under the same held owner and unchanged
standing evaluation-source/protocol authority; it expires after <=300s for cell
admission. No lease is rewritten and no new human availability announcement is
invented. The original startup lease may expire during the bounded outer run;
new cell admission uses fresh release/owner checks. STOP, pause, owner/source/file
changes are checked between requests, outside request measurement timing, and
charged to the existing outer wall allowance.

Full scope is six candidates plus all three original Q4 controls and target-only.
`scripts/evaluate_nine_model_native.py:evaluate` shares the original native loop.
Guards register server birth identities, discover descendants, stop the server
and clean descendants even when discovery or shutdown raises. Each cell must
prove process-group/CUDA absence, empty foreign CUDA/DXG census, same boot/UUID/
SM120 and host/GPU resource return before the next launch. Controller cleanup
also checks SubprocessRunner's complete evaluator/descendant registry. Loader
markers, per-projection diagnostic CUDA dispatch and no-dense-fallback gates
remain original. The separate collection receipt, measurements **and aggregate
report** preserve collection/runtime source identity and every original candidate
and control association. It never labels a collection SHA as a training bundle.

Final admission uses the integrated heldout feature (original55466f8, test branch
dependencyd6593ba). Boolean final_set_authorized cannot admit final inputs.
validate_collection authenticates the frozen selection/disjointness/protocol
before this adapter, retaining opaque prompt sealing during freeze/inspection.
The first prompt-byte read occurs in the shared loop after actual runtime gates.
No final rows, allowance or recipes were selected by this feature owner.

## Checks and practical limits

Mac arm64/CPU, primary .venv Python3.11.15. Fourteen runtime tests plus the
collection/collection-QA/final-admission/final-QA modules:29focused checks PASS.
They use genuine original producer record shapes at modeled OS/observer/runner
boundaries; the existing tiny serialized producer tests cover heavy checkpoint
joins. Runtime tests include real local nonblocking flock contention, six original
censuses plus separate export/watcher census, actual 60-cell shared-loop callback
ordering, stage-to-loop model/receipt joins and aggregate ancestry, stale/wrong/
nonexclusive/device lease, wrong boot/source/model/control/owner, foreign holders,
STOP/pause, checkpoint mutation, missing upstream birth proof, uncovered export,
new process census and failed per-cell resource return. They distinguish terminal
metadata from actual observer process absence. CPU fixtures are not CUDA proof.

Related final nine-model suite after final-admission dependency:259tests,
254PASS and five existing device/platform skips. Final rerun and independent QA evidence are recorded in the ignored
results/nine-model-qat-overnight/collection-runtime-qa directory. Changed-file
Ruff check/format and git diff check pass. Initial fixture/QA failures and their
corrections are retained; no source blocker is hidden by a CPU pass.

No remote/tmux/GPU, pretrained weights, real checkpoints/captures/final prompt
bytes, live30a/nativecc9/watcher/lease edits, native launch or extra live monitor
was performed. Root alone owns production supervision and durable goal records.
Actual readiness still needs all six genuine completed exports, original three
controls, authenticated held-out scope/selection, complete original producer
records (including watcher), fresh availability/current host release and actual
native load/dispatch/resource evidence. Existing missing scientific/operational
inputs remain PENDING. No acceptance/latency/throughput claim is made.

One efficiency limit remains explicit: heavy serialized joins run once in each
process's initial collection validation, hence controller and evaluator child
perform two total validation passes before native execution. They do not rerun
per cell/request; immutable Files inode/stat/hash guards retain all validated
artifacts thereafter and no checkpoint tensors are retained. A future trusted
parent-to-child validation handoff could remove that duplicate startup pass;
this source adapter deliberately retains independent child validation. Charge
both passes to the authenticated outer allowance and assess their real resource
cost before production use. Child continuation verification follows its heavy
validation and expiry remains strict; if cold validation consumes the <=300s
continuation, fail closed and obtain a new fresh continuation/retry after immutable
joins are validated. Cold startup/resource cost is unknown. This is a reported gap against an exactly-once
whole-controller requirement, not a claim that fixtures establish runtime ready.

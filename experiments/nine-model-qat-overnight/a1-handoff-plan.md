# Source-pinned direct A1 handoff metadata

Bounded source feature based on main `21174a8`, owned in temporary worktree
`/Users/pippo/github/binary-eagle-a1-handoff`, branch `feat/a1-handoff-plan`.
The deliverable is an immutable, inspectable dependency plan; it does not
implement a staged driver or complete automatic A8 to A1 dispatch. Existing
packet bind, lane runner and endpoint scripts and source gates are unchanged.
Root retains the durable goal, decisions and sole live GPU authority.

## Contract and use

`prepare_nine_model_lane_handoff.py --inputs INPUTS.json --output PLAN.json`
validates and exclusively publishes the metadata plan. Inspect with
`--plan PLAN.json --plan-sha256 EXACT_SHA`. Optional `--availability LEASE.json`
uses the existing `require_available` contract and still grants no execution.
The tool uses standard-library Python only and loads the existing stdlib-only
pipeline file directly to avoid the package's eager PyTorch/model imports.
It runs only read-only Git source inspection, never executes stage commands,
queries hardware, imports Torch/NumPy/models,
reads model/checkpoint/raw capture bytes, changes a packet or lease, chooses a
recipe/budget, or touches remote hosts. Metadata pins and Python sources are
rehashed; large target/corpus locators are preserved without byte access.

Input schema is `nine_model_lane_handoff_inputs_v1`:

- `upstream_endpoint_plan`: existing immutable A8 `nine_model_lane_endpoint_plan_v1`
  path/SHA locator. Its pinned Python sources and exact kernel identities are
  checked as saved metadata; original runtime validation remains mandatory.
- `source_checkout`: canonical root of the exact original A1 command checkout
  (actual packet uses `c2544aa7928b0d0c454099a56ae912262c6b0ab5`). `source` is
  the existing endpoint-style relative Python filename to path/SHA inventory,
  covering every `.py` under `scripts` and `src/w1a1_eagle` in that checkout.
  `source_revision` pins the exact 40-hex Git revision; HEAD must match, the
  inventory must match that revision and every Python blob must match its Git
  content identity. Changed dependency bytes cannot be admitted by repinning a
  physical path while retaining the original revision.
  No source pin is rewritten to current main. Actual initial request pins trainer
  SHA `f1437b4e588dbfe3672a795e064fc7027730617ddb2c687d6af417941280834d`;
  executing a retention-modified trainer against that request must reject.
- `packet`: exact path/SHA locators named `commands`, `initial_request`, `config`,
  `resolved_inputs`, `budget`, `source_joins`, `generation_replay_link`,
  `replay_receipts`. These point to the existing A1 files and original physical
  replay receipt log. Config, request, calibration, precision, command source
  and exact argv joins are checked. Historical link generation/replay producer
  hashes and original TRAIN tokens/history/ancestry joins are checked without
  relabeling a producer or regenerating receipts. Raw payload validation stays
  with the existing runtime capture/bind contract.
- `policy`: an existing `nine_model_lane_endpoint_policy_v1` path/SHA locator,
  with finite bounds and the same standing authorization-record byte hash.
  Copied authorization locators may differ; their content/provenance may not.
  The budget remains `human_selected=false` with delegated operational provenance;
  no new policy is selected by freezing a plan.
- `outputs`: complete path-only late-bound records `{path: ABSOLUTE, status: PENDING}`
  for `initial_receipt`, `export_audit`, `independent_qa`, `production_inputs`,
  `admission_plan`, `lane`, `train_receipt`, `endpoint_inputs`, `endpoint_plan`,
  `endpoint_result`. Known packet output names must match the original packet.
  Output paths must be unpublished; a missing, promoted or already published
  record refuses freezing. Later actual outputs use the existing runtime schemas
  and must be frozen/admitted separately; this plan does not fabricate them.
- Optional `upstream_snapshot`: exact saved locators for `supervisor`, `lane_state`,
  `train_receipt`, `endpoint_state`. Absence remains PENDING. If supplied, natural
  terminal success/no signal, exact controller/supervisor PIDs, original lane/config,
  positive committed approved-budget endpoint, full allocated trainer seconds,
  SM120 GPU identity, owned-release declaration and successful matched report joins
  must agree. Training receipt belongs beneath the original run's `attempts`;
  report belongs at the endpoint state's `evaluation/report.json`. Internally
  consistent foreign-parent copies, STOP, premature budget and wrong-lane results
  reject. Saved metadata never proves live process identity absence or GPU release.

Plan stages are ordered: natural A8 endpoint, A1 initial actor, export, independent
QA, full bind, admission plan, lane bundle, actual SM120 admission, long QAT,
endpoint plan, endpoint evaluation. Initial actor/export argv is copied and
validated against the original packet; bind argv uses that same pinned helper
and original replay log/link. Later argv stays null until its actual admission,
QA, lease and supervisor inputs exist. Admission and long QAT are logical gates:
existing `run_nine_model_lane.py` performs both within one supervised invocation.
No new separate admission-only driver is implied.

Every stage remains PENDING. Every GPU stage explicitly requires live exact
boot/birth identity absence, owned process/CUDA release, no foreign CUDA/DXG
holders, the shared nonblocking GPU flock, adjacent fresh same-boot SM120 physical
hardware/resource observation, fresh stage-bound maximum-300-second lease and
live host pause=false, plus detached remote_job supervision and release checks.
`fresh_live_verification_required=true`, `production_ready=false`, and
`execution_authorized_by_this_file=false` are unconditional. JSON cannot prove a
kernel release. Planner/pipeline source pins are also frozen for inspection.
Publication writes/fsyncs a temporary file and atomically links it exclusively;
failed serialization publishes nothing, and an existing plan cannot be replaced.
Incomplete input/output inventories refuse publication.

## Acceptance and remaining integration

Mac arm64, Python 3.11 standard-library synthetic/filesystem/tiny-Git checks only. Owner
`python3 -m unittest discover -s tests -p test_nine_model_lane_handoff.py -v`
passes 20 checks. Changed-file Ruff check and format pass. Tests cover actual CLI
freeze/inspect/overwrite refusal, failed serialization, model-free imports, old
source versus current drift, repinned command mismatch, changed argv/link hashes,
original Git revision/dependency identity and exact checkpoint manifest joins,
forged generation ancestry/physical replay producer, incomplete/promoted/published
outputs, standing-delegation forgery, stale lease, stopped/premature/wrong-parent
saved endpoint metadata and internally consistent foreign-parent receipt copies.
No CUDA, actual A1 actor/export/admission, latency, acceptance or throughput claim.

Independent Luna review found that internally consistent relocated receipt/report
metadata lacked path ancestry checks in the first draft. The owner added parent
containment and expected-report-path validation plus a regression test. The
atomic exclusive publication and serialization-failure checks also address the
reviewed partial-file risk. Independent final findings and exact test output are
preserved under ignored `results/nine-model-qat-overnight/a1-handoff-qa/` and must
be copied to primary main before worker retirement.

Root must materialize the actual metadata descriptor with CUDA hidden after
review, using the exact original A1 source checkout and upstream frozen endpoint.
No actual descriptor/plan was materialized by this worker. The durable active goal
should record this report/source commit at integration. Actual A1 zero-update
actor, all-nine export, full bind/current independent QA, lane/admission bundle,
fresh SM120 admission, long QAT and endpoint remain unfinished. A staged sequencer
that resolves fresh output pins, issues adjacent leases, holds the GPU flock and
advances existing supervised CLI stages remains an explicit implementation gap.
Healthy A8 and passive endpoint continue unchanged under root ownership.

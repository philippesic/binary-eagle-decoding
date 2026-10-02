# Configured-shape QAT allocation and lifetime ledger

October 2, 2026. CPU arithmetic and tiny storage calibration only. Source base
`ce5da6bf484346181a1e9d6d916a313d3d346f3b`; exact source SHA256 records and
regenerable values are in [configured-ledger.json](configured-ledger.json).
This closes one host-admission arithmetic gap and does **not** establish CUDA
fit, memory reservation, native acceptance, or throughput. No full model,
configured-shape tensor, GPU, Metal, SSH, or capture was used.

**Concrete finding:** curriculum save charges cloned tensor payload plus 16 MiB,
but `_cpu_tree` executes `detach().cpu().clone()`. For a CUDA input, CPU transfer
storage and its CPU clone overlap. The last large Adam head moment is copied
after almost all payload has already been retained. Core plus Adam alone needs
2,947,020,836 additional live host bytes versus 2,636,640,328 charged bytes: an
undercount of **310,380,508 bytes (296.002 MiB)**. RNG and metadata are excluded
from both totals, not assumed zero. Adding their actual sizes cannot repair this
omitted large transfer. The fixed/Adam layout is an admitted configuration. The ordered case is a
fully populated Adam state after ordinary updates; legal sparse state or other
insertion orders require their own event order, not this exact total.

One narrow estimate proposal: retain the current host floor and 16 MiB allowance,
and charge `tensor_bytes(raw_payload) + largest_cuda_tensor_bytes(raw_payload)
+ 16 MiB`. The largest configured F32 tensor is 327,680,000 bytes. This covers
the transfer overlap conservatively without making payload aliases free: the
recursive clone makes a new storage for **each occurrence**. No production
helper or live job was changed. Workspace/allocator effects remain unbounded.

## Units and unique owners

Let W=218,234,880 selected weights, R=65,280 output rows, P=W+R=218,300,160,
L=327,680,000 bytes (81,920,000 F32 head elements). B=4P=873,200,640 bytes,
I=4R=261,120 bytes, M=8P=1,746,401,280 bytes. The configured shared frozen
aggregate F=777,912,320 bytes is a supplied estimate, not a measured storage
inventory; actual model parameters/buffers must validate it. Selected weights
are F32 trainable masters themselves; there is no extra master copy. The old
dense selected weights are removed during installation and do not belong in
steady-state frozen storage. Frozen biases, if present, add their actual F32
row counts; none can be inferred from the shape list alone.

| Unique allocation | Bytes | Location and ownership |
| --- | ---: | --- |
| Latent signs + scale offsets | B per lane | Device; independent A8/A1 parameters |
| Initial row scales | I per lane | Device frozen buffers; not scale masters or moments |
| Adam first and second moments | M per lane | Device; exactly two buffers per owned parameter |
| Adam scalar steps | 4 per parameter tensor | CPU for current noncapturable Adam; 18 base tensors/lane |
| One lane's gradients | B | Device; freed with `set_to_none=True` before next lane |
| Shared frozen drafter aggregate | F supplied | Device once across lanes; target remains CPU/provider-owned |
| Hard sign outputs | 4W=872,939,520 | Device, one cached output per projection per round |
| Before-update sign snapshots | W=218,234,880 | Device bool; not packed; live through optimizer/diagnostics |
| Before-update scale snapshots | 4R | Device; live through optimizer/diagnostics |
| Row NPZ latent/effective scales | B per export | Host copied NumPy arrays, one lane exported at a time |
| Packed GGUF signs + row scales | 27,540,480/lane | Host output arrays; packing retains no F32 master set |
| Optional four-mask sign diagnostics | 218,234,880 for two lanes | Host packed NumPy masks; distinct from device snapshots |

Learned activations have **six** independent F32 scalars/lane, not nine: Q/K/V
share one and gate/up share one. Bank registration, projection registration,
state_dict entries and optimizer ownership are aliases and counted once while
resident. Each scalar adds 4 master + 8 Adam moment + 4 active gradient bytes,
plus one CPU step. All-row affine midpoint adds R F32 masters/lane (FC-only
adds 2,560). Rank-1 fusion adds 10,240 F32 factor elements; rank-4 adds 40,960;
optional output bias adds 2,560. Each has the same master/moment/gradient cost.
F16 deployment factors are derived copies, not optimizer masters. Forward
`_F16FactorSTE` holds rounded F32 factors and F16 conversion temporaries, plus
token-by-rank intermediate and token-by-2,560 correction output. Affine sums
and learned support/derivative tensors depend on real token invocation counts.

## Stage allocation and release ledger

The numbers below are logical tensor storage. Host additions are beyond the
resident process at the admission snapshot, not additions to a blank process.
CPU model/target/source loader costs cannot be reconstructed exactly from nine
selected shapes; denote their actual unique resident storage H. Current config
uses a 12 GiB CPU load allowance; H is not proved bounded by that allowance.

| Phase | Device live/additional allocations | Host live/additional allocations and release |
| --- | --- | --- |
| Initialization | No target transfer. After installation, two independent B+I binary lanes plus shared F; moments absent until smoke. | H includes CPU target/dense draft; installation retains original selected modules in `pending` until return, all replacements accumulated, and per-matrix F32 conversion/abs/latent/clone temporaries. At second draft deepcopy admission, add current unique draft storage, including frozen storage temporarily duplicated before rebinding; old frozen duplicate freed after alias restoration. Loader dtype and external retained target references unknown. |
| Paired zero-update smoke | Both Adam sets eagerly initialized, then serial lane graph, hard signs, one lane gradients. Persistent subtotal 2(B+I+M)+F=6,017,638,400 bytes (5.604 GiB), excluding small steps and optional state. Gradients cleared and graph references deleted per lane; initialized moments retained. Repeated/resumed smoke evaluates `zeros_like` arguments to `setdefault` even for existing keys: one extra temporary up to L during the preallocation loop, freed immediately when unused. | Existing model/target/teacher host operands and RNG; no new full CPU checkpoint. Smoke is not missing lazy Adam admission. |
| Paired optimizer step | Same persistent subtotal. One active graph and gradients; `joint_train_step` adds bool sign/scale snapshots until its return. For a round with all base gradients allocated, known post-backward subtotal = persistent+B+W+4R=7,109,335,040 (6.621 GiB), excluding optional state and remaining graph/workspace. | Teacher/cursor/RNG/diagnostic state resident. Packed diagnostics, if enabled, create one current packed set across all linears, then per-matrix unpack masks; largest unpack is 81,920,000 bytes. `state_dict` deepcopies the four-mask set during save; this is an additional host copy, not an alias. |
| Continuous resume.pt save | Module/optimizer state_dict tensors alias live storages, so no second device model/moment set. torch.save serialization behavior/workspace is unresolved. | Canonical recipe payloads clone CPU scalars/factors/midpoints; these coexist with live recipe state. Repeated learned registration entries alias within state_dict; canonical recipe clones are separate. Diagnostics state deepcopy is separate. `torch.save` preserves tensor storage aliases; serialization scratch not assumed zero. |
| Continuous row NPZ export | Same resident training state; effective scale expressions allocate per-row device results. | Arrays retain B for one lane; per-matrix CUDA→CPU transfer plus NumPy F32 copy adds at most L beyond retained arrays. Fusion/affine native_payload adds separate derived arrays; learned deployment values are Python scalars. Existing helper charges 2B+L+16 MiB=2,090,858,496 (1.947 GiB), conservatively above fixed-row B+L. Two lane exports are serial, not simultaneous. |
| Standalone GGUF export during development | Training is offloaded; no device master conversion required. | `load_checkpoint` reads one F32 latent at a time, retains packed/sign-scale outputs, and briefly reloads that same latent for its hash while the first remains live: up to 2L, plus packed outputs and small scale arrays. The `pack` step has a full bool sign mask plus packed output. Q/K row reorder creates copies at their smaller shapes. `GGUFReader` maps base/output files; writer retains views/packed arrays, not a second full dense model. Resident mapped pages and IO/page-cache pressure are unknown. Auxiliary deployment arrays add once each. |
| Continuous resume | Existing model/moments remain while loading CPU payload. Core `load_state_dict` copies into existing parameters. Moment restore makes destination device moments; old and replacement moment storages can overlap transiently. No second trainable master set is required on device. | Entire deserialized CPU unique checkpoint C remains until return, including moments and canonical recipe copies. Admission charges 1.25×file size; it is a heuristic, not a guaranteed allocator/RSS bound. Validation includes full-sized bool/abs temporaries and tiny isolated auxiliary probes. |
| Development offload | Training gradients are cleared before the callback. Parameters/buffers/moments are transferred to CPU preserving optimizer trainable object identity and frozen parameter aliases. CUDA originals release incrementally; synchronize/empty_cache occurs after transfer. Native process/CPU evaluation later has its own device allocation. | Current `lane_storage_bytes(cuda)` counts unique parameters/buffers/moments once and charges precisely their additional destination storage before offload. Adam CPU steps already resident. If a future caller leaves gradients live, helper omits them; current caller clears them. CPU trainer state stays resident during native export and development target/drafter load (separate 12 GiB admission). Return transfer releases CPU storage incrementally; no second full CPU trainer clone is implied. |
| Curriculum smoke | One B+I+F lane; temporary `moment_probe` reserves **2B** on CUDA, even without optimizer state. On resumed smoke, real M is already resident and probe adds another 2B. Graph/gradients also live. Probes clear on exit, not resident Adam state. | Existing provider/model host state; tiny RNG preservation for resume smoke. Current paired eager-init behavior must not be confused with this separate single-lane probe strategy. |
| Curriculum save | Model state_dict aliases live core; canonical auxiliaries copy to CPU before final admission. Existing optimizer device storage remains live throughout save. | `_cpu_tree` duplicates every tensor occurrence, including CPU auxiliaries and CPU step/RNG tensors. Raw core and optimizer alias live source and do not constitute additional device copies. For each GPU tensor, transfer temporary overlaps retained clones and new clone; exact ordered core+Adam peak above. Raw auxiliary CPU copies are already resident at the final host snapshot and their new clones add again. |
| Curriculum resume | Existing current model/moments remain initially; `_bind_phase` creates fresh activation bank/optimizer, frees old moments when references drop. `_load_model` copies core into existing masters; optimizer load transfers CPU moments to device. No full second dense model. | torch.load creates C on CPU. Current curriculum path has only a host floor before load, **no checkpoint-sized additional admission**; actual C and validation temporaries remain unmeasured. This is a documented separate measurement gap, not a second patch proposal. |
| Curriculum transition | Boundary save returns before `_bind_phase`. Core masters/scale buffers/fusion/midpoints retain identity. New six-scalar bank briefly overlaps old bank; optimizer reassignment then releases old moments. New optimizer initially has empty state; later first update allocates moments, or smoke uses temporary probes. | First save payload releases at return; second save after rebind does not overlap first complete payload. Old checkpoint file still retained on disk, not tensor RAM. Adapter/closure retention cannot be proved by shape arithmetic. |

Largest known matrix temporaries: hard-sign construction executes a bool mask,
negative F32 ones, positive F32 ones and F32 output simultaneously. During head
construction, earlier cached signs contribute 4(W-head); head operands/output
contribute 3L plus L/4 mask, totaling 1,610,219,520 temporary bytes. The saved
backward weight is the existing latent alias, not another master. Sign snapshot
construction uses a full F32 clone plus bool mask. Gradient finite checks use
full bool masks; clipped sign backward uses abs/support/cast intermediates;
non-foreach Adam has per-parameter arithmetic temporaries (including a head-sized
sqrt denominator). These are different points in execution, not a sum of
simultaneous guaranteed allocations. The JSON's 7.434 GiB known envelope sums
hard signs, gradients and snapshots conservatively and is explicitly **not a
logical lower bound**: hard signs may release during backward before all
gradients exist. Graph-held signs must never be counted again as a separate
full graph component without subtracting this known ownership.

The GGUF mapping/writer observations were read from main’s pinned llama.cpp
`9e2c7a90051e738751aab7d7bd7c2d8201fb76e3` (`gguf_reader.py:138`,
`gguf_writer.py:379–403`): little-endian writer holds references; endianness
conversion would make copies. The worker worktree has no initialized submodule.

The current `memory_estimate` gives 10,112,064,512 bytes (9.418 GiB): persistent
+one gradient+assumed 3 GiB graph/temporary allowance. It omits optional state,
step scalars and several explicit phase temporaries as named line items; the
3 GiB allowance might cover them but is not a measured bound. There is no
decisive device-cap contradiction from known storage alone. Do **not** lower any
floor or convert the estimate into fit approval. Current main's learned-head
correction `59ef557` selects serial grad-enabled learned-head training rather
than pooled head execution; captured graph/timing evidence must bind to that
selection. The ledger's persistent sizes and save finding are unaffected, while
token graph lifetimes are unresolved and must use the current execution path.

## Independent gate, checks and remaining work

The arithmetic acceptance gate is: every concrete storage is counted once per
stage, aliases free only when storage is genuinely shared, separate clone
occurrences charged separately, tiny calibration agrees, and unknown graph,
CUDA workspace, serialization scratch and allocator residency stay unresolved.
Configured tensors were never instantiated. See [validation.md](validation.md)
for independent CPU storage identities/calibration and exact commands.

Owner repeated the independent calibration on PyTorch 2.14.0 CPU using
`PYTHONPATH=src /Users/pippo/github/binary-eagle-decoding/.venv/bin/python
research/parallel20261002/memory_ledger/validation/storage_calibration.py`; exact
52/152/232/16,777,304-byte results matched the independent PyTorch 2.8.0 run.
Raw result: ignored `runs/parallel20261002/memory-ledger-validation/owner-torch214.json`.

Owner checks: `python research/parallel20261002/memory_ledger/reference/ledger.py`
regenerates the JSON; `python -m unittest discover -s
research/parallel20261002/memory_ledger/reference -p 'test_*.py' -v` verifies five
arithmetic/lifetime gates. Source hashes pin the analysis. Parent owns active
goal update, review, main integration, push, and worktree cleanup. Required next
deployment evidence remains actual phase-specific peak allocated/reserved and
host available/RSS measurements for initialization, smoke, train, checkpoint,
resume and development/curriculum transitions under the exact recipe. This
report authorizes no experiment.

## Separate proposal delta: transactional resume staging

Team 2's CPU prototype deserializes C, wraps staged core Parameters around those
CPU checkpoint storages, and aliases candidate optimizer moments to checkpoint
moments. Frozen non-core storage is shared; auxiliary activation/fusion/midpoint
candidate allocations add A. Thus proposed additional host residency is C+A
+bounded validation temporaries+RNG/metadata; live current model/moments are
already resident. No full rollback/dense-model clone is required. This is a
proposal, not current `CurriculumRunner.resume` behavior. A future CUDA commit
would require resumed core masters+moments+auxiliaries on device concurrently
with live current state until publication (approximately B+M+I for Adam core,
plus auxiliaries); host payload still remains during transfer. No CUDA fit claim
follows from the CPU prototype. Team 2’s PyTorch 2.14 tiny combined-recipe measurement after one update found
9,188 B live tensor storage, 16,484 B loaded checkpoint storage, and 428 B
additional candidate auxiliary storage: resident union 26,100 B exactly equals
9,188+16,484+428. Candidate core/moments alias loaded checkpoint storage; the
candidate also shares 208 B live frozen storage. Validation probes/hash
temporaries are additional and sequential, not covered by that resident union.
Team 2’s independent alias evidence is recorded in its own report; do not charge
shared checkpoint moments twice on host.

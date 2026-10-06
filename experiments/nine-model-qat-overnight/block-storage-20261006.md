# Incremental block campaign storage

The four remaining block candidates cannot yet be conservatively admitted to
the observed 189,381,636,096 B of free RTX5080-host disk with every output kept
there. Capture-only envelopes and the first DSpark A8 phase fit. Neither exposure
proposal is selected, and these are planning sizes, not production measurements.

The ledger excludes the already allocated F16 target, original BF16 bases,
FC/norm references, TRAIN source and indexed runtime build. It preserves the
maximum packet05 capture geometry and includes one 10 GiB free reserve.

| Envelope | 450 TRAIN prompts | 750 TRAIN prompts |
|---|---:|---:|
| Capture + reserve | 77,121,547,123 B | 110,557,799,923 B |
| First DSpark A8 phase, recent retention 3 | 144,062,520,645 B | 177,498,773,445 B |
| Four-block publication peak, recent retention 1 | 210,417,182,611 B | 243,853,435,411 B |
| Four-block publication peak, recent retention 3 | 249,707,063,187 B | 283,143,315,987 B |

The formula follows actual saver/retention and timed-export source:
`capture + fixed preparation + 4I + [4(r + 2) + 1]P + 12Z + 3 sum(G)`.
`r` is the recent-checkpoint window, with initial and 4/8-hour checkpoints
protected separately; the final 12-hour checkpoint occupies the recent window.
The extra `P` is the next writer before pruning. Reference sizes are
`I = 1,659,748,352 B`, `P = 4,911,235,072 B`, `Z = 1,625,752,030 B` for a full
projection NPZ, and `G = 2,030,864,256 / 1,875,275,904 B` for DSpark/DFlash GGUF.
Fixed preparation includes four initial GGUF/NPZ files, four fusion candidates
and four physical controls, metadata and 4,483,964,480 B for missing original Q4
controls. Subtract only actual verified existing files, not assumed allocations.

Checkpoint tensor geometry is 406,323,200 latent elements and 112,640 row scales.
Initial tensors use `4N + 8S = 1,626,193,920 B`; progressed FP32 Adam state uses
`12N + 16S = 4,877,680,640 B`. Each planning slot adds 32 MiB for serialization,
RNG and metadata. That allowance is not a measured production bound. NPZ/GGUF
temporary rename does not require a second full copy of its new final output.

Sources: `src/w1a1_eagle/block_training.py` checkpoint saver/retention;
`scripts/train_nine_model_qat.py` protected timed checkpoints and separate NPZs;
`scripts/fit_block_fusion.py` candidate/control outputs;
`scripts/evaluate_nine_model_timed_checkpoint.py` CPU disk admission. The latter
also requires fresh free disk at least twice the actual resume plus original-base
bytes at evaluation time; this is a separate phase check.

STOP/failure anchors, logs, failed staging and other new allocations remain
unbudgeted. Short actual chains may reduce capture size; these envelopes do not
prove physical impossibility. Remaining EAGLE A1 storage is also outside this
four-block ledger. No checkpoint deletion or archival has occurred.

Root observed approximately 142 GiB free on the Mac on October 6 at 01:51 PDT.
A possible operational route is byte-verified archival of completed models'
inactive intermediate checkpoints/exports before subsequent lanes. Preserve the
initial and final comparison artifacts, all raw receipts and exact source pins;
rehydrate authenticated bytes if later needed. Before any remote removal, require
an explicit per-file ledger, complete independent destination hashes, fresh disk
reserves on both hosts and proof that no active job references those files. This
route is not yet implemented or storage admission. No unrelated protected data
may be removed, and no RTX2080Ti access is needed.

Read-only dependency follow-up: after natural completed12h evaluation and owned
release, old4h/8h NPZ and GGUF payloads are outside the final collector's required
online graph. Moving just those payloads preserves normal timed-state validation;
estimated reclaim is7,313,232,572B per DSpark or7,002,055,868B per DFlash. Keep all
JSON/raw evidence online. Select by authenticated export receipts, not globs.

Old4h/8h resume files are different: `TimedEvaluation.__init__` hashes every
completed request checkpoint. They can be archived only with mandatory exact
original-path restoration before any timed recovery/reconstruction. Final
12h resume/sidecar/NPZ/manifest/GGUF/audit, original base/target/initialGGUF/audit,
fusion initializer and source graph remain final-collector dependencies.
Never rewrite settled ledgers or timed state to bypass these links. The normal
controller may finish with completed4h/8h and pending12h because it returns
after final evaluation without another trainer acknowledgment.

The larger illustrative archive subset (old two timed resumes/exports plus two
unreferenced recent slots for retention3) could reclaim about99.85GiB across
four lanes. This needs actual per-file dependency checks, sizes, independent
destination hashes, fresh reserves and restore verification; source read-only
lookup is not archive execution or storage admission. Restoring during a live
collector would violate its cached inode/stat identity checks; use a fresh
validation process after byte-identical restoration.

# Mac-only preparation checkpoint

RTX5080 remains paused following “pause all 5080 usage continue mac only.”
No remote call followed operator closure. The only earlier authorized remote job
was dependency setup; no CUDA build, model, capture, training or evaluation ran.
All subsequent evidence is Apple M3 Max CPU. No real optimizer update or held-out
quality/evaluation was performed. All owned CPU process groups have exited.

Published source/report parent `b3d02b9`, native `624f50e74` (published in the
user fork before parent gitlinks). Final independent ledger snapshot source
`566b954` has 51 file hashes, all still current; later commits are reports only.

| Completed bounded evidence | Scope |
|---|---|
| Native FFN15 and FFN15+FC16 export/load/graphs, both families/A8/A1 | Original released weights, synthetic graph inputs; private heads/embeddings preserved |
| Actual original-model CPU forward/backward | Four synthetic-input zero-update cells, finite selected/later-state/K/V gradients |
| Authentic target-only CPU TRAIN capture | Nine original prompts, three domains × train/fit/validation; six physical replay golden calls |
| Fusion fitting | Four FC-only scale fits, 96 fit + 96 prompt-disjoint validation rows, all three domains; reference magnitudes/rescue off |
| Calibrated actual-model CPU forward/backward | Four cells, one prose TRAIN block each, hard CE, exact sparse initializer/private ownership, zero steps/moments |
| Calibrated native export/load/graphs | Four FC16 cells, exact 32 NPY members, protected BF16 bytes and 16 packed W1 operations; synthetic graph inputs |
| Source/lifecycle tests | Final 172 campaign tests OK with five Linux/device skips; 78 block tests PASS; earlier 98 QAT/export regressions PASS; changed-file lint/format PASS |

Capture attempt03 passed in93 seconds; cumulative three attempts139 seconds
within1800. Four fits passed in16.3 seconds. Calibrated Torch cells stayed below
16GiB RSS; native export below8GiB. These are scoped resource observations,
not CUDA memory, speed or throughput evidence. Earlier failed attempts01/02
remain unchanged and closed; their receipt-interface defects are fixed and
independently tested. Root source/QA checks bind all resulting artifacts.

The source now includes strict generated/replay receipt semantics, data lineage,
calibrated magnitude policy, hard-forward QAT/zero-update smoke, exact optimizer/
RNG/cursor resume, export/load/graphs, supervisor/STOP/resource return, fresh SM120
admission and native evaluation/reporting. Prelaunch and post-training completion
are separate gates, preventing a first-launch deadlock. Synthetic proof cannot
satisfy actual data/model/export requirements.

The [inspectable draft packet](pipeline-draft-packet.md) is saved outside Git.
Real inspection exits0/PENDING, with102 missing field/status entries and six
unselected candidates. The packet receipt/hash table and exact command are in
that report; it includes genuine available CPU artifacts without promoting them.
The one future detached launch command is also documented there.

Production freeze and launch still need:

- Human-selected coverage, objective/recipe, long data exposure and training/
  evaluation budgets; existing proposed options are in `docs/DECISIONS.md`.
- Serious production coverage and appropriate calibration/admissions; the CPU
  pilot is bounded development evidence, not a long-QAT dataset allocation.
- Exact original frozen DSpark/DFlash Q4 files, currently absent locally;
  fresh BF16 conversions/calibrated prototypes do not replace those controls.
- Appropriate production portability inputs and freshly authorized SM120
  kernel/model/backward/full-moment-memory/resource-return checks. SM75 is a
  separate unmeasured track. Current GPU pause is preserved.

Native quality, convergence, throughput, full-L1 real-model execution and
three-domain real-model backward are not established by the above checks.
All nine aggregate and prelaunch production statuses remain PENDING.

The active project goal remains unfinished at that external-input boundary.
All owned feature/QA/operator workers are checkpointed; merged temporary
worktrees/branches are retired. Raw artifacts, CPU binaries/current and backed-up
libraries, earlier failures and QA archives remain outside Git in main. Other
teams’ worktrees and original untracked overnight files are untouched.

## EAGLE supplement — October 5, 04:36 UTC

Published `3ecb1c1`/`24a8a8a` diagnostic source and ancestry guard, `a90a9ff`
actual report, `b8bbf88` independent QA. Both direct A8/A1 CPU cells passed
18 selected finite/nonzero gradients, later-state/K/V gradients and nine packed
projection serialization checks with zero optimizer updates/moments. PeakRSS
5.80GB. Seven new focused tests and changed-file lint/format PASS. The complete
local TRAIN diagnostic is preparation-only, training-ineligible; all four
historical identity/numeric/mask/completeness gates remain unresolved. Original
unmodified pinned AngelSlim leaf modules were loaded through a private namespace;
this is not an official full-framework loader or warm-lifecycle admission.

The new actual report and additive QA ledger are tracked; large NPZ/GGUF/raw
artifacts stay under main's ignored results directory. Original 51-pin ledger
and draft identities remain unchanged. This supplement does not freeze choices
or production readiness. RTX5080 pause and Mac-only boundary persist.

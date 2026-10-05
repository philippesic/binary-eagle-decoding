# Frozen artifact read-access identity correction

Owner `/root/overnight_eagle_data`; isolated worktree
`/private/tmp/nine-model-qat-20261004/overnight-artifact-guard`, original base
`a16c0d327a06d1896b0a156f1e6e729d1a9e6159`. Bounded ownership is the shared
`Files` fingerprint/check/cache guard and the helper-file metadata comparisons
inside `_privileged_dxg_holders`, plus focused tests and this report. No other
source, process identity/holder logic or live training checkout was modified.

The coordinator reproduced the original failure on an unchanged **synthetic**
WSL file with stale access time: a successful SHA256 read updated only
`st_atime_ns`, while device/inode/size/mtime/ctime and expected content SHA
remained unchanged. The old `Files.check` compared complete `os.stat_result`
before and after hashing and falsely refused it as an artifact mutation.
This source-level observation addresses the earlier preserved bind transient
and a concrete risk when endpoint inputs are first read after a long wait.
It does not independently classify every historical failure as this cause.

The corrected shared fingerprint includes device, inode, size, full file mode,
UID, GID, link count, nanosecond mtime and nanosecond ctime. Access time is
excluded. A fresh process still hashes bytes against the external expected SHA;
a successful first read must have identical substantive identity before and
after hashing. A failed hash/read race adds no cache or identity binding.

A verified `(canonical path, expected SHA)` binding now remains immutable
within that `Files` instance: a changed substantive identity refuses even if
bytes happen to be identical, rather than accepting a new stat fingerprint after rehashing identical bytes.
Repeated access-time-only reads use the same in-process successful byte hash,
with a second identity observation before returning a cached artifact.
No stat-only trust is persisted between processes. Canonical regular-file and
exact path/SHA locator policy remain unchanged.

The fixed read-only DXG census helper uses the same fingerprint around both
its initial hash and its privileged read-only invocation. Its exact helper SHA,
fixed Ubuntu/root interpreter invocation, complete/read-only/root-UID protocol,
boot/PID namespace, holder identities and release-PENDING refusals remain
unchanged. `scripts/read_only_dxg_census.py` itself is untouched. This owner ran
no actual privileged census, WSL command, SSH, GPU query or model/data load.

Checks on Apple M3 Max/Mac CPU:

- Eleven focused tests pass: access-time mutation during hash, cached reads,
  a real stale-atime file read on the current filesystem; byte/size/replacement/
  chmod/mtime/link changes; every device/inode/permission/owner/link/nanosecond
  identity field; content and metadata races; cache-race refusal/no cache poison;
  and mocked read-only helper access-time versus substantive mutation.
- Thirty-four existing pipeline tests run successfully:32 passed and two known
  Linux `/proc` process cleanup skips. This verifies the changed shared guard in lifecycle/source/
  STOP/resume/resource/dispatch/report consumers; no model or GPU execution.
- Seven existing read-only census protocol tests pass, including wrong helper
  hash, UID, namespace, missing/denied observer and real-holder refusal fixtures.
- Changed-file Ruff/format and diff whitespace checks pass.

The first local focused run exposed two test-fixture setup errors after its
chmod case: subsequent subcases attempted to write the now-read-only fixture.
The fixture now restores owner-write permission before rewriting; all eleven
checks pass. This was a fixture issue, not a production guard waiver.

Integration: root reviews/cherry-picks this source into main and pushes it;
repin shared module/source QA in a new endpoint checkout/plan. Do not alter the
running trainer's frozen `30a` checkout or its source/config bindings. Independent
QA and an operator-owned bounded synthetic Linux read check remain the next
verification actions; they require no model or GPU work. Actual long-wait
endpoint execution remains distinct from these local source tests.

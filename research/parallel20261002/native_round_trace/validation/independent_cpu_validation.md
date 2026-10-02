# Independent CPU validation

This run exercised the adapter with independently authored records shaped like
the pinned native `w1ax_eagle_round_v1` producer. It used synthetic token IDs;
there were no model weights, captures, GPU, Metal, or remote host involved.

- **Worktree:** `/private/tmp/eagle-parallel-20261002/native-round-trace`
- **Branch/base at run:** `research/20261002-native-round-trace` at `bb0d30495571b14b1fe751facb1853c6485304e2`
- **Native source:** `third_party/llama.cpp` at `9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`
- **Environment:** macOS 27.0 arm64; Python 3.11.15; CPU only
- **Run directory:** `runs/parallel20261002/native-round-trace-validation/`
- **Exact command:** `set -o pipefail; PYTHONPATH=. /Users/pippo/github/binary-eagle-decoding/.venv/bin/python -m unittest -v tests.test_parallel20261002_native_round_trace_independent 2>&1 | tee runs/parallel20261002/native-round-trace-validation/independent-02.log`
- **Result:** exit 0; 10 tests passed in 0.001 seconds.
- **Raw output:** `runs/parallel20261002/native-round-trace-validation/independent-02.log`
- **Test SHA-256:** `5bf2b3c18395043739a79f33277a7270ccbdb104349c241e362bb840abd27747`
- **Adapter SHA-256:** `946a09df40a8851dd33f78c20ef54247656fb81a9174cb97dbfdf5663f8c8779`
- **Raw log SHA-256:** `3fe6cafa9b54101ec3cbc61db8548dc5c826208d3946770cb3a0988260938d20`

Coverage includes full acceptance with bonus, rejected-token correction,
no-proposal target output, checkpoint/replay accounting, EOS and cap
termination, leading untraced seed handling, root/cap metadata checks,
aggregate reconciliation, interval union and overlap accounting, and malformed
token/span/attempt rejection. The source semantics establish that `begin_us` is
outside `round_us`, stage times are CPU wall intervals, target decode includes
the existing synchronization, and nested stage intervals can overlap. Replay
emissions remain explicitly unresolved by role because the native statistics
exclude the reused accepted token.

**Cleanup:** the test process exited; no persistent process or device resource
was left. Python import bytecode is disposable and not part of the commit. The
raw test log remains in the ignored run directory.

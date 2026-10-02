# Native round-cost trace adapter

The initialized native source already provides a useful opt-in trace, so this
feature implements an adapter/validator and a small UNAPPLIED extension instead
of introducing another instrumentation stream. It connects emitted transcript
roots and remaining caps to the complete-session objective while explicitly
preserving missing timing/private-state domains. No native Q4_0 performance or
acceptance result is claimed.

Source inspected: llama.cpp `9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`.
`tools/server/server-context.cpp` SHA256
`ebb8b4310280b691ca5cebe37dac18a08401cd0063ca5cbccaa4724a7a5060be`.
The producing path is `emit_round_trace` and its pre_decode/decode/post_decode
callbacks, enabled by `W1AX_ROUND_TRACE_JSONL`. Verification calls actual
`common_sampler_sample_and_accept_n` in `common/sampling.cpp`: target sampling
and ordered match, with a target correction or bonus. This is not ratio acceptance
or residual sampling. Existing capture and analyzers already map task groups,
allow one missing leading seed, exclude checkpoint attempts from quality,
reconcile native counter differences, and analyze named-stage overlap.

The new adapter adds strict session validation and output-history root joins,
separates correction versus bonus versus no-proposal emissions, conserves counts,
and reconciles supplied native aggregates exactly. Replay quality roles remain
explicitly unresolved because native code subtracts a reused-token offset from
accepted count. Source records raw accepted decisions before terminal emission
truncation; these remain separate from actually emitted accepted tokens.

The additive source extension records cached prompt + pending seed, generation
position, effective remaining cap, decoder position, stop enum, and target/process
callback intervals and executed shapes. Astra source review confirmed that the
cached prompt may shift and terminal cache contains un-emitted accepted suffix;
therefore the extension does not call cache bytes the committed transcript.
Legacy target/process stage bounds are envelopes; actual callback vectors are
separate. All new producer work is under existing enabled-trace guards and no
synchronization, backend call, sampling, or cache mutation is added.
The extension changes in-memory trace struct size even when disabled; zero new
event generation/allocation/clock work is the structural claim, not measured
zero runtime overhead.

Acceptance: owner11 + independent Luna10 = **21/21 CPU tests passed**, plus
**compiled actual patched C++ emitter fixture passed** and Ruff/diff checks.
Compiled fixture: proposed3, verifier accepted3, emitted accepted2, un-emitted
accepted1, EOS1 and cap1; native aggregates proposed3/accepted3/rounds1 reconcile
exactly. Twenty arbitrary synthetic CPU-wall units contain named envelope union16
and outside-envelope gap4; producer residual is3 because named stages overlap1.
Disabled emitter second call appends zero bytes and calls no clock.
Synthetic fixture costs are not hardware latency measurements.

Environment: macOS27 arm64, system Python3.11.3 for owner suite, Apple Clang21 C++17
CPU-only stubs; independent Python3.11.15 CPU. No native full-server build, model,
GPU, Metal, SSH, captured tensor, held-out prompt, optimizer update, or new
production run occurred. Main/submodule files and gitlink were not edited.

Exact reproduction from repository root:

```
python3 -m unittest discover -s tests -p 'test_parallel20261002_native_round_trace*.py' -v
PYTHONPATH=. python3 research/parallel20261002/native_round_trace/reference/run_callback_fixture.py
```

Detailed input/output contract, limits, timing boundaries and adoption prerequisites
are in [PROTOCOL.md](PROTOCOL.md). Independent evidence is in
[validation](../../../research/parallel20261002/native_round_trace/validation/independent_cpu_validation.md).
Source and patch hashes are in
[source-binding](../../../research/parallel20261002/native_round_trace/reference/source-binding.json).
The patch is intentionally UNAPPLIED and the frozen actual GPU preparation
pipeline remains unaffected.

Remaining work: full native compilation of callback insertions before adoption;
private cache/feature/processor state identity; replay logical-round/role linkage;
external full-session clock covering prefill/bootstrap/inter-round/serialization;
fresh native ancestry/processed-output equivalence under an already admitted run.
The source extension and CPU protocol provide reviewable instrumentation work,
not an enabled training or deployment decision.

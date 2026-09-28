# Bounded RTX 5080 recurrent capture smoke

**2026-09-28 UTC.** One frozen prose training prompt ran for eight output
tokens under candidate-D own-history drafting on the RTX 5080 (SM120a),
CUDA 13.1, pinned FP16 Qwen3-4B target and candidate-D binary draft. The
target, draft and frozen training-file SHA256s were
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`,
`10e8e98e616480b25ff7600f195ba7ea3e0fd7c24832765013f960783c7609cf`
and `80e365bbc6d2caf4abd5e216e53d72ce62d80f9cf1a668e6862efb845a185e74`.
The native fork was `0abe6e586`; the checked CUDA server binary SHA256 was
`0e49bbfa0514bf037dbf1c9f5dadf959e87f5c793ebd79db5bc753f97d1df8f9`.
The host's previously recorded private CUDA/glibc compatibility include was
required to compile CUDA 13.1 against GNU 15.2; no system header was changed.
CUDA was the sole enabled accelerator. The server log confirms all 37 target
and two draft layers offloaded to CUDA0, F16 target/draft caches, A16 draft
activation mode and Flash Attention enabled.

The bounded runner captured four native rounds, 16 draft-head rows, 16 raw
target-verifier logit rows and 53 raw target-feature rows. Its exact-prefix
continuity and raw-response audits passed. All eight emitted raw token IDs
`[32, 3283, 38921, 646, 1824, 2176, 7557, 323]` matched the earlier
Apple M3 Max CPU capture of the same prompt. The CUDA feature, verifier-logit
and head-state payload hashes differed from CPU; this smoke does not set a
cross-backend numerical tolerance or prove target-feature parity. The
manifest SHA256 is
`44b189aed1f603d9c56a8143c5c4e274d0f583ce5a4fe2198205def7435f267f`.
The raw feature, target-logit and head-state payload SHA256s are
`5c78dafb0e6e9b0c03bdbe75c2231491eef0dc54d2b319ad3b9d469b3ccf3241`,
`dd276d2f14798ea2d237f0332877224a6ab6aa4978c8d273fa8be78b7ad99b15`
and `532d1b6b80fec8cbddadd3fcbe4cdabe280dda1845cc428311af828c4d2e3ceb`.

The supervised run was
`checkouts/recurrent-gpu-capture-20260928/runs/recurrent-cuda-smoke-20260928/`
under the registered remote project directory. Its `state.json` reports exit
code zero. Afterward the RTX 5080 showed 0% utilization, 1,372 MiB whole
device use, no compute app and no project server process. This was an
instrumented capture check, not a quality or timing comparison. No model
training, development/final prompt, or Q4_0 baseline run occurred.

The full frozen 96-prompt capture has not begun. The runner's former 8,192
target-logit and 65,536 target-feature row caps could truncate the 128-output
token protocol. The expanded explicit caps of 65,536 and 131,072 bound a
complete worst-case single-sequence D/D capture; actual files grow only for
rows written. A full run must still pass its manifest, request ownership,
continuity and preparation-bundle audits before any training use.

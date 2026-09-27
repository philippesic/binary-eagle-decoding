# Mixed precision rescue and frozen-body head adaptation on RTX 5080

Status: active, established 2026-09-27 UTC.

## Approved objective

Complete BOTH the four-subset Q8_0 rescue screen starting from unchanged fitted-group D and one target-aligned FP16 output-head fit on frozen D-body states, then meaningful graph-enabled native performance evaluation. Q4_0 EAGLE is primary; FP16 is diagnostic. Target remains pinned FP16. RTX 5080 ONLY; no RTX 2080 Ti connection. Existing studies remain sealed. Reserved-final prompts remain untouched.

## Ownership

Independent orchestrator chat 01a0e482-0a83-7c10-adc7-79738acb5f65, sole GPU owner. No dependency or reporting to originator. Starting main bf1f7b42b4100000d046eb640664a7e619074ff3; runtime 2e8d2e8dac6354798037e4e9aca455cd898bf06f.
Managed local worktree /Users/pippo/.codex/worktrees/binary-rescue-head/binary-eagle-decoding. Parent owns docs, decisions, integration, experiment execution/report. Sol runtime worker /root/runtime owns third_party/llama.cpp loader/graph changes; Sol /root/measurement owns new benchmark/analyzer/tests; Astra /root/alignment_advisor is read-only alignment reviewer. Workers share explicit file partitions, no independent GPU access. tmux MCP session rescue-head-5080 ($13, window @29); pane to be recorded after inventory. No owned GPU job started.

## Frozen scope and rules

Four independent swaps to standard Q8_0 from the original checkpoint: fusion; Q/K/V/output attention; FFN down only; vocabulary head. Every other D tensor unchanged. Audit actual types, payloads, scales, nonselected parameters and execution precision (Q8_1 conversion differs from D A16).
At most one combination: admit only if at least two single subsets each improve accepted/round by >=10% relative to unchanged D and each improves >=16/24 prompts. Combine the two largest pooled acceptance gains (tie: smaller weight share, then lexical name). No combination otherwise; no iterative refinement.

Quality: one deterministic pass of 24 frozen development prompts, D5/p0, max128 tokens, ctx2048, F16 target/draft KV, temperature0, seed42, no thinking, concurrency1, no prompt reuse. Include D,C,Q4_0,FP16,target-only and untrained D-body/original-FP16-head. Compare all raw emitted IDs.

Body/head factorial: Q4/D bodies and Q4/D heads plus D/original-FP16 head on common forced histories/round boundaries; preserve each body's real recurrent state/cache. Report verifier agreement/rank/margin at first/later depths. Capture true target labels at exact prefixes, positions and parent/depth, with valid and unsupported-label masks.

One head fit only: frozen D body/signs/group scales/norm/vocabulary and target; original FP16 head initialization; mapped true-verifier next-token CE plus one initialization regularizer. Only96 train prompts, max8192 valid states balanced by prompt/depth; max500 optimizer steps or45 optimizer minutes, checkpoints0/100/250/500. Hyperparameters and selection/stop rules must be frozen before fitting. Select by live native development accepted/round (tie: earlier checkpoint), compare gain beyond untrained dense head. Stop on invalid alignment/nonfinite/export mismatch. No objective or hyperparameter sweep. A failed bounded fit does not prove irrecoverability.

Performance required for all singles and selected head endpoint, D,C,untrained dense-head,Q4_0,FP16,target-only and admitted combination. Graphs enabled/verified where supported; bounded compatibility work authorized. Disable heavy dumps in primary timing; separate instrumented behavior-matched profiling. Warmups and >=5 balanced/alternating repetitions, per-request pairing, aggregate tokens/time, prompt distributions and uncertainty versus Q4. Record client wall, server prefill/decode, TTFT/stream arrivals, counts/conditional acceptance with censoring, memory/bit cost/telemetry, exact builds/hashes. Profile complete rounds and real shapes selectively. Recompute candidate-specific fixed-trajectory removable-span headroom without summing overlapping spans.

Optional depth menu D={1,3,5},p0 only if finalist reaches >=80% Q4 accepted/round at fixed policy; include Q4 matched blocks and distinguish selected from fixed results. Longer diagnostic: freeze six development IDs (first two per category in manifest), append a deterministic repeated neutral context to approximately1024 input tokens, max256 output, D5/p0; finalists/Q4 only, separately labeled. Never touch final prompts.

## State and next actions

Required operations/evaluation/scale reports and precision/readout guidance read. Local registry pause flag false at ownership establishment. Remote preflight pending. Initialize isolated remote project, implement and test mixed loader/export, alignment capture and graph/timing support. Publish tested runtime commits before parent gitlinks. Update this file at milestones with commands/artifacts/jobs and actual evidence.

13:21UTC preflight: RTX5080 16303MiB total/2115MiB used,0% util,no compute processes; pausefalse. tmux pane%29; no GPU job yet. Goal establishment3b4283e published main. Added Sol workers /root/capture (native semantic state/label capture), /root/export (audited mixed GGUF exporter), /root/head_fit (dataset/preparation and bounded trainer). No worker may use GPU/SSH. Alignment review complete: native decoder memory positionP consumes token[P+1] and predicts P+d+2; capture normalized head input and join verifier rowd, including labeled off-policy prefixes and excluding bonus. Existing scale captures are insufficient for training joins.

Head fit hyperparameters frozen before training: AdamW lr1e-4,batch64,weight_decay0,seed42; F32 master with FP16 deployment forward, explicit relative L2 coefficient0.01: mean((W-W0)^2)/(mean(W0^2)+1e-12). Select <=17 supported valid states per prompt/depth by ordered bucket centers (max8160); keep invalid/unsupported totals. Checkpoints0/100/250/500 plus last finite time-cap endpoint if2700 optimizer seconds reached. Select highest native development accepted/round; tie earlier checkpoint. No search. Export-zero and offline/native head-logit/prediction parity required before optimization; exact activation convention must match native head.

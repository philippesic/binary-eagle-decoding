# Nine model research slate checkpoint

October 4, 2026. The human requested eight research agents followed by peer
review of every report and integration of worthwhile findings into the
[nine-model recommendation](../nine-model-qat-research-plan-2026-10-04.md).
This is a read-only CPU/web planning slate, not model implementation or a new
training goal/budget. Existing A8 goal remains complete; no GPU, host-control,
remote query, weights, model load, test/benchmark or monitoring is authorized
or launched by this slate. Root owns shared plan/checkpoint/status writes in the temporary worktree.
After the read-only research, each agent received a reporting-only assignment
to persist its complete analysis in a uniquely owned Markdown file; no shared
code or implementation writes are permitted.

## Agent assignments

| Agent | Method | Scope | Status |
| --- | --- | --- | --- |
| /root/literature_quantizers | Research first, Sol high | Published low-bit quantizer, initialization and gradient methods | Complete, report saved |
| /root/literature_training_data | Research first, Sol high | Published drafter objectives, data and distillation | Complete, report saved |
| /root/literature_train_systems | Research first, Sol high | Published training execution and memory improvements | Complete, report saved |
| /root/literature_inference | Research first, Sol high | Published native inference and serving improvements | Complete, report saved |
| /root/reason_root_acceptance | Reason first, Astra medium | Root acceptance, fusion geometry and activation information | Complete, report saved; derivation preceded web |
| /root/reason_trajectory | Reason first, Astra medium | Recurrent/block propagation and data/gradient ancestry | Complete, report saved; derivation preceded web |
| /root/reason_latency | Reason first, Astra medium | Measured latency, execution and overhead | Complete, report saved; derivation preceded web |
| /root/reason_cost_quality | Reason first, Astra medium | Cheap cost/quality controls and representation freedom | Complete, report saved; derivation preceded web |

Research-first agents use primary papers/author implementations as their
starting point. Reason-first agents sent their locally derived hypotheses
before first web validation, then use intermittent primary-source checks.
All read the existing plan to avoid presenting its shortlist or completed
negative experiments as new discoveries. Final reports must separate measured
efficacy from hypotheses, identify implementation hooks/effort and cost, name
a falsifier/minimal experiment, and recommend incorporate, probe or defer.

## Root synthesis and peer review

All eight scientific reports are complete and saved as 01 through 08 in this
directory, with SHA256 and word counts in report-manifest.json. Four focused
Opus 5.5 pair audits are running and cover all eight complete texts. Root will
deduplicate related mechanisms, verify
material evidence/precision distinctions, and send every report to Opus 5.5
through peer-review MCP in bounded prompts. Use exact supported model identifier
claude-opus-5-5-high; no Fable fallback. Earlier broad max-reasoning prompt timed
out at 180 seconds, so use focused report-pair audits and a final synthesis.
Reviewer advice is not evidence of model performance. Record accepted changes,
qualifications and rejected suggestions in the report and existing plan.

Preserve exact target/verifier/capture ancestry, frozen Q4 baselines and final
split. Training and inference benefits need distinct tests. Shared runtime
changes must apply to applicable Q4 controls. A1 positive normalization gains
do not automatically change sign codes; adaptive stopping changes trajectories;
source-reported operation counts do not establish latency savings.

Raw peer output will stay under ignored runs/nine-model-research-slate-20261004
in the primary workspace. Compact eight-agent reports, peer audit summaries and
root decisions will be committed in this directory, with the updated plan and
DECISIONS. No implementation is selected merely because a report recommends it.

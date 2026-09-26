# All-layer W1Ax activation-precision study on RTX 2080 Ti

**State:** frozen measurement protocol; implementation and primary matrices complete, broader diagnostics in progress. See the [running result report](w1ax-activation-precision-results.md).
**Purpose:** separate the quality effect of activation precision from the cost of
executing the same binary weights, then judge end-to-end performance against
both ordinary FP16 EAGLE and Q4_0 EAGLE. This is a pre-QAT baseline study;
trained variants must be labeled and evaluated separately.

## Matched comparison matrix

Run these eight primary paths in one matched SM75 matrix, with a single fixed
FP16 Qwen3-4B target, unchanged verifier, and the same pinned EAGLE-3 source:

| Path | Draft arithmetic | Role |
| --- | --- | --- |
| Target-only | No drafter | Correctness and timing context; no draft-time metric. |
| Ordinary EAGLE | FP16 draft weights | Mandatory throughput anchor. |
| Q8_0 EAGLE | Standard Q8_0 draft GGUF | Weight-format control. |
| Q4_0 EAGLE | Standard Q4_0 draft GGUF | Mandatory throughput anchor. |
| W1A16 EAGLE | Binary weights, FP16 activations at each selected op | Isolate binary-weight damage without low-bit activations. |
| W1A8 EAGLE | Same binary weights, signed eight-bit activations | Intermediate activation precision. |
| W1A4 EAGLE | Same binary weights, signed four-bit activations | Bit-serial kernel candidate. |
| W1A1 EAGLE | Same binary weights, runtime-packed signs | Existing low-precision endpoint. |

"All layers" means the nine drafter-owned linears: feature fusion; attention
Q/K/V/O; FFN gate/up/down; and the 32,000-row vocabulary head. Embedding lookup,
RMSNorm, RoPE, attention score/value operations, softmax, SiLU, residuals, KV
cache work, token selection, and the target remain at their recorded ordinary
precision. Q8_0/Q4_0 name GGUF weight formats whose observed CUDA path converts
activations to Q8_1; neither is the separate whole-row/whole-token W8A8/W4A4
research quantizer. If the combined build permits, include genuine W8A8 and
W4A4 as secondary same-run controls for the effect of weight precision at A8
and A4. Do not substitute them for Q8_0/Q4_0.

## Freeze the numerical contracts before exporting

- Use one source checkpoint and identical packed weight signs and F32 row
  scales across W1A16/W1A8/W1A4/W1A1. Audit every exported row after the
  converter's layout transforms; record tensor hashes and ensure no dense
  shadow of a selected weight remains loaded or executes.
- Define A16 at the operator boundary explicitly: cast the incoming activation
  to FP16, then specify accumulator/output precision and scale order. The GGML
  graph may present F32 inputs, so merely omitting an activation quantizer is
  not an A16 contract. W1A16 therefore measures the binary-weight effect plus
  this explicit cast. On captured inputs, separately compare the ordinary
  graph with and without the same cast to isolate its numerical effect.
- Reuse the existing symmetric signed per-token absmax quantizer for A8/A4
  unless a pre-run amendment says otherwise: code ranges [-127, 127] and
  [-7, 7], nearest-even rounding, documented clipping and zero-vector rules,
  and F32 token scales. Keep W1A1's existing sign/mean-absolute rule explicit.
  Fix output dtype, bias handling, and scale multiplication order across W1Ax.
- Implement W1A16 as sign-select/add accumulation and W1A8/W1A4 with actual
  low-bit CUDA arithmetic, including a named bit-serial path for W1A4. The
  conventional W1A4 timing comparator must also be a specified real kernel
  using the same quantized operands. A dense dequantized fallback is a separate
  diagnostic and may not be reported as native low-bit speed.

## Gates before timing

1. Compare every W1A8/W1A4 integer kernel, including the W1A4 bit-serial
   path, with an independent scalar integer reference and a conventional
   same-code integer implementation: require exact INT32 dots. For W1A16,
   compare against a specified sign-add floating reference. Report
   floating-output tolerance separately from integer-dot equality.
2. Cover zero and negative-zero signs, signed extrema, clipping thresholds,
   all-zero vectors, dirty padding/tails, strided layouts, scale precision and
   order, reused scratch, all real logical K values, N=1/2 and observed larger
   N, and representative captured EAGLE activations. Check each of the nine
   projections and full-model logits/decoded IDs against its own reference.
3. On the actual 2080 Ti, prove loader and CUDA dispatch for every selected
   layer and observed shape. Preserve kernel names, precision, selectors,
   fallback evidence, and disassembly when claiming a specific instruction.
   Check no target or verifier code/precision change entered the comparison.

## Measurements to preserve

### Quality and proposal behavior

- For every request and round: proposed draft count, accepted draft count,
  emitted target/draft tokens, verification-round count, accepted/proposed
  ratio, accepted drafts/round, emitted tokens/round, accepted-prefix length,
  and acceptance conditional on depth. Record actual draft length and early
  stopping/confidence decisions; the configured maximum is not the actual
  proposal count. Preserve EOS/length behavior and raw generated token IDs,
  not only decoded strings.
- Report per-prompt, prose/code/reasoning, context-length, and repetition
  distributions. Compare each speculative path to FP16 EAGLE, Q4_0 EAGLE, and
  target-only; record first token divergence and target/verifier logits for
  known target-only mismatches. Label target-only rate ratios as timing
  observations until strict target equivalence is established.
- For the later QAT decision, record per-layer activation clipping/zero rates,
  quantization error, head top-k agreement/logit margins, and target-token
  coverage of the 32,000-token draft vocabulary. Select by online acceptance,
  not cached KL alone.

### Three levels of cost

1. **Identical-input operator replay:** capture and reuse actual activation
   tensors and layer shapes across precisions so changed draft trajectories do
   not confound arithmetic speed. For all nine linears and observed N (especially
   N=1/2 decode and N about 37/38), report scale/quantization, bit-plane
   packing, dot, rescale/output, and complete operator latency. Include
   packing-inclusive and already-packed views, launch/scratch costs, invocation
   counts, and live shape histograms. Preserve request/round IDs, layer/group
   identifiers, and CUDA graph capture/replay settings in traces and captures.
   Profile FP16, Q8_0, and Q4_0 kernels too;
   prior profiles lack a matched FP16/Q4_0 component comparison.
2. **Complete EAGLE round:** time `begin`, `process` feature extraction/copies,
   fusion and draft-context catch-up, every `draft` seed/step and sampler,
   target `llama_decode` verification, proposal checking, `accept`, checkpoint
   and KV rollback/repair, transfers/synchronization, and residual host work.
   Record both CPU wall spans and CUDA events/software traces, with defined
   boundaries and an unattributed remainder. Do not add overlapping GPU spans
   as if serial. The old `draft_ms` covers `draft()` only; `accept_ms` is only an
   acceptance hook, not isolated verifier latency.
3. **Serving outcome:** generated tokens / decode wall time; tokens / full
   request wall time including prefill; prefill duration, streaming time to
   first token, full round wall time, median/tail request and round latency,
   loaded/peak GPU memory, scratch and model bytes. Report total as well as
   per-round, per-proposal, and per-emitted-token costs. Run profiler diagnostics
   separately from uninstrumented throughput trials and quantify profiler
   overhead.

## Workload, analysis, and decision rules

- Keep the target, tokenizer/chat template, prompt hashes, thinking mode,
  context/batch/microbatch settings, F16 KV, placement, concurrency=1, sampling,
  seed, stopping rules, draft maximum/confidence policy, cache policy and output
  cap identical. Use the original 12 prompts for historical regression and the
  frozen 24 QAT-development prompts for new selection. Reserve the untouched
  24 QAT-final prompts until the **eventual trained QAT candidate and policy**
  are frozen; evaluate that final candidate once, not after this untrained
  precision screen.
- Freeze a separate context diagnostic set before execution, with actual
  tokenized prompt lengths in short (up to 256), medium (257-768), and long
  (769-1536) bins under the same 2048-token context limit. Run 32- and
  128-token output caps for each bin. Label this a context/latency diagnostic,
  separate from the QAT development/final quality sets, and preserve its hashes.
- Make maximum draft length D=5 and confidence floor `p_min=0.0` the common
  comparison. On development prompts, use the predeclared diagnostic grid
  D={1,2,3,5} and `p_min`={0.0,0.1,0.3}; keep fixed-policy and
  individually tuned-policy results separate. D denotes draft length; kernel
  K denotes the dot-product reduction dimension. Freeze any grid amendment
  before inspecting results.
- Use two warmups and at least five measured repetitions, rotating variant
  order with one server and no competing GPU job. Record clocks, temperature,
  power/throttling, utilization, free VRAM, host load, CUDA/driver/compiler,
  project/submodule commits, model and binary hashes, and exact resolved config.
- Report pooled token/time rates, paired ratios and prompt-level uncertainty
  against **both** FP16 and Q4_0, plus Q8_0 and target-only context. Repetitions
  estimate timing variability, not independent quality examples. Use the
  measured full-round decomposition for optimistic zero-draft-cost and
  break-even-acceptance screens; label them counterfactuals, not speed claims.
  State which costs are held fixed: changing acceptance can alter proposal
  length, verifier batch shapes, feature processing and cache repair, so a
  single break-even number is not a universal threshold.
- Preserve commands, raw per-request/per-round records, token IDs, activation
  captures, CUDA traces, environment snapshots, analysis version, and hashes
  under ignored run directories. Publish a compact report linking those
  artifacts. A candidate is a practical speed result only after repeated
  end-to-end measurements support gains over both throughput anchors.

The prior 2080 Ti FP16/Q8_0/Q4_0/W1A1 `draft()` times (6.471/4.995/4.178/
3.558 ms per round) remain historical checks. They do not supply missing
`process()`/verifier spans or matched denominators for a new build, so rerun
all primary paths together. Reuse byte-identical source audits, correctness
vectors, prompt manifests and old raw runs for provenance, not as replacements
for the new paired timing matrix.

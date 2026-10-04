# Independent validation: real TRAIN fusion A8 fit

2026-10-03. One authenticated CPU fit ran on Apple M3 Max, macOS 27.0 arm64,
Python 3.11.15, NumPy 2.4.6 and PyTorch 2.14.0. OpenBLAS, OpenMP, Accelerate,
MKL and NumExpr were limited to two threads; `CUDA_VISIBLE_DEVICES` was empty.
No GPU, Metal, SSH or remote process was used for this validation.

## Source operands and training eligibility

The fitted input package is
`results/fusion-binary-real-a8-20261003/data-v2/`. Its provenance receipt is
SHA256 `75cbf2b83fdcc14b2dc338a5ac2fe27188a41cd6957fa311ea7d0e6bcd857c87`;
the fit manifest is
`32db46b48d6f9737aefb2a4be01b87c7376e645bbc71972329da0f68cc338619`, and the
operand archive is
`04ad9777d23df2d42863aadde2a8cf672981bcedee83e601cae60da8e8bb6681`. The
receipt hash was independently pinned before fitting and the fitter rechecked
the receipt, original weight/base model hashes, source inventory, selected
prompt membership and every row hash.

The archive contains 384 finite raw F32 inputs of width 7,680 and the frozen
BF16 `fc.weight` promoted to F32 at shape 2,560×7,680. It has 256 fitting rows
from eight prompts and 128 independent validation rows from four other prompts;
each prompt contributes 32 rows. All twelve prompts belong to the original
TRAIN inventory. Fit and validation prompt groups and content hashes are
disjoint. The source weight and full F16 base GGUF are pinned as
`58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e` and
`c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1`.

I preserved the historical capture’s own status: its manifest says
`training_eligible:false`, `readiness:preparation_only`, and activation bits 16.
The authorization comes from the separately hash-bound readiness and native
provider records, which grant full exact-prefix training with no unresolved
gates. The provider binds the same capture manifest; its binding points to a
completed prepare-only run with no optimization run. This distinction stays
visible in the raw receipts and in the fit report. No sealed final data was
read.

I inspected the producer from the actual data-capture revision
`b4e366d4f0a30cac07f14d51c54c5b1329b3f485`, rather than the newer CPU runtime
used for loading the candidate. In that revision, `server-context.cpp` reads
the draft-declared target taps in order, writes their raw float rows without
conversion, and joins every speculative input to an exact token prefix and an
accepted/rejected disposition. `eagle3.cpp` declares a F32 fusion input and
sends it directly to `fc`; the exact source GGUF records taps `[2,18,33]`, a
2,560-wide target hidden state, and `norm_before_fc=false`. The persisted source
copies of the feature selector, capture auditor, and native step have SHA256
`8756216b7fc3ef4e4f3e24716cdbea4a2b4de422e1331830294dd81b9c8055e6`,
`5fb1c1e703bb217e77a0786d90f8c58aff7ec734449117866c3448cd4bb4b6b2`, and
`844e52095ee0518cf507c81a10a895cdab7a61e32a1cb75836f2edb70e91d1c9`.
The historical native `eagle3.cpp` and server capture writer hashes are
`b70954fe1760857a1eb441bf480bb3cbc36b7c3c1a091d38a3f1de8f14a9e9ac` and
`ebb8b4310280b691ca5cebe37dac18a08401cd0063ca5cbccaa4724a7a5060be`.

The independent row audit checked all 384 IDs and content hashes, verified a
unique native feature-row join per operand, and confirmed the ordered tap IDs,
F32 dtype, pre-encoder boundary, exact prefix tokens, and retained accepted
prefix disposition. The raw feature values therefore enter the declared A8
quantizer before the fusion layer, with no upstream cast or normalization.

## Fit and held-out reconstruction

The fitter used source SHA256
`77d5a6c60725d4130f949e4b605eedaf282ce06cf7716f461caaca84f317abe6` for the
first run and `c5f59386bace25374b32cdd986e4866863becb537806b4cda2e8181a61d6fd84`
for the byte-identical streamed-serialization reproduction. Both used the
unchanged A8 config SHA256
`e921e3afb88e98e4c6e0ef009394ba50b53ee03845961f93bd16d458088cd4cb`. It fixes
the F32 A8 contract, nonnegative row scales, two alternating passes, two scans
per pass, at most 32 accepted flips per output row, a 1 GiB workspace estimate
cap, and a 43,200 second CPU time cap.

All 2,560 rows of the matched scale-only control passed the continuous KKT and
finite exported-scale neighbor checks. Maximum relative KKT residual was
`8.06153e-16`; every row’s neighboring-scale gain stayed below its configured
objective margin. The four scans accepted 69,663 individual sign moves, each
verified as a strict improvement of the current finite SSE. The flip histogram
was 383 zero-scale rows with zero moves, one active row with 31 moves, and
2,176 active rows at the 32-move cap. Final signs differ from the original
initializer in 66,803 positions; fitted scales differ from scale-only on 2,176
rows and from the initializer on all 2,560 rows. Thus the search improved its
finite objective while hitting its declared sign-move cap on nearly every
movable output row; this result does not establish a sign local optimum.

The frozen candidate improves held-out relative squared reconstruction error
from `0.07449856` for converged scale-only to `0.05439720` for sign-and-scale,
a 27.0% reduction. The independent audit recomputed this on each held-out
prompt from the original F32 inputs and frozen BF16-promoted weights:

| Held-out prompt hash prefix | Scale-only relative error | Sign-and-scale relative error |
| --- | ---: | ---: |
| `10aca91f` | 0.0727954 | 0.0543144 |
| `d05c6422` | 0.0763759 | 0.0576805 |
| `651293cf` | 0.0757694 | 0.0536175 |
| `c204c55e` | 0.0730432 | 0.0519648 |

On the fitting prompts, relative squared error fell from `0.0753544` to
`0.0498554`.

The fusion-coordinate sign agreement and argmax agreement improved over
scale-only in the aggregate and on each of the four prompts. Against the
original binary initializer, however, the fitted candidate’s aggregate mean
row cosine was `0.66560` versus `0.66788`, coordinate sign agreement was
`73.61%` versus `74.25%`, and argmax agreement was `10.16%` versus `15.63%`.
These are fusion-output coordinates, not vocabulary decisions. Reconstruction
does not establish native draft acceptance.

The initial packed candidate SHA256 is
`96ac80053802ae01ad6c7b9a8c328adbe4386da9d5807ebdca215c483373320b`; the
full fusion-only GGUF SHA256 is
`b9eae46c19b95ca904855c403c67b2b0d0d8134040b97c9c089e96ab50a68574`. The
packed-sign array SHA256 is
`698ff8f5819ef630c258804ca0fb2914e12c122b6150a745d9d8e07ca7443931`; the
fitted row-scale array SHA256 is
`b08b777148d105fc40fec7f558b6a107d6e1e4da19184c49f8db45b439faa722`. The
final reproduction has the exact same NPZ, control-scale NPZ
(`15a64f5b6b093246eeeea2834791a1ba91e1ba1fbb62077b4fd46e5bb0302aeb`) and
GGUF hashes. The run 1 fit report SHA256 is
`695885607a363548edff651371b69d1cb99e1bbc237a17502077baec072b2584`; the run
2 report SHA256 is
`55ff66daddb62ee6b9cc0491396425939425a1ec2e330f613dbde8543b38a701`. The run
2 fitter and streaming exporter source hashes are
`c5f59386bace25374b32cdd986e4866863becb537806b4cda2e8181a61d6fd84` and
`49fe27f700f0b02a72b52634aa331480c07b9a24bbf35a581a717bfba10ff262`.
Candidate/reloaded signs and F32 scales match; the exporter
preserved all 13 nonfusion tensors and every original metadata value apart
from GGUF structural tensor/KV counts. A separate streamed verification with
1 MiB payload chunks confirmed the identical GGUF byte stream, all 13
nonfusion payloads and original KVs at 248 MB peak RSS. The full CPU model
loader accepted both the base and candidate. The native CPU A8 operator replay
checked 20,480 outputs with zero absolute and relative error on Apple M3 Max.
These checks validate format and arithmetic on this CPU, not SM75 performance,
end-to-end acceptance or throughput.

The bounded streaming serializer report SHA256 is
`8f4392ca0d2976462c3d8764443cc539dc6a8451e56032f4e352bd364a6106dc`; the
native operator replay report SHA256 is
`363c98b1023bc5fddaca2d1ccdfc5878d44ae6b90ef087975a9c62b9ab369ce7`. The
data capture used native source commit `b4e366d4f0a30cac07f14d51c54c5b1329b3f485`;
the candidate CPU loader/replay used llama.cpp commit
`9e2c7a90051e738751aab7d7bd7c2d8201fb76e3`.

## Commands, memory and cleanup

Both runs used `/private/tmp/eagle-fusion-real-report` as the working directory
and were wrapped in `/usr/bin/time -l`. The full argv, environment overrides,
hashes and hardware are preserved in
`results/fusion-binary-real-a8-20261003/fit-real-01.command.json` and
`fit-real-02.command.json`; both records were written before launch. The common
command form was:

```sh
env EAGLE_GGUF_PY=/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  /usr/bin/time -l /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  scripts/fit_fusion_binary_discrete.py \
  --manifest /Users/pippo/github/binary-eagle-decoding/results/fusion-binary-real-a8-20261003/data-v2/fusion-manifest.json \
  --config /private/tmp/eagle-fusion-real-report/configs/fusion_binary_discrete_a8.json \
  --source-weights /Users/pippo/github/binary-eagle-decoding/models/hf/Qwen3-4B_eagle3/model.safetensors \
  --base-gguf /Users/pippo/github/binary-eagle-decoding/models/gguf/Qwen3-4B-eagle3-f16.gguf \
  --provenance-receipt /Users/pippo/github/binary-eagle-decoding/results/fusion-binary-real-a8-20261003/data-v2/provenance-receipt.json \
  --provenance-receipt-sha256 75cbf2b83fdcc14b2dc338a5ac2fe27188a41cd6957fa311ea7d0e6bcd857c87 \
  --output-dir /Users/pippo/github/binary-eagle-decoding/results/fusion-binary-real-a8-20261003/fit-real-02
```

The final independent audit ran from `/private/tmp/eagle-fusion-real-validator`
with the same two-thread CPU limits:

```sh
env EAGLE_FUSION_FIT_SCRIPT=/private/tmp/eagle-fusion-real-report/scripts/fit_fusion_binary_discrete.py \
  EAGLE_FUSION_REAL_RUN_ROOT=/Users/pippo/github/binary-eagle-decoding/results/fusion-binary-real-a8-20261003 \
  EAGLE_GGUF_PY=/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp/gguf-py \
  OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES= \
  /usr/bin/time -l /Users/pippo/github/binary-eagle-decoding/.venv/bin/python \
  -m unittest -v tests.test_fusion_binary_real_independent

/Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff format \
  tests/test_fusion_binary_real_independent.py
/Users/pippo/github/binary-eagle-decoding/.venv/bin/ruff check \
  tests/test_fusion_binary_real_independent.py
```

Final test output and Ruff output are retained in
`results/fusion-binary-real-a8-20261003/independent-validation/` outside Git.

Run 1 used the same arguments with `--output-dir .../fit-real-01`, fitter
source SHA `77d5a6c6...`, and wrote stdout/resource logs beside that directory.
It completed successfully in 11.67 seconds, but `/usr/bin/time -l` measured
2,796,650,496 bytes max RSS against the 1 GiB workspace cap. Its report records
2.44 seconds in the fitting routine, while whole-process timing also includes
operand loading, export and serialization; that run did not have phase RSS
telemetry, so the peak cannot be assigned to a subphase.

Run 2 used fitter SHA `c5f59386...` with streamed GGUF serialization. It
completed successfully in 5.39 seconds with 770,818,048 bytes max RSS, 0 swaps,
and the exact same frozen candidate, controls and GGUF. Its phase telemetry
places peak RSS at `validation_complete` before export; streamed payload write,
reload verification and report writing added no observed RSS growth. The
reproduction execution record SHA256 is
`7993b65cd393ea4c38498c1327cb34b2fb3bed5e96c4a398329cc72cc9870b5b`.
Neither process retained a session or remained running.

The run 1 command and execution records have SHA256
`8bc5118b4aaa80eb8f282dfecaea44d177765946acd48c3123022624928a96bf` and
`a09dc38ccd8ca40aae16846d39c390638c64d5626c5710c5fcbd54531c53c057`. The
run 2 command record SHA256 is
`e9e782fa641cbf00fe010777dd4ac918874b5e1843e4ce06026a64d8f6854209`; its
stdout and timing logs are
`71daf526af9757dc32cfd05a069279fa45fb991dae8e0f4180c6433bb71b55b7` and
`ef8c8e6cd7478ef774a816202296143219c72524e5b4572c25abd32378126569`.

An earlier independent test pass used full in-memory `GGUFReader` array
comparisons and reached 3,122,364,416 bytes max RSS. It also exposed two test
expectation defects: the GGUF structural counters change when fusion metadata
and packed tensors are added, and the manifest pin string in the test had a
typo. The final audit replaced that comparison with the bounded 1 MiB streaming
verification above. The final independent unittest suite passed 6/6 in 0.97
seconds at 701,399,040 bytes max RSS; Ruff passed. These first two over-cap
observations remain recorded; not every audit step stayed under 1 GiB.

The final test source SHA256 is
`5256905a567892d0f894da3e0dae176b82e2a7e42f72ee50832c970fc480b7c3`; its
Ruff log SHA256 is
`82b3e6a6c090a57601d22943bd23fca9218d1031dbe5a7b754092f9a156b4f18`, and the
final unittest log SHA256 is
`fcaacede7e401621e9fa8464d6501b89de6c4c8d4cedcace75126727a4cf7d1a`. The
independent per-prompt validation metrics have SHA256
`21cc352fd3b9a57a9397f6b628dacd586d7336c6bcb7dd7785d94be99723bbbe`.

All fit processes exited successfully and a follow-up process query found none
running. Model files, capture packages, candidates and raw outputs remain in
ignored `results/`; only this report and its independent test source are in
Git. No native target requests, Q4_0 comparisons, or throughput experiments
were run.

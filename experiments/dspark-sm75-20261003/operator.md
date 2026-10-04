# RTX2080Ti operator record

Operator: Luna. All RTX2080Ti access was through the tmux MCP transport to the
registry host alias `rtx2080ti` in
`~/.config/binary-eagle-decoding/hosts.toml`; no ordinary-shell SSH, SCP, or
rsync was used. Registry user `philip`, port `22`, workdir
`~/binary-eagle-decoding`, initial pause flag false. The current pause is recorded below. The registry address remains
machine-local; only the host alias is recorded here.

## Host and initial admission evidence

Read-only commands, issued over the tmux MCP pane (DS_SCREEN_HOST is the registry host; exact commands remain in ignored operator evidence):

```sh
ssh -p 22 philip@"$DS_SCREEN_HOST" 'hostname; date -Is; uname -a; nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw --format=csv; nvidia-smi; ps -eo pid,ppid,pgid,sid,stat,etime,cmd --sort=pid | head -100; tmux -L binary-eagle-runtime ls 2>&1'
ssh -p 22 philip@"$DS_SCREEN_HOST" '/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw --format=csv'
```

The host was `DESKTOP-T4JKGJB`, WSL2 kernel `6.18.33.2-microsoft-standard-WSL2`.
`nvidia-smi` is `/usr/lib/wsl/lib/nvidia-smi` (not on PATH). Hardware is an
NVIDIA GeForce RTX 2080 Ti, UUID `GPU-35b7b96c-d577-95f0-0050-40699c9faef7`,
11,264 MiB total, 522 MiB used, driver 610.74, P8, 28 C, 300/405 MHz, 17.57 W.
The only reported GPU context was Xwayland PID 29; it used about 522 MiB. This
is an observed baseline, not a claim that the GPU is free. `/dev/dxg` exists.
No project runtime tmux server existed at the first check. No model was loaded.

Current immutable checkout is under the registered project directory:
`~/binary-eagle-decoding/runs/checkouts/dspark-sm75-20261003-current`, detached
at published parent commit `c4eb52b85253e82d58d4c6d2c8b937c18ee0882a`, clean;
the root `/runs/` ignore rule covers it. The earlier setup checkout at
`~/binary-eagle-decoding-worktrees/dspark-sm75-20261003-32cbdd8` was created
before correcting its location. Its two CPU durability-probe run directories
are preserved there until copied and verified under the registered workdir.
The first probe failed immediately from shell escaping (`SyntaxError`) and is
retained as failure evidence; the second succeeded.

## Fixed target and existing native baseline

From `~/binary-eagle-decoding`:

```sh
for f in models/gguf/Qwen3-4B-f16.gguf models/gguf/Qwen3-4B-eagle3-f16.gguf models/gguf/Qwen3-4B-eagle3-q4_0.gguf; do stat -c '%n %s bytes %y' "$f"; sha256sum "$f"; done
build/llama-cuda/bin/llama-cli --version
grep -E 'CMAKE_BUILD_TYPE:|CMAKE_CUDA_ARCHITECTURES:|CMAKE_CUDA_COMPILER:|CMAKE_CXX_COMPILER:|GGML_CUDA:' build/llama-cuda/CMakeCache.txt
```

| Artifact | Size | SHA256 |
| --- | ---: | --- |
| Target `Qwen3-4B-f16.gguf` | 8,051,285,280 B | `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6` |
| Q4_0 EAGLE primary draft | 128,988,160 B | `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280` |
| FP16 EAGLE diagnostic draft | 442,700,800 B | `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1` |

Existing CLI reports llama.cpp `0.5.0-dev` build 11154, commit `34e21b7d8`.
Existing `build/llama-cuda` is Release, CUDA enabled, architecture 75; compiler
paths are under `runs/toolchain-bootstrap/env/bin`. Its source/build are read
only for this task. `cmake` and `nvcc` are not on PATH; the configured CUDA
compiler is `~/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/nvcc`.
Initial `nvidia-smi` free capacity was about 10.7 GiB. The on-disk FP16 target
plus release drafter leaves little room for CUDA/runtime allocations at the
frozen context of 2,048; no load or memory pass has been claimed. Retaining
either draft embedding or output head may raise residency substantially, so
the released-vs-frozen target tensor equality receipt is required before
drop/borrow decisions.

## WSL durability

WSL version queried through the absolute Windows PowerShell executable from the
tmux MCP path: `wsl.exe --version` reported WSL 2.7.14.0 and kernel
6.18.33.2-2. `/mnt/c/Users/philip/.wslconfig` and `/home/philip/.wslconfig`
were absent before this task. With no file to preserve, the approved durability
setting was created at `/mnt/c/Users/philip/.wslconfig` with bytes
`[general]\ninstanceIdleTimeout=-1\n`; SHA256
`4808e110c79c1924e9c61bb9ff8dd5683230f68a6ae8a56202cb0383f630bbe1`.
It will be read on the next WSL startup. The current-boot CPU proof below
passed before this setting was created; no WSL-wide shutdown/restart was
performed. Root accepted proceeding without restart and requires the startup
caveat to remain explicit.

CPU disconnect proof used the project supervisor in a detached Linux tmux
session, not a foreground SSH process:

```sh
tmux -L binary-eagle-runtime new-session -d -s dspark-durability-proof-2 -c "$PWD" "exec python3 scripts/remote_job.py dspark-sm75-durability-proof-20261003b -- sleep 100"
```

Supervisor state: `~/binary-eagle-decoding-worktrees/dspark-sm75-20261003-32cbdd8/runs/dspark-sm75-durability-proof-20261003b/state.json`;
supervisor PID/PGID 1114, child PID/PGID 1117, start `2026-10-03T22:57:56.747307Z`,
finish `2026-10-03T22:59:45.574331Z`, status `finished`, exit 0. SSH was closed
for over 80 seconds after launch and reconnected after the 100-second child
finished. The earlier failed probe is in sibling run
`dspark-sm75-durability-proof-20261003`; it exited 1 at launch, with its raw
SyntaxError in `stdout.log`. Both complete run directories have been copied
under `~/binary-eagle-decoding/runs/dspark-sm75-operator-evidence-20261003/`;
the initial external worktree remains intact as well.

## Released model download (complete)

Source owner pinned these immutable files:

| Model | Revision | Expected bytes | Expected SHA256 |
| --- | --- | ---: | --- |
| `deepseek-ai/dspark_qwen3_4b_block7` | `3457dff1417cb84927f6098a5fcb7cee85c934b7` | 2,786,273,970 | `f9e31587608441f235d46410e7201f8cb1647be5cd077065c89cc02c37ae86a7` |
| `deepseek-ai/dflash_qwen3_4b_block7` | `02d530b7962ea1412beaf41a05c0b8e36d5f9b1d` | 2,630,685,488 | `68d4138ec35864c47c4856d9f54004f1299d151818c33eacafd8ba19aad9cb9f` |

`curl -sSI -L --max-time 30` validated HTTP 200 after redirects and matching
`x-linked-etag`/content lengths for both pinned revisions. Download is detached
under remote `tmux -L binary-eagle-runtime`, session `dspark-release-downloads`,
supervised by `remote_job.py` run ID `dspark-sm75-release-fetch-20261003`.
Exact child command from its `state.json`:

```sh
curl -fL --retry 5 -o runs/dspark-sm75-release-fetch-20261003/dspark_model.safetensors https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7/resolve/3457dff1417cb84927f6098a5fcb7cee85c934b7/model.safetensors --next -fL --retry 5 -o runs/dspark-sm75-release-fetch-20261003/dflash_model.safetensors https://huggingface.co/deepseek-ai/dflash_qwen3_4b_block7/resolve/02d530b7962ea1412beaf41a05c0b8e36d5f9b1d/model.safetensors --next -fL --retry 5 -o runs/dspark-sm75-release-fetch-20261003/dspark_config.json https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7/resolve/3457dff1417cb84927f6098a5fcb7cee85c934b7/config.json --next -fL --retry 5 -o runs/dspark-sm75-release-fetch-20261003/dflash_config.json https://huggingface.co/deepseek-ai/dflash_qwen3_4b_block7/resolve/02d530b7962ea1412beaf41a05c0b8e36d5f9b1d/config.json
```

Both safetensors downloads completed in `dspark-sm75-release-fetch-20261003`
with exit 0; the terminal supervisor state is under
`runs/dspark-sm75-release-fetch-20261003/state.json`. Exact expected file sizes
and SHA256 values matched. Config files were also downloaded and hashed.
SHA256 job `dspark-sm75-release-hashes-20261003` completed exit 0. The artifact
paths are:

```text
~/binary-eagle-decoding/runs/checkouts/dspark-sm75-20261003-current/runs/dspark-sm75-release-fetch-20261003/dspark_model.safetensors
~/binary-eagle-decoding/runs/checkouts/dspark-sm75-20261003-current/runs/dspark-sm75-release-fetch-20261003/dflash_model.safetensors
```

Config SHA256s: DSpark `494e5665481ff8216c4e857a531c401f72b7fb795434a8201f45252c0ad7568a`;
DFlash `e4d3ad2f79210e91cf8b4d78547d23afda9a8cd2a90c4c6b97c9ea617d8f94f1`.
No model load has started.

## Native CUDA build (prepared)

Authorized native source repository:
`https://github.com/philippesic/llama.cpp.git`, branch
`feature/dspark-admission`. Final approved commit is
`76847aa877261817026e2f29472f080235b6a829`, clean detached source at
`~/binary-eagle-decoding/runs/checkouts/llama-dspark-admission-76847aa`. Earlier
clean commit `a4884f5dd15f7dcdd2d9c16a8d754467a776dfb3` remains preserved at
`~/binary-eagle-decoding/runs/checkouts/llama-dspark-admission-a4884f5`.

First supervised configure attempt used bootstrap CMake/nvcc directly and
failed because Ninja was pointed at the default Unix Makefiles generator; it is
preserved as `dspark-sm75-native-config-a4884f5-20261003`. A second attempt
added `-G Ninja` but still failed at CUDAToolkit discovery; its log is
`dspark-sm75-native-config-a4884f5-ninja-20261003`. It reported CMake 4.4's
missing `cuda_runtime.h` at `env/include` (actual header is at
`env/targets/x86_64-linux/include`) followed by unknown helper
`_CUDAToolkit_find_and_add_import_lib`. A third attempt passed toolchain paths
relative to the base checkout while `remote_job.py` had changed cwd to the
isolated checkout, so it ended in `starting` without spawning a child; no
process remained after verification. Its state is preserved at
`dspark-native-config-76847aa-mamba-20261003`.

The known-good activation from
`experiments/rtx2080ti-quantization-suite.md` uses
`runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env env`
with explicit `CC`, `CXX`, and `CUDAHOSTCXX`. This resolved CUDAToolkit 12.8.93
and configured successfully. Supervised configure run:
`dspark-native-config-76847aa-abs-20261003`. Build directory:
`~/binary-eagle-decoding/runs/builds/dspark-native-76847aa-mamba-abs-sm75`.
Build completed successfully, 356/356 Ninja targets, in remote session
`dspark-native-build-76847aa`, supervised run ID
`dspark-native-build-76847aa-20261003`, four parallel jobs. Exit code 0. The
binary reports `0.5.0-dev (build 1, commit 76847aa)`, GNU 13.4.0; its SHA256 is
`a048ca99a13a0a2e847ce12e827f5bb0e39daa1cfa2dd5f6ebf068f7e103a60a`.
All 356 targets and the final link are preserved in `stdout.log`. CUDA emitted
only its existing Conda `compiler-bindir` redefinition warning. Process group
3531/3533 and direct PIDs are absent after completion; `nvidia-smi
--query-compute-apps` is empty and the GPU remains at its 549 MiB Xwayland
display baseline / 0% utilization. The build did not load a model.

Canonical source checks first used published parent helper commit
`c1b1071f92f67b2d38cc1a88b978c4190eed4177`, staged clean under
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-c1b1071`. The guarded
replacement is published parent commit
`a64b65d171ecebe7c778a63b3caea06d93cb6ddf`, staged clean under
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-a64b65d`. Its scripts are
`scripts/dspark_screen/compare_frozen.py`, `check_export.py`, and
`validate_native.py`. Existing conversion environment is
`~/binary-eagle-decoding/runs/toolchain-bootstrap/convert-env`, Python 3.12.14,
NumPy 2.2.6, safetensors 0.8.0. `compare_frozen.py --help` passed under the
supervised tools run `dspark-admission-tools-20261003`.

Independent supervised inventory (`dspark-target-tensor-names-20261003`) found
target HF `models/hf/Qwen3-4B/config.json` has `tie_word_embeddings=true`;
target config SHA256 is
`8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a`. The target
GGUF contains `token_embd.weight` F16 shape `[2560,151936]` and no
`output.weight`. Released DSpark has both `embed_tokens.weight` and
`lm_head.weight`, each BF16 shape `[151936,2560]`; its own config does not set
the tie flag. Native source `76847aa877261817026e2f29472f080235b6a829`,
`src/models/qwen3.cpp`, creates optional `output.weight` and when it is absent
creates output from `token_embd.weight` as a duplicated tensor. The first
comparator run failed on missing `output.weight` and is preserved at
`runs/dspark-canonical-dspark-compare-20261003/stdout.log`. The guarded helper
commit `a64b65d171ecebe7c778a63b3caea06d93cb6ddf` requires the pinned target
config, target architecture/tie flag, and exact selected Qwen3 loader fallback
before using target `token_embd.weight` for the head comparison. It was run on
both source checkpoints. Each report found 77,045 differing values of
388,956,160, max absolute difference `2.9802322387695312e-08`, 451 BF16-to-F16
underflows, and zero nonfinites/overflows. In both reports released embed and
head canonical hashes were identical to each other
(`75599dcbc3345b7aa22c4db6be909c5ae109a710cd3e15d842c5e7bcd00620aa`), while the
fixed target hash was
`33f5c1db6eb5f998ec7f5ecc4ea37f36aab01183c5f2caa55333703ac57dd982`; exact
borrowing is false for both roles.

DSpark comparison run `dspark-canonical-dspark-v2-20261003` exited 0; report
SHA256 `545ce55da1a96218288277b09fec4b1212d7a6b4196607ab55ed27afe2e46269`,
copied config SHA256
`e403743c27eeb271c505212b9999d0734cd950458b669a95941a612f9e465d89`.
DFlash comparison run `dflash-canonical-dflash-v2-20261003` exited 0; report
SHA256 `dcceb28b14baae88453ec28957818115013ffbf304e6e9b24ddd5f9c666ca0ac`,
copied config SHA256
`bf7deaec638cb7a0bebc1738b5f35b6d855a6b82ebc764ce7531cfa8afa470e1`.
Both copied configs retain the original BF16 `embed_tokens.weight` and
`lm_head.weight` with `has_embed_tokens=true`, `retain_lm_head=true`; the fixed
target and released source files remain unmodified.

Separate conversion-input directories under the ignored current-checkout
`runs/` contain each copied `config.json` and `model.safetensors` symlink to the
pinned immutable source file. Both BF16 conversions used native source 76847aa,
existing convert-env, `--outtype bf16`, and the unchanged target tokenizer.
DSpark converter job `dspark-release-convert-20261003` exited 0, 64 tensors,
2.79 GB file, draft SHA256
`dc5299bdb1e7e906b003334a9b806ba502e72111341432ac5a9fd11de22f520c`. DFlash
converter job `dflash-release-convert-20261003` exited 0, 60 tensors, 2.63 GB
file, draft SHA256
`92925d9e4be49a82d1e4f9d9671aae0e3647c1146fccbd4004bfc8afe0c9896f`.

DSpark export audit `dspark-release-export-audit-20261003` passed. It records
2,786,331,140 native tensor bytes, BF16 matrices, no borrowing, and retained
embedding/head raw hashes equal to source (`eabe5625fc0575bf517c424041e9701c0fd521889e0f547c8522d2aa20e8c0f8`).
Export report SHA256 `b0aa8f6686a822258be033707242c641cd8ef757f4dc90784580e82c774d4ae9`.
DFlash audit `dflash-release-export-audit-20261003` also passed, 2,630,743,040
native tensor bytes; its retained embedding/head raw hashes match source with the
same value. Export report SHA256
`fa96afef39453b9bec6bac3a874681c0c878a103dc7bbd994250d060a49c3463`.

The initial full throughput/probe checkout was
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-03f6c81`, parent commit
`03f6c81a247407cd3c308a8fcd65d9e63d8ee5d8`, with protocol SHA256
`6244571641791b4f47e2109295f9e28e2827d2c63c290670548f60893c16cf3c`. Later
root updates froze common batch/ubatch32 and mask validation in complete parent
checkout `~/binary-eagle-decoding/runs/checkouts/dspark-screen-84c1f24`, commit
`84c1f24b058d0367917aa09185fded9e1fa51138`, protocol SHA256
`367431663597d312bf4cc75544d3e8c1994260a0d3cd266558cdd388b48ceca7`. The later
probe and baseline readiness runs below used 84c1f24; the earlier 03f6c81 and
d9197c9 failed run artifacts remain preserved.

## GPU process/resource state

At initial admission the only context was Xwayland PID 29, 522 MiB. No project
GPU process was launched. Repeat `nvidia-smi` and process/context inspection
before the first load, record peak memory for every arm, then inspect the exact
supervisor process group and `nvidia-smi` contexts after each stop. Do not call
the GPU free until project groups and contexts are absent.

Before the authorized probe could start, a fresh local
`python3 scripts/agent_env.py status` showed
`rtx2080ti.pause_requested=true` (updated `2026-10-03T22:52:45.606236Z`). No
probe Popen, model load, or benchmark request was issued after observing this.
Read-only remote verification at `2026-10-03T23:54:24Z`: no
`binary-eagle-runtime` tmux server, no project `llama-server`, supervisor,
converter, comparator, export checker, or probe process in `ps`, and
`nvidia-smi --query-compute-apps` returned only its header. At that check the
RTX 2080 Ti remained at the Xwayland baseline, 549 MiB / 11,264 MiB and 0%
utilization. This was the paused state before the user's explicit resume.

## Resumed probe attempts and current preparation (2026-10-04)

The human explicitly renewed study-only access to `rtx2080ti`; no RTX 5080 or
other task was touched. The local pause status was rechecked before work. The
full immutable 03f6c81 probe attempt `dspark-sm75-admission-probe-20261003`
loaded only the frozen FP16 target, completed warmups and both prompt requests,
and preserved target device hashes. It then failed the one-word EOS validator:
raw response content was `YES`, token IDs were `[14004,151645]`, and the native
stop/EOS fields were correct, while the convenience message content included
empty `<think>` markup. The root-reviewed probe correction was integrated in
complete checkout `d9197c90c8dce59afb7e3bf4e5a24190418c26af`.

One retry, supervised run `dspark-sm75-admission-probe-eos-retry-20261003`,
used that immutable checkout, original pinned assets/config, and server teardown
grace 30 seconds. It preserved the corrected EOS result and the partial native
records. DSpark proposal-3 exposed the author-output-reservation defect
(requested seven noise positions but reserved four; it emitted zero proposals).
DSpark proposal-7 emitted nonzero proposals but was only partial proof. The
retry then loaded DFlash with inherited batch/ubatch 2048/512. CUDA memory
exhausted during a later VMM reserve: target compute buffer had grown to 599
MiB; at failure GPU accounting was total 11,263 MiB = free 0 + process 8,057
MiB + unaccounted 3,205 MiB. The server aborted with SIGABRT, and the supervisor
removed its group in 0.164 seconds. No admission receipt was issued. Both probe
runs and their failures remain intact; this retry is not an architecture or
throughput result.

Root froze `--batch-size 32 --ubatch-size 32` for all arms, preserving the
2,048 context, 128 output cap, FP16 target/KV, and verifier behavior. The
reviewed native capacity fix is pinned at `fcdf5822c5b78f9dbcfd1f5c7106f09c3f0c9b1a`.
An isolated source checkout is staged at
`~/binary-eagle-decoding/runs/checkouts/llama-dspark-admission-fcdf582`; a
separate supervised CUDA build ran under tmux session
`native-fcdf-build`, run ID `native-fcdf-build-20261004`, with build directory
`~/binary-eagle-decoding/runs/builds/dspark-native-fcdf582-mamba-abs-sm75`.
The build uses the existing activated toolchain, CUDA architecture 75, and
four Ninja jobs. It completed 356/356 steps and the supervised run exited 0.
The built server reports `0.5.0-dev (build 11206, commit fcdf5822)` and has
SHA256 `1bd67cdf74d6ced49454ca2546d9a462e71d4e695b8fb5e52d7359fa68473822`.
The prior 76847aa source/build is preserved unchanged.

FFN-only Q4_0 CPU candidates are being staged under the supervised workflow;
these quantize only the 15 five-block FFN gate/up/down matrices and preserve all
other BF16 source tensors. DSpark candidate:
`runs/precision-q4-20261004/dspark-ffn-q4.gguf`, 2,255,282,336 bytes.
DFlash candidate:
`runs/precision-q4-20261004/dflash-ffn-q4.gguf`, 2,099,693,984 bytes.
DFlash quantization run `q4-ffn-dflash-quantize-20261004` exited 0; its log
reports model size 2,508.87 MiB to quantized size 1,996.76 MiB in 2.022 s.
Both quantizer runs hid CUDA devices and did not use the GPU. The old DSpark
precision receipt check failed only because GGUF reported
`conf_proj.weight` as `[2816]` while the source was `[2816,1]`; all non-FFN
types/bytes matched. This failure is preserved. Native owner published helper
fix `52bb528f3fe3072b7c01a5649a40b59be97700c4`, accepting only equal
bytes/types/native extents with a trailing singleton collapse; root integrated
it in 84c1f24. Source-bound FFN-only Q4 checks passed for both candidates under
supervised runs `q4-check-dspark-20261004` and `q4-check-dflash-20261004`, using
the convert-env Python and fcdf source. Both reports have `passed=true`, 15 Q4_0
FFN matrices, and immutable non-FFN tensors. DSpark report SHA256 is
`766a810b2f0bd0d5ac9a44bd775804e85708d1eb0be9bbdfbb1b03d51c699130`; DFlash
report SHA256 is
`880d696977fdea0f111615d5dd53fffc8c7488dae4cce35e572815688aa4bd44`.
Candidate hashes are DSpark
`a95358608ce2e1c77f9a8253b692e697e3745dd68ba66b859833d41c480193fe` and DFlash
`6f23297bb623217e3df6352c8e83521f59f3b307b26f27f1262269775552b7c3`. DSpark's
only shape normalization was `conf_proj.weight` source `[2816,1]` to candidate
`[2816]`, with native extents `[2816,1,1,1]`; raw BF16 bytes and type matched.
DFlash had no collapsed singleton. Earlier failed Python/env/path runs and the
pre-fix singleton failure remain in their own supervised run folders.

## Baseline readiness and final native probe (2026-10-04)

Baseline readiness used 84c1f24's `baseline_probe.py`, config
`runs/baseline-readiness-fcdf-20261004/config.json`, and supervised run
`baseline-probe-fcdf-job-20261004`. The config pinned only the final fcdf server,
fixed FP16 target, existing Q4_0 EAGLE baseline, and current protocol; it had no
candidate-admission flag. It ran two target-only and two Q4-EAGLE warmups,
two development prompts, and the raw YES/EOS fixture with 128-token greedy
requests. Both server groups (target PID/PGID 20386, Q4 PID/PGID 20466) returned
0. Target-only memory loaded at 8,511 MiB and Q4 EAGLE at 8,681 MiB; Q4 generated
1,289 draft proposals and accepted 246 over this readiness smoke. The raw EOS
fixture passed in both arms: IDs `[14004,151645]`, raw content `YES`, native EOS
stop. This smoke is not a throughput denominator or candidate admission.

Its `readiness.json` marks output IDs unequal for prompt-00 and its warmup at
token89 (target 72,499; Q4/speculative verifier 9,920); prompt-01 and EOS match.
Repeated requests are deterministic within each arm. The same discrepancy
later appeared in every candidate arm, pointing to a shared target verification
path difference. Preserve this as an unresolved exactness finding.

After a fresh pause/process/context check, the authorized one-shot native probe
used config SHA256
`b9a1f5d83ee0aff9b0ef56ef85e2ce4ec5edb633aba1d2a96d21bf681b502b38` and supervised
run `dspark-native-probe-fcdf-20261004` with teardown grace30. Its only config
changes from the prior pinned probe were the fcdf binary/hash and 84c1f24 B32
protocol hash; target, released BF16 drafts, canonical reports, exports, and
source hashes stayed pinned. The output directory is
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-84c1f24/runs/native-admission-fcdf-20261004/results`.

All five target/DSpark3/DSpark7/DFlash3/DFlash7 servers loaded and exited 0;
all had actual proposals, injections, and nonzero acceptances. DSpark3 logged
203 author-noise events, 205 native rounds, 603 proposed and 304 accepted
tokens; DSpark7 logged 173 noise events, 177 rounds, 1,183 proposed and 332
accepted; DFlash3 logged 213 noise events, 215 rounds, 635 proposed and 295
accepted; DFlash7 logged 197 noise events, 201 rounds, 1,355 proposed and 309
accepted. The token-level comparison passed for prompt-01 and the raw EOS
fixture in all four arms; prompt-00 and its warmup diverged at token89 in all
four arms (target 72,499 vs verified speculative token 9,920). Speculative
traces show the correction token was chosen by target verification with zero
draft acceptance in the corresponding round, so the divergence is not an
accepted draft token.

The 84c1 validator stopped before creating `admission.json`, failing with
`ValueError: missing actual seven-row attention masks` after all 25 requests
had completed. Each draft state trace contains exactly two leading bootstrap
mask rows before its first noise event: anchor0, query positions0 and1, visible
positions `[0,1]`, `max_visible_clean_position=-1`, seq0. Apart from these two
rows, the logged mask blocks are complete: every noise event has its seven
query positions in order with the correct visible positions and clean-prefix
bound. Counts are DSpark3 203 noise/1,423 masks, DSpark7 173/1,213, DFlash3
213/1,493, and DFlash7 197/1,381 (each mask total is exactly two above
`7 * noise`). This is preserved as a narrow validator-accounting issue, not
claimed admission. The independent token comparison also found the prompt-00
exactness discrepancy, so no admission or timing is justified by this run.

Peak `gpu-loaded.json` readings were target 8,511 MiB, DSpark3/7 10,509 MiB,
and DFlash3/7 10,361 MiB. Final per-server CUDA breakdowns were: target
11,263 = 2,089 free + 7,965 self (7,672 model + 288 context + 4 compute) +
1,209 unaccounted MiB; DSpark3 11,263 = 0 free + 7,966 self (7,672+288+6) +
3,296 unaccounted; DSpark7 11,263 = 0 + 7,967 (7,672+288+6) + 3,296
unaccounted; DFlash3 11,263 = 97 free + 7,966 self (7,672+288+6) + 3,199
unaccounted; DFlash7 11,263 = 97 + 7,967 (7,672+288+6) + 3,199 unaccounted.
All reported Host model maps were 742 MiB (741 MiB process map).

At the final cleanup check, the tmux session/server, remote-job supervisor
PID/PGID21046, probe PID/PGID21048, and server groups 21262/21346/21394/21477
were absent; `nvidia-smi` returned only the Xwayland 364 MiB baseline and no
compute-app rows. No full screen or component profile ran. Cumulative
inference time remains 0 seconds. Any future GPU transaction requires a fresh
pause/process/context check; preserve every run and do not call the GPU free
until its supervised process group and CUDA contexts are absent.

## Bounded raw-logit diagnosis and Q4 output comparison (2026-10-04)

After root published checkout `5048bdea77d1cef1c1415e38d7cd4e30101cbec9`, I ran
one bounded baseline-only trace with the unchanged final fcdf binary, frozen
FP16 target, Q4_0 EAGLE primary, protocol, cache, prompts, seeds, greedy
sampling, full 128-token cap, and warmups. Only the diagnostic positions `[87,
88,89,90]` and port changed. The supervised run was
`near-tie-logit-baseline-20261004`, exit 0 from 01:44:52.217Z to 01:45:38.094Z.
Its config is
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-5048bde/runs/neartie-logit-diagnostic-fcdf-20261004/config.json`
with SHA256
`f7e0d053393534894e2b5c29f7f29315f18dc3576924c633500630341e10329c`. It binds
protocol SHA `367431663597d312bf4cc75544d3e8c1994260a0d3cd266558cdd388b48ceca7`
and gate file SHA
`c3d57320ce79bb3dc3d7ea4decda00307f26f3fdb244e927490b62dee951fc45`.

The raw target verifier traces are in `results/target_only/verify.jsonl`, SHA256
`3bcfc92f20e8fdbd51cd11b5a0a3bebb8fc9eeb3032829878d543f8e3fefd3eb`; Q4 traces
are in `results/eagle_q4_0/verify.jsonl`, SHA256
`787b54ca1c4c4cd1dc2ab5ad2d98a41051c82c5391bd55acd185e430705782d0`. All ten
target/Q4 request outputs exactly reproduce the earlier baseline readiness
run; the same token-ID mismatch remains at position89 in prompt-00/warmup-00,
with all other prompt/EOS pairs equal. Each arm's position87–90 top-five logits
and selected IDs were repeatable across the matching warmup and prompt.

At the actual matched prompt-00 prefix, target position89 selected/emitted
72,499 from top-five `[(72499,18.92755126953125),(9920,18.919330596923828),
(34855,18.052734375),(28727,17.546527862548828),(17486,17.408985137939453)]`.
The actual Q4 sampled/emitted correction row selected 9,920 from
`[(9920,18.92258644104004),(72499,18.919464111328125),
(34855,18.052085876464844),(28727,17.555221557617188),
(17486,17.406150817871094)]`. It is Q4 trace task142, verification round48,
generated position89, base position89, row0, empty speculative prefix, proposal
token34,855 rejected, no replay. The associated round-trace record (task142,
round47) shows five proposals `[34855,315,279,326,1246]`, zero accepted,
verified/emitted correction `[9920]`, and no replay. The actual prefix IDs through
position88 are identical across target and Q4. The target margin is
`0.008220672607421875`; Q4's opposing margin is `0.0031223297119140625`.
Both IDs appear in both finite top-five lists, both NaN counts are zero, and
the centered common-top-five maximum difference is `0.008617782592773437`.
All predeclared numeric-gate checks pass. This supports the bounded near-tie
sensitivity conclusion only; it does not establish bit parity or prove batch
arithmetic caused the difference.

A separate CPU comparison bound every candidate arm's five raw output-ID arrays
to the Q4_0 EAGLE primary readiness outputs and identical request JSON. Its
source-bound comparison report is
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-5048bde/runs/neartie-logit-diagnostic-fcdf-20261004/results/q4-primary-output-comparison.json`,
SHA256 `e4dd913e4f1eebf2bc32a8c23fce8ba81954a6d9288da873eff0a24ec878433b`.
DSpark7 and DFlash7 match Q4_0 EAGLE exactly for all five cases. DSpark3 and
DFlash3 match on prompt-01 and EOS, while prompt-00 and its warmup first differ
at index94; each has 33 suffix differences at positions94–111 and113–127.
No comparison row claims candidate admission. The report pins native probe
config, protocol, binary/target, source/checkpoint conversions, canonical and
export receipts, diagnostic verify logs, Q4 readiness files, all per-case
measurement and request paths/hashes, and the preserved probe failure record.

After the diagnosis, the operator rechecked teardown: remote-job supervisor
group22662, probe group22664, and model groups22717/22810 were absent; the
runtime tmux server was gone; RTX2080Ti was at 366/11,264MiB with no compute-app
rows and 0% utilization. No screen timing was launched; cumulative inference
time remains 0s. Root later published a CPU-only mask-accounting fix and
continues with CPU revalidation of the old probe; no second model probe has
been run.

## Follow-up at short-N first divergence (2026-10-04)

After root accepted the bounded position89 sensitivity gate, I verified
DSpark3 and DFlash3 produce identical full 128-token IDs on all five existing
native-probe cases. CPU report:
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-84c1f24/runs/native-admission-fcdf-20261004/results/dspark3-dflash3-exact-id-comparison.json`,
SHA256 `3f71a59693b06e0b3dffb7a0ede843fe26c75ef06ae28198d5ea79118fdf8cc4`. Both
short arms diverge from primary Q4 on warmup-00/prompt-00 at token94; all other
cases and tokens matched Q4. That makes token94 the next concrete near-tie
check.

The one additional 114cc2f diagnostic ran only Q4 EAGLE and DSpark3 with the
unchanged 128-token cap, context2048, B32/µ32, seed42, greedy sampling, prompts,
cache behavior, and two warmups. It traced generated positions92–95. Config:
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-114cc2f/runs/neartie-shortnmax-fcdf-20261004/config.json`,
SHA256 `63c0729299f2f51d4d6a5bc1d8221c9ee5b1f21b8c59a22c43ad9773f5febc25`;
arms `[eagle_q4_0,dspark_3]`, port18386. The supervised run
`neartie-shortnmax-diagnostic-20261004` ran from 02:13:55.265Z to 02:14:42.708Z,
exit0. Q4 server PID/PGID23855 and DSpark3 PID/PGID23931 each returned0. GPU
load snapshots were Q4 8,683MiB and DSpark3 10,511MiB. At cleanup the supervisor,
probe, both server groups, and runtime tmux were absent; RTX2080Ti returned to
366/11,264MiB with no compute-app rows.

The exact supervised launch was:

```sh
tmux -L binary-eagle-runtime new-session -d -s neartie-shortnmax -c "$PWD" "exec python3 scripts/remote_job.py neartie-shortnmax-diagnostic-20261004 --stop-grace-seconds 15 -- python3 /home/philip/binary-eagle-decoding/runs/checkouts/dspark-screen-114cc2f/scripts/dspark_screen/baseline_probe.py /home/philip/binary-eagle-decoding/runs/checkouts/dspark-screen-114cc2f/runs/neartie-shortnmax-fcdf-20261004/config.json /home/philip/binary-eagle-decoding/runs/checkouts/dspark-screen-114cc2f/runs/neartie-shortnmax-fcdf-20261004/results"
```

The DSpark cell set `DSPARK_REQUIRE_AUTHOR_LAYOUT=1`; both cells captured
`W1AX_VERIFY_TRACE_JSONL` at positions `92,93,94,95` and the existing round
trace. No profiler or throughput timing flags were set.

Q4 and DSpark3 outputs share the complete prefix through token93 and first
diverge at94: Q4 emits2331; DSpark3 emits23035. Their actual raw verifier rows
at94 use the same reached target prefix, with base position92 and draft-prefix
IDs `[2348,279]`; both are sampled/emitted, nonreplay rows. Q4 top five are
`[(2331,21.159774780273438),(23035,21.15850067138672),
(85849,20.836078643798828),(43006,20.798328399658203),
(6946,20.02191162109375)]`. DSpark3 top five are
`[(23035,21.15709114074707),(2331,21.15159034729004),
(85849,20.83428192138672),(43006,20.794246673583984),
(6946,20.019554138183594)]`. Both actual IDs appear in both finite lists,
with no NaNs. Q4 margin is `0.00127410888671875`; DSpark3 margin is
`0.00550079345703125`; centered common-top-five maximum difference is
`0.004618453979492188`. The Q4 native round verifies/emits `[2348,279,2331]`
after two accepted drafts; DSpark3 verifies/emits `[2348,279,23035,3685]` after
three accepted drafts. The bounded position94 gate passes.

CPU evidence summary:
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-114cc2f/runs/neartie-shortnmax-fcdf-20261004/results/neartie-shortnmax-summary.json`,
SHA256 `cd8dc701673452565bb11adaa4de30437ea12a298ca236b58ab465e1662009aa`. Its
request/measurement, verifier-trace, and round-trace artifacts remain in the
same ignored run directory. The source-bound evidence manifest is
`.../neartie-shortnmax-fcdf-20261004/results/neartie-shortnmax-evidence-manifest.json`,
SHA256 `2a1165e28d1bd7aa10cc101a1ef3b60432fc7702676670f208e2792ff4f22741`, with
all bounded near-tie and cache-prefix checks passing. No full five-repeat timing
or extra GPU run occurred; cumulative inference time remains 0 seconds.

## Resumed six-repeat reference screen — 2026-10-04

After the human renewed the RTX2080Ti study authorization, I refreshed the
registry through the local tmux MCP pane: `.29` (`rtx2080ti`) was unpaused;
`.24` remained paused. SSH to `.29` entered WSL2 kernel
`6.18.33.2-microsoft-standard-WSL2`; the WSL GPU utility is
`/usr/lib/wsl/lib/nvidia-smi`. At 02:42:58Z, before launch, the RTX2080Ti UUID
was `GPU-35b7b96c-d577-95f0-0050-40699c9faef7`, driver 610.74, 11,264 MiB,
366 MiB allocated, 0% utilization, P8/31C. No compute-app rows, matching
supervisor/probe/benchmark/server/validator processes, runtime tmux server, or
listener on 18384 existed. WSL boot was 2026-10-03 16:08:09; the `.wslconfig`
idle-timeout setting still awaits WSL restart, with the prior 109-second
disconnect/reconnect proof as current-boot durability evidence.

CPU-only numeric receipt generation used frozen original evidence, without
loading a model. It covers 20/20 original native-probe measurement outputs and
has 0 uncovered outputs. Position 94 has opposing raw margins
`[+0.00127410888671875,-0.00550079345703125]` and centered common-top-five
maximum `0.004618453979492898`. Receipt:
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-a9dded1/runs/numeric-admission-fcdf-20261004/results/numeric-receipt.json`,
SHA256 `d9d1b1a62ae2038470579db9fdc3df2884eb24dcf84ac1e495caf074658f3c17`.
Copy of the immutable probe manifest with the receipt binding was validated
CPU-only; `admission.json` passed with memory, anchor-first, target-immutable,
cache-contract and greedy-semantics fields all true. Its SHA256 is
`305a56fdeeb112af076edf25da04d6cace7daf57a511c9d0a2000faaa7fe481b`; the
admitted final fcdf582 binary hash is
`1bd67cdf74d6ced49454ca2546d9a462e71d4e695b8fb5e52d7359fa68473822`.

The full BF16 reference timing uses immutable checkout
`dspark-screen-a9dded1` at `a9dded1bb2e554d7394472f93d6160db24f333a9`, frozen
protocol SHA256 `367431663597d312bf4cc75544d3e8c1994260a0d3cd266558cdd388b48ceca7`,
final target/hash `Qwen3-4B-f16.gguf` /
`05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6`, Q4_0
EAGLE/hash `Qwen3-4B-eagle3-q4_0.gguf` /
`2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280`, DSpark
BF16/hash `dc5299bdb1e7e906b003334a9b806ba502e72111341432ac5a9fd11de22f520c`,
and DFlash BF16/hash
`92925d9e4be49a82d1e4f9d9671aae0e3647c1146fccbd4004bfc8afe0c9896f`. The
run configuration is
`runs/reference-sixrep-fcdf-20261004/config.json`, SHA256
`ff3412d462960484339196a3d219419d0eb7598194280230f92b4e143131fd49`; it binds
the passed admission by path/SHA and begins at0 prior inference seconds.

Exact supervised launch from the clean immutable checkout:

```sh
PATH="/usr/lib/wsl/lib:$PATH" tmux -L binary-eagle-runtime new-session -d \
  -s dspark-ref-6x6-20261004 -c "$PWD" \
  "exec python3 scripts/remote_job.py dspark-reference-sixrep-fcdf-20261004 \
  --stop-grace-seconds 10 -- env PATH=/usr/lib/wsl/lib:$PATH PYTHONUNBUFFERED=1 \
  python3 scripts/benchmark_dspark_screen.py \
  $PWD/runs/reference-sixrep-fcdf-20261004/config.json \
  $PWD/runs/reference-sixrep-fcdf-20261004/results"
```

The six-arm, six-repeat run completed all 936 records: 864 measured requests
and72 warmups, with1246.091775789 seconds of phase inference. The finished raw
receipt is `runs/reference-sixrep-fcdf-20261004/results/measurements.json`,
SHA256 `74b9c9dfc4f16bf549502efe0fe779f22ac7e1a8db5839e008a9ee655aa31381`.
Its source-bound CPU summary is `summary.json`, SHA256
`07c85d24f846c06fd9eea4a1b9188d28dd9d5548efb008fa8620d87472b78f99`. The
supervisor ended03:10:43.771869Z with exit0. At03:11:33Z, exact-owned process
and compute-app queries were empty, the runtime tmux server was gone, port18384
was free, and the RTX2080Ti returned to366MiB/0%/P8.

Candidate FFN-Q4 source config is staged for later
use at `runs/q4-native-admission-fcdf-20261004/config.json` in the a9 checkout;
its config SHA is `ae224dde8f3ecac093266218b6416457edc52b0bec91a5e7ce8f884b73f1fc32`.
The same file is staged in checkout `0c2f2923ebe34eac1fba3398f26678593c05c360`;
it references the
persisted plan-selected candidate files and passed exact 15-matrix receipts
without hashing either large candidate during the timed reference run.

## Q4 native admission and timing — 2026-10-04

The sole Q4 native probe used immutable source checkout
`0c2f2923ebe34eac1fba3398f26678593c05c360`, config
`runs/q4-native-admission-fcdf-20261004/config.json`, and output directory
`runs/q4-native-admission-fcdf-20261004/results`. Its one supervised Linux tmux
session was `dspark-q4-probe-20261004`, remote job ID
`dspark-q4-probe-supervisor-20261004`, with30-second supervisor grace. All25
requests completed across target-only plus DSpark/DFlash short and maximum
cells, including warmups and EOS; the target and all four Q4 candidates loaded
on the RTX2080Ti, with no OOM. The probe's first validator run exited1 after
preserving `manifest.json` and all raw traces: `failure.json` says
`uncovered target-only token mismatch`, records_completed25. This was the
expected unscoped position89 target-versus-Q4 sensitivity that its first
exact-ID validator could not yet consume; there was no GPU rerun.

The scoped Q4 numeric receipt was composed CPU-only from the original baseline,
position89 and position94 diagnostics, and the retained Q4 BF16 references,
precision receipts and complete outputs. It passes20/20 output bindings, with0
uncovered; Q4 ancestry covers only exact full-ID and termination paths, not raw
candidate-Q4 logits. Receipt
`runs/q4-native-admission-fcdf-20261004/results/numeric-receipt.json` has SHA256
`d9e75322b43b79ea147fe5fb2a1eb72847f169671eaeb3172c0b4a422112cb79`. CPU
revalidation on immutable helper source
`a7c3f502a0bab7d4874c31b1bb45cea3ddc6a94d` produced a passed actual native
admission at
`runs/q4-native-admission-fcdf-20261004/results/admission-history-aware.json`,
SHA256 `b3055c5ee2c76d20aff23b425027530ba81ff36ad91bc9a70b79372d0f29704e`.
The fixed fcdf582 binary, target, protocol and Q4 EAGLE hashes match the
reference receipt. Admission reports memory, anchor-first, target-immutable,
cache-contract and greedy-semantics checks true.

The history-aware comparison qualifies all five first-noise blocks for both
architectures and finds no disagreement at identical input history. DSpark
has5 matched-history joins and122 token-matched/different-history joins, with2
later decision changes; DFlash has5 matched-history joins and146
token-matched/different-history joins, with6 later decision changes. The initial
ordered prefill/injection signatures and first three proposals match at all five
requests, including EOS. Subsequent target injections differ in span and
aggregate feature hash as n_max changes, before the later proposal-decision
differences; state traces contain no rollback events and the corresponding
rounds are nonreplay. `state.jsonl` records per-injection FNV64 hashes, feature
spans, cache maxima, taps and rc; raw target-feature tensor values are not
captured. The original failed validator manifest and error remain preserved
alongside the history-aware admission.

The six-arm Q4 timing used immutable checkout
`a7c3f502a0bab7d4874c31b1bb45cea3ddc6a94d`, the frozen protocol, and
`prior_timing_receipts` bound to the completed BF16 reference measurements SHA256
`74b9c9dfc4f16bf549502efe0fe779f22ac7e1a8db5839e008a9ee655aa31381`. Its config
`runs/q4-timing-sixrep-fcdf-20261004/config.json` has SHA256
`59218dc3306a6e8015e620359f05ce804b505aab20ad25714fd01bd056502bb3`. The two
phases charged1246.0917757890202 and1306.1840105780866 inference seconds,
2552.275786367107 combined from the shared7200-second limit. The Q4 phase
completed936 records (864 measured,72 warmups) and exited0. Its raw
`measurements.json` SHA256 is
`92c4d132df8ce330bfbc0a1dfab13e849fd5b9ce5a8752c3910ada75e457433b`; CPU summary
`summary.json` SHA256 is
`138be59ba135f5e421afc49ed8afc1dedc219cec7e17d263fbb513db7380c91e`.

| Arm | Decode / request TPS | Mean / p95 latency (ms) | Proposed / usable accepted | Q4-primary mismatches |
| --- | ---: | ---: | ---: | ---: |
| Target only | 58.340 / 60.570 | 2179.76 / 2218.31 | — | — |
| Q4 EAGLE | 82.752 / 87.587 | 1536.71 / 1863.88 | 41,850 / 9,630 | primary |
| DSpark3 | 99.141 / 106.261 | 1282.69 / 1627.67 | 17,454 / 12,246 | 12 |
| DSpark7 | 128.774 / 141.125 | 987.52 / 1541.73 | 29,292 / 13,872 | 0 |
| DFlash3 | 105.227 / 113.447 | 1208.50 / 1569.37 | 18,048 / 12,066 | 12 |
| DFlash7 | 128.989 / 141.276 | 985.87 / 1612.67 | 32,094 / 13,464 | 0 |

The Q4 supervisor (`dspark-q4-time-6x6-20261004`, run ID
`dspark-q4-timing-supervisor-20261004`, supervisor PID/PGID30875, child
PID/PGID30877) started03:59:17.479593Z and exited0 at04:25:43.704522Z. At
04:26:56Z the owned process groups and compute-app list were empty, the runtime
tmux server was gone, port18384 was free, and the card was at366MiB/0%/P8/44C.
The measured phase2 target and Q4 EAGLE rates were about8–9% slower than phase1,
while candidate absolute rates were roughly flat. The already-recorded startup
snapshots show the same device, driver and idle P8 clocks in both phases; no
per-request clock telemetry was recorded, so the relative-ratio increase is not
attributed to FFN quantization.

## CUDA event / sparse MATMUL profiles and p90 check — 2026-10-04

The four short event runs used immutable source checkout
`8165b0f43add6f0312cf703bb42f9ba7422595bd` and the existing admitted BF16 and Q4
assets. Each ran two arms with one 16-token warmup and one 16-token measured
request per arm. The exact command form was:

```sh
tmux -L binary-eagle-runtime new-session -d -s SESSION -c "$PWD" \
  "exec python3 scripts/remote_job.py RUN_ID --stop-grace-seconds 10 -- env \
  PATH=/usr/lib/wsl/lib:\$PATH PYTHONUNBUFFERED=1 \
  python3 scripts/dspark_screen/baseline_probe.py CONFIG RESULTS"
```

The four sessions, run IDs and config hashes were:

| Session / run ID | Config | Config SHA256 | Start–end UTC | Result |
| --- | --- | --- | --- | --- |
| `dspark-profile-bf16-dspark-20261004` | `runs/profile-diagnostics-20261004/configs/bf16-dspark.json` | `a2887d2ea6ad376347b4692cd38ec7a97f6da08878c893b2ba5ab15d1c6757aa` | 04:39:02.139734–04:39:39.661551 | exit0 |
| `dspark-profile-bf16-dflash-20261004` | `runs/profile-diagnostics-20261004/configs/bf16-dflash.json` | `cca9a75088983e3dcf1418e4a8325591154d85e2e042cd5ab44c013645ecd56f` | 04:40:28.683654–04:41:08.705907 | exit0 |
| `dspark-profile-q4-dspark-20261004` | `runs/profile-diagnostics-20261004/configs/q4-dspark.json` | `d7db91c2af1a7378cd015de473478196762ac782cc640d0d113ab606b26b1063` | 04:42:13.542910–04:42:50.066634 | exit0 |
| `dspark-profile-q4-dflash-20261004` | `runs/profile-diagnostics-20261004/configs/q4-dflash.json` | `f913b30d0f45122d658d3c093c61d93f50fd21e73a6f9a20380b313efec6536f` | 04:43:31.859477–04:44:10.881391 | exit0 |

All four runs returned the card to366MiB/0%/P8 before the next launch. The
DSpark event analyzer accepted25 graph frames for each arm without truncation;
each pair reproduced its output IDs. Aggregated component CUDA-event union time
in milliseconds (confidence, context injection, draft body, feature fusion,
full head, Markov/output assembly) was:

| Profile | DSpark3 | DSpark7 |
| --- | --- | --- |
| BF16 | 1.010 / 7.209 / 59.231 / 23.380 / 31.071 / 54.455 | 1.088 / 8.926 / 57.090 / 21.264 / 31.096 / 55.460 |
| FFN Q4 | 0.927 / 7.540 / 49.464 / 25.307 / 30.773 / 56.903 | 0.832 / 10.176 / 44.586 / 20.955 / 30.606 / 55.573 |

The DFlash general event analyzer read27,502 CUDA event records across2,086
frames for each precision phase; `trace_truncated=false`, with zero graph
inventory-only or orphan inventory records. The DFlash-specific DSpark event
analyzer then attributed candidate nodes by model context. Component CUDA-event
union milliseconds (context injection, draft body, feature fusion, full head,
Markov/output assembly) were:

| Profile | DFlash3 | DFlash7 |
| --- | --- | --- |
| BF16 | 7.617 / 74.459 / 26.807 / 41.413 / 24.872 | 9.115 / 63.347 / 15.910 / 36.140 / 20.720 |
| FFN Q4 | 7.484 / 58.981 / 27.405 / 43.771 / 29.953 | 10.371 / 53.282 / 19.478 / 37.714 / 22.581 |

These synchronized event spans include dispatch and stream idle; they are
perturbed diagnostics, not throughput estimates. The sparse audit found34
MATMUL inventory rows per Q4 candidate, including six Q4_0 records for
`blk.0.ffn_gate.weight`, `ffn_up.weight` and `ffn_down.weight`, at n=2 and7 via
MMVQ with Q8_1 activation. Gate/up shapes were [9728,2560] and down
[2560,9728], with I32-dot/F32-scaled accumulation. Candidate-context event joins
were DSpark3 `0x5b02d7b2d320` (2,022 event nodes), DSpark7
`0x58fd83533ce0` (2,022), DFlash3 `0x6006087b8440` (1,839), and DFlash7
`0x5efca6dd9d20` (1,601); the DFlash candidate `dspark_feature_fusion` and
`result_output` rows use the same candidate contexts as the Q4 MATMUL records.
This is sparse dispatch inventory, not kernel execution counts or full
15-matrix coverage. Raw Q4 log SHAs and canonical event/context join hashes:

| Arm | Raw server log SHA256 | Canonical join SHA256 |
| --- | --- | --- |
| DSpark3 | `82310c1ac740c3d5665a24969dc1260655cc5606f344d3dd95d4d5757a5bf66f` | `91d0d9478547bd1eb009020266ff951d043e7eb76d986e9f55aff7f75f5efb4c` |
| DSpark7 | `51db96ae929ba6d6a497b5e8715d2e3bf73acbddc5e8e8f78a874b64ae66f54a` | `302265d69e2325ac80d88d349216dfb2e8474c62cf8217fa90ad72c21cfbec64` |
| DFlash3 | `71735b4a26dbbeea6e143046b2266bd09fee62f675c3163dae8d80b78d86e9c8` | `9e8e30c645547b081797e016db191bda0d3ce17ac2e609d1952842f6868a2857` |
| DFlash7 | `b25f73ccef20e6e15761d1876692b465afd7fdec9aa1c19e21d94c20d65bbd74` | `4b9ee96faf30df9748b8e30e8373b87531e8317abfa07e2c6d76486303a7b1e1` |

The four source-bound event component summaries are preserved as
`dspark_3/events.json`, `dspark_7/events.json`, or
`dflash_3/dspark-event-analysis.json` and `dflash_7/dspark-event-analysis.json`
in each profile result directory. Their hashes in the same order as the tables
above are BF16 DSpark `a2aa2aef227c432dd021b02dbb25132175c406fb17a6b4980a13f3bfc399226d`,
`cf8fce14593d813aa96e2855ead846f826adb018bbb6c9f67881a331a26d94c2`; Q4 DSpark
`8caffa78ab995fe01ae992e06d8710e5237e96968334e5fb1be773b89023e9d5`,
`ea8a09b753b6ea9c2a7d8bc7c8fa892b641cf3abe6eef67823d3310c352b3172`; BF16
DFlash `99ab7d3176f313fd7fc87abeb8a1cf06ca037d78bc38277bf0c6305b3aeaf451`,
`0181a374160a45db52e7c671e49da821ef67703961290212968cb326a2ee5758`; and Q4
DFlash `e5c343f52d3fbd8e9d45f00ca9b0207d16523e7605d4b5816bb00636c31d56f9`,
`2a2e19efd45f8004563937c8bdc13f3fddf490f082567a14ba46fa6ce857a49b`.

The single raw-logit diagnostic used immutable runner checkout
`7b6ae9c` and run directory
`runs/p90-shortarm-fcdf-20261004/results`. Input config SHA256 was
`f6351ebd3b7e60d7e1fcd5b1853ed77ff42c9832b826271f118bdc0c56f51c9d`; the
runner-preserved config SHA256 was
`94fa23aff6668434dcd7467a7e1e9d1a46ebb329f51490c6ce9d095cc9154f14`. It compared
`eagle_q4_0` with `dspark_3` using the frozen F16 target and BF16 DSpark draft, six frozen prompts, two
warmups per arm, output cap128 and verify positions88–95. Its detached Linux
tmux session was `dspark-p90-shortarm-20261004`, run ID the same, supervisor
PID/PGID34989 and child PID/PGID34991, 04:46:30.461023–04:47:27.999559Z, exit0;
each arm completed all8 requests. The paired full outputs differ on warmup00,
prompt00 and prompt05; prompt05's first differing ID is at position90.

The CPU audit was run from final source commit
`4d15874ab49b7f5850c46dbce2ddfa0296431e5a` against the original BF16 reference
results and predeclared `configs/dspark-screen/numeric-gate.json`, with explicit
`--protocol` pointing to the immutable frozen source protocol
`367431663597d312bf4cc75544d3e8c1994260a0d3cd266558cdd388b48ceca7`. The command
was:

```sh
python3 scripts/dspark_screen/audit_completed_pair.py \
  /home/philip/binary-eagle-decoding/runs/checkouts/dspark-screen-7b6ae9c/runs/p90-shortarm-fcdf-20261004/results \
  /home/philip/binary-eagle-decoding/runs/checkouts/dspark-screen-a9dded1/runs/reference-sixrep-fcdf-20261004/results \
  configs/dspark-screen/numeric-gate.json \
  /home/philip/binary-eagle-decoding/runs/checkouts/dspark-screen-7b6ae9c/runs/p90-shortarm-fcdf-20261004/results/audit-prompt05-position90-final.json \
  --protocol /home/philip/binary-eagle-decoding/runs/checkouts/dspark-screen-7b6ae9c/configs/dspark-screen/protocol.json \
  --case prompt-05 --position 90
```

The final audit passed; SHA256
`95685d8d6034204161d2c6e7a49e78b4e684b66e86f1385e36f4525b2a4167cb`. Competing
IDs were16062 and28071; signed raw-logit gaps were+0.002527237 and−0.002305985,
with centered common-top5 maximum0.005583191. Both raw top-5 lists were finite,
contained both IDs, and selected the actual emitted token. This accepts only the
scoped near-tie sensitivity. The first audit failure and a staged-protocol
intermediate pass remain preserved; the benchmark's sorted protocol JSON has a
different byte hash from the source, so the final audit checks both source bytes
and parsed saved semantics.

The already-recorded `environment.json` files have SHA256
`e62bcc318593b9535f591445573e8b3dad5e659119bb3f92d5411b887d0a4656` (BF16) and
`096ad521eb0f553c6cb197d785e7336780c84f28fbee90cae44fa48fec7f5cfc` (Q4).
Both report the same RTX 2080 Ti UUID, driver610.74,11264MiB, compute capability
7.5, CUDA driver-reported version13.3 and linked `libcudart.so.12`. Their
saved `nvidia-smi -q` snapshots were idle P8 at31C, graphics/SM300MHz and memory
405MHz, with instantaneous draw24.28W and24.63W respectively (max graphics/SM
2190MHz, memory7000MHz). No per-request clocks were captured during either
timing run. This supports recording the phase difference, but does not explain
it or justify assigning the candidate ratio change to Q4 quantization.

At the final post-run check, all profile and p90 supervisor/server groups were
stopped, their ports18386–18390 were free, and GPU memory returned to366MiB. The
card was still the registry-selected RTX 2080 Ti UUID
`GPU-35b7b96c-d577-95f0-0050-40699c9faef7`, driver610.74, 11264MiB, SM75.

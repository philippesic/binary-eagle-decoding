# RTX2080Ti operator record

Operator: Luna. All RTX2080Ti access was through the tmux MCP transport to the
registry host alias `rtx2080ti` in
`~/.config/binary-eagle-decoding/hosts.toml`; no ordinary-shell SSH, SCP, or
rsync was used. Registry user `philip`, port `22`, workdir
`~/binary-eagle-decoding`, initial pause flag false. The current pause is recorded below. The registry address remains
machine-local; only the host alias is recorded here.

## Host and initial admission evidence

Read-only commands, issued over the tmux MCP pane (host address redacted here; exact commands remain in ignored operator evidence):

```sh
ssh -p 22 philip@<registry-host> 'hostname; date -Is; uname -a; nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw --format=csv; nvidia-smi; ps -eo pid,ppid,pgid,sid,stat,etime,cmd --sort=pid | head -100; tmux -L binary-eagle-runtime ls 2>&1'
ssh -p 22 philip@<registry-host> '/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version,pstate,temperature.gpu,clocks.current.sm,clocks.current.memory,power.draw --format=csv'
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

The complete throughput/probe checkout is clean at
`~/binary-eagle-decoding/runs/checkouts/dspark-screen-03f6c81`, parent commit
`03f6c81a247407cd3c308a8fcd65d9e63d8ee5d8`; frozen protocol SHA256 is
`6244571641791b4f47e2109295f9e28e2827d2c63c290670548f60893c16cf3c` (six
balanced repeats). Both native draft exports are passed and no model has been
loaded; probe config/staging and fresh pre-load hardware checks remain pending.

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
`nvidia-smi --query-compute-apps` returned only its header. The RTX 2080 Ti
remained at the Xwayland baseline, 549 MiB / 11,264 MiB and 0% utilization. All
owned remote run process groups checked after completion are absent. Probe and
throughput screen remain paused pending a fresh explicit resume; full six-repeat
screen protocol checkout is 03f6c81 with SHA256 recorded above. The cumulative
inference budget remains 0 seconds.

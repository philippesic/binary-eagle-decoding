# RTX5080 overnight QAT operator record

## Authorization and ownership

The user authorized RTX5080 use for preflight and relevant QAT, with approximately
30-minute health checks and continued operation while healthy. RTX2080Ti remains
outside this assignment. Root granted source-bound staging for parent
`d7bbdeef970219b4eecae2a598276d0065879dee` and native submodule
`624f50e74f51b6af93bf6b879f84703e726df172`; no training, model load, data
capture, or evaluation has been authorized by that narrow GO.

This operator owns one managed Git worktree at
`/private/tmp/nine-model-qat-20261004/overnight-operator`, branch
`prep/nine-model-overnight-operator`, and the sole RTX5080 transport/job
partition. The separate human-facing project chat is the coordinator. Existing
dirty checkouts, other operators' worktrees, captured data, and run files remain
untouched.

## Fresh remote inventory

Read-only inventory used the shared `rtx5080` registry entry through tmux MCP.
The remote login opened Windows SSH and then Ubuntu WSL. The actual registry
address is intentionally not copied into this report. MCP transport initially
used local tmux session `$257`, window `@286`, pane `%288`; all SSH/WSL commands
ran in that pane. No shell SSH/scp/rsync command was used outside tmux MCP.

On 2026-10-05 06:45 UTC the WSL device reported NVIDIA GeForce RTX 5080, driver
616.92, 16,303 MiB VRAM, 2,714 MiB used, 13,264 MiB free and 0% utilization; no
GPU processes appeared in `nvidia-smi`. `free -h` showed 19 GiB total and 18 GiB
available, 5 GiB swap; `/home` had 1,007 GiB total and 199 GiB free. WSL uses a
20 GB memory cap. `/mnt/c/Users/philip/.wslconfig` contains
`[general] instanceIdleTimeout=-1`.

`nvcc` is CUDA 13.1.115; GCC is 15.2.0. The old root `.venv` reports Python
3.11.15, CMake 3.31.10, Torch 2.14.0+cu130, Torch CUDA runtime 13.0, and
`torch.cuda.is_available() == True`. System `cmake` is absent from `PATH`; the
existing locked environment provides `.venv/bin/cmake`. This import/device query
did not load a model or run a CUDA kernel.

No `binary-eagle-runtime` or default Linux tmux server existed at initial
inventory. `pgrep` found no `remote_job.py`, trainer, continuous-QAT, or teacher
capture command. The pre-existing root checkout is `main` at
`7547d253b6bf7d8a04ddb3c868e997afad39f31a`, 555 commits behind its configured
origin, with `M scripts/train_continuous_w1ax.py`, untracked `checkouts/`, and
untracked `rescue-head-20260927/`. Nine historical detached worktrees were
listed. All were preserved. The previous `nineprep-5080-uvsync-20261004` job
ended naturally with exit 0 at 2026-10-04 23:45:20 UTC; it only installed locked
dependencies into the old nine-model checkout. No build or CUDA model check was
performed there.

The remote model directory includes target Qwen3-4B HF F16 shards at
`models/hf/Qwen3-4B/` and Eagle3 at `models/hf/Qwen3-4B_eagle3/model.safetensors`.
SHA256 values read from the files:

| File | SHA256 |
|---|---|
| `models/hf/Qwen3-4B/model-00001-of-00003.safetensors` | `328a91d3122359d5547f9d79521205bc0a46e1f79a792dfe650e99fc2d651223` |
| `models/hf/Qwen3-4B/model-00002-of-00003.safetensors` | `6cd087b316306a68c562436b5492edbcf6e16c6dba3a1308279caa5a58e21ca5` |
| `models/hf/Qwen3-4B/model-00003-of-00003.safetensors` | `e4bf436957184f4eeb86a80e9db394503f1f56446b2e6b7edeac5b81470f4ca1` |
| `models/hf/Qwen3-4B_eagle3/model.safetensors` | `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e` |

The exact old EAGLE preparation receipt and its retained stage tree are present:
`runs/qat-optimization-readiness/prepared-full-source-20261003-01/provider-binding.json`
binds to `runs/qat-optimization-readiness/retained-capture-adoption-20261002-01/checkout/runs/retained-a8-a1-preparation-20261002-01/preparation-ready.json`;
the receipt hash matches
`bdfa56f8b10e44e82a6d807a71f32d68c39143af7094e6f8f0da63504d41a498`. Its
`resolved_config.json`, 320 `stages/provider-*.json` files, and retained stages
manifest exist. The stage manifest is
`runs/qat-optimization-readiness/retained-capture-adoption-20261002-01/inputs/retained-stages.json`
with SHA256
`ccc43a104a3a09e1a910746c319f55df33761bec5616378254a90e9339dbdafe`.
The frozen configs and stages implementation are under
`runs/qat-optimization-readiness/retained-capture-adoption-20261002-01/checkout/`
(`configs/continuous_w1ax.json`, `configs/continuous_w1ax_sources.json`,
`scripts/w1ax_continuous_stages.py`). A first TRAIN index record confirms
metadata keys `category`, `content_sha256`, `domain`, `group`, `id`,
`input_tokens`, `source_id`, `source_row_id`, and `topic`; only the index record
was read, not prompt text.

The current remote root model files include only Qwen3 target F16, Eagle3 F16,
and an Eagle3 head W1A1 GGUF. The inventory has not yet established presence and
identity of the nine original frozen Q4 controls or any DSpark/DFlash Q4 block
control. No candidate is admitted from filenames alone.

## Source staging

Created one clean source clone at
`/home/philip/binary-eagle-decoding/runs/checkouts/nine-model-qat-overnight-d7bbdee`
and checked out exact detached parent
`d7bbdeef970219b4eecae2a598276d0065879dee`, with clean
`third_party/llama.cpp` at
`624f50e74f51b6af93bf6b879f84703e726df172`. The old dirty checkout and the
previous `nine-model-qat-5080-b998f4f` checkout were not modified. New clone
staging command, executed inside MCP pane `%288`:

```sh
cd ~/binary-eagle-decoding && test ! -e runs/checkouts/nine-model-qat-overnight-d7bbdee && git clone --depth=1 --no-checkout https://github.com/philippesic/binary-eagle-decoding.git runs/checkouts/nine-model-qat-overnight-d7bbdee && git -C runs/checkouts/nine-model-qat-overnight-d7bbdee fetch --depth=1 origin d7bbdeef970219b4eecae2a598276d0065879dee && git -C runs/checkouts/nine-model-qat-overnight-d7bbdee checkout --detach d7bbdeef970219b4eecae2a598276d0065879dee && git -C runs/checkouts/nine-model-qat-overnight-d7bbdee submodule update --init --depth=1 third_party/llama.cpp
```

The command completed successfully. `git status --short --branch` on the clone
reported clean detached `HEAD`; `git submodule status` reported the requested
native revision. No GPU build has run.

## Bounded CPU disconnect proof

The older `prep09-disconnect-supervisor-20261002-01` attempt was interrupted
with SIGINT at the 240-second mark and is not accepted as a completed proof. A
new stdlib-only heartbeat probe was staged at
`runs/operator-tools/cpu_disconnect_probe.py` inside the clean source clone. The
supervised run id is `nine-model-overnight-cpu-disconnect-20261005-01`, with
detached Linux tmux socket/session `binary-eagle-runtime` /
`nine-model-overnight-disconnect-20261005`. It writes raw output only under the
ignored new clone run directory.

Exact launch command:

```sh
tmux -L binary-eagle-runtime new-session -d -s nine-model-overnight-disconnect-20261005 -c /home/philip/binary-eagle-decoding/runs/checkouts/nine-model-qat-overnight-d7bbdee 'exec /home/philip/binary-eagle-decoding/.venv/bin/python /home/philip/binary-eagle-decoding/runs/checkouts/nine-model-qat-overnight-d7bbdee/scripts/remote_job.py nine-model-overnight-cpu-disconnect-20261005-01 -- /usr/bin/python3 /home/philip/binary-eagle-decoding/runs/checkouts/nine-model-qat-overnight-d7bbdee/runs/operator-tools/cpu_disconnect_probe.py --heartbeat /home/philip/binary-eagle-decoding/runs/checkouts/nine-model-qat-overnight-d7bbdee/runs/nine-model-overnight-cpu-disconnect-20261005-01/heartbeat.jsonl --max-seconds 240 --interval-seconds 1
```

At 2026-10-05 06:55:59.249835 UTC, `remote_job.py` recorded supervisor PID/PGID
44534 and child PID/PGID 44540, both in their expected process groups. The
initial probe snapshot had eight fsynced heartbeat rows and no received signal.
GPU remained at its 2,714 MiB idle baseline and 0% utilization. I then closed
both WSL and Windows SSH shells through MCP; the pane printed
`Connection to <registered rtx5080 host> closed.` with exit status 0. A fresh MCP
transport reconnected after the full bound. `remote_job.py` ended naturally at
2026-10-05 06:59:59.796772 UTC with exit0/no signal; the probe summary records
240.011 seconds of runtime and the same child PID/PGID/SID. All240 fsynced
heartbeat/event lines were present. Supervisor44534 and child44540 were absent,
the detached Linux tmux server had exited, and the GPU remained at 2,714MiB/0%.

## EAGLE production initializer

The original retained full TRAIN receipt and all requested domain/group/topic/
content quotas passed with the reviewed source fix. Three earlier attempts are
preserved: `eagle-fixed-a8-initializer-20261005-01` lacked original commit6f in
the shallow clone; attempt02 lacked original producer commit7547; attempt03
reached the original data and correctly refused one selected longest round
without a CE label. None of those attempts wrote an initializer or used the GPU.
The dedicated source clone received only exact commit6f and commit7547 history
objects; HEAD and tracked source remained unchanged.

Attempt04 used the clean detached helper checkout at parent
`0b2ca0ab2687d91e72e5a26cbd4ef1d89026a8d8`, native submodule624, and the
existing locked Python environment. The supervised child used the original ready
receipt SHA `bdfa56f8b10e44e82a6d807a71f32d68c39143af7094e6f8f0da63504d41a498`,
resolved config SHA
`09e8afa76bb82ca8260cd373fbcb3f22fdd184187d4e7f67bb873c038404943b`, source
weights SHA `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`,
32 fit +16 validation prompts per domain,16 rows per prompt, fixed A8,
300-second fit cap, 900-second phase cap, two CPU threads, CUDA hidden, and an
enforced8GiB scope. It exited naturally with code0 at07:28:03.802240Z;
supervisor45952 and child group45958 are absent.

The output initializer is
`/home/philip/binary-eagle-decoding/data/nine-model-overnight/eagle-fixed-a8-initializer-04/initializer.npz`,
SHA256 `ae910cd8524ea266a47ac58a0150bc8e37b2427b12831eecd001f90313b83167`. It
contains only `fc.latent` and `fc.scale`, with reference magnitudes preserved at
0.5. Calibration consumed 1,536 fit rows and 768 validation rows from a
prompt/group/topic/content-disjoint TRAIN holdout. Per-domain validation
relative squared errors were0.06236 (code),0.06353 (prose), and0.06147
(reasoning); scale-only fit error was0.06265. These are fusion-coordinate
diagnostics only, not native acceptance, throughput, or model quality. The
initializer is not a full nine-projection export or native candidate admission.
Other artifact hashes: initializer manifest
`7d2b89a606005f07134222aa543df2baea056768b07f9e96d0825c6e5717fb09`, report
`4c2a75db9b0788aa71d4c4fe3926378f4356ad0b4e9acaf5eb57b9a57ae4597e`, row
evidence `acec4f98f58cbc8e1c39484fecd01c83b5c99047ef7e0060f2e5bbdda709860e`,
and raw selected inputs
`b05e4101335cadc9c5b122a64a2e8385915e75a7e240f4d414f18c735cd1a96e`.

The reviewed selection fix scans native rounds by descending prefix length and
chooses the longest structurally audited round with at least one admitted CE
label. Invalid ancestry still refuses. A bounded first-shard diagnostic located
the original refusal at source position23 for
`magicoder:line-012903-index-35307`: longest round82/prefix433 had one valid row
and zero CE labels; earlier round81/prefix431 had three valid rows and two CE
labels. It read no prompt body or raw tokens and did not repeat the corpus
audit.

## Native624 SM120 build

The native CUDA build ran from the clean parent
`5f53740776567d567e58d7a3930ca2164408c912` and exact submodule624. Ignored build
directory: `runs/build/native624-sm120-20261005`; supervisor run:
`runs/native624-sm120-build-20261005-01`. Linux tmux socket/session are
`binary-eagle-runtime` / `nine-model-native-build-20261005`. The build script is
`runs/operator-tools/native624_sm120_build.sh`, SHA256
`89f4d089b68103e4739e9beb501f4b460161267530a4dd159d298d5a10d829cc`. At launch,
supervisor PID/PGID46114 and child timeout PID/PGID46120 started07:30:45.709980Z.
The actual scope enforces `memory.max=12884901888` (12GiB); Ninja uses two jobs
and an outer2700-second timeout. Adjacent host memory was20,251,287,552 bytes
available and the project filesystem had198GiB free.

Environment/compiler flags: Python3.11.15 environment lock SHA
`33515d09381d8dba1a5059d0dbac474f9d3a9a84ebc5bce718a97d37253cac82`, CMake3.31.10,
NVCC13.1.115, GCC/G++15.2, CUDA arch120, `GGML_CUDA=ON`, FA/graphs ON,
forced cuBLAS/MMQ OFF, tests ON, and the existing
`cuda-glibc-compat/include` passed as `CMAKE_CUDA_FLAGS`. That include tree has
2,109 files and manifest SHA256
`134527a42f07c26488ae4630b046c40191abd0312a5f241a08dcade863b802f8`.
Configure completed successfully. The first run reached366/372 Ninja actions,
then the optional `test-block-binary` executable failed to link because
`libllama.so.0.5.0` references undefined
`llama_model_loader::get_key<bool>(std::string const&, bool&, bool)`. The
incremental follow-up of the required first-lane targets stopped on the same
reference when linking `llama-bench` and `llama-cli`. A bounded source inspection
found `llama-model-loader.cpp` explicitly instantiates the enum-key bool overload
and string-key uint32/string overloads, but omits string-key bool. This is a
native624 source/link issue that affects `libllama` consumers. The backend test
binary did link (SHA256
`2d3bd713bdefa07c7e3474244398b960fdcd39b9e79a34ff07e52c3e231f63d9`); its CUDA
runtime dependencies resolve to the build's `libggml-cuda.so.0` plus CUDA
13.1/cuBLAS/WSL driver. That binary was not run. The server, CLI, bench and
block-teacher executable paths are absent, so native runtime admission remains
pending. Both supervisors/process groups and the Linux tmux server exited; GPU
remained at its2,714MiB/0% idle baseline. The original and incremental failures
remain in `runs/native624-sm120-build-20261005-01/stdout.log` and
`runs/native624-sm120-build-20261005-02/stdout.log`; no source or gitlink was
edited by the operator.

## Native link repair and cc9-bound rebuild

The native fix was published as
`cc9cab3c64f61580cf63e5ef050b075b11cd1fb9` and adds the missing explicit
string-key `bool` instantiation. After Git-checking the owned build clone's
native submodule to that exact commit, I reran the original configure script
and all six required targets in the same isolated build directory. Run ID:
`native624-sm120-build-20261005-04`; Linux tmux socket/session:
`binary-eagle-runtime` / `nine-model-native-reconfigure-20261005`; local
transport MCP session `$258`, window `@287`, pane `%289`. Command:

```sh
tmux -L binary-eagle-runtime new-session -d -s nine-model-native-reconfigure-20261005 -c "$PWD" 'exec systemd-run --user --scope --property=MemoryMax=12G --property=MemorySwapMax=2G /home/philip/binary-eagle-decoding/.venv/bin/python scripts/remote_job.py native624-sm120-build-20261005-04 -- /usr/bin/timeout --signal=TERM --kill-after=10s 2700s /bin/bash /home/philip/binary-eagle-decoding/runs/checkouts/nine-model-qat-overnight-5f53740/runs/operator-tools/native624_sm120_build.sh'
```

The script SHA remains
`89f4d089b68103e4739e9beb501f4b460161267530a4dd159d298d5a10d829cc`. It
configured CUDA arch120 (CMake selected 120a), NVCC13.1, GCC/G++15.2, Release,
`GGML_CUDA=ON`, FlashAttention and CUDA graphs ON, forced cuBLAS/MMQ OFF, tests
ON, using the existing `cuda-glibc-compat/include` flag; Ninja parallelism2.
The user scope reported `memory.max=12884901888` and `MemorySwapMax=2G`; outer
timeout2700s. Supervisor51309 and child timeout51311 started08:01:03.376536Z,
then both exited naturally with code0 at08:01:06.377870Z. CMake's actual log
records `ggml commit: cc9cab3`, and the generated build-info source contains
`LLAMA_COMMIT = "cc9cab3"`.

The six target paths exist: `llama-server`, `llama-cli`, `llama-bench`,
`llama-block-teacher`, `test-backend-ops`, and `test-block-binary`. SHA256s in
that order are `7d315453eeab54964f5d5abba38d4d562561115d0e3989ff5dc7ddf835545d3c`,
`136e584530ec0482239d6b3b967d5043586c3dd9e5c3766659c6dacc409c0614`,
`c01cee2dfac32f9093966660c60f5e38e92e802445d8ee950fb9c88336b7dc97`,
`3b73926c29523b07675a29d908b3dde5a2a5e21c5c00e8b68a7934682528b497`,
`2d3bd713bdefa07c7e3474244398b960fdcd39b9e79a34ff07e52c3e231f63d9`, and
`4852e031389169560acad9b532e80b941b5dc1268945e8418596ed212d9ad684`.
Runtime libraries: `libllama.so.0.5.0`
`91ca9dfb2e1eb3b2972bf96b8e2e7073b349116a5958bdd53d9968cf07d0fdf4`,
`libggml-cuda.so.0.25.1`
`0571d53637fc20c2da5667f1b1d4b2515de300a2d2419397c8e977613a164966`,
`libggml.so.0.25.1`
`1a7c59ab10300bd6a9d1ee2c585d15603f2ede95c8b65235c78024053199b778`,
`libggml-base.so.0.25.1`
`36082d6378d124611b172ac932013cccbea6adc21a0555636f0c9d7d1fd4e562`, and
`libggml-cpu.so.0.25.1`
`e6b204dbe192269ea94e5142fa4fce90e87433ef30b3d9898d71b0d6b5f2cde4`. `nm`
confirms both intended `get_key<bool>` definitions are exported by libllama and
none remain undefined. The UI fallback asset archive was retained in the
ignored build directory; `dist.tar.gz` SHA256 is
`2f3f728fbfd7f8ea1a937bc49a34b9d17e2a06e10c09c0e54a091d8f950b5fd6`, and
`.ui-embed.sha256` contains the same archive digest. No target was executed and
no model was loaded.

Final release checks found supervisor/child PIDs absent, no Linux tmux server
or job process, and RTX5080 returned to 2,714MiB used / 13,264MiB free / 0%.
The actual build source parent was `5f53740776567d567e58d7a3930ca2164408c912`
with the intentional temporary gitlink change to cc9. After process release I
moved only this owned build clone to published parent
`0760c52c007c47bbb6f919864e135b2c27f4a1fa`, which pins native cc9, leaving the
checkout clean. The old dirty checkout and all historical runs remain
untouched.

## Remaining pre-QAT gates

This is a successful native compile/link preflight, not runtime admission.
Still required are the source-bound runtime JSON packet with actual build and
calibration pins, an authenticated candidate export from the sparse
initializer, native target/drafter acceptance and memory checks, the selected
Q4_0 comparator/control set for all requested model families, and an
independent review of those admission artifacts. AngelSlim package preflight
has passed read-only: the existing locked environment reports version0.5.0
with `direct_url.json` Git commit
`0358da9c651e6a7d7ccafea26ced4b9c98d11681`; the production Eagle3Config,
Llama3Eagle3Drafter, Eagle3Model, ModelLoader, AutoTokenizer, and Transformers
ROPE imports succeeded with CUDA hidden. Transformers is4.57.6 and its default
RoPE entry exists. No model/device load occurred. No QAT or model-quality
evaluation has started; RTX2080Ti remains unused.

The metadata-only source packet prepare was authorized and attempted as
`eagle-packet-metadata-20261005-01` with CUDA hidden and a 2GiB memory cap. It
exited1 at08:14:18Z before creating an output directory, leaving its exact
command and traceback in the ignored supervisor run directory. The production
initializer report SHA remains unchanged; its safe `fit_config` scalar fields
are `activation_bits=8`, `latent_initialization=preserve_reference_magnitudes`,
`max_coordinate_flips_per_row=0`, `max_seconds=300.0`,
`reference_kind=eagle_fixed_reference_0.5`, and
`zero_scale_orientation_rescue=false`. The packet helper currently requests
`orientation_rescue`, so its schema join needs an owner-reviewed compatibility
fix mapping the existing field. This failure did not read prompt content,
instantiate a model, allocate CUDA, or modify the completed initializer.

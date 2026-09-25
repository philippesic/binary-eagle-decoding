# RTX 2080 Ti quantization suite

**Date:** 2026-09-24–25  
**Owner:** SM75 experiment operator  
**Status:** Production W1A1 correctness gates and the frozen nine-variant paired matrix are complete on the RTX 2080 Ti. The integrated W1A1 MMA candidate also passed backend/SASS and short model parity gates. The separate W8A8/W4A4 branch gates remain pending; Q4_0/Q8_0 kernel/activation paths are source-inferred, not profiler-confirmed.

## Initial preflight (resolved below)

The following host/probe/run-directory notes capture the first access checkpoint. At that point no remote supervisor run had been started and the absent system toolchain looked blocking. That was resolved through a user-space CUDA 12.8 environment, supervised model staging/conversion, and the successful SM75 builds and tests documented below; the early stop note is historical.

- Host registry: `/Users/pippo/.config/binary-eagle-decoding/hosts.toml`, read via `python3 scripts/agent_env.py status`; `rtx2080ti` resolves to `philip@192.168.4.31:22`, workdir `~/binary-eagle-decoding`. Address is machine-local and is not committed.
- GPU pause state: no `rtx2080ti` pause entry was present in the returned `gpu_control` map; the 5080 flag was false.
- SSH was issued only through tmux MCP session `eagle-2080ti-preflight` (pane `%0`). This session already existed at start; its pane showed an idle local shell after earlier preflight commands. No remote run was started.
- Remote OS: Ubuntu 24.04 LTS under WSL2, kernel `6.18.33.2-microsoft-standard-WSL2`; user `philip`.
- Remote resources: 15 GiB RAM (14 GiB available), swap 4 GiB unused, home filesystem 1007 GiB total / 955 GiB available.
- GPU: NVIDIA GeForce RTX 2080 Ti, compute capability 7.5, 11,264 MiB VRAM. WSL reports NVIDIA-SMI 610.53, KMD 610.74, CUDA UMD 13.3. At the final preflight sample it was 31 C, P8, 0% utilization, 855 MiB used, 10,187 MiB free. The only listed GPU process was Xwayland (PID 36); compute-app query was empty.

## Exact commands and results

Local registry and repository status:

```sh
python3 scripts/agent_env.py status
git status --short --branch
```

The registry showed the values above. The checkout was on `main`; unrelated in-progress edits existed in `docs/RTX2080TI_RUNBOOK.md` and `docs/goals/rtx2080ti-quantization-suite.md` and were left untouched.

The current tmux pane also showed these earlier reachability probes (visible in its captured scrollback):

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=accept-new philip@192.168.4.31 'id -un; uname -sr; nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader; df -h ~'
ssh -o BatchMode=yes -o ConnectTimeout=8 philip@192.168.4.31 'whoami; ls -l /dev/dxg /usr/lib/wsl/lib/nvidia-smi 2>&1; /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader 2>&1; command -v nvcc || true; free -h'
ssh -o BatchMode=yes -o ConnectTimeout=8 philip@192.168.4.31 'printf "USER=%s\n" "$(id -un)"; cat /etc/os-release | head -6; for x in git python3 cmake ninja g++ clang++ curl nvcc; do printf "%s=" "$x"; command -v "$x" || true; done; ldd --version | head -1; /usr/lib/wsl/lib/nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,clocks.sm,power.draw --format=csv,noheader; /usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>&1; sudo -n true >/dev/null 2>&1; printf "SUDO_NOPASS=%s\n" "$?"'
```

The above established SSH reachability and WSL/GPU visibility. `nvidia-smi` was not in `PATH`; WSL's `/usr/lib/wsl/lib/nvidia-smi` worked. Ubuntu reports glibc 2.39. `sudo -n true` returned 1.

Final resource/process/toolchain check, issued via `mcp__tmux__execute_command` in pane `%0`:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=8 philip@192.168.4.31 'printf "GPU\n"; /usr/lib/wsl/lib/nvidia-smi; printf "COMPUTE\n"; /usr/lib/wsl/lib/nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv; printf "PROCESSES\n"; ps -eo pid,pgid,stat,etime,args | grep -E "remote_job.py|llama-(server|quantize)|test-backend-ops|sm75_mma_probe|nvcc|cmake --build" | grep -v grep || true; printf "RUNS\n"; ls -1 ~/binary-eagle-decoding/runs 2>/dev/null | tail -20 || true; printf "TOOLCHAIN\n"; for x in nvcc cmake ninja g++ curl; do printf "%s=" "$x"; command -v "$x" || true; done; df -h ~'
```

At that initial checkpoint there were no matching job/build processes or run directories; `nvcc`, `cmake`, `ninja`, and `g++` were absent from `PATH`, and noninteractive sudo was unavailable. This was an initial environment blocker only. The user-space bootstrap below resolved it, after which the build, correctness gates, conversions, and measured suite ran under supervision.

## Run directory and cleanup

At this initial preflight only, no `scripts/remote_job.py` run had been launched, so there was no preflight run ID or raw run directory. No project process had been started. The tmux session was pre-existing and left available; SSH probes completed with no remote shell/job left running. Later supervised runs and final cleanup are recorded below.

## SM75 correctness gates completed

The pinned production CUDA build completed under `runs/sm75-production-build-retry-20260924/` at 2026-09-24 22:49:09 UTC, exit 0. It compiled 368/368 targets for CUDA architecture 75 using CUDA 12.8.93 and GNU 13.4.0. Build stdout preserves configuration and build output. The `test-backend-ops` binary SHA256 is `5d3d81634e6b3254ed09db348de1d7bcf74aa1c623e29910336d5f94ce316b8f`; `llama-server` SHA256 is `443575e1a2dfb505b8f9310c0cd9cadb96a74e1d0346186dba6b5bfb1010eb18`.

The five-case W1A1 operator gate was run exactly as:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py sm75-w1a1-op-gate-20260924 -- build/llama-cuda/bin/test-backend-ops test -b CUDA0 -o W1A1_MUL_MAT
```

Run `runs/sm75-w1a1-op-gate-20260924/` finished exit 0 in about one second. On the NVIDIA RTX 2080 Ti, compute capability 7.5, the binary logged `CUDA packed W1A1 XOR/POPCOUNT dispatch` and passed K=31, 32, 33, 2560, and strided K=33 (5/5). The target was CUDA0, with CPU skipped as the non-target backend. The host showed 10,105 MiB free during the test. Post-run memory returned to 855 MiB / 0% utilization, the Xwayland desktop allocation only; no project process remained.

The standalone binary-MMA probe compile, SASS inspection, and default runtime test were one supervised run:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py sm75-mma-probe-20260924 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env bash -c 'set -e; nvcc -std=c++17 -O2 -arch=sm_75 kernels/sm75_mma_probe.cu -o runs/sm75-mma-probe-20260924/sm75_mma_probe; cuobjdump --dump-sass runs/sm75-mma-probe-20260924/sm75_mma_probe > runs/sm75-mma-probe-20260924/sm75_mma_probe.sass; grep -F "BMMA.88128.XOR.POPC" runs/sm75-mma-probe-20260924/sm75_mma_probe.sass; runs/sm75-mma-probe-20260924/sm75_mma_probe'
```

Run `runs/sm75-mma-probe-20260924/` finished exit 0 in about three seconds. `cuobjdump` found six static `BMMA.88128.XOR.POPC` sites in the SM75 SASS. The real 2080 Ti identified itself as SM75; default mode passed 21 shape/tail cases, totaling 880 exact integer dots against the independent dense CPU reference. Source SHA256 `c89695021e077248cee5700888c74030351ee58ac2bb1828ed293acca4aaebfd`; executable SHA256 `4aa93578867b82f1aa8c62a47fb174d1c03fe2b6dc8fe46248c466838d4b278f`; SASS SHA256 `a734083453c51649d6bc7e9a73da3637c6edc1866ead59a4f151603248fc78b0`; raw stdout SHA256 `ea8b261680c52263053003e1aa0d509a6564877ab0b5cf2020a5355b0f6a66a5`. After the run, process group 23309 was absent and GPU returned to 855 MiB / 0% with only Xwayland listed.

The optional integrated candidate was fetched into an isolated worktree at `runs/llama-sm75-mma-src/` at commit `9bb01a682ed4ba5e506870c8a38338589830b164`; the pinned production submodule remains at `8d2b18a9c9b3a42927404a91799a823c758c09b6`. Its supervised CUDA build completed 358/358 targets, exit 0, in `runs/sm75-integrated-mma-build-20260924/`; candidate binaries live under `build/llama-cuda-mma/`.

The candidate portable and opt-in binary-MMA operator gates were run separately:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py sm75-integrated-mma-portable-gate-20260924 -- env GGML_CUDA_W1A1_MMA=0 build/llama-cuda-mma/bin/test-backend-ops test -b CUDA0 -o W1A1_MUL_MAT
python3 scripts/remote_job.py sm75-integrated-mma-tensorcore-gate-20260924 -- env GGML_CUDA_W1A1_MMA=1 build/llama-cuda-mma/bin/test-backend-ops test -b CUDA0 -o W1A1_MUL_MAT
```

Both exited 0 on the RTX 2080 Ti, CC 7.5, and passed 8/8 scalar-reference cases, including the five base shapes plus K=128, a partial/tail K=129 tile, and the real-head shape `(K=2560, rows=320, tokens=1)`. Logs identify separate dispatch: `CUDA packed W1A1 portable XOR/POPCOUNT dispatch` for selector 0 and `CUDA packed W1A1 binary MMA dispatch (K=31, rows=7, tokens=3, cc=750)` for selector 1. GPU returned to 855 MiB / 0% after each gate.

The SASS inspection was supervised as `runs/sm75-integrated-mma-sass-20260924/`. CUDA library `build/llama-cuda-mma/bin/libggml-cuda.so.0.25.1` SHA256 is `1985378050b10c13958dd3c2394cc28bcc940a03c0d1f49b98f6b84891e679f3`; `cuobjdump --dump-sass` output SHA256 is `9a4eb5027d197146337b5ba0ceea2655c276eba4d26701afc1ebc09a1c37c1da`; the extracted seven BMMA instruction lines SHA256 is `c272c95c3de53b7304e3db4bc9e84f325898ef24250fc98a27bfa32885727f82`. The dump contains 7 static `BMMA.88128.XOR.POPC` sites. The full text dump is retained in the ignored run directory as produced by the supervised process; it is larger than the compact extracted site file.

Pinned target and draft source snapshots are staged under `models/hf/Qwen3-4B/` and `models/hf/Qwen3-4B_eagle3/`. Supervised run `runs/fetch-pinned-models-20260924/` resolved target `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` and draft `AngelSlim/Qwen3-4B_eagle3@fd331e59626c8e95c392381a16ee59d518727fbb`. All 18 source files' byte counts and SHA256 values match local `results/local-prep/model_manifest.json` (local manifest SHA256 `397937106d9de6f74d556455ccc3a292983e7b200a814dd69451f9a3705efa6c`). The remote manifest is `runs/fetch-pinned-models-20260924/model-manifest.json`, SHA256 `e665b63bc4470f1a3d59015fe641eedb3e4ce824894c6ebae9d7eb0a7aa2d539`. No `.part` files remained at the final download check.

## Next action

The production paired matrix is sealed and its raw outputs are hashed below. Continue with the separate integrated W1A1 MMA four-path matrix, then the published W8A8/W4A4 source correctness gates in isolated supervised runs. Keep weight-only Q4_0/Q8_0 timing labels distinct from the W8A8/W4A4 research contracts until an executed-kernel trace proves the path.

## WSL user-space setup and build checkpoint

Python probe: Python 3.12.3, OpenSSL 3.0.13, Python HTTPS to `https://pypi.org/simple/` returned HTTP 200, and DNS resolved PyPI and GitHub. The machine provides Git and curl, but no compiler, CMake, Ninja, pip, or CUDA toolkit in the base WSL image. GitHub SSH host verification failed; HTTPS Git access succeeded. The remote project was cloned recursively over HTTPS with SSH submodule URLs rewritten to HTTPS, then fast-forwarded to current `main`.

Remote project source state after fetch:

- Parent commit: `c425ca2eb504a6d7bd55ef322b103f16846d0567`
- llama.cpp commit: `8d2b18a9c9b3a42927404a91799a823c758c09b6`
- Checkout: clean `main...origin/main`

Micromamba 2.9.0 is installed under ignored `runs/toolchain-bootstrap/`; archive SHA256 is `8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd`. The environment prefix is `runs/toolchain-bootstrap/env`, package root is `runs/toolchain-bootstrap/mamba-root`, and setup logs remain in that ignored directory. Noninteractive sudo is unavailable; no system files were modified.

Exact remote bootstrap and package commands (all SSH invocations above ran via tmux MCP):

```sh
mkdir -p ~/bootstrap
git -c "url.https://github.com/.insteadOf=git@github.com:" clone --recurse-submodules https://github.com/philippesic/binary-eagle-decoding.git ~/binary-eagle-decoding
git -C ~/binary-eagle-decoding fetch origin main
git -C ~/binary-eagle-decoding merge --ff-only origin/main
git -C ~/binary-eagle-decoding submodule update --init --recursive
mkdir -p ~/binary-eagle-decoding/runs/toolchain-bootstrap/bin
curl --fail --location --silent --show-error https://micro.mamba.pm/api/micromamba/linux-64/2.9.0 -o ~/binary-eagle-decoding/runs/toolchain-bootstrap/micromamba.tar.bz2
python3 -c 'import tarfile; p="/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/micromamba.tar.bz2"; t=tarfile.open(p,"r:bz2"); m=t.extractfile("bin/micromamba"); q="/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/bin/micromamba"; open(q,"wb").write(m.read())'
chmod +x ~/binary-eagle-decoding/runs/toolchain-bootstrap/bin/micromamba
export MAMBA_ROOT_PREFIX=~/binary-eagle-decoding/runs/toolchain-bootstrap/mamba-root
~/binary-eagle-decoding/runs/toolchain-bootstrap/bin/micromamba create --yes --prefix ~/binary-eagle-decoding/runs/toolchain-bootstrap/env --channel nvidia --channel conda-forge cuda-nvcc=12.8 cuda-cudart-dev=12.8 cmake ninja gcc_linux-64=13 gxx_linux-64=13 python=3.12 git
~/binary-eagle-decoding/runs/toolchain-bootstrap/bin/micromamba install --yes --prefix ~/binary-eagle-decoding/runs/toolchain-bootstrap/env --channel nvidia --channel conda-forge cuda-libraries-dev=12.8 cuda-driver-dev=12.8
```

The env resolves CUDA compiler 12.8.93, GCC/G++ 13.4.0, CMake 4.4.3, Ninja 1.13.2, and Python 3.12.14. NVIDIA's minimal `cuda-nvcc` plus `cuda-cudart-dev` packages initially omitted cuBLAS; a supervised configure stopped at missing `CUDA::cublas`. Adding `cuda-libraries-dev=12.8` and `cuda-driver-dev=12.8` corrected that provisioning failure. Its install log is `runs/toolchain-bootstrap/cuda-libs-install.log`; the initial package plan `cuda-cublas-dev=12.8` was invalid and the NVIDIA-only solve lacked `ocl-icd`, so the successful solve included conda-forge.

The failed supervised configure/build attempt is preserved as `runs/sm75-production-build-20260924/` with `state.json` and `stdout.log` (exit 1, CMake generation failed before compilation or GPU work). The corrected production build completed all 368 targets in `runs/sm75-production-build-retry-20260924/` with exit 0. It was launched by:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py sm75-production-build-retry-20260924 -- ~/binary-eagle-decoding/runs/toolchain-bootstrap/bin/micromamba run -p ~/binary-eagle-decoding/runs/toolchain-bootstrap/env env CC=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-gcc CXX=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ python3 scripts/build_llama.py cuda --with-tests --cuda-arch 75 --jobs 4
```

The corrected environment is CUDA 12.8.93 / GNU 13.4.0. The repeated nvcc `compiler-bindir` redefinition warning did not prevent a successful build. Model sources and GGUFs are staged under ignored paths; no system packages were installed. Full conversion, audit, model smoke, and matrix provenance follows.

## Pinned GGUF artifacts and row audits

The remote source tree was `~/binary-eagle-decoding`; models and raw results are ignored paths on the WSL host. The 18 source files were fetched from target `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c` and draft `AngelSlim/Qwen3-4B_eagle3@fd331e59626c8e95c392381a16ee59d518727fbb`. Remote source-manifest SHA256: `e665b63bc4470f1a3d59015fe641eedb3e4ce824894c6ebae9d7eb0a7aa2d539`.

F16 conversions used the pinned llama.cpp converter with CPU-only environment `runs/toolchain-bootstrap/convert-env`:

```sh
python3 scripts/remote_job.py convert-target-f16-20260924 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/convert-env env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B --outtype f16 --outfile models/gguf/Qwen3-4B-f16.gguf
python3 scripts/remote_job.py convert-draft-f16-20260924 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/convert-env env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B_eagle3 --target-model-dir models/hf/Qwen3-4B --outtype f16 --outfile models/gguf/Qwen3-4B-eagle3-f16.gguf
```

The same converter emitted the five W1A1 group variants with `--w1a1-eagle-groups fusion|attention|ffn|all` and `--w1a1-eagle-head` for the head-only draft. All conversions ran under unique `scripts/remote_job.py` IDs. The Q4_0/Q8_0 standard weight-only controls were made with supervised `llama-quantize --pure <f16-draft> <output> Q4_0 8` and `Q8_0 8` commands; the production converter's original undefined-symbol failure and the successful quantizer from the isolated candidate build are both preserved in their run directories.

| GGUF | Bytes | SHA256 |
|---|---:|---|
| Target F16 | 8,051,285,280 | `05a259dca043f1089ec94ace1edc2a0086e4264c805eee81f57cc57f2dc720a6` |
| Ordinary EAGLE draft F16 | 442,700,800 | `c1f895a130b64cd3d5a97fba7aa7605dc7fe3a389dd6d48e6751128614ee76d1` |
| Fusion W1A1 | 405,847,648 | `5b47b7b017b100edb266412f6e689e0ace5f0f1d7af26178caa3ae4bd91f8e0b` |
| Attention W1A1 | 364,094,112 | `3fa4aca2cfe6c5a59449bcbab8d6bcdb72a3ffcd9fc3f0193fbf009776a83f67` |
| FFN W1A1 | 302,707,008 | `518b8f624209921a3a6c971c12c20350dbd0e8e8d6a02fc196085093d8a69ae8` |
| Head W1A1 | 289,229,312 | `b2095130b5196574a9a08a88d2fb9a32ac1ea870ff3cf587ae7d6e7f64e7819f` |
| All W1A1 | 33,774,784 | `098e1ecbb299aa16e2c968663acc49e60c0fcf16b053766d9f558114f79d011c` |
| Draft Q4_0 | 128,988,160 | `2db40f99d27e404298b80b2865671b9fd0136060ffb503007cb2ae23759e7280` |
| Draft Q8_0 | 238,105,600 | `29ee91f09971555458cf420461fdfeeeabc9236f8a2c20662b7333a980a49507` |

The five W1A1 converter source-to-GGUF audits found zero sign-word and packed-row mismatches. Audit JSON SHA256 / raw log SHA256:

| Scope | Audit SHA256 | Log SHA256 |
|---|---|---|
| Fusion (1 tensor, 2,560 rows) | `63cf006a333ab6ba973db35f6c31cc7e4e9886b33fabac2c1bd9999a248b6a8a` | `17b4f3e1d2a8b021c5eddc387f0da18c870235c6466f859f6364a92b8127d010` |
| Attention (4 tensors, 8,704 rows) | `2d61ec64a1b20549c38a3c75357a42348910a336636582fd24d38095e1009cc7` | `2a458c26a160871553211e9d5ff46abf7aa186d7f78c9f196412525a0d08541a` |
| FFN (3 tensors, 22,016 rows) | `6a15a827b77ae02ad80aefd0af3b2ee4fdd95baf7bdb9bb336cde29db3a81318` | `6bb71f9c706c1046cb5cf4fd78806952f3d7989090208927e0ba614987ec5e8c` |
| Head (1 tensor, 32,000 rows) | `795bedb90af9c706462ebd0b0758b5f772a2c8d766c4601a39be8330f82f4edc` | `96fb397e4563d68581c12667016e8cc480c648f064da5d9ea2ca55badaa2c6bc` |
| All (9 tensors, 65,280 rows) | `872719a0eb911bb2446bb485a22923a4677fc4955fb3b17b518f12dc9cb6ed4b` | `13bdcba3c28a8956f0bcfb2b977336d1473324e3d638f1eb77c8c0e466ef3e6b` |

For all five W1A1 scopes, maximum absolute scale error was at most `7.4506e-9`; the fusion audit's maximum was `5.5879e-9`. The GGUF tensor audit found nine Q4_0 or Q8_0 linears respectively, four F32 tensors, and one I64 mapping tensor. Q4_0/Q8_0 name the stored GGUF weight formats. The pinned source predicts Q4_0 × Q8_1 nibble expansion plus byte DP4A and Q8_0 × Q8_1 signed-byte DP4A at decode batch 1; those exact runtime paths were not profiler-traced in this experiment and must not be presented as the separate W4A4/W8A8 contract.

## Frozen nine-variant production matrix

The measured run was launched from the clean remote parent commit `f99affa3625a9d3631c99bafc748bbcf429368ea`, with llama.cpp gitlink and checkout both `34e21b7d85c17e25d5a91ce2ab1074d4c18bfe39`. CUDA architecture was SM75; the hardware identified as an NVIDIA GeForce RTX 2080 Ti with 11,264 MiB. Raw artifacts remain at `/home/philip/binary-eagle-decoding/results/native-nine-quantization-full-run-20260924/` on the remote host.

The exact supervised command was:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py native-nine-quantization-full-supervisor-20260924 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/benchmark_native_eagle.py --config runs/toolchain-bootstrap/native-full.toml --run-id native-nine-quantization-full-run-20260924
```

The frozen config hash is `3537fbddbb70b0da833c5048f086b106d71f2f66180840fbbc76118b7791fc31`. It used 12 prompts, 2 warmups, 5 measured repetitions, 128-token cap, `seed=42`, `temperature=0.0`, `cache_prompt=false`, `enable_thinking=false`, and one server at a time. Variant order was rotated across repetitions. The supervisor ran from `2026-09-25 00:02:32 UTC` to `00:28:07 UTC`, finished with exit 0, and collected 540/540 records (60 per variant). All five packed W1A1 group variants have CUDA dispatch confirmation in all five repetitions. No record had an error. Finish reasons were `length` for 495 requests and `stop` for 45.

The pooled request rate is completion tokens / client request wall time, including prefill. Decode rate is completion tokens / server `predicted_ms`, excluding prompt processing. Acceptance is accepted draft tokens / proposed draft tokens; acceptance per round is also shown. `target_only` has no draft acceptance counters.

| Variant | Request tok/s | Decode tok/s | Accepted / proposed | Rounds | Accepted / round | Per-prompt request tok/s range | Per-prompt decode tok/s range |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target only | 59.792 | 61.668 | — | — | — | 59.12–60.34 | 61.36–62.10 |
| Ordinary EAGLE | 77.348 | 82.488 | 3,970 / 16,740 (23.72%) | 3,400 | 1.168 | 58.55–97.20 | 60.40–103.19 |
| Draft Q4_0 | 85.197 | 91.508 | 3,990 / 16,615 (24.01%) | 3,375 | 1.182 | 66.19–104.97 | 68.56–112.09 |
| Draft Q8_0 | 81.947 | 87.732 | 3,970 / 16,735 (23.72%) | 3,400 | 1.168 | 63.16–102.51 | 65.33–108.84 |
| W1A1 fusion | 47.441 | 49.319 | 1,565 / 28,490 (5.49%) | 5,780 | 0.271 | 43.55–51.96 | 44.99–52.92 |
| W1A1 attention | 47.311 | 49.172 | 1,410 / 29,245 (4.82%) | 5,925 | 0.238 | 42.65–52.48 | 43.64–54.63 |
| W1A1 FFN | 56.324 | 59.048 | 2,360 / 24,630 (9.58%) | 5,005 | 0.472 | 48.98–64.90 | 50.28–67.34 |
| W1A1 head | 70.514 | 74.813 | 3,405 / 19,480 (17.48%) | 3,960 | 0.860 | 55.03–87.54 | 56.67–101.12 |
| W1A1 all | 44.649 | 46.305 | 385 / 34,195 (1.13%) | 6,940 | 0.055 | 42.08–46.52 | 43.89–47.71 |

The table's per-prompt ranges pool the five repetitions for each prompt, then take the min/max over the 12 prompts. The separate category summaries in `analysis.json` show decode-rate ranges of 45.05–46.97 tok/s for W1A1 all, 46.39–50.62 for attention, 52.90–64.89 for FFN, 47.95–51.40 for fusion, and 64.31–86.18 for head. The corresponding accepted-per-round ranges across prose/code/reasoning prompts are 0.027–0.070, 0.166–0.278, 0.321–0.617, 0.234–0.325, and 0.597–1.153. Q4_0, Q8_0, and ordinary EAGLE category decode-rate ranges were 75.47–103.35, 71.32–101.22, and 66.82–95.08 tok/s.

The server's cumulative EAGLE timing lines also yielded the following measured-only host timing decomposition. `begin_ms`, `draft_ms`, and `accept_ms` are from `common_speculative_impl`, not isolated GPU-kernel timings; draft ms/round divides the pooled draft-call time by the measured verification rounds. `accept_ms` is the logged acceptance-hook span and is not total target-verification latency. The run does not expose an isolated verifier span, so that cost cannot be separated here.

| Variant | begin ms | draft ms | accept ms | draft ms / verification round |
|---|---:|---:|---:|---:|
| Ordinary EAGLE | 0.087 | 22,001.561 | 4.151 | 6.471 |
| Draft Q4_0 | 0.091 | 14,102.330 | 4.178 | 4.178 |
| Draft Q8_0 | 0.089 | 16,981.566 | 4.110 | 4.995 |
| W1A1 fusion | 0.087 | 37,109.169 | 6.988 | 6.420 |
| W1A1 attention | 0.088 | 34,683.928 | 7.105 | 5.854 |
| W1A1 FFN | 0.091 | 27,299.239 | 5.877 | 5.454 |
| W1A1 head | 0.094 | 20,840.759 | 4.985 | 5.263 |
| W1A1 all | 0.079 | 24,695.913 | 8.531 | 3.558 |

Target-only has no EAGLE timing decomposition. These are available for all eight speculative variants; the native `draft_ms` measure is host wall time around draft calls and should not be read as standalone GPU kernel duration.

For the all-group W1A1 draft, draft-call cost per verification round fell from 6.471 ms with ordinary EAGLE to 3.558 ms, while rounds increased from 3,400 to 6,940; pooled draft-call time therefore rose from 22.0 s to 24.7 s. Its aggregate acceptance rate was only 1.13%, so the shorter proposal call did not offset the additional rounds.

Paired bootstrap intervals used 2,000 resamples with seed 42, resampling prompts and repetitions together. Representative 95% intervals for pooled candidate/ordinary-EAGLE rate ratios are:

| Variant | Request-rate ratio median [95% interval] | Decode-rate ratio median [95% interval] |
|---|---:|---:|
| Q4_0 | 1.101 [1.074, 1.127] | 1.109 [1.081, 1.135] |
| Q8_0 | 1.060 [1.048, 1.071] | 1.064 [1.053, 1.075] |
| W1A1 fusion | 0.613 [0.564, 0.671] | 0.597 [0.545, 0.660] |
| W1A1 attention | 0.611 [0.571, 0.657] | 0.596 [0.552, 0.646] |
| W1A1 FFN | 0.728 [0.691, 0.765] | 0.715 [0.677, 0.756] |
| W1A1 head | 0.911 [0.875, 0.950] | 0.907 [0.869, 0.946] |
| W1A1 all | 0.577 [0.530, 0.630] | 0.561 [0.511, 0.617] |

The intervals are descriptive for this prompt set and machine, not corrections for systematic bias. Relative to target-only, the W1A1 head pooled rates were 1.179 request-rate and 1.213 decode-rate ratios, but target-only and all speculative outputs had a stable text difference on one prompt; those ratios are timing observations, not strict lossless speedups.

### Paired completion validation and timing limits

All eight speculative variants (ordinary EAGLE, five W1A1 scopes, Q4_0, and Q8_0) matched one another's completion hashes on all 60 prompt/repetition pairs. Target-only output was also identical across all five repetitions for every prompt. Against target-only, each speculative variant differed only on `reasoning-02`, in the same way for all five repetitions: the answer's markdown heading is `### Step-by-Step Solution:` for target-only versus `### Step-by-step solution:` for the speculative variants. The first differing character is 440; the full strings are 503 and 507 characters. Both use the same request body, with `seed=42`, `temperature=0`, `max_tokens=128`, and `enable_thinking=false`. This is a stable formatting difference, not evidence of different reasoning content. The server omitted generated token IDs from all 540 responses despite `return_tokens=true`, so token-level equality cannot be checked.

Server-predicted decode durations were present and positive for all 540 records. Client time-to-first-token was unavailable in all records because the run used non-streaming responses; no TTFT comparison is claimed. The Q4_0/Q8_0 format labels are verified at GGUF storage/load time; exact activation conversion and CUDA kernel dispatch remain untraced in this matrix.

### Run integrity, hashes, and cleanup

The initial GPU snapshot was RTX 2080 Ti, driver `610.74`, 11,264 MiB total / 855 MiB used, at 300 MHz. The largest live memory reading during the matrix was 10,160 MiB used / 868 MiB free; the maximum `gpu-loaded.json` snapshot was 10,110 MiB. The final sample was 855 MiB used / 10,173 MiB free at 0% utilization. The supervisor, benchmark process, and llama-server process were absent at final check; the GPU is released.

Raw result SHA256 values:

| Artifact | SHA256 |
|---|---|
| `manifest.json` | `8fe6358677f4891b71e78f0f6be7a23211979764a24ac6d915a8711d09333fd9` |
| `report.json` | `54e4ba42f4130856000db25ce36cf1b93026bc66a76dc839339e9b5d0114d13f` |
| `records.json` | `04c4a1a71244f8b8f18077e5d0a192455d36625d780c990a1742da6628507889` |
| `prompts.jsonl` | `0d6a698d6816592c6ff435fed2fea4cdafe9f5248393d2a5ac091e1551919476` |
| `analysis.json` | `9e1a8b5b2f48394b927e6825927edb8f86d58bf2908eec52af8777f9d20d14a1` |

The analysis command also ran under `scripts/remote_job.py` with run ID `native-nine-quantization-analysis-20260924`:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py native-nine-quantization-analysis-20260924 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env python scripts/analyze_native_benchmark.py --run-dir results/native-nine-quantization-full-run-20260924 --samples 2000 --seed 42 --output results/native-nine-quantization-full-run-20260924/analysis.json
```

Raw `manifest.json`, `report.json`, `records.json`, `prompts.jsonl`, every per-request response/metrics file, GPU snapshots, and per-variant server logs remain in the ignored remote run directory. No raw run data was added to Git.

## Four-path portable versus binary-MMA head matrix

After the sealed production run, a second independent matrix compared target-only, ordinary EAGLE, portable packed W1A1 head, and the opt-in packed-head binary-MMA candidate. The remote parent was `decec7a7ef5c09e51a1afb3fe3766560b338efcb`; the production submodule gitlink remained `34e21b7d85c17e25d5a91ce2ab1074d4c18bfe39`. The isolated candidate build tree was commit `9bb01a682ed4ba5e506870c8a38338589830b164`. Its CUDA binary was `build/llama-cuda-mma/bin/llama-server`, SHA256 `365edec352eb3cc57fa7649317ea3354410363a475afdeced0ea345d748355cf`.

The config `runs/toolchain-bootstrap/native-mma-head-full.toml` was copied from `configs/native_benchmark.toml`, selected the integrated candidate server, and enabled `evaluation.binary_mma = true`. Its SHA256 is `3fa9a7d1254c47a3867dba30b72c212525363b8e3eb455997bea01afc85b1012`. The exact supervised commands were:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py native-mma-head-full-supervisor-20260925 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/benchmark_native_eagle.py --config runs/toolchain-bootstrap/native-mma-head-full.toml --run-id native-mma-head-full-run-20260925
python3 scripts/remote_job.py native-mma-head-analysis-20260925 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env python scripts/analyze_native_benchmark.py --run-dir results/native-mma-head-full-run-20260925 --samples 2000 --seed 42 --output results/native-mma-head-full-run-20260925/analysis.json
```

The run used the same 12 prompts, 2 warmups, five measured repetitions, 128-token cap, target F16 model, ordinary F16 EAGLE draft, and frozen W1A1-head GGUF as the production matrix. It finished exit 0 at `2026-09-25 00:54:48 UTC` with 240/240 records (60 per path). Every portable-head record records selector `0`; every MMA-head record records selector `1`. Both portable and MMA variants have confirmed CUDA dispatch across all five repetitions, and the report confirms the expected binary-MMA marker separately from portable XOR/POPCOUNT.

| Variant | Request tok/s | Decode tok/s | Accepted / proposed | Rounds | Accepted / round |
|---|---:|---:|---:|---:|---:|
| Target only | 59.515 | 61.462 | — | — | — |
| Ordinary EAGLE | 77.129 | 82.261 | 3,970 / 16,740 (23.72%) | 3,400 | 1.168 |
| Portable W1A1 head | 70.388 | 74.591 | 3,405 / 19,480 (17.48%) | 3,960 | 0.860 |
| Binary-MMA W1A1 head | 70.523 | 74.720 | 3,405 / 19,480 (17.48%) | 3,960 | 0.860 |

MMA/portable pooled rate ratios were 1.0019 request and 1.0017 decode. Paired 95% bootstrap intervals (2,000 prompt-and-repetition resamples, seed 42) were `[0.9966, 1.0068]` for request rate and `[0.9973, 1.0062]` for decode rate, so this matrix measured no head-path speed difference. The report's MMA/ordinary ratios were 0.914 request and 0.908 decode. The target-only ratios were 1.185 request and 1.216 decode, subject to the same stable reasoning-02 formatting divergence described above.

Portable W1A1 and binary-MMA W1A1 produced identical decoded text on all 60 paired inputs. Ordinary EAGLE, portable W1A1, and MMA W1A1 each had the same five formatting-only text differences from target-only on `reasoning-02`; target-only remained stable over repetitions. Generated token IDs were omitted by the server in all 240 responses despite `return_tokens=true`. Decode timing is available, but non-streaming requests did not provide client TTFT. Finish reasons were `length` for 220 records and `stop` for 20.

The initial GPU sample was 855 MiB used / 10,173 MiB free; the largest sampled matrix reading was 10,160 MiB used / 868 MiB free. After the analysis job, no `llama-server`, benchmark runner, or supervisor process remained and GPU memory returned to 855 MiB / 0% utilization.

Raw files are on the WSL host at `/home/philip/binary-eagle-decoding/results/native-mma-head-full-run-20260925/`:

| Artifact | SHA256 |
|---|---|
| `manifest.json` | `ca7f6647b1731fcac8b6d6494cf3a7d0d348fb8e34bed7b2be17ce9cdb48b8a1` |
| `report.json` | `0ee964ea70cc76ab7a479eea6291334bc783af370b36c8d07dabe1f12afba665` |
| `records.json` | `ad53f8f9875b786117894e1dd54a637d7bea04976aead6609631a5746ef9ce3f` |
| `prompts.jsonl` | `0d6a698d6816592c6ff435fed2fea4cdafe9f5248393d2a5ac091e1551919476` |
| `analysis.json` | `9a996f457fa5864166e920dd08f185596b80da75e1009585fc1f8d2b9ad229c2` |

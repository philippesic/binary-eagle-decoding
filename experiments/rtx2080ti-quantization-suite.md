# RTX 2080 Ti quantization suite

**Date:** 2026-09-24–25  
**Owner:** SM75 experiment operator  
**Status:** The full W1A1, W8A8, W4A4, Q4_0/Q8_0, and opt-in Tensor Core comparisons are complete on the RTX 2080 Ti. The combined MMA candidate passed its SM75 backend/probe/model/SASS gates, then its full 12-prompt/five-repetition matrix measured both INT8 and INT4 MMA kernels slower than their same-format default paths with identical acceptance. A separate executed-path trace confirmed Q4_0/Q8_0 use Q8_1 activation quantization, MMVQ at one/two-token shapes, and MMQ at an observed 38-token shape. All raw matrices and the trace are hashed below.

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

The production, W1A1 MMA, default W8A8/W4A4 vector, and opt-in W8A8/W4A4 MMA matrices and the Q4_0/Q8_0 path diagnostic are sealed with raw hashes below. Keep Q4_0/Q8_0 block-scale format labels distinct from the W8A8/W4A4 research contracts.

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

## Combined W8A8/W4A4 SM75 correctness and export audit

The combined default-vector source was built from llama.cpp commit `d0724427b61f6ff4733b502d0ce3d800a2f4cd86` in isolated worktree `runs/llama-sm75-lowbit-src/`. The parent was `b9596675225271634881f631146d1e76bfce1a74`; its committed llama.cpp gitlink stayed `34e21b7d85c17e25d5a91ce2ab1074d4c18bfe39`. Before the low-bit full matrix, the production submodule working tree was temporarily checked out to d0724427b so the benchmark manifest records the actual compiled source; after the supervised run ended and no server process remained, it was restored to 34e21b7. The parent gitlink never changed.

The supervised CUDA build was `sm75-eagle-int-lowbit-build-d0724427b-20260925`: 360/360 targets, exit 0, CUDA 12.8.93, architecture SM75. It built the isolated candidate into `build/llama-cuda-lowbit/`. The preserved build-log SHA256 is `aed016472e038a7501b7ec3e1a82e91be5f561c06b0a9e1ad6e03569f79ad969`; `test-backend-ops` SHA256 is `d6271d940deb54799ebc7fa2850d4cc1f2cbf7bde97c78b7e8804bb8cb9cf045`, `llama-server` is `bffe57951e5c4fafd47bba9bca970931ac008e7fe4b97418eac3b7c257213b69`, and `libggml-cuda.so.0.25.1` is `042c8184102f7b9512733f0f9711ea01add20dc39bc0c377b5cde92727326c86`. The only repeated build diagnostic was the known Conda `compiler-bindir` redefinition warning.

The build command, run through `scripts/remote_job.py`, was:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py sm75-eagle-int-lowbit-build-d0724427b-20260925 -- bash -c "set -eu; runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env env CC=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-gcc CXX=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ cmake -S runs/llama-sm75-lowbit-src -B build/llama-cuda-lowbit -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF -DGGML_CUDA=ON -DGGML_METAL=OFF -DCMAKE_CUDA_ARCHITECTURES=75; runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env env CC=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-gcc CXX=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=/home/philip/binary-eagle-decoding/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ cmake --build build/llama-cuda-lowbit --parallel 4 --target test-backend-ops llama-server"
```

Two independent supervised backend gates passed on the RTX 2080 Ti, compute capability 7.5:

| Gate | Run ID | Result | Runtime marker | Raw log SHA256 |
|---|---|---:|---|---|
| `W8A8_MUL_MAT` | `sm75-w8a8-backend-ops-d0724427b-20260925` | 3/3, K=8/33/9728 | `CUDA W8A8 signed INT8 dot/I32 accumulation dispatch` | `f609fa21de5594acbf1e43b8e5b4cede00c47656068e640ab3ec046c277ff85b` |
| `W4A4_MUL_MAT` | `sm75-w4a4-backend-ops-d0724427b-20260925` | 2/2, K=9/9728 | `CUDA W4A4 signed-nibble vector dot, scalar integer MUL/ADD; no INT4 Tensor Core MMA` | `3c2e75fcedf08c0d58ee279bb396581e8b7ab4621dc2384847dd214656843686` |

These are the backend cases registered in this source revision; the expanded W8A8 real-row/tail coverage in the source-gate plan remains a follow-up. Both tests returned the GPU to its idle baseline of 855 MiB / 0% utilization.

The separate signed-I4 MMA probe, `sm75-w4a4-signed-i4-mma-probe-d0724427b-20260925`, is not wired into production W4A4 dispatch. It passed six shape/basis cases with 308 exact I32 outputs, including K=9/33/9728 and N=1/N=9. `cuobjdump --dump-sass` found `IMMA.8832.S4.S4.SAT` in the executed probe binary. Source SHA256 `a475b66cf7ebbbc5bf2bda6a8296070a0b68d811b2cb2982def01af7047d5285`, executable SHA256 `09bf81398bf102be029bfb6929b84a463967003cf6634b5ebd1c53bfb7f030d0`, SASS SHA256 `d4bc243e1db7d32d37353e95bb673cb7d0574ad19a723dbda5db396a386f419d`, and stdout SHA256 `94ee22ceb47369791c8ab0fcdd971a0014ab1010650f17a277e3efcad6d839f3`. This proves the standalone instruction-layout probe on SM75; it does not prove production W4A4 dispatch or speed. The probe job returned the GPU to 855 MiB / 0%.

### WSL W8A8/W4A4 export and source audit

Both draft GGUFs were generated under `scripts/remote_job.py` from the pinned BF16 checkpoint `models/hf/Qwen3-4B_eagle3/model.safetensors`, SHA256 `58ac5bbfdd71047ebaa5d5535b895c2af37004eb820ca2dda55bd7666658853e`, using the converter in the d0724427b worktree:

```sh
python3 scripts/remote_job.py convert-w8a8-d0724427b-20260925 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/convert-env env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python runs/llama-sm75-lowbit-src/convert_hf_to_gguf.py models/hf/Qwen3-4B_eagle3 --target-model-dir models/hf/Qwen3-4B --outtype f16 --w8a8-eagle --outfile models/gguf/Qwen3-4B-eagle3-w8a8-d0724427b.gguf
python3 scripts/remote_job.py convert-w4a4-d0724427b-20260925 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/convert-env env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python runs/llama-sm75-lowbit-src/convert_hf_to_gguf.py models/hf/Qwen3-4B_eagle3 --target-model-dir models/hf/Qwen3-4B --outtype f16 --w4a4-eagle --outfile models/gguf/Qwen3-4B-eagle3-w4a4-d0724427b.gguf
```

An ignored WSL audit script, SHA256 `81cfbe97a344638cd388a024265a97c783d09a79e270bb8d6c61a922a576d3df`, read each GGUF back and re-quantized all nine pinned BF16 source tensors after the 32-Q-head / 8-K-head RoPE row permutation. It checked versioned metadata, all-nine tensor names and logical K, raw I8 code/packed matrices, F32 row scales, tensor inventory and absence of dense shadows.

| Format | GGUF bytes | GGUF SHA256 | Audit run | Code mismatches | F32 scale byte mismatches |
|---|---:|---|---|---:|---:|
| W8A8 | 224,730,560 | `48d8c517253ee24278412efc18eaf38340d6e9eede4fc819f64ab268dab590d8` | `audit-w8a8-source-d0724427b-20260925` | 0 across 9 linears | 0; all scales byte-exact |
| W4A4 | 115,613,408 | `0471dd2a1ac7628ae97018dad5d24aaf08cbfc6a258758a975d4e60b0d40beed` | `audit-w4a4-source-d0724427b-20260925` | 0 across 9 linears | 0; all scales byte-exact |

W8A8 conversion/audit log SHA256 values are `1de672a30b1c4c275f428e131dfdb46fb9e58e8825ff8323a9a5067d4bc1a99c` and `e0b5124e14ecb8263e61dce07d48b6ce35e6095a9085c3b172dac152e2622c39`. W4A4 conversion/audit log SHA256 values are `57275f135f40243952a38dae744e460dde2590a8bb5a49262e5b8ccb00d9fc36` and `dd5add325a24807b7a8ff22e1a314114acb2441178c1dd26db3c46b058a8e9fc`. Both exports match the earlier local CPU-gate GGUF hashes byte-for-byte. No host-dependent F32 scale differences were observed on WSL.

### All-nine WSL model-load/dispatch smoke

The harness dry-run validated an ignored one-prompt, five-repetition config (seed 42, temperature 0, 128-token cap, F16 target/KV/context 2048). The supervised model smoke was `sm75-eagle-int-lowbit-model-smoke-d0724427b-20260925`, result directory `results/native-lowbit-model-smoke-d0724427b-20260925`. It compared target-only, ordinary EAGLE, portable W1A1 head, W8A8, and W4A4 on `prose-01` only (25 measured requests total). W8A8 and W4A4 both passed all-nine loader and exact CUDA operator markers. Every variant matched target-only text 5/5 on this prompt. Token IDs were omitted by the server in all 25 responses.

| Smoke variant | Accepted / proposed | Rounds | Accepted / round | Acceptance |
|---|---:|---:|---:|---:|
| Ordinary EAGLE | 230 / 1,960 | 400 | 0.575 | 11.73% |
| W1A1 head | 180 / 2,200 | 450 | 0.400 | 8.18% |
| W8A8 vector | 230 / 1,960 | 400 | 0.575 | 11.73% |
| W4A4 vector | 45 / 2,875 | 585 | 0.077 | 1.57% |

The W4A4 smoke shows a severe acceptance loss on this one prompt; the full prompt suite below measures its aggregate impact. The maximum loaded GPU snapshot was 10,110 MiB; the final state was 855 MiB / 0% with no server or runner process. Smoke raw manifest/report/records/prompts SHA256 values are, respectively: `e14a98240c09a7013dc1e8a9f6b92b57f790c6f560b4585d481c2c05ba898694`, `d8cbc1a2b1a212c4c3ebdcb77a324e18ee2082768521d39c6c89cd081c985183`, `a1d772d94f73b33177885dd823d676346184f59724137aefa3d0775d3d542693`, and `ddb4868a2717c813a852713a7c455c35438d784ed6e334c70b2ae2b93af68164`.

## Five-path W8A8/W4A4 vector matrix

The full default-vector comparison used the audited d0724427b build and GGUFs, with W1A1, W8A8 and W4A4 Tensor Core selectors off. The candidate binary was SHA256 `bffe57951e5c4fafd47bba9bca970931ac008e7fe4b97418eac3b7c257213b69`; W8A8 and W4A4 GGUF hashes are listed above. The config SHA256 is `b69ae15c9b342736b5648847055acf708063bc421d7ed2f2ed6f31da4a14f9ee`; prompts are the same frozen 12 in the production suite. Settings were 2 warmups, five measured repetitions, 128-token cap, seed 42, temperature 0, same F16 target/drafter, F16 KV, context 2048, parallel 1, one server at a time.

Exact supervised launch and analysis commands:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py sm75-eagle-int-lowbit-vector-five-rep-d0724427b-20260925 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/benchmark_native_eagle.py --config runs/toolchain-bootstrap/native-lowbit-full.toml --run-id native-lowbit-vector-five-rep-d0724427b-20260925
python3 scripts/remote_job.py analyze-native-lowbit-vector-d0724427b-20260925 -- runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env python scripts/analyze_native_benchmark.py --run-dir results/native-lowbit-vector-five-rep-d0724427b-20260925 --samples 2000 --seed 42 --output results/native-lowbit-vector-five-rep-d0724427b-20260925/analysis.json
```

The runner completed 300/300 requests (60 per variant) at `2026-09-25 01:56:18 UTC`, exit 0. Its manifest records parent commit `b9596675225271634881f631146d1e76bfce1a74`, parent gitlink `34e21b7d85c17e25d5a91ce2ab1074d4c18bfe39`, and actual temporary llama.cpp checkout `d0724427b61f6ff4733b502d0ce3d800a2f4cd86` with `llama_checkout_matches_gitlink=false`. The parent submodule working tree was restored to 34e21b7 after the server/runner process group ended; post-run `git status` is clean. The maximum loaded-GPU snapshot was 10,110 MiB; the highest live sample was 10,160 MiB used / 868 MiB free. Final GPU memory returned to 855 MiB / 0% with no project process.

| Variant | Request tok/s | Decode tok/s | Accepted / proposed | Rounds | Accepted / round | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| Target only | 58.670 | 60.604 | — | — | — | — |
| Ordinary EAGLE | 76.007 | 80.994 | 3,970 / 16,740 | 3,400 | 1.168 | 23.72% |
| W1A1 portable head | 69.401 | 73.580 | 3,405 / 19,480 | 3,960 | 0.860 | 17.48% |
| W8A8 vector | 75.139 | 80.182 | 3,905 / 17,055 | 3,465 | 1.127 | 22.90% |
| W4A4 vector | 40.048 | 41.404 | 620 / 33,055 | 6,710 | 0.092 | 1.88% |

Pooled request/decode ratios against ordinary EAGLE were `0.989/0.990` for W8A8 and `0.527/0.511` for W4A4. Paired 95% bootstrap intervals (2,000 samples, seed 42, resampling prompts and repetitions) were:

| Variant vs ordinary EAGLE | Request ratio median [95% interval] | Decode ratio median [95% interval] |
|---|---:|---:|
| W8A8 vector | 0.988 [0.974, 1.003] | 0.990 [0.975, 1.005] |
| W4A4 vector | 0.526 [0.484, 0.575] | 0.510 [0.466, 0.561] |

Against target-only, W8A8 ratios were 1.281 request and 1.323 decode; W4A4 ratios were 0.683 and 0.683. Each comparison with target-only contains the same stable reasoning-02 markdown difference described above, so those target-only ratios are timing observations rather than strict lossless speedups. All four speculative outputs (ordinary, W1A1 head, W8A8, W4A4) matched one another on all 60 prompt/repetition pairs. Against target-only, each matched 55/60; the five differences were the known `reasoning-02` heading capitalization, while target-only itself was stable across repetitions. Generated token IDs were omitted in all 300 responses. All five dispatch confirmations were true for both W8A8 and W4A4. W8A8 ran its signed-INT8 DP4A-style source path; W4A4 ran the signed-nibble scalar integer MUL/ADD vector path, not INT4 Tensor Core MMA.

Per-prompt pooled request/decode ranges across the 12 prompts were 57.70–92.98 / 59.66–98.34 tok/s for W8A8 and 36.64–42.06 / 38.86–44.06 for W4A4. Category decode tok/s for prose/code/reasoning was 65.91/85.25/92.88 for W8A8 and 40.29/41.74/42.11 for W4A4; accepted-per-round by category was 0.749/1.263/1.461 and 0.063/0.100/0.113, respectively. Ordinary EAGLE category decode was 66.82/89.67/95.08 tok/s, accepted-per-round 0.749/1.358/1.510.

Server `common_speculative_impl` timing decomposition is host wall time, not GPU-kernel time. `accept_ms` is the acceptance-hook span, not total target-verification latency; the runner has no isolated verifier span. Measured-only totals and draft-call cost per round were:

| Variant | begin ms | draft ms | accept-hook ms | draft ms / verification round |
|---|---:|---:|---:|---:|
| Ordinary EAGLE | 0.096 | 22,499.275 | 4.281 | 6.617 |
| W8A8 vector | 0.094 | 21,929.817 | 4.364 | 6.329 |
| W4A4 vector | 0.094 | 44,696.353 | 8.287 | 6.661 |

Raw artifacts remain under `/home/philip/binary-eagle-decoding/results/native-lowbit-vector-five-rep-d0724427b-20260925/`:

| Artifact | SHA256 |
|---|---|
| `manifest.json` | `4dd772b7efd3bcc113528f51502e89fa763f362c4fb01b1bae7569091eb2051f` |
| `report.json` | `72eb46d17da9d67cdc5f0211d3184eb53653cb66cde231b6323f18156aec1f27` |
| `records.json` | `39bb90a4c105de4056a8104bd592cfd61524982aa6cf5f6c28d4392a0243eb12` |
| `prompts.jsonl` | `0d6a698d6816592c6ff435fed2fea4cdafe9f5248393d2a5ac091e1551919476` |
| `analysis.json` | `08302212b00d2f9e3ced34645f5d29a722c0a7feab5fa8ee6d881212b5035208` |

## Seven-path opt-in W8A8/W4A4 MMA matrix

The combined opt-in candidate at llama.cpp `2d9712cde8d7808bb869e59e56a565e0e4fa2918` passed its SM75 build, backend-op selectors, exact-dot probes, candidate-library SASS checks, and the short all-nine model parity smoke before timing. The actual live-library extracts contain `IMMA.8816.S8.S8` in `_Z15w8a8_signed_mma...` (extract SHA256 `2d9c549f1df31c39122891a70f36cee5bed076f98c8d63c9df10b55217c1dfe1`) and `IMMA.8832.S4.S4.SAT` in `_Z17w4a4_sm75_mma_dot...` (extract SHA256 `0b83964bb3d96ee47df168b3fbd35c75fe8967bee3db0ff17d4038a3c9999782`). These are the candidate-library SASS excerpts, separate from the standalone probes. Both MMA paths were opt-in; the paired default rows kept their selectors disabled.

The full run used the exact candidate server binary SHA256 `07bb2339b000ba63f71cb4c5c7b74304a0816e466b5796d40f7e09712ff33b5a`, the audited d0724427b W8A8 and W4A4 GGUFs above, the same F16 target and ordinary draft, 12 frozen prompts, two warmups per server, five repetitions, 128-token cap, seed 42, temperature 0, F16 KV, context 2048, and one server at a time. The ignored config `runs/toolchain-bootstrap/native-operand-mma-full.toml` has SHA256 `9084a6e0059da4ac478a03c5096680f34102b712a67749f0b4a27bed4140f6e5`. It enabled both default and MMA rows for W8A8 and W4A4 using the same audited GGUF for each pair, required the exact distinct operator marker, and forbade the paired path's marker.

Supervised run `native-lowbit-mma-full-supervisor-timed-2d9712cde-20260925` launched `scripts/benchmark_native_eagle.py` with result ID `native-lowbit-mma-full-timed-2d9712cde-20260925`. It completed 420/420 measured requests (60 for each of seven variants) with supervisor exit 0. The manifest records parent commit `9fe5321c350bdea554935f2f28603c4e72e118da`, parent gitlink `34e21b7d85c17e25d5a91ce2ab1074d4c18bfe39`, and actual temporary llama.cpp checkout `2d9712cde8d7808bb869e59e56a565e0e4fa2918`. The initial dry-run correctly generated its manifest but occupied its requested result directory; the first timed invocation stopped before starting a server with `FileExistsError`. Its raw output was preserved, and the completed timed run used a fresh result ID. No timed records were discarded or repeated.

| Variant | Request tok/s | Decode tok/s | Accepted / proposed | Rounds | Accepted / round | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| Target only | 58.814 | 60.739 | — | — | — | — |
| Ordinary EAGLE | 76.399 | 81.435 | 3,970 / 16,740 | 3,400 | 1.168 | 23.72% |
| W1A1 portable head | 69.430 | 73.620 | 3,405 / 19,480 | 3,960 | 0.860 | 17.48% |
| W8A8 DP4A | 75.279 | 80.301 | 3,905 / 17,055 | 3,465 | 1.127 | 22.90% |
| W8A8 SM75 MMA | 62.496 | 65.889 | 3,905 / 17,055 | 3,465 | 1.127 | 22.90% |
| W4A4 signed-nibble vector | 40.114 | 41.502 | 620 / 33,055 | 6,710 | 0.092 | 1.88% |
| W4A4 signed-I4 SM75 MMA | 35.478 | 36.526 | 620 / 33,055 | 6,710 | 0.092 | 1.88% |

The opt-in Tensor Core kernels had a measured cost loss with unchanged acceptance. W8A8 MMA was 17.0% slower in pooled request rate and 17.9% slower in decode rate than its DP4A default; W4A4 MMA was 11.6% slower in request rate and 12.0% slower in decode rate than its signed-nibble vector default. Paired prompt/repetition bootstrap intervals (2,000 draws, seed 42) exclude parity:

| MMA vs same-format default | Request ratio median [95% interval] | Decode ratio median [95% interval] |
|---|---:|---:|
| W8A8 MMA / DP4A | 0.830 [0.826, 0.836] | 0.821 [0.817, 0.824] |
| W4A4 MMA / vector | 0.884 [0.880, 0.889] | 0.880 [0.876, 0.884] |

Against ordinary EAGLE, the corresponding MMA request/decode ratios were 0.818 [0.807, 0.828] / 0.809 [0.797, 0.821] for W8A8 and 0.464 [0.426, 0.506] / 0.448 [0.409, 0.493] for W4A4. The candidate rows and their default rows produced identical completion-text hashes and identical acceptance/proposal/round counts on all 60 pairs for each format. Each speculative variant matched target-only on 55/60 requests; all six speculative variants shared the same five stable `reasoning-02` heading-capitalization differences. Token IDs were omitted by the server for all requests.

Every W8A8 and W4A4 row had the intended selector tuple: W8A8 default `(W8=0,W4=0,W1=0)`, W8A8 MMA `(1,0,0)`, W4A4 default `(0,0,0)`, W4A4 MMA `(0,1,0)`. Harness dispatch evidence confirmed all-nine loader markers and each exact selected CUDA marker across repetitions, with the forbidden opposite-path marker absent. Both MMA candidates therefore executed the distinct Tensor Core paths whose SASS was inspected. Host draft-call timings show the cost increase: W8A8 default 22,009 ms total / 6.352 ms per verification round versus MMA 41,767 ms / 12.054 ms; W4A4 vector 44,543 ms / 6.638 ms versus MMA 68,368 ms / 10.189 ms. These are host wall times inside draft calls, not isolated GPU-kernel durations. The logged `accept_ms` is only the acceptance-hook span, not total target verification latency; the runner has no isolated verifier span.

The highest live GPU sample during timing was 10,160 MiB used / 868 MiB free. After the supervisor exited, the RTX 2080 Ti returned to 855 MiB used / 10,173 MiB free at 0% utilization, with no server process. The temporary submodule checkout was restored from `2d9712cde` to the committed production gitlink `34e21b7`; post-restore parent status was clean and the gitlink stayed unchanged.

Exact supervised launch and analysis commands:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py native-lowbit-mma-full-supervisor-timed-2d9712cde-20260925 -- python3 scripts/benchmark_native_eagle.py --config runs/toolchain-bootstrap/native-operand-mma-full.toml --run-id native-lowbit-mma-full-timed-2d9712cde-20260925
python3 scripts/analyze_native_benchmark.py --run-dir results/native-lowbit-mma-full-timed-2d9712cde-20260925 --samples 2000 --seed 42 --output results/native-lowbit-mma-full-timed-2d9712cde-20260925/analysis.json
```

Raw artifacts remain under `/home/philip/binary-eagle-decoding/results/native-lowbit-mma-full-timed-2d9712cde-20260925/`:

| Artifact | SHA256 |
|---|---|
| `manifest.json` | `c1accfcc98eafb63230470beea84d149fee9bce8ed1708850de43ca4eb224779` |
| `report.json` | `e51ce493b519aa5e6e6c37c23eb86aad836f98491015aa2b0ec97fd2b42a825d` |
| `records.json` | `c254f858db9273054f71a24d3198dbcf0454f3da3792f4e1cb735df9198152e9` |
| `analysis.json` | `92e77f518c32bfa575b5946306da1690e775d8c28beb1dd21fad8232c12eb632` |

## Q4_0/Q8_0 CUDA dispatch trace

To resolve the remaining weight-only execution labels, an isolated source worktree at llama.cpp commit `5c52067b71b5dc43854054506f555a6babe09150` was built against the production base `34e21b7d85c17e25d5a91ce2ab1074d4c18bfe39`. The trace change is opt-in and the production checkout/binary remained unchanged. The supervised build `sm75-qformat-trace-build-5c52067-20260925` compiled 356/356 Ninja targets for SM75 using CUDA 12.8.93 and GCC 13.4.0. Build stdout SHA256 is `cd78daf2ac677946776e934e453167e5b69ea0d169e499fa36eaea9c6b9b6c63`; the isolated `llama-server` binary SHA256 is `0fd2d394a5136526c0e5fce96e0d704a27d53a3445d0d0135386034c02425199`.

The trace-only smoke used `GGML_CUDA_QFORMAT_DISPATCH_TRACE=1`, the same F16 target, ordinary draft, Q4_0 and Q8_0 drafts, F16 KV, context 2048, one `prose-01` prompt, one warmup, five measured repetitions, and a 32-token cap. The config is `runs/toolchain-bootstrap/qformat-trace-smoke.toml`, SHA256 `d8d5cd837a38388ea50794531aa61ae1143b6fdf0261a6bb29a71d15e4538fe6`. Supervised inference ID `native-qformat-dispatch-trace-supervisor-5c52067-20260925` completed with exit 0 at 2026-09-25 03:38:25 UTC. It produced 25/25 measured records (five each target, ordinary EAGLE, W1A1 head, Q4_0 and Q8_0), no failed requests, and stable completion text within each variant. This is a path diagnostic, not a timed quantization comparison.

For both stored weight types, the trace selected MMVQ and MMQ, with Q8_1 as the internally quantized activation kernel type. The trace reports `src1=f32` because the activation tensor enters as F32; `activation_quantized=yes` records the actual conversion used by these operators. The logged shapes were:

| Stored weight | CUDA family | K | M | N | Batch | Activation kernel | Other flags |
|---|---|---:|---:|---:|---|---|---|
| Q4_0 | MMVQ | 5120 | 4096 | 2 | 1×1 | Q8_1, quantized | no matmul IDs, not fused |
| Q4_0 | MMQ | 7680 | 2560 | 38 | 1×1 | Q8_1, quantized | no matmul IDs, not fused |
| Q4_0 | MMVQ | 4096 | 2560 | 1 | 1×1 | Q8_1, quantized | no matmul IDs, fused |
| Q8_0 | MMVQ | 5120 | 4096 | 2 | 1×1 | Q8_1, quantized | no matmul IDs, not fused |
| Q8_0 | MMQ | 7680 | 2560 | 38 | 1×1 | Q8_1, quantized | no matmul IDs, not fused |
| Q8_0 | MMVQ | 4096 | 2560 | 1 | 1×1 | Q8_1, quantized | no matmul IDs, fused |

Each format's trace logger emitted these three distinct records in every repetition (15 trace lines per format across five server processes). This confirms the N=38 MMQ prefill path and the N=1 fused MMVQ decode path; the N=2 nonfused MMVQ call was also observed. No cuBLAS Q4_0/Q8_0 dispatch was selected in this smoke. The trace identifies the GGML kernel family and activation conversion; instruction-level DP4A remains a source-backed property of the selected MMVQ implementation rather than an instruction trace from this run.

The highest sampled live GPU use during the smoke was 9,090 MiB; after supervision ended, the GPU returned to 855 MiB used / 10,173 MiB free at 0%, with no `llama-server` process. The production submodule stayed at gitlink `34e21b7`, and the parent working tree remained clean.

The exact build and inference commands were:

```sh
cd ~/binary-eagle-decoding
python3 scripts/remote_job.py sm75-qformat-trace-build-5c52067-20260925 -- bash -c 'set -eu; env CC=$PWD/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-gcc CXX=$PWD/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=$PWD/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env cmake -S runs/llama-sm75-qformat-trace-src -B build/llama-cuda-qformat-trace -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF -DGGML_CUDA=ON -DGGML_METAL=OFF -DCMAKE_CUDA_ARCHITECTURES=75; env CC=$PWD/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-gcc CXX=$PWD/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ CUDAHOSTCXX=$PWD/runs/toolchain-bootstrap/env/bin/x86_64-conda-linux-gnu-g++ runs/toolchain-bootstrap/bin/micromamba run -p runs/toolchain-bootstrap/env cmake --build build/llama-cuda-qformat-trace --parallel 4 --target llama-server'
python3 scripts/remote_job.py native-qformat-dispatch-trace-supervisor-5c52067-20260925 -- python3 scripts/benchmark_native_eagle.py --config runs/toolchain-bootstrap/qformat-trace-smoke.toml --run-id native-qformat-dispatch-trace-smoke-5c52067-20260925
```

Raw smoke artifacts remain at `/home/philip/binary-eagle-decoding/results/native-qformat-dispatch-trace-smoke-5c52067-20260925/`:

| Artifact | SHA256 |
|---|---|
| `manifest.json` | `28c40fcf79cb570b349bab254e0b07f56db0672b513202a514fbac98952fc3d7` |
| `report.json` | `56d28fe2323d5f14747911c1c8a59aa7a84faaa5c0e976129a3ec7622698c411` |
| `records.json` | `ca9566b44c74fa9cece3ca84122a0029a02cad88f8a691b46aa64c5bbb887f90` |
| `prompts.jsonl` | `ddb4868a2717c813a852713a7c455c35438d784ed6e334c70b2ae2b93af68164` |

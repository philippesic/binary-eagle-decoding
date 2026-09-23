# Repository setup

## Bootstrap

This is a standalone Git repository. Its initial checkout includes llama.cpp as
a shallow submodule. Future clones should use `git clone --recurse-submodules`.
For an existing checkout:

```sh
make setup
make check
make doctor
```

Install Git, uv, and a C/C++ toolchain first. On macOS, the Xcode command-line
tools provide the compiler; on Linux, use a supported GCC/Clang toolchain.
`make setup` restores the recorded submodule and installs the locked CMake,
Ninja, and Ruff tools in `.venv`. `.python-version` selects Python 3.11.
It does not install PyTorch, training frameworks, or download model weights.

## Runtime builds

```sh
make build-cpu       # portable development / CLI smoke check
make build-metal    # macOS development
make build-cuda     # CUDA toolkit on PATH; CMAKE_CUDA_ARCHITECTURES=75
```

Each build has its own directory under `build/llama-<backend>`. All build
`llama-server`, `llama-cli`, and `llama-bench` in its `bin/` directory. Builds use
Release mode, with tests and the embedded web UI disabled. HTTPS support is
disabled to avoid an OpenSSL dependency; use local model files.

To configure without compiling or limit build parallelism:

```sh
uv run --locked python scripts/build_llama.py cpu --configure-only
uv run --locked python scripts/build_llama.py cpu --jobs 4
```

For the SM75 binary experiment, use Linux, an RTX 2080 Ti, a matching NVIDIA
driver, and a CUDA toolkit/host compiler combination that supports SM75. Record
the exact toolkit rather than assuming a version from the workstation. The
scaffold explicitly targets architecture 75; CUDA compilation and binary
instruction availability still require validation on that host. Neither an
Apple GPU build nor a CPU build validates the Turing kernel hypothesis.

Capture the machine and revisions with:

```sh
uv run --locked python scripts/environment.py > results/environment.json
```

An uncommitted project has no HEAD yet; the manifest records the failed revision
query and working-tree state explicitly. A manifest alone does not preserve
uncommitted code: commit or archive the exact changes before reporting results.

## First model run (next milestone)

1. Resolve immutable Hugging Face revisions for the target and draft, and record
   them in an experiment-specific copy of `configs/baseline.toml`.
2. Download those snapshots into `models/hf/Qwen3-4B` and
   `models/hf/Qwen3-4B_eagle3`; record file hashes and conversion commands.
3. Set up a separate conversion environment using the pinned upstream
   `requirements/requirements-convert_hf_to_gguf.txt`. Select and lock the
   ML/CUDA dependencies on the experimental host before training work begins.
4. From the repo root, with that conversion environment active:

```sh
mkdir -p models/gguf
python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B \
  --outtype f16 --outfile models/gguf/Qwen3-4B-f16.gguf
python third_party/llama.cpp/convert_hf_to_gguf.py models/hf/Qwen3-4B_eagle3 \
  --target-model-dir models/hf/Qwen3-4B \
  --outtype f16 --outfile models/gguf/Qwen3-4B-eagle3-f16.gguf
```

The pinned upstream [conversion and EAGLE instructions](../third_party/llama.cpp/docs/speculative.md)
describe the model pairing and target metadata requirement. FP16 is the proposed
baseline; do not treat a BF16 checkpoint file as proof of native BF16 execution
on Turing.

Run these servers separately, using the same prompts and request parameters:

```sh
# Target only
./build/llama-cuda/bin/llama-server \
  -m models/gguf/Qwen3-4B-f16.gguf --spec-type none \
  --n-gpu-layers all --ctx-size 2048 --parallel 1 --fit off \
  --host 127.0.0.1 --port 8080

# Normal EAGLE-3
./build/llama-cuda/bin/llama-server \
  -m models/gguf/Qwen3-4B-f16.gguf \
  -md models/gguf/Qwen3-4B-eagle3-f16.gguf --spec-type draft-eagle3 \
  --spec-draft-n-max 5 --n-gpu-layers all --spec-draft-ngl all \
  --ctx-size 2048 --parallel 1 --fit off \
  --host 127.0.0.1 --port 8080
```

These are starting commands, not validated benchmark results. Check actual
memory placement, KV cache settings, chat template/thinking mode, and successful
generation before freezing a baseline. FP16 target weights plus draft, KV cache,
and workspaces may strain available memory; measure this before increasing
context. Preserve the same settings across comparisons. See
[evaluation](EVALUATION.md) for the full reporting protocol.

`llama-bench` is useful for target-only diagnostics, but the speculative
comparison must exercise the actual draft + verifier path.

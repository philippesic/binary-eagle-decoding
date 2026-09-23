# W1A1 EAGLE speculative decoding

A bounded systems/ML experiment: can a genuinely binary EAGLE drafter improve
end-to-end LLM throughput enough to offset lower speculative-token acceptance?

Start with the [project overview](docs/PROJECT_OVERVIEW.md), which defines the
goal, scope, milestones, and decision criteria. See [setup](docs/SETUP.md) for
build instructions and [evaluation](docs/EVALUATION.md) for measurement rules.
For Codex desktop teamwork, see [agent operations](docs/AGENT_OPERATIONS.md).
Invoke `$start-goal-team` with a rough objective to open the next official goal.

## Quick start

Requires Git, Python 3.11+, [uv](https://docs.astral.sh/uv/), and a C/C++ compiler.
Run from this repository's root:

```sh
make setup
make check
make doctor
make build-cpu
./build/llama-cpu/bin/llama-server --version
```

`make build-metal` is available for Apple development machines.
`make build-cuda` targets SM75 on a Linux machine with NVIDIA's CUDA toolkit.
CPU and Metal runs are development checks, not evidence for the GPU hypothesis.

## Layout

```text
configs/                 Versioned experiment plans (not yet a runner interface)
data/                    Local prompts, calibration data, and hidden states
docs/                    Project guidance, setup, and evaluation protocol
experiments/             Experiment notes and small, reviewable reports
kernels/                 Future packed binary CUDA prototypes
models/                  Local checkpoints and converted GGUF files
results/                 Local raw runs, logs, and environment manifests
scripts/                 Bootstrap checks, builds, and environment capture
src/w1a1_eagle/           Future simulation, QAT, export, and analysis code
third_party/llama.cpp/   Pinned upstream Git submodule
```

This initial scaffold includes the upstream runtime and build tooling. W1A1
simulation, training, kernels, model downloads, and benchmark execution are
future milestones; no speedup or model compatibility has been measured yet.

Development tools are pinned in `uv.lock`. The llama.cpp revision is pinned by
the submodule gitlink; see [third-party notes](third_party/README.md). Large
artifacts stay out of Git. Do not use `git submodule update --remote` for routine
setup, because it changes the experimental runtime version.

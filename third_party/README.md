# Upstream runtime

`llama.cpp/` is a Git submodule fetched from the user's fork,
https://github.com/philippesic/llama.cpp. The original project is
https://github.com/ggml-org/llama.cpp.
Initial pin: `6e60f35608ec6918b44a9839c0c433687165f086`.

Restore with `git submodule update --init --recursive`. This uses the parent
repository's recorded commit. The initial clone is shallow; fetch additional
history explicitly if needed. The upstream LICENSE remains inside the submodule.

Useful integration points at the initial pin:

- `conversion/llama.py`: EAGLE checkpoint conversion.
- `src/models/eagle3.cpp`: drafter graph construction.
- `src/models/qwen3.cpp`: target graph.
- `common/speculative.cpp`: EAGLE drafting / verification coordination.
- `ggml/src/ggml-cuda/`: CUDA backend and matrix operation dispatch.
- `docs/speculative.md`: runtime usage.

For runtime changes, create a branch inside the submodule. Push its commits to
the user's fork before updating the parent gitlink. A gitlink to a local-only
commit is not sufficient for another machine to clone the work.
Do not advance upstream during experiments without recording the change and
re-running the baseline. Keep the planning config's revision consistent too.

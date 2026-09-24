# Combined INT draft shared-library link fix

**2026-09-24:** The combined llama.cpp fork branch `feat/eagle-int-cuda-integration` now includes commit `d0724427b`, a clean cherry-pick of the published `34e21b7` fix. It explicitly instantiates the out-of-line `llama_model_loader::get_arr<std::string>(const std::string &, std::vector<std::string> &, bool)` overload in `src/llama-model-loader.cpp`. This is the only change after combined CUDA source commit `b881a6346`; the parent gitlink remains pinned.

Local Apple M3 Max checks: `git diff HEAD^ HEAD --check` passed; a CPU CMake rebuild linked `libllama.dylib`; `nm -gU build/int-cuda-cpu/bin/libllama.dylib | c++filt` displayed the exact overload as a global text symbol (`T`). Both `test-w8a8-eagle-load --accept` and `test-w4a4-eagle-load --accept` loaded their real exported GGUFs with all nine draft linears after the rebuild. Ignored logs are under `results/int-integration/`.

This verifies the symbol on the local shared build. The GPU operator still needs to rebuild the exact combined commit on WSL and confirm the production `libllama.so` exports/resolves it before SM75 precision and benchmark gates resume. No GPU work or parent gitlink update occurred in this stage.

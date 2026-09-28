// Replay pinned EAGLE key RoPE through the ggml CPU backend.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<char> read_exact(const std::string & path, size_t size) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || static_cast<size_t>(file.tellg()) != size) {
        throw std::runtime_error("missing or wrong-size operand: " + path);
    }
    std::vector<char> bytes(size);
    file.seekg(0);
    file.read(bytes.data(), size);
    if (!file) throw std::runtime_error("cannot read operand: " + path);
    return bytes;
}

int main(int argc, char ** argv) {
    if (argc != 3) return 2; // operand directory, token count
    try {
        const std::string dir = argv[1];
        const int64_t tokens = std::stoll(argv[2]);
        if (tokens < 1 || tokens > 256) return 2;
        constexpr int64_t head_width = 128, heads = 8;
        auto raw = read_exact(dir + "/key_raw.f32", tokens * heads * head_width * sizeof(float));
        auto positions = read_exact(dir + "/positions.i32", tokens * sizeof(int32_t));

        ggml_init_params params = {4 * 1024 * 1024, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml_init failed");
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(backend, 1);

        ggml_tensor * key = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_width, heads, tokens);
        ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
        ggml_tensor * rotated = ggml_rope_ext(
            ctx, key, pos, nullptr, head_width, GGML_ROPE_TYPE_NORMAL,
            40960, 1000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        );
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
        ggml_build_forward_expand(graph, rotated);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("ggml backend allocation failed");
        ggml_backend_tensor_set(key, raw.data(), 0, raw.size());
        ggml_backend_tensor_set(pos, positions.data(), 0, positions.size());
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CPU RoPE graph compute failed");
        }
        std::vector<char> output(ggml_nbytes(rotated));
        ggml_backend_tensor_get(rotated, output.data(), 0, output.size());
        std::ofstream file(dir + "/key_rope.f32", std::ios::binary);
        file.write(output.data(), output.size());
        if (!file) throw std::runtime_error("cannot write RoPE result");
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}

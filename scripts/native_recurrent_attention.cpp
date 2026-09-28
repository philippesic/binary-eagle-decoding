// Replay a captured EAGLE decoder's ggml CPU Flash Attention operation.
// Inputs are physical, post-write cache slots and the original F16 mask.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
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
    std::vector<char> data(size);
    file.seekg(0);
    file.read(data.data(), size);
    if (!file) throw std::runtime_error("cannot read operand: " + path);
    return data;
}

int main(int argc, char ** argv) {
    if (argc != 5) return 2; // operand directory, query tokens, K/V slots, threads
    try {
        const std::string dir = argv[1];
        const int64_t tokens = std::stoll(argv[2]);
        const int64_t slots = std::stoll(argv[3]);
        const int threads = std::stoi(argv[4]);
        if (tokens < 1 || tokens > 512 || slots < 1 || slots > 2048 || threads < 1 || threads > 128) {
            return 2;
        }
        constexpr int64_t head_width = 128, q_heads = 32, kv_heads = 8;
        auto q_data = read_exact(dir + "/query.f32", tokens * q_heads * head_width * sizeof(float));
        auto k_data = read_exact(dir + "/keys.f16", slots * kv_heads * head_width * sizeof(ggml_fp16_t));
        auto v_data = read_exact(dir + "/values.f16", slots * kv_heads * head_width * sizeof(ggml_fp16_t));
        auto mask_data = read_exact(dir + "/mask.f16", tokens * slots * sizeof(ggml_fp16_t));

        ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml_init failed");
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(backend, threads);

        // Layout matches eagle3.cpp: Q/K/V are [head_width, heads, tokens or slots],
        // then build_attn_mha permutes axes 1 and 2 before Flash Attention.
        ggml_tensor * q_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_width, q_heads, tokens);
        ggml_tensor * k_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_width, kv_heads, slots);
        ggml_tensor * v_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_width, kv_heads, slots);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, slots, tokens, 1, 1);
        ggml_tensor * q = ggml_permute(ctx, q_raw, 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(ctx, k_raw, 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(ctx, v_raw, 0, 2, 1, 3);
        const float scale = 1.0f / sqrtf(float(head_width));
        ggml_tensor * attention = ggml_flash_attn_ext(ctx, q, k, v, mask, scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_n_kv_max(attention, 0);
        ggml_prec_set_acc(attention, GGML_PREC_F32);
        ggml_tensor * output = ggml_reshape_2d(ctx, attention, head_width * q_heads, tokens);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("ggml backend allocation failed");
        ggml_backend_tensor_set(q_raw, q_data.data(), 0, q_data.size());
        ggml_backend_tensor_set(k_raw, k_data.data(), 0, k_data.size());
        ggml_backend_tensor_set(v_raw, v_data.data(), 0, v_data.size());
        ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size());
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CPU Flash Attention graph compute failed");
        }
        std::vector<char> result(ggml_nbytes(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size());
        std::ofstream file(dir + "/attention.f32", std::ios::binary);
        file.write(result.data(), result.size());
        if (!file) throw std::runtime_error("cannot write attention result");
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}

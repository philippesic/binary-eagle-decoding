// Identical-Q/K/V Qwen3 block-0 Flash Attention, output projection and residual.
#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t TOKENS = 29;
constexpr int64_t KV_SLOTS = 256;
constexpr int64_t HEAD = 128;
constexpr int64_t Q_HEADS = 32;
constexpr int64_t KV_HEADS = 8;
constexpr int64_t HIDDEN = 2560;
constexpr int64_t Q_WIDTH = HEAD * Q_HEADS;
constexpr int64_t KV_WIDTH = HEAD * KV_HEADS;

std::vector<char> read_exact(const std::string & path, size_t size) {
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
} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) return 2; // operand directory, qf32 or qf16cast
    try {
        const std::string dir = argv[1];
        const std::string mode = argv[2];
        if (mode != "qf32" && mode != "qf16cast") return 2;
        auto q_bytes = read_exact(dir + "/q_rope.f32", TOKENS * Q_WIDTH * sizeof(float));
        auto k_bytes = read_exact(dir + "/k_cache.f16", KV_SLOTS * KV_WIDTH * sizeof(ggml_fp16_t));
        auto v_bytes = read_exact(dir + "/v_cache.f16", KV_SLOTS * KV_WIDTH * sizeof(ggml_fp16_t));
        auto mask_bytes = read_exact(dir + "/causal_mask.f16", TOKENS * KV_SLOTS * sizeof(ggml_fp16_t));
        auto weight_bytes = read_exact(dir + "/o_weight.f16", HIDDEN * Q_WIDTH * sizeof(ggml_fp16_t));
        auto residual_bytes = read_exact(dir + "/layer_input.f32", TOKENS * HIDDEN * sizeof(float));

        ggml_backend_load_all();
        ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
        if (!backend) throw std::runtime_error("ggml CUDA backend unavailable");
        const std::string device = ggml_backend_dev_description(ggml_backend_get_device(backend));
        if (device.find("RTX 5080") == std::string::npos) {
            throw std::runtime_error("GPU backend is not the registered RTX 5080");
        }
        ggml_init_params params = {8 * 1024 * 1024, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml_init failed");

        ggml_tensor * q_raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD, Q_HEADS, TOKENS, 1);
        ggml_tensor * k_cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HEAD, KV_HEADS, KV_SLOTS, 1);
        ggml_tensor * v_cache = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, HEAD, KV_HEADS, KV_SLOTS, 1);
        ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, KV_SLOTS, TOKENS, 1, 1);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, Q_WIDTH, HIDDEN);
        ggml_tensor * residual = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, TOKENS);
        ggml_tensor * q_operand = mode == "qf16cast" ? ggml_cast(ctx, q_raw, GGML_TYPE_F16) : q_raw;
        ggml_tensor * q = ggml_permute(ctx, q_operand, 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(ctx, k_cache, 0, 2, 1, 3);
        ggml_tensor * v = ggml_permute(ctx, v_cache, 0, 2, 1, 3);
        ggml_tensor * attn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f / sqrtf(float(HEAD)), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_n_kv_max(attn, 0);
        ggml_prec_set_acc(attn, GGML_PREC_F32);
        ggml_tensor * heads = ggml_reshape_2d(ctx, attn, Q_WIDTH, TOKENS);
        ggml_tensor * projected = ggml_mul_mat(ctx, weight, heads);
        ggml_tensor * output = ggml_add(ctx, projected, residual);
        if (output->type != GGML_TYPE_F32 || output->ne[0] != HIDDEN || output->ne[1] != TOKENS) {
            throw std::runtime_error("ggml attention residual returned wrong geometry");
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("ggml CUDA tensor allocation failed");
        ggml_backend_tensor_set(q_raw, q_bytes.data(), 0, q_bytes.size());
        ggml_backend_tensor_set(k_cache, k_bytes.data(), 0, k_bytes.size());
        ggml_backend_tensor_set(v_cache, v_bytes.data(), 0, v_bytes.size());
        ggml_backend_tensor_set(mask, mask_bytes.data(), 0, mask_bytes.size());
        ggml_backend_tensor_set(weight, weight_bytes.data(), 0, weight_bytes.size());
        ggml_backend_tensor_set(residual, residual_bytes.data(), 0, residual_bytes.size());
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml CUDA attention graph failed");
        }
        std::vector<char> result(ggml_nbytes(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size());
        std::ofstream file(dir + "/ggml_attn_" + mode + "_residual.f32", std::ios::binary);
        file.write(result.data(), result.size());
        if (!file) throw std::runtime_error("cannot write ggml attention residual");
        printf("%s: %zu bytes on %s\n", mode.c_str(), result.size(), device.c_str());
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}

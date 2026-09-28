// Identical-input Qwen3 block-0 Q projection, per-head norm and RoPE on ggml CUDA.
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t TOKENS = 29;
constexpr int64_t HIDDEN = 2560;
constexpr int64_t Q_WIDTH = 4096;
constexpr int64_t HEAD_WIDTH = 128;
constexpr int64_t HEADS = Q_WIDTH / HEAD_WIDTH;

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
    if (argc != 3) return 2; // operand directory, f32 or f16cast
    try {
        const std::string dir = argv[1];
        const std::string mode = argv[2];
        if (mode != "f32" && mode != "f16cast") return 2;
        auto input_bytes = read_exact(dir + "/native_norm.f32", TOKENS * HIDDEN * sizeof(float));
        auto weight_bytes = read_exact(dir + "/q_weight.f16", HIDDEN * Q_WIDTH * sizeof(ggml_fp16_t));
        auto norm_bytes = read_exact(dir + "/q_norm_weight.f32", HEAD_WIDTH * sizeof(float));
        auto position_bytes = read_exact(dir + "/positions.i32", TOKENS * sizeof(int32_t));

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
        ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HIDDEN, TOKENS);
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, HIDDEN, Q_WIDTH);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HEAD_WIDTH);
        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, TOKENS);
        ggml_tensor * operand = mode == "f16cast" ? ggml_cast(ctx, input, GGML_TYPE_F16) : input;
        ggml_tensor * projected = ggml_mul_mat(ctx, weight, operand);
        if (projected->type != GGML_TYPE_F32 || projected->ne[0] != Q_WIDTH || projected->ne[1] != TOKENS) {
            throw std::runtime_error("ggml Q projection returned wrong geometry");
        }
        ggml_tensor * heads = ggml_reshape_3d(ctx, projected, HEAD_WIDTH, HEADS, TOKENS);
        ggml_tensor * normed = ggml_mul(ctx, ggml_rms_norm(ctx, heads, 1e-6f), norm_weight);
        ggml_tensor * output = ggml_rope_ext(
            ctx, normed, positions, nullptr, HEAD_WIDTH, GGML_ROPE_TYPE_NEOX,
            40960, 1000000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        );
        ggml_set_output(projected);
        ggml_set_output(normed);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 48, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("ggml CUDA tensor allocation failed");
        ggml_backend_tensor_set(input, input_bytes.data(), 0, input_bytes.size());
        ggml_backend_tensor_set(weight, weight_bytes.data(), 0, weight_bytes.size());
        ggml_backend_tensor_set(norm_weight, norm_bytes.data(), 0, norm_bytes.size());
        ggml_backend_tensor_set(positions, position_bytes.data(), 0, position_bytes.size());
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml CUDA Q projection/norm/RoPE failed");
        }
        for (auto [tensor, label] : {std::pair<ggml_tensor *, std::string>{projected, "raw"},
                                     {normed, "normed"}, {output, "rope"}}) {
            std::vector<char> result(ggml_nbytes(tensor));
            ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
            std::ofstream file(dir + "/ggml_q_" + mode + "_" + label + ".f32", std::ios::binary);
            file.write(result.data(), result.size());
            if (!file) throw std::runtime_error("cannot write ggml Q output");
            printf("%s %s: %zu bytes on %s\n", mode.c_str(), label.c_str(), result.size(), device.c_str());
        }
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}

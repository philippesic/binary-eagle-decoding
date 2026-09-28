// Identical-input Qwen3 block-0 V projection on the pinned ggml CUDA backend.
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t TOKENS = 29;
constexpr int64_t HIDDEN = 2560;
constexpr int64_t V_WIDTH = 1024;

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
        auto weight_bytes = read_exact(dir + "/v_weight.f16", HIDDEN * V_WIDTH * sizeof(ggml_fp16_t));

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
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, HIDDEN, V_WIDTH);
        ggml_tensor * operand = mode == "f16cast" ? ggml_cast(ctx, input, GGML_TYPE_F16) : input;
        ggml_tensor * output = ggml_mul_mat(ctx, weight, operand);
        if (output->type != GGML_TYPE_F32 || output->ne[0] != V_WIDTH || output->ne[1] != TOKENS) {
            throw std::runtime_error("ggml V projection returned wrong geometry");
        }
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("ggml CUDA tensor allocation failed");
        ggml_backend_tensor_set(input, input_bytes.data(), 0, input_bytes.size());
        ggml_backend_tensor_set(weight, weight_bytes.data(), 0, weight_bytes.size());
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml CUDA V projection failed");
        }
        std::vector<char> result(ggml_nbytes(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size());
        const std::string output_name = mode == "f32" ? "ggml_v_f32.f32" : "ggml_v_f16cast.f32";
        std::ofstream file(dir + "/" + output_name, std::ios::binary);
        file.write(result.data(), result.size());
        if (!file) throw std::runtime_error("cannot write ggml V projection output");
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

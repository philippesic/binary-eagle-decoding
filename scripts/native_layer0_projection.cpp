// Bounded CPU ggml graph for the target's pre-RoPE layer-0 norm and Q/K/V.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
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

static void write_tensor(const std::string & path, const ggml_tensor * tensor) {
    std::vector<char> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());
    std::ofstream file(path, std::ios::binary);
    file.write(data.data(), data.size());
    if (!file) throw std::runtime_error("cannot write result: " + path);
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        return 2; // operand directory, token count, hidden, q rows, k rows, v rows
    }
    try {
        const std::string dir = argv[1];
        const int64_t tokens = std::stoll(argv[2]);
        const int64_t hidden = std::stoll(argv[3]);
        const int64_t rows[3] = {std::stoll(argv[4]), std::stoll(argv[5]), std::stoll(argv[6])};
        if (tokens < 1 || tokens > 128 || hidden < 1 || hidden > 8192) return 2;
        for (auto row : rows) if (row < 1 || row > 16384) return 2;

        auto x_data = read_exact(dir + "/embedding.f32", hidden * tokens * sizeof(float));
        auto norm_data = read_exact(dir + "/attn_norm.f32", hidden * sizeof(float));
        std::vector<char> weight_data[3];
        const char * labels[3] = {"q", "k", "v"};
        for (int i = 0; i < 3; ++i) {
            weight_data[i] = read_exact(dir + "/" + labels[i] + ".f16",
                                        hidden * rows[i] * sizeof(ggml_fp16_t));
        }

        ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
        ggml_context * ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml_init failed");
        ggml_backend_t backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(backend, 1);

        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
        ggml_tensor * norm_weight = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden);
        ggml_tensor * weights[3];
        for (int i = 0; i < 3; ++i) {
            weights[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, hidden, rows[i]);
        }
        ggml_tensor * normalized = ggml_mul(ctx, ggml_rms_norm(ctx, x, 1e-6f), norm_weight);
        ggml_tensor * projections[3];
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, normalized);
        for (int i = 0; i < 3; ++i) {
            projections[i] = ggml_mul_mat(ctx, weights[i], normalized);
            ggml_build_forward_expand(graph, projections[i]);
        }
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buffer) throw std::runtime_error("ggml backend allocation failed");
        ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size());
        ggml_backend_tensor_set(norm_weight, norm_data.data(), 0, norm_data.size());
        for (int i = 0; i < 3; ++i) {
            ggml_backend_tensor_set(weights[i], weight_data[i].data(), 0, weight_data[i].size());
        }
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("CPU graph compute failed");
        }
        write_tensor(dir + "/native_norm.f32", normalized);
        for (int i = 0; i < 3; ++i) {
            write_tensor(dir + "/native_" + std::string(labels[i]) + ".f32", projections[i]);
        }
        ggml_backend_buffer_free(buffer);
        ggml_backend_free(backend);
        ggml_free(ctx);
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}

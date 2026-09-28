// Bounded Qwen3 block-0 CUDA graph taps for one frozen 29-token prefill.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int32_t TOKENS = 29;
constexpr size_t MAX_BYTES = 16 * 1024 * 1024;

const std::map<std::string, int64_t> SELECTED = {
    {"attn_norm-0", 2560}, {"Qcur_normed-0", 4096}, {"Kcur_normed-0", 1024},
    {"Qcur-0", 4096}, {"Kcur-0", 1024}, {"Vcur-0", 1024},
    {"ffn_inp-0", 2560}, {"ffn_norm-0", 2560}, {"ffn_out-0", 2560},
    {"l_out-0", 2560},
};

struct capture_state {
    std::string directory;
    std::ofstream index;
    std::map<std::string, int> seen;
    size_t bytes_written = 0;
    bool complete = false;
    std::string error;

    static bool callback(ggml_tensor * tensor, bool ask, void * user_data) {
        auto & state = *static_cast<capture_state *>(user_data);
        const std::string name = tensor->name;
        const auto selected = SELECTED.find(name);
        if (selected == SELECTED.end() || state.complete || !state.error.empty()) return false;
        if (ask) return true;
        try {
            if (tensor->type != GGML_TYPE_F32 || ggml_nelements(tensor) != selected->second * TOKENS) {
                throw std::runtime_error("selected block-0 tensor has unexpected type or element count: " + name);
            }
            const size_t bytes = ggml_nbytes(tensor);
            if (bytes == 0 || bytes > MAX_BYTES - state.bytes_written) {
                throw std::runtime_error("block-0 tensor capture exceeds byte cap");
            }
            const int ordinal = state.seen[name]++;
            if (ordinal >= 3) throw std::runtime_error("repeated block-0 tensor exceeds ordinal cap: " + name);
            const std::string filename = name + "_" + std::to_string(ordinal) + ".f32";
            std::vector<char> data(bytes);
            ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
            std::ofstream output(state.directory + "/" + filename, std::ios::binary);
            output.write(data.data(), data.size());
            if (!output) throw std::runtime_error("cannot write block-0 tensor " + filename);
            state.index << name << '\t' << ordinal << '\t' << bytes << '\t' << filename;
            for (int i = 0; i < GGML_MAX_DIMS; ++i) state.index << '\t' << tensor->ne[i];
            for (int i = 0; i < GGML_MAX_DIMS; ++i) state.index << '\t' << tensor->nb[i];
            state.index << '\n';
            if (!state.index) throw std::runtime_error("cannot write block-0 index");
            state.bytes_written += bytes;
            if (name == "l_out-0") state.complete = true;
        } catch (const std::exception & error) {
            state.error = error.what();
            return false;
        }
        return true;
    }
};

std::vector<llama_token> read_tokens(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != TOKENS * (std::streamoff) sizeof(int32_t)) {
        throw std::runtime_error("expected exactly 29 I32 token IDs");
    }
    std::vector<llama_token> tokens(TOKENS);
    input.seekg(0);
    input.read(reinterpret_cast<char *>(tokens.data()), TOKENS * sizeof(int32_t));
    if (!input) throw std::runtime_error("cannot read token IDs");
    for (llama_token token : tokens) if (token < 0) throw std::runtime_error("negative token ID");
    return tokens;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 4) return 2; // target GGUF, 29-token I32 file, output directory
    try {
        auto tokens = read_tokens(argv[2]);
        capture_state state;
        state.directory = argv[3];
        state.index.open(state.directory + "/index.tsv", std::ios::out | std::ios::trunc);
        if (!state.index) throw std::runtime_error("cannot open block-0 capture index");

        llama_backend_init();
        bool expected_gpu = false;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto * device = ggml_backend_dev_get(i);
            const std::string description = ggml_backend_dev_description(device);
            if (ggml_backend_dev_type(device) == GGML_BACKEND_DEVICE_TYPE_GPU &&
                description.find("RTX 5080") != std::string::npos) {
                expected_gpu = true;
                printf("target backend device: %s\n", description.c_str());
            }
        }
        if (!expected_gpu) throw std::runtime_error("RTX 5080 CUDA backend not registered");
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 99;
        llama_model * model = llama_model_load_from_file(argv[1], model_params);
        if (!model) throw std::runtime_error("cannot load target GGUF");
        if (llama_model_n_embd(model) != 2560 || llama_model_n_layer(model) != 36) {
            throw std::runtime_error("target model is not the pinned Qwen3-4B geometry");
        }
        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = 2048;
        context_params.n_batch = 2048;
        context_params.n_ubatch = 512;
        context_params.n_seq_max = 1;
        context_params.n_threads = 10;
        context_params.n_threads_batch = 10;
        context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        context_params.type_k = GGML_TYPE_F16;
        context_params.type_v = GGML_TYPE_F16;
        context_params.cb_eval = capture_state::callback;
        context_params.cb_eval_user_data = &state;
        llama_context * context = llama_init_from_model(model, context_params);
        if (!context) throw std::runtime_error("cannot create target CUDA context");
        llama_batch batch = llama_batch_get_one(tokens.data(), TOKENS);
        const int32_t decoded = llama_decode(context, batch);
        state.index.flush();
        if (decoded != 0 || !state.error.empty() || !state.complete) {
            throw std::runtime_error(
                !state.error.empty() ? state.error : "target prefill did not complete block-0 capture"
            );
        }
        for (const auto & [name, width] : SELECTED) {
            (void) width;
            if (!state.seen.count(name)) throw std::runtime_error("missing block-0 tensor: " + name);
        }
        printf("captured %zu bytes of block-0 tensors\n", state.bytes_written);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
    } catch (const std::exception & error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}

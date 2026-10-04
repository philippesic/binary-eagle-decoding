// CPU-only full-model loading gate; no context, inference, or target loading.
#include "llama.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: check_fusion_binary_model_cpu candidate.gguf\n");
        return 2;
    }
    // The provided candidate declares A8; the native loader cross-checks this selector.
    setenv("GGML_W1AX_ACT_BITS", "8", 1);
    llama_backend_init();
    auto params = llama_model_default_params();
    params.n_gpu_layers = 0;
    params.check_tensors = true;
    auto * model = llama_model_load_from_file(argv[1], params);
    if (!model) {
        llama_backend_free();
        return 1;
    }
    const int layers = llama_model_n_layer(model);
    const int width = llama_model_n_embd(model);
    std::printf("{\"cpu_model_load_passed\":true,\"decoder_layers\":%d,"
                "\"hidden_width\":%d,\"gpu_layers\":0,\"inference_performed\":false}\n",
                layers, width);
    llama_model_free(model);
    llama_backend_free();
    return layers == 1 && width == 2560 ? 0 : 1;
}

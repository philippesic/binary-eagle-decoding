// Compiles the actual extracted (patched) server trace struct + emitter.
// No native contexts, models, schedulers or device APIs exist in this fixture.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <vector>
#include "nlohmann/json.hpp"
using json = nlohmann::ordered_json;
using llama_tokens = std::vector<int32_t>;
struct common_speculative_process_trace {
    int64_t feature_copy_us=0, encoder_us=0, batch_build_us=0, draft_decode_us=0;
    int32_t n_tokens=0, n_draft_decode=0;
};
struct common_speculative_draft_trace {
    int64_t seed_decode_us=0;
    std::vector<int64_t> step_decode_us, sampler_us;
};
static int64_t fixture_time=0;
static int64_t fixture_clock_calls=0;
static int64_t ggml_time_us() { ++fixture_clock_calls; return fixture_time; }
enum stop_type { STOP_TYPE_NONE, STOP_TYPE_EOS, STOP_TYPE_LIMIT, STOP_TYPE_WORD };
#include "round_struct.inc"
struct server_slot {
    int id=0;
    server_round_trace spec_trace_round;
    stop_type stop=STOP_TYPE_NONE;
    struct { int64_t n_gen=0; } stats;
    int remaining=0;
    int n_remaining() const {return remaining;}
};
struct fixture_server {
    std::ofstream round_trace_file;
    struct { struct { struct { int n_max=5; float p_min=0.75f; } draft; } speculative; } params_base;
#include "round_emitter.inc"
};
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    fixture_server server;
    server.round_trace_file.open(argv[1]);
    server_slot slot;
    auto &tr=slot.spec_trace_round;
    tr.active=true; tr.task_id=7; tr.round_index=0; tr.start_us=100;
    tr.draft_start_us=101; tr.draft_end_us=106;
    tr.target_start_us=105; tr.target_end_us=112;
    tr.process_start_us=112; tr.process_end_us=117;
    tr.proposed={10,99,12}; tr.n_accepted=3; tr.emitted={10,99};
    tr.session_context={{"generated_before",1},{"remaining_before",2},
                        {"cached_prompt_token_ids",{1,2}}, {"pending_seed_token_id",9}};
    tr.target_batches.push_back({{"start_us",105},{"end_us",112},{"batch_tokens",4},{"slot_tokens",4},{"slot_output_rows",4}});
    slot.stats.n_gen=3; slot.remaining=0; slot.stop=STOP_TYPE_EOS;
    fixture_time=120;
    server.emit_round_trace(slot,"complete");
    const auto bytes=server.round_trace_file.tellp();
    // Actual disabled emitter's guard returns without any event or clock access.
    const auto clocks=fixture_clock_calls;
    fixture_time=999;
    server.emit_round_trace(slot,"complete");
    if (server.round_trace_file.tellp()!=bytes || clocks!=fixture_clock_calls) return 3;
    return 0;
}

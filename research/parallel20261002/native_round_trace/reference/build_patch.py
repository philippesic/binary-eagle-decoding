"""Build an unapplied additive trace extension against pinned native source."""

import difflib
import hashlib
import json
from pathlib import Path

PIN = "9e2c7a90051e738751aab7d7bd7c2d8201fb76e3"
REL = "tools/server/server-context.cpp"
BASE = Path("/Users/pippo/github/binary-eagle-decoding/third_party/llama.cpp")
OUT = Path(__file__).parent
source = (BASE / REL).read_text()
modified = source


def replace(old, new):
    global modified
    assert modified.count(old) == 1, f"nonunique/missing source boundary {old[:60]}"
    modified = modified.replace(old, new)


replace(
    "    common_speculative_process_trace process_detail;\n",
    """    // Additive opt-in accounting metadata; populated only in active traces.
    json session_context;
    json target_batches = json::array();
    std::vector<json> process_batches;
    common_speculative_process_trace process_detail;
""",
)
replace(
    "                tr.start_us = ggml_time_us();\n",
    """                tr.start_us = ggml_time_us();
                // The pending sampled token has already been emitted. Cached
                // prompt tokens may have been context-shifted; preserve these
                // separately rather than naming them the full transcript.
                tr.session_context = {
                    {"generated_before", slot.stats.n_gen},
                    {"remaining_before", slot.n_remaining()},
                    {"decoder_position", slot.prompt.tokens.pos_next()},
                    {"cached_prompt_token_ids", slot.prompt.tokens.get_text_tokens()},
                    {"pending_seed_token_id", slot.sampled},
                    {"root_identity_kind", "shiftable_cached_prompt_plus_pending_seed"},
                    {"private_state_identity", nullptr},
                    {"attempt_index", tr.round_index},
                    {"replay", tr.replay},
                };
""",
)
# Use vector rather than eager JSON arrays so disabled/reset path has no new allocations.
modified = modified.replace(
    "    json target_batches = json::array();", "    std::vector<json> target_batches;"
)
replace(
    "                    if (!tr.target_start_us) tr.target_start_us = t_target;\n",
    """                    int32_t slot_rows = 0;
                    int32_t slot_output_rows = 0;
                    for (int i = 0; i < batch_view.n_tokens; ++i) {
                        for (int j = 0; j < batch_view.n_seq_id[i]; ++j) {
                            if (batch_view.seq_id[i][j] == slot.id) {
                                ++slot_rows;
                                if (batch_view.logits && batch_view.logits[i]) ++slot_output_rows;
                                break;
                            }
                        }
                    }
                    tr.target_batches.push_back({
                        {"start_us", t_target}, {"end_us", t_end},
                        {"batch_tokens", batch_view.n_tokens}, {"slot_tokens", slot_rows},
                        {"slot_output_rows", slot_output_rows}, {"decode_return", ret},
                        {"existing_output_sync", ret == 0 && has_output},
                        {"execution_domain", "native_process_target_context"},
                        {"timing_kind", "host_wall_existing_sync_when_output"},
                    });
                    if (!tr.target_start_us) tr.target_start_us = t_target;
""",
)
replace(
    "                        if (!tr.process_start_us) tr.process_start_us = t_process;\n",
    """                        tr.process_batches.push_back({
                            {"start_us", t_process}, {"end_us", t_end},
                            {"batch_tokens", batch_view.n_tokens},
                            {"feature_tokens", detail.n_tokens},
                            {"draft_decode_tokens", detail.n_draft_decode},
                            {"execution_domain", "native_process_draft_context"},
                            {"timing_kind", "host_call_wall_completion_unproven"},
                        });
                        if (!tr.process_start_us) tr.process_start_us = t_process;
""",
)
replace(
    "        round_trace_file << record.dump() << '\\n';\n",
    """        tr.session_context["native_stop_type"] = slot.stop == STOP_TYPE_EOS ? "eos" :
            slot.stop == STOP_TYPE_LIMIT ? "limit" : slot.stop == STOP_TYPE_WORD ? "word" : "none";
        tr.session_context["remaining_after"] = slot.n_remaining();
        tr.session_context["generated_after"] = slot.stats.n_gen;
        tr.session_context["target_batches"] = tr.target_batches;
        tr.session_context["process_batches"] = tr.process_batches;
        tr.session_context["clock_scope"] = "one native process incarnation; bind externally";
        tr.session_context["gpu_time_us"] = nullptr;
        tr.session_context["role_accounting"] =
            tr.replay ? "replay_offset_unresolved" : "ordered_prefix";
        // Stop processing may leave un-emitted accepted ids in slot.prompt.
        // A next committed transcript is root transcript + emitted_token_ids;
        // never infer it from the terminal cache snapshot.
        record["session_context_v1"] = tr.session_context;
        round_trace_file << record.dump() << '\\n';
""",
)
patch = "".join(
    difflib.unified_diff(
        source.splitlines(True), modified.splitlines(True), fromfile="a/" + REL, tofile="b/" + REL
    )
)
(OUT / "native-session-context.unapplied.patch").write_text(patch)
(OUT / "source-binding.json").write_text(
    json.dumps(
        {
            "native_commit": PIN,
            "source_path": REL,
            "original_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "patched_sha256": hashlib.sha256(modified.encode()).hexdigest(),
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "state": "UNAPPLIED; no native build or runtime adoption",
        },
        indent=2,
    )
    + "\n"
)

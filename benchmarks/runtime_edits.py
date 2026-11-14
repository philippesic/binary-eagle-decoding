SOURCE_EDITS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "checkpoint_warm_buffers_v1": (
        (
            "common/common.cpp",
            r"""#include <climits>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
""",
            r"""#include <climits>
#include <cmath>
#include <chrono>
#include <cstdarg>
#include <condition_variable>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
""",
        ),
        (
            "common/common.cpp",
            r"""#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
""",
            r"""#include <fcntl.h>
#include <io.h>
#else
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#if !defined(_WIN32)
""",
        ),
        (
            "common/common.cpp",
            r"""
    return true;
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

""",
            r"""
    return true;
}

struct common_state_pool {
    std::mutex mtx;
    std::condition_variable cv;
    std::multimap<size_t, void *> idle;
    std::unordered_map<void *, size_t> caps;
    size_t idle_bytes = 0;
    size_t recent[4] = { 0, 0, 0, 0 };
    size_t n_recent = 0;
    bool   worker = false;
};

static common_state_pool & common_state_pool_get() {
    static common_state_pool * pool = new common_state_pool();
    return *pool;
}

static constexpr size_t COMMON_STATE_POOL_MIN  = 1u << 20;
static constexpr size_t COMMON_STATE_POOL_ALGN = 2u << 20;
static constexpr size_t COMMON_STATE_IDLE_MAX  = (size_t) 2 << 30;
static constexpr size_t COMMON_STATE_RESERVE   = 4;

static void * common_state_map(size_t cap) {
#if defined(__linux__)
    void * ptr = mmap(nullptr, cap, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::bad_alloc();
    }
    return ptr;
#else
    return ::operator new(cap);
#endif
}

static void common_state_unmap(void * ptr, size_t cap) {
#if defined(__linux__)
    munmap(ptr, cap);
#else
    GGML_UNUSED(cap);
    ::operator delete(ptr);
#endif
}

static size_t common_state_reserve_cap(const common_state_pool & pool) {
    size_t cap = 0;
    for (size_t i = 0; i < std::min<size_t>(pool.n_recent, 4); ++i) {
        cap = std::max(cap, pool.recent[i]);
    }
    return cap;
}

static size_t common_state_reserve_count(const common_state_pool & pool, size_t cap) {
    size_t count = 0;
    for (auto it = pool.idle.lower_bound(cap); it != pool.idle.end(); ++it) {
        count++;
    }
    return count;
}

static void common_state_reserve_loop() {
    auto & pool = common_state_pool_get();
    std::unique_lock<std::mutex> lock(pool.mtx);
    while (true) {
        pool.cv.wait(lock, [&] {
            const size_t cap = common_state_reserve_cap(pool);
            return cap > 0 && common_state_reserve_count(pool, cap) < COMMON_STATE_RESERVE &&
                   pool.idle_bytes + cap <= COMMON_STATE_IDLE_MAX;
        });
        const size_t cap = common_state_reserve_cap(pool);
        lock.unlock();
        void * ptr = nullptr;
        try {
            ptr = common_state_map(cap);
            std::memset(ptr, 0, cap);
        } catch (const std::bad_alloc &) {
            ptr = nullptr;
        }
        lock.lock();
        if (ptr == nullptr) {
            pool.n_recent = 0;
            continue;
        }
        pool.caps[ptr] = cap;
        pool.idle.emplace(cap, ptr);
        pool.idle_bytes += cap;
    }
}

void * common_state_alloc(size_t size, bool pooled) {
    if (!pooled || size < COMMON_STATE_POOL_MIN) {
        return ::operator new(size);
    }
    auto & pool = common_state_pool_get();
    std::lock_guard<std::mutex> lock(pool.mtx);
    const size_t cap = GGML_PAD(size + size/8, COMMON_STATE_POOL_ALGN);
    pool.recent[pool.n_recent++ % 4] = cap;
    if (!pool.worker) {
        pool.worker = true;
        std::thread(common_state_reserve_loop).detach();
    }
    void * ptr = nullptr;
    auto it = pool.idle.lower_bound(size);
    if (it != pool.idle.end() && it->first <= 2*cap) {
        ptr = it->second;
        pool.idle_bytes -= it->first;
        pool.idle.erase(it);
    } else {
        ptr = common_state_map(cap);
        pool.caps[ptr] = cap;
    }
    pool.cv.notify_one();
    return ptr;
}

void common_state_free(void * ptr, size_t size) {
    if (ptr == nullptr) {
        return;
    }
    if (size >= COMMON_STATE_POOL_MIN) {
        auto & pool = common_state_pool_get();
        std::lock_guard<std::mutex> lock(pool.mtx);
        auto it = pool.caps.find(ptr);
        if (it != pool.caps.end()) {
            const size_t cap = it->second;
            if (pool.idle_bytes + cap <= COMMON_STATE_IDLE_MAX) {
                pool.idle.emplace(cap, ptr);
                pool.idle_bytes += cap;
            } else {
                pool.caps.erase(it);
                common_state_unmap(ptr, cap);
            }
            return;
        }
    }
    ::operator delete(ptr);
}

size_t common_prompt_checkpoint::size() const {
    return data_tgt.size() + data_dft.size() + data_spec.size();
}

""",
        ),
        (
            "common/common.h",
            r"""#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
""",
            r"""#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <new>
#include <type_traits>
#include <utility>

#if defined(_WIN32) && !defined(_WIN32_WINNT)
#define _WIN32_WINNT 0x0A00
#endif
""",
        ),
        (
            "common/common.h",
            r"""//
// prompt utils
//

struct common_prompt_checkpoint {
    int64_t n_tokens;

    // (optional) id of the task that created the checkpoint
""",
            r"""//
// prompt utils
//

void * common_state_alloc(size_t size, bool pooled);
void   common_state_free(void * ptr, size_t size);

template <typename T>
struct common_state_allocator {
    using value_type = T;
    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap            = std::true_type;
    using is_always_equal                        = std::true_type;

    bool pooled = true;

    common_state_allocator() = default;
    explicit common_state_allocator(bool pooled) : pooled(pooled) {}
    template <typename U>
    common_state_allocator(const common_state_allocator<U> & other) : pooled(other.pooled) {}

    T * allocate(size_t n) { return static_cast<T *>(common_state_alloc(n * sizeof(T), pooled)); }
    void deallocate(T * ptr, size_t n) { common_state_free(ptr, n * sizeof(T)); }

    template <typename U>
    void construct(U * ptr) noexcept { ::new (static_cast<void *>(ptr)) U; }
    template <typename U, typename... Args>
    void construct(U * ptr, Args &&... args) { ::new (static_cast<void *>(ptr)) U(std::forward<Args>(args)...); }

    common_state_allocator select_on_container_copy_construction() const { return common_state_allocator(false); }

    template <typename U>
    bool operator==(const common_state_allocator<U> &) const { return true; }
    template <typename U>
    bool operator!=(const common_state_allocator<U> &) const { return false; }
};

using common_state_bytes = std::vector<uint8_t, common_state_allocator<uint8_t>>;

struct common_prompt_checkpoint {
    int64_t n_tokens;

    // (optional) id of the task that created the checkpoint
""",
        ),
        (
            "common/common.h",
            r"""
    llama_pos pos_min;
    llama_pos pos_max;

    std::vector<uint8_t> data_tgt;
    std::vector<uint8_t> data_dft;

    // (optional) speculative-decoding implementation state stashed with the checkpoint
    // (e.g. eagle3's deferred-boundary g_embd row)
    std::vector<uint8_t> data_spec;
""",
            r"""
    llama_pos pos_min;
    llama_pos pos_max;

    common_state_bytes data_tgt;
    common_state_bytes data_dft;

    // (optional) speculative-decoding implementation state stashed with the checkpoint
    // (e.g. eagle3's deferred-boundary g_embd row)
    std::vector<uint8_t> data_spec;
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        n_read += size;
        return true;
    }

    static bool ckpt_read_buf(std::ifstream & ifs, std::vector<uint8_t> & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
""",
            r"""        n_read += size;
        return true;
    }

    template <typename Bytes>
    static bool ckpt_read_buf(std::ifstream & ifs, Bytes & buf, size_t n_avail, size_t & n_read) {
        uint64_t n = 0;
        // check the size against the bytes left in the file before allocating, the size field may be corrupted
        if (!ckpt_read(ifs, &n, sizeof(n), n_read) || n > n_avail - n_read) {
            return false;
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        ofs.write((const char *) src, size);
        n_written += size;
    }

    static void ckpt_write_buf(std::ofstream & ofs, const std::vector<uint8_t> & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
""",
            r"""        ofs.write((const char *) src, size);
        n_written += size;
    }

    template <typename Bytes>
    static void ckpt_write_buf(std::ofstream & ofs, const Bytes & buf, size_t & n_written) {
        const uint64_t n = buf.size();
        ckpt_write(ofs, &n, sizeof(n), n_written);
        if (n > 0) {
            ckpt_write(ofs, buf.data(), n, n_written);
""",
        ),
    ),
    "device_feature_handoff_v1": (
        (
            "common/speculative.cpp",
            r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
""",
            r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
""",
        ),
        (
            "common/speculative.cpp",
            r"""    common_batch batch_inject; // target features for KV cache injection

    std::vector<float> features_buf; // [n_chunk, n_embd_enc] gathered target features

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;
""",
            r"""    common_batch batch_inject; // target features for KV cache injection

    std::vector<float> features_buf; // [n_chunk, n_embd_enc] gathered target features

    bool                features_dev = false;
    llama_batch_ext_ptr batch_dev;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;
""",
        ),
        (
            "common/speculative.cpp",
            r"""        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2 && !is_lilicorr);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }
""",
            r"""        for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
            llama_set_embeddings_layer_inp(ctx_tgt, (uint32_t) target_layer_ids[k], true);
        }

        bool ids_ascending = true;
        for (uint32_t k = 1; k < target_layer_ids_n; ++k) {
            ids_ascending = ids_ascending && target_layer_ids[k] > target_layer_ids[k - 1];
        }
        features_dev = ids_ascending && !is_lilicorr && llama_model_dflash_device_features(model_dft) &&
                       std::getenv("LLAMA_SPEC_HOST_FEATURES") == nullptr;
        if (features_dev) {
            llama_set_embeddings_layer_inp_device(ctx_tgt, true);
            batch_dev.reset(llama_batch_ext_init(ctx_dft));
        }
        LOG_INF("%s: - target features handed over on the %s\n", __func__, features_dev ? "device" : "host");

        // DFlash2 reads its selector lattice from h_nextn and never consumes raw logits.
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ !is_dflash2 && !is_lilicorr);
        llama_set_causal_attn(ctx_dft, causal_attn); // DFlash needs non-causal attention unless the model says otherwise
    }
""",
        ),
        (
            "common/speculative.cpp",
            r"""
            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
""",
            r"""
            for (int32_t offset = 0; offset < n_rows; offset += n_ubatch) {
                const int32_t n_chunk = std::min(n_ubatch, n_rows - offset);

                if (features_dev) {
                    const ggml_tensor * src = llama_get_embeddings_layer_inp_device(ctx_tgt);
                    if (!src) {
                        GGML_ABORT("DFlash: target layer inputs not extracted on the device.");
                    }
                    llama_batch_ext * b = batch_dev.get();
                    llama_batch_ext_clear(b);
                    for (int32_t i = 0; i < n_chunk; ++i) {
                        const llama_pos p = batch_in.tokens[i_batch_beg[seq_id] + offset + i].pos[0];
                        const llama_pos pos_arr[4] = { p, p, p, 0 };
                        llama_pos pos_row[GGML_MROPE_SECTIONS] = { 0, 0, 0, 0 };
                        for (int32_t j = 0; j < batch_inject.n_pos; ++j) {
                            pos_row[j] = pos_arr[j];
                        }
                        const int32_t idx = llama_batch_ext_add_embd_device(b, seq_id);
                        if (idx < 0 || !llama_batch_ext_set_pos(b, idx, pos_row)) {
                            GGML_ABORT("DFlash: failed to add device feature row %d", i);
                        }
                    }
                    llama_batch_ext_set_embd_device(b, src, i_batch_beg[seq_id] + offset);
                    const int32_t rc = llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, b);
                    if (rc != 0) {
                        LOG_ERR("%s: llama_process(ctx_dft) failed rc=%d (n_tokens=%d, offset=%d)\n",
                                __func__, rc, (int) n_chunk, (int) offset);
                        return false;
                    }
                    llama_embeddings_layer_inp_consumed(ctx_tgt, ctx_dft);
                    continue;
                }

                // gather target features per extract layer; the fused decode encodes and
                // injects them into the K/V cache at the target positions
                features_buf.resize((size_t) n_chunk * n_embd_enc);
                for (uint32_t k = 0; k < target_layer_ids_n; ++k) {
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    }

    const bool mixed = n_tok_only > 0 && n_embd_only > 0;

    if (mixed && !allow_mixed) {
        LLAMA_LOG_ERROR("%s: this model or context does not support batches mixing token and embedding entries\n", __func__);
        return false;
    }
""",
            r"""    }

    const bool mixed = n_tok_only > 0 && n_embd_only > 0;

    if (batch_inp.embd_device && (mixed || n_both > 0 || batch_inp.embd_dev == nullptr)) {
        LLAMA_LOG_ERROR("%s: device embedding rows need an embedding-only batch with a device source\n", __func__);
        return false;
    }

    if (mixed && !allow_mixed) {
        LLAMA_LOG_ERROR("%s: this model or context does not support batches mixing token and embedding entries\n", __func__);
        return false;
    }
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""                const float * src = batch_inp.embd.data() + batch_inp.tokens[i].embd_off;
                std::copy(src, src + n_embd, embd_vec.data() + (size_t) i*n_embd);
            }
        }
    } else if (has_embd) {
        embd_vec = batch_inp.embd;
    }

    if (has_state) {
        state_vec = batch_inp.state;
    }

""",
            r"""                const float * src = batch_inp.embd.data() + batch_inp.tokens[i].embd_off;
                std::copy(src, src + n_embd, embd_vec.data() + (size_t) i*n_embd);
            }
        }
    } else if (has_embd && !batch_inp.embd_device) {
        embd_vec = batch_inp.embd;
    }

    embd_dev      = batch_inp.embd_device ? batch_inp.embd_dev : nullptr;
    embd_dev_row0 = batch_inp.embd_device ? batch_inp.embd_dev_row0 : 0;

    if (has_state) {
        state_vec = batch_inp.state;
    }

""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    //

    batch.n_tokens = n_tok;
    batch.token    = has_token ? token_vec.data() : nullptr;
    batch.embd     = has_embd  ? embd_vec.data()  : nullptr;
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.logits   = output.data();
""",
            r"""    //

    batch.n_tokens = n_tok;
    batch.token    = has_token ? token_vec.data() : nullptr;
    batch.embd     = has_embd  ? (embd_dev ? llama_ubatch_embd_device_marker() : embd_vec.data()) : nullptr;
    batch.pos      = pos.data();
    batch.n_seq_id = n_seq_id.data();
    batch.seq_id   = seq_id.data();
    batch.logits   = output.data();
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    batch = {};

    token_vec   .clear();
    embd_vec    .clear();
    is_embd_vec .clear();
    state_vec   .clear();
    seq_id_data .clear();
    pos         .clear();
""",
            r"""    batch = {};

    token_vec   .clear();
    embd_vec    .clear();
    embd_dev      = nullptr;
    embd_dev_row0 = 0;
    is_embd_vec .clear();
    state_vec   .clear();
    seq_id_data .clear();
    pos         .clear();
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    const bool use_token = batch.token && !(mixed_batch && n_embd_rows == n_tokens);
    const bool use_embd  = batch.embd  && !(mixed_batch && n_embd_rows == 0);
    const bool has_state = !state_vec.empty();

    const int64_t n_embd_all  = use_embd  ? (int64_t) n_tokens*n_embd       : 0;
    const int64_t n_state_all = has_state ? (int64_t) n_tokens*n_embd_state : 0;
    const int64_t n_pos_all   =             (int64_t) n_tokens*n_pos_per_embd;

    udata->token     .resize(n_tokens);
""",
            r"""    const bool use_token = batch.token && !(mixed_batch && n_embd_rows == n_tokens);
    const bool use_embd  = batch.embd  && !(mixed_batch && n_embd_rows == 0);
    const bool has_state = !state_vec.empty();

    const bool    on_device   = use_embd && embd_dev != nullptr;
    const int64_t n_embd_all  = use_embd && !on_device ? (int64_t) n_tokens*n_embd : 0;
    const int64_t n_state_all = has_state ? (int64_t) n_tokens*n_embd_state : 0;
    const int64_t n_pos_all   =             (int64_t) n_tokens*n_pos_per_embd;

    udata->token     .resize(n_tokens);
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""        if (use_token) {
            udata->token[i] = batch.token[idxs[i]];
        }

        if (use_embd) {
            memcpy(udata->embd.data() + i*n_embd, batch.embd + (int64_t) idxs[i]*n_embd, n_embd*sizeof(float));
        }

        if (mixed) {
""",
            r"""        if (use_token) {
            udata->token[i] = batch.token[idxs[i]];
        }

        if (on_device) {
            GGML_ASSERT(idxs[i] == idxs[0] + (int32_t) i && "device embedding rows must stay contiguous");
        } else if (use_embd) {
            memcpy(udata->embd.data() + i*n_embd, batch.embd + (int64_t) idxs[i]*n_embd, n_embd*sizeof(float));
        }

        if (mixed) {
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""        /*.n_seqs_unq   =*/ (uint32_t) udata->seq_id_unq.size(),
        /*.n_pos        =*/ n_pos_per_embd,

        /*.token        =*/ use_token ? udata->token.data() : nullptr,
        /*.embd         =*/ use_embd  ? udata->embd.data()  : nullptr,
        /*.embd_state   =*/ has_state ? udata->embd_state.data() : nullptr,
        /*.pos          =*/ udata->pos.data(),
        /*.n_seq_id     =*/ udata->n_seq_id.data(),
        /*.seq_id       =*/ udata->seq_id.data(),
""",
            r"""        /*.n_seqs_unq   =*/ (uint32_t) udata->seq_id_unq.size(),
        /*.n_pos        =*/ n_pos_per_embd,

        /*.token        =*/ use_token ? udata->token.data() : nullptr,
        /*.embd         =*/ use_embd  ? (on_device ? llama_ubatch_embd_device_marker() : udata->embd.data()) : nullptr,
        /*.embd_state   =*/ has_state ? udata->embd_state.data() : nullptr,
        /*.pos          =*/ udata->pos.data(),
        /*.n_seq_id     =*/ udata->n_seq_id.data(),
        /*.seq_id       =*/ udata->seq_id.data(),
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""        /*.decision_order =*/ udata->decision_order.empty() ? nullptr : udata->decision_order.data(),
        /*.data         =*/ std::move(udata),
    };

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: added ubatch to split:\n", __func__);

        ubatch_print(res, debug);
""",
            r"""        /*.decision_order =*/ udata->decision_order.empty() ? nullptr : udata->decision_order.data(),
        /*.data         =*/ std::move(udata),
    };

    if (on_device) {
        res.embd_dev      = embd_dev;
        res.embd_dev_row0 = embd_dev_row0 + idxs[0];
    }

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: added ubatch to split:\n", __func__);

        ubatch_print(res, debug);
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    tokens.clear();
    embd  .clear();
    state .clear();
    n_embd = 0;
}

int32_t llama_batch_ext::add_token(llama_seq_id seq_id) {
    if (tokens.size() >= n_tokens_max) {
""",
            r"""    tokens.clear();
    embd  .clear();
    state .clear();
    n_embd = 0;
    embd_device   = false;
    embd_dev      = nullptr;
    embd_dev_row0 = 0;
}

float * llama_ubatch_embd_device_marker() {
    static float marker = 0.0f;
    return &marker;
}

bool llama_batch_ext::set_token_embd_device(int32_t idx) {
    if (idx < 0 || idx >= (int32_t) tokens.size() || n_embd_inp_enc == 0 || !embd.empty()) {
        return false;
    }
    if (n_embd == 0) {
        n_embd = n_embd_inp_enc;
    } else if (n_embd != n_embd_inp_enc) {
        return false;
    }

    token & t = tokens[idx];

    if (t.has_embd) {
        return false;
    }

    t.has_embd  = true;
    t.embd_off  = 0;
    embd_device = true;

    return true;
}

int32_t llama_batch_ext::add_token(llama_seq_id seq_id) {
    if (tokens.size() >= n_tokens_max) {
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""bool llama_batch_ext::set_token_embd(int32_t idx, llama_embd embd_in) {
    if (idx < 0 || idx >= (int32_t) tokens.size()) {
        return false;
    }
    if (!embd_in.data) {
        return false;
    }

    const size_t n_total = embd_in.n_rows * embd_in.n_embd;
""",
            r"""bool llama_batch_ext::set_token_embd(int32_t idx, llama_embd embd_in) {
    if (idx < 0 || idx >= (int32_t) tokens.size()) {
        return false;
    }
    if (!embd_in.data || embd_device) {
        return false;
    }

    const size_t n_total = embd_in.n_rows * embd_in.n_embd;
""",
        ),
        (
            "src/llama-batch.cpp",
            r"""    }
    return idx;
}

bool llama_batch_ext_add_seq(llama_batch_ext * batch, int32_t idx, llama_seq_id seq_id) {
    return batch->add_seq(idx, seq_id);
}

""",
            r"""    }
    return idx;
}

int32_t llama_batch_ext_add_embd_device(llama_batch_ext * batch, llama_seq_id seq_id) {
    int32_t idx = batch->add_token(seq_id);
    if (idx < 0) {
        return idx;
    }
    if (!batch->set_token_embd_device(idx)) {
        return -2;
    }
    return idx;
}

void llama_batch_ext_set_embd_device(llama_batch_ext * batch, const struct ggml_tensor * src, int64_t row0) {
    batch->embd_dev      = src;
    batch->embd_dev_row0 = row0;
}

bool llama_batch_ext_add_seq(llama_batch_ext * batch, int32_t idx, llama_seq_id seq_id) {
    return batch->add_seq(idx, seq_id);
}

""",
        ),
        (
            "src/llama-batch.h",
            r"""    };

    // the llama_ubatch pointers above point to this data if set. otherwise - point to external non-owning data
    std::shared_ptr<data_t> data;
};

// crash if a mixed ubatch reaches code that expects only tokens or only embd
#define ASSERT_EMBD_OR_TOKEN(ubatch) GGML_ASSERT(!(ubatch).is_mixed() && "mixed token/embd ubatch is not supported here")

struct llama_hparams;
""",
            r"""    };

    // the llama_ubatch pointers above point to this data if set. otherwise - point to external non-owning data
    std::shared_ptr<data_t> data;

    const struct ggml_tensor * embd_dev = nullptr;
    int64_t embd_dev_row0 = 0;
};

float * llama_ubatch_embd_device_marker();

// crash if a mixed ubatch reaches code that expects only tokens or only embd
#define ASSERT_EMBD_OR_TOKEN(ubatch) GGML_ASSERT(!(ubatch).is_mixed() && "mixed token/embd ubatch is not supported here")

struct llama_hparams;
""",
        ),
        (
            "src/llama-batch.h",
            r"""    std::vector<token> tokens;
    std::vector<float> embd;
    std::vector<float> state;

    llama_batch_ext(llama_context * ctx);

    // build without a llama_context, used by tests
    llama_batch_ext(
""",
            r"""    std::vector<token> tokens;
    std::vector<float> embd;
    std::vector<float> state;

    bool embd_device = false;
    const struct ggml_tensor * embd_dev = nullptr;
    int64_t embd_dev_row0 = 0;

    llama_batch_ext(llama_context * ctx);

    // build without a llama_context, used by tests
    llama_batch_ext(
""",
        ),
        (
            "src/llama-batch.h",
            r"""
    bool add_seq(int32_t idx, llama_seq_id seq_id);
    bool set_token_id(int32_t idx, llama_token id);
    bool set_token_embd(int32_t idx, llama_embd embd_in);
    bool set_token_state(int32_t idx, llama_embd state_in);
    bool set_token_pos(int32_t idx, const llama_pos * pos_in);
    bool set_output(int32_t idx, bool output_last);
    bool set_decision_order(int32_t idx, int32_t order);
""",
            r"""
    bool add_seq(int32_t idx, llama_seq_id seq_id);
    bool set_token_id(int32_t idx, llama_token id);
    bool set_token_embd(int32_t idx, llama_embd embd_in);
    bool set_token_embd_device(int32_t idx);
    bool set_token_state(int32_t idx, llama_embd state_in);
    bool set_token_pos(int32_t idx, const llama_pos * pos_in);
    bool set_output(int32_t idx, bool output_last);
    bool set_decision_order(int32_t idx, int32_t order);
""",
        ),
        (
            "src/llama-batch.h",
            r"""    uint32_t n_outputs;

    std::vector<llama_token>    token_vec;    // owned token IDs built from llama_batch_ext
    std::vector<float>          embd_vec;     // owned embeddings built from llama_batch_ext
    std::vector<int8_t>         is_embd_vec;  // mixed batch only (= 1 if embd, 0 if text token)
    std::vector<float>          state_vec;    // owned state embeddings built from llama_batch_ext, llama_batch has no slot for them
    std::vector<llama_seq_id>   seq_id_data;  // flat storage for seq_id pointers below

""",
            r"""    uint32_t n_outputs;

    std::vector<llama_token>    token_vec;    // owned token IDs built from llama_batch_ext
    std::vector<float>          embd_vec;     // owned embeddings built from llama_batch_ext
    const struct ggml_tensor *  embd_dev = nullptr;
    int64_t                     embd_dev_row0 = 0;
    std::vector<int8_t>         is_embd_vec;  // mixed batch only (= 1 if embd, 0 if text token)
    std::vector<float>          state_vec;    // owned state embeddings built from llama_batch_ext, llama_batch has no slot for them
    std::vector<llama_seq_id>   seq_id_data;  // flat storage for seq_id pointers below

""",
        ),
        (
            "src/llama-context.cpp",
            r"""#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
""",
            r"""#include "llama-context.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
""",
        ),
        (
            "src/llama-context.cpp",
            r"""
    return embd_layer_inp[lid].data;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
""",
            r"""
    return embd_layer_inp[lid].data;
}

const ggml_tensor * llama_context::get_embeddings_layer_inp_device() {
    return t_layer_inp_dev;
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

""",
            r"""    cparams.embeddings_nextn        = value;
    cparams.embeddings_nextn_masked = masked;
}

void llama_context::set_embeddings_layer_inp_device(bool enable) {
    synchronize();

    layer_inp_device = enable;

    t_layer_inp_dev = nullptr;
    buf_layer_inp_dev.reset();
    ctx_layer_inp_dev.reset();
}

void llama_context::embeddings_layer_inp_consumed(llama_context * consumer) {
    if (t_layer_inp_dev == nullptr) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buf_layer_inp_dev.get()));
    ggml_backend_t backend = nullptr;
    for (auto & cur : consumer->backends) {
        if (dev != nullptr && ggml_backend_get_device(cur.get()) == dev) {
            backend = cur.get();
            break;
        }
    }
    if (backend != nullptr && !layer_inp_event) {
        layer_inp_event.reset(ggml_backend_event_new(dev));
    }
    if (backend == nullptr || !layer_inp_event) {
        consumer->synchronize();
        return;
    }
    ggml_backend_event_record(layer_inp_event.get(), backend);
    layer_inp_event_pending = true;
}

void llama_context::set_embeddings_layer_inp(uint32_t lid, bool enable) {
    LLAMA_LOG_DEBUG("%s: lid = %d, enable = %d\n", __func__, lid, enable);

    GGML_ASSERT(lid <= model.hparams.n_layer());

    cparams.embeddings_layer_inp[lid] = enable;

    t_layer_inp_dev = nullptr;
    buf_layer_inp_dev.reset();
    ctx_layer_inp_dev.reset();

    // note: without this reserve, the draft acceptance drops to zero. not sure why - this is unexpected
    sched_need_reserve = true;
}

""",
        ),
        (
            "src/llama-context.cpp",
            r"""                    }
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
""",
            r"""                    }
            }
        }

        if (layer_inp_device) {
            GGML_ASSERT(ubatch.data && ubatch.data->batch_idxs.size() == ubatch.n_tokens);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                GGML_ASSERT(ubatch.data->batch_idxs[i] == (int32_t) (n_tokens_prev + i) && "device layer inputs need batch-ordered ubatches");
            }
        }

        // [TAG_EXTRACT_TARGET_EMBEDDINGS]
        bool extract_all_idxs = extract_layer_inputs(res, n_tokens_prev, ubatch.n_tokens);

        // extract nextn embeddings before
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
""",
            r"""    return n_outputs_max;
}

bool llama_context::extract_layer_inputs(const llm_graph_result * res, size_t token_offset, size_t n_tokens) {
    if (layer_inp_device) {
        std::vector<uint32_t> lids;
        for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
            if (cparams.embeddings_layer_inp[il]) {
                lids.push_back(il);
            }
        }
        if (lids.empty()) {
            return false;
        }

        ggml_tensor * first = res->get_layer_inp((int) lids[0]);
        GGML_ASSERT(first != nullptr);
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched.get(), first);
        GGML_ASSERT(backend != nullptr);

        const int64_t n_embd_l = model.hparams.n_embd;
        const int64_t n_rows   = cparams.n_batch;

        if (t_layer_inp_dev == nullptr) {
            ggml_init_params params_dev = { ggml_tensor_overhead(), nullptr, true };
            ctx_layer_inp_dev.reset(ggml_init(params_dev));
            t_layer_inp_dev = ggml_new_tensor_3d(ctx_layer_inp_dev.get(), GGML_TYPE_F32, n_embd_l, n_rows, (int64_t) lids.size());
            buf_layer_inp_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_layer_inp_dev.get(), ggml_backend_get_default_buffer_type(backend)));
            GGML_ASSERT(buf_layer_inp_dev != nullptr && "failed to allocate the device layer input buffer");

            ggml_init_params params_views = { ggml_tensor_overhead() * (LLAMA_MAX_LAYERS + 1), nullptr, true };
            ctx_layer_inp_views.reset(ggml_init(params_views));
        }

        GGML_ASSERT(t_layer_inp_dev->ne[2] == (int64_t) lids.size());
        GGML_ASSERT((int64_t) (token_offset + n_tokens) <= n_rows);

        ggml_reset(ctx_layer_inp_views.get());

        if (layer_inp_event_pending) {
            ggml_backend_event_wait(backend, layer_inp_event.get());
            layer_inp_event_pending = false;
        }

        for (size_t k = 0; k < lids.size(); ++k) {
            ggml_tensor * t = res->get_layer_inp((int) lids[k]);
            GGML_ASSERT(t != nullptr && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
            GGML_ASSERT(t->ne[0] == n_embd_l && ggml_nelements(t) == n_embd_l * (int64_t) n_tokens);
            GGML_ASSERT(ggml_backend_sched_get_tensor_backend(sched.get(), t) == backend);

            const size_t offset = k * t_layer_inp_dev->nb[2] + token_offset * t_layer_inp_dev->nb[1];
            ggml_tensor * dst = ggml_view_4d(ctx_layer_inp_views.get(), t_layer_inp_dev,
                    t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[1], t->nb[2], t->nb[3], offset);
            GGML_ASSERT(ggml_backend_view_init(dst) == GGML_STATUS_SUCCESS);
            ggml_backend_tensor_copy_async(backend, backend, t, dst);
        }

        return false;
    }

    bool extracted = false;
    for (uint32_t il = 0; il < cparams.embeddings_layer_inp.size(); ++il) {
        if (!cparams.embeddings_layer_inp[il]) {
            continue;
""",
        ),
        (
            "src/llama-context.cpp",
            r"""
    return ctx->get_embeddings_layer_inp(lid);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

""",
            r"""
    return ctx->get_embeddings_layer_inp(lid);
}

void llama_set_embeddings_layer_inp_device(llama_context * ctx, bool value) {
    ctx->set_embeddings_layer_inp_device(value);
}

void llama_embeddings_layer_inp_consumed(llama_context * ctx, llama_context * consumer) {
    ctx->embeddings_layer_inp_consumed(consumer);
}

const ggml_tensor * llama_get_embeddings_layer_inp_device(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings_layer_inp_device();
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

""",
        ),
        (
            "src/llama-context.h",
            r"""    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

""",
            r"""    float * get_embeddings_nextn();
    float * get_embeddings_nextn_ith(int32_t i);

    float * get_embeddings_layer_inp(uint32_t lid);
    const ggml_tensor * get_embeddings_layer_inp_device();

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

""",
        ),
        (
            "src/llama-context.h",
            r"""
    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

""",
            r"""
    void set_embeddings (bool value);
    void set_embeddings_nextn(bool value, bool masked);
    void set_embeddings_layer_inp(uint32_t lid, bool enable);
    void set_embeddings_layer_inp_device(bool enable);
    void embeddings_layer_inp_consumed(llama_context * consumer);
    void set_nextn_layer_offset(int32_t offset);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

""",
        ),
        (
            "src/llama-context.h",
            r"""    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

""",
            r"""    // populated when cparams.output_layer_inp[il] is true
    std::vector<buffer_view<float>> embd_layer_inp;
    std::vector<int32_t> embd_batch_idxs; // extracted index -> original batch index

    bool                    layer_inp_device = false;
    ggml_context_ptr        ctx_layer_inp_dev;
    ggml_context_ptr        ctx_layer_inp_views;
    ggml_backend_buffer_ptr buf_layer_inp_dev;
    ggml_tensor *           t_layer_inp_dev = nullptr;
    ggml_backend_event_ptr  layer_inp_event;
    bool                    layer_inp_event_pending = false;

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

""",
        ),
        (
            "src/llama-ext.h",
            r"""// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
""",
            r"""// mirrors:
// LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);
LLAMA_API float * llama_get_embeddings_layer_inp(struct llama_context * ctx, uint32_t lid);

LLAMA_API void llama_set_embeddings_layer_inp_device(struct llama_context * ctx, bool value);

LLAMA_API const struct ggml_tensor * llama_get_embeddings_layer_inp_device(struct llama_context * ctx);

LLAMA_API void llama_embeddings_layer_inp_consumed(struct llama_context * ctx, struct llama_context * consumer);

LLAMA_API int32_t llama_batch_ext_add_embd_device(struct llama_batch_ext * batch, llama_seq_id seq_id);

LLAMA_API void llama_batch_ext_set_embd_device(struct llama_batch_ext * batch, const struct ggml_tensor * src, int64_t row0);

LLAMA_API bool llama_model_dflash_device_features(const struct llama_model * model);

LLAMA_API llama_context * llama_get_ctx_other(struct llama_context * ctx);

//
// model/context data extraction
""",
        ),
        (
            "src/llama-graph.cpp",
            r"""    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
""",
            r"""    }

    if (ubatch->embd && embd && !ubatch->is_mixed()) {
        GGML_ASSERT(n_embd == embd->ne[0]);
        GGML_ASSERT(ubatch->embd_dev == nullptr && "this graph does not read device embedding rows");

        const int64_t n_tokens = ubatch->n_tokens;

        ggml_backend_tensor_set(embd, ubatch->embd, 0, n_tokens*n_embd*ggml_element_size(embd));
""",
        ),
        (
            "src/llama-graph.cpp",
            r"""    }
    return n;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
""",
            r"""    }
    return n;
}

bool llm_graph_input_embd_dev::can_reuse(const llm_graph_params & params) {
    return params.ubatch.embd_dev == src && params.ubatch.embd_dev_row0 == row0;
}

bool llm_graph_input_embd::can_reuse(const llm_graph_params & params) {
    bool res = true;

    res &= params.ubatch.embd_dev == nullptr;

    res &= (!params.ubatch.token) || (tokens && tokens->ne[0] == params.ubatch.n_tokens);
    res &= (!params.ubatch.embd)  || (embd   &&   embd->ne[1] == params.ubatch.n_tokens);
    res &= (!mixed_tokens) || mixed_tokens->ne[0] == llm_graph_n_tok_rows(params.ubatch);
    res &= (!mixed_embd)   || mixed_embd->ne[1]   == params.ubatch.n_tokens;
""",
        ),
        (
            "src/llama-graph.h",
            r"""};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;
""",
            r"""};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd_dev : public llm_graph_input_i {
public:
    llm_graph_input_embd_dev(const ggml_tensor * src, int64_t row0) : src(src), row0(row0) {}
    virtual ~llm_graph_input_embd_dev() = default;

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->embd_dev == src && ubatch->embd_dev_row0 == row0);
    }

    bool can_reuse(const llm_graph_params & params) override;

    const ggml_tensor * src;
    const int64_t row0;
};

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;
""",
        ),
        (
            "src/llama-model.cpp",
            r"""int32_t llama_model_dflash_selector_top_k(const llama_model * model) {
    return model->hparams.dflash_selector_top_k;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

""",
            r"""int32_t llama_model_dflash_selector_top_k(const llama_model * model) {
    return model->hparams.dflash_selector_top_k;
}

bool llama_model_dflash_device_features(const llama_model * model) {
    const auto & hparams = model->hparams;
    return model->arch == LLM_ARCH_DFLASH && hparams.n_lora_kv == 0 && hparams.dsv4_hc_mult == 0 && hparams.lilicorr_candidate_topk == 0;
}

int32_t llama_model_n_head(const llama_model * model) {
    return model->hparams.n_head();
}

""",
        ),
        (
            "src/models/dflash.cpp",
            r"""
    // KV cache injection
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
        ggml_set_input(inp->embd);

        ggml_tensor * inp_target = inp->embd;
        cb(inp_target, "inp_target_features", -1);

        res->add_input(std::move(inp));

        // fuse the target features through the encoder
        ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);
        inp_g = build_norm(inp_g, model.output_norm_enc->type==GGML_TYPE_F32?model.output_norm_enc:ggml_cast(ctx0,model.output_norm_enc,GGML_TYPE_F32), NULL, LLM_NORM_RMS, -1);
""",
            r"""
    // KV cache injection
    ASSERT_EMBD_OR_TOKEN(ubatch);
    if (ubatch.embd) {
        ggml_tensor * inp_target = nullptr;

        if (ubatch.embd_dev) {
            ggml_tensor * src = const_cast<ggml_tensor *>(ubatch.embd_dev);
            GGML_ASSERT(src->type == GGML_TYPE_F32 && src->ne[0] * src->ne[2] == n_embd_inp);
            GGML_ASSERT(ubatch.embd_dev_row0 >= 0 && ubatch.embd_dev_row0 + n_tokens <= src->ne[1]);

            ggml_tensor * rows = ggml_view_3d(ctx0, src, src->ne[0], n_tokens, src->ne[2], src->nb[1], src->nb[2], ubatch.embd_dev_row0 * src->nb[1]);
            inp_target = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, rows, 0, 2, 1, 3)), n_embd_inp, n_tokens);

            res->add_input(std::make_unique<llm_graph_input_embd_dev>(ubatch.embd_dev, ubatch.embd_dev_row0));
        } else {
            auto inp = std::make_unique<llm_graph_input_embd>(n_embd_inp);

            inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_inp, n_tokens);
            ggml_set_input(inp->embd);

            inp_target = inp->embd;

            res->add_input(std::move(inp));
        }
        cb(inp_target, "inp_target_features", -1);

        // fuse the target features through the encoder
        ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);
        inp_g = build_norm(inp_g, model.output_norm_enc->type==GGML_TYPE_F32?model.output_norm_enc:ggml_cast(ctx0,model.output_norm_enc,GGML_TYPE_F32), NULL, LLM_NORM_RMS, -1);
""",
        ),
    ),
    "round_profile_nvtx_v1": (
        (
            "src/CMakeLists.txt",
            r"""target_link_libraries(llama PUBLIC ggml)
""",
            r"""target_link_libraries(llama PUBLIC ggml)
find_package(CUDAToolkit REQUIRED)
target_include_directories(llama PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
""",
        ),
        (
            "ggml/src/ggml-cuda/ggml-cuda.cu",
            r"""#include "ggml-cuda.h"
#include "ggml-impl.h"
""",
            r"""#include "ggml-cuda.h"
#include "ggml-impl.h"
#include <cstdio>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>

static bool bitdraft_nvtx_nodes() {
    static const bool enabled = getenv("BITDRAFT_NVTX_NODES") != nullptr;
    return enabled;
}

static void bitdraft_nvtx_push_node(const ggml_tensor * node, int index, const char * kind) {
    char label[4096];
    int used = snprintf(label, sizeof(label), "%s|%d|%s|%s|%lld,%lld,%lld,%lld", kind, index, ggml_op_desc(node), node->name,
        (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2], (long long) node->ne[3]);
    for (int j = 0; j < GGML_MAX_SRC && used > 0 && used < (int) sizeof(label); ++j) {
        const ggml_tensor * src = node->src[j];
        if (src == nullptr) {
            continue;
        }
        const int weight = src->buffer != nullptr && ggml_backend_buffer_get_usage(src->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
        used += snprintf(label + used, sizeof(label) - used, "|%d:%s:%s:%lld,%lld,%lld,%lld:%zu:%d", j, src->name, ggml_type_name(src->type),
            (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2], (long long) src->ne[3], ggml_nbytes(src), weight);
    }
    nvtxRangePushA(label);
}
""",
        ),
        (
            "ggml/src/ggml-cuda/ggml-cuda.cu",
            r"""                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
""",
            r"""                if (bitdraft_nvtx_nodes()) {
                    bitdraft_nvtx_push_node(node, i, "node");
                }
                int nodes_to_skip = ggml_cuda_try_fuse(cuda_ctx, cgraph, i);
                if (bitdraft_nvtx_nodes() && nodes_to_skip != 0) {
                    for (int fused = 1; fused <= nodes_to_skip; ++fused) {
                        bitdraft_nvtx_push_node(cgraph->nodes[i + fused], i + fused, "fused");
                        nvtxRangePop();
                    }
                    nvtxRangePop();
                }
""",
        ),
        (
            "ggml/src/ggml-cuda/ggml-cuda.cu",
            r"""                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
""",
            r"""                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (bitdraft_nvtx_nodes()) {
                    nvtxRangePop();
                }
""",
        ),
        (
            "src/llama-context.cpp",
            r"""#include "llama-context.h"
""",
            r"""#include "llama-context.h"
#include <nvtx3/nvToolsExt.h>
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    const auto gparams = graph_params(res, ubatch, mctx, gtype);
""",
            r"""    const auto gparams = graph_params(res, ubatch, mctx, gtype);
    nvtxRangePushA("llama.graph");
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    // set the input data for the input tensors
    {
""",
            r"""    nvtxRangePop();
    nvtxRangePushA("llama.inputs");
    {
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
""",
            r"""    nvtxRangePop();
    nvtxRangePushA("llama.compute");
    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    nvtxRangePop();
""",
        ),
        (
            "src/llama-context.cpp",
            r"""    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens""",
            r"""    nvtxRangePushA("llama.sync");
    ggml_backend_sched_synchronize(sched.get());
    nvtxRangePop();

    // FIXME: if multiple single tokens""",
        ),
        (
            "common/speculative.cpp",
            r"""#include "speculative.h"
""",
            r"""#include "speculative.h"
#include <nvtx3/nvToolsExt.h>

namespace {
struct bitdraft_range {
    explicit bitdraft_range(const char * label) { nvtxRangePushA(label); }
    ~bitdraft_range() { nvtxRangePop(); }
    bitdraft_range(const bitdraft_range &) = delete;
    bitdraft_range & operator=(const bitdraft_range &) = delete;
};
}
""",
        ),
        (
            "common/speculative.cpp",
            r"""            LOG_WRN("%s: llama_process returned %d\n", __func__, ret);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
""",
            r"""            LOG_WRN("%s: llama_process returned %d\n", __func__, ret);
            return;
        }

        bitdraft_range bitdraft_select("dft.select");
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_block_beg[seq_id] < 0) {
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""#include "server-context.h"
""",
            r"""#include "server-context.h"
#include <nvtx3/nvToolsExt.h>
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
""",
            r"""            nvtxRangePushA("srv.draft");
            queue_tasks.yield_to_queue([&]() {
                common_speculative_draft(spec.get());
            });
            nvtxRangePop();
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        int ret = 0;
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });
""",
            r"""        int ret = 0;
        nvtxRangePushA(("srv.target|" + std::to_string(batch.view.size())).c_str());
        queue_tasks.yield_to_queue([&]() {
            ret = llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.view.get());
            if (ret == 0 && has_output) {
                llama_synchronize(ctx_tgt);
            }
        });
        nvtxRangePop();
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });
""",
            r"""            nvtxRangePushA("srv.process");
            queue_tasks.yield_to_queue([&]() {
                ok = common_speculative_process(spec.get(), batch.view);
            });
            nvtxRangePop();
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        // make checkpoints if needed
""",
            r"""        nvtxRangePushA("srv.checkpoint");
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        // update the batch with the sampled/drafted tokens
""",
            r"""        nvtxRangePop();
""",
        ),
        (
            "tools/server/server-context.cpp",
            r"""        // speculative decoding - main model sample and accept
""",
            r"""        nvtxRangePushA("srv.verify");
        struct bitdraft_verify_end {
            ~bitdraft_verify_end() { nvtxRangePop(); }
        } bitdraft_verify;
""",
        ),
    ),
}

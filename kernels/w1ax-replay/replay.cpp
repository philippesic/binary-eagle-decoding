#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

template <typename T> static T read_le(std::istream & input) {
    std::array<unsigned char, sizeof(T)> bytes{};
    input.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    require(bool(input), "truncated capture header");
    T value = 0;
    for (size_t i = 0; i < bytes.size(); ++i) value |= T(bytes[i]) << (8*i);
    return value;
}

static std::string escape_json(const std::string & input) {
    std::string result;
    for (unsigned char c : input) {
        if (c == '"' || c == '\\') { result.push_back('\\'); result.push_back(char(c)); }
        else if (c >= 32 && c < 127) result.push_back(char(c));
        else result += "?";
    }
    return result;
}

struct capture {
    fs::path path;
    uint64_t sequence, k, m, n;
    uint32_t source_bits;
    std::string name;
    std::vector<float> activations;
};

static capture read_capture(const fs::path & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    require(bool(input), "cannot open capture: " + path.string());
    const auto length = input.tellg();
    input.seekg(0);
    char magic[8];
    input.read(magic, sizeof(magic));
    require(bool(input) && std::memcmp(magic, "W1AXACT1", 8) == 0, "invalid capture magic: " + path.string());
    capture c;
    c.path = path;
    c.sequence = read_le<uint64_t>(input);
    c.k = read_le<uint64_t>(input);
    c.m = read_le<uint64_t>(input);
    c.n = read_le<uint64_t>(input);
    c.source_bits = read_le<uint32_t>(input);
    char name[128];
    input.read(name, sizeof(name));
    require(bool(input) && std::memchr(name, 0, sizeof(name)), "invalid capture weight name");
    c.name = name;
    require(c.k > 0 && c.m > 0 && c.n > 0 && c.k <= 1000000 && c.m <= 1000000 && c.n <= 65535,
            "invalid capture dimensions");
    require(c.source_bits == 1 || c.source_bits == 4 || c.source_bits == 8 || c.source_bits == 16,
            "invalid source precision");
    require(c.n <= SIZE_MAX / c.k / sizeof(float), "capture activation size overflow");
    const size_t bytes = size_t(c.n*c.k*sizeof(float));
    require(length == std::streamoff(8 + 4*8 + 4 + 128 + bytes), "capture byte length mismatch: " + path.string());
    c.activations.resize(size_t(c.n*c.k));
    input.read(reinterpret_cast<char *>(c.activations.data()), bytes);
    require(bool(input), "truncated activation data");
    for (float value : c.activations) require(std::isfinite(value), "nonfinite activation in " + path.string());
    return c;
}

struct gguf_deleter { void operator()(gguf_context * p) const { if (p) gguf_free(p); } };
struct ggml_deleter { void operator()(ggml_context * p) const { if (p) ggml_free(p); } };
struct backend_deleter { void operator()(ggml_backend_t p) const { if (p) ggml_backend_free(p); } };
struct buffer_deleter { void operator()(ggml_backend_buffer_t p) const { if (p) ggml_backend_buffer_free(p); } };

class weight_file {
public:
    explicit weight_file(const fs::path & path) : path_(path), input_(path, std::ios::binary),
        meta_(gguf_init_from_file(path.string().c_str(), {true, nullptr})) {
        require(bool(input_) && bool(meta_), "cannot read GGUF: " + path.string());
    }

    void load(const capture & c, std::vector<uint32_t> & words, std::vector<float> & scales) {
        const auto packed_id = gguf_find_tensor(meta_.get(), c.name.c_str());
        require(packed_id >= 0 && gguf_get_tensor_type(meta_.get(), packed_id) == GGML_TYPE_I32,
                "missing I32 packed tensor: " + c.name);
        const std::string suffix = ".w1a1_packed";
        require(c.name.size() > suffix.size() && c.name.compare(c.name.size()-suffix.size(), suffix.size(), suffix) == 0,
                "capture name is not a packed W1A1 tensor: " + c.name);
        const std::string scale_name = c.name.substr(0, c.name.size()-suffix.size()) + ".w1a1_scale";
        const auto scale_id = gguf_find_tensor(meta_.get(), scale_name.c_str());
        require(scale_id >= 0 && gguf_get_tensor_type(meta_.get(), scale_id) == GGML_TYPE_F32,
                "missing F32 row scales: " + scale_name);
        const int64_t * wne = gguf_get_tensor_ne(meta_.get(), packed_id);
        const int64_t * sne = gguf_get_tensor_ne(meta_.get(), scale_id);
        const uint64_t count = (c.k + 31)/32;
        require(wne[0] == int64_t(count) && wne[1] == int64_t(c.m) && wne[2] == 1 && wne[3] == 1 &&
                sne[0] == int64_t(c.m) && sne[1] == 1 && sne[2] == 1 && sne[3] == 1,
                "capture/GGUF tensor shape mismatch: " + c.name);
        words.resize(size_t(count*c.m));
        scales.resize(size_t(c.m));
        read_tensor(packed_id, words.data(), words.size()*sizeof(uint32_t));
        read_tensor(scale_id, scales.data(), scales.size()*sizeof(float));
        for (float scale : scales) require(std::isfinite(scale), "nonfinite GGUF row scale");
    }

    std::vector<uint8_t> load_anchor(const capture & c, ggml_type expected_type) {
        const std::string suffix = ".w1a1_packed";
        require(c.name.size() > suffix.size() && c.name.compare(c.name.size()-suffix.size(), suffix.size(), suffix) == 0,
                "capture name is not a packed W1A1 tensor: " + c.name);
        const std::string dense_name = c.name.substr(0, c.name.size()-suffix.size()) + ".weight";
        const auto id = gguf_find_tensor(meta_.get(), dense_name.c_str());
        require(id >= 0, "anchor GGUF missing tensor: " + dense_name);
        const auto type = gguf_get_tensor_type(meta_.get(), id);
        require(type == expected_type, "anchor GGUF type mismatch for " + dense_name + ": expected " +
                ggml_type_name(expected_type) + ", got " + ggml_type_name(type));
        const int64_t * shape = gguf_get_tensor_ne(meta_.get(), id);
        require(shape[0] == int64_t(c.k) && shape[1] == int64_t(c.m) && shape[2] == 1 && shape[3] == 1,
                "anchor GGUF shape mismatch: " + dense_name);
        require(c.k % ggml_blck_size(type) == 0, "anchor K is not divisible by its quantization block size");
        const size_t row_bytes = ggml_row_size(type, int64_t(c.k));
        require(c.m <= SIZE_MAX/row_bytes, "anchor tensor size overflow");
        std::vector<uint8_t> data(size_t(c.m)*row_bytes);
        read_tensor(id, data.data(), data.size());
        return data;
    }

private:
    void read_tensor(int64_t id, void * data, size_t bytes) {
        require(gguf_get_tensor_size(meta_.get(), id) == bytes, "GGUF tensor byte size mismatch");
        const auto offset = gguf_get_data_offset(meta_.get()) + gguf_get_tensor_offset(meta_.get(), id);
        input_.clear();
        input_.seekg(std::streamoff(offset));
        input_.read(static_cast<char *>(data), bytes);
        require(bool(input_), "truncated GGUF tensor data");
    }
    fs::path path_;
    std::ifstream input_;
    std::unique_ptr<gguf_context, gguf_deleter> meta_;
};

static std::vector<size_t> sample_rows(size_t m, size_t requested) {
    std::vector<size_t> rows;
    if (requested == 0 || requested >= m) {
        rows.resize(m);
        std::iota(rows.begin(), rows.end(), 0);
        return rows;
    }
    rows.reserve(requested + 2);
    for (size_t i = 0; i < requested; ++i) rows.push_back(i*(m-1)/std::max<size_t>(1, requested-1));
    rows.push_back(0);
    rows.push_back(m-1);
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}

static float reference(const capture & c, const std::vector<uint32_t> & w,
        const std::vector<float> & scales, int bits, size_t token, size_t row) {
    const size_t words = (c.k+31)/32;
    const float * x = c.activations.data() + token*c.k;
    double abs_sum = 0.0;
    float absmax = 0.0f;
    for (size_t i = 0; i < c.k; ++i) {
        abs_sum += std::abs(double(x[i]));
        absmax = std::max(absmax, std::abs(x[i]));
    }
    const int qmax = bits == 8 ? 127 : 7;
    const float inv = absmax == 0 ? 0 : float(qmax)/absmax;
    if (bits == 4 || bits == 8) require(std::isfinite(inv), "activation absmax reciprocal overflow");
    int32_t dot = 0;
    float half_dot = 0.0f;
    for (size_t i = 0; i < c.k; ++i) {
        const int sign = (w[row*words + i/32] & (uint32_t(1) << (i%32))) ? 1 : -1;
        if (bits == 1) dot += sign * (x[i] >= 0 ? 1 : -1);
        else if (bits == 16) {
            const float h = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x[i]));
            half_dot += sign > 0 ? h : -h;
        } else {
            const float scaled = x[i]*inv;
            const int q = std::max(-qmax, std::min(qmax, int(std::nearbyint(scaled))));
            dot += sign*q;
        }
    }
    const float first = (bits == 16 ? half_dot : float(dot))*scales[row];
    if (bits == 16) return first;
    const float act_scale = bits == 1 ? float(abs_sum/double(c.k)) : absmax/float(qmax);
    return first*act_scale;
}

struct options {
    fs::path gguf, captures;
    fs::path fp16_gguf, q8_gguf, q4_gguf;
    std::string backend = "gpu";
    int warmups = 2, samples = 5;
    size_t check_rows = 8, limit = 0;
    bool require_nine = false;
};

static options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--require-nine") { o.require_nine = true; continue; }
        require(i+1 < argc, "missing value after " + arg);
        const std::string value = argv[++i];
        if (arg == "--gguf") o.gguf = value;
        else if (arg == "--fp16-gguf") o.fp16_gguf = value;
        else if (arg == "--q8-gguf") o.q8_gguf = value;
        else if (arg == "--q4-gguf") o.q4_gguf = value;
        else if (arg == "--capture-dir") o.captures = value;
        else if (arg == "--backend") o.backend = value;
        else if (arg == "--warmups") o.warmups = std::stoi(value);
        else if (arg == "--samples") o.samples = std::stoi(value);
        else if (arg == "--check-rows") o.check_rows = std::stoull(value);
        else if (arg == "--limit") o.limit = std::stoull(value);
        else throw std::runtime_error("unknown argument: " + arg);
    }
    require(!o.gguf.empty() && !o.captures.empty() && fs::is_regular_file(o.gguf) && fs::is_directory(o.captures),
            "supply existing --gguf and --capture-dir");
    require(o.backend == "cpu" || o.backend == "gpu", "--backend must be cpu or gpu");
    for (const auto & path : {o.fp16_gguf, o.q8_gguf, o.q4_gguf}) {
        require(path.empty() || fs::is_regular_file(path), "anchor GGUF path does not exist: " + path.string());
    }
    require(o.warmups >= 0 && o.samples > 0 && o.samples <= 10000, "invalid warmups or samples");
    return o;
}

static std::string group_of(const std::string & name) {
    if (name.rfind("output.", 0) == 0) return "head";
    if (name.rfind("fc.", 0) == 0) return "fusion";
    if (name.find(".attn_") != std::string::npos) return "attention";
    if (name.find(".ffn_") != std::string::npos) return "ffn";
    return "unknown";
}

struct activation_diagnostics {
    size_t source_zeros = 0, code_zeros = 0, clipped = 0, saturated = 0;
    double mae = 0, rmse = 0, max_abs_error = 0;
};

static activation_diagnostics diagnose_activations(const capture & c, int bits) {
    activation_diagnostics d;
    double error_sum = 0, error_squared_sum = 0;
    for (size_t token = 0; token < c.n; ++token) {
        const float * x = c.activations.data() + token*c.k;
        double abs_sum = 0;
        float absmax = 0;
        for (size_t i = 0; i < c.k; ++i) {
            abs_sum += std::abs(double(x[i]));
            absmax = std::max(absmax, std::abs(x[i]));
        }
        const int qmax = bits == 8 ? 127 : 7;
        const float inv = absmax == 0 ? 0 : float(qmax)/absmax;
        if (bits == 4 || bits == 8) require(std::isfinite(inv), "activation absmax reciprocal overflow");
        const float scale = bits == 1 ? float(abs_sum/double(c.k)) : absmax/float(qmax);
        for (size_t i = 0; i < c.k; ++i) {
            if (x[i] == 0) ++d.source_zeros;
            float reconstructed;
            if (bits == 16) {
                reconstructed = ggml_fp16_to_fp32(ggml_fp32_to_fp16(x[i]));
                if (!std::isfinite(reconstructed)) ++d.clipped;
                if (reconstructed == 0) ++d.code_zeros;
            } else if (bits == 1) {
                reconstructed = (x[i] >= 0 ? 1.0f : -1.0f)*scale;
            } else {
                const int raw = int(std::nearbyint(x[i]*inv));
                if (raw < -qmax || raw > qmax) ++d.clipped;
                const int code = std::max(-qmax, std::min(qmax, raw));
                if (code == 0) ++d.code_zeros;
                if (std::abs(code) == qmax) ++d.saturated;
                reconstructed = float(code)*scale;
            }
            // An overflowing FP16 cast is a distinct clipping failure; it
            // cannot yield a meaningful finite reconstruction error.
            if (!std::isfinite(reconstructed)) continue;
            const double error = std::abs(double(reconstructed)-double(x[i]));
            error_sum += error;
            error_squared_sum += error*error;
            d.max_abs_error = std::max(d.max_abs_error, error);
        }
    }
    const double count = double(c.k*c.n);
    d.mae = error_sum/count;
    d.rmse = std::sqrt(error_squared_sum/count);
    return d;
}

static std::vector<float> replay(const capture & c, const std::vector<uint32_t> & w, const std::vector<float> & scales,
        ggml_backend_t backend, int bits, const options & opt) {
    ggml_init_params params = {2*1024*1024, nullptr, true};
    std::unique_ptr<ggml_context, ggml_deleter> ctx(ggml_init(params));
    require(bool(ctx), "ggml context allocation failed");
    auto * wt = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, (c.k+31)/32, c.m);
    auto * st = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, c.m);
    auto * at = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.k, c.n);
    ggml_set_name(wt, c.name.c_str());
    auto * out = ggml_w1ax_mul_mat(ctx.get(), wt, st, at, c.k, bits);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    std::unique_ptr<ggml_backend_buffer, buffer_deleter> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    require(bool(buffer), "backend tensor allocation failed");
    ggml_backend_tensor_set(wt, w.data(), 0, w.size()*sizeof(uint32_t));
    ggml_backend_tensor_set(st, scales.data(), 0, scales.size()*sizeof(float));
    ggml_backend_tensor_set(at, c.activations.data(), 0, c.activations.size()*sizeof(float));
    auto compute = [&]() {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "native ggml graph failed");
        ggml_backend_synchronize(backend);
    };
    compute();
    std::vector<float> actual(size_t(c.m*c.n));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));
    const auto rows = sample_rows(size_t(c.m), opt.check_rows);
    double max_abs = 0, max_rel = 0;
    size_t failures = 0;
    for (size_t token = 0; token < c.n; ++token) for (size_t row : rows) {
        const float expected = reference(c, w, scales, bits, token, row);
        const float got = actual[token*c.m + row];
        const double delta = std::abs(double(got)-double(expected));
        max_abs = std::max(max_abs, delta);
        max_rel = std::max(max_rel, delta/(1+std::abs(double(expected))));
        if (!std::isfinite(got) || delta > 0.003 + 0.0002*std::abs(double(expected))) ++failures;
    }
    require(failures == 0, "scalar parity failed for " + c.path.string() + " bits=" + std::to_string(bits));
    const activation_diagnostics diagnostic = diagnose_activations(c, bits);
    for (int i = 0; i < opt.warmups; ++i) compute();
    std::vector<double> us;
    us.reserve(opt.samples);
    for (int i = 0; i < opt.samples; ++i) {
        const auto start = std::chrono::steady_clock::now();
        compute();
        const auto end = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(end-start).count());
    }
    std::vector<double> sorted = us;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) { return sorted[size_t(std::ceil(p*(sorted.size()-1)))]; };
    const double activation_count = double(c.k*c.n);
    std::cout << std::setprecision(9) << "{\"record_type\":\"operator_replay\",\"capture\":\"" << escape_json(c.path.filename().string())
              << "\",\"sequence\":" << c.sequence << ",\"name\":\"" << escape_json(c.name)
              << "\",\"group\":\"" << group_of(c.name) << "\",\"K\":" << c.k << ",\"M\":" << c.m
              << ",\"N\":" << c.n << ",\"source_bits\":" << c.source_bits << ",\"replay_bits\":" << bits
              << ",\"backend_type\":\"" << opt.backend
              << "\",\"backend\":\"" << escape_json(ggml_backend_dev_description(ggml_backend_get_device(backend)))
              << "\",\"timing\":\"synchronized_host_wall_full_ggml_graph_us\",\"checked_rows\":" << rows.size()
              << ",\"checked_outputs\":" << rows.size()*c.n << ",\"max_abs_error\":" << max_abs
              << ",\"max_rel_error\":" << max_rel << ",\"min_us\":" << sorted.front()
              << ",\"median_us\":" << percentile(0.5) << ",\"p95_us\":" << percentile(0.95)
              << ",\"activation\":{\"elements\":" << size_t(c.k*c.n)
              << ",\"source_zero_rate\":" << diagnostic.source_zeros/activation_count
              << ",\"code_zero_rate\":";
    if (bits == 1) std::cout << "null";
    else std::cout << diagnostic.code_zeros/activation_count;
    std::cout << ",\"clip_rate\":";
    if (bits == 1) std::cout << "null";
    else std::cout << diagnostic.clipped/activation_count;
    std::cout << ",\"saturation_rate\":";
    if (bits == 1 || bits == 16) std::cout << "null";
    else std::cout << diagnostic.saturated/activation_count;
    std::cout << ",\"mae\":" << diagnostic.mae << ",\"rmse\":" << diagnostic.rmse
              << ",\"max_abs_error\":" << diagnostic.max_abs_error << "},\"samples_us\":[";
    for (size_t i = 0; i < us.size(); ++i) std::cout << (i ? "," : "") << us[i];
    std::cout << "]}" << std::endl;
    return c.name == "output.w1a1_packed" ? std::move(actual) : std::vector<float>{};
}

static std::vector<size_t> top_indices(const float * scores, size_t rows, size_t count) {
    std::vector<size_t> indices(rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin()+count, indices.end(), [&](size_t a, size_t b) {
        return scores[a] == scores[b] ? a < b : scores[a] > scores[b];
    });
    indices.resize(count);
    return indices;
}

static void compare_head(const capture & c, const std::array<std::vector<float>, 4> & outputs) {
    require(c.m == 32000, "head comparison requires the full 32,000-row output tensor");
    for (const auto & output : outputs) {
        require(output.size() == size_t(c.m*c.n), "incomplete head output vector");
        for (float score : output) require(std::isfinite(score), "nonfinite head score");
    }
    constexpr std::array<int, 4> bits = {16, 8, 4, 1};
    for (size_t token = 0; token < c.n; ++token) {
        const float * reference = outputs[0].data() + token*c.m;
        const auto ref_top = top_indices(reference, size_t(c.m), 10);
        for (size_t mode = 1; mode < bits.size(); ++mode) {
            const float * candidate = outputs[mode].data() + token*c.m;
            const auto candidate_top = top_indices(candidate, size_t(c.m), 10);
            const auto overlap = [&](size_t k) {
                size_t matches = 0;
                for (size_t i = 0; i < k; ++i) {
                    if (std::find(candidate_top.begin(), candidate_top.begin()+k, ref_top[i]) != candidate_top.begin()+k) ++matches;
                }
                return matches;
            };
            std::cout << std::setprecision(9) << "{\"record_type\":\"head_comparison\",\"capture\":\""
                      << escape_json(c.path.filename().string()) << "\",\"sequence\":" << c.sequence
                      << ",\"token\":" << token << ",\"rows\":" << c.m
                      << ",\"reference_bits\":16,\"candidate_bits\":" << bits[mode]
                      << ",\"reference\":\"same_binary_weights_w1a16\",\"top1_agree\":"
                      << (ref_top[0] == candidate_top[0] ? "true" : "false")
                      << ",\"topk_set_overlap\":{\"1\":" << overlap(1)
                      << ",\"5\":" << overlap(5) << ",\"10\":" << overlap(10)
                      << "},\"reference_top1_id\":" << ref_top[0]
                      << ",\"candidate_top1_id\":" << candidate_top[0]
                      << ",\"reference_top1_margin\":" << double(reference[ref_top[0]])-reference[ref_top[1]]
                      << ",\"candidate_top1_margin\":" << double(candidate[candidate_top[0]])-candidate[candidate_top[1]]
                      << ",\"reference_top1_score\":" << reference[ref_top[0]]
                      << ",\"candidate_score_at_reference_top1\":" << candidate[ref_top[0]]
                      << ",\"candidate_top1_score\":" << candidate[candidate_top[0]]
                      << ",\"reference_score_at_candidate_top1\":" << reference[candidate_top[0]]
                      << "}" << std::endl;
        }
    }
}

static void replay_anchor(const capture & c, const std::vector<uint8_t> & weights,
        ggml_type type, const char * format, ggml_backend_t backend, const options & opt) {
    ggml_init_params params = {2*1024*1024, nullptr, true};
    std::unique_ptr<ggml_context, ggml_deleter> ctx(ggml_init(params));
    require(bool(ctx), "ggml context allocation failed");
    auto * wt = ggml_new_tensor_2d(ctx.get(), type, c.k, c.m);
    auto * at = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.k, c.n);
    auto * out = ggml_mul_mat(ctx.get(), wt, at);
    auto * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    std::unique_ptr<ggml_backend_buffer, buffer_deleter> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    require(bool(buffer), "anchor backend tensor allocation failed");
    require(ggml_nbytes(wt) == weights.size(), "anchor GGUF/backend tensor byte size mismatch");
    ggml_backend_tensor_set(wt, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(at, c.activations.data(), 0, c.activations.size()*sizeof(float));
    auto compute = [&]() {
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "native anchor ggml graph failed");
        ggml_backend_synchronize(backend);
    };
    compute();
    std::vector<float> actual(size_t(c.m*c.n));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));
    for (float value : actual) require(std::isfinite(value), "anchor output contains nonfinite values");

    // This is a diagnostic reference, not a parity gate: quantized ggml CUDA
    // matmul may quantize F32 activations to Q8_1, whereas this uses the exact
    // captured F32 inputs and dequantized *weight rows* outside the timed span.
    const auto * traits = ggml_get_type_traits(type);
    require(traits && traits->to_float, "anchor type lacks a row dequantizer");
    const size_t row_bytes = ggml_row_size(type, int64_t(c.k));
    std::vector<float> dense_row(size_t(c.k));
    const auto rows = sample_rows(size_t(c.m), opt.check_rows);
    double ref_max_abs = 0, ref_max_rel = 0;
    for (size_t row : rows) {
        traits->to_float(weights.data()+row*row_bytes, dense_row.data(), int64_t(c.k));
        for (float value : dense_row) require(std::isfinite(value), "anchor dequantized weight contains nonfinite value");
        for (size_t token = 0; token < c.n; ++token) {
            const float * activation = c.activations.data()+token*c.k;
            double sum = 0;
            for (size_t i = 0; i < c.k; ++i) sum += double(dense_row[i])*double(activation[i]);
            require(std::isfinite(sum), "anchor scalar reference contains nonfinite value");
            const double delta = std::abs(double(actual[token*c.m+row])-sum);
            ref_max_abs = std::max(ref_max_abs, delta);
            ref_max_rel = std::max(ref_max_rel, delta/(1+std::abs(sum)));
        }
    }

    for (int i = 0; i < opt.warmups; ++i) compute();
    std::vector<double> us;
    us.reserve(opt.samples);
    for (int i = 0; i < opt.samples; ++i) {
        const auto start = std::chrono::steady_clock::now();
        compute();
        const auto end = std::chrono::steady_clock::now();
        us.push_back(std::chrono::duration<double, std::micro>(end-start).count());
    }
    std::vector<double> sorted = us;
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&](double p) { return sorted[size_t(std::ceil(p*(sorted.size()-1)))]; };
    std::cout << std::setprecision(9) << "{\"record_type\":\"anchor_operator_replay\",\"capture\":\""
              << escape_json(c.path.filename().string()) << "\",\"sequence\":" << c.sequence
              << ",\"name\":\"" << escape_json(c.name) << "\",\"group\":\"" << group_of(c.name)
              << "\",\"K\":" << c.k << ",\"M\":" << c.m << ",\"N\":" << c.n
              << ",\"source_bits\":" << c.source_bits << ",\"anchor_format\":\"" << format
              << "\",\"weight_type\":\"" << ggml_type_name(type) << "\",\"backend_type\":\"" << opt.backend
              << "\",\"backend\":\"" << escape_json(ggml_backend_dev_description(ggml_backend_get_device(backend)))
              << "\",\"timing\":\"synchronized_host_wall_full_ggml_graph_us\",\"finite_outputs\":" << actual.size()
              << ",\"validation\":\"finite_output_gate_with_sampled_f32_activation_reference_non_gate\""
              << ",\"reference_rows\":" << rows.size() << ",\"reference_outputs\":" << rows.size()*c.n
              << ",\"reference_max_abs_error\":" << ref_max_abs << ",\"reference_max_rel_error\":" << ref_max_rel
              << ",\"min_us\":" << sorted.front() << ",\"median_us\":" << percentile(0.5)
              << ",\"p95_us\":" << percentile(0.95) << ",\"samples_us\":[";
    for (size_t i = 0; i < us.size(); ++i) std::cout << (i ? "," : "") << us[i];
    std::cout << "]}" << std::endl;
}

int main(int argc, char ** argv) {
    try {
        const options opt = parse(argc, argv);
        require(!std::getenv("GGML_W1AX_CAPTURE_DIR"),
                "unset GGML_W1AX_CAPTURE_DIR before replay; capture synchronizes and corrupts timing");
        ggml_backend_load_all();
        std::unique_ptr<ggml_backend, backend_deleter> backend(
                ggml_backend_init_by_type(opt.backend == "gpu" ? GGML_BACKEND_DEVICE_TYPE_GPU : GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
        require(bool(backend), "requested ggml backend unavailable");
        weight_file gguf(opt.gguf);
        std::unique_ptr<weight_file> fp16_anchor, q8_anchor, q4_anchor;
        if (!opt.fp16_gguf.empty()) fp16_anchor = std::make_unique<weight_file>(opt.fp16_gguf);
        if (!opt.q8_gguf.empty()) q8_anchor = std::make_unique<weight_file>(opt.q8_gguf);
        if (!opt.q4_gguf.empty()) q4_anchor = std::make_unique<weight_file>(opt.q4_gguf);
        std::vector<fs::path> paths;
        for (const auto & entry : fs::directory_iterator(opt.captures)) {
            if (entry.is_regular_file() && entry.path().extension() == ".bin" &&
                    entry.path().filename().string().rfind("op-", 0) == 0) paths.push_back(entry.path());
        }
        std::sort(paths.begin(), paths.end());
        require(!paths.empty(), "capture directory contains no op-*.bin files");
        if (opt.limit && paths.size() > opt.limit) paths.resize(opt.limit);
        std::set<std::string> seen_names;
        for (const auto & path : paths) {
            const capture c = read_capture(path);
            seen_names.insert(c.name);
            std::vector<uint32_t> weights;
            std::vector<float> scales;
            gguf.load(c, weights, scales);
            std::array<std::vector<float>, 4> head_outputs;
            size_t mode = 0;
            for (int bits : {16, 8, 4, 1}) {
                head_outputs[mode++] = replay(c, weights, scales, backend.get(), bits, opt);
            }
            if (c.name == "output.w1a1_packed") compare_head(c, head_outputs);
            if (fp16_anchor) replay_anchor(c, fp16_anchor->load_anchor(c, GGML_TYPE_F16), GGML_TYPE_F16,
                    "fp16", backend.get(), opt);
            if (q8_anchor) replay_anchor(c, q8_anchor->load_anchor(c, GGML_TYPE_Q8_0), GGML_TYPE_Q8_0,
                    "q8_0", backend.get(), opt);
            if (q4_anchor) replay_anchor(c, q4_anchor->load_anchor(c, GGML_TYPE_Q4_0), GGML_TYPE_Q4_0,
                    "q4_0", backend.get(), opt);
        }
        if (opt.require_nine) {
            const std::set<std::string> expected = {
                "fc.w1a1_packed", "output.w1a1_packed", "blk.0.attn_q.w1a1_packed",
                "blk.0.attn_k.w1a1_packed", "blk.0.attn_v.w1a1_packed", "blk.0.attn_output.w1a1_packed",
                "blk.0.ffn_gate.w1a1_packed", "blk.0.ffn_down.w1a1_packed", "blk.0.ffn_up.w1a1_packed",
            };
            require(seen_names == expected, "capture set does not cover exactly the nine EAGLE packed tensors");
        }
    } catch (const std::exception & error) {
        std::cerr << "w1ax_operator_replay: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}

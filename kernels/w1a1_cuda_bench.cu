#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "w1a1_cuda.cuh"

namespace {

void check(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t elements) : elements_(elements) {
    check(cudaMalloc(reinterpret_cast<void **>(&data_), elements * sizeof(T)),
          "cudaMalloc");
  }
  ~DeviceBuffer() {
    if (data_)
      cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  T *get() const { return data_; }
  void upload(const std::vector<T> &source) {
    if (source.size() != elements_)
      throw std::runtime_error("upload size mismatch");
    check(cudaMemcpy(data_, source.data(), elements_ * sizeof(T),
                     cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");
  }
  std::vector<T> download() const {
    std::vector<T> result(elements_);
    check(cudaMemcpy(result.data(), data_, elements_ * sizeof(T),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");
    return result;
  }

private:
  size_t elements_;
  T *data_ = nullptr;
};

bool nonnegative(float value) { return value >= 0.0f; }

float mean_abs(const float *row, int k) {
  double sum = 0.0;
  for (int feature = 0; feature < k; ++feature)
    sum += std::fabs(row[feature]);
  return static_cast<float>(sum / k);
}

uint32_t valid_mask(int k) {
  return (k & 31) == 0 ? UINT32_MAX : (uint32_t{1} << (k & 31)) - 1;
}

std::vector<uint32_t> cpu_pack(const std::vector<float> &matrix, int rows,
                               int k) {
  const int words = w1a1::words_per_row(k);
  std::vector<uint32_t> packed(static_cast<size_t>(rows) * words, 0);
  for (int row = 0; row < rows; ++row) {
    for (int feature = 0; feature < k; ++feature) {
      if (nonnegative(matrix[static_cast<size_t>(row) * k + feature])) {
        packed[static_cast<size_t>(row) * words + feature / 32] |=
            uint32_t{1} << (feature % 32);
      }
    }
  }
  return packed;
}

int cpu_dot(const float *weight, const float *activation, int k) {
  int dot = 0;
  for (int feature = 0; feature < k; ++feature) {
    dot += nonnegative(weight[feature]) == nonnegative(activation[feature])
               ? 1
               : -1;
  }
  return dot;
}

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

// Test data include all-zero rows, both signed zeros, alternating signs,
// random values, and values adjacent to zero. The CPU dot uses source signs
// directly rather than reusing the CUDA packed-bit calculation.
void correctness_case(int k) {
  constexpr int rows = 7;
  constexpr int tokens = 3;
  const int words = w1a1::words_per_row(k);
  std::mt19937 generator(0x51a1u + static_cast<unsigned>(k));
  std::uniform_real_distribution<float> random(-3.0f, 3.0f);
  std::vector<float> weights(static_cast<size_t>(rows) * k);
  std::vector<float> activations(static_cast<size_t>(tokens) * k);
  for (int row = 0; row < rows; ++row) {
    for (int feature = 0; feature < k; ++feature) {
      float value = random(generator);
      if (row == 0)
        value = 0.0f;
      if (row == 1)
        value = feature & 1 ? -0.0f : +0.0f;
      if (row == 2)
        value = feature & 1 ? -0.75f : +0.75f;
      if (row == 3 && feature % 17 == 0)
        value = -0.0f;
      if (row == 4 && feature % 19 == 0)
        value = +0.0f;
      if (row == 5 && feature % 23 == 0)
        value = -std::numeric_limits<float>::denorm_min();
      weights[static_cast<size_t>(row) * k + feature] = value;
    }
  }
  for (int token = 0; token < tokens; ++token) {
    for (int feature = 0; feature < k; ++feature) {
      float value = random(generator);
      if (token == 0)
        value = 0.0f;
      if (token == 1)
        value = feature & 1 ? -0.0f : +0.0f;
      if (token == 2 && feature % 11 == 0)
        value = -0.0f;
      if (token == 2 && feature % 29 == 0)
        value = -std::numeric_limits<float>::denorm_min();
      activations[static_cast<size_t>(token) * k + feature] = value;
    }
  }

  std::vector<float> weight_scales(rows);
  std::vector<float> activation_scales(tokens);
  for (int row = 0; row < rows; ++row)
    weight_scales[row] =
        mean_abs(weights.data() + static_cast<size_t>(row) * k, k);
  for (int token = 0; token < tokens; ++token)
    activation_scales[token] =
        mean_abs(activations.data() + static_cast<size_t>(token) * k, k);
  const auto cpu_weights = cpu_pack(weights, rows, k);
  const auto cpu_activations = cpu_pack(activations, tokens, k);

  DeviceBuffer<float> d_input(activations.size());
  DeviceBuffer<uint32_t> d_weights(cpu_weights.size());
  DeviceBuffer<uint32_t> d_activations(cpu_activations.size());
  DeviceBuffer<float> d_weight_scales(weight_scales.size());
  DeviceBuffer<float> d_activation_scales(activation_scales.size());
  DeviceBuffer<float> d_partials(static_cast<size_t>(tokens) * words);
  DeviceBuffer<int32_t> d_dots(static_cast<size_t>(tokens) * rows);
  DeviceBuffer<float> d_output(static_cast<size_t>(tokens) * rows);
  d_input.upload(activations);
  d_weights.upload(cpu_weights);
  d_weight_scales.upload(weight_scales);
  check(w1a1::pack_activations(d_input.get(), tokens, k, d_activations.get(),
                               d_activation_scales.get(), d_partials.get(), 0),
        "pack_activations");
  check(cudaDeviceSynchronize(), "pack synchronize");
  require(d_activations.download() == cpu_activations,
          "activation bits differ for K=" + std::to_string(k));
  const auto device_scales = d_activation_scales.download();
  for (int token = 0; token < tokens; ++token) {
    const float expected = activation_scales[token];
    const float tolerance = 2.0e-7f + 5.0e-5f * std::fabs(expected);
    require(std::fabs(device_scales[token] - expected) <= tolerance,
            "activation mean_abs differs for K=" + std::to_string(k));
  }

  // Dirty padding must not alter any integer dot. Deliberately make the
  // invalid bits of weights and activations disagree in the last word.
  if ((k & 31) != 0) {
    auto dirty_weights = cpu_weights;
    auto dirty_activations = cpu_activations;
    for (int row = 0; row < rows; ++row)
      dirty_weights[static_cast<size_t>(row) * words + words - 1] |=
          ~valid_mask(k);
    for (int token = 0; token < tokens; ++token)
      dirty_activations[static_cast<size_t>(token) * words + words - 1] &=
          valid_mask(k);
    d_weights.upload(dirty_weights);
    d_activations.upload(dirty_activations);
  }
  check(w1a1::binary_linear(d_weights.get(), d_weight_scales.get(),
                            d_activations.get(), d_activation_scales.get(),
                            rows, tokens, k, d_dots.get(), d_output.get(), 0),
        "binary_linear");
  check(cudaDeviceSynchronize(), "linear synchronize");
  const auto dots = d_dots.download();
  const auto output = d_output.download();
  for (int token = 0; token < tokens; ++token) {
    for (int row = 0; row < rows; ++row) {
      const size_t index = static_cast<size_t>(token) * rows + row;
      const int expected_dot =
          cpu_dot(weights.data() + static_cast<size_t>(row) * k,
                  activations.data() + static_cast<size_t>(token) * k, k);
      require(dots[index] == expected_dot,
              "integer dot differs at K=" + std::to_string(k) + ", row=" +
                  std::to_string(row) + ", token=" + std::to_string(token));
      const float weighted =
          static_cast<float>(expected_dot) * weight_scales[row];
      const float expected = weighted * activation_scales[token];
      // The pack reduction has a different F32 summation tree than the
      // CPU double sum. Its relative error is bounded here at 5e-5;
      // 2e-4 absolute covers results near zero and the last multiply.
      const float tolerance = 2.0e-4f + 5.0e-5f * std::fabs(expected);
      require(std::fabs(output[index] - expected) <= tolerance,
              "scaled output differs at K=" + std::to_string(k) + ", row=" +
                  std::to_string(row) + ", token=" + std::to_string(token));
    }
  }
}

struct Shape {
  const char *name;
  int rows;
  int k;
};
constexpr Shape shapes[] = {
    {"fusion", 2560, 7680},      {"attention_q", 4096, 5120},
    {"attention_k", 1024, 5120}, {"attention_v", 1024, 5120},
    {"attention_o", 2560, 4096}, {"ffn_gate", 9728, 2560},
    {"ffn_up", 9728, 2560},      {"ffn_down", 2560, 9728},
    {"head", 32000, 2560},
};

struct Timing {
  float minimum;
  float median;
  float p95;
  std::vector<float> samples;
};

Timing summarize(std::vector<float> samples) {
  auto sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  return {sorted.front(), sorted[sorted.size() / 2],
          sorted[(sorted.size() * 95 + 99) / 100 - 1], std::move(samples)};
}

template <typename Operation>
float measure_once(Operation operation, cudaEvent_t start, cudaEvent_t stop) {
  check(cudaEventRecord(start), "cudaEventRecord start");
  operation();
  check(cudaEventRecord(stop), "cudaEventRecord stop");
  check(cudaEventSynchronize(stop), "cudaEventSynchronize");
  float milliseconds = 0.0f;
  check(cudaEventElapsedTime(&milliseconds, start, stop),
        "cudaEventElapsedTime");
  return milliseconds;
}

template <typename Packed, typename Inclusive>
std::pair<Timing, Timing> time_pair(Packed packed, Inclusive inclusive,
                                    int warmups, int samples) {
  for (int index = 0; index < warmups; ++index) {
    packed();
    inclusive();
  }
  check(cudaDeviceSynchronize(), "warmup synchronize");
  cudaEvent_t start = nullptr, stop = nullptr;
  check(cudaEventCreate(&start), "cudaEventCreate start");
  try {
    check(cudaEventCreate(&stop), "cudaEventCreate stop");
    std::vector<float> packed_times, inclusive_times;
    packed_times.reserve(samples);
    inclusive_times.reserve(samples);
    for (int index = 0; index < samples; ++index) {
      if (index & 1) {
        inclusive_times.push_back(measure_once(inclusive, start, stop));
        packed_times.push_back(measure_once(packed, start, stop));
      } else {
        packed_times.push_back(measure_once(packed, start, stop));
        inclusive_times.push_back(measure_once(inclusive, start, stop));
      }
    }
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    return {summarize(std::move(packed_times)),
            summarize(std::move(inclusive_times))};
  } catch (...) {
    if (stop)
      cudaEventDestroy(stop);
    cudaEventDestroy(start);
    throw;
  }
}

struct BenchmarkResult {
  Shape shape;
  int tokens;
  Timing packed;
  Timing inclusive;
};

BenchmarkResult benchmark_shape(Shape shape, int tokens, int warmups,
                                int samples) {
  const int words = w1a1::words_per_row(shape.k);
  std::mt19937 generator(0x5080u + static_cast<unsigned>(shape.k + shape.rows));
  std::uniform_real_distribution<float> activation_random(-1.0f, 1.0f);
  std::uniform_real_distribution<float> scale_random(0.05f, 0.5f);
  std::vector<uint32_t> weights(static_cast<size_t>(shape.rows) * words);
  std::vector<float> weight_scales(shape.rows);
  std::vector<float> activations(static_cast<size_t>(tokens) * shape.k);
  for (auto &word : weights)
    word = generator();
  for (int row = 0; row < shape.rows; ++row)
    weights[static_cast<size_t>(row) * words + words - 1] &=
        valid_mask(shape.k);
  for (auto &scale : weight_scales)
    scale = scale_random(generator);
  for (auto &value : activations)
    value = activation_random(generator);

  DeviceBuffer<uint32_t> d_weights(weights.size());
  DeviceBuffer<float> d_weight_scales(weight_scales.size());
  DeviceBuffer<float> d_input(activations.size());
  DeviceBuffer<uint32_t> d_activations(static_cast<size_t>(tokens) * words);
  DeviceBuffer<float> d_activation_scales(tokens);
  DeviceBuffer<float> d_partials(static_cast<size_t>(tokens) * words);
  DeviceBuffer<float> d_output(static_cast<size_t>(tokens) * shape.rows);
  d_weights.upload(weights);
  d_weight_scales.upload(weight_scales);
  d_input.upload(activations);
  check(w1a1::pack_activations(d_input.get(), tokens, shape.k,
                               d_activations.get(), d_activation_scales.get(),
                               d_partials.get(), 0),
        "initial pack");
  check(cudaDeviceSynchronize(), "initial pack synchronize");
  const auto linear = [&] {
    check(w1a1::binary_linear(d_weights.get(), d_weight_scales.get(),
                              d_activations.get(), d_activation_scales.get(),
                              shape.rows, tokens, shape.k, nullptr,
                              d_output.get(), 0),
          "timed binary_linear");
  };
  const auto pack_and_linear = [&] {
    check(w1a1::pack_activations(d_input.get(), tokens, shape.k,
                                 d_activations.get(), d_activation_scales.get(),
                                 d_partials.get(), 0),
          "timed pack");
    linear();
  };
  auto [packed, inclusive] =
      time_pair(linear, pack_and_linear, warmups, samples);
  return {shape, tokens, std::move(packed), std::move(inclusive)};
}

void print_timing(const Timing &timing) {
  std::cout << "{\"min_ms\":" << timing.minimum
            << ",\"median_ms\":" << timing.median
            << ",\"p95_ms\":" << timing.p95 << ",\"samples_ms\":[";
  for (size_t index = 0; index < timing.samples.size(); ++index) {
    if (index)
      std::cout << ',';
    std::cout << timing.samples[index];
  }
  std::cout << "]}";
}

int positive_arg(const char *text, const char *name) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!*text || *end || value <= 0 || value > 100000)
    throw std::runtime_error(std::string(name) +
                             " must be a positive integer <= 100000");
  return static_cast<int>(value);
}

} // namespace

int main(int argc, char **argv) {
  try {
    int warmups = 10;
    int samples = 30;
    bool correctness_only = false;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--warmups" && index + 1 < argc)
        warmups = positive_arg(argv[++index], "warmups");
      else if (argument == "--samples" && index + 1 < argc)
        samples = positive_arg(argv[++index], "samples");
      else if (argument == "--correctness-only")
        correctness_only = true;
      else
        throw std::runtime_error("usage: w1a1_cuda_bench [--warmups N] "
                                 "[--samples N] [--correctness-only]");
    }
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device),
          "cudaGetDeviceProperties");
    int runtime = 0, driver = 0;
    check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    constexpr int widths[] = {31, 32, 33, 2560, 4096, 5120, 7680, 9728};
    for (int k : widths)
      correctness_case(k);
    std::cout << std::fixed << std::setprecision(6);
    std::cout
        << "{\"schema\":1,\"gpu\":\"" << properties.name
        << "\",\"compute_capability\":\"" << properties.major << '.'
        << properties.minor << "\",\"cuda_runtime\":" << runtime
        << ",\"cuda_driver\":" << driver
        << ",\"correctness\":{\"widths\":[31,32,33,2560,4096,5120,7680,9728],"
           "\"pairs_per_width\":21,\"integer_dots_exact\":true,"
           "\"max_scale_relative_tolerance\":0.00005,"
           "\"output_absolute_tolerance\":0.0002,"
           "\"output_relative_tolerance\":0.00005}"
        << ",\"warmups\":" << warmups << ",\"samples\":" << samples
        << ",\"timing_scope\":\"CUDA event GPU span on default stream; "
           "preallocated buffers; no H2D or cudaMalloc; alternating order; "
           "no graph replay\""
        << ",\"results\":[";
    if (!correctness_only) {
      bool first = true;
      for (const Shape &shape : shapes) {
        // The CUDA drafter profile observed 10-token FFN/head calls.
        // A single-token call shows the launch-bound end as well.
        for (int tokens : {1, 10}) {
          const auto result = benchmark_shape(shape, tokens, warmups, samples);
          if (!first)
            std::cout << ',';
          first = false;
          std::cout << "{\"shape\":\"" << result.shape.name
                    << "\",\"rows\":" << result.shape.rows
                    << ",\"k\":" << result.shape.k
                    << ",\"tokens\":" << result.tokens
                    << ",\"packed_weight_bytes\":"
                    << static_cast<size_t>(result.shape.rows) *
                           w1a1::words_per_row(result.shape.k) *
                           sizeof(uint32_t)
                    << ",\"activation_scratch_bytes\":"
                    << static_cast<size_t>(result.tokens) *
                           w1a1::words_per_row(result.shape.k) * sizeof(float)
                    << ",\"already_packed\":";
          print_timing(result.packed);
          std::cout << ",\"packing_inclusive\":";
          print_timing(result.inclusive);
          std::cout << '}';
        }
      }
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "w1a1_cuda_bench: " << error.what() << '\n';
    return 1;
  }
}

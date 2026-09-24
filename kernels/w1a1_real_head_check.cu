// Full-vocabulary numerical parity for a real captured EAGLE-3 head fixture.
// Build beside w1a1_cuda.cu; this program performs no benchmark timing.
#include "w1a1_cuda.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr char kMagic[8] = {'W', '1', 'A', '1', 'R', 'H', '\0', '\0'};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kHeaderBytes = 96;

void cuda_check(cudaError_t result, const char *action) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(result));
}

uint32_t read_u32(const std::vector<uint8_t> &file, size_t offset) {
  if (offset + 4 > file.size())
    throw std::runtime_error("truncated fixture header");
  return static_cast<uint32_t>(file[offset]) |
         (static_cast<uint32_t>(file[offset + 1]) << 8) |
         (static_cast<uint32_t>(file[offset + 2]) << 16) |
         (static_cast<uint32_t>(file[offset + 3]) << 24);
}

std::string hash_hex(const std::vector<uint8_t> &file, size_t offset) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (size_t i = 0; i < 32; ++i) {
    const uint8_t byte = file[offset + i];
    out.push_back(digits[byte >> 4]);
    out.push_back(digits[byte & 15]);
  }
  return out;
}

uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::runtime_error("fixture dimensions overflow");
  return a * b;
}

template <typename T>
std::vector<T> take(const std::vector<uint8_t> &file, size_t &offset, uint64_t count) {
  static_assert(sizeof(T) == 4, "fixture arrays contain four-byte elements");
  const uint64_t nbytes = multiply(count, sizeof(T));
  if (nbytes > file.size() - offset)
    throw std::runtime_error("truncated fixture payload");
  std::vector<T> out(static_cast<size_t>(count));
  std::memcpy(out.data(), file.data() + offset, static_cast<size_t>(nbytes));
  offset += static_cast<size_t>(nbytes);
  return out;
}

struct Fixture {
  int k;
  int rows;
  int tokens;
  int words;
  std::string capture_hash;
  std::string gguf_hash;
  std::vector<int32_t> selected;
  std::vector<uint32_t> weights;
  std::vector<float> weight_scales;
  std::vector<float> activations;
  std::vector<uint32_t> expected_packed;
  std::vector<float> expected_scales;
  std::vector<int32_t> expected_dots;
  std::vector<float> expected_outputs;
};

Fixture read_fixture(const std::string &path) {
  static_assert(sizeof(float) == 4 && sizeof(int32_t) == 4 && sizeof(uint32_t) == 4);
  if (std::numeric_limits<float>::is_iec559 == false)
    throw std::runtime_error("host float is not IEEE-754");
  const uint32_t endian_probe = 1;
  if (*reinterpret_cast<const uint8_t *>(&endian_probe) != 1)
    throw std::runtime_error("fixture requires a little-endian host");
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    throw std::runtime_error("could not open fixture");
  const auto length = stream.tellg();
  if (length < kHeaderBytes || length > static_cast<std::streamoff>(1ULL << 34))
    throw std::runtime_error("fixture size is invalid");
  std::vector<uint8_t> file(static_cast<size_t>(length));
  stream.seekg(0);
  if (!stream.read(reinterpret_cast<char *>(file.data()), length))
    throw std::runtime_error("could not read fixture");
  if (std::memcmp(file.data(), kMagic, sizeof(kMagic)) != 0 ||
      read_u32(file, 8) != kVersion || read_u32(file, 12) != kHeaderBytes)
    throw std::runtime_error("unsupported fixture magic, version, or header size");
  Fixture f{};
  f.k = static_cast<int>(read_u32(file, 16));
  f.rows = static_cast<int>(read_u32(file, 20));
  f.tokens = static_cast<int>(read_u32(file, 24));
  f.words = static_cast<int>(read_u32(file, 28));
  if (f.k <= 0 || f.rows <= 0 || f.tokens <= 0 || f.words != (f.k + 31) / 32 ||
      f.rows > 1000000 || f.tokens > 4096 || f.k > 1000000)
    throw std::runtime_error("fixture dimensions are invalid");
  f.capture_hash = hash_hex(file, 32);
  f.gguf_hash = hash_hex(file, 64);
  const uint64_t tw = multiply(f.tokens, f.words);
  const uint64_t rw = multiply(f.rows, f.words);
  const uint64_t tk = multiply(f.tokens, f.k);
  const uint64_t tr = multiply(f.tokens, f.rows);
  const uint64_t elements = static_cast<uint64_t>(f.tokens) + rw + f.rows + tk +
                            tw + f.tokens + tr + tr;
  const uint64_t expected_size = kHeaderBytes + multiply(elements, 4);
  if (expected_size != file.size())
    throw std::runtime_error("fixture file length does not match dimensions");
  size_t offset = kHeaderBytes;
  f.selected = take<int32_t>(file, offset, f.tokens);
  f.weights = take<uint32_t>(file, offset, rw);
  f.weight_scales = take<float>(file, offset, f.rows);
  f.activations = take<float>(file, offset, tk);
  f.expected_packed = take<uint32_t>(file, offset, tw);
  f.expected_scales = take<float>(file, offset, f.tokens);
  f.expected_dots = take<int32_t>(file, offset, tr);
  f.expected_outputs = take<float>(file, offset, tr);
  if (offset != file.size())
    throw std::runtime_error("fixture trailing bytes");
  for (float value : f.activations)
    if (!std::isfinite(value))
      throw std::runtime_error("fixture contains nonfinite input");
  return f;
}

template <typename T> struct DeviceBuffer {
  T *ptr = nullptr;
  explicit DeviceBuffer(size_t elements) {
    cuda_check(cudaMalloc(reinterpret_cast<void **>(&ptr), elements * sizeof(T)),
               "cudaMalloc");
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  ~DeviceBuffer() { cudaFree(ptr); }
  void upload(const std::vector<T> &host) {
    cuda_check(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T),
                          cudaMemcpyHostToDevice), "cudaMemcpy upload");
  }
  std::vector<T> download(size_t elements) const {
    std::vector<T> host(elements);
    cuda_check(cudaMemcpy(host.data(), ptr, elements * sizeof(T),
                          cudaMemcpyDeviceToHost), "cudaMemcpy download");
    return host;
  }
};

bool near(float actual, float expected, double absolute, double relative) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(static_cast<double>(actual) - expected) <=
             absolute + relative * std::abs(static_cast<double>(expected));
}

int run(const std::string &path) {
  const Fixture f = read_fixture(path);
  cudaDeviceProp device{};
  int device_id = 0;
  cuda_check(cudaGetDevice(&device_id), "cudaGetDevice");
  cuda_check(cudaGetDeviceProperties(&device, device_id), "cudaGetDeviceProperties");
  const size_t tw = static_cast<size_t>(f.tokens) * f.words;
  const size_t rw = static_cast<size_t>(f.rows) * f.words;
  const size_t tk = static_cast<size_t>(f.tokens) * f.k;
  const size_t tr = static_cast<size_t>(f.tokens) * f.rows;
  DeviceBuffer<float> input(tk), weight_scales(f.rows), activation_scales(f.tokens),
      partials(tw), output(tr);
  DeviceBuffer<uint32_t> weights(rw), packed(tw);
  DeviceBuffer<int32_t> dots(tr);
  input.upload(f.activations);
  weights.upload(f.weights);
  weight_scales.upload(f.weight_scales);
  cuda_check(w1a1::pack_activations(input.ptr, f.tokens, f.k, packed.ptr,
                                    activation_scales.ptr, partials.ptr, 0),
             "pack_activations launch");
  cuda_check(w1a1::binary_linear(weights.ptr, weight_scales.ptr, packed.ptr,
                                 activation_scales.ptr, f.rows, f.tokens, f.k,
                                 dots.ptr, output.ptr, 0),
             "binary_linear launch");
  cuda_check(cudaDeviceSynchronize(), "CUDA kernel execution");
  const auto got_packed = packed.download(tw);
  const auto got_scales = activation_scales.download(f.tokens);
  const auto got_dots = dots.download(tr);
  const auto got_output = output.download(tr);
  size_t pack_mismatches = 0, dot_mismatches = 0, scale_mismatches = 0,
         output_mismatches = 0;
  double max_scale_error = 0, max_output_error = 0;
  for (size_t i = 0; i < tw; ++i)
    pack_mismatches += got_packed[i] != f.expected_packed[i];
  for (size_t i = 0; i < f.expected_scales.size(); ++i) {
    const double error = std::abs(static_cast<double>(got_scales[i]) - f.expected_scales[i]);
    max_scale_error = std::max(max_scale_error, error);
    scale_mismatches += !near(got_scales[i], f.expected_scales[i], 2e-7, 5e-5);
  }
  for (size_t i = 0; i < tr; ++i) {
    dot_mismatches += got_dots[i] != f.expected_dots[i];
    const double error = std::abs(static_cast<double>(got_output[i]) - f.expected_outputs[i]);
    max_output_error = std::max(max_output_error, error);
    output_mismatches += !near(got_output[i], f.expected_outputs[i], 2e-4, 5e-5);
  }
  const bool pass = !(pack_mismatches || dot_mismatches || scale_mismatches ||
                      output_mismatches);
  std::cout << std::setprecision(9) << "{\"schema_version\":1,\"pass\":"
            << (pass ? "true" : "false") << ",\"gpu\":\"" << device.name
            << "\",\"compute_capability\":\"" << device.major << "." << device.minor
            << "\",\"k\":" << f.k << ",\"rows\":" << f.rows
            << ",\"tokens\":" << f.tokens << ",\"pack_mismatches\":"
            << pack_mismatches << ",\"dot_mismatches\":" << dot_mismatches
            << ",\"scale_mismatches\":" << scale_mismatches
            << ",\"output_mismatches\":" << output_mismatches
            << ",\"max_scale_abs_error\":" << max_scale_error
            << ",\"max_output_abs_error\":" << max_output_error
            << ",\"capture_sha256\":\"" << f.capture_hash
            << "\",\"gguf_sha256\":\"" << f.gguf_hash << "\"}" << std::endl;
  return pass ? 0 : 2;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: w1a1_real_head_check FIXTURE.bin\n";
    return 1;
  }
  try {
    return run(argv[1]);
  } catch (const std::exception &error) {
    std::cerr << "real-head check: " << error.what() << '\n';
    return 1;
  }
}

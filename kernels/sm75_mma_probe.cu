// Standalone SM75 binary Tensor Core correctness probe. No project GPU
// dispatch. Build: nvcc -std=c++17 -O2 -arch=sm_75 -o sm75_mma_probe
// kernels/sm75_mma_probe.cu The exact fragment mapping and opcode come from
// NVIDIA PTX ISA 8.5, sections 9.7.15.4.5 and 9.7.15.4.14 (CUDA 12.6
// documentation).

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 8;
constexpr int kBitsPerMma = 128;

void check(cudaError_t error, const char *operation) {
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(error));
  }
}

// A is row-major [rows, ceil(K/32)]. B is token-major [tokens, ceil(K/32)],
// corresponding to a column-major logical K x tokens matrix. One warp owns
// one 8 x 8 output tile. PTX lanes: group=lane/4, thread=lane%4; A has row
// group and K word 4*tile+thread, B has token group and the same K word.
// Accumulators belong to row group, columns 2*thread and 2*thread+1.
__global__ void binary_mma_probe(const uint32_t *a, const uint32_t *b,
                                 int *dots, int rows, int tokens, int k) {
  const int lane = threadIdx.x;
  const int group = lane >> 2;
  const int in_group = lane & 3;
  const int row = blockIdx.y * kTile + group;
  const int token_for_b = blockIdx.x * kTile + group;
  const int words = (k + 31) / 32;
  const int tiles_k = (k + kBitsPerMma - 1) / kBitsPerMma;
  int mismatch0 = 0;
  int mismatch1 = 0;

  // The loop bound and control path are uniform across all 32 warp lanes.
  // Invalid rows/tokens and all padding K bits supply zero operands; the
  // logical K (rather than padded length) converts the final count to a dot.
  for (int tile = 0; tile < tiles_k; ++tile) {
    const int word = 4 * tile + in_group;
    uint32_t a_word = 0;
    uint32_t b_word = 0;
    if (word < words) {
      const int remaining = k - 32 * word;
      const uint32_t valid_mask =
          remaining >= 32 ? 0xffffffffu : ((1u << remaining) - 1u);
      if (row < rows)
        a_word = a[row * words + word] & valid_mask;
      if (token_for_b < tokens) {
        b_word = b[token_for_b * words + word] & valid_mask;
      }
    }
    asm volatile("mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc "
                 "{%0, %1}, {%2}, {%3}, {%0, %1};"
                 : "+r"(mismatch0), "+r"(mismatch1)
                 : "r"(a_word), "r"(b_word));
  }

  const int token0 = blockIdx.x * kTile + 2 * in_group;
  if (row < rows && token0 < tokens) {
    dots[row * tokens + token0] = k - 2 * mismatch0;
  }
  if (row < rows && token0 + 1 < tokens) {
    dots[row * tokens + token0 + 1] = k - 2 * mismatch1;
  }
}

// Deterministic signs with signed zero and varied nonzero values. The CPU
// expectation below reads these dense values, never the packed GPU operands.
float value_at(int outer, int feature, uint32_t salt) {
  uint32_t x = static_cast<uint32_t>(outer + 1) * 0x9e3779b9u;
  x ^= static_cast<uint32_t>(feature + 1) * 0x85ebca6bu;
  x ^= salt;
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  if (x % 17u == 0u)
    return (x & 1u) ? -0.0f : 0.0f;
  return (x & 2u) ? -static_cast<float>((x % 11u) + 1u)
                  : static_cast<float>((x % 13u) + 1u);
}

std::vector<uint32_t> pack(const std::vector<float> &dense, int outer, int k) {
  const int words = (k + 31) / 32;
  std::vector<uint32_t> packed(static_cast<size_t>(outer) * words, 0);
  for (int row = 0; row < outer; ++row) {
    for (int feature = 0; feature < k; ++feature) {
      if (dense[static_cast<size_t>(row) * k + feature] >= 0.0f) {
        packed[static_cast<size_t>(row) * words + feature / 32] |=
            1u << (feature % 32);
      }
    }
  }
  return packed;
}

void run_case(int rows, int tokens, int k) {
  const int words = (k + 31) / 32;
  std::vector<float> weights(static_cast<size_t>(rows) * k);
  std::vector<float> activations(static_cast<size_t>(tokens) * k);
  for (int row = 0; row < rows; ++row) {
    for (int feature = 0; feature < k; ++feature) {
      weights[static_cast<size_t>(row) * k + feature] =
          value_at(row, feature, 0x12bc34deu);
    }
  }
  for (int token = 0; token < tokens; ++token) {
    for (int feature = 0; feature < k; ++feature) {
      activations[static_cast<size_t>(token) * k + feature] =
          value_at(token, feature, 0xfedc7654u);
    }
  }
  auto packed_a = pack(weights, rows, k);
  auto packed_b = pack(activations, tokens, k);
  // Deliberately dirty unused tail bits: the kernel must mask them before
  // MMA. A and B receive different dirt so accidental XOR is observable.
  if (k % 32 != 0) {
    const uint32_t tail = ~((1u << (k % 32)) - 1u);
    for (int row = 0; row < rows; ++row) {
      packed_a[static_cast<size_t>(row) * words + words - 1] |=
          tail & (row & 1 ? 0xaaaaaaaau : 0x55555555u);
    }
    for (int token = 0; token < tokens; ++token) {
      packed_b[static_cast<size_t>(token) * words + words - 1] |=
          tail & (token & 1 ? 0x33333333u : 0xccccccccu);
    }
  }

  uint32_t *device_a = nullptr;
  uint32_t *device_b = nullptr;
  int *device_dots = nullptr;
  try {
    check(cudaMalloc(reinterpret_cast<void **>(&device_a),
                     packed_a.size() * sizeof(uint32_t)),
          "cudaMalloc A");
    check(cudaMalloc(reinterpret_cast<void **>(&device_b),
                     packed_b.size() * sizeof(uint32_t)),
          "cudaMalloc B");
    check(cudaMalloc(reinterpret_cast<void **>(&device_dots),
                     static_cast<size_t>(rows) * tokens * sizeof(int)),
          "cudaMalloc dots");
    check(cudaMemcpy(device_a, packed_a.data(),
                     packed_a.size() * sizeof(uint32_t),
                     cudaMemcpyHostToDevice),
          "copy A");
    check(cudaMemcpy(device_b, packed_b.data(),
                     packed_b.size() * sizeof(uint32_t),
                     cudaMemcpyHostToDevice),
          "copy B");
    const dim3 grid((tokens + 7) / 8, (rows + 7) / 8);
    binary_mma_probe<<<grid, 32>>>(device_a, device_b, device_dots, rows,
                                   tokens, k);
    check(cudaGetLastError(), "binary_mma_probe launch");
    std::vector<int> actual(static_cast<size_t>(rows) * tokens);
    check(cudaMemcpy(actual.data(), device_dots, actual.size() * sizeof(int),
                     cudaMemcpyDeviceToHost),
          "copy dots");
    for (int row = 0; row < rows; ++row) {
      for (int token = 0; token < tokens; ++token) {
        int expected = 0;
        for (int feature = 0; feature < k; ++feature) {
          const bool a_sign =
              weights[static_cast<size_t>(row) * k + feature] >= 0.0f;
          const bool b_sign =
              activations[static_cast<size_t>(token) * k + feature] >= 0.0f;
          expected += a_sign == b_sign ? 1 : -1;
        }
        const int observed = actual[static_cast<size_t>(row) * tokens + token];
        if (observed != expected) {
          char message[192];
          std::snprintf(
              message, sizeof(message),
              "K=%d rows=%d tokens=%d output=(%d,%d): got %d expected %d", k,
              rows, tokens, row, token, observed, expected);
          throw std::runtime_error(message);
        }
      }
    }
  } catch (...) {
    cudaFree(device_dots);
    cudaFree(device_b);
    cudaFree(device_a);
    throw;
  }
  check(cudaFree(device_dots), "cudaFree dots");
  check(cudaFree(device_b), "cudaFree B");
  check(cudaFree(device_a), "cudaFree A");
  std::printf("PASS K=%d rows=%d tokens=%d outputs=%d\n", k, rows, tokens,
              rows * tokens);
}

} // namespace

int main() {
  try {
    cudaDeviceProp device{};
    check(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
    if (device.major != 7 || device.minor != 5) {
      throw std::runtime_error("runtime probe requires an SM75 GPU");
    }
    std::printf("SM75 device: %s\n", device.name);
    for (int k : {31, 32, 33, 128, 129, 2560, 4096, 5120, 7680, 9728}) {
      run_case(8, 8, k); // all 64 outputs of a full tile
      run_case(5, 3, k); // partial row and token tile
    }
    run_case(9, 10, 129); // second tile in both output dimensions
    std::puts(
        "All SM75 binary-MMA integer dots matched the dense CPU reference.");
    return 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
}

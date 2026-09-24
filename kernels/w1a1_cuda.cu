#include "w1a1_cuda.cuh"

#include <cmath>

namespace w1a1 {
namespace {

// One thread owns one word. Its partial absolute sum is reduced across all
// words of the token by scale_rows. The tail is never read or packed.
__global__ void pack_words(const float *__restrict__ input, int k, int words,
                           uint32_t *__restrict__ packed,
                           float *__restrict__ partials) {
  const int word = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = blockIdx.y;
  if (word >= words)
    return;

  uint32_t bits = 0;
  float absolute_sum = 0.0f;
  const int first = word * 32;
  const int remaining = k - first;
  const int end = remaining < 32 ? remaining : 32;
  const float *row = input + static_cast<size_t>(token) * k;
  for (int bit = 0; bit < end; ++bit) {
    const float value = row[first + bit];
    const uint32_t representation = __float_as_uint(value);
    // Inspect the sign bits so negative subnormals retain their sign even
    // if a caller later compiles arithmetic with flush-to-zero enabled.
    const bool positive_sign = (representation & 0x80000000u) == 0 ||
                               (representation & 0x7fffffffu) == 0;
    bits |= static_cast<uint32_t>(positive_sign) << bit;
    absolute_sum += fabsf(value);
  }
  const size_t offset = static_cast<size_t>(token) * words + word;
  packed[offset] = bits;
  partials[offset] = absolute_sum;
}

__global__ void scale_rows(const float *__restrict__ partials, int words, int k,
                           float *__restrict__ scales) {
  __shared__ float sums[256];
  const int token = blockIdx.x;
  float sum = 0.0f;
  for (int word = threadIdx.x; word < words; word += blockDim.x) {
    sum += partials[static_cast<size_t>(token) * words + word];
  }
  sums[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride)
      sums[threadIdx.x] += sums[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0)
    scales[token] = sums[0] / static_cast<float>(k);
}

// Four output rows per block, one warp per row. Each lane owns every 32nd
// packed word, then a warp reduction accumulates mismatches. This portable
// __popc path does not depend on binary tensor-core instructions.
__global__ void xor_popc_linear(const uint32_t *__restrict__ weights,
                                const float *__restrict__ weight_scales,
                                const uint32_t *__restrict__ activations,
                                const float *__restrict__ activation_scales,
                                int rows, int words, int k,
                                int32_t *__restrict__ integer_dots,
                                float *__restrict__ output) {
  const int row = blockIdx.x * 4 + threadIdx.y;
  const int token = blockIdx.y;
  if (row >= rows)
    return;

  const int lane = threadIdx.x;
  const uint32_t tail_mask =
      (k & 31) == 0 ? UINT32_MAX : (uint32_t{1} << (k & 31)) - 1;
  const size_t weight_offset = static_cast<size_t>(row) * words;
  const size_t activation_offset = static_cast<size_t>(token) * words;
  int mismatches = 0;
  for (int word = lane; word < words; word += 32) {
    const uint32_t mask = word == words - 1 ? tail_mask : UINT32_MAX;
    mismatches += __popc((weights[weight_offset + word] ^
                          activations[activation_offset + word]) &
                         mask);
  }
  for (int delta = 16; delta > 0; delta /= 2) {
    mismatches += __shfl_down_sync(0xffffffff, mismatches, delta);
  }
  if (lane == 0) {
    const int32_t dot = k - 2 * mismatches;
    const size_t index = static_cast<size_t>(token) * rows + row;
    if (integer_dots)
      integer_dots[index] = dot;
    const float weighted = static_cast<float>(dot) * weight_scales[row];
    output[index] = weighted * activation_scales[token];
  }
}

} // namespace

int words_per_row(int k) { return (k + 31) / 32; }

cudaError_t pack_activations(const float *input, int tokens, int k,
                             uint32_t *packed, float *scales,
                             float *scratch_partials, cudaStream_t stream) {
  if (!input || !packed || !scales || !scratch_partials || tokens <= 0 ||
      k <= 0)
    return cudaErrorInvalidValue;
  const int words = words_per_row(k);
  pack_words<<<dim3((words + 127) / 128, tokens), 128, 0, stream>>>(
      input, k, words, packed, scratch_partials);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
    return error;
  scale_rows<<<tokens, 256, 0, stream>>>(scratch_partials, words, k, scales);
  return cudaGetLastError();
}

cudaError_t binary_linear(const uint32_t *packed_weights,
                          const float *weight_scales,
                          const uint32_t *packed_activations,
                          const float *activation_scales, int rows, int tokens,
                          int k, int32_t *integer_dots, float *output,
                          cudaStream_t stream) {
  if (!packed_weights || !weight_scales || !packed_activations ||
      !activation_scales || !output || rows <= 0 || tokens <= 0 || k <= 0)
    return cudaErrorInvalidValue;
  xor_popc_linear<<<dim3((rows + 3) / 4, tokens), dim3(32, 4), 0, stream>>>(
      packed_weights, weight_scales, packed_activations, activation_scales,
      rows, words_per_row(k), k, integer_dots, output);
  return cudaGetLastError();
}

} // namespace w1a1

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace w1a1 {

// All matrices are row major. Packed rows contain ceil(k / 32) uint32 words;
// bit (feature % 32) is one for a nonnegative input, including either zero.
// Inputs are finite. The caller owns all device buffers and the stream.
int words_per_row(int k);

// scratch_partials has tokens * words_per_row(k) floats. Each activation row
// receives its F32 mean(abs(x)) over logical k, with no padded contribution.
cudaError_t pack_activations(const float *input, int tokens, int k,
                             uint32_t *packed, float *scales,
                             float *scratch_partials, cudaStream_t stream);

// Integer dots can be omitted by passing nullptr. Dirty high bits in the last
// word are masked, so only logical k contributes. Output is F32 in (token,row)
// order: (F32(dot) * F32(weight_scale)) * F32(activation_scale).
cudaError_t binary_linear(const uint32_t *packed_weights,
                          const float *weight_scales,
                          const uint32_t *packed_activations,
                          const float *activation_scales, int rows, int tokens,
                          int k, int32_t *integer_dots, float *output,
                          cudaStream_t stream);

} // namespace w1a1

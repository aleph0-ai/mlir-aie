//===- layer_norm.cc ------------------------------------------*- C++ -*-===//
//
// Generic LayerNormalization kernel for AIE2.
// Normalizes across the last dimension (cols).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#if defined(DTYPE_BF16)
using scalar_t = bfloat16;
#elif defined(DTYPE_F32)
using scalar_t = float;
#else
#error "Define DTYPE_BF16 or DTYPE_F32"
#endif

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

static inline float bits_to_float(int32_t bits) {
    union {
        int32_t i;
        float f;
    } u;
    u.i = bits;
    return u.f;
}

extern "C" {
void KERNEL_NAME(
    scalar_t *input,
    scalar_t *scale,
    scalar_t *bias,
    scalar_t *output,
    int32_t rows,
    int32_t cols,
    int32_t epsilon_bits) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    float epsilon = bits_to_float(epsilon_bits);
    for (int32_t r = 0; r < rows; ++r) {
        float mean = 0.0f;
        float var = 0.0f;
        int32_t base = r * cols;
        for (int32_t c = 0; c < cols; ++c) {
            mean += static_cast<float>(input[base + c]);
        }
        mean /= static_cast<float>(cols);
        for (int32_t c = 0; c < cols; ++c) {
            float diff = static_cast<float>(input[base + c]) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(cols);
        float denom = 1.0f / aie::sqrt(var + epsilon);
        for (int32_t c = 0; c < cols; ++c) {
            float x = static_cast<float>(input[base + c]);
            float y = (x - mean) * denom;
            float s = static_cast<float>(scale[c]);
            float b = static_cast<float>(bias[c]);
            output[base + c] = static_cast<scalar_t>(y * s + b);
        }
    }
}
}

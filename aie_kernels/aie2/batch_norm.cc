//===- batch_norm.cc ------------------------------------------*- C++ -*-===//
//
// Generic BatchNormalization kernel for AIE2 (NCHW).
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
    scalar_t *mean,
    scalar_t *var,
    scalar_t *output,
    int32_t batch,
    int32_t channels,
    int32_t height,
    int32_t width,
    int32_t epsilon_bits) {
    if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0) {
        return;
    }
    float epsilon = bits_to_float(epsilon_bits);
    int32_t hw = height * width;
    for (int32_t n = 0; n < batch; ++n) {
        for (int32_t c = 0; c < channels; ++c) {
            float mean_c = static_cast<float>(mean[c]);
            float var_c = static_cast<float>(var[c]);
            float scale_c = static_cast<float>(scale[c]);
            float bias_c = static_cast<float>(bias[c]);
            float denom = 1.0f / aie::sqrt(var_c + epsilon);
            int32_t base = (n * channels + c) * hw;
            for (int32_t idx = 0; idx < hw; ++idx) {
                float x = static_cast<float>(input[base + idx]);
                float y = (x - mean_c) * denom * scale_c + bias_c;
                output[base + idx] = static_cast<scalar_t>(y);
            }
        }
    }
}
}

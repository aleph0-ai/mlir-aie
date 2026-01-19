//===- clip_generic.cc -----------------------------------------*- C++ -*-===//
//
// Generic clip kernel for AIE2.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <string.h>

#include <aie_api/aie.hpp>

#if defined(DTYPE_BF16)
using scalar_t = bfloat16;
#elif defined(DTYPE_F16)
using scalar_t = _Float16;
#elif defined(DTYPE_F32)
using scalar_t = float;
#elif defined(DTYPE_I8)
using scalar_t = int8_t;
#else
#error "Define one of DTYPE_BF16/DTYPE_F16/DTYPE_F32/DTYPE_I8"
#endif

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

static inline float bits_to_float(int32_t bits) {
    uint32_t u = static_cast<uint32_t>(bits);
    float f;
    memcpy(&f, &u, sizeof(float));
    return f;
}

extern "C" {
void KERNEL_NAME(scalar_t *in, scalar_t *out, int32_t input_size, int32_t min_bits,
                 int32_t max_bits) {
    if (input_size <= 0) {
        return;
    }
#if defined(DTYPE_I8)
    int32_t min_val = min_bits;
    int32_t max_val = max_bits;
    if (min_val < -128) {
        min_val = -128;
    }
    if (max_val > 127) {
        max_val = 127;
    }
    for (int32_t i = 0; i < input_size; ++i) {
        int32_t v = static_cast<int32_t>(in[i]);
        if (v < min_val) {
            v = min_val;
        }
        if (v > max_val) {
            v = max_val;
        }
        out[i] = static_cast<scalar_t>(v);
    }
#else
    float min_val = bits_to_float(min_bits);
    float max_val = bits_to_float(max_bits);
    for (int32_t i = 0; i < input_size; ++i) {
        float v = static_cast<float>(in[i]);
        if (v < min_val) {
            v = min_val;
        }
        if (v > max_val) {
            v = max_val;
        }
        out[i] = static_cast<scalar_t>(v);
    }
#endif
}
}

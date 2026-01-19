//===- pad_generic.cc ------------------------------------------*- C++ -*-===//
//
// Generic pad kernel for AIE2 (rank-1/2, constant mode).
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

static inline scalar_t pad_value_from_i32(int32_t value) {
#if defined(DTYPE_I8)
    if (value > 127) {
        return static_cast<scalar_t>(127);
    }
    if (value < -128) {
        return static_cast<scalar_t>(-128);
    }
    return static_cast<scalar_t>(value);
#else
    uint32_t bits = static_cast<uint32_t>(value);
    float f;
    memcpy(&f, &bits, sizeof(float));
    return static_cast<scalar_t>(f);
#endif
}

extern "C" {
void KERNEL_NAME(scalar_t *in, scalar_t *out, int32_t in_rows, int32_t in_cols,
                 int32_t out_rows, int32_t out_cols, int32_t pad_top,
                 int32_t pad_left, int32_t pad_value) {
    if (out_rows <= 0 || out_cols <= 0 || in_rows <= 0 || in_cols <= 0) {
        return;
    }
    scalar_t pad_val = pad_value_from_i32(pad_value);
    for (int32_t r = 0; r < out_rows; ++r) {
        int32_t in_r = r - pad_top;
        for (int32_t c = 0; c < out_cols; ++c) {
            int32_t in_c = c - pad_left;
            if (in_r < 0 || in_c < 0 || in_r >= in_rows || in_c >= in_cols) {
                out[r * out_cols + c] = pad_val;
            } else {
                out[r * out_cols + c] = in[in_r * in_cols + in_c];
            }
        }
    }
}
}

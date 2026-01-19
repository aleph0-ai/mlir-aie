//===- concat_generic.cc ---------------------------------------*- C++ -*-===//
//
// Generic concat kernel for AIE2 (rank-1/2).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

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

extern "C" {
void KERNEL_NAME(scalar_t *in0, scalar_t *in1, scalar_t *out, int32_t in0_rows,
                 int32_t in0_cols, int32_t in1_rows, int32_t in1_cols,
                 int32_t out_rows, int32_t out_cols) {
#if defined(CONCAT_AXIS0)
    for (int32_t r = 0; r < in0_rows; ++r) {
        for (int32_t c = 0; c < in0_cols; ++c) {
            out[r * out_cols + c] = in0[r * in0_cols + c];
        }
    }
    for (int32_t r = 0; r < in1_rows; ++r) {
        int32_t out_r = r + in0_rows;
        for (int32_t c = 0; c < in1_cols; ++c) {
            out[out_r * out_cols + c] = in1[r * in1_cols + c];
        }
    }
#elif defined(CONCAT_AXIS1)
    for (int32_t r = 0; r < in0_rows; ++r) {
        for (int32_t c = 0; c < in0_cols; ++c) {
            out[r * out_cols + c] = in0[r * in0_cols + c];
        }
        for (int32_t c = 0; c < in1_cols; ++c) {
            int32_t out_c = c + in0_cols;
            out[r * out_cols + out_c] = in1[r * in1_cols + c];
        }
    }
#else
#error "Define CONCAT_AXIS0 or CONCAT_AXIS1"
#endif
}
}

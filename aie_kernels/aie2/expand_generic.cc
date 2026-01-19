//===- expand_generic.cc ---------------------------------------*- C++ -*-===//
//
// Generic expand kernel for AIE2 (rank-1/2).
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
void KERNEL_NAME(scalar_t *in, scalar_t *out, int32_t in_rows, int32_t in_cols,
                 int32_t out_rows, int32_t out_cols) {
    if (out_rows <= 0 || out_cols <= 0 || in_rows <= 0 || in_cols <= 0) {
        return;
    }
    for (int32_t r = 0; r < out_rows; ++r) {
        int32_t src_r = (in_rows == 1) ? 0 : r;
        for (int32_t c = 0; c < out_cols; ++c) {
            int32_t src_c = (in_cols == 1) ? 0 : c;
            out[r * out_cols + c] = in[src_r * in_cols + src_c];
        }
    }
}
}

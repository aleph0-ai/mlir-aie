//===- gather_generic.cc ---------------------------------------*- C++ -*-===//
//
// Generic gather kernel for AIE2 (rank-1/2, indices int32).
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
void KERNEL_NAME(scalar_t *in, int32_t *indices, scalar_t *out, int32_t in_rows,
                 int32_t in_cols, int32_t out_rows, int32_t out_cols) {
#if defined(GATHER_AXIS0)
    for (int32_t r = 0; r < out_rows; ++r) {
        int32_t idx = indices[r];
        for (int32_t c = 0; c < out_cols; ++c) {
            out[r * out_cols + c] = in[idx * in_cols + c];
        }
    }
#elif defined(GATHER_AXIS1)
    for (int32_t r = 0; r < out_rows; ++r) {
        for (int32_t c = 0; c < out_cols; ++c) {
            int32_t idx = indices[c];
            out[r * out_cols + c] = in[r * in_cols + idx];
        }
    }
#else
#error "Define GATHER_AXIS0 or GATHER_AXIS1"
#endif
}
}

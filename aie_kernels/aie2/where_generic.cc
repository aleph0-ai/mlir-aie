//===- where_generic.cc ---------------------------------------*- C++ -*-===//
//
// Generic Where kernels for AIE2 (rank-1/2).
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
void KERNEL_NAME(
    int8_t *cond,
    scalar_t *lhs,
    scalar_t *rhs,
    scalar_t *out,
    int32_t rows,
    int32_t cols) {
    if (rows <= 0 || cols <= 0) {
        return;
    }
    const int32_t total = rows * cols;
    for (int32_t i = 0; i < total; ++i) {
        out[i] = cond[i] ? lhs[i] : rhs[i];
    }
}
}

//===- arg_reduce_generic.cc ----------------------------------*- C++ -*-===//
//
// Generic ArgMax/ArgMin kernels for AIE2 (rank-1 only).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <math.h>

#include <aie_api/aie.hpp>

#if defined(DTYPE_BF16)
using scalar_t = bfloat16;
using acc_t = float;
#elif defined(DTYPE_F16)
using scalar_t = _Float16;
using acc_t = float;
#elif defined(DTYPE_F32)
using scalar_t = float;
using acc_t = float;
#elif defined(DTYPE_I8)
using scalar_t = int8_t;
using acc_t = int32_t;
#else
#error "Define one of DTYPE_BF16/DTYPE_F16/DTYPE_F32/DTYPE_I8"
#endif

#if defined(IDX_I32)
using index_t = int32_t;
#elif defined(IDX_I64)
using index_t = int64_t;
#else
#error "Define IDX_I32 or IDX_I64"
#endif

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

static inline acc_t to_acc(scalar_t x) {
#if defined(DTYPE_I8)
    return static_cast<acc_t>(x);
#else
    return static_cast<acc_t>(x);
#endif
}

extern "C" {
void KERNEL_NAME(scalar_t *in, index_t *out, int32_t input_size) {
    if (input_size <= 0) {
        return;
    }
    acc_t best = to_acc(in[0]);
    index_t best_idx = 0;
    for (int32_t i = 1; i < input_size; ++i) {
        acc_t v = to_acc(in[i]);
#if defined(OP_MAX)
        if (v > best) {
            best = v;
            best_idx = static_cast<index_t>(i);
        }
#elif defined(OP_MIN)
        if (v < best) {
            best = v;
            best_idx = static_cast<index_t>(i);
        }
#else
#error "Define OP_MAX or OP_MIN"
#endif
    }
    *out = best_idx;
}
}

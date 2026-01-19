//===- reduce_generic.cc ---------------------------------------*- C++ -*-===//
//
// Generic reduce kernels for AIE2.
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
#else
#error "Define one of DTYPE_BF16/DTYPE_F16/DTYPE_F32"
#endif

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

static inline acc_t to_acc(scalar_t x) {
    return static_cast<acc_t>(x);
}

static inline scalar_t from_acc(acc_t x) {
    return static_cast<scalar_t>(x);
}

extern "C" {
void KERNEL_NAME(scalar_t *in, scalar_t *out, int32_t input_size) {
    if (input_size <= 0) {
        return;
    }
#if defined(OP_SUM)
    acc_t acc = 0;
    for (int32_t i = 0; i < input_size; ++i) {
        acc += to_acc(in[i]);
    }
    *out = from_acc(acc);
#elif defined(OP_MEAN)
    acc_t acc = 0;
    for (int32_t i = 0; i < input_size; ++i) {
        acc += to_acc(in[i]);
    }
    acc = acc / static_cast<acc_t>(input_size);
    *out = from_acc(acc);
#elif defined(OP_MAX)
    acc_t acc = to_acc(in[0]);
    for (int32_t i = 1; i < input_size; ++i) {
        acc_t v = to_acc(in[i]);
        if (v > acc) {
            acc = v;
        }
    }
    *out = from_acc(acc);
#elif defined(OP_MIN)
    acc_t acc = to_acc(in[0]);
    for (int32_t i = 1; i < input_size; ++i) {
        acc_t v = to_acc(in[i]);
        if (v < acc) {
            acc = v;
        }
    }
    *out = from_acc(acc);
#elif defined(OP_PROD)
    acc_t acc = 1;
    for (int32_t i = 0; i < input_size; ++i) {
        acc *= to_acc(in[i]);
    }
    *out = from_acc(acc);
#else
#error "Define OP_SUM/OP_MEAN/OP_MAX/OP_MIN/OP_PROD"
#endif
}
}

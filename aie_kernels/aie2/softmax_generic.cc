//===- softmax_generic.cc --------------------------------------*- C++ -*-===//
//
// Generic softmax kernels for AIE2.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <math.h>

#include <aie_api/aie.hpp>

#if defined(DTYPE_F16)
using scalar_t = _Float16;
using acc_t = float;
#elif defined(DTYPE_F32)
using scalar_t = float;
using acc_t = float;
#else
#error "Define one of DTYPE_F16/DTYPE_F32"
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
    acc_t max_val = to_acc(in[0]);
    for (int32_t i = 1; i < input_size; ++i) {
        acc_t v = to_acc(in[i]);
        if (v > max_val) {
            max_val = v;
        }
    }
    acc_t sum = 0;
    for (int32_t i = 0; i < input_size; ++i) {
        acc_t v = to_acc(in[i]) - max_val;
        acc_t e = expf(v);
        out[i] = from_acc(e);
        sum += e;
    }
    if (sum == 0) {
        return;
    }
    acc_t inv_sum = 1.0f / sum;
    for (int32_t i = 0; i < input_size; ++i) {
        acc_t v = to_acc(out[i]) * inv_sum;
        out[i] = from_acc(v);
    }
}
}

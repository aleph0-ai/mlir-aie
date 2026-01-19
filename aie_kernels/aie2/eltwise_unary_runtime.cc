//===- eltwise_unary_runtime.cc --------------------------------*- C++ -*-===//
//
// Generic unary elementwise kernels with runtime element count for AIE2.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <math.h>

#include <aie_api/aie.hpp>

#if defined(DTYPE_BF16)
using scalar_t = bfloat16;
using acc_t = float;
#else
#error "Define DTYPE_BF16"
#endif

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

static inline acc_t to_acc(scalar_t x) { return static_cast<acc_t>(x); }

static inline scalar_t from_acc(acc_t x) { return static_cast<scalar_t>(x); }

static inline scalar_t unary_eval(scalar_t x) {
#if defined(OP_COPY)
    return x;
#elif defined(OP_ABS)
    acc_t v = to_acc(x);
    if (v < 0.0f) {
        v = -v;
    }
    return from_acc(v);
#elif defined(OP_NEG)
    return from_acc(-to_acc(x));
#elif defined(OP_RELU)
    acc_t v = to_acc(x);
    return from_acc(v < 0.0f ? 0.0f : v);
#elif defined(OP_EXP)
    return from_acc(expf(to_acc(x)));
#elif defined(OP_LOG)
    return from_acc(logf(to_acc(x)));
#elif defined(OP_SQRT)
    return from_acc(sqrtf(to_acc(x)));
#elif defined(OP_TANH)
    return from_acc(tanhf(to_acc(x)));
#elif defined(OP_SIGMOID)
    acc_t v = to_acc(x);
    return from_acc(1.0f / (1.0f + expf(-v)));
#elif defined(OP_ERF)
    return from_acc(erff(to_acc(x)));
#else
#error "Define a unary OP_* macro"
#endif
}

extern "C" {
void KERNEL_NAME(scalar_t *in, scalar_t *out, int32_t elements) {
    if (elements <= 0) {
        return;
    }
    for (int32_t i = 0; i < elements; ++i) {
        out[i] = unary_eval(in[i]);
    }
}
}

//===- eltwise_binary_runtime.cc ------------------------------*- C++ -*-===//
//
// Generic binary elementwise kernels with runtime element count for AIE2.
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

static inline scalar_t binary_eval(scalar_t a, scalar_t b) {
#if defined(OP_ADD)
    return from_acc(to_acc(a) + to_acc(b));
#elif defined(OP_SUB)
    return from_acc(to_acc(a) - to_acc(b));
#elif defined(OP_MUL)
    return from_acc(to_acc(a) * to_acc(b));
#elif defined(OP_DIV)
    return from_acc(to_acc(a) / to_acc(b));
#elif defined(OP_POW)
    return from_acc(powf(to_acc(a), to_acc(b)));
#else
#error "Define a binary OP_* macro"
#endif
}

extern "C" {
void KERNEL_NAME(scalar_t *a, scalar_t *b, scalar_t *out, int32_t elements) {
    if (elements <= 0) {
        return;
    }
    for (int32_t i = 0; i < elements; ++i) {
        out[i] = binary_eval(a[i], b[i]);
    }
}
}

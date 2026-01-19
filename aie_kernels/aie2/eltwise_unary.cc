//===- eltwise_unary.cc ----------------------------------------*- C++ -*-===//
//
// Generic unary elementwise kernels for AIE2.
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

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

#ifndef TILE_ELEMS
#define TILE_ELEMS 256
#endif

static inline acc_t to_acc(scalar_t x) {
#if defined(DTYPE_I8)
    return static_cast<acc_t>(x);
#else
    return static_cast<acc_t>(x);
#endif
}

static inline scalar_t from_acc(acc_t x) {
#if defined(DTYPE_I8)
    if (x > 127) {
        return static_cast<scalar_t>(127);
    }
    if (x < -128) {
        return static_cast<scalar_t>(-128);
    }
    return static_cast<scalar_t>(x);
#else
    return static_cast<scalar_t>(x);
#endif
}

static inline scalar_t unary_eval(scalar_t x) {
#if defined(OP_COPY)
    return x;
#elif defined(OP_ABS)
#if defined(DTYPE_I8)
    acc_t v = to_acc(x);
    if (v < 0) {
        v = -v;
    }
    return from_acc(v);
#else
    acc_t v = to_acc(x);
    if (v < 0.0f) {
        v = -v;
    }
    return from_acc(v);
#endif
#elif defined(OP_NEG)
#if defined(DTYPE_I8)
    return from_acc(-to_acc(x));
#else
    return from_acc(-to_acc(x));
#endif
#elif defined(OP_RELU)
#if defined(DTYPE_I8)
    acc_t v = to_acc(x);
    return from_acc(v < 0 ? 0 : v);
#else
    acc_t v = to_acc(x);
    return from_acc(v < 0.0f ? 0.0f : v);
#endif
#elif defined(OP_EXP)
#if defined(DTYPE_I8)
#error "OP_EXP not supported for I8"
#else
    return from_acc(expf(to_acc(x)));
#endif
#elif defined(OP_LOG)
#if defined(DTYPE_I8)
#error "OP_LOG not supported for I8"
#else
    return from_acc(logf(to_acc(x)));
#endif
#elif defined(OP_SQRT)
#if defined(DTYPE_I8)
#error "OP_SQRT not supported for I8"
#else
    return from_acc(sqrtf(to_acc(x)));
#endif
#elif defined(OP_TANH)
#if defined(DTYPE_I8)
#error "OP_TANH not supported for I8"
#else
    return from_acc(tanhf(to_acc(x)));
#endif
#elif defined(OP_SIGMOID)
#if defined(DTYPE_I8)
#error "OP_SIGMOID not supported for I8"
#else
    acc_t v = to_acc(x);
    return from_acc(1.0f / (1.0f + expf(-v)));
#endif
#elif defined(OP_ERF)
#if defined(DTYPE_I8)
#error "OP_ERF not supported for I8"
#else
    return from_acc(erff(to_acc(x)));
#endif
#elif defined(OP_NOT)
#if defined(DTYPE_I8)
    return static_cast<scalar_t>(static_cast<int8_t>(~x));
#else
#error "OP_NOT only supported for I8"
#endif
#else
#error "Define a unary OP_* macro"
#endif
}

extern "C" {
void KERNEL_NAME(scalar_t *in, scalar_t *out) {
    for (int i = 0; i < TILE_ELEMS; ++i) {
        out[i] = unary_eval(in[i]);
    }
}
}

//===- eltwise_binary.cc ---------------------------------------*- C++ -*-===//
//
// Generic binary elementwise kernels for AIE2.
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

static inline scalar_t binary_eval(scalar_t a, scalar_t b) {
#if defined(OP_ADD)
    return from_acc(to_acc(a) + to_acc(b));
#elif defined(OP_SUB)
    return from_acc(to_acc(a) - to_acc(b));
#elif defined(OP_MUL)
    return from_acc(to_acc(a) * to_acc(b));
#elif defined(OP_DIV)
#if defined(DTYPE_I8)
    acc_t denom = to_acc(b);
    if (denom == 0) {
        return static_cast<scalar_t>(0);
    }
    return from_acc(to_acc(a) / denom);
#else
    return from_acc(to_acc(a) / to_acc(b));
#endif
#elif defined(OP_POW)
#if defined(DTYPE_I8)
#error "OP_POW not supported for I8"
#else
    return from_acc(powf(to_acc(a), to_acc(b)));
#endif
#elif defined(OP_XOR)
#if defined(DTYPE_I8)
    return static_cast<scalar_t>(static_cast<int8_t>(a ^ b));
#else
#error "OP_XOR only supported for I8"
#endif
#else
#error "Define a binary OP_* macro"
#endif
}

extern "C" {
void KERNEL_NAME(scalar_t *a, scalar_t *b, scalar_t *out) {
    for (int i = 0; i < TILE_ELEMS; ++i) {
        out[i] = binary_eval(a[i], b[i]);
    }
}
}

//===- bitwise_bf16_binary.cc -------------------------------*- C++ -*-===//
//
// Generic BF16 bitwise binary kernels for AIE2.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#if defined(DTYPE_BF16)
using scalar_t = uint16_t;
#else
#error "Define DTYPE_BF16"
#endif

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

#ifndef TILE_ELEMS
#define TILE_ELEMS 256
#endif

static inline scalar_t bitwise_eval(scalar_t a, scalar_t b) {
#if defined(OP_AND)
    return static_cast<scalar_t>(a & b);
#elif defined(OP_OR)
    return static_cast<scalar_t>(a | b);
#elif defined(OP_XOR)
    return static_cast<scalar_t>(a ^ b);
#else
#error "Define a bitwise OP_* macro"
#endif
}

extern "C" {
void KERNEL_NAME(scalar_t *a, scalar_t *b, scalar_t *out) {
    for (int i = 0; i < TILE_ELEMS; ++i) {
        out[i] = bitwise_eval(a[i], b[i]);
    }
}
}

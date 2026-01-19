//===- bitwise_bf16_unary.cc --------------------------------*- C++ -*-===//
//
// Generic BF16 bitwise unary kernels for AIE2.
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

static inline scalar_t bitwise_eval(scalar_t a) {
#if defined(OP_NOT)
    return static_cast<scalar_t>(~a);
#else
#error "Define a bitwise OP_* macro"
#endif
}

extern "C" {
void KERNEL_NAME(scalar_t *in, scalar_t *out) {
    for (int i = 0; i < TILE_ELEMS; ++i) {
        out[i] = bitwise_eval(in[i]);
    }
}
}

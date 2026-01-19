//===- compare_binary.cc ---------------------------------------*- C++ -*-===//
//
// Generic binary compare kernels for AIE2.
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

#ifndef TILE_ELEMS
#define TILE_ELEMS 256
#endif

static inline uint8_t compare_eval(scalar_t a, scalar_t b) {
#if defined(OP_EQ)
    return static_cast<uint8_t>(a == b);
#elif defined(OP_GT)
    return static_cast<uint8_t>(a > b);
#elif defined(OP_LT)
    return static_cast<uint8_t>(a < b);
#else
#error "Define a compare OP_* macro"
#endif
}

extern "C" {
void KERNEL_NAME(scalar_t *a, scalar_t *b, uint8_t *out) {
    for (int i = 0; i < TILE_ELEMS; ++i) {
        out[i] = compare_eval(a[i], b[i]);
    }
}
}

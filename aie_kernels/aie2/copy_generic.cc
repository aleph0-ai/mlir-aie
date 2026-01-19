//===- copy_generic.cc -----------------------------------------*- C++ -*-===//
//
// Generic copy kernel for AIE2.
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

extern "C" {
void KERNEL_NAME(scalar_t *in, scalar_t *out, int32_t input_size) {
    if (input_size <= 0) {
        return;
    }
    for (int32_t i = 0; i < input_size; ++i) {
        out[i] = in[i];
    }
}
}

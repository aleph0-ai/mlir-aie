//===- cast_generic.cc ----------------------------------------*- C++ -*-===//
//
// Generic Cast kernels for AIE2.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>
#include <math.h>

#include <aie_api/aie.hpp>

#if defined(SRC_BF16)
using src_t = bfloat16;
using acc_t = float;
#elif defined(SRC_F16)
using src_t = _Float16;
using acc_t = float;
#elif defined(SRC_F32)
using src_t = float;
using acc_t = float;
#elif defined(SRC_I8)
using src_t = int8_t;
using acc_t = int32_t;
#else
#error "Define one of SRC_BF16/SRC_F16/SRC_F32/SRC_I8"
#endif

#if defined(DST_BF16)
using dst_t = bfloat16;
#elif defined(DST_F16)
using dst_t = _Float16;
#elif defined(DST_F32)
using dst_t = float;
#elif defined(DST_I8)
using dst_t = int8_t;
#else
#error "Define one of DST_BF16/DST_F16/DST_F32/DST_I8"
#endif

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

static inline acc_t to_acc(src_t v) {
#if defined(SRC_I8)
    return static_cast<acc_t>(v);
#else
    return static_cast<acc_t>(v);
#endif
}

static inline dst_t from_acc(acc_t v) {
#if defined(DST_I8)
    if (v > 127) {
        return static_cast<dst_t>(127);
    }
    if (v < -128) {
        return static_cast<dst_t>(-128);
    }
    return static_cast<dst_t>(v);
#else
    return static_cast<dst_t>(v);
#endif
}

extern "C" {
void KERNEL_NAME(src_t *in, dst_t *out, int32_t input_size) {
    if (input_size <= 0) {
        return;
    }
    for (int32_t i = 0; i < input_size; ++i) {
        out[i] = from_acc(to_acc(in[i]));
    }
}
}

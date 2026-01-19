//===- pool_generic.cc ----------------------------------------*- C++ -*-===//
//
// Generic MaxPool/AveragePool kernels for AIE2 (NCHW).
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

static inline acc_t min_init() {
#if defined(DTYPE_I8)
    return static_cast<acc_t>(-128);
#else
    return -INFINITY;
#endif
}

extern "C" {
void KERNEL_NAME(
    scalar_t *in,
    scalar_t *out,
    int32_t n,
    int32_t c,
    int32_t in_h,
    int32_t in_w,
    int32_t out_h,
    int32_t out_w,
    int32_t kernel_h,
    int32_t kernel_w,
    int32_t stride_h,
    int32_t stride_w,
    int32_t pad_top,
    int32_t pad_left) {
    if (n <= 0 || c <= 0 || in_h <= 0 || in_w <= 0 || out_h <= 0 || out_w <= 0) {
        return;
    }
    int32_t out_idx = 0;
    for (int32_t bn = 0; bn < n; ++bn) {
        for (int32_t ch = 0; ch < c; ++ch) {
            for (int32_t oh = 0; oh < out_h; ++oh) {
                for (int32_t ow = 0; ow < out_w; ++ow) {
                    const int32_t in_h_base = oh * stride_h - pad_top;
                    const int32_t in_w_base = ow * stride_w - pad_left;
                    acc_t acc = 0;
                    int32_t count = 0;
#if defined(OP_MAX)
                    acc = min_init();
#endif
                    for (int32_t kh = 0; kh < kernel_h; ++kh) {
                        const int32_t ih = in_h_base + kh;
                        if (ih < 0 || ih >= in_h) {
                            continue;
                        }
                        for (int32_t kw = 0; kw < kernel_w; ++kw) {
                            const int32_t iw = in_w_base + kw;
                            if (iw < 0 || iw >= in_w) {
                                continue;
                            }
                            const int32_t in_idx = ((bn * c + ch) * in_h + ih) * in_w + iw;
                            const acc_t v = to_acc(in[in_idx]);
#if defined(OP_MAX)
                            if (v > acc) {
                                acc = v;
                            }
#elif defined(OP_AVG)
                            acc += v;
#else
#error "Define OP_MAX or OP_AVG"
#endif
                            count += 1;
                        }
                    }
#if defined(OP_AVG)
                    if (count > 0) {
                        acc = acc / static_cast<acc_t>(count);
                    } else {
                        acc = 0;
                    }
#endif
                    out[out_idx++] = from_acc(acc);
                }
            }
        }
    }
}
}

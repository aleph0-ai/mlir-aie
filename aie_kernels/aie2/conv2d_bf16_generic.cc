//===- conv2d_bf16_generic.cc ----------------------------------*- C++ -*-===//
//
// Generic BF16 Conv2d kernel for AIE2 (NCHW, FCHW).
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#if defined(DTYPE_BF16)
using scalar_t = bfloat16;
#else
#error "Define DTYPE_BF16"
#endif

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

extern "C" {
void KERNEL_NAME(
    scalar_t *input,
    scalar_t *weights,
    scalar_t *output,
    int32_t batch,
    int32_t in_h,
    int32_t in_w,
    int32_t in_c,
    int32_t out_c,
    int32_t k_h,
    int32_t k_w,
    int32_t out_h,
    int32_t out_w,
    int32_t stride_h,
    int32_t stride_w,
    int32_t pad_top,
    int32_t pad_left) {
    if (batch <= 0 || in_h <= 0 || in_w <= 0 || in_c <= 0 || out_c <= 0 || k_h <= 0 || k_w <= 0) {
        return;
    }
    if (stride_h <= 0 || stride_w <= 0) {
        return;
    }
    for (int32_t n = 0; n < batch; ++n) {
        for (int32_t oc = 0; oc < out_c; ++oc) {
            for (int32_t oh = 0; oh < out_h; ++oh) {
                for (int32_t ow = 0; ow < out_w; ++ow) {
                    float acc = 0.0f;
                    for (int32_t ic = 0; ic < in_c; ++ic) {
                        for (int32_t kh = 0; kh < k_h; ++kh) {
                            for (int32_t kw = 0; kw < k_w; ++kw) {
                                int32_t ih = oh * stride_h - pad_top + kh;
                                int32_t iw = ow * stride_w - pad_left + kw;
                                if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                                    continue;
                                }
                                int32_t in_idx = (((n * in_c + ic) * in_h) + ih) * in_w + iw;
                                int32_t w_idx = (((oc * in_c) + ic) * k_h + kh) * k_w + kw;
                                acc += static_cast<float>(input[in_idx]) * static_cast<float>(weights[w_idx]);
                            }
                        }
                    }
                    int32_t out_idx = (((n * out_c + oc) * out_h) + oh) * out_w + ow;
                    output[out_idx] = static_cast<scalar_t>(acc);
                }
            }
        }
    }
}
}

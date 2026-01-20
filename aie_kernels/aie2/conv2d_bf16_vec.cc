//===- conv2d_bf16_vec.cc ------------------------------------*- C++ -*-===//
//
// Vectorized BF16 Conv2d kernel for AIE2 (NCHW, FCHW-packed weights).
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
    int32_t groups,
    int32_t out_c,
    int32_t k_h,
    int32_t k_w,
    int32_t out_h,
    int32_t out_w,
    int32_t stride_h,
    int32_t stride_w,
    int32_t pad_top,
    int32_t pad_left) {
#ifdef KERNEL_K
    const int32_t k_h_eff = KERNEL_K;
    const int32_t k_w_eff = KERNEL_K;
#else
    const int32_t k_h_eff = k_h;
    const int32_t k_w_eff = k_w;
#endif

    if (batch <= 0 || in_h <= 0 || in_w <= 0 || in_c <= 0 || out_c <= 0) {
        return;
    }
    if (k_h_eff <= 0 || k_w_eff <= 0) {
        return;
    }
    if (stride_h <= 0 || stride_w <= 0) {
        return;
    }
    if (groups != 1) {
        return;
    }
    constexpr int32_t k_vec = 8;
    if ((out_c % k_vec) != 0) {
        return;
    }

    const int32_t out_c_blocks = out_c / k_vec;
    const int32_t k_total = in_c * k_h_eff * k_w_eff;
    const int32_t out_plane = out_h * out_w;

    for (int32_t n = 0; n < batch; ++n) {
        for (int32_t ocb = 0; ocb < out_c_blocks; ++ocb) {
            for (int32_t oh = 0; oh < out_h; ++oh) {
                for (int32_t ow = 0; ow < out_w; ++ow) {
                    aie::accum<accfloat, k_vec> acc = aie::zeros<accfloat, k_vec>();
                    for (int32_t ic = 0; ic < in_c; ++ic) {
                        for (int32_t kh = 0; kh < k_h_eff; ++kh) {
                            const int32_t ih = oh * stride_h - pad_top + kh;
                            if (ih < 0 || ih >= in_h) {
                                continue;
                            }
                            for (int32_t kw = 0; kw < k_w_eff; ++kw) {
                                const int32_t iw = ow * stride_w - pad_left + kw;
                                if (iw < 0 || iw >= in_w) {
                                    continue;
                                }
                                const int32_t in_idx = (((n * in_c + ic) * in_h) + ih) * in_w + iw;
                                const int32_t k_idx = (ic * k_h_eff + kh) * k_w_eff + kw;
                                const int32_t w_base = (ocb * k_total + k_idx) * k_vec;
                                const aie::vector<scalar_t, k_vec> w_vec =
                                    aie::load_v<k_vec>(weights + w_base);
                                acc = aie::add(acc, aie::mul(w_vec, input[in_idx]));
                            }
                        }
                    }
                    alignas(32) scalar_t tmp[k_vec];
                    const aie::vector<scalar_t, k_vec> out_vec = acc.to_vector<scalar_t>();
                    aie::store_v(tmp, out_vec);
                    const int32_t out_base =
                        (((n * out_c + ocb * k_vec) * out_h) + oh) * out_w + ow;
                    for (int32_t lane = 0; lane < k_vec; ++lane) {
                        output[out_base + lane * out_plane] = tmp[lane];
                    }
                }
            }
        }
    }
}
}

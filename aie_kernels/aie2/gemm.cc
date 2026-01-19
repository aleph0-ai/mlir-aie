//===- gemm.cc -------------------------------------------------*- C++ -*-===//
//
// Generic GEMM kernel for AIE2.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include <aie_api/aie.hpp>

#if defined(DTYPE_BF16)
using scalar_t = bfloat16;
#elif defined(DTYPE_F32)
using scalar_t = float;
#else
#error "Define DTYPE_BF16 or DTYPE_F32"
#endif

#ifndef KERNEL_NAME
#error "Define KERNEL_NAME"
#endif

static inline float bits_to_float(int32_t bits) {
    union {
        int32_t i;
        float f;
    } u;
    u.i = bits;
    return u.f;
}

extern "C" {
#if defined(GEMM_HAS_C)
void KERNEL_NAME(
    scalar_t *a,
    scalar_t *b,
    scalar_t *c,
    scalar_t *out,
    int32_t m,
    int32_t n,
    int32_t k,
    int32_t trans_a,
    int32_t trans_b,
    int32_t alpha_bits,
    int32_t beta_bits,
    int32_t c_rows,
    int32_t c_cols) {
#else
void KERNEL_NAME(
    scalar_t *a,
    scalar_t *b,
    scalar_t *out,
    int32_t m,
    int32_t n,
    int32_t k,
    int32_t trans_a,
    int32_t trans_b,
    int32_t alpha_bits) {
#endif
    if (m <= 0 || n <= 0 || k <= 0) {
        return;
    }
    float alpha = bits_to_float(alpha_bits);
#if defined(GEMM_HAS_C)
    float beta = bits_to_float(beta_bits);
#endif
    for (int32_t i = 0; i < m; ++i) {
        for (int32_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int32_t kk = 0; kk < k; ++kk) {
                float a_val = trans_a ? static_cast<float>(a[kk * m + i])
                                      : static_cast<float>(a[i * k + kk]);
                float b_val = trans_b ? static_cast<float>(b[j * k + kk])
                                      : static_cast<float>(b[kk * n + j]);
                acc += a_val * b_val;
            }
#if defined(GEMM_HAS_C)
            float c_val = 0.0f;
            if (c_rows <= 1 && c_cols <= 1) {
                c_val = static_cast<float>(c[0]);
            } else if (c_rows <= 1) {
                c_val = static_cast<float>(c[j]);
            } else if (c_cols <= 1) {
                c_val = static_cast<float>(c[i]);
            } else {
                c_val = static_cast<float>(c[i * c_cols + j]);
            }
            float y = alpha * acc + beta * c_val;
#else
            float y = alpha * acc;
#endif
            out[i * n + j] = static_cast<scalar_t>(y);
        }
    }
}
}

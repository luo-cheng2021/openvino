// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

#include "vnode_utils.hpp"

#if defined(HAVE_AVX2)
#    include <immintrin.h>
inline void hsum(__m256& x) {
    __m256 y;                             // x:  0 1 2 3   4 5 6 7
    y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
    x = _mm256_add_ps(x, y);              // X:  01 12 23 30  45 56 67 74
    y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
    x = _mm256_add_ps(x, y);              // x: 0123 x x x   4567 x x x
    y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
    x = _mm256_add_ps(x, y);              // x: 01234567 x x x x x x x
}
inline __m256i get_mask(int N7) {
    static __m256i mask[] = {
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),
        _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),
        _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),
        _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),
        _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),
        _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1),
    };
    return _mm256_loadu_si256(&mask[N7]);
}
#endif
namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void rms_norm(float* dst, float* src, float eps, float* weight, size_t size) {
#if defined(HAVE_AVX2)
    size_t i;
    __m256i mask;
    auto sum2 = _mm256_setzero_ps();
    for (i = 0; i < size - 8; i += 8) {
        auto d0 = _mm256_loadu_ps(src + i);
        auto d1 = _mm256_mul_ps(d0, d0);
        sum2 = _mm256_add_ps(sum2, d1);
    }
    if (i < size) {
        mask = get_mask(size - i);
        auto d0 = _mm256_maskload_ps(src + i, mask);
        auto d1 = _mm256_mul_ps(d0, d0);
        sum2 = _mm256_add_ps(sum2, d1);
    }
    auto v_eps = _mm256_set1_ps(eps);
    hsum(sum2);
    auto v_sum = _mm256_div_ps(sum2, _mm256_set1_ps(size));
    auto scale = _mm256_add_ps(v_sum, v_eps);
    scale = _mm256_sqrt_ps(scale);
    scale = _mm256_div_ps(_mm256_set1_ps(1.0f), scale);
    for (i = 0; i < size - 8; i += 8) {
        auto d0 = _mm256_loadu_ps(src + i);
        auto dw = _mm256_loadu_ps(weight + i);
        auto d1 = _mm256_mul_ps(_mm256_mul_ps(d0, scale), dw);
        _mm256_storeu_ps(dst + i, d1);
    }
    if (i < size) {
        auto d0 = _mm256_maskload_ps(src + i, mask);
        auto dw = _mm256_maskload_ps(weight + i, mask);
        auto d1 = _mm256_mul_ps(_mm256_mul_ps(d0, scale), dw);
        _mm256_maskstore_ps(dst + i, mask, d1);
    }
#else
    float sum2 = 0.0f;
    for (size_t i = 0; i < size; i++) {
        sum2 += src[i] * src[i];
    }
    auto scale = 1.0f / sqrt(sum2 / size + eps);
    for (size_t i = 0; i < size; i++) {
        dst[i] = src[i] * scale * weight[i];
    }
#endif
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
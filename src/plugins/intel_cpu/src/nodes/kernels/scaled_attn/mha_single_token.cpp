// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/parallel.hpp"
#include "mha_single_token.hpp"
#include "common.hpp"
#include "softmax_kernel.hpp"
#include "utils/profiler.hpp"
#define _GNU_SOURCE
#include <unistd.h>
#include <sys/types.h>

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;

#if defined(HAVE_AVX2)

#define prefetch_bytes(bytes, sel, advance, src) {  \
    auto *p = reinterpret_cast<char *>(src);        \
    for (size_t i = 0; i < bytes; i += 64)          \
        _mm_prefetch(p + i + advance, sel);         \
}

#else

#define prefetch_bytes(bytes, sel, advance, src)

#endif

template<typename T>
static void attn_acc_value(float* out, float weight, T* v, size_t S, float scale, float zp) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
    for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
        auto v_value = mm512_uni_loadu_ps(v + i);
        auto v_out = mm512_uni_loadu_ps(out + i);
        v_out = _mm512_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        _mm512_storeu_ps(out + i, v_out);
    }
#elif defined(HAVE_AVX2)
    auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto v_value = mm256_uni_loadu_ps(v + i);
        auto v_out = mm256_uni_loadu_ps(out + i);
        v_out = _mm256_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        mm256_uni_storeu_ps(out + i, v_out);
    }
#endif
    for (; i < S; i++) {
        out[i] += weight * v[i];
    }
}

static void attn_acc_value(float* out, float weight, uint8_t* v, size_t S, float scale, float zp) {
    size_t i = 0;
    weight *= scale;
#if defined(HAVE_AVX512F)
    auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
    auto v_zp = _mm512_set1_ps(zp);
    for (; i + 4 * vec_len_f32_avx512 <= S; i += 4* vec_len_f32_avx512) {
        auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i));
        auto v1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx512));
        auto v2_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx512 * 2));
        auto v3_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx512 * 3));

        auto v0_out = mm512_uni_loadu_ps(out + i);
        auto v1_out = mm512_uni_loadu_ps(out + i + vec_len_f32_avx512);
        auto v2_out = mm512_uni_loadu_ps(out + i + vec_len_f32_avx512 * 2);
        auto v3_out = mm512_uni_loadu_ps(out + i + vec_len_f32_avx512 * 3);

        auto v0_256 = _mm512_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm512_cvtepu8_epi32(v1_128);
        auto v2_256 = _mm512_cvtepu8_epi32(v2_128);
        auto v3_256 = _mm512_cvtepu8_epi32(v3_128);

        auto v0_value = _mm512_cvtepi32_ps(v0_256);
        auto v1_value = _mm512_cvtepi32_ps(v1_256);
        auto v2_value = _mm512_cvtepi32_ps(v2_256);
        auto v3_value = _mm512_cvtepi32_ps(v3_256);

        v0_value = _mm512_sub_ps(v0_value, v_zp);
        v1_value = _mm512_sub_ps(v1_value, v_zp);
        v2_value = _mm512_sub_ps(v2_value, v_zp);
        v3_value = _mm512_sub_ps(v3_value, v_zp);

        v0_out = _mm512_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm512_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);
        v2_out = _mm512_fmadd_ps(attn_w_vec_fp32, v2_value, v2_out);
        v3_out = _mm512_fmadd_ps(attn_w_vec_fp32, v3_value, v3_out);

        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 0, v0_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 1, v1_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 2, v2_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 3, v3_out);
    }
    if (i + 2 * vec_len_f32_avx512 <= S) {
        auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i));
        auto v1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx512));

        auto v0_out = mm512_uni_loadu_ps(out + i);
        auto v1_out = mm512_uni_loadu_ps(out + i + vec_len_f32_avx512);

        auto v0_256 = _mm512_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm512_cvtepu8_epi32(v1_128);

        auto v0_value = _mm512_cvtepi32_ps(v0_256);
        auto v1_value = _mm512_cvtepi32_ps(v1_256);

        v0_value = _mm512_sub_ps(v0_value, v_zp);
        v1_value = _mm512_sub_ps(v1_value, v_zp);

        v0_out = _mm512_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm512_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);

        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 0, v0_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 1, v1_out);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= S) {
        auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + i));
        auto v0_out = mm512_uni_loadu_ps(out + i);
        auto v0_256 = _mm512_cvtepu8_epi32(v0_128);
        auto v0_value = _mm512_cvtepi32_ps(v0_256);
        v0_value = _mm512_sub_ps(v0_value, v_zp);
        v0_out = _mm512_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        mm512_uni_storeu_ps(out + i + vec_len_f32_avx512 * 0, v0_out);
        i += vec_len_f32_avx512;
    }
#elif defined(HAVE_AVX2)
    auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
    auto v_zp = _mm256_set1_ps(zp);
    for (; i + 4 * vec_len_f32_avx2 <= S; i += 4* vec_len_f32_avx2) {
        auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i));
        auto v1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx2));
        auto v2_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx2 * 2));
        auto v3_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx2 * 3));

        auto v0_out = mm256_uni_loadu_ps(out + i);
        auto v1_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2);
        auto v2_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2 * 2);
        auto v3_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2 * 3);

        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm256_cvtepu8_epi32(v1_128);
        auto v2_256 = _mm256_cvtepu8_epi32(v2_128);
        auto v3_256 = _mm256_cvtepu8_epi32(v3_128);

        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        auto v1_value = _mm256_cvtepi32_ps(v1_256);
        auto v2_value = _mm256_cvtepi32_ps(v2_256);
        auto v3_value = _mm256_cvtepi32_ps(v3_256);

        v0_value = _mm256_sub_ps(v0_value, v_zp);
        v1_value = _mm256_sub_ps(v1_value, v_zp);
        v2_value = _mm256_sub_ps(v2_value, v_zp);
        v3_value = _mm256_sub_ps(v3_value, v_zp);

        v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm256_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);
        v2_out = _mm256_fmadd_ps(attn_w_vec_fp32, v2_value, v2_out);
        v3_out = _mm256_fmadd_ps(attn_w_vec_fp32, v3_value, v3_out);

        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 0, v0_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 1, v1_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 2, v2_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 3, v3_out);
    }
    if (i + 2 * vec_len_f32_avx2 <= S) {
        auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i));
        auto v1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i + vec_len_f32_avx2));

        auto v0_out = mm256_uni_loadu_ps(out + i);
        auto v1_out = mm256_uni_loadu_ps(out + i + vec_len_f32_avx2);

        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v1_256 = _mm256_cvtepu8_epi32(v1_128);

        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        auto v1_value = _mm256_cvtepi32_ps(v1_256);

        v0_value = _mm256_sub_ps(v0_value, v_zp);
        v1_value = _mm256_sub_ps(v1_value, v_zp);

        v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        v1_out = _mm256_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);

        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 0, v0_out);
        mm256_uni_storeu_ps(out + i + vec_len_f32_avx2 * 1, v1_out);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= S) {
        auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i));
        auto v0_out = mm256_uni_loadu_ps(out + i);
        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        v0_value = _mm256_sub_ps(v0_value, v_zp);
        v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
        mm256_uni_storeu_ps(out + i, v0_out);
        i += vec_len_f32_avx2;
    }
#endif
    for (; i < S; i++) {
        out[i] += weight * (v[i] - zp);
    }
}

template<typename T>
static float sum_q_head(T* a, size_t n) {
    float sum = 0.0f;
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto vsum0 = _mm512_set1_ps(0.0f);
    auto vsum1 = _mm512_set1_ps(0.0f);
    auto vsum2 = _mm512_set1_ps(0.0f);
    auto vsum3 = _mm512_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx512 <= n; i += vec_len_f32_avx512 * 4) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        vsum0 = _mm512_add_ps(va0, vsum0);
        vsum1 = _mm512_add_ps(va1, vsum1);
        vsum2 = _mm512_add_ps(va2, vsum2);
        vsum3 = _mm512_add_ps(va3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        vsum0 = _mm512_add_ps(va0, vsum0);
        vsum1 = _mm512_add_ps(va1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        vsum0 = _mm512_add_ps(va0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    sum = _mm512_reduce_add_ps(vsum0);
#elif defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        vsum0 = _mm256_add_ps(va0, vsum0);
        vsum1 = _mm256_add_ps(va1, vsum1);
        vsum2 = _mm256_add_ps(va2, vsum2);
        vsum3 = _mm256_add_ps(va3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        vsum0 = _mm256_add_ps(va0, vsum0);
        vsum1 = _mm256_add_ps(va1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        vsum0 = _mm256_add_ps(va0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);
#endif

    for (; i < n; i++) {
        float tmp = a[i];
        sum += tmp;
    }
    return sum;
}

template<typename TA, typename TB>
static __attribute__((always_inline)) inline float dot_product(TA* a, TB* b, size_t n, float* scale, float* zp, float* head_sum) {
    size_t i = 0;
    float sum = 0.0f;
#if defined(HAVE_AVX512F)
    auto vsum0 = _mm512_setzero_ps();
    auto vsum1 = _mm512_setzero_ps();
    auto vsum2 = _mm512_setzero_ps();
    auto vsum3 = _mm512_setzero_ps();
    for (; i + 4 * vec_len_f32_avx512 <= n; i += 4 * vec_len_f32_avx512) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);
        auto vb2 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 2);
        auto vb3 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 3);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto vb0 = mm512_uni_loadu_ps(b + i);
        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    sum = _mm512_reduce_add_ps(vsum0);
#elif defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        auto vb0 = mm256_uni_loadu_ps(b + i);
        auto vb1 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2);
        auto vb2 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2 * 2);
        auto vb3 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2 * 3);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        auto vb0 = mm256_uni_loadu_ps(b + i);
        auto vb1 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto vb0 = mm256_uni_loadu_ps(b + i);
        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);
#endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

template<typename TA>
static __attribute__((always_inline)) inline float dot_product(TA* a, uint8_t* b, size_t n, float* scale, float* zp, float* head_sum) {
    size_t i = 0;
    float sum = 0.0f;
#if defined(HAVE_AVX512F)
    auto vsum0 = _mm512_set1_ps(0.0f);
    auto vsum1 = _mm512_set1_ps(0.0f);
    auto vsum2 = _mm512_set1_ps(0.0f);
    auto vsum3 = _mm512_set1_ps(0.0f);
    auto v_zp = _mm512_set1_ps(*zp);
    for (; i + 4 * vec_len_f32_avx512 <= n; i += vec_len_f32_avx512 * 4) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512));
        auto vb2_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512 * 2));
        auto vb3_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512 * 3));

        auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);
        auto vb2_256 = _mm512_cvtepu8_epi32(vb2_128);
        auto vb3_256 = _mm512_cvtepu8_epi32(vb3_128);

        auto vb0 = _mm512_cvtepi32_ps(vb0_256);
        auto vb1 = _mm512_cvtepi32_ps(vb1_256);
        auto vb2 = _mm512_cvtepi32_ps(vb2_256);
        auto vb3 = _mm512_cvtepi32_ps(vb3_256);

        vb0 = _mm512_sub_ps(vb0, v_zp);
        vb1 = _mm512_sub_ps(vb1, v_zp);
        vb2 = _mm512_sub_ps(vb2, v_zp);
        vb3 = _mm512_sub_ps(vb3, v_zp);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512));

        auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);

        auto vb0 = _mm512_cvtepi32_ps(vb0_256);
        auto vb1 = _mm512_cvtepi32_ps(vb1_256);

        vb0 = _mm512_sub_ps(vb0, v_zp);
        vb1 = _mm512_sub_ps(vb1, v_zp);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i));
        auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
        auto vb0 = _mm512_cvtepi32_ps(vb0_256);
        vb0 = _mm512_sub_ps(vb0, v_zp);
        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    sum = _mm512_reduce_add_ps(vsum0);
    for (; i < n; i++) {
        sum += a[i] * (b[i] - *zp);
    }
    return scale[0] * sum;

#elif defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2));
        auto vb2_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2 * 2));
        auto vb3_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2 * 3));

        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);
        auto vb2_256 = _mm256_cvtepu8_epi32(vb2_128);
        auto vb3_256 = _mm256_cvtepu8_epi32(vb3_128);

        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        auto vb1 = _mm256_cvtepi32_ps(vb1_256);
        auto vb2 = _mm256_cvtepi32_ps(vb2_256);
        auto vb3 = _mm256_cvtepi32_ps(vb3_256);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2));

        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);

        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        auto vb1 = _mm256_cvtepi32_ps(vb1_256);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i));
        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    // B = scale * (b - zero)
    // Σ (A * B) = Σ (a * scale * (b - zero)) = scale * (Σ a * b - zero Σ a) = scale * (sum - zp * head_sum)
    return scale[0] * (sum - zp[0] * head_sum[0]);
#else
    for (; i < n; i++) {
        sum += a[i] * (b[i] - *zp);
    }
    return scale[0] * sum;
#endif
}

template<typename T>
static void attn_reduce(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= S; i+= vec_len_f32_avx512) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm512_setzero_ps();
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = _mm512_loadu_ps(src);
            result_vec_fp32 = _mm512_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        // save to bf16
        mm512_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm256_set1_ps(0.0f);
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = mm256_uni_loadu_ps(src);
            result_vec_fp32 = _mm256_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        mm256_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#endif
    for (; i < S; i++) {
        auto* src = temp + i;
        float sum = 0.0f;
        // sum result from all threads partition
        for (size_t m = 0; m < M; m++) {
            sum += src[0];
            src += temp_stride;
        }
        dst[i] = sum;
    }
}
struct timer {
    struct one_thread {
        size_t ns;
        size_t count;
        std::chrono::steady_clock::time_point start;
        size_t pad[8 - 3];
    } threads[8];
    timer() {
        memset(this, 0, sizeof(*this));
    }
    void start(size_t id) {
        threads[id].start = std::chrono::steady_clock::now();
    }
    void end(size_t id) {
        auto end_ns = std::chrono::steady_clock::now();
        threads[id].ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_ns - threads[id].start).count();
        threads[id].count++;
    }
    void reset(size_t id) {
        threads[id].ns = 0;
        threads[id].count = 0;
    }
    ~timer() {
        for (int i = 0; i < 8; i++) {
            std::cout << "id " << i << " ns " << threads[i].ns << " count " << threads[i].count
                << " avg ns " << static_cast<double>(threads[i].ns) / threads[i].count << "\n";
        }
    }
} g_timer;

template <typename T, typename T2>
static void mha_single_token_kernel(const ov::intel_cpu::PlainTensor& query,
                             const ov::intel_cpu::PlainTensor& present_key,
                             const ov::intel_cpu::PlainTensor& present_value,
                             const ov::intel_cpu::PlainTensor& alibi_mask,
                             const ov::intel_cpu::PlainTensor& attention_mask,
                             const ov::intel_cpu::PlainTensor& beams,
                             ov::intel_cpu::PlainTensor& output_emb,
                             ov::intel_cpu::PlainTensor& buf_attn_w,
                             ov::intel_cpu::PlainTensor& buf_attn_score,
                             bool has_out_transpose,
                             bool auto_causal,
                             float d_scale,
                             const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                             const ov::intel_cpu::PlainTensor& past_v_scale_zp,
                             ov::intel_cpu::PlainTensor& head_sum) {
    PROFILE(_prof, "1tok_q*k");
    ov::intel_cpu::PlainTensor causal_mask;
    bool select_nfltmax_at_0 = false;
    auto B = query.size(0);
    auto H = query.size(1);
    auto q_len = query.size(2);
    auto S = query.size(3);
    auto kv_len = present_key.size(2);
    auto h_group_num = present_key.size(1);
    size_t h_each_group_len = 1;
    if (h_group_num != H) {
        h_each_group_len = H / h_group_num;
    }
    if (d_scale == 0.0f)
        d_scale = 1.0f / sqrt(S);
    auto nthr = static_cast<size_t>(parallel_get_max_threads());

    // use per-token kernel, for each k,v token
    //  attn mask is a matrix of q_len(kv_len)
    buf_attn_w.resize<float>({B, H, q_len, kv_len});
#if defined(HAVE_AVX2) && !defined(HAVE_AVX512F)
    // avx2 will pre-compute the zero point and try to save the sub instruction in the dot_product,
    //  but it seems not necessary for avx512. Possible reason may be that for avx2 the cost of dot_product
    //  is larger than the memory access time, but for avx512 is not and the cost of pre-compute is a pure increase.
    bool pastkv_is_int8 = past_k_scale_zp;
    if (pastkv_is_int8) {
        // be sure no false sharing
        head_sum.resize<float>({B, H, q_len, 16});
        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            head_sum.at<float>(b, h, pq, false) = sum_q_head(&query.at<T>(b, h, pq, false), S);
        });
    }
#endif
    //std::cout << "b=" << B << ", H=" << h_group_num << ", S=" << S << "L=" << kv_len << "\n";
    auto id = gettid();
    // _prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_mmset");
    // parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
    //     // std::string name = std::string("1tok_mmset") + std::to_string(ithr);
    //     // _prof = ov::intel_cpu::profilerManagerInstance.startProfile(name);//"1tok_mmset1");
    //     kv_len = std::min(1025ul, kv_len);
    //     auto cur = gettid();
    //     if (cur == id)
    //         _prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_mmset0");
    // //for (size_t ithr = 0; ithr < nthr; ithr++) {
    //     size_t start{0}, end{0};
    //     splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

    //     size_t b, h_group, pk;
    //     if (start < end) {
    //         parallel_it_init(start, b, B, h_group, h_group_num, pk, kv_len);
    //         auto p_k = &present_key.at<T2>(beams.at<int32_t>(b, pk), h_group, pk, false);
    //         for (size_t iwork = start; iwork < end; ++iwork) {
    //             //auto p = &past_k_scale_zp.at<float>(b_kv, h_group, pk, false);
    //             //auto p_k = &present_key.at<T2>(b_kv, h_group, pk, false);
    //             // _mm_prefetch(p_k + 4096*1, _MM_HINT_T0);
    //             // _mm_prefetch(p_k + 4096*1 + 64, _MM_HINT_T0);
    //             for (size_t i = 0; i < S; i += 64) {
    //                 *(volatile int*)(p_k + i);
    //             }
    //             p_k += S;
    //             if (++pk == kv_len) {
    //                 pk = 0;
    //                 //p_k += present_key.m_strides[1] - S * kv_len;
    //                 if (++h_group == h_group_num) {
    //                     h_group = 0;
    //                     if (++b == B)
    //                         break;
    //                 }
    //             }
    //         }
    //     }
    //     if (cur == id)
    //         _prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_mmset0_end");
    //     //break;
    // //}
    // });

    // _prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_fake");
    // parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
    //     auto curid = gettid();
    //     if (curid == id)
    //     _prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_mmsetf0");
    //     size_t start{0}, end{0};
    //     kv_len = std::min(1025ul, kv_len);
    //     splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

    //     size_t b, h_group, pk;
    //     if (start < end) {
    //         parallel_it_init(start, b, B, h_group, h_group_num, pk, kv_len);
    //         auto p_k = &present_key.at<T2>(beams.at<int32_t>(b, pk), h_group, pk, false);
    //         auto p_q = &query.at<T>(b, h_group, false);
    //         float tmp[3] = {1.1f, 0.0f, 1.0f};
    //         auto p = tmp;
    //         auto p_sum = tmp + 2;
    //         for (size_t iwork = start; iwork < end; ++iwork) {
    //             auto b_kv = beams ? beams.at<int32_t>(b, pk) : b;
    //             //auto p = &past_k_scale_zp.at<float>(b_kv, h_group, pk, false);
    //             //auto p_k = &present_key.at<T2>(b_kv, h_group, pk, false);
    //             _mm_prefetch(p_k + 4096*1, _MM_HINT_T0);
    //             _mm_prefetch(p_k + 4096*1 + 64, _MM_HINT_T0);
    //             buf_attn_w.at<float>(b, h_group, 0, pk) =
    //                     dot_product(p_q, p_k,
    //                         S, p, p + 1, p_sum);
    //             //parallel_it_step(b, B, h_group, h_group_num, pk, kv_len);
    //             //p += 2;
    //             p_k += S;
    //             if (++pk == kv_len) {
    //                 pk = 0;
    //                 p_k += present_key.m_strides[1] - S * kv_len;
    //                 //p += past_k_scale_zp.m_strides[1] - 2 * kv_len;
    //                 //p_q += S;
    //                 //p_sum += 16;
    //                 if (++h_group == h_group_num) {
    //                     h_group = 0;
    //                     if (++b == B)
    //                         break;
    //                 }
    //             }
    //         }
    //     }
    //     if (curid == id)
    //     _prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_mmsetf0_end");
    // });

    _prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_raw");
    //std::atomic<bool> flag;
    parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
        // volatile size_t count = 0;
        // if (ithr == 0) {
        //     while (count++ < 100000) {}
        //     flag = true;
        // }
        // while (flag.load() == false) {}
        auto curid = gettid();
        auto cur_kv_len = kv_len;
        if (cur_kv_len == 1025)
            g_timer.reset(ithr);
        if (1 || curid == id) {
            //_prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_mmsetr0");
            g_timer.start(ithr);
        }
        size_t start{0}, end{0};
        if (cur_kv_len >= 1025)
            cur_kv_len = 1025 * 1;
        splitter(B * h_group_num * cur_kv_len, nthr, ithr, start, end);

        size_t b, h_group, pk;
        if (start < end) {
            parallel_it_init(start, b, B, h_group, h_group_num, pk, cur_kv_len);
            if (1 || (q_len == 1 && h_each_group_len == 1)) {
                if (0 && B == 1) {
                    // the memory will be continuous when b==1
                    for (size_t iwork = start; iwork < end; ++iwork) {
                        auto p = &past_k_scale_zp.at<float>(0, h_group, pk, false);
                        auto p_k = &present_key.at<T2>(0, h_group, pk, false);
                        prefetch_bytes(S, _MM_HINT_T0, 4096, p_k);
                        buf_attn_w.at<float>(0, h_group, 0, pk) =
                                dot_product(&query.at<T>(0, h_group, false), p_k,
                                    S, p, p + 1, &head_sum.at<float>(0, h_group, false));
                        parallel_it_step(b, B, h_group, h_group_num, pk, cur_kv_len);
                    }
                } else {
                    auto p_k = &present_key.at<T2>(b/*beams.at<int32_t>(b, pk)*/, h_group, pk, false);
                    auto p_q = &query.at<T>(b, h_group, false);
                    auto p = &past_k_scale_zp.at<float>(b, h_group, pk, false);
                    auto p_sum = &head_sum.at<float>(b, h_group, false);
                    //g_timer.start(ithr);
                    for (size_t iwork = start; iwork < end; ++iwork) {
                        //auto b_kv = beams ? beams.at<int32_t>(b, pk) : b;
                        //auto p = &past_k_scale_zp.at<float>(b_kv, h_group, pk, false);
                        //auto p_k = &present_key.at<T2>(b_kv, h_group, pk, false);
                        _mm_prefetch(p_k + 4096*1, _MM_HINT_T0);
                        _mm_prefetch(p_k + 4096*1 + 64, _MM_HINT_T0);
                        buf_attn_w.at<float>(b, h_group, 0, pk) =
                                dot_product(p_q, p_k,
                                    S, p, p + 1, &head_sum.at<float>(b, h_group, false));
                        //parallel_it_step(b, B, h_group, h_group_num, pk, cur_kv_len);
                        p += 2;
                        p_k += S;
                        if (++pk == cur_kv_len) {
                            pk = 0;
                            //p_k += present_key.m_strides[1] - S * cur_kv_len;
                            //p += past_k_scale_zp.m_strides[1] - 2 * cur_kv_len;
                            p_q += S;
                            //p_sum += 16;
                            if (++h_group == h_group_num) {
                                h_group = 0;
                                if (++b == B)
                                    break;
                            }
                        }
                    }
                }
            } else {
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at<int32_t>(b, pk) : b;
                    for (size_t pq = 0; pq < q_len; pq++) {
                        auto p = &past_k_scale_zp.at<float>(b_kv, h_group, pk, false);
                        for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                            buf_attn_w.at<float>(b, h, pq, pk) =
                                    dot_product(&query.at<T>(b, h, pq, false), &present_key.at<T2>(b_kv, h_group, pk, false),
                                        S, p, p + 1, &head_sum.at<float>(b, h, pq, false));
                        }
                    }
                    parallel_it_step(b, B, h_group, h_group_num, pk, cur_kv_len);
                }
            }
        }
        if (1 || curid == id) {
            //_prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_mmsetr0_end");
            g_timer.end(ithr);
        }
    });

    _prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_softmax");
    parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
        // apply attention mask & sofmax
        auto ncausal = auto_causal ? (kv_len - q_len + pq + 1) : kv_len;
        float* alibi_ptr = alibi_mask ? &alibi_mask.at<float>({b, h, pq, 0}, true) : nullptr;
        float* attn_mask_ptr = attention_mask ? &attention_mask.at<float>({b, h, pq, 0}, true) : nullptr;
        uint8_t* cmask_ptr = causal_mask ? &causal_mask.at<uint8_t>({b, h, pq, 0}, true) : nullptr;
        attn_softmax_kernel(&buf_attn_w.at<float>(b, h, pq, false),
                            &buf_attn_w.at<float>(b, h, pq, false),
                            d_scale,
                            alibi_ptr,
                            attn_mask_ptr,
                            cmask_ptr,
                            select_nfltmax_at_0,
                            ncausal,
                            kv_len,
                            ov::element::f32);
    });

    // attn_w * V
    _prof = ov::intel_cpu::profilerManagerInstance.startProfile("1tok_k*v");
    buf_attn_score.resize<float>({static_cast<size_t>(nthr), B, q_len, H, S});
    // buf_attn_w {B, H, q_len, kv_len}
    parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
        size_t start{0}, end{0};
        splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

        memset(&buf_attn_score.at<float>({ithr, 0, 0, 0, 0}), 0, buf_attn_score.stride(0) * sizeof(float));

        size_t b, h_group, pv;
        if (start < end) {
            parallel_it_init(start, b, B, h_group, h_group_num, pv, kv_len);
            if (q_len == 1 && h_each_group_len == 1) {
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at<int32_t>(b, pv) : b;
                    auto* v = &present_value.at<T2>(b_kv, h_group, pv, false);
                    auto p = &past_v_scale_zp.at<float>(b_kv, h_group, pv, false);
                    attn_acc_value(&buf_attn_score.at<float>(ithr, b, 0, h_group, false),
                                buf_attn_w.at<float>(b, h_group, 0, pv),
                                v,
                                S,
                                p[0],
                                p[1]);
                    parallel_it_step(b, B, h_group, h_group_num, pv, kv_len);
                }
            } else {
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.at<int32_t>(b, pv) : b;
                    auto* v = &present_value.at<T2>(b_kv, h_group, pv, false);
                    auto p = &past_v_scale_zp.at<float>(b_kv, h_group, pv, false);
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                            attn_acc_value(&buf_attn_score.at<float>(ithr, b, pq, h, false),
                                        buf_attn_w.at<float>(b, h, pq, pv),
                                        v,
                                        S,
                                        p[0],
                                        p[1]);
                        }
                    }
                    parallel_it_step(b, B, h_group, h_group_num, pv, kv_len);
                }
            }
        }
    });

    parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
        auto* temp = &buf_attn_score.at<float>(0, b, pq, h, false);
        size_t temp_stride = buf_attn_score.stride(0);
        auto* dst = has_out_transpose ? &output_emb.at<T>(b, pq, h * S, false) : &output_emb.at<T>(b, h, pq, false);
        attn_reduce(dst, temp, nthr, S, temp_stride);
    });
}

void mha_single_token(const ov::intel_cpu::PlainTensor& query,
                      const ov::intel_cpu::PlainTensor& present_key,
                      const ov::intel_cpu::PlainTensor& present_value,
                      const ov::intel_cpu::PlainTensor& alibi_mask,
                      const ov::intel_cpu::PlainTensor& attention_mask,
                      const ov::intel_cpu::PlainTensor& beams,
                      ov::intel_cpu::PlainTensor& output_emb,
                      ov::intel_cpu::PlainTensor& buf_attn_w,
                      ov::intel_cpu::PlainTensor& buf_attn_score,
                      bool has_out_transpose,
                      bool auto_causal,
                      float d_scale,
                      const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                      const ov::intel_cpu::PlainTensor& past_v_scale_zp,
                      ov::intel_cpu::PlainTensor& head_sum) {
    if (query.get_precision() == ov::element::bf16) {
        if (present_key.get_precision() == ov::element::u8) {
            mha_single_token_kernel<ov::bfloat16, uint8_t>(query,
                                                           present_key,
                                                           present_value,
                                                           alibi_mask,
                                                           attention_mask,
                                                           beams,
                                                           output_emb,
                                                           buf_attn_w,
                                                           buf_attn_score,
                                                           has_out_transpose,
                                                           auto_causal,
                                                           d_scale,
                                                           past_k_scale_zp,
                                                           past_v_scale_zp,
                                                           head_sum);
        } else {
            mha_single_token_kernel<ov::bfloat16, ov::bfloat16>(query,
                                                                present_key,
                                                                present_value,
                                                                alibi_mask,
                                                                attention_mask,
                                                                beams,
                                                                output_emb,
                                                                buf_attn_w,
                                                                buf_attn_score,
                                                                has_out_transpose,
                                                                auto_causal,
                                                                d_scale,
                                                                past_k_scale_zp,
                                                                past_v_scale_zp,
                                                                head_sum);
        }
    } else if (query.get_precision() == ov::element::f32) {
        if (present_key.get_precision() == ov::element::u8) {
            mha_single_token_kernel<float, uint8_t>(query,
                                                    present_key,
                                                    present_value,
                                                    alibi_mask,
                                                    attention_mask,
                                                    beams,
                                                    output_emb,
                                                    buf_attn_w,
                                                    buf_attn_score,
                                                    has_out_transpose,
                                                    auto_causal,
                                                    d_scale,
                                                    past_k_scale_zp,
                                                    past_v_scale_zp,
                                                    head_sum);
        } else if (present_key.get_precision() == ov::element::f16) {
            mha_single_token_kernel<float, ov::float16>(query,
                                                        present_key,
                                                        present_value,
                                                        alibi_mask,
                                                        attention_mask,
                                                        beams,
                                                        output_emb,
                                                        buf_attn_w,
                                                        buf_attn_score,
                                                        has_out_transpose,
                                                        auto_causal,
                                                        d_scale,
                                                        past_k_scale_zp,
                                                        past_v_scale_zp,
                                                        head_sum);
        } else {
            mha_single_token_kernel<float, float>(query,
                                                present_key,
                                                present_value,
                                                alibi_mask,
                                                attention_mask,
                                                beams,
                                                output_emb,
                                                buf_attn_w,
                                                buf_attn_score,
                                                has_out_transpose,
                                                auto_causal,
                                                d_scale,
                                                past_k_scale_zp,
                                                past_v_scale_zp,
                                                head_sum);
        }
    } else {
        OPENVINO_THROW("Unsupported precision: ", query.get_precision());
    }
}
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
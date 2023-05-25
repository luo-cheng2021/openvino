#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <thread>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif
#include "mvn_custom.hpp"
#include "common_custom.hpp"

namespace ov {
namespace intel_cpu {

static float sum(bfloat16* src, size_t ele_num) {
    size_t i = 0;
    auto one = _mm512_set1_epi32(0x3f803f80);
    __m512 s;
    s = _mm512_xor_ps(s, s);
    for (; i < ele_num / 32 * 32; i += 32) {
        auto a0 = _mm512_loadu_epi16(src);
        s = _mm512_dpbf16_ps(s, (__m512bh)a0, (__m512bh)one);
        src += 32;
    }
    if (i != ele_num) {
        __mmask32 msk = _cvtu32_mask32(0xFFFFFFFFu >> (32 - (ele_num % 32)));
        auto a0 = _mm512_maskz_loadu_epi16(msk, src);
        s = _mm512_dpbf16_ps(s, (__m512bh)a0, (__m512bh)one);
    }
    // https://stackoverflow.com/questions/26896432/horizontal-add-with-m512-avx512
    return _mm512_reduce_add_ps(s);
}

static float sum_power2(bfloat16* src, float mean, size_t ele_num) {
    size_t i = 0;
    __m512 s;
    s = _mm512_xor_ps(s, s);
    auto m = _mm512_set1_ps(mean);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        s = _mm512_fmadd_ps(a0_f, a0_f, s);
        src += 16;
    }
    if (i != ele_num) {
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_maskz_sub_ps(msk, a0_f, m);
        s = _mm512_fmadd_ps(a0_f, a0_f, s);
    }
    return _mm512_reduce_add_ps(s);
}

static void mvn(bfloat16* src, float mean, float var, size_t ele_num, bfloat16* dst) {
    size_t i = 0;
    auto m = _mm512_set1_ps(mean);
    auto v = _mm512_set1_ps(var);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        auto regOut = _mm512_cvtne2ps_pbh(a0_f, a0_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), _mm512_extracti64x4_epi64(regOut, 0));

        src += 16;
        dst += 16;
    }
    if (i != ele_num) {
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        auto regOut = _mm512_cvtne2ps_pbh(a0_f, a0_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_mask_storeu_epi16(dst, msk, _mm512_extracti64x4_epi64(regOut, 0));
    }
}

static void mvn_i8(bfloat16* src, float mean, float var, size_t ele_num, int8_t* dst, float* quant) {
    size_t i = 0;
    auto m = _mm512_set1_ps(mean);
    auto v = _mm512_set1_ps(var);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto q = _mm512_loadu_ps(quant);
        auto a0 = _mm256_loadu_epi16(src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        a0_f = _mm512_mul_ps(a0_f, q);
        auto a0_i = _mm512_cvtps_epi32(a0_f);
        auto a0_i8 = _mm512_cvtsepi32_epi8(a0_i);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), a0_i8);

        src += 16;
        dst += 16;
        quant += 16;
    }
    if (i != ele_num) {
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto q = _mm512_maskz_loadu_ps(msk, quant);
        auto a0 = _mm256_maskz_loadu_epi16(msk, src);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        a0_f = _mm512_mul_ps(a0_f, q);
        auto a0_i = _mm512_cvtps_epi32(a0_f);
        auto a0_i8 = _mm512_cvtsepi32_epi8(a0_i);
        store_n(a0_i8, ele_num % 16, dst);
    }
}

static void mvn_weight_bias(bfloat16* src, float mean, float var, size_t ele_num, bfloat16* dst, float* weight, float* bias) {
    size_t i = 0;
    auto m = _mm512_set1_ps(mean);
    auto v = _mm512_set1_ps(var);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(src);
        auto w = _mm512_loadu_ps(weight);
        auto b = _mm512_loadu_ps(bias);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        a0_f = _mm512_fmadd_ps(a0_f, w, b);
        auto regOut = _mm512_cvtne2ps_pbh(a0_f, a0_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), _mm512_extracti64x4_epi64(regOut, 0));

        src += 16;
        dst += 16;
        weight += 16;
        bias += 16;
    }
    if (i != ele_num) {
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, src);
        auto w = _mm512_maskz_loadu_ps(msk, weight);
        auto b = _mm512_maskz_loadu_ps(msk, bias);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        a0_f = _mm512_fmadd_ps(a0_f, w, b);
        auto regOut = _mm512_cvtne2ps_pbh(a0_f, a0_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_mask_storeu_epi16(dst, msk, _mm512_extracti64x4_epi64(regOut, 0));
    }
}

static void mvn_i8_weight_bias(bfloat16* src, float mean, float var, size_t ele_num, int8_t* dst, float* quant, float* weight, float* bias) {
    size_t i = 0;
    auto m = _mm512_set1_ps(mean);
    auto v = _mm512_set1_ps(var);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto q = _mm512_loadu_ps(quant);
        auto a0 = _mm256_loadu_epi16(src);
        auto w = _mm512_loadu_ps(weight);
        auto b = _mm512_loadu_ps(bias);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        a0_f = _mm512_fmadd_ps(a0_f, w, b);
        a0_f = _mm512_mul_ps(a0_f, q);
        auto a0_i = _mm512_cvtps_epi32(a0_f);
        auto a0_i8 = _mm512_cvtsepi32_epi8(a0_i);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(dst), a0_i8);

        src += 16;
        dst += 16;
        quant += 16;
        weight += 16;
        bias += 16;
    }
    if (i != ele_num) {
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto q = _mm512_maskz_loadu_ps(msk, quant);
        auto a0 = _mm256_maskz_loadu_epi16(msk, src);
        auto w = _mm512_maskz_loadu_ps(msk, weight);
        auto b = _mm512_maskz_loadu_ps(msk, bias);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        a0_f = _mm512_sub_ps(a0_f, m);
        a0_f = _mm512_mul_ps(a0_f, v);
        a0_f = _mm512_fmadd_ps(a0_f, w, b);
        a0_f = _mm512_mul_ps(a0_f, q);
        auto a0_i = _mm512_cvtps_epi32(a0_f);
        auto a0_i8 = _mm512_cvtsepi32_epi8(a0_i);
        store_n(a0_i8, ele_num % 16, dst);
    }
}

void mvn_line(bfloat16* src, size_t ele_num, float eps, bool inside_sqrt, bfloat16 *dst) {
    // mean
    float mean = sum(src, ele_num) / ele_num;
    // var
    float var = sum_power2(src, mean, ele_num) / ele_num;
    var = 1.0f / (inside_sqrt ? std::sqrt(var + eps) : std::sqrt(var) + eps);
    // mvn
    mvn(src, mean, var, ele_num, dst);
}

void mvn_line(bfloat16* src, size_t ele_num, float eps, bool inside_sqrt, int8_t *dst, float* quant) {
    // mean
    float mean = sum(src, ele_num) / ele_num;
    // var
    float var = sum_power2(src, mean, ele_num) / ele_num;
    var = 1.0f / (inside_sqrt ? std::sqrt(var + eps) : std::sqrt(var) + eps);
    // mvn
    mvn_i8(src, mean, var, ele_num, dst, quant);
}

void mvn_line_weight_bias(bfloat16* src, size_t ele_num, float eps, bool inside_sqrt, bfloat16 *dst, float* weight, float* bias) {
    // mean
    float mean = sum(src, ele_num) / ele_num;
    // var
    float var = sum_power2(src, mean, ele_num) / ele_num;
    var = 1.0f / (inside_sqrt ? std::sqrt(var + eps) : std::sqrt(var) + eps);
    // mvn
    mvn_weight_bias(src, mean, var, ele_num, dst, weight, bias);
}

void mvn_line_weight_bias(bfloat16* src, size_t ele_num, float eps, bool inside_sqrt, int8_t *dst, float* quant, float* weight, float* bias) {
    // mean
    float mean = sum(src, ele_num) / ele_num;
    // var
    float var = sum_power2(src, mean, ele_num) / ele_num;
    var = 1.0f / (inside_sqrt ? std::sqrt(var + eps) : std::sqrt(var) + eps);
    // mvn
    mvn_i8_weight_bias(src, mean, var, ele_num, dst, quant, weight, bias);
}

}
}
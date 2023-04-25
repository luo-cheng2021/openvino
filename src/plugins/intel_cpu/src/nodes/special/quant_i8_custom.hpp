#pragma once

#include <stdint.h>
#include <openvino/core/type/bfloat16.hpp>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif
#include "common_custom.hpp"

namespace ov {
namespace intel_cpu {

inline void quant_i8(void* dst, void* src, size_t ele_num, float scale) {
    size_t i = 0;
    bfloat16* a = reinterpret_cast<bfloat16*>(src);
    int8_t* d = reinterpret_cast<int8_t*>(dst);
    auto s = _mm512_set1_ps(scale);
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(a);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto d_f = _mm512_mul_ps(a0_f, s);
        auto d_i = _mm512_cvtps_epi32(d_f);
        auto d_i8 = _mm512_cvtsepi32_epi8(d_i);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(d), d_i8);
        a += 16;
        d += 16;
    }
    if (i != ele_num) {
        // https://stackoverflow.com/questions/40391708/convert-16-bit-mask-mmask16-to-m128i-control-byte-mask-on-knl-xeon-phi-72
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, a);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto d_f = _mm512_mul_ps(a0_f, s);
        auto d_i = _mm512_cvtps_epi32(d_f);
        auto d_i8 = _mm512_cvtsepi32_epi8(d_i);
        store_n(d_i8, ele_num % 16, d);
    }
}

// NOTE: did not handle tail because there should be enough room
inline void cvt_i32_f32(float* dst, int32_t* src, size_t ele_num) {
    for (int i = 0; i < (ele_num + 15) / 16 * 16; i += 16) {
        auto a0 = _mm512_load_epi32(src);
        auto a_f = _mm512_cvtepi32_ps(a0);
        _mm512_storeu_ps(dst, a_f);
        src += 16;
        dst += 16;
    }
}

}
}
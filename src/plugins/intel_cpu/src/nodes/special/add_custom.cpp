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
#include "add_custom.hpp"

namespace ov {
namespace intel_cpu {

/// Convert Packed BF16 Data to Packed float Data.
///
/// \headerfile <x86intrin.h>
///
/// \param __A
///    A 256-bit vector of [16 x bfloat].
/// \returns A 512-bit vector of [16 x float] come from convertion of __A
static __inline__ __m512 _mm512_cvtpbh_ps(__m256bh __A) {
  return _mm512_castsi512_ps((__m512i)_mm512_slli_epi32(
      (__m512i)_mm512_cvtepi16_epi32((__m256i)__A), 16));
}

void add3(bfloat16* a, bfloat16 *b, bfloat16 *c, bfloat16 *dst, size_t ele_num) {
    size_t i = 0;
    for (; i < ele_num / 16 * 16; i += 16) {
        auto a0 = _mm256_loadu_epi16(a);
        auto b0 = _mm256_loadu_epi16(b);
        auto c0 = _mm256_loadu_epi16(c);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto b0_f = _mm512_cvtpbh_ps((__m256bh)b0);
        auto c0_f = _mm512_cvtpbh_ps((__m256bh)c0);
        auto d_f = _mm512_add_ps(a0_f, b0_f);
        d_f = _mm512_add_ps(d_f, c0_f);
        auto regOut = _mm512_cvtne2ps_pbh(d_f, d_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst), _mm512_extracti64x4_epi64(regOut, 0));
        a += 16;
        b += 16;
        c += 16;
        dst += 16;
    }
    if (i != ele_num) {
        // https://stackoverflow.com/questions/40391708/convert-16-bit-mask-mmask16-to-m128i-control-byte-mask-on-knl-xeon-phi-72
        __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (ele_num % 16)));
        auto a0 = _mm256_maskz_loadu_epi16(msk, a);
        auto b0 = _mm256_maskz_loadu_epi16(msk, b);
        auto c0 = _mm256_maskz_loadu_epi16(msk, c);
        auto a0_f = _mm512_cvtpbh_ps((__m256bh)a0);
        auto b0_f = _mm512_cvtpbh_ps((__m256bh)b0);
        auto c0_f = _mm512_cvtpbh_ps((__m256bh)c0);
        auto d_f = _mm512_add_ps(a0_f, b0_f);
        d_f = _mm512_add_ps(d_f, c0_f);
        auto regOut = _mm512_cvtne2ps_pbh(d_f, d_f); // only 16 bfloat16 results in lower 256bits 
        _mm256_mask_storeu_epi16(dst, msk, _mm512_extracti64x4_epi64(regOut, 0));
    }
}

}
}
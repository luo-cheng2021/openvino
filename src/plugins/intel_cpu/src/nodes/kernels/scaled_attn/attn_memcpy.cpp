// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <assert.h>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <limits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "common.hpp"
#include "attn_memcpy.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

// float16 <- float
template<typename TA, typename TB>
void attn_copy_inner(TA* a, TB* b, size_t n) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto vb = mm512_uni_loadu_ps(b + i);
        mm512_uni_storeu_ps(a + i, vb);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto vb = mm256_uni_loadu_ps(b + i);
        mm256_uni_storeu_ps(a + i, vb);
    }
#endif
    for (; i < n; i++) {
        uni_store_to_float16(&a[i], b[i]);
    }
}

void attn_copy(void* a, void* b, size_t N, ov::element::Type a_precision, ov::element::Type b_precision) {
    if (a_precision == b_precision) {
        memcpy(a, b, N * a_precision.size());
    } else {
        assert(a_precision == ov::element::f16);
        assert(b_precision == ov::element::f32);
        auto a_ptr = static_cast<ov::float16*>(a);
        auto b_ptr = static_cast<float*>(b);
        attn_copy_inner(a_ptr, b_ptr, N);
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
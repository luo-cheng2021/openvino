// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This file include jit and raw function wrapper define. Compiling in clang.
// Note: The function jit called should not contain static class variable
//   because we do not resolve the undefined _cxx_* problem
#include "mkldnn_adaptive_pooling_opt.h"
#include <math.h>
#include <vector>
#include <easy/jit.h>
#include "xsimd.h"

template <class Arch>
void poolAvgT(Arch, const float *srcData, float *dstData, int od, int oh, int ow, size_t spatIndOff,
    const size_t inStrides[5], size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW);
static void poolAvgInner(const xsimd::detail::supported_arch arch, const float *srcData, float *dstData, int od, int oh, int ow, size_t spatIndOff,
    const size_t inStrides[5], size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW) {
    if (arch.avx512f)
        poolAvgT(xsimd::avx512f(), srcData, dstData, od, oh, ow, spatIndOff, inStrides, ID, OD, IH, OH, IW, OW);
    else if (arch.avx2)
        poolAvgT(xsimd::avx2(), srcData, dstData, od, oh, ow, spatIndOff, inStrides, ID, OD, IH, OH, IW, OW);
    else if (arch.sse4_1)
        poolAvgT(xsimd::sse4_1(), srcData, dstData, od, oh, ow, spatIndOff, inStrides, ID, OD, IH, OH, IW, OW);
}

void poolAvg(const float *srcData, float *dstData, int od, int oh, int ow, size_t spatIndOff,
    const size_t inStrides[5], size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW) {
    poolAvgInner(xsimd::available_architectures(), srcData, dstData, od, oh, ow, spatIndOff, inStrides, ID, OD, IH, OH, IW, OW);
}

AvgFunc getAvgFunc(const easy::RawBytes& raw, size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW) {
    using namespace std::placeholders;
    auto avg = easy::jit_(raw, poolAvgInner, xsimd::available_architectures(), _1, _2, _3, _4, _5, _6, _7, ID, OD, IH, OH, IW, OW);
    return avg;
}

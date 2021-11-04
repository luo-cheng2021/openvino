// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// This file include common function needed by the optimization function
// Note: 1, Common function should be static to avoid linker pick the wrong architecture version
// 2, if want use specific instrisc can use the predefined macro: HAVE_AVX2/HAVE_SSE42/HAVE_AVX512
#include <math.h>
#include <vector>
#include <easy/jit.h>
#include "xsimd.h"
#include "fuse_private.h"

static inline void setBinBorders(size_t *startPtr, size_t *endPtr, size_t idx, size_t inputLength, size_t outputLength) {
    *(startPtr) = idx * inputLength / outputLength;
    *(endPtr) = ceil(static_cast<float>((idx + 1) * inputLength) / outputLength);
}

template <class Arch>
void poolAvgT(Arch*, const float *srcData, float *dstData, int od, int oh, int ow, size_t spatIndOff,
    const size_t inStrides[5], size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW, const FuseConstAlgParamPrivate<float, Arch> params_c, const FuseMutableParams& params_m) {
    using b_type = xsimd::batch<float, Arch>;
    //std::size_t inc = b_type::size;

    size_t dStart, dEnd, hStart, hEnd, wStart, wEnd;
    setBinBorders(&dStart, &dEnd, od, ID, OD);
    setBinBorders(&hStart, &hEnd, oh, IH, OH);
    setBinBorders(&wStart, &wEnd, ow, IW, OW);
    auto binSize = (dEnd - dStart) * (hEnd - hStart) * (wEnd - wStart);
    // if (binSize == 0)
    //     IE_THROW() << errorPrefix << "has empty bin";
    //float sum = 0;
    b_type sum(0);
    for (size_t pixD = dStart; pixD < dEnd; pixD++) {
        for (size_t pixH = hStart; pixH < hEnd; pixH++) {
            for (size_t pixW = wStart; pixW < wEnd; pixW++) {
                //float curr = srcData[pixD * inStrides[2] + pixH * inStrides[3] + pixW * inStrides[4]];
                auto offset = pixD * inStrides[2] + pixH * inStrides[3] + pixW * inStrides[4];
                b_type curr = b_type::load(&srcData[offset], xsimd::unaligned_mode());
                //sum = sum + curr;
                sum += curr;
            }
        }
    }
    //*dstData = sum / binSize;
    auto res = sum / binSize;
    res = seq_fuse(res, 0, 0, 0, params_m, params_c);
    xsimd::store(dstData, res, xsimd::unaligned_mode());
}

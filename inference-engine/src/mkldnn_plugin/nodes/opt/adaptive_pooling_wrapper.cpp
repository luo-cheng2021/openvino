// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This file include jit and raw function wrapper define. Compiling in clang.
// Note: The function jit called should not contain static class variable
//   because we do not resolve the undefined _cxx_* problem
#include <math.h>
#include <vector>
#include <easy/jit.h>
#include "xsimd.h"
#include "adaptive_pooling_public.h"
#include "fuse_private.h"

template <class Arch>
void poolAvgT(Arch*, const float *srcData, float *dstData, int od, int oh, int ow, size_t spatIndOff,
    const size_t inStrides[5], size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW, const FuseConstAlgParamPrivate<float, Arch> param_c, const FuseMutableParams& params_m);

void poolAvg(const float *srcData, float *dstData, int od, int oh, int ow, size_t spatIndOff,
    const size_t inStrides[5], size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW, const FuseConstParams& params_c, const FuseMutableParams& params_m) {
    auto arch = xsimd::available_architectures();
    if (arch.avx512f) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::avx512f>(params_c);
        xsimd::avx512f arch;
        poolAvgT(&arch, srcData, dstData, od, oh, ow, spatIndOff, inStrides,
            ID, OD, IH, OH, IW, OW, paramPrivate, params_m);
    }
    else if (arch.avx2) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::avx2>(params_c);
        xsimd::avx2 arch;
        poolAvgT(&arch, srcData, dstData, od, oh, ow, spatIndOff, inStrides,
            ID, OD, IH, OH, IW, OW, paramPrivate, params_m);
    }
    else if (arch.sse4_2) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::sse4_2>(params_c);
        xsimd::sse4_2 arch;
        poolAvgT(&arch, srcData, dstData, od, oh, ow, spatIndOff, inStrides,
            ID, OD, IH, OH, IW, OW, paramPrivate, params_m);
    }
}

AvgFunc getAvgFunc(const easy::RawBytes& raw, size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW, const FuseConstParams& params) {
    using namespace std::placeholders;
    auto arch = xsimd::available_architectures();
    if (arch.avx512f) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::avx512f>(params);
        xsimd::avx512f arch;
        auto avg = easy::jit_(raw, poolAvgT<xsimd::avx512f>, &arch, _1, _2, _3, _4, _5, _6, _7,
            ID, OD, IH, OH, IW, OW, paramPrivate, _8);
        return avg;
    }
    else if (arch.avx2) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::avx2>(params);
        xsimd::avx2 arch;
        auto avg = easy::jit_(raw, poolAvgT<xsimd::avx2>, &arch, _1, _2, _3, _4, _5, _6, _7,
            ID, OD, IH, OH, IW, OW, paramPrivate, _8, easy::options::dump_ir("zzz.ll"),
            easy::options::opt_level(2, 0));
        return avg;
    }
    else if (arch.sse4_2) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::sse4_2>(params);
        xsimd::sse4_2 arch;
        auto avg = easy::jit_(raw, poolAvgT<xsimd::sse4_2>, &arch, _1, _2, _3, _4, _5, _6, _7,
            ID, OD, IH, OH, IW, OW, paramPrivate, _8);
        return avg;
    }
    return AvgFunc{nullptr};
}

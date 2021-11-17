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
#include "eltwise_public.h"
#include "fuse_private.h"

template <class Type, class Arch>
void eltwiseT(Arch*, const EltwiseConstParam params_c, const EltwiseMutableParam& params_m, const FuseConstAlgParamPrivate<Type, Arch> fuse_params_c, const FuseMutableParams& fuse_params_m);

void eltwise(const EltwiseConstParam& params_c, const EltwiseMutableParam& params_m, const FuseConstParams& fuse_params_c, const FuseMutableParams& fuse_params_m) {
    auto arch = xsimd::available_architectures();
    if (arch.avx512f) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::avx512f>(fuse_params_c);
        xsimd::avx512f arch;
        eltwiseT(&arch, params_c, params_m, paramPrivate, fuse_params_m);
    }
    else if (arch.avx2) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::avx2>(fuse_params_c);
        xsimd::avx2 arch;
        eltwiseT(&arch, params_c, params_m, paramPrivate, fuse_params_m);
    }
    else if (arch.sse4_2) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::sse4_2>(fuse_params_c);
        xsimd::sse4_2 arch;
        eltwiseT(&arch, params_c, params_m, paramPrivate, fuse_params_m);
    }
}

EltwiseFunc getEltwiseFunc(const easy::RawBytes& raw, const EltwiseConstParam& params_c, const FuseConstParams& fuse_params_c) {
    using namespace std::placeholders;
    auto arch = xsimd::available_architectures();
    if (arch.avx512f) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::avx512f>(fuse_params_c);
        xsimd::avx512f arch;
        auto f = easy::jit_(raw, eltwiseT<float, xsimd::avx512f>, &arch, params_c, _1, paramPrivate, _2);
        return f;
    }
    else if (arch.avx2) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::avx2>(fuse_params_c);
        xsimd::avx2 arch;
        auto f = easy::jit_(raw, eltwiseT<float, xsimd::avx2>, &arch, params_c, _1, paramPrivate, _2,
            easy::options::dump_ir("zzz.ll"), easy::options::opt_level(2, 0));
        return f;
    }
    else if (arch.sse4_2) {
        const auto paramPrivate = ConvertFuseParams<float, xsimd::sse4_2>(fuse_params_c);
        xsimd::sse4_2 arch;
        auto f = easy::jit_(raw, eltwiseT<float, xsimd::sse4_2>, &arch, params_c, _1, paramPrivate, _2);
        return f;
    }
    return EltwiseFunc{nullptr};
}

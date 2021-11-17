// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This file include avx2 specific optimization function. Compiling in clang.
// Note: 1, avoid use static class variable
//       2, do not depend function in other cpp
#include "eltwise_private.h"

template void EASY_JIT_EXPOSE eltwiseT<float, xsimd::avx2>(xsimd::avx2*, const EltwiseConstParam params_c, const EltwiseMutableParam& params_m,
    const FuseConstAlgParamPrivate<float, xsimd::avx2> fuse_params_c, const FuseMutableParams& fuse_params_m);

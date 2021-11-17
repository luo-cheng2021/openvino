// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This file include jit and raw function declare. Compiling in both gcc and clang.
// Note: Because this file will be seen both in gcc and clang we should avoid
//   define too complex class here
#pragma once

#include "fuse_public.h"
#include <easy/jit.h>

struct EltwiseMutableParam {
    void *src_x;    // src x addr
    int strides_x[3];

    void *src_y;    // src y addr
    int strides_y[3];

    void *dst_d;    // dst addr
    int strides_d[3];

    int dims[3];
    // EltwiseMutableParam() {
    //     memset(this, 0, sizeof(*this));
    // }
};

struct EltwiseConstParam {
    AlgType alg_type;
    float alpha;
    float beta;
    int precision_x;       // input precision
    int precision_y;
    int precision_inner;   // compute will use this precision
    int precision_d;       // dst precision
    EltwiseConstParam() {
        memset(this, 0, sizeof(*this));
    }
};

using EltwiseFunc = easy::FunctionWrapper<void(const EltwiseMutableParam&, const FuseMutableParams&)>;

// raw optimization function
//  when debugging we can call this function directly
void eltwise(const EltwiseConstParam& params_c, const EltwiseMutableParam& params_m, const FuseConstParams& fuse_params_c, const FuseMutableParams& fuse_params_m);

// get jit function, the parameters will become runtime constants
//  when debugging complete we should
//  1, Add a AvgFunc variable such as _avg in the class
//  2, Make the call '_avg = getAvgFunc' if _avg is null
//  3, Call _avg() to do the actual compute
// TODO: multithread support
EltwiseFunc getEltwiseFunc(const easy::RawBytes& raw, const EltwiseConstParam& params_c, const FuseConstParams& fuse_params_c);

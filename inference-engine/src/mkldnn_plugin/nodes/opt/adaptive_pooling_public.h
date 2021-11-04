// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// This file include jit and raw function declare. Compiling in both gcc and clang.
// Note: Because this file will be seen both in gcc and clang we should avoid
//   define too complex class here
#pragma once

#include "fuse_public.h"
#include <easy/jit.h>

using AvgFunc = easy::FunctionWrapper<void(const float *, float *, int, int, int, size_t, const size_t inStrides[5], const FuseMutableParams&)>;

// raw optimization function
//  when debugging we can call this function directly
void poolAvg(const float *srcData, float *dstData, int od, int oh, int ow, size_t spatIndOff,
    const size_t inStrides[5], size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW, const FuseConstParams& params_c, const FuseMutableParams& params_m);

// get jit function, the parameters will become runtime constants
//  when debugging complete we should
//  1, Add a AvgFunc variable such as _avg in the class
//  2, Make the call '_avg = getAvgFunc' if _avg is null
//  3, Call _avg() to do the actual compute
// TODO: multithread support
AvgFunc getAvgFunc(const easy::RawBytes& raw, size_t ID, size_t OD, size_t IH, size_t OH, size_t IW, size_t OW, const FuseConstParams& params_c);

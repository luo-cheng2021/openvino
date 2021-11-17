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
#include "eltwise_public.h"

template <class Type, class Arch>
void eltwiseT(Arch*, const EltwiseConstParam params_c, const EltwiseMutableParam& params_m, const FuseConstAlgParamPrivate<Type, Arch> fuse_params_c, const FuseMutableParams& fuse_params_m) {
    constexpr std::size_t inc = b_t<Type, Arch>::size;

    const auto *src_x_k = (uint8_t*)params_m.src_x;
    const auto *src_y_k = (uint8_t*)params_m.src_y;
    const auto *dst_d_k = (uint8_t*)params_m.dst_d;
    for (int k = 0; k < params_m.dims[0]; k++) {
        const uint8_t* src_x = src_x_k;
        const uint8_t* src_y = src_y_k;
        const uint8_t* dst_d = dst_d_k;
        for (int j = 0; j < params_m.dims[1]; j++) {
            int i = 0;
            for (; i < params_m.dims[2] / inc; i++) {
                // TODO: add type convert here
                auto x = b_t<Type, Arch>::load((Type*)(src_x + i * inc * sizeof(Type)), xsimd::unaligned_mode());
                auto y = b_t<Type, Arch>::load((Type*)(src_y + i * inc * sizeof(Type)), xsimd::unaligned_mode());
                b_t<Type, Arch> d;
                if (params_c.alg_type == AlgType::Add) {
                    d = x + y;
                }
                // TODO: FIXME: check if need a new function to update the address in fuse params
                d = seq_fuse<Type, Arch>(d, i, j, k, fuse_params_m, fuse_params_c);
                // TODO: add type convert here
                d.store_unaligned((Type*)(dst_d + i * inc * sizeof(Type)));
            }
            if (params_m.dims[2] % inc) {
                Type buf[inc];
                // TODO: FIXME: malloc more data at least align simd width
                auto x = b_t<Type, Arch>::load((Type*)(src_x + i * inc * sizeof(Type)), xsimd::unaligned_mode());
                auto y = b_t<Type, Arch>::load((Type*)(src_y + i * inc * sizeof(Type)), xsimd::unaligned_mode());
                b_t<Type, Arch> d;
                if (params_c.alg_type == AlgType::Add) {
                    d = x + y;
                }
                // TODO: FIXME: check if need a new function to update the address in fuse params
                d = seq_fuse<Type, Arch>(d, i, j, k, fuse_params_m, fuse_params_c);
                d.store_unaligned(buf);
                memcpy((void*)(dst_d + i * inc * sizeof(Type)), (void*)buf, (params_m.dims[2] % inc) * sizeof(Type));
            }
            src_x += params_m.strides_x[1];
            src_y += params_m.strides_y[1];
            dst_d += params_m.strides_d[1];
        }
        src_x_k += params_m.strides_x[0];
        src_y_k += params_m.strides_y[0];
        dst_d_k += params_m.strides_d[0];
    }
}

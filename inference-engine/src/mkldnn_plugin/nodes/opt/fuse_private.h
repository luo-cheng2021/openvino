// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// This file includes all fused function needed by the optimization function
// Note: 1, Common function should be static to avoid linker pick the wrong architecture version
// 2, if want use specific instrisc can use the predefined macro: HAVE_AVX2/HAVE_SSE42/HAVE_AVX512

#include <utility>
#include <map>
#include <easy/jit.h>
#include "xsimd.h"
#include "fuse_public.h"

template<class Type, class Arch>
using b_t = xsimd::batch<Type, Arch>;
struct CallerContext {
    int x, y, z;    // loop idx, inner -> outer
};

template<class Type, class Arch>
using b_func_type = b_t<Type, Arch>(*)(b_t<Type, Arch>, const CallerContext context, const FuseMutableAlgParam, const FuseConstAlgParam);

template<class Type, class Arch>
struct FuseConstAlgParamPrivate {
    b_func_type<Type, Arch> funcs[MAX_FUSE_NUM];
    int funcs_num;
    FuseConstAlgParam params[MAX_FUSE_NUM];
};

#define DECLARE_EXTERN(func) \
    extern template b_t<float, xsimd::sse4_2> func(b_t<float, xsimd::sse4_2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c); \
    extern template b_t<float, xsimd::avx2> func(b_t<float, xsimd::avx2> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c); \
    extern template b_t<float, xsimd::avx512f> func(b_t<float, xsimd::avx512f> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c);

#define ADDR(c, p, type) (type*)(p.addr + c.z * p.stride_xy + c.y * p.stride_x + c.x * b_t<T, A>::size * sizeof(T))
template<class T, class A>
b_t<T, A> add(b_t<T, A> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c) {
    b_t<T, A> y = b_t<T, A>::load(ADDR(context, param_m, float), xsimd::unaligned_mode());
    return x + y;
}
DECLARE_EXTERN(add)

template<class T, class A>
b_t<T, A> sub(b_t<T, A> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c) {
    b_t<T, A> y = b_t<T, A>::load(ADDR(context, param_m, float), xsimd::unaligned_mode());
    return x - y;
}
DECLARE_EXTERN(sub)

template<class T, class A>
b_t<T, A> mul(b_t<T, A> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c) {
    b_t<T, A> y = b_t<T, A>::load(ADDR(context, param_m, float), xsimd::unaligned_mode());
    return x * y;
}
DECLARE_EXTERN(mul)

template<class T, class A>
b_t<T, A> abs(b_t<T, A> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c) {
    return xsimd::abs(x);
}
DECLARE_EXTERN(abs)

template<class T, class A>
b_t<T, A> add_c(b_t<T, A> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c) {
    b_t<T, A> c(param_c.x1);
    return x + c;
}
DECLARE_EXTERN(add_c)

template<class T, class A>
b_t<T, A> sub_c(b_t<T, A> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c) {
    b_t<T, A> c(param_c.x1);
    return x - c;
}
DECLARE_EXTERN(sub_c)

template<class T, class A>
b_t<T, A> mul_c(b_t<T, A> x, const CallerContext context, const FuseMutableAlgParam param_m, const FuseConstAlgParam param_c) {
    b_t<T, A> c(param_c.x1);
    return x * c;
}
DECLARE_EXTERN(mul_c)

// convert
template<class Type, class Arch>
FuseConstAlgParamPrivate<Type, Arch> ConvertFuseParams(const FuseConstParams& params) {
    static std::map<AlgType, b_func_type<Type, Arch>> all_funcs = {
        {AlgType::Abs, abs<Type, Arch>},
        {AlgType::Add, add<Type, Arch>},
        {AlgType::Sub, sub<Type, Arch>},
        {AlgType::Mul, mul<Type, Arch>},
        {AlgType::Add_C, add_c<Type, Arch>},
        {AlgType::Sub_C, sub_c<Type, Arch>},
        {AlgType::Mul_C, mul_c<Type, Arch>},
    };
    FuseConstAlgParamPrivate<Type, Arch> paramPrivate;
    memset(&paramPrivate, 0, sizeof(paramPrivate));
    for (size_t i = 0; i < params.num; i++) {
        paramPrivate.funcs[i] = all_funcs[params.types[i]];
    }

    paramPrivate.funcs_num = params.num;
    memcpy(paramPrivate.params, params.params, sizeof(paramPrivate.params));
    
    return paramPrivate;
}

// sequence fuse interface: result = g(f(x))
template<class Type, class Arch>
static b_t<Type, Arch> seq_fuse(b_t<Type, Arch> x, int idx_x, int idx_y, int idx_z, const FuseMutableParams& param_m, const FuseConstAlgParamPrivate<Type, Arch>& params) {
    CallerContext context{idx_x, idx_y, idx_z};
    for (int i = 0; i < params.funcs_num; i++) {
        x = params.funcs[i](x, context, param_m.params[i], params.params[i]);
    }

    return x;
}

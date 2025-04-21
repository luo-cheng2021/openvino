// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"


#if TO_TILED
KERNEL (linear_to_tiled)(
    const __global uchar* src,
    __global uchar* dst)
{
    int K = get_global_size(0);
    int N = get_global_size(1);

    int k0 = get_global_id(0);
    int n0 = get_global_id(1);

    src += n0 * K + k0;
    dst += (n0 / 32) * ( 32 * K) + k0 * 32 + (n0 % 32);

    dst[0] = src[0];
}

#else

KERNEL (tiled_to_linear) (
    const __global uchar* src,
    __global uchar* dst)
{
    int K = get_global_size(0);
    int N = get_global_size(1);

    int k0 = get_global_id(0);
    int n0 = get_global_id(1);

    dst += n0 * K + k0;
    src += (n0 / 32) * ( 32 * K) + k0 * 32 + (n0 % 32);

    dst[0] = src[0];
}
#endif


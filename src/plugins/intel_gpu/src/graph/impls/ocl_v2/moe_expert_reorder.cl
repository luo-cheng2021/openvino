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

    uchar value_0 = src[0];
    uchar value_1 = src[N/2];

    uchar value = (n0 & 0) ? ((value_1 & 0xf) << 4) | (value_0 & 0xf) : (value_1 & 0xf0) | ((value_0 & 0xf0) >> 4); 
    intel_sub_group_block_write_uc((__global ushort *)dst , value);
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

    uchar value = intel_sub_group_block_read_uc((const __global ushort *)src);
    if(n0 & 0) {
        dst[0] = (value & 0xf) | ((sub_group_broadcast(value, n0 + 1) & 0xf) << 4);
        dst[N/2] = ((value & 0xf0) >> 4) | (sub_group_broadcast(value, n0 + 1) & 0xf0);
    }

}
#endif


// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#define WEI_UINT4 1

typedef struct {
    __global void* weight[3];
    __global void* zp[3];
    __global void* scale[3];
    int routing_offset;
    int pad;
} FUNC(expert_info);


#if GATE_UP_ENABLE

inline int get_4bit_weight_index_no_isv(int k, int n, int K, int N, int OSV) {
    return (n / OSV) * (OSV * K / 2) + (k / 2) * OSV;
}

inline void thread_task_splitter(const int group_num, const int thr_num, const int thr_id, int* n_start, int* n_end) {
    if (thr_num <= 1 || group_num == 0) {
        *n_start = 0;
        *n_end = group_num;
    } else {
        int num = (group_num + thr_num - 1) / thr_num;
        int num_minus = num - 1;
        int last = group_num - num_minus * thr_num;
        *n_end = thr_id < last ? num : num_minus;
        *n_start = thr_id <= last ? thr_id * num : last * num + (thr_id - last) * num_minus;
    }
    *n_end += *n_start;
}

inline tile_gemv(
    const __global uchar* weights,
    const __global half* scales,
    const __global uchar* zps,
    __global half* input,   // [1, K]
    __global half* output,  // [1, N]
    int N, int K,
    __local float* all_sum_even,
    const bool silu
    ) {
    // global:[X, N, 16]
    // local: [1, SUBGROUP_SIZE, 16]
    int n = get_global_id(1);              // N
    int thr_id = get_local_id(2);          // 0~thr_num-1
    int thr_num = get_local_size(2);       // 32
    int wi_id = get_sub_group_local_id();  // 0~31

    int gk0, gk1;
    int group_num = K / GROUP_SIZE;
    thread_task_splitter(group_num, thr_num, thr_id, &gk0, &gk1);

    //if(wi_id==0 && thr_id==0) {
    //    printf("gws = (%d, %d,%d), ", get_global_size(0), get_global_size(1), get_global_size(2));
    //    printf("lws = (%d, %d,%d), ", get_local_size(0), get_local_size(1), get_local_size(2));
    //    printf("K = %d, N = %d, SUBGROUP_SIZE = %d, THR_NUM=%d, ", K,N, SUBGROUP_SIZE, thr_num);
    //    printf("gk0 = %d, gk1 = %d, group_num = %d, thr_id = %d, thr_num = %d\n", gk0, gk1, group_num, thr_id, thr_num);
    //}

    // Scale layout is byfx
    scales += n;
    zps += n;

    float sum_all = 0;
    for (int gk = gk0; gk < gk1; gk++) {
        __global half* A = input + gk * GROUP_SIZE;
        const __global uchar* B =
            weights + get_4bit_weight_index_no_isv(gk * GROUP_SIZE, n, K, N, SUBGROUP_SIZE);

        float8 sum = 0;
        float scale_1 = convert_float(scales[gk * N]);
        half zpx16 = (half)(zps[gk * N]);

        __attribute__((opencl_unroll_hint(4))) for (int g = 0; g < GROUP_SIZE; g += 32, B += 16 * SUBGROUP_SIZE) {
            ushort input_value = intel_sub_group_block_read_us((const __global ushort*)(A + g));
            char16 bx16 = as_char16(intel_sub_group_block_read_uc16(B));

#    if WEI_UINT4
            half16 i4x16_even = convert_half16((bx16 & (char16)0xF)) - zpx16;
            half16 i4x16_odd = convert_half16(as_char16(as_uchar16(bx16) >> 4)) - zpx16;
#    else
            char16 i4x16_even_c16 = (bx16 & (char16)0xF);
            char16 i4x16_odd_c16 = (as_char16(as_uchar16(bx16) >> 4));
            i4x16_even_c16 = select(i4x16_even_c16, i4x16_even_c16 - (char16)16, i4x16_even_c16 > (char16)7);
            i4x16_odd_c16 = select(i4x16_odd_c16, i4x16_odd_c16 - (char16)16, i4x16_odd_c16 > (char16)7);
            half16 i4x16_even = convert_half16(i4x16_even_c16) - zpx16;
            half16 i4x16_odd = convert_half16(i4x16_odd_c16) - zpx16;
#    endif

            sum[0] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s0 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.s2 +
                      as_half(sub_group_broadcast(input_value, 8)) * i4x16_even.s4 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even.s6;
            sum[1] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s0 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.s2 +
                      as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd.s4 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd.s6;

            sum[2] += as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s1 +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.s3 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even.s5 +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even.s7;
            sum[3] += as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s1 +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.s3 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd.s5 +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd.s7;

            sum[4] += as_half(sub_group_broadcast(input_value, 16)) * i4x16_even.s8 +
                      as_half(sub_group_broadcast(input_value, 20)) * i4x16_even.sa +
                      as_half(sub_group_broadcast(input_value, 24)) * i4x16_even.sc +
                      as_half(sub_group_broadcast(input_value, 28)) * i4x16_even.se;
            sum[5] += as_half(sub_group_broadcast(input_value, 17)) * i4x16_odd.s8 +
                      as_half(sub_group_broadcast(input_value, 21)) * i4x16_odd.sa +
                      as_half(sub_group_broadcast(input_value, 25)) * i4x16_odd.sc +
                      as_half(sub_group_broadcast(input_value, 29)) * i4x16_odd.se;

            sum[6] += as_half(sub_group_broadcast(input_value, 18)) * i4x16_even.s9 +
                      as_half(sub_group_broadcast(input_value, 22)) * i4x16_even.sb +
                      as_half(sub_group_broadcast(input_value, 26)) * i4x16_even.sd +
                      as_half(sub_group_broadcast(input_value, 30)) * i4x16_even.sf;
            sum[7] += as_half(sub_group_broadcast(input_value, 19)) * i4x16_odd.s9 +
                      as_half(sub_group_broadcast(input_value, 23)) * i4x16_odd.sb +
                      as_half(sub_group_broadcast(input_value, 27)) * i4x16_odd.sd +
                      as_half(sub_group_broadcast(input_value, 31)) * i4x16_odd.sf;
        }

        sum_all += (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]) * scale_1;
    }


    *(all_sum_even + thr_num*wi_id + thr_id) = sum_all;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(thr_id==0) {
        float sum_value = 0.0;
        for (int i = 0; i < thr_num; i++) {
            sum_value += *(all_sum_even + thr_num * wi_id + i);
        }

        if (silu) {
            sum_value = sum_value / (1 + exp(-sum_value));
        }
        output[n] = sum_value;
    }
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_gate_up)(
    const __global FUNC(expert_info)* info_ptrs,
    __global TYPE* x,                        // [1, HIDDEN_SIZE]
    __global TYPE* y) {                      // [MAX_TOPK, INTERMEDIATE_SIZE]
    // gws: [expert, N, THR_NUM(16)]
    // lws: [1, SUBGROUP_SIZE, THR_NUM(16)]
    int expert_no = get_global_id(0);
    y += expert_no * INTERMEDIATE_SIZE;
    const __global FUNC(expert_info)* info_ptr = info_ptrs + expert_no;
    // up, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* up_weight = (__global uchar*)info_ptr->weight[1];
    __global half* up_scale = (__global half*)info_ptr->scale[1];
    __global uchar* up_zp = (__global uchar*)info_ptr->zp[1];
    // gate, [HIDDEN_SIZE, INTERMEDIATE_SIZE]
    __global uchar* gate_weight = (__global uchar*)info_ptr->weight[0];
    __global half* gate_scale = (__global half*)info_ptr->scale[0];
    __global uchar* gate_zp = (__global uchar*)info_ptr->zp[0];

    //__local float all_sum_even[SUBGROUP_SIZE][16];  // [wi_id, thr_id]
    __local float all_sum_even[SUBGROUP_SIZE * THR_NUM];

    //if(get_global_id(0)==0 && get_global_id(1)==0 && get_global_id(2)==0) {
    //    printf("gate:gws = (%d, %d,%d), ", get_global_size(0), get_global_size(1), get_global_size(2));
    //    printf("lws = (%d, %d,%d), ", get_local_size(0), get_local_size(1), get_local_size(2));
    //    printf("K = %d, N = %d, SUBGROUP_SIZE = %d, THR_NUM=%d\n", HIDDEN_SIZE,INTERMEDIATE_SIZE, SUBGROUP_SIZE, THR_NUM);
    //}

    tile_gemv(up_weight, up_scale, up_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, all_sum_even, false);
    tile_gemv(gate_weight, gate_scale, gate_zp, x, y, INTERMEDIATE_SIZE, HIDDEN_SIZE, all_sum_even, true);
}

#elif DOWN_ENABLE

inline int get_4bit_weight_index_no_isv_down(int k, int n, int K, int N, int OSV) {
    return (n / OSV) * (OSV * K / 2) + (k / 2) * OSV;
}

inline void thread_task_splitter_down(const int group_num, const int thr_num, const int thr_id, int* n_start, int* n_end) {
    if (thr_num <= 1 || group_num == 0) {
        *n_start = 0;
        *n_end = group_num;
    } else {
        int num = (group_num + thr_num - 1) / thr_num;
        int num_minus = num - 1;
        int last = group_num - num_minus * thr_num;
        *n_end = thr_id < last ? num : num_minus;
        *n_start = thr_id <= last ? thr_id * num : last * num + (thr_id - last) * num_minus;
    }
    *n_end += *n_start;
}

inline tile_gemv_down(
    const __global uchar* weights,
    const __global half* scales,
    const __global uchar* zps,
    __global half* input,   // [1, K]
    __global half* output,  // [1, N]
    int N, int K,
    __local float* all_sum_even,
    half routing_weights

    ) {
    // global:[X, N, 16]
    // local: [1, SUBGROUP_SIZE, 16]
    int n = get_global_id(1);              // N
    int thr_id = get_local_id(2);          // 0~thr_num-1
    int thr_num = get_local_size(2);       // 32
    int wi_id = get_sub_group_local_id();  // 0~31

    int gk0, gk1;
    int group_num = K / GROUP_SIZE;
    thread_task_splitter_down(group_num, thr_num, thr_id, &gk0, &gk1);

    // Scale layout is byfx
    scales += n;
    zps += n;

    float sum_all = 0;
    for (int gk = gk0; gk < gk1; gk++) {
        __global half* A = input + gk * GROUP_SIZE;
        const __global uchar* B =
            weights + get_4bit_weight_index_no_isv_down(gk * GROUP_SIZE, n, K, N, SUBGROUP_SIZE);

        float8 sum = 0;
        float scale_1 = convert_float(scales[gk * N]);
        half zpx16 = (half)(zps[gk * N]);

        __attribute__((opencl_unroll_hint(4))) for (int g = 0; g < GROUP_SIZE; g += 32, B += 16 * SUBGROUP_SIZE) {
            ushort input_value = intel_sub_group_block_read_us((const __global ushort*)(A + g));
            char16 bx16 = as_char16(intel_sub_group_block_read_uc16(B));

#    if WEI_UINT4
            half16 i4x16_even = convert_half16((bx16 & (char16)0xF)) - zpx16;
            half16 i4x16_odd = convert_half16(as_char16(as_uchar16(bx16) >> 4)) - zpx16;
#    else
            char16 i4x16_even_c16 = (bx16 & (char16)0xF);
            char16 i4x16_odd_c16 = (as_char16(as_uchar16(bx16) >> 4));
            i4x16_even_c16 = select(i4x16_even_c16, i4x16_even_c16 - (char16)16, i4x16_even_c16 > (char16)7);
            i4x16_odd_c16 = select(i4x16_odd_c16, i4x16_odd_c16 - (char16)16, i4x16_odd_c16 > (char16)7);
            half16 i4x16_even = convert_half16(i4x16_even_c16) - zpx16;
            half16 i4x16_odd = convert_half16(i4x16_odd_c16) - zpx16;
#    endif

            sum[0] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s0 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.s2 +
                      as_half(sub_group_broadcast(input_value, 8)) * i4x16_even.s4 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even.s6;
            sum[1] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s0 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.s2 +
                      as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd.s4 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd.s6;

            sum[2] += as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s1 +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.s3 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even.s5 +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even.s7;
            sum[3] += as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s1 +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.s3 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd.s5 +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd.s7;

            sum[4] += as_half(sub_group_broadcast(input_value, 16)) * i4x16_even.s8 +
                      as_half(sub_group_broadcast(input_value, 20)) * i4x16_even.sa +
                      as_half(sub_group_broadcast(input_value, 24)) * i4x16_even.sc +
                      as_half(sub_group_broadcast(input_value, 28)) * i4x16_even.se;
            sum[5] += as_half(sub_group_broadcast(input_value, 17)) * i4x16_odd.s8 +
                      as_half(sub_group_broadcast(input_value, 21)) * i4x16_odd.sa +
                      as_half(sub_group_broadcast(input_value, 25)) * i4x16_odd.sc +
                      as_half(sub_group_broadcast(input_value, 29)) * i4x16_odd.se;

            sum[6] += as_half(sub_group_broadcast(input_value, 18)) * i4x16_even.s9 +
                      as_half(sub_group_broadcast(input_value, 22)) * i4x16_even.sb +
                      as_half(sub_group_broadcast(input_value, 26)) * i4x16_even.sd +
                      as_half(sub_group_broadcast(input_value, 30)) * i4x16_even.sf;
            sum[7] += as_half(sub_group_broadcast(input_value, 19)) * i4x16_odd.s9 +
                      as_half(sub_group_broadcast(input_value, 23)) * i4x16_odd.sb +
                      as_half(sub_group_broadcast(input_value, 27)) * i4x16_odd.sd +
                      as_half(sub_group_broadcast(input_value, 31)) * i4x16_odd.sf;
        }

        sum_all += (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]) * scale_1;
    }


    *(all_sum_even + thr_num * wi_id + thr_id) = sum_all;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(thr_id==0) {
        float sum_value = 0.0;
        for (int i = 0; i < thr_num; i++) {
            sum_value += *(all_sum_even + thr_num * wi_id + i);
        }
        output[n] = sum_value * routing_weights;
    }

}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_down)(
    const __global FUNC(expert_info)* info_ptrs,
    const __global TYPE* x,                               // [MAX_TOPK, INTERMEDIATE_SIZE]
    __global TYPE* routing_weights,                       // [MAX_TOPK]
    __global TYPE* y) {                                   // [MAX_TOPK, HIDDEN_SIZE]
    // gws: [expert, N, THR_NUM(16)]
    // lws: [1, SUBGROUP_SIZE, THR_NUM(16)]
    int expert_no = get_global_id(0);
    x += expert_no * INTERMEDIATE_SIZE;
    y += expert_no * HIDDEN_SIZE;
    const __global FUNC(expert_info)* info_ptr = info_ptrs + expert_no;
    // down, [INTERMEDIATE_SIZE, HIDDEN_SIZE]
    __global uchar* down_weight = (__global uchar*)info_ptr->weight[2];
    __global half* down_scale = (__global half*)info_ptr->scale[2];
    __global uchar* down_zp = (__global uchar*)info_ptr->zp[2];
    __local float all_sum_even[SUBGROUP_SIZE * THR_NUM];

    //if(get_global_id(0)==0 && get_global_id(1)==0 && get_global_id(2)==0) {
    //    printf("down:gws = (%d, %d,%d), ", get_global_size(0), get_global_size(1), get_global_size(2));
    //    printf("lws = (%d, %d,%d), ", get_local_size(0), get_local_size(1), get_local_size(2));
    //    printf("K = %d, N = %d, SUBGROUP_SIZE = %d, THR_NUM = %d\n",INTERMEDIATE_SIZE, HIDDEN_SIZE, SUBGROUP_SIZE,THR_NUM);
    //}

    tile_gemv_down(down_weight, down_scale, down_zp, x, y, HIDDEN_SIZE, INTERMEDIATE_SIZE, all_sum_even, routing_weights[info_ptr->routing_offset]);
}

#else

//__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
KERNEL (mlp_reduce)(const __global TYPE* x,                // [MAX_TOPK, HIDDEN_SIZE]
    __global TYPE* y) {                                    // [1, HIDDEN_SIZE]
    int n = get_global_id(1);
    float sum = 0;
    //if(get_global_id(0)==0 && get_global_id(1)==0 && get_global_id(2)==0) {
    //    printf("reduce:gws = (%d, %d,%d), ", get_global_size(0), get_global_size(1), get_global_size(2));
    //    printf("lws = (%d, %d,%d), ", get_local_size(0), get_local_size(1), get_local_size(2));
    //    printf("N = %d, SUBGROUP_SIZE = %d\n", HIDDEN_SIZE, SUBGROUP_SIZE);
    //}
    for (int i = 0; i < MAX_TOPK; i++) {
        sum += x[n];
        x += HIDDEN_SIZE;
    }
    y[n] = sum;
}
#endif

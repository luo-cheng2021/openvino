// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <string>
#include <vector>
#include <cpu/x64/brgemm/brgemm.hpp>
#include <cpu/x64/matmul/brgemm_matmul_copy_utils.hpp>
#include <cpu/x64/matmul/brgemm_matmul_utils.hpp>
#include <cpu/x64/amx_tile_configure.hpp>

namespace ov {
namespace intel_cpu {
namespace gpt {

// pattern is:
// query:[batch, num_heads, query_seq_len, head_size]  key:[batch, num_heads, key_seq_len, head_size]
//    \                                                 |
//     \                                           Transpose0: [batch, num_heads, head_size, key_seq_len]
//      \                                              /
//       \                                            /
//        \                                          /
//        MatMul0: [batch, num_heads, query_seq_len, key_seq_len]
//          |
//          |   norm_factor(const): [1]
//          |       /
//       Multiply: [batch, num_heads, query_seq_len, key_seq_len]
//          |
//          |   causal_mask: [1, 1, query_seq_len, key_seq_len]
//          |       /
//       Select(only for 1x300): [batch, num_heads, query_seq_len, key_seq_len]
//          |
//          |   attention_mask:[batch, 1, 1, key_seq_len]
//          |       /
//       Add: [batch, num_heads, query_seq_len, key_seq_len]
//          |
//       SoftMax: [batch, num_heads, query_seq_len, key_seq_len]
//          |
//           \  value:[batch, num_heads, key_seq_len, head_size]
//            \     /
//             MatMul1: [batch, num_heads, query_seq_len, head_size]
//               |
//            Transpose1(only for 1x300): [batch, query_seq_len, num_heads * head_size]
// usage:
// MHAGPT mha;
// MHAGPT::CreateParam create_param = {...};
// auto scratch_size = mha.query_scratch_size();
// auto scratch_buffer = malloc(scratch_size);
// mha.create(create_param, scratch_buffer);
// MHAGPT::ExecParam exec_param = {...};
// mha.exec(exec_param);
class MHAGPT {
public:
    struct CreateParam {
        size_t num_heads, head_size, head_size_aligned;
        float normal_factor;
        InferenceEngine::Precision qkv_precision;
        InferenceEngine::Precision dst_precision;
        size_t max_seq_len;
        bool is_qkv_quant_per_tensor;
    };
    struct ExecParam {
        size_t batch, query_seq_len, key_seq_len;
        size_t first_valid_softmax_items; // only for 1x300: valid items in 1st row, next row will be increased by 1
        uint8_t* q;
        std::vector<uint8_t*>& k;
        std::vector<uint8_t*>& v;
        float* attention_mask;
        uint8_t* attn_output;
        size_t head_stride_in_q;            // q stride for next head
        size_t batch_stride_in_q;           // q stride for next batch
        size_t head_stride_in_kv;           // kv stride for next head
        size_t batch_stride_in_attn_mask;   // attn_mask stride for next batch
        size_t head_stride_in_attn;         // attn stride for next head
        size_t batch_stride_in_attn;        // attn stride for next batch
        float q_dequant;
        float k_dequant;
        float qk_quant;
        float v_dequant;
        std::vector<float>& qkv_quant;
    };
    MHAGPT();
    static int query_scratch_size(const CreateParam& param);
    void create(const CreateParam& param);

    //void close();
    void exec(const ExecParam& param);
    impl_desc_type get_impl_type();

private:
    struct Impl;
    std::shared_ptr<Impl> impl;
};

}   // namespace gpt
}   // namespace intel_cpu
}   // namespace ov

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v10 {
/// \brief      Parameterized, part of GPT Neox attention operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GPTAttn : public Op {
public:
    OPENVINO_OP("GPTAttn", "opset10");

    GPTAttn() = default;

    /// \brief      Constructs operation.
    ///
    /// \param      data   Input tensor.
    ///
    GPTAttn(const Output<Node>& qkv,
        const Output<Node>& past_keys_num,
        const Output<Node>& beam_idx,
        const Output<Node>& attn_mask,
        const Output<Node>& position_ids,
        int layer_num = 32,
        int head_num = 32,
        int size_per_head = 80,
        int rotary_emb_base = 10000,
        float rotary_pct = 0.25,
        int cur_layer_num = 0,
        int max_seq_len = 400,
        bool use_position2d = false,
        float q_quant = 0.0f,
        float k_quant = 0.0f,
        float qk_quant = 0.0f,
        float v_quant = 0.0f
        );

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    size_t m_layer_num = 32;
    size_t m_head_num = 32;
    size_t m_size_per_head = 80;
    size_t m_rotary_emb_base = 10000;
    float m_rotary_pct = 0.25;
    size_t m_cur_layer_num = 0;
    size_t m_max_seq_len = 400;
    bool m_use_position2d = false;
    float m_q_quant = 0.0f;
    float m_k_quant = 0.0f;
    float m_qk_quant = 0.0f;
    float m_v_quant = 0.0f;
};
}  // namespace v10
}  // namespace op
}  // namespace ov

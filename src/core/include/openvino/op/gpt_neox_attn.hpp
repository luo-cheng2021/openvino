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
class OPENVINO_API GPTNeoxAttn : public Op {
public:
    OPENVINO_OP("GPTNeoxAttn", "opset10");

    GPTNeoxAttn() = default;

    /// \brief      Constructs operation.
    ///
    /// \param      data   Input tensor.
    ///
    GPTNeoxAttn(const Output<Node>& qkv, const Output<Node>& past_keys_num,
        const Output<Node>& beam_idx,
        int layer_num = 32,
        int head_num = 32,
        int size_per_head = 80,
        int hidden_size = 32 * 80,
        int max_position_embeddings = 2048,
        int rotary_emb_base = 10000,
        float rotary_pct = 0.25,
        int max_seq_len = 400);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int m_layer_num = 32;
    int m_head_num = 32;
    int m_size_per_head = 80;
    int m_hidden_size = 32 * 80;
    int m_max_position_embeddings = 2048;
    int m_rotary_emb_base = 10000;
    float m_rotary_pct = 0.25;
    int m_max_seq_len = 400;
};
}  // namespace v10
}  // namespace op
}  // namespace ov

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class ConditionalBlock : public Op {
public:
    OPENVINO_OP("ConditionalBlock", "internal");
    BWDCMP_RTTI_DECLARATION;

    ConditionalBlock() = default;

    ConditionalBlock(const OutputVector& inputs,
                     const Output<Node>& cond,
                     bool is_scalar_condition,
                     int32_t sub_block_index,
                     int32_t m_num_outputs);
    ConditionalBlock(const Output<Node>& cond,
                     bool is_scalar_condition,
                     int32_t sub_block_index,
                     int32_t m_num_outputs);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    /// \return A vector containing the values for each input except "cond".
    const OutputVector get_inputs_from_parent() const;

    const int32_t get_subblock_index() const {
        return m_sub_block_index;
    }

private:
    bool m_is_scalar_condition;
    int32_t m_sub_block_index;

    int32_t m_num_outputs;
};

}  // namespace internal
}  // namespace op
}  // namespace ov

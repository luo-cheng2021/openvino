// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class While : public Op {
public:
    OPENVINO_OP("While", "internal");
    BWDCMP_RTTI_DECLARATION;

    While() = default;

    While(const OutputVector& inputs, int32_t sub_block, const std::vector<std::string>& output_names);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int32_t m_sub_block = 0;

    std::vector<std::string> m_output_names;

private:
};

}  // namespace internal
}  // namespace op
}  // namespace ov

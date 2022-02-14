// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class SelectInput : public Op {
public:
    OPENVINO_OP("SelectInput", "internal");
    BWDCMP_RTTI_DECLARATION;

    SelectInput() = default;

    SelectInput(const Output<Node>& args0,
                const Output<Node>& args1,
                const Output<Node>& mask,
                const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    std::vector<std::pair<ov::element::Type, ov::PartialShape>> m_output_infos;
};

}  // namespace internal
}  // namespace op
}  // namespace ov

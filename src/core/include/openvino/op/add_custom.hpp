// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v10 {
/// \brief      Parameterized, part of Add operation.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API AddCustom : public Op {
public:
    OPENVINO_OP("AddCustom", "opset10");

    AddCustom() = default;

    /// \brief      Constructs operation.
    ///
    /// \param      data   Input tensor.
    ///
    AddCustom(const Output<Node>& node1, const Output<Node>& node2,
        const Output<Node>& node3);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};
}  // namespace v10
}  // namespace op
}  // namespace ov

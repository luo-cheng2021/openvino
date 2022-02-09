// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class TensorArrayAppend : public Op {
public:
    OPENVINO_OP("TensorArrayAppend", "internal");
    BWDCMP_RTTI_DECLARATION;

    TensorArrayAppend() = default;

    TensorArrayAppend(const Output<Node>& input, const Output<Node>& index);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    bool has_evaluate() const { return true; }

private:
};

}  // namespace internal
}  // namespace op
}  // namespace ov

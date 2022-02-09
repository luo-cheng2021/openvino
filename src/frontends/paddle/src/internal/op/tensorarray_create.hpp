// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class TensorArrayCreate : public Op {
public:
    OPENVINO_OP("TensorArrayCreate", "internal");
    BWDCMP_RTTI_DECLARATION;

    TensorArrayCreate() = default;

    TensorArrayCreate(const Output<Node>& args0);

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    void validate_and_infer_types() override;
    bool has_evaluate() const { return true; }
    bool constant_fold(OutputVector& output_values, const OutputVector& inputs_values) override;

private:
};

}  // namespace internal
}  // namespace op
}  // namespace ov

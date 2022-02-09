// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace internal {
class TensorArrayToTensor : public Op {
public:
    struct ConcatParam {
        int axis;       // the axis will be concat
    };
    struct SliceParam {
        int index;      // will get the index
    };
    OPENVINO_OP("TensorArrayToTensor", "internal");
    BWDCMP_RTTI_DECLARATION;

    TensorArrayToTensor() = default;

    TensorArrayToTensor(const Output<Node>& arg0, const ConcatParam& param);
    TensorArrayToTensor(const Output<Node>& arg0, const SliceParam& param);
    TensorArrayToTensor(const Output<Node>& arg0, const bool is_concat, const ConcatParam& concat_param, const SliceParam& slice_param);

    void validate_and_infer_types() override;

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override;
    bool has_evaluate() const { return true; }

    ConcatParam m_concat_param;
    SliceParam m_slice_param;
private:
    bool m_is_concat = true;
};

}  // namespace internal
}  // namespace op
}  // namespace ov

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/tensorarray_to_tensor.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "ngraph/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

BWDCMP_RTTI_DEFINITION(op::internal::TensorArrayToTensor);

op::internal::TensorArrayToTensor::TensorArrayToTensor(const Output<Node>& arg0)
    : Op({arg0}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::TensorArrayToTensor::clone_with_new_inputs(const OutputVector& new_args) const {
    return make_shared<TensorArrayToTensor>(new_args[0]);
}

bool op::internal::TensorArrayToTensor::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

void op::internal::TensorArrayToTensor::validate_and_infer_types() {
    set_output_type(0, element::dynamic, {Dimension::dynamic()});
}

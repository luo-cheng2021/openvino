// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/unary_dyn.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "ngraph/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

BWDCMP_RTTI_DEFINITION(op::internal::UnaryDyn);

op::internal::UnaryDyn::UnaryDyn(const Output<Node>& args0) : Op({args0}){
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::UnaryDyn::clone_with_new_inputs(const OutputVector& new_args) const {
    return make_shared<UnaryDyn>(new_args[0]);
}

void op::internal::UnaryDyn::validate_and_infer_types() {
    auto ps = get_input_partial_shape(0);

    // TODO: which axis should be concat?
    ps[1] = ov::Dimension(); // HARDCODE axis?
    set_output_type(0, get_input_element_type(0), ps);
}

bool op::internal::UnaryDyn::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    outputs[0]->set_unary(inputs[0]);
    outputs[0]->write(inputs[0]->get_data_ptr(), outputs[0]->get_size_in_bytes());
    return true;
}

bool op::internal::UnaryDyn::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    return false;
}
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/tensorarray_write.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "ngraph/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

BWDCMP_RTTI_DEFINITION(op::internal::TensorArrayWrite);

op::internal::TensorArrayWrite::TensorArrayWrite(const Output<Node>& input, const Output<Node>& index, const std::string& output_names)
    : Op({input, index}),
    m_output_name(output_names) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::TensorArrayWrite::clone_with_new_inputs(const OutputVector& new_args) const {
    return make_shared<TensorArrayWrite>(new_args[0], new_args[1], m_output_name);
}

bool op::internal::TensorArrayWrite::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

void op::internal::TensorArrayWrite::validate_and_infer_types() {
    auto ps0 = get_input_partial_shape(0);
    ov::PartialShape ps;
    // dimension 1 is the list
    ps.resize(ps0.size() + 1);
    set_output_type(0, get_input_element_type(0), ps);
}

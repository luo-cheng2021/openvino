// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add_custom.hpp"

#include "itt.hpp"

namespace ov {
op::v10::AddCustom::AddCustom(const Output<Node>& node1, const Output<Node>& node2,
        const Output<Node>& node3) :
        op::Op({node1, node2, node3}) {
    constructor_validate_and_infer_types();
}

bool op::v10::AddCustom::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_AddCustom_visit_attributes);
    return true;
}

void op::v10::AddCustom::validate_and_infer_types() {
    OV_OP_SCOPE(v10_AddCustom_validate_and_infer_types);
    // NODE_VALIDATION_CHECK(this,
    //                       get_input_element_type(0).is_dynamic() || get_input_element_type(0).is_real(),
    //                       "The element type of the input tensor must be a floating point number.");
    // set_output_type(0, element::boolean, get_input_partial_shape(0));
    auto ps = get_input_partial_shape(0);
    set_output_type(0, get_input_element_type(0), ps);
}

std::shared_ptr<Node> op::v10::AddCustom::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_AddCustom_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v10::AddCustom>(new_args.at(0), new_args.at(1), new_args.at(2));
}
}  // namespace ov

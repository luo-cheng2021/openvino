// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/mvn_custom.hpp"

#include "itt.hpp"

namespace ov {
template <>
NGRAPH_API EnumNames<ov::op::MVNEpsMode>& EnumNames<ov::op::MVNEpsMode>::get() {
    static auto enum_names = EnumNames<ov::op::MVNEpsMode>(
        "op::MVNEpsMode",
        {{"OUTSIDE_SQRT", ov::op::MVNEpsMode::OUTSIDE_SQRT}, {"INSIDE_SQRT", ov::op::MVNEpsMode::INSIDE_SQRT}});
    return enum_names;
}

op::v10::MVNCustom::MVNCustom(const Output<Node>& data,
                 const Output<Node>& reduction_axes,
                 const Output<Node>& weight,
                 const Output<Node>& bias,
                 bool normalize_variance,
                 float eps,
                 MVNEpsMode eps_mode)
    : Op({data, reduction_axes, weight, bias}),
      m_normalize_variance{normalize_variance},
      m_eps{eps},
      m_eps_mode{eps_mode} {
    constructor_validate_and_infer_types();
}

void op::v10::MVNCustom::validate_and_infer_types() {
    OV_OP_SCOPE(v10_MVNCustom_validate_and_infer_types);
    const auto data = get_input_partial_shape(0);
    const auto axes = get_input_partial_shape(1);

    if (axes.is_static()) {
        NODE_VALIDATION_CHECK(this, is_vector(axes.to_shape()), "Expected 1D tensor for the 'axes' input. Got: ", axes);

        NODE_VALIDATION_CHECK(
            this,
            data.rank().is_dynamic() || data.rank().get_length() >= static_cast<int64_t>(axes.get_shape()[0]),
            "Expected rank for the 'data' input to be higher than axes shape. Got: ",
            data);
    }

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> op::v10::MVNCustom::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_MVN_clone_with_new_inputs);
    NODE_VALIDATION_CHECK(this,
                          new_args.size() == 4,
                          "Expected 4 element in new_args for the MVNCustom op but got ",
                          new_args.size());
    return std::make_shared<op::v10::MVNCustom>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_normalize_variance, m_eps, m_eps_mode);
}

bool op::v10::MVNCustom::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_MVN_visit_attributes);
    visitor.on_attribute("eps", m_eps);
    visitor.on_attribute("normalize_variance", m_normalize_variance);
    visitor.on_attribute("eps_mode", m_eps_mode);
    return true;
}
}  // namespace ov

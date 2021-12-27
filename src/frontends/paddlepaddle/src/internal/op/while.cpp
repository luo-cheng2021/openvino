// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/while.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "ngraph/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

BWDCMP_RTTI_DEFINITION(op::internal::While);

op::internal::While::While(const OutputVector& inputs, int32_t sub_block, const std::vector<std::string>& output_names)
    : Op(inputs),
      m_sub_block(sub_block),
      m_output_names(output_names) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::While::clone_with_new_inputs(const OutputVector& new_args) const {
    return make_shared<While>(new_args, m_sub_block, m_output_names);
}

bool op::internal::While::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("sub_block", m_sub_block);
    return true;
}

void op::internal::While::validate_and_infer_types() {
    std::map<std::string, size_t> inputname_idx;
    for (auto i = 0; i < get_input_size(); i++) {
        auto node = get_input_node_ptr(i);
        inputname_idx.insert({node->get_friendly_name(), i});
    }

    for (auto i = 0; i < m_output_names.size(); i++) {
        const auto& name = m_output_names[i];
        const auto input_idx = inputname_idx[name];
        set_output_type(i, get_input_element_type(input_idx), get_input_partial_shape(input_idx));
    }
}

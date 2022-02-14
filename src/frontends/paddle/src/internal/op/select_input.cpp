// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/select_input.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "ngraph/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

BWDCMP_RTTI_DEFINITION(op::internal::SelectInput);

op::internal::SelectInput::SelectInput(const Output<Node>& args0,
                                       const Output<Node>& args1,
                                       const Output<Node>& mask,
                                       const std::vector<std::pair<ov::element::Type, ov::PartialShape>>& output_infos)
    : Op({args0, args1, mask}),
      m_output_infos(output_infos) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::SelectInput::clone_with_new_inputs(const OutputVector& new_args) const {
    return make_shared<SelectInput>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_infos);
}

bool op::internal::SelectInput::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

void op::internal::SelectInput::validate_and_infer_types() {
    for (auto i = 0; i < m_output_infos.size(); i++) {
        set_output_type(i, m_output_infos[i].first, m_output_infos[i].second);
    }
}
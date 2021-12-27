// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

//#define DEFAULT_OPSET

#ifdef DEFAULT_OPSET
#    include "default_opset.hpp"
#    include "ngraph/op/util/variable.hpp"
#else
#    include "openvino/opsets/opset3.hpp"
#endif

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs assign(const NodeContext& node) {
    auto x = node.get_ng_input("X");

#ifdef DEFAULT_OPSET
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{PartialShape::dynamic(), element::dynamic, "ID"});
    auto read_value = std::make_shared<default_opset::ReadValue>(x, variable);
    auto assign = std::make_shared<default_opset::Assign>(read_value, variable);
#else
    // auto read_value = std::make_shared<ov::opset3::ReadValue>(x, "variable_id");
    // auto assign = std::make_shared<ov::opset3::Assign>(read_value, "variable_id");
    auto assign = std::make_shared<ov::opset3::Convert>(x, x.get_element_type());
#endif

    return node.default_single_output_mapping({assign}, {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/select_input.hpp"

#include "openvino/frontend/paddle/node_context.hpp"

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs select_input_(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    const auto mask = node.get_input("Mask");

    const element::Type output_type = node.get_out_port_type("Out");
    auto placehodler = std::make_shared<default_opset::Select>(std::make_shared<default_opset::Convert>(mask, element::boolean), x[1], x[0]);

    return node.default_single_output_mapping({placehodler}, {"Out"});
}

NamedOutputs select_input(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    const auto mask = node.get_input("Mask");

    auto outputs_info = node.get_output_port_infos("Out");
    auto placehodler = std::make_shared<ov::op::internal::SelectInput>(x[0], x[1], mask, outputs_info);

    return node.default_single_output_mapping({placehodler}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
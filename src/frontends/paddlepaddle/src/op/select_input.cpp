// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/select_input.hpp"

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs select_input(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    const auto mask = node.get_ng_input("Mask");

    const element::Type output_type = node.get_out_port_type("Out");
    auto placehodler = std::make_shared<ov::op::internal::SelectInput>(x[0], x[1], mask, output_type);

    return node.default_single_output_mapping({placehodler}, {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
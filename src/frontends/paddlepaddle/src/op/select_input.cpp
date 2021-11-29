// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs select_input(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    const auto mask = node.get_ng_input("Mask");

    const auto dummy_node = default_opset::Constant::create(element::i64, {1,2}, {0});
    return node.default_single_output_mapping(
        {dummy_node},
        {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs conditional_block(const NodeContext& node) {
    const auto cond = node.get_ng_input("Cond");
    auto sub_block = node.get_attribute<ov::BlockIndex>("sub_block");
    const auto is_scalar_condition = node.get_attribute<bool>("is_scalar_condition", true);

    std::cout << "conditional_block sub_block " << sub_block.get() << std::endl;

    (void)sub_block;
    (void)is_scalar_condition;

    const auto dummy_node = default_opset::Constant::create(element::i64, {3,4}, {0});
    return node.default_single_output_mapping(
        {dummy_node},
        {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
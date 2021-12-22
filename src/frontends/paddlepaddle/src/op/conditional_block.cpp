// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"
#include "internal/op/conditional_block.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs conditional_block(const NodeContext& node) {   
    const auto cond = node.get_ng_input("Cond");
    auto sub_block = node.get_attribute<ov::BlockIndex>("sub_block");
    const auto is_scalar_condition = node.get_attribute<bool>("is_scalar_condition", true);

    std::cout << "conditional_block sub_block " << sub_block.get() << std::endl;

    int32_t num_outputs = node.get_output_size("Out");

    std::shared_ptr<Node> placehodler;
    if (node.has_ng_input("Input")) {
        const auto inputs = node.get_ng_inputs("Input");
        placehodler = std::make_shared<ov::op::internal::ConditionalBlock>(inputs, cond, is_scalar_condition, sub_block.get(), num_outputs);
    }
    else {
        placehodler = std::make_shared<ov::op::internal::ConditionalBlock>(cond, is_scalar_condition, sub_block.get(), num_outputs);
    }     
    const auto split_outputs = placehodler->outputs();

    auto out_names = node.get_output_names();
    auto it = std::find(out_names.begin(), out_names.end(), "Out");
    PDPD_ASSERT(it != out_names.end(), "Expected output not found");

    NamedOutputs named_outputs;
    for (const auto& split_output : split_outputs) {
        named_outputs[*it].push_back(split_output);
    }
    return named_outputs;
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
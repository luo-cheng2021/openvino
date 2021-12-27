// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

#include "internal/op/while.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {

using namespace default_opset;

NamedOutputs while_(const NodeContext& node) {
    const auto data = node.get_ng_inputs("X");
    const auto cond = node.get_ng_input("Condition");
    const auto sub_block = node.get_attribute<ov::BlockIndex>("sub_block").get();
    const auto output_names = node.get_output_var_names("Out");

    ov::OutputVector inputs = data;
    inputs.push_back(cond);
    NamedOutputs named_outputs;
    named_outputs["Out"] = std::make_shared<ov::op::internal::While>(inputs, sub_block, output_names)->outputs();
    return named_outputs;
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov

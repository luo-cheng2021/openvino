// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"

#include "default_opset.hpp"
#include "internal/op/tensorarray_to_tensor.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs tensor_array_to_tensor(const NodeContext& node) {
    const auto x = node.get_input("X");
    auto axis = node.get_attribute<int32_t>("axis", 0);
    PADDLE_OP_CHECK(node, axis == 0, "axis should be 0, got: ", axis);

    auto placeholder = std::make_shared<default_opset::Squeeze>(x, default_opset::Constant::create(element::i32, {1}, {0}));
    // get rid of pre-added content
    auto out = std::make_shared<default_opset::VariadicSplit>(placeholder, default_opset::Constant::create(element::i32, {1}, {0}),
        default_opset::Constant::create(element::i32, {2}, {1, -1}));

    NamedOutputs named_outputs;
    // TODO: handle empty tensor array case
    named_outputs["Out"] = {out->output(1)};
    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
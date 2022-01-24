// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"
#include "paddlepaddle_frontend/utility.hpp"
#include "internal/op/tensorarray_to_tensor.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs tensor_array_to_tensor(const NodeContext& node) {
    const auto x = node.get_ng_input("X");

    //auto placeholder = std::make_shared<default_opset::Convert>(x, x.get_tensor().get_element_type());
    auto placeholder = std::make_shared<default_opset::Squeeze>(x, default_opset::Constant::create(element::i32, {1}, {0}));

    return node.default_single_output_mapping({placeholder}, {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
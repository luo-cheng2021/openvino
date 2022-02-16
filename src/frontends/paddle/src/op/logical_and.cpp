// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs logical_and(const NodeContext& node) {
    auto x = node.get_input("X");
    auto y = node.get_input("Y");
    return node.default_single_output_mapping({std::make_shared<default_opset::LogicalAnd>(x, y)}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov

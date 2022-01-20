// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"
#include "paddlepaddle_frontend/utility.hpp"
#include "internal/op/tensorarray_write.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs write_to_array(const NodeContext& node) {
    const auto x = node.get_ng_input("X");
    const auto index = node.get_ng_input("I");
    const auto output_names = node.get_output_var_names("Out");

    auto placehodler = std::make_shared<ov::op::internal::TensorArrayWrite>(x, index, output_names[0]);

    return node.default_single_output_mapping({placehodler}, {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
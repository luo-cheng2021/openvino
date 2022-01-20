// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"
#include "paddlepaddle_frontend/utility.hpp"
#include "internal/op/tensorarray_length.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs lod_array_length(const NodeContext& node) {
    const auto x = node.get_ng_input("X");

    auto placehodler = std::make_shared<ov::op::internal::TensorArrayLength>(x);

    return node.default_single_output_mapping({placehodler}, {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
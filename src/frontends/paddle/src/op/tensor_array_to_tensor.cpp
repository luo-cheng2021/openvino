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

    ov::op::internal::TensorArrayToTensor::ConcatParam param{axis};
    auto placeholder = std::make_shared<ov::op::internal::TensorArrayToTensor>(x, param);

    return node.default_single_output_mapping({placeholder}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
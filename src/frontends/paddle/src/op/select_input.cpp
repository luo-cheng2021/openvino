// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/select_input.hpp"

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs select_input(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    const auto mask = node.get_input("Mask");

    const element::Type output_type = node.get_out_port_type("Out");
    const auto cond = std::make_shared<default_opset::Convert>(mask, element::boolean);
    const auto ps0 = x[0].get_partial_shape();
    const auto ps1 = x[1].get_partial_shape();

    if (ps0.compatible(ps1)) {
        auto placehodler = std::make_shared<default_opset::Select>(cond, x[1], x[0]);
        return node.default_single_output_mapping({placehodler}, {"Out"});
    } else {
        if (ps0.rank() != ps1.rank()) {
            const auto fix_idx = [&](int idx) {
                const auto ps = x[idx].get_partial_shape();
                if (ps.is_static()) {
                    const Shape shape(ps.get_shape());
                    const auto size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
                    if (size == 0)
                        return 1 - idx;
                }
                return idx;
            };
            auto placehodler = std::make_shared<default_opset::Select>(cond, x[fix_idx(1)], x[fix_idx(0)]);
            return node.default_single_output_mapping({placehodler}, {"Out"});
        }
        PADDLE_OP_CHECK(node, false, "input shapes should be compatible.");

        return {};
    }
    //     // fallback to if ops
    //     const auto if_node = std::make_shared<default_opset::If>(cond);
    //     const auto then_param = std::make_shared<default_opset::Parameter>(x[1].get_element_type(), ps1);
    //     const auto then_result = std::make_shared<default_opset::Result>(then_param);
    //     const auto then_branch = std::make_shared<Model>(ResultVector{then_result}, ParameterVector{then_param});
    //     const auto else_param = std::make_shared<default_opset::Parameter>(x[0].get_element_type(), ps0);
    //     const auto else_result = std::make_shared<default_opset::Result>(else_param);
    //     const auto else_branch = std::make_shared<Model>(ResultVector{else_result}, ParameterVector{else_param});
    //     if_node->set_then_body(then_branch);
    //     if_node->set_else_body(else_branch);
    //     if_node->set_input(x[1], then_param, nullptr);
    //     if_node->set_input(x[0], nullptr, else_param);
    //     if_node->set_output(then_result, else_result);
    //     return node.default_single_output_mapping({if_node}, {"Out"});
    // }
}

NamedOutputs select_input_(const NodeContext& node) {
    const auto x = node.get_ng_inputs("X");
    const auto mask = node.get_input("Mask");

    auto outputs_info = node.get_output_port_infos("Out");
    auto placehodler = std::make_shared<ov::op::internal::SelectInput>(x[0], x[1], mask, outputs_info);

    return node.default_single_output_mapping({placehodler}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
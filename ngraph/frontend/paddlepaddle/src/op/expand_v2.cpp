// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "expand_v2.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs expand_v2 (const NodeContext& node) {
    auto x = node.get_ng_input("X");
    Output<Node> shapeExpectedNode;
    if (node.has_ng_input("Shape")) {
        shapeExpectedNode = node.get_ng_input("Shape");
    } else if (node.has_ng_input("expand_shapes_tensor")) {
        auto inputs = node.get_ng_inputs("expand_shapes_tensor");
        ngraph::NodeVector node_vec;
        for (auto &&node : inputs) {
            auto cast = std::make_shared<ngraph::opset6::Convert>(node, element::i32);
            node_vec.push_back(cast);
        }
        shapeExpectedNode = std::make_shared<ngraph::opset6::Concat>(node_vec, 0);
    } else {
        std::vector<int32_t> shapeExpected;
        if (node.has_attribute<std::vector<int32_t>>("shape")) {
            shapeExpected = node.get_attribute<std::vector<int32_t>>("shape");
        } else {
            throw std::runtime_error("expand: has no shape attribute");
        }
        shapeExpectedNode = ngraph::opset6::Constant::create(ngraph::element::i32, { shapeExpected.size() }, shapeExpected);
    }
    // if -1 in shape we will copy the orginal value from input
    auto zeroNode = ngraph::opset6::Constant::create(ngraph::element::i32, {1}, {0});
    auto maskNode = std::make_shared<ngraph::opset6::Greater>(shapeExpectedNode, zeroNode);
    auto inputShapeNode = std::make_shared<ngraph::opset6::ShapeOf>(x, element::i32);
    auto fixedShapeNode = std::make_shared<ngraph::opset6::Select>(maskNode, shapeExpectedNode, inputShapeNode);

    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Broadcast>(x, fixedShapeNode)}, {"Out"});
}

}}}}
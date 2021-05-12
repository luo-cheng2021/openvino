// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "pow.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs pow (const NodeContext& node) {
    auto x = node.get_ng_input("X");
    Output<Node> factorNode;
    if (node.has_ng_input("FactorTensor")) {
        factorNode = node.get_ng_input("FactorTensor");
    } else {
        auto factor = 1.0f;
        if (node.has_attribute<float>("factor")) {
            factor = node.get_attribute<float>("factor");
        }
        factorNode = ngraph::opset6::Constant::create(ngraph::element::f32, {}, {factor});
    }
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Power>(x, factorNode)}, {"Out"});
}

}}}}
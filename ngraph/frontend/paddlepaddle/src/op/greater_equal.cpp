// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "greater_equal.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs greater_equal (const NodeContext& node) {
    auto x = node.get_ng_input("X");
    auto y = node.get_ng_input("Y");
    // TODO: support the data type of 'Out' equal to the type of input
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::GreaterEqual>(x, y)}, {"Out"});
}

}}}}
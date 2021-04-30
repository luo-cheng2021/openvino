//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <ngraph/opsets/opset6.hpp>
#include "argmax.hpp"

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs argmax (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    bool flatten = node.get_attribute<bool>("flatten");
    const element::Type& index_element_type = element::i64;
    const Output<ngraph::Node> k = ngraph::opset6::Constant::create(ngraph::element::i64, {}, {1});

    if(!flatten) {
        auto axis = node.get_attribute<int64_t>("axis");
        const auto axis_to_remove = ngraph::opset6::Constant::create(element::u64, Shape{}, {axis});
        auto node_topk = std::make_shared<ngraph::opset6::TopK>(data, k, axis, "max", "index", index_element_type);
        const auto reshaped_indices = std::make_shared<ngraph::opset6::Squeeze>(node_topk->output(1), axis_to_remove);
        return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Convert>(reshaped_indices, element::i64)}, {"Out"});
    } else {
        int64_t axis = 0;
        const Output<ngraph::Node> reshape_flatten = ngraph::opset6::Constant::create(ngraph::element::i64, {1}, {-1});
        auto node_reshape = std::make_shared<ngraph::opset6::Reshape>(data, reshape_flatten, true);
        auto node_topk = std::make_shared<ngraph::opset6::TopK>(node_reshape, k, axis, "max", "index", index_element_type);
        return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Convert>(node_topk->output(1), element::i64)}, {"Out"});
    }
}

}}}}

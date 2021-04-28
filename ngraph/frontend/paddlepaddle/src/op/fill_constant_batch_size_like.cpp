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
#include "fill_constant_batch_size_like.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

static std::shared_ptr<Node> get_val(int32_t idx, const Output<Node>& data) {
    auto startsNode = ngraph::opset6::Constant::create(element::i32, { 1 }, { idx });
    auto endsNode = ngraph::opset6::Constant::create(element::i32, { 1 }, { idx + 1 });
    auto stridesNode = ngraph::opset6::Constant::create(element::i32, { 1 }, { 1 });
    return std::make_shared<ngraph::opset6::StridedSlice>(data,
        startsNode, 
        endsNode, 
        stridesNode,
        std::vector<int64_t>(1, 0),
        std::vector<int64_t>(1, 0));
}

static std::shared_ptr<Node> set_val(int32_t idx, std::shared_ptr<Node> val_node, std::shared_ptr<Node> array_node) {
    NodeVector nodes;
    if (idx > 0) {
        // [0, idx)
        auto startsNode = ngraph::opset6::Constant::create(element::i32, { 1 }, { 0 });
        auto endsNode = ngraph::opset6::Constant::create(element::i32, { 1 }, { idx });
        auto stridesNode = ngraph::opset6::Constant::create(element::i32, { 1 }, { 1 });
        auto head = std::make_shared<ngraph::opset6::StridedSlice>(array_node,
            startsNode, 
            endsNode, 
            stridesNode,
            std::vector<int64_t>(1, 0),
            std::vector<int64_t>(1, 0));
        nodes.push_back(head);
    }
    nodes.push_back(val_node);
    // [idx + 1, max)
    auto startsNode = ngraph::opset6::Constant::create(element::i32, { 1 }, { idx + 1 });
    auto endsNode = ngraph::opset6::Constant::create(element::i32, { 1 }, { INT_MAX });
    auto stridesNode = ngraph::opset6::Constant::create(element::i32, { 1 }, { 1 });
    auto tail = std::make_shared<ngraph::opset6::StridedSlice>(array_node,
        startsNode, 
        endsNode, 
        stridesNode,
        std::vector<int64_t>(1, 0),
        std::vector<int64_t>(1, 0));
    nodes.push_back(tail);
    
    return std::make_shared<ngraph::opset6::Concat>(nodes, 0);
}

NamedOutputs fill_constant_batch_size_like (const NodeContext& node) {
    auto input_dim_idx = node.get_attribute<int32_t>("input_dim_idx");
    auto output_dim_idx = node.get_attribute<int32_t>("output_dim_idx");
    auto value = node.get_attribute<float>("value");
    auto shapes = node.get_attribute<std::vector<int32_t> >("shape");
    auto input = node.get_ng_input("Input");
    auto input_shape = std::make_shared<ngraph::opset6::ShapeOf>(input, element::i32);
    // because Gather&Scatter does not support evaluate then
    // 1, cat the array:
    //   shape[0, shape[output_dim_idx]) + input_shape[input_dim_idx] + shape[shape[output_dim_idx + 1], -1]
    auto input_val_node = get_val(input_dim_idx, input_shape);
    auto shapes_node = ngraph::opset6::Constant::create(ngraph::element::i32, { shapes.size() }, shapes);
    auto shape_node = set_val(output_dim_idx, input_val_node, shapes_node);
    // 2, use the shape broadcast the node
    auto val_const = ngraph::opset6::Constant::create(ngraph::element::f32, { 1 }, { value });
    return node.default_single_output_mapping(
        {std::make_shared<ngraph::opset6::Broadcast>(val_const, shape_node)}, 
        {"Out"});
}

}}}}
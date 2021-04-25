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
#include "flatten_contiguous_range.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs flatten_contiguous_range (const NodeContext& node) {
    auto data = node.get_ng_input("X");

    PartialShape input_shape = data.get_partial_shape();
    int32_t input_rank = input_shape.rank().get_length();

    auto start_axis = node.get_attribute<int32_t>("start_axis") < 0 ? 0 : node.get_attribute<int32_t>("start_axis");
    auto stop_axis = node.get_attribute<int32_t>("stop_axis") > input_rank ? input_rank : node.get_attribute<int32_t>("stop_axis");
    stop_axis = (stop_axis == -1) ? (input_rank - 1) : stop_axis;

    int64_t flattened_rank = input_rank - (stop_axis - start_axis);
    auto flattened_shape = std::vector<int64_t>(flattened_rank, 1);

    int32_t i = 0, j = 0;

    for (i = 0; i < start_axis; i++, j++)
        flattened_shape[j] = input_shape[i].get_length();

    for (i = start_axis; i <= stop_axis; i++)
        flattened_shape[j] *= input_shape[i].get_length();

    j++;

    for (i = stop_axis + 1; i < input_rank; i++, j++)
        flattened_shape[j] = input_shape[i].get_length();

    auto shape_node = ngraph::opset6::Constant::create(ngraph::element::i64, {flattened_shape.size()}, flattened_shape);
//    return {std::make_shared<ngraph::opset6::Reshape>(data, shape_node, true)};
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Reshape>(data, shape_node, true)}, {"Out"});
}
}
}
}
}

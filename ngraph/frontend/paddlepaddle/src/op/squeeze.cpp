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
#include "squeeze.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

NamedOutputs squeeze (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    std::vector<int32_t> axes;
    if (node.has_attribute<std::vector<int32_t>>("axes")) {
        axes = node.get_attribute<std::vector<int32_t>>("axes");
    }
    
    auto axesNode = ngraph::opset6::Constant::create(ngraph::element::i32, {axes.size()}, axes);
    return node.default_single_output_mapping({std::make_shared<ngraph::opset6::Squeeze>(data, axesNode)}, {"Out"});
}

} // namespace op
} // namespace pdpd
} // namespace frontend
} // namespace ngraph
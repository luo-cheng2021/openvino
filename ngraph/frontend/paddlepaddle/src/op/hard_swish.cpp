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
#include "hard_swish.hpp"
#include <paddlepaddle_frontend/utility.hpp>

namespace ngraph {
namespace frontend {
namespace pdpd {
namespace op {

OutputVector hard_swish (const NodeContext& node) {
    auto data = node.get_ng_input("X");
    if (node.has_attribute<float>("threshold")) {
        auto threshold = node.get_attribute<float>("threshold");
        MY_ASSERT(std::abs(threshold - 6.0) < 0.001, "hard_swish: threshold must = 6.0.");
    }
    if (node.has_attribute<float>("scale")) {
        auto scale = node.get_attribute<float>("scale");
        MY_ASSERT(std::abs(scale - 6.0) < 0.001, "hard_swish: scale must = 6.0.");
    }
    if (node.has_attribute<float>("offset")) {    
        auto offset = node.get_attribute<float>("offset");
        MY_ASSERT(std::abs(offset - 3.0) < 0.001, "hard_swish: offset must = 3.0.");
    }
    return {std::make_shared<ngraph::opset6::HSwish>(data)};
}

}}}}
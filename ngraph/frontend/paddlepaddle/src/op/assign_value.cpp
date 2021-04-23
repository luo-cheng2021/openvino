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
#include "assign_value.hpp"
namespace ngraph {
    namespace frontend {
        namespace pdpd {
            namespace op {

                OutputVector assign_value (const NodeContext& node) {
                    std::vector<float> values = node.get_attribute<std::vector<float>>("fp32_values");
                    std::vector<int32_t> shape = node.get_attribute<std::vector<int32_t>>("shape");
                    return {opset6::Constant::create(element::f32, Shape{shape.begin(), shape.end()}, values)};
                }

            }
        }
    }
}

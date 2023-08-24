// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset4.hpp>

namespace ov {
namespace intel_cpu {

class VNodeFusion: public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("VNodeFusion", "0");
    VNodeFusion();
};

}   // namespace intel_cpu
}   // namespace ov

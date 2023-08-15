// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/graph_rewrite.hpp>

namespace ov {
namespace intel_cpu {

class RoPEFusion : public ngraph::pass::GraphRewrite {
public:
    OPENVINO_RTTI("RoPEFusion", "0");
    RoPEFusion();
};

}  // namespace intel_cpu
}  // namespace ov

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace pass {

class TransformIf : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::TransformIf");
    TransformIf(std::vector<std::shared_ptr<Function>> functions);

private:
    std::vector<std::shared_ptr<Function>> m_functions;
};

}  // namespace pass
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov

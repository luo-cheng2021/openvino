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

class TransformCond : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::TransformCond");
    TransformCond(std::vector<std::shared_ptr<Function>> functions);

private:
    std::vector<std::shared_ptr<Function>> m_functions;
};

class TransformIfIf : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::TransformIfIf");
    TransformIfIf(std::vector<std::shared_ptr<Function>> functions);

private:
    std::vector<std::shared_ptr<Function>> m_functions;
};

class TensorArrayWriteConcatenation : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::TensorArrayWriteConcatenation");
    TensorArrayWriteConcatenation(std::shared_ptr<Function> func);
};

class ConditionalBlockTensorArrayOutputSlice : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::ConditionalBlockTensorArrayOutputSlice");
    ConditionalBlockTensorArrayOutputSlice(std::vector<std::shared_ptr<Function>> functions);

private:
    std::vector<std::shared_ptr<Function>> m_functions;
};

}  // namespace pass
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov

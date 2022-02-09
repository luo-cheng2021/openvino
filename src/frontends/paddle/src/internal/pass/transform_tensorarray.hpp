// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace pass {

class TransformTensorArray : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::TransformTensorArray");
    TransformTensorArray(std::vector<std::shared_ptr<Model>> functions);

private:
    std::vector<std::shared_ptr<Model>> m_functions;
};

class TransformEliminateConvert: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::TransformEliminateConvert");
    TransformEliminateConvert();
};

class TransformMarkupTensorArray : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ov::frontend::pass::TransformMarkupTensorArray");

    TransformMarkupTensorArray() = default;

    bool run_on_function(std::shared_ptr<ov::Model> f) override;

private:
};
}  // namespace pass
}  // namespace paddle
}  // namespace frontend
}  // namespace ov

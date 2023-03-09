// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <string>
#include <vector>
#include "mha_gpt.h"

namespace ov {
namespace intel_cpu {
namespace node {

class AddCustom : public Node {
public:
    AddCustom(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override { return getType() == Type::AddCustom; }

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    void prepareParams() override;
    bool needShapeInfer() const override { return false; }

private:
    InferenceEngine::Precision dataPrecision;
    size_t totalElements;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <tuple>

namespace ov {
namespace intel_cpu {
namespace node {

class MVNCustom : public Node {
public:
    MVNCustom(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeInPlace() const override {
        return false;
    }

    bool canFuse(const NodePtr& node) const override;
    void prepareParams() override;

    // Defines way to add epsilon: inside sqrt or outside.
    enum MVNEpsMode {
        INSIDE_SQRT,
        OUTSIDE_SQRT
    };

private:
    bool initAcrossChannels_;
    bool execAcrossChannels_;
    bool normalizeVariance_;
    float epsValue_;
    MVNEpsMode epsMode_;

    enum Fastpath_Postops {
        Fastpath_Postops_NotSupport,
        Fastpath_Postops_No,
        Fastpath_Postops_FQ
    } supportedPostops = Fastpath_Postops_NotSupport;
    std::vector<float> qkv_quant;     // next node quant scale
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

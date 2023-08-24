// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include <string>
#include <memory>
#include <vector>
#include <dnnl_extension_utils.h>

#include "transformations/cpu_opset/common/op/vnode.hpp"

#include "vnode_executor.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class VNode : public Node {
public:
    VNode(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool needShapeInfer() const override {return false;};
    bool needPrepareParams() const override {return false;};
    bool isExecutable() const override { return true; }
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    std::string m_vtype;
    std::string errorPrefix;
    std::shared_ptr<ov::intel_cpu::VNode> m_vnode;

    std::shared_ptr<vnode_executor> m_executor;
    ov::element::Type inType;
    std::map<std::string, double> m_symbol_name2value;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
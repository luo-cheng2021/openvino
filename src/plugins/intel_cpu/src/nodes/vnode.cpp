// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vnode.h"

#include <ngraph/opsets/opset1.hpp>
#include <string>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <vector>

#include "ie_parallel.hpp"
#include "transformations/cpu_opset/common/op/vnode.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

struct VnodeExecutorFactory {
    std::map<std::string, std::function<std::shared_ptr<vnode_executor>()>> vem;

    template <typename executor>
    void enroll() {
        std::string signature = executor::get_signature();
        std::function<std::shared_ptr<vnode_executor>()> creator = []() {
            return std::make_shared<executor>();
        };
        vem[signature] = creator;
    }

    VnodeExecutorFactory() {
        enroll<gptneox_attention_executor<KT_REF, float>>();
        enroll<gptneox_attention_executor<KT_REF, ov::bfloat16>>();

        enroll<gptj_attention_executor<KT_REF, float>>();
        enroll<gptj_attention_executor<KT_REF, ov::bfloat16>>();

        enroll<falcon_attention_executor<KT_REF, float>>();
        enroll<falcon_attention_executor<KT_REF, ov::bfloat16>>();

        enroll<llama2_attention_executor<KT_REF, float>>();
        enroll<llama2_attention_executor<KT_REF, ov::bfloat16>>();

        enroll<llama_RMSNorm_executor<KT_REF, float>>();

        #ifdef OV_CPU_WITH_MLAS
        enroll<gptneox_attention_executor<KT_MLAS, float>>();
        enroll<gptj_attention_executor<KT_MLAS, float>>();
        enroll<falcon_attention_executor<KT_MLAS, float>>();
        enroll<llama2_attention_executor<KT_MLAS, float>>();
        #endif
        #ifdef OV_CPU_WITH_LLMDNN
        enroll<gptneox_attention_executor<KT_LLMDNN, ov::bfloat16>>();
        enroll<gptj_attention_executor<KT_LLMDNN, ov::bfloat16>>();
        enroll<falcon_attention_executor<KT_LLMDNN, ov::bfloat16>>();
        enroll<llama2_attention_executor<KT_LLMDNN, ov::bfloat16>>();
        #endif
    }

    std::shared_ptr<vnode_executor> create(std::string signature) {
        auto it = vem.find(signature);
        if (it != vem.end()) {
            auto exec = it->second();
            exec->signature = signature;
            return exec;
        }
        return nullptr;
    }

    std::shared_ptr<vnode_executor> create(std::string vtype, InferenceEngine::Precision prec_hint) {
        static int use_ref = std::getenv("USE_REF") ? atoi(std::getenv("USE_REF")) : 0;
        std::shared_ptr<vnode_executor> ret;
        std::string signature;
        std::string impl_type = "REF";
        if (use_ref) {
            impl_type = "REF";
        } else if (prec_hint == InferenceEngine::Precision::FP32) {
            impl_type = "MLAS";
        } else if (prec_hint == InferenceEngine::Precision::BF16) {
            impl_type = "LLMDNN";
        }
        signature = vtype + "," + impl_type + "," + prec_hint.name();
        if (ret = create(signature))
            return ret;
        signature = vtype + ",REF," + prec_hint.name();
        if (ret = create(signature))
            return ret;
        signature = vtype + ",REF," + InferenceEngine::Precision(InferenceEngine::Precision::FP32).name();
        if (ret = create(signature))
            return ret;
        return nullptr;
    }
};

bool VNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    const auto vnode = std::dynamic_pointer_cast<const ov::intel_cpu::VNode>(op);
    if (!vnode) {
        errorMessage = "Only VNode operation is supported";
        return false;
    }

    return true;
}

VNode::VNode(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "VNode layer with name '" + getName() + "'";

    m_vnode = std::dynamic_pointer_cast<ov::intel_cpu::VNode>(op);
    m_vtype = m_vnode->get_vtype();
    m_symbol_name2value = op->get_rt_info()["symbol_name2value"].as<decltype(m_symbol_name2value)>();
}

void VNode::getSupportedDescriptors() {
    if (!descs.empty())
        return;

    if (getParentEdges().size() != m_vnode->get_input_size())
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (getChildEdges().size() < m_vnode->get_output_size())
        IE_THROW() << errorPrefix << "has incorrect number of output edges: " << getChildEdges().size();
}

namespace {

struct VnodeKey {
    std::string vtype;
    Precision::ePrecision prec;

    size_t hash() const;
    bool operator==(const VnodeKey& rhs) const;
};

size_t VnodeKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;
    size_t seed = 0;
    seed = hash_combine(seed, vtype);
    seed = hash_combine(seed, prec);
    return seed;
}

bool VnodeKey::operator==(const VnodeKey &rhs) const {
    return vtype == rhs.vtype && prec == rhs.prec;
}
} // namespace

void VNode::initSupportedPrimitiveDescriptors() {
    static VnodeExecutorFactory factory;

    if (!supportedPrimitiveDescriptors.empty())
        return;

    // orginal precision at input port 0 as a hint of runtime precisions
    auto runtime_precision = getOriginalInputPrecisionAtPort(0);

    VnodeKey key{m_vtype, runtime_precision};
    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, [&](const VnodeKey& key) {
        auto executor = factory.create(key.vtype, key.prec);
        if (executor)
            std::cout << getName() << " created executor: " << executor->signature << std::endl;
        return executor;
    });
    m_executor = result.first;
    if (!m_executor) {
        IE_THROW() << errorPrefix << " unsupported vnode " << m_vtype << " with " << runtime_precision;
    }

    std::vector<ov::intel_cpu::PortConfigurator> inPortConfigs;
    std::vector<ov::intel_cpu::PortConfigurator> outPortConfigs;
    for (auto* p : m_executor->inputs) {
        inPortConfigs.emplace_back(LayoutType::ncsp, p->get_precision());
    }
    for (auto* p : m_executor->outputs) {
        auto inplace_to = m_executor->output_inplace_to(p);
        outPortConfigs.emplace_back(LayoutType::ncsp, p->get_precision(), false, inplace_to);
    }
    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref);
}

void VNode::execute(dnnl::stream strm) {
    if (m_executor) {
        m_executor->exec(this, strm, m_symbol_name2value);
    } else {
        IE_THROW() << errorPrefix << " Not implemented for " << m_vtype;
    }
}

void VNode::executeDynamicImpl(dnnl::stream strm) {
    if (m_executor) {
        m_executor->exec(this, strm, m_symbol_name2value);
    } else {
        IE_THROW() << errorPrefix << " Not implemented for " << m_vtype;
    }
}

bool VNode::created() const {
    return getType() == Type::VNode;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "add_custom.h"
#include <ngraph/opsets/opset10.hpp>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include <cpu/x64/jit_generator.hpp>
#include "emitters/jit_dnnl_emitters.hpp"
#include "emitters/jit_load_store_emitters.hpp"
#include "special/add_custom.hpp"
#include "ie_parallel.hpp"

using namespace InferenceEngine;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

bool AddCustom::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<op::v10::AddCustom>(op)) {
            errorMessage = "Not supported AddCustom operation version. CPU plug-in supports only 10th version.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

AddCustom::AddCustom(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }
}

void AddCustom::initSupportedPrimitiveDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(0);

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, dataPrecision}},
                         {{LayoutType::ncsp, dataPrecision}},
                          impl_desc_type::ref_any);
}

void AddCustom::createPrimitive() {
    Node::createPrimitive();
}

void AddCustom::prepareParams() {
    const auto& dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    totalElements =
        std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
}

void AddCustom::executeDynamicImpl(dnnl::stream strm) {
    auto& dims = getParentEdgeAt(0)->getMemoryPtr()->getStaticDims();
    redefineOutputMemory({dims});

    execute(strm);
}

void AddCustom::execute(dnnl::stream strm) {
    auto* node0 = reinterpret_cast<bfloat16*>(getParentEdgeAt(0)->getMemoryPtr()->GetPtr());
    auto* node1 = reinterpret_cast<bfloat16*>(getParentEdgeAt(1)->getMemoryPtr()->GetPtr());
    auto* node2 = reinterpret_cast<bfloat16*>(getParentEdgeAt(2)->getMemoryPtr()->GetPtr());
    auto* dst = reinterpret_cast<bfloat16*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    
    auto total = totalElements;
    auto count = total / 1024;
    parallel_for(count, [&](int i) {
        size_t size = 1024 * i > total ? total - 1024 * i : 1024;
        add3(node0 + i * 1024, node1 + i * 1024, node2 + i * 1024, dst + i * 1024, size);
    });    
}

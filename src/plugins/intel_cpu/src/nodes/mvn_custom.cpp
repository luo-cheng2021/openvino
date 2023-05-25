// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_custom.h"

#include <algorithm>
#include <string>
#include <vector>

#include "fake_quantize.h"
#include "eltwise.h"
#include <dnnl_extension_utils.h>
#include "utils/bfloat16.hpp"
#include "ie_parallel.hpp"

#include <cpu/x64/jit_generator.hpp>
#include <cpu/x64/jit_uni_eltwise.hpp>
#include <cpu/x64/injectors/jit_uni_depthwise_injector.hpp>
#include <cpu/x64/injectors/jit_uni_quantization_injector.hpp>
#include <cpu/x64/injectors/jit_uni_eltwise_injector.hpp>

#include <ngraph/opsets/opset6.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "utils/cpu_utils.hpp"
#include "special/mvn_custom.hpp"

using namespace dnnl;
using namespace InferenceEngine;
using namespace dnnl::impl;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::utils;

#define GET_OFF(field) offsetof(jit_mvn_call_args, field)

namespace ov {
namespace intel_cpu {
namespace node {

bool MVNCustom::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (op->get_output_partial_shape(0).rank().is_dynamic()) {
            errorMessage = "Unsupported dynamic input rank.";
            return false;
        }
        const auto& inDataRank = op->get_output_partial_shape(0).rank().get_length();
        if (inDataRank != 3) {
            errorMessage = "First input accepts ranks 3. Actual: " + std::to_string(inDataRank);
            return false;
        }

        if (auto mvnOp = ngraph::as_type_ptr<const ngraph::op::v10::MVNCustom>(op)) {
            auto axesOp = ngraph::as_type_ptr<ngraph::op::Constant>(mvnOp->get_input_node_shared_ptr(1));
            if (!axesOp) {
                errorMessage = "Constant expected as the second input.";
                return false;
            }

            auto epsMode = mvnOp->get_eps_mode();
            if (epsMode != ngraph::op::MVNEpsMode::INSIDE_SQRT &&
                    epsMode != ngraph::op::MVNEpsMode::OUTSIDE_SQRT) {
                errorMessage = std::string("Just INSIDE_SQRT and OUTSIDE_SQRT epsilon mods are supported. Actual: ") +
                        std::to_string(static_cast<int>(epsMode));
                return false;
            }
            // Validates MVNCustom node axes to check whether it can be executed on the current CPU implementation.
            // Supported cases:
            // 1D: axes: [0]
            // 2D: axes: [1]
            // 3D: axes: [1,2], [2]
            // 4D: axes: [1,2,3], [2,3]
            // 5D: axes: [1,2,3,4], [2,3,4]
            auto axesVal = axesOp->cast_vector<int>();
            for (int& axe : axesVal)
                axe = axe < 0 ? axe + inDataRank : axe;
            std::sort(axesVal.begin(), axesVal.end());
            if (inDataRank == 1) {
                if (axesVal.size() != 1 || axesVal[0] != 0) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
            } else {
                if (inDataRank > 5 || (inDataRank != axesVal.size() + 1 && inDataRank != axesVal.size() + 2)) {
                    errorMessage = "Unsupported axes.";
                    return false;
                }
                int value = inDataRank - 1;
                for (int i = axesVal.size() - 1; i >= 0; i--, value--) {
                    if (axesVal[i] != value) {
                        errorMessage = "Unsupported axes.";
                        return false;
                    }
                }
            }
        } else {
            errorMessage = "Node is not an instance of the MVNCustom operation.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MVNCustom::MVNCustom(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
        : Node(op, context, NgraphShapeInferFactory(op, EMPTY_PORT_MASK)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    epsMode_ = INSIDE_SQRT;
    if (auto mvnOp = ngraph::as_type_ptr<ngraph::op::v10::MVNCustom>(op)) {
        normalizeVariance_ = mvnOp->get_normalize_variance();
        if (normalizeVariance_ == false)
            IE_THROW(NotImplemented) << "normalizeVariance_ = false not supported.";
        epsValue_ = mvnOp->get_eps();
        if (mvnOp->get_eps_mode() == ngraph::op::MVNEpsMode::OUTSIDE_SQRT) {
            epsMode_ = OUTSIDE_SQRT;
        }

        initAcrossChannels_ = false;
        const auto& inDataShapeSize = getInputShapeAtPort(0).getRank();
        if (inDataShapeSize == mvnOp->input_value(1).get_shape()[0] + 1 || inDataShapeSize == 1)
            IE_THROW(NotImplemented) << "initAcrossChannels = true not supported.";
    }
    execAcrossChannels_ = initAcrossChannels_;
}

void MVNCustom::getSupportedDescriptors() {}

void MVNCustom::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    Precision inputPrecision = Precision::BF16;
    Precision outputPrecision = Precision::BF16;

    if (!fusedWith.empty()) {
        outputPrecision = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        if (fusedWith.size() == 1) {
            auto& node = fusedWith[0];
            if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
                auto quant = fakeQuantizeNode->getInputScale();
                if (quant.size() != 1) {
                    qkv_quant = std::move(quant);
                    supportedPostops = Fastpath_Postops_FQ;
                } else {
                    auto& shape = getInputShapeAtPort(0);
                    if (shape.getRank() == 3) {
                        auto dim = shape.getDims()[2];
                        if (dim != Shape::UNDEFINED_DIM) {
                            // std::cout << "fq " << dim << " scale " << quant[0] << " out precision:"
                            //     << outputPrecision << " \n";

                            qkv_quant = std::vector<float>(dim, quant[0]);
                            supportedPostops = Fastpath_Postops_FQ;
                        }
                    }
                }
            }
        }
    } else {
        supportedPostops = Fastpath_Postops_No;
    }

    // TODO [DS]: inplace
    bool canBeInplace = !isDynamicNode() && (inputPrecision.size() == outputPrecision.size()) &&
                        (getParentEdgeAt(0)->getParent()->getChildEdges().size() == 1) &&
                        !getParentEdgeAt(0)->getParent()->isConstant();

    const size_t inputsNum = getParentEdges().size();
    NodeConfig config;
    config.dynBatchSupport = false;
    config.inConfs.resize(inputsNum);
    config.outConfs.resize(1);
    config.inConfs[0].constant(false);
    config.outConfs[0].constant(false);
    config.inConfs[0].inPlace(-1);
    config.outConfs[0].inPlace(canBeInplace ? 0 : -1);

    config.inConfs[1].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(InferenceEngine::Precision::I32, getInputShapeAtPort(1)));
    config.inConfs[1].constant(true);

    config.inConfs[2].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(InferenceEngine::Precision::FP32, getInputShapeAtPort(2)));
    config.inConfs[2].constant(true);

    config.inConfs[3].setMemDesc(std::make_shared<CpuBlockedMemoryDesc>(InferenceEngine::Precision::FP32, getInputShapeAtPort(3)));
    config.inConfs[3].constant(true);

    auto& creatorsMap = BlockedDescCreator::getCommonCreators();
    auto pushDesc = [&](LayoutType format, impl_desc_type impl_type) {
        config.inConfs[0].setMemDesc(creatorsMap.at(format)->createSharedDesc(inputPrecision, getInputShapeAtPort(0)));
        config.outConfs[0].setMemDesc(creatorsMap.at(format)->createSharedDesc(outputPrecision, getOutputShapeAtPort(0)));
        supportedPrimitiveDescriptors.push_back({config, impl_type});
    };

    impl_desc_type impl_type;
    if (mayiuse(cpu::x64::avx512_core)) {
        impl_type = impl_desc_type::jit_avx512;
    } else if (mayiuse(cpu::x64::avx2)) {
        impl_type = impl_desc_type::jit_avx2;
    } else if (mayiuse(cpu::x64::sse41)) {
        impl_type = impl_desc_type::jit_sse42;
    } else {
        impl_type = impl_desc_type::ref;
    }

    // planar
    if (canBeInplace)
        config.inConfs[0].inPlace(0);
    pushDesc(LayoutType::ncsp, impl_type);
}

void MVNCustom::prepareParams() {
}

void MVNCustom::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void MVNCustom::execute(dnnl::stream strm) {
    auto &dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto &srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    auto &weightMemPtr = getParentEdgeAt(2)->getMemoryPtr();
    auto &biasMemPtr = getParentEdgeAt(3)->getMemoryPtr();

    uint8_t *dst_data = reinterpret_cast<uint8_t*>(dstMemPtr->GetPtr());
    uint8_t *src_data = reinterpret_cast<uint8_t*>(srcMemPtr->GetPtr());
    float *weight_data = reinterpret_cast<float*>(weightMemPtr->GetPtr());
    float *bias_data = reinterpret_cast<float*>(biasMemPtr->GetPtr());
    const SizeVector& in_dims = srcMemPtr->getStaticDims();
    size_t C1 = in_dims[2];
    size_t C2 = C1 * in_dims[1];
    auto eps = epsValue_;
    bool inside_sqrt = epsMode_ == INSIDE_SQRT;

    if (supportedPostops == Fastpath_Postops_No) {
        parallel_for2d(in_dims[0], in_dims[1], [&](size_t b, size_t c) {
            auto offset = b * C2 + c * C1;
            mvn_line_weight_bias(reinterpret_cast<bfloat16*>(src_data) + offset, in_dims[2], eps, inside_sqrt,
                reinterpret_cast<bfloat16*>(dst_data) + offset, weight_data, bias_data);
        });
    } else {
        parallel_for2d(in_dims[0], in_dims[1], [&](size_t b, size_t c) {
            auto offset = b * C2 + c * C1;
            mvn_line_weight_bias(reinterpret_cast<bfloat16*>(src_data) + offset, in_dims[2], eps, inside_sqrt,
                reinterpret_cast<int8_t*>(dst_data) + offset, qkv_quant.data(), weight_data, bias_data);
        });
    }
}

bool MVNCustom::canFuse(const NodePtr& node) const {
    // limit post ops to unary when shape transformed on channel
    // 1D only fused with unary
    int inputRank = getInputShapeAtPort(0).getRank();
    bool unaryEltwise = one_of(node->getAlgorithm(), Algorithm::EltwiseRelu,
                                                     Algorithm::EltwiseGelu,
                                                     Algorithm::EltwiseElu,
                                                     Algorithm::EltwiseSigmoid,
                                                     Algorithm::EltwiseClamp,
                                                     Algorithm::EltwiseTanh,
                                                     Algorithm::EltwiseSwish,
                                                     Algorithm::EltwiseHswish,
                                                     Algorithm::EltwiseMish,
                                                     Algorithm::EltwiseHsigmoid,
                                                     Algorithm::EltwiseRoundHalfToEven,
                                                     Algorithm::EltwiseRoundHalfAwayFromZero,
                                                     Algorithm::EltwiseAbs,
                                                     Algorithm::EltwiseSqrt,
                                                     Algorithm::EltwiseSoftRelu);
    if ((inputRank == 1 && !unaryEltwise) ||
        (inputRank == 2 && !unaryEltwise && initAcrossChannels_)) {
        return false;
    }

    return canFuseSimpleOperation(node);
}

bool MVNCustom::created() const {
    return getType() == Type::MVNCustom;
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

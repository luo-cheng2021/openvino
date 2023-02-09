// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gpt_neox_attn.h"
#include <ngraph/opsets/opset10.hpp>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>

using namespace InferenceEngine;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

bool GPTNeoxAttn::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<op::v10::GPTNeoxAttn>(op)) {
            errorMessage = "Not supported GPTNeoxAttn operation version. CPU plug-in supports only 10th version.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

GPTNeoxAttn::GPTNeoxAttn(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto attn_op = ov::as_type_ptr<ov::op::v10::GPTNeoxAttn>(op);
    layerNum = attn_op->m_layer_num;
    headNum = attn_op->m_head_num;
    sizePerHead = attn_op->m_size_per_head;
    hiddenSize = attn_op->m_hidden_size;
    intermediateSize = attn_op->m_intermediate_size;
    layerNormEps = attn_op->m_layer_norm_eps;
    maxPositionEmbeddings = attn_op->m_max_position_embeddings;
    rotaryEmbBase = attn_op->m_rotary_emb_base;
    rotaryPct = attn_op->m_rotary_pct;
    useParallelResidual = attn_op->m_use_parallel_residual;
    vocabSize = attn_op->m_vocab_size;
    maxSeqLen = attn_op->m_max_seq_len;
    curLayerNum = attn_op->m_cur_layer_num;
}

void GPTNeoxAttn::initSupportedPrimitiveDescriptors() {
    auto dataPrecision = getOriginalInputPrecisionAtPort(IN_QKV);

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, Precision::I32}},
                         {{LayoutType::ncsp, dataPrecision}},
                          impl_desc_type::ref_any);
}

void GPTNeoxAttn::createPrimitive() {
    Node::createPrimitive();
}

void GPTNeoxAttn::prepareParams() {
    auto& qkvMemPtr = getParentEdgeAt(IN_QKV)->getMemoryPtr();
    auto& pastKeysMemPtr = getParentEdgeAt(IN_PAST_KEYS)->getMemoryPtr();
    auto& pastKeysNumMemPtr = getParentEdgeAt(IN_PAST_KEYS_NUM)->getMemoryPtr();
}

void GPTNeoxAttn::executeDynamicImpl(dnnl::stream strm) {
    auto qkvDataDims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    qkvDataDims.back() = qkvDataDims.back() / 3;
    redefineOutputMemory({qkvDataDims});

    execute(strm);
}

void GPTNeoxAttn::execute(dnnl::stream strm) {
    const void* qkv = getParentEdgeAt(IN_QKV)->getMemoryPtr()->GetPtr();
    const void* past_keys = getParentEdgeAt(IN_PAST_KEYS)->getMemoryPtr()->GetPtr();
    const int* past_keys_num = reinterpret_cast<const int*>(getParentEdgeAt(IN_PAST_KEYS_NUM)->getMemoryPtr()->GetPtr());
    const void* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    // robery embbeding
    // mha
}

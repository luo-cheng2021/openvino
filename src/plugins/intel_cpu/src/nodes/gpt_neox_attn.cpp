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
    normalFactor = 1.0f / sqrtf(static_cast<float>(sizePerHead));

    rotaryNdims = static_cast<int>(sizePerHead * rotaryPct);
}

void GPTNeoxAttn::initSupportedPrimitiveDescriptors() {
    auto dataPrecision = getOriginalInputPrecisionAtPort(IN_QKV);
    dataTypeSize = dataPrecision.size();

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
    auto dataPrecision = getOriginalInputPrecisionAtPort(IN_QKV);

    // if (!dataMemPtr || !dataMemPtr->isAllocated()) {
    //     THROW_ERROR << " has not allocated input data memory.";
    // }
    // for (int i = 0; i < 4; i++) {
    //     if (definedOutputs[i]) {
    //         auto& dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
    //         if (!dstMemPtr || !dstMemPtr->isAllocated()) {
    //             THROW_ERROR << " has not allocated output memory at port " << i;
    //         }
    //     }
    // }
    // if (getSelectedPrimitiveDescriptor() == nullptr) {
    //     THROW_ERROR << " has unidentified preferable primitive descriptor.";
    // }

    // size_t srcLen = 1;
    // if (flattened) {
    //     srcLen = getParentEdgeAt(IN_DATA)->getMemoryPtr()->GetSize() / dataTypeSize;
    // } else {
    //     auto dstDataShape = getParentEdgeAt(IN_DATA)->getMemoryPtr()->getStaticDims();
    //     srcLen = dstDataShape[axis];
    // }
    // firstUniTmp.resize(srcLen, 0);
    // inToOutTmp.resize(srcLen);
    // occurTmp.resize(srcLen);
    const auto& qkvDims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    const auto batch = static_cast<int>(qkvDims[0]);
    const auto seqLen = static_cast<int>(qkvDims[1]);
    layerOffsetInPastKey = dataTypeSize * (layerNum * 2 * batch * headNum * maxSeqLen * sizePerHead);
    layerOffsetInPastValue = layerOffsetInPastKey + dataTypeSize * (1 * batch * headNum * maxSeqLen * sizePerHead);
    gpt::MHAGPT::CreateParam param = {
        batch, headNum, maxSeqLen, sizePerHead, maxSeqLen, 1.0f, dataPrecision, true
    };
    auto curScratchSize = gpt::MHAGPT::query_scratch_size(param);
    if (scratchSize < curScratchSize) {
        scratchMem.reset(new Memory(getEngine()));
        CpuBlockedMemoryDesc desc(dataPrecision, {static_cast<size_t>(curScratchSize)});
        scratchMem->Create(desc, nullptr, false);

        scratchSize = curScratchSize;
    }
}

void GPTNeoxAttn::executeDynamicImpl(dnnl::stream strm) {
    auto qkvDataDims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    qkvDataDims.back() = qkvDataDims.back() / 3;
    redefineOutputMemory({qkvDataDims});

    execute(strm);
}

void GPTNeoxAttn::execute(dnnl::stream strm) {
    // [batch, seq_len, (num_heads * 3 * head_size)]
    auto* qkv = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_QKV)->getMemoryPtr()->GetPtr());
    // [layer_num, 2, batch, num_heads, seq_len, head_size]
    const auto* pastKeys = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_PAST_KEYS)->getMemoryPtr()->GetPtr());
    const int* pastKeysNum = reinterpret_cast<const int*>(getParentEdgeAt(IN_PAST_KEYS_NUM)->getMemoryPtr()->GetPtr());
    auto* dstData = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto& qkvDims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    const auto batch = qkvDims[0];
    const auto newSeqOffset = pastKeysNum[0];

    // first token will write to pastKeys offset 0
    bool firstToken = newSeqOffset == 0;
    if (firstToken) {
        assert(false);
    } else {
        // seq_len == 1 branch
        // [batch, seq_len, (num_heads * 3 * head_size)]
        //   --> [batch, seq_len, num_heads, 3 * head_size]
        // [batch, seq_len, num_attention_heads, 3 * head_size] --> 3 [batch, num_attention_heads, seq_len, head_size]
        auto query = qkv;                       // qkv[..., : self.head_size].permute(0, 2, 1, 3)
        auto key = qkv + sizePerHead;           // qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
        auto value = qkv + 2 * sizePerHead;     // qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)

        // robery embbeding start
        // // Compute rotary embeddings on rotary_ndims
        // auto query_rot = query[..., : self.rotary_ndims]
        // auto query_pass = query[..., self.rotary_ndims :]
        // auto key_rot = key[..., : self.rotary_ndims]
        // auto key_pass = key[..., self.rotary_ndims :]
        // // Compute token offset for rotary embeddings (when decoding)
        // seq_len = key.shape[-2]
        // offset = 0
        // if has_layer_past:
        //     offset = layer_past[0].shape[-2]
        //     seq_len += offset
        // // rotary_emb
        // cos = cos_cached[:seq_len, ...]
        // sin = sin_cached[:seq_len, ...]
        // q_embed, k_embed;
        // apply_rotary_pos_emb(query_rot, key_rot, cos, sin, offset, q_embed, k_embed);
        // query = torch.cat((query, query_pass), dim=-1)
        // key = torch.cat((key, key_pass), dim=-1)
        // robery embbeding end

        // copy memory to pastKeys
        auto newPastKeyPtr = pastKeys + layerOffsetInPastKey + dataTypeSize * newSeqOffset * sizePerHead;
        auto newPastValuePtr = pastKeys + layerOffsetInPastValue + dataTypeSize * newSeqOffset * sizePerHead;
        // dst, line_width, dst_stride, line_height, src, src_stride
        // key = torch.cat((past_key, key), dim=-2)
        // MemcpyStride(newPastKeyPtr, sizePerHead, sizePerHead * maxSeqLen, headNum,
        //     key, sizePerHead * 3);
        // key_rot part: TODO
        // key_pass part
        // auto key_pass = key[..., self.rotary_ndims :]
        // key = torch.cat((key, key_pass), dim=-1)
        // MemcpyStride(newPastKeyPtr + rotaryNdims, sizePerHead - rotaryNdims, sizePerHead * maxSeqLen, headNum,
        //     key + rotaryNdims, sizePerHead * 3);
        // // value = torch.cat((past_value, value), dim=-2)
        // MemcpyStride(newPastValuePtr, sizePerHead, sizePerHead * maxSeqLen, headNum,
        //     value, sizePerHead * 3);
        // attn_output = _attn(query, key, value)

        // Reshape outputs
        // auto _merge_heads = [] (tensor, num_attention_heads, attn_head_size) {
        //     // tensor [bs, num_attention_heads, seq_len, attn_head_size]
        //     tensor = tensor.permute(0, 2, 1, 3).contiguous()
        //     // -> [bs, seq_len, num_attention_heads, attn_head_size]
        //     tensor = tensor.view(tensor.size(0), tensor.size(1), num_attention_heads * attn_head_size)
        //     // -> [bs, seq_len, hidden_size]
        // };
        // attn_output = _merge_heads(attn_output, self.num_attention_heads, self.head_size)
        auto& mha = mhaGPTs[(static_cast<size_t>(batch) << 32) + static_cast<size_t>(newSeqOffset + 1)];
        if (!mha) {
            auto dataPrecision = getOriginalInputPrecisionAtPort(IN_QKV);

            gpt::MHAGPT::CreateParam param = {
                batch, headNum, 1, sizePerHead, newSeqOffset + 1, normalFactor, dataPrecision, false
            };
            mha = std::make_shared<gpt::MHAGPT>();
            mha->create(param, static_cast<uint8_t*>(scratchMem->GetData()));
        }
        auto head_stride_in_kv = sizePerHead * maxSeqLen;
        auto batch_stride_in_kv = head_stride_in_kv * headNum;
        // q: [batch, num_heads, query_seq_len, head_size]
        // k: [batch, num_heads, key_seq_len, head_size]
        // v: [batch, num_heads, value_seq_len, head_size]
        // attention_mask: [batch, 1, 1, key_seq_len]
        // attn_output: [batch, num_heads, query_seq_len, head_size]
        gpt::MHAGPT::ExecParam param = {
            query, key, value,
            nullptr, // TODO
            dstData,
            sizePerHead, sizePerHead * headNum,
            head_stride_in_kv, batch_stride_in_kv,
            0,      // TODO
        };

        mha->exec(param);
    }

    // robery embbeding
    // mha
}

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class attn_gpt;
class GPTAttn : public Node {
public:
    GPTAttn(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override { return getType() == Type::GPTAttn; }

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    void prepareParams() override;
    bool needShapeInfer() const override { return false; }
    bool canFuse(const NodePtr& node) const override;

private:
    void extractQuantParam();
    size_t layerNum = 32;
    size_t headNum = 32;
    size_t sizePerHead = 80;
    size_t rotaryEmbBase = 10000;
    float rotaryPct = 0.25;
    size_t curLayerNum = 0;
    size_t maxSeqLen = 400;
    float normalFactor = 0.0f;
    bool usePosition2d = false;
    // aligned to cache line
    size_t sizePerHeadAligned = 80;
    InferenceEngine::Precision inputDataType;
    InferenceEngine::Precision outputDataType;
    InferenceEngine::Precision mhaInputDataType;
    int64_t inputDataTypeSize = 1;
    int64_t mhaInputDataTypeSize = 1;
    float q_quant = 0.0f;
    float k_quant = 0.0f;
    float qk_quant = 0.0f;
    float v_quant = 0.0f;
    std::vector<float> qkv_quant;     // next node quant scale
    bool useInt8 = false;

    std::shared_ptr<attn_gpt> attnGPT;

    static constexpr size_t IN_QKV           = 0;
    static constexpr size_t IN_PAST_KEYS_NUM = 1;
    static constexpr size_t IN_BEAM_IDX      = 2;
    static constexpr size_t IN_ATTN_MASK     = 3;
    static constexpr size_t IN_POSITION_IDS  = 4;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

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

struct jit_rotary_compile_params {
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    size_t head_num;
    size_t rotary_ndims;
    size_t hidden_size;
    size_t max_seq_len;
    size_t size_per_head;
    size_t size_per_head_aligned;
};

struct jit_rotary_call_args {
    void* q_src;
    void* k_src;
    float* cos;
    float* sin;
    void* q_dst;
    void* k_dst;
    size_t q_dst_stride;
    float* q_quant;
    float* k_quant;
};

struct jit_uni_rotary_kernel {
    void (*ker_)(const jit_rotary_call_args*);

    void operator()(const jit_rotary_call_args* call_args) {
        assert(ker_);
        ker_(call_args);
    }

    explicit jit_uni_rotary_kernel(const jit_rotary_compile_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_rotary_kernel() {}

    virtual void create_ker() = 0;

    jit_rotary_compile_params jcp_;
};

class GPTNeoxAttn : public Node {
public:
    GPTNeoxAttn(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override { return getType() == Type::GPTNeoxAttn; }

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    void prepareParams() override;
    bool needShapeInfer() const override { return false; }
    void initRotery(size_t maxSeqLen);
    void reinitAttentionMask(size_t batch, size_t max_seq_len);
    void applyRotaryPosEmb(uint8_t* q_src, uint8_t* k_src, uint8_t* q_dst, const std::vector<uint8_t*>& k_dst, size_t k_start,
                           float* cos_cached, float* sin_cached, size_t batch, size_t qSeqLen, size_t offset);
    void updateAttnMask(const int* attn_mask, size_t batch, size_t seq_len);
    bool canFuse(const NodePtr& node) const override;

private:
    void extractQuantParam();
    size_t layerNum = 32;
    size_t headNum = 32;
    size_t sizePerHead = 80;
    size_t hiddenSize = 32 * 80;
    size_t maxPositionEmbeddings = 2048;
    size_t rotaryEmbBase = 10000;
    float rotaryPct = 0.25;
    size_t vocabSize = 50304;
    size_t maxSeqLen = 400;
    float normalFactor = 0.0f;
    // aligned to cache line
    size_t sizePerHeadAligned = 80;


    InferenceEngine::Precision inputDataType;
    InferenceEngine::Precision outputDataType;
    InferenceEngine::Precision mhaInputDataType;
    int64_t inputDataTypeSize = 1;
    int64_t mhaInputDataTypeSize = 1;
    int rotaryNdims = 0;
    std::unique_ptr<gpt::MHAGPT> mhaGPT;
    std::shared_ptr<float> attnMasks;
    std::shared_ptr<float> cosCached;
    std::shared_ptr<float> sinCached;
    std::shared_ptr<uint8_t> queryTranspose;
    std::unique_ptr<jit_uni_rotary_kernel> rotaryKernel;

    float q_quant = 0.0f;
    float k_quant = 0.0f;
    float qk_quant = 0.0f;
    float v_quant = 0.0f;
    std::vector<float> qkv_quant;     // next node quant scale
    bool useInt8 = false;

    static constexpr size_t IN_QKV           = 0;
    static constexpr size_t IN_PAST_KEYS_NUM = 1;
    static constexpr size_t IN_BEAM_IDX      = 2;
    static constexpr size_t ATTN_MASK_IDX    = 3;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

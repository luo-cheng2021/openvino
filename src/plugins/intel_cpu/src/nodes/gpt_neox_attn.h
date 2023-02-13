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
    size_t head_num;
    size_t rotary_ndims;
    size_t hidden_size;
    size_t q_seq_len;
    size_t max_seq_len;
    size_t size_per_head;
    size_t src_stride;
    size_t q_dst_stride;
    size_t k_dst_stride;
};

struct jit_rotary_call_args {
    void* q_src;
    void* k_src;
    float* cos;
    float* sin;
    void* q_dst;
    void* k_dst;
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
    void applyRotaryPosEmb(uint8_t* q_src, uint8_t* k_src, uint8_t* q_dst, uint8_t* k_dst,
                           float* cos_cached, float* sin_cached, size_t batch, size_t qSeqLen, size_t offset);

private:
    size_t layerNum = 32;
    size_t headNum = 32;
    size_t sizePerHead = 80;
    size_t hiddenSize = 32 * 80;
    size_t intermediateSize = 10240;
    float layerNormEps = 1e-5;
    size_t maxPositionEmbeddings = 2048;
    size_t rotaryEmbBase = 10000;
    float rotaryPct = 0.25;
    bool useParallelResidual = true;
    size_t vocabSize = 50304;
    size_t maxSeqLen = 400;
    size_t curLayerNum = 0;
    float normalFactor = 0.0f;

    InferenceEngine::Precision dataPrecision;
    int64_t dataTypeSize = 1;
    int64_t layerOffsetInPastKey = 0;
    int64_t layerOffsetInPastValue = 0;
    int rotaryNdims = 0;
    // mha kernels, key = batch(high 32bit) + value length(low 32bit, including current and past)
    std::unordered_map<size_t, std::shared_ptr<gpt::MHAGPT>> mhaGPTs;
    std::vector<std::vector<float>> attnMasks;
    std::vector<std::vector<float>> cosCached;
    std::vector<std::vector<float>> sinCached;
    std::vector<uint8_t> queryTranspose;
    std::unique_ptr<jit_uni_rotary_kernel> rotaryKernel;

    static constexpr size_t IN_QKV           = 0;
    static constexpr size_t IN_PAST_KEYS     = 1;
    static constexpr size_t IN_PAST_KEYS_NUM = 2;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

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

private:
    int layerNum = 32;
    int headNum = 32;
    int sizePerHead = 80;
    int hiddenSize = 32 * 80;
    int intermediateSize = 10240;
    float layerNormEps = 1e-5;
    int maxPositionEmbeddings = 2048;
    int rotaryEmbBase = 10000;
    float rotaryPct = 0.25;
    bool useParallelResidual = true;
    int vocabSize = 50304;
    int maxSeqLen = 400;
    int curLayerNum = 0;
    float normalFactor = 0.0f;

    int64_t dataTypeSize = 1;
    int64_t layerOffsetInPastKey = 0;
    int64_t layerOffsetInPastValue = 0;
    int rotaryNdims = 0;
    MemoryPtr scratchMem;
    size_t scratchSize = 0;
    std::unordered_map<size_t, std::shared_ptr<gpt::MHAGPT>> mhaGPTs;

    static constexpr size_t IN_QKV           = 0;
    static constexpr size_t IN_PAST_KEYS     = 1;
    static constexpr size_t IN_PAST_KEYS_NUM = 2;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov

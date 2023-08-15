// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope.h"

#include <dnnl_extension_utils.h>
#include <onednn/dnnl.h>

#include <chrono>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cpu/x64/jit_generator.hpp>
#include <ie_ngraph_utils.hpp>
#include <shape_inference/shape_inference_internal_dyn.hpp>
#include <string>
#include <vector>

#include "common/bfloat16.hpp"
#include "common/cpu_memcpy.h"
#include "emitters/x64/jit_dnnl_emitters.hpp"
#include "emitters/x64/jit_load_store_emitters.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "nodes/common/cpu_convert.h"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

RoPE::RoPE(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context)
    : Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    const auto node = std::dynamic_pointer_cast<const RoPENode>(op);
    m_config = node->get_config();
}

template <typename T>
void rope_kernel(bool interleave,
                 T* dst,
                 T* src,
                 float* cos_tab,
                 float* sin_tab,
                 size_t rotary_dims,
                 size_t feature_size) {
    auto half_rotary_dims = rotary_dims / 2;
    size_t i = 0;
    if (interleave) {
        size_t j = 0;
        for (i = 0; i < rotary_dims; i += 2, j++) {
            dst[i] = cos_tab[j] * src[i] - sin_tab[j] * src[i + 1];
            dst[i + 1] = cos_tab[j] * src[i + 1] + sin_tab[j] * (src[i]);
        }
    } else {
        // even terms
        for (; i < half_rotary_dims; i++) {
            dst[i] = cos_tab[i] * src[i] + sin_tab[i] * (-src[i + half_rotary_dims]);
        }
        // odd terms
        for (; i < rotary_dims; i++) {
            dst[i] = cos_tab[i] * src[i] + sin_tab[i] * (src[i - half_rotary_dims]);
        }
    }
    // non-embedded
    for (; i < feature_size; i++) {
        dst[i] = src[i];
    }
}

template <typename T>
struct RoPEExecutor : public RoPE::Executor {
    void execute(RoPE* pnode) override {
        auto& config = pnode->getConfig();

        if (config.is_interleaved) {
            execute_interleaved(pnode);
        } else {
            execute_normal(pnode);
        }
    }

    void execute_normal(RoPE* pnode) {
        auto& config = pnode->getConfig();

        IE_ASSERT(config.is_cos_sin_combined == false);

        ov::intel_cpu::PlainTensor<T> t_src(pnode->getParentEdgeAt(0)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<float> t_cos(pnode->getParentEdgeAt(1)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<float> t_sin(pnode->getParentEdgeAt(2)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<int32_t> gather;
        ov::intel_cpu::PlainTensor<T> t_past;
        if (config.reshape_H) {
            assert(t_src.m_rank == 3);
            assert((t_src.size(2) % config.reshape_H) == 0);
            auto reshape_X = t_src.size(2) / config.reshape_H;
            t_src = t_src.reshape({t_src.size(0), t_src.size(1), config.reshape_H, reshape_X});
        }
        if (config.slice_stop - config.slice_start > 0) {
            t_src = t_src.slice(3, config.slice_start, config.slice_stop);
        }
        if (config.input_trans0213) {
            t_src = t_src.permute({0, 2, 1, 3});
        }
        if (config.gather_position_arg_id > 0) {
            gather.reset(pnode->getParentEdgeAt(config.gather_position_arg_id)->getMemoryPtr());
        }

        auto batch_size = t_src.size(0);
        auto head_cnt = t_src.size(1);
        auto seq_len = t_src.size(2);
        auto feature_size = t_src.size(3);

        size_t past_len = 0;
        if (config.concat_with_past_arg_id > 0) {
            t_past.reset(pnode->getParentEdgeAt(config.concat_with_past_arg_id)->getMemoryPtr());
            past_len = t_past.size(2);
        }

        VectorDims result_shape{batch_size, head_cnt, past_len + seq_len, feature_size};

        pnode->redefineOutputMemory({result_shape});

        ov::intel_cpu::PlainTensor<T> t_dst(pnode->getChildEdgeAt(0)->getMemoryPtr());

        auto rotary_dims = config.ndims;
        auto half_rotary_dims = rotary_dims / 2;

        parallel_for3d(batch_size, head_cnt, seq_len, [&](size_t b, size_t h, size_t p) {
            auto cos_pos = p;
            if (gather) {
                if (gather.m_rank == 4)
                    cos_pos = gather.at({b, h, p, 0}, true);
                else
                    cos_pos = gather.at({b, p}, true);
            }
            auto* src = &t_src.at({b, h, p, 0});
            auto* cos = &t_cos.at({b, h, cos_pos, 0}, true);
            auto* sin = &t_sin.at({b, h, cos_pos, 0}, true);
            auto* dst = &t_dst.at({b, h, past_len + p, 0});

            if (past_len) {
                memcpy(&t_dst.at({b, h, 0, 0}), &t_past.at({b, h, 0, 0}), past_len * feature_size * sizeof(T));
            }

            rope_kernel(false, dst, src, cos, sin, rotary_dims, feature_size);
        });
    }

    void execute_interleaved(RoPE* pnode) {
        auto& config = pnode->getConfig();
        IE_ASSERT(config.concat_with_past_arg_id == 0);
        IE_ASSERT(config.input_trans0213 == false);
        IE_ASSERT(config.gather_position_arg_id == 0);

        ov::intel_cpu::PlainTensor<T> t_src(pnode->getParentEdgeAt(0)->getMemoryPtr());
        ov::intel_cpu::PlainTensor<float> t_cos;
        ov::intel_cpu::PlainTensor<float> t_sin;

        if (config.is_cos_sin_combined) {
            // B, L, 64
            ov::intel_cpu::PlainTensor<float> combined(pnode->getParentEdgeAt(1)->getMemoryPtr());
            auto ndims = combined.size(2);
            t_sin = combined.slice(2, 0, ndims / 2);
            t_cos = combined.slice(2, ndims / 2, ndims);
        } else {
            t_cos.reset(pnode->getParentEdgeAt(1)->getMemoryPtr());
            t_sin.reset(pnode->getParentEdgeAt(2)->getMemoryPtr());
        }

        if (config.reshape_H) {
            assert(t_src.m_rank == 3);
            assert((t_src.size(2) % config.reshape_H) == 0);
            auto reshape_X = t_src.size(2) / config.reshape_H;
            t_src = t_src.reshape({t_src.size(0), t_src.size(1), config.reshape_H, reshape_X});
        }

        if (config.slice_stop - config.slice_start > 0) {
            t_src = t_src.slice(3, config.slice_start, config.slice_stop);
        }

        // B,L,H,S
        auto batch_size = t_src.size(0);
        auto seq_len = t_src.size(1);
        auto head_cnt = t_src.size(2);
        auto feature_size = t_src.size(3);

        if (config.output_trans0213) {
            pnode->redefineOutputMemory({{batch_size, head_cnt, seq_len, feature_size}});
        } else {
            pnode->redefineOutputMemory({{batch_size, seq_len, head_cnt, feature_size}});
        }
        ov::intel_cpu::PlainTensor<T> t_dst(pnode->getChildEdgeAt(0)->getMemoryPtr());

        auto rotary_dims = config.ndims ? config.ndims : (t_cos.size(3) * 2);
        if (rotary_dims > feature_size)
            rotary_dims = feature_size;

        parallel_for3d(batch_size, seq_len, head_cnt, [&](size_t b, size_t p, size_t h) {
            auto* x = &t_src.at({b, p, h, 0});
            float* cos = &t_cos.at({b, p, 0}, true);
            float* sin = &t_sin.at({b, p, 0}, true);
            T* dst;
            if (config.output_trans0213) {
                dst = &t_dst.at({b, h, p, 0});
            } else {
                dst = &t_dst.at({b, p, h, 0});
            }

            rope_kernel(true, dst, x, cos, sin, rotary_dims, feature_size);
        });
    }
};

namespace {
struct RoPEKey {
    InferenceEngine::Precision::ePrecision prec;

    size_t hash() const {
        using namespace dnnl::impl;
        using namespace dnnl::impl::primitive_hashing;
        size_t seed = 0;
        seed = hash_combine(seed, prec);
        return seed;
    }
    bool operator==(const RoPEKey& rhs) const {
        return prec == rhs.prec;
    }
};
}  // namespace

void RoPE::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;
    auto srcPrecision = getOriginalInputPrecisionAtPort(0);
    auto dstPrecision = getOriginalOutputPrecisionAtPort(0);

    auto rtPrecision = srcPrecision;
    auto CosSinPrecision = InferenceEngine::Precision::FP32;  // rtPrecision

    RoPEKey key{rtPrecision};
    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, [&](const RoPEKey& key) {
        std::shared_ptr<Executor> executor;
        if (key.prec == InferenceEngine::Precision::BF16) {
            executor = std::make_shared<RoPEExecutor<ov::bfloat16>>();
        } else {
            executor = std::make_shared<RoPEExecutor<float>>();
        }
        if (executor)
            std::cout << getName() << " created RoPEExecutor." << std::endl;
        return executor;
    });

    m_executor = result.first;
    if (!m_executor) {
        IE_THROW() << getName() << ": failed to create RoPEExecutor.";
    }

    // initialize input ports
    std::vector<PortConfigurator> inPortConfigs;
    inPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getInputShapeAtPort(0), false, -1);
    inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(1), false, -1);

    if (!m_config.is_cos_sin_combined) {
        inPortConfigs.emplace_back(LayoutType::ncsp, CosSinPrecision, getInputShapeAtPort(2), false, -1);
    }

    if (m_config.gather_position_arg_id > 0) {
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   InferenceEngine::Precision::I32,
                                   getInputShapeAtPort(m_config.gather_position_arg_id),
                                   false,
                                   -1);
    }
    if (m_config.concat_with_past_arg_id > 0) {
        inPortConfigs.emplace_back(LayoutType::ncsp,
                                   rtPrecision,
                                   getInputShapeAtPort(m_config.concat_with_past_arg_id),
                                   false,
                                   -1);
    }
    // initialize output port
    std::vector<PortConfigurator> outPortConfigs;
    outPortConfigs.emplace_back(LayoutType::ncsp, rtPrecision, getOutputShapeAtPort(0), false, -1);

    addSupportedPrimDesc(inPortConfigs, outPortConfigs, impl_desc_type::ref_any);
}

void RoPE::execute(dnnl::stream strm) {
    m_executor->execute(this);
}

bool RoPE::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        const auto node = std::dynamic_pointer_cast<const RoPENode>(op);
        if (!node) {
            errorMessage = "Only RoPENode operation is supported";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov

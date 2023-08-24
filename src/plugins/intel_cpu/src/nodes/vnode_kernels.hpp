
// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <dnnl_extension_utils.h>
#include <node.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/parallel.hpp"
#include "utils/profiler.hpp"
#include "vnode_utils.hpp"
#include "x86intrin.h"
#ifdef OV_CPU_WITH_LLMDNN
#    include "llm_emb_gpt.hpp"
#    include "llm_mha_gpt.hpp"
#    include "llm_mm.hpp"
#endif
#include "mlas/sgemm.hpp"
#include "utils/plain_tensor.hpp"
namespace ov {
namespace intel_cpu {
namespace node {

//============================ kernels ============================
enum KernelTypes { KT_REF, KT_LLMDNN, KT_MLAS };

template <KernelTypes KType>
struct ktype_name_of {
    static constexpr char * value = "?";
};

template <>
struct ktype_name_of<KT_REF> {
    static constexpr char * value = "REF";
};
template <>
struct ktype_name_of<KT_LLMDNN> {
    static constexpr char * value = "LLMDNN";
};
template <>
struct ktype_name_of<KT_MLAS> {
    static constexpr char * value = "MLAS";
};

// default implementation: reference
template <KernelTypes KType, typename T>
struct RoPE_kernel {
    RoPE_kernel() = default;

    // [B, L] mapping position of tokens in current query/key to position in cos/sin table
    PlainTensor<int32_t> gather_pos_idx;

    void operator()(PlainTensor<T>& cur_query,   // [B,L,H,S] or [B,L,G,gH,S]
                    PlainTensor<T>& cur_key,     // [B,L,H,S] or [B,L,G,1,S]
                    PlainTensor<T>& cur_value,   // [B,L,H,S] or [B,L,G,1,S]
                    PlainTensor<T>& past_key,    // B,H,L0,S
                    PlainTensor<T>& past_value,  // B,H,L0,S
                    PlainTensor<T>& query_emb,   // B,H,L,S
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    PlainTensor<float>& rotary_emb_cos,
                    PlainTensor<float>& rotary_emb_sin) {
        auto B = cur_query.size(0);
        auto L1 = cur_query.size(1);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        size_t G = 0;
        size_t gH = 0;
        assert(cur_key.m_rank == cur_query.m_rank);
        assert(cur_value.m_rank == cur_query.m_rank);
        if (cur_query.m_rank == 5) {
            // 5D case: multi-query
            G = cur_query.size(2);
            gH = cur_query.size(3);
            assert(cur_key.size(2) == G && cur_key.size(3) == 1);
            assert(cur_value.size(2) == G && cur_value.size(3) == 1);
        }
        auto rotary_dims = rotary_emb_cos.size(3);
        auto half_rotary_dims = rotary_dims / 2;

        if (L0 > 0) {
            PROFILE(prof1, "copyPast");
            auto max_nt = parallel_get_max_threads();
            if (B*H  < max_nt) {
                parallel_for3d(B, H, L0, [&](size_t b, size_t h, size_t l0) {
                    memcpy(&present_value.at({b, h, l0, 0}), &past_value.at({b, h, l0, 0}), sizeof(T) * S);
                    memcpy(&present_key.at({b, h, l0, 0}), &past_key.at({b, h, l0, 0}), sizeof(T) * S);
                });
            } else {
                parallel_for2d(B, H, [&](size_t b, size_t h) {
                    memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(T) * L0 * S);
                    memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(T) * L0 * S);
                });
            }
        }

        PROFILE(prof2, "RoPE_REF");
        // rotary embedding for word vector at each position p
        // meanwhile concat is performed
        parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t p) {
            auto p1 = p + L0;
            auto pos_cos_sin = p1;
            if (gather_pos_idx) {
                pos_cos_sin = gather_pos_idx.at({b, p}, true);
            }
            T* q;
            T* k;
            T* v;
            if (gH > 0) {
                // multi-query: h = G*gH
                size_t g = h / gH;
                size_t hg = h % gH;
                q = &cur_query.at({b, p, g, hg, 0});
                k = &cur_key.at({b, p, g, 0, 0});
                v = &cur_value.at({b, p, g, 0, 0});
            } else {
                q = &cur_query.at({b, p, h, 0});
                k = &cur_key.at({b, p, h, 0});
                v = &cur_value.at({b, p, h, 0});
            }
            auto* present_k = &present_key.at({b, h, p1, 0});    // f32[B, H, L0+L1, 64]
            auto* present_v = &present_value.at({b, h, p1, 0});  // f32[B, H, L0+L1, 64]
            auto* q_embed = &query_emb.at({b, h, p, 0});
            // q_embed = (q * cos) + (rotate_half(q) * sin)
            // k_embed = (k * cos) + (rotate_half(k) * sin)
            auto* cos = &rotary_emb_cos({0, 0, pos_cos_sin, 0});
            auto* sin = &rotary_emb_sin({0, 0, pos_cos_sin, 0});

            size_t s = 0;
            for (; s < half_rotary_dims; s++) {
                q_embed[s] = cos[s] * q[s] + sin[s] * (-q[s + half_rotary_dims]);
                present_k[s] = cos[s] * k[s] + sin[s] * (-k[s + half_rotary_dims]);
                present_v[s] = v[s];
            }
            for (; s < rotary_dims; s++) {
                q_embed[s] = cos[s] * q[s] + sin[s] * (q[s - half_rotary_dims]);
                present_k[s] = cos[s] * k[s] + sin[s] * (k[s - half_rotary_dims]);
                present_v[s] = v[s];
            }
            for (; s < S; s++) {
                q_embed[s] = q[s];
                present_k[s] = k[s];
                present_v[s] = v[s];
            }
        });
    }
};

// GPT-J style RoPE
template <KernelTypes KType, typename T>
struct RoPE2_kernel {
    RoPE2_kernel() = default;

    void operator()(PlainTensor<T>& cur_query,           // B,L,H,S
                    PlainTensor<T>& cur_key,             // B,L,H,S
                    PlainTensor<T>& cur_value,           // B,L,H,S
                    PlainTensor<T>& past_key,            // B,H,L0,S
                    PlainTensor<T>& past_value,          // B,H,L0,S
                    PlainTensor<T>& query_emb,           // B,H,L,S
                    PlainTensor<T>& present_key,         // B, H, L0, S
                    PlainTensor<T>& present_value,       // B, H, L0, S
                    PlainTensor<float>& rotary_emb_cos,  // B, L1, half_rotary_dims
                    PlainTensor<float>& rotary_emb_sin) {
        auto B = cur_query.size(0);
        auto L1 = cur_query.size(1);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        auto half_rotary_dims = rotary_emb_cos.size(2);
        auto rotary_dims = half_rotary_dims * 2;
        // copy past kv into present
        if (L0 > 0) {
            PROFILE(prof, "copyPast");
            auto max_nt = parallel_get_max_threads();
            if (B*H  < max_nt) {
                parallel_for3d(B, H, L0, [&](size_t b, size_t h, size_t l0) {
                    memcpy(&present_value.at({b, h, l0, 0}), &past_value.at({b, h, l0, 0}), sizeof(T) * S);
                    memcpy(&present_key.at({b, h, l0, 0}), &past_key.at({b, h, l0, 0}), sizeof(T) * S);
                });
            } else {
                parallel_for2d(B, H, [&](size_t b, size_t h) {
                    memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(T) * L0 * S);
                    memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(T) * L0 * S);
                });
            }
        }

        PROFILE(prof2, "RoPE2_REF");
        // rotary embedding for word vector at each position p
        // meanwhile concat is performed
        parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t p) {
            auto p1 = p + L0;
            auto* q = &cur_query.at({b, p, h, 0});
            auto* k = &cur_key.at({b, p, h, 0});
            auto* v = &cur_value.at({b, p, h, 0});
            auto* present_k = &present_key.at({b, h, p1, 0});    // f32[B, H, L0+L1, 64]
            auto* present_v = &present_value.at({b, h, p1, 0});  // f32[B, H, L0+L1, 64]
            auto* q_embed = &query_emb.at({b, h, p, 0});
            auto* cos = &rotary_emb_cos({0, p, 0});
            auto* sin = &rotary_emb_sin({0, p, 0});

            size_t s = 0;
            size_t i = 0;

            for (; s < rotary_dims; i++, s += 2) {
                q_embed[s] = cos[i] * q[s] - sin[i] * q[s + 1];
                q_embed[s + 1] = cos[i] * q[s + 1] + sin[i] * q[s];

                present_k[s] = cos[i] * k[s] - sin[i] * k[s + 1];
                present_k[s + 1] = cos[i] * k[s + 1] + sin[i] * k[s];

                present_v[s] = v[s];
                present_v[s + 1] = v[s + 1];
            }

            for (; s < S; s++) {
                q_embed[s] = q[s];
                present_k[s] = k[s];
                present_v[s] = v[s];
            }
        });
    }
};

template <typename RT>
struct KVConcatKernel {
    void operator()(const PlainTensor<RT>& k_input,       // "f32[B, L1, H, S]"
                    const PlainTensor<RT>& v_input,       // "f32[B, L1, H, S]"
                    const PlainTensor<RT>& past_key,      // "f32[B, H, L0, S]"
                    const PlainTensor<RT>& past_value,    // "f32[B, H, L0, S]"
                    const PlainTensor<RT>& present_key,   // "f32[B, H, L0+L1, S]"
                    const PlainTensor<RT>& present_value  // "f32[B, H, L0+L1, S]"
    ) {
        auto B = past_key.size(0);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        auto L1 = k_input.size(1);
        auto past_ks_stride = past_key.stride(3);
        parallel_for3d(2, B, H, [&](size_t kv_id, size_t b, size_t h) {
            if (kv_id == 0) {
                if (past_ks_stride == 1) {
                    if (L0 > 0) {
                        memcpy(&present_key.at({b, h, 0, 0}), &past_key.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
                    }
                    for (size_t p = 0; p < L1; p++) {
                        memcpy(&present_key.at({b, h, L0 + p, 0}), &k_input.at({b, p, h, 0}), sizeof(RT) * S);
                    }
                } else {
                    // special layout for bloom past/present_key, [B, H, S, L0]
                    if (L0 > 0) {
                        for (size_t s = 0; s < S; s++) {
                            memcpy(&present_key.at({b, h, 0, s}), &past_key.at({b, h, 0, s}), sizeof(RT) * L0);
                        }
                    }
                    for (size_t s = 0; s < S; s++) {
                        for (size_t p = 0; p < L1; p++) {
                            present_key.at({b, h, L0 + p, s}) = k_input.at({b, p, h, s});
                        }
                    }
                }
            } else {
                // past_key/value
                if (L0 > 0) {
                    memcpy(&present_value.at({b, h, 0, 0}), &past_value.at({b, h, 0, 0}), sizeof(RT) * L0 * S);
                }
                for (size_t p = 0; p < L1; p++) {
                    memcpy(&present_value.at({b, h, L0 + p, 0}), &v_input.at({b, p, h, 0}), sizeof(RT) * S);
                }
            }
        });
    }
};

#ifdef OV_CPU_WITH_LLMDNN
template <typename DT>
llmdnn::tensor Convert2LLMTensor(const PlainTensor<DT>& src) {
    llmdnn::tensor dst;
    dst.m_capacity = 0;
    dst.m_rank = src.m_rank;
    dst.m_ptr = src.m_ptr;
    memcpy(dst.m_dims, src.m_dims, sizeof(size_t) * src.m_rank);
    dst.m_element_size = sizeof(DT);
    dst.m_dtype = llmdnn::precision_of<DT>::value;
    for (size_t i = 0; i < src.m_rank; i++) {
        dst.m_strides[i] = src.m_strides[i] * sizeof(DT);
    }
    return std::move(dst);
}

// specialization on llmdnn
template <>
struct RoPE_kernel<KT_LLMDNN, ov::bfloat16> {
    RoPE_kernel() = default;

    void operator()(PlainTensor<ov::bfloat16>& cur_query,   // B,L,H,S
                    PlainTensor<ov::bfloat16>& cur_key,     // B,L,H,S
                    PlainTensor<ov::bfloat16>& cur_value,   // B,L,H,S
                    PlainTensor<ov::bfloat16>& past_key,    // f32[B, H, L0, S]
                    PlainTensor<ov::bfloat16>& past_value,  // f32[B, H, L0, S]
                    PlainTensor<ov::bfloat16>& query_emb,
                    PlainTensor<ov::bfloat16>& present_key,
                    PlainTensor<ov::bfloat16>& present_value,
                    PlainTensor<float>& rotary_emb_cos,
                    PlainTensor<float>& rotary_emb_sin) {
        PROFILE(prof2, "RoPE_LLMDNN");
        llmdnn::emb_gpt(Convert2LLMTensor(cur_query),   // B,L,H,S
                        Convert2LLMTensor(cur_key),     // B,L,H,S
                        Convert2LLMTensor(cur_value),   // B,L,H,S
                        Convert2LLMTensor(past_key),    // f32[B, H, L0, S]
                        Convert2LLMTensor(past_value),  // f32[B, H, L0, S]
                        Convert2LLMTensor(query_emb),
                        Convert2LLMTensor(present_key),
                        Convert2LLMTensor(present_value),
                        Convert2LLMTensor(rotary_emb_cos),
                        Convert2LLMTensor(rotary_emb_sin),
                        llmdnn::tensor());
    }
};
#endif

// default implementation: reference
template <KernelTypes KType, typename T>
struct MHA_kernel {
    MHA_kernel() = default;

    template <typename D>
    float dot_product(const D* a, const D* b, int len, int stride_b = 1) {
        float result = 0;
        if (stride_b == 1) {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i]);
        } else {
            for (int i = 0; i < len; i++)
                result += static_cast<float>(a[i]) * static_cast<float>(b[i * stride_b]);
        }
        return result;
    }

    void softmax(float* a, int len) {
        float max = *std::max_element(a, a + len);
        float sum = 0.0f;
        for (int i = 0; i < len; i++) {
            a[i] = exp(a[i] - max);
            sum += a[i];
        }
        float scale = 1.0f / sum;
        for (int i = 0; i < len; i++) {
            a[i] *= scale;
        }
    }

    template <typename D>
    void accumulate(float* acc, const D* v, int len, float weight = 1.0f) {
        for (int i = 0; i < len; i++) {
            acc[i] += static_cast<float>(v[i]) * weight;
        }
    }

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // output_emb    [B, q_len, H*S]
    void operator()(PlainTensor<T>& query,
                    PlainTensor<T>& present_key,
                    PlainTensor<T>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<T>& output_emb,
                    float d_scale = 0.0f) {
        PROFILE(prof, "MHA_REF");
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        auto k_stride_s = present_key.stride(3);
        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        parallel_for2d(B, H, [&](size_t b, size_t h) {
            std::vector<float> attn_score(kv_len);
            std::vector<float> word_vec(head_size, 0.0f);

            // auto key = &present_key.at({b, h, 0, 0});
            // auto value = &present_value.at({b, h, 0, 0});
            // auto output = &output_emb.at({b, 0, h * head_size});
            for (size_t m = 0; m < q_len; m++) {
                // dot-product to get attention scores
                auto* q = &query.at({b, h, m, 0});
                // how many key/values can be accessed causally
                auto ncausal = kv_len;  // kv_len - q_len + m + 1;
                // no causall mask is set and it's not fused into attention_mask
                if (auto_causal)
                    ncausal = kv_len - q_len + m + 1;
                for (size_t n = 0; n < ncausal; n++) {
                    auto* k = &present_key.at({b, h, n, 0});
                    attn_score[n] = dot_product(q, k, head_size, k_stride_s) * d_scale;

                    // apply alibi tensor
                    if (alibi_mask)
                        attn_score[n] += alibi_mask.at({b, h, m, n}, true);

                    // apply attention mask (maybe combined with causal_mask)
                    if (attention_mask)
                        attn_score[n] += attention_mask.at({b, h, m, n}, true);

                    // apply causal_mask
                    if (causal_mask) {
                        bool is_zero = causal_mask.at({b, h, m, n}, true) == 0;
                        if (select_nfltmax_at_0) {
                            if (is_zero)
                                attn_score[n] = -FLT_MAX;
                        } else {
                            if (!is_zero) {
                                attn_score[n] = -FLT_MAX;
                            }
                        }
                    }
                }

                // softmax
                softmax(&attn_score[0], ncausal);

                // linearly combine value
                word_vec.assign(head_size, 0.0f);
                for (size_t n = 0; n < ncausal; n++) {
                    auto* v = &present_value.at({b, h, n, 0});
                    accumulate(word_vec.data(), v, head_size, attn_score[n]);
                }

                // output [B, L1, H*head_size]
                auto* out = &output_emb.at({b, m, h * head_size});
                std::copy(word_vec.begin(), word_vec.end(), out);
            }
        });
    }
};

#ifdef OV_CPU_WITH_MLAS
template <>
struct MHA_kernel<KT_MLAS, float> {
    size_t m_block_size;
    // buffer to hold qk temp
    std::vector<PlainTensor<float>> qk_buffers;

    MHA_kernel() {
        m_block_size = std::getenv("MBLK") ? atoi(std::getenv("MBLK")) : 4;
        qk_buffers.resize(parallel_get_max_threads(), PlainTensor<float>(true));
    }

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor<float>& query,
                    PlainTensor<float>& present_key,
                    PlainTensor<float>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,
                    PlainTensor<float>& output_emb,
                    float d_scale = 0.0f) {
        PROFILE(prof, "MHA_MLAS");
        auto B = query.size(0);
        auto H = query.size(1);
        auto q_len = query.size(2);
        auto head_size = query.size(3);
        auto kv_len = present_key.size(2);

        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);
        auto k_stride_s = present_key.stride(3);

        auto m_blocks = (q_len + m_block_size - 1) / m_block_size;
        bool auto_causal = attention_mask.size(2) == 1 && !causal_mask;

        if (false) {
            std::cout << "===============" << std::endl;
            std::cout << "alibi_mask = " << alibi_mask << std::endl;
            std::cout << "attention_mask = " << attention_mask << std::endl;
            std::cout << "causal_mask = " << causal_mask << std::endl;
            std::cout << "select_nfltmax_at_0 = " << select_nfltmax_at_0 << std::endl;
            std::cout << "k_stride_s = " << k_stride_s << std::endl;
            std::cout << "present_key = " << present_key << std::endl;
        }

        parallel_for3d(B, H, m_blocks, [&](size_t b, size_t h, size_t m_blk) {
            size_t thread_id = static_cast<size_t>(parallel_get_thread_num());
            auto& qk_buf = qk_buffers[thread_id];

            auto m_start = m_blk * m_block_size;
            auto m_end = std::min(m_start + m_block_size, q_len);
            auto m_cnt = m_end - m_start;

            auto kv_len_cache_align = (((kv_len * sizeof(float)) + 63) / 64 * 64) / sizeof(float);
            qk_buf.resize({m_block_size, kv_len_cache_align});
            const float* q_ptr = &query.at({b, h, m_start, 0});
            const float* k_ptr = &present_key.at({b, h, 0, 0});
            const float* v_ptr = &present_value.at({b, h, 0, 0});

            float* alibi_ptr = nullptr;
            auto alibi_stride = 0;
            if (alibi_mask) {
                alibi_ptr = &alibi_mask.at({b, h, 0, 0}, true);
                if (alibi_mask.size(2) > 1)
                    alibi_stride = alibi_mask.stride(2);
            }
            float* attn_mask_ptr = nullptr;
            auto attn_mask_stride = 0;
            if (attention_mask) {
                attn_mask_ptr = &attention_mask.at({b, h, 0, 0}, true);
                if (attention_mask.size(2) > 1)
                    attn_mask_stride = attention_mask.stride(2);
            }
            uint8_t* cmask_ptr = nullptr;
            auto cmask_stride = 0;
            if (causal_mask) {
                cmask_ptr = &causal_mask.at({b, h, 0, 0}, true);
                if (causal_mask.size(2) > 1)
                    cmask_stride = causal_mask.stride(2);
            }

            float* qk = &(qk_buf.at({0, 0}));
            auto qk_m_stride = qk_buf.stride(0);

            if (k_stride_s == 1)
                mlas_sgemm("N",
                           "T",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(2),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);
            else
                mlas_sgemm("N",
                           "N",
                           m_cnt,
                           kv_len,
                           head_size,
                           1.0f,
                           q_ptr,
                           query.stride(2),
                           k_ptr,
                           present_key.stride(3),
                           0.f,
                           qk,
                           qk_m_stride,
                           1);

            for (size_t m = m_start; m < m_end; m++) {
                // apply attention mask
                // sofmax
                auto ncausal = auto_causal ? (kv_len - q_len + m + 1): kv_len;
                InferenceEngine::Extensions::Cpu::XARCH::scale_add_softmax(qk + (m - m_start) * qk_m_stride,
                                                                           d_scale,
                                                                           alibi_ptr + m * alibi_stride,
                                                                           attn_mask_ptr + m * attn_mask_stride,
                                                                           cmask_ptr + m * cmask_stride,
                                                                           select_nfltmax_at_0,
                                                                           ncausal,
                                                                           kv_len);
            }
            mlas_sgemm("N",
                       "N",
                       m_cnt,
                       head_size,
                       kv_len,
                       1.0f,
                       qk,
                       qk_m_stride,
                       v_ptr,
                       present_value.stride(2),
                       0.f,
                       &output_emb.at({b, m_start, h * head_size}),
                       output_emb.stride(1),
                       1);
            // output [B, L1, H*S]
            // for (size_t m = 0; m < q_len; m++) {
            //    const float* src = dst + m * head_size;
            //    std::copy(src, src + head_size, &output_emb.at({b, m, h * head_size}));
            //}
        });
    }
};
#endif

#ifdef OV_CPU_WITH_LLMDNN
template <>
struct MHA_kernel<KT_LLMDNN, ov::bfloat16> {
    llmdnn::mha_gpt m_kernel;
    bool m_kernel_initialized = false;
    MHA_kernel() {}

    PlainTensor<uint8_t> causal_mask;
    bool select_nfltmax_at_0;  // set attn_score to -FLT_MAX when causal_mask[...] equal to this
    void set_causal_mask(PlainTensor<uint8_t> mask, bool _select_nfltmax_at_0) {
        causal_mask = mask;
        select_nfltmax_at_0 = _select_nfltmax_at_0;
    }

    void operator()(PlainTensor<ov::bfloat16>& query,
                    PlainTensor<ov::bfloat16>& present_key,
                    PlainTensor<ov::bfloat16>& present_value,
                    const PlainTensor<float>& alibi_mask,
                    const PlainTensor<float>& attention_mask,  // [batch, 1, query_seq_len, key_seq_len]
                    PlainTensor<ov::bfloat16>& attn_output,
                    float d_scale = 0.0f) {
        PROFILE(prof, "MHA_LLMDNN");
        auto head_size = query.size(3);
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(head_size);

        // llmdnn::tensor alibi;
        bool use_causal_mask = (!causal_mask) && (attention_mask.size(2) == 1);
        m_kernel.exec(Convert2LLMTensor(query),
                      Convert2LLMTensor(present_key),
                      Convert2LLMTensor(present_value),
                      Convert2LLMTensor(attn_output),
                      Convert2LLMTensor(attention_mask),
                      Convert2LLMTensor(alibi_mask),
                      Convert2LLMTensor(causal_mask),
                      select_nfltmax_at_0,
                      d_scale,
                      use_causal_mask);
    }
};
#endif

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
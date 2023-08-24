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
#include "utils/plain_tensor.hpp"
#include "vnode_kernels.hpp"
#include "vnode_rms_norm.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

struct vnode_executor {
    std::string signature;
    std::vector<PlainTensorBase*> inputs;
    std::vector<PlainTensorBase*> outputs;
    std::map<PlainTensorBase*, int> out_inplace;

    vnode_executor() {}

    virtual InferenceEngine::Precision get_precision() = 0;

    void register_inputs() {}
    template <typename T0, typename... Ts>
    void register_inputs(PlainTensor<T0>& in0, PlainTensor<Ts>&... ins) {
        inputs.push_back(&in0);
        register_inputs(ins...);
    }

    void register_outputs() {}
    template <typename T0, typename... Ts>
    void register_outputs(PlainTensor<T0>& out0, PlainTensor<Ts>&... outs) {
        outputs.push_back(&out0);
        register_outputs(outs...);
    }

    int output_inplace_to(PlainTensorBase* out) {
        if (out_inplace.count(out)) {
            return out_inplace[out];
        }
        return -1;
    }
    void output_inplace_to(PlainTensorBase& out, int to_input) {
        out_inplace[&out] = to_input;
    }

    void update_inputs(Node* node) {
        int ii = 0;
        for (auto& inp : inputs) {
            inp->reset(node->getParentEdgeAt(ii++)->getMemoryPtr());
        }
    }

    void update_outputs(Node* node) {
        int oo = 0;
        for (auto& outp : outputs) {
            outp->reset(node->getChildEdgeAt(oo++)->getMemoryPtr());
        }
    }

    virtual void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) = 0;
};

#define EXECUTOR_SIGNATURE(vtype_name)                                                               \
    static constexpr char* vtype = vtype_name;                                                       \
    static constexpr char* impl_type = ktype_name_of<KType>::value;                                  \
    static constexpr InferenceEngine::Precision::ePrecision prec = precision_of<RT>::value;          \
    static inline std::string get_signature() {                                                      \
        return std::string(vtype) + "," + impl_type + "," + InferenceEngine::Precision(prec).name(); \
    }                                                                                                \
    InferenceEngine::Precision get_precision() override {                                            \
        return prec;                                                                                 \
    }

template <KernelTypes KType, typename RT>
struct gptneox_attention_executor : public vnode_executor {
    EXECUTOR_SIGNATURE("gptneox_attention");

    PlainTensor<RT> qkv_input;                      // f32[B, L1, H*3*S] => [B, L1, H, 3, S]
    PlainTensor<RT> past_key;                       // f32[B, H, L0, S]
    PlainTensor<RT> past_value;                     // f32[B, H, L0, S]
    PlainTensor<float> attention_mask;              // f32[B, 1, 1, L0 + L1]
    PlainTensor<float> rotary_emb_cos;              // f32[1,1,2048,16]
    PlainTensor<float> rotary_emb_sin;              // f32[1,1,2048,16]
    PlainTensor<int32_t> layer0_repeat_Tile;        // i32[?,1,?,24]
    PlainTensor<int32_t> ListConstruct_205_Concat;  // "i32[4]"

    PlainTensor<RT> output_emb;     // f32[B, L1, H*S]
    PlainTensor<RT> present_key;    // f32[B, H, L0+L1, S]
    PlainTensor<RT> present_value;  // f32[B, H, L0+L1, S]

    MHA_kernel<KType, RT> kernel;
    RoPE_kernel<KType, RT> rope_kernel;

    gptneox_attention_executor() : m_query_emb(true) {
        register_inputs(qkv_input,
                        past_key,
                        past_value,
                        attention_mask,
                        rotary_emb_cos,
                        rotary_emb_sin,
                        layer0_repeat_Tile,
                        ListConstruct_205_Concat);
        register_outputs(output_emb, present_key, present_value);
    }

    PlainTensor<RT> m_query_emb;  // query with embed

    size_t B;
    size_t H;
    size_t L0;
    size_t L1;
    size_t S;
    size_t rotary_dims;
    size_t max_position_embeddings;

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        B = past_key.size(0);
        H = past_key.size(1);   // 8
        L0 = past_key.size(2);  // number of tokens to be encoded
        S = past_key.size(3);   // 64
        L1 = qkv_input.size(1);
        rotary_dims = rotary_emb_cos.size(3);
        max_position_embeddings = rotary_emb_cos.size(2);

        {
            PROFILE(prof, "redefineOutputMemory");
            node->redefineOutputMemory({{B, L1, H * S}, {B, H, L0 + L1, S}, {B, H, L0 + L1, S}});
            update_outputs(node);
        }

        attention_mask.assert_dims({B, 1, 1, L0 + L1});
        past_key.assert_dims({B, H, L0, S});
        past_value.assert_dims({B, H, L0, S});
        qkv_input.assert_dims({B, L1, H * 3 * S});
        rotary_emb_cos.assert_dims({1, 1, max_position_embeddings, rotary_dims});
        rotary_emb_sin.assert_dims({1, 1, max_position_embeddings, rotary_dims});

        m_query_emb.resize({B, H, L1, S});

        // PlainTensor<RT> qkv_input;          // f32[B, L1, H*3*S] => [B, L1, H, 3, S]
        auto qkv_4d = qkv_input.reshape({B, L1, H, 3 * S});
        auto q_input = qkv_4d.slice(3, 0, S);
        auto k_input = qkv_4d.slice(3, S, 2 * S);
        auto v_input = qkv_4d.slice(3, 2 * S, 3 * S);
        rope_kernel(q_input,
                    k_input,
                    v_input,
                    past_key,
                    past_value,
                    m_query_emb,
                    present_key,
                    present_value,
                    rotary_emb_cos,
                    rotary_emb_sin);

        kernel(m_query_emb, present_key, present_value, {}, attention_mask, output_emb);
    }
};

template <KernelTypes KType, typename RT>
struct gptj_attention_executor : public vnode_executor {
    EXECUTOR_SIGNATURE("gptj_attention");
    PlainTensor<RT> q_input;                // f32[B, L1, H*S]
    PlainTensor<RT> k_input;                // f32[B, L1, H*S]
    PlainTensor<RT> v_input;                // f32[B, L1, H*S]
    PlainTensor<RT> past_key;               // f32[B, H, L0, S]
    PlainTensor<RT> past_value;             // f32[B, H, L0, S]
    PlainTensor<float> attention_mask;      // f32[B, 1, 1, L0 + L1]
    PlainTensor<float> rotary_emb_sin_cos;  // f32[?,?,64]

    PlainTensor<RT> output_emb;     // f32[B, L1, H*S]
    PlainTensor<RT> present_key;    // f32[B, H, L0+L1, S]
    PlainTensor<RT> present_value;  // f32[B, H, L0+L1, S]

    MHA_kernel<KType, RT> kernel;
    RoPE2_kernel<KType, RT> rope_kernel;

    gptj_attention_executor() : m_query_emb(true) {
        register_inputs(q_input, k_input, v_input, past_key, past_value, attention_mask, rotary_emb_sin_cos);
        register_outputs(output_emb, present_key, present_value);
    }

    PlainTensor<RT> m_query_emb;  // query with embed

    size_t B;
    size_t H;
    size_t L0;
    size_t L1;
    size_t S;
    size_t rotary_dims;
    size_t max_position_embeddings;

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        B = past_key.size(0);
        H = past_key.size(1);
        L0 = past_key.size(2);
        S = past_key.size(3);
        L1 = q_input.size(1);
        rotary_dims = rotary_emb_sin_cos.size(2);
        // DEBUG_LOG_TEMP(" B=", B, " H=", H, " S=", S, " L0=", L0, " L1=", L1);
        {
            PROFILE(prof, "redefineOutputMemory");
            node->redefineOutputMemory({{B, L1, H * S}, {B, H, L0 + L1, S}, {B, H, L0 + L1, S}});
            update_outputs(node);
        }

        attention_mask.assert_dims({B, 1, 1, L0 + L1});
        past_key.assert_dims({B, H, L0, S});
        past_value.assert_dims({B, H, L0, S});
        q_input.assert_dims({B, L1, H * S});
        k_input.assert_dims({B, L1, H * S});
        v_input.assert_dims({B, L1, H * S});
        rotary_emb_sin_cos.assert_dims({1, L1, rotary_dims});
        present_key.assert_dims({B, H, L0 + L1, S});
        present_value.assert_dims({B, H, L0 + L1, S});
        m_query_emb.resize({B, H, L1, S});

        auto rope_q = q_input.reshape({B, L1, H, S});
        auto rope_k = k_input.reshape({B, L1, H, S});
        auto rope_v = v_input.reshape({B, L1, H, S});
        auto sin_tab = rotary_emb_sin_cos.slice(2, 0, rotary_dims / 2);
        auto cos_tab = rotary_emb_sin_cos.slice(2, rotary_dims / 2, rotary_dims);

        rope_kernel(rope_q,
                    rope_k,
                    rope_v,
                    past_key,
                    past_value,
                    m_query_emb,
                    present_key,
                    present_value,
                    cos_tab,
                    sin_tab);

        kernel(m_query_emb, present_key, present_value, {}, attention_mask, output_emb);
    }
};

template <KernelTypes KType, typename RT>
struct falcon_attention_executor : public vnode_executor {
    EXECUTOR_SIGNATURE("falcon_attention");
    PlainTensor<RT> qkv_proj;             // f32[B, L1, G*(gH+2)*S=(H + 2G)*S]
    PlainTensor<RT> past_key;             // f32[B, G*gH=H, L0, S]
    PlainTensor<RT> past_value;           // f32[B, G*gH=H, L0, S]
    PlainTensor<int32_t> past_kv_shape;   // i32[4]
    PlainTensor<float> attn_causal_mask;  // f32[B, 1, L1, L0 + L1]
    PlainTensor<float> cos_tab;           // f32[1,2048,64]
    PlainTensor<float> sin_tab;           // f32[1,2048,64]

    PlainTensor<RT> output_emb;     // f32[B, L1, H*S]
    PlainTensor<RT> present_key;    // f32[B*H, L0+L1, S]
    PlainTensor<RT> present_value;  // f32[B*H, L0+L1, S]

    MHA_kernel<KType, RT> kernel;
    RoPE_kernel<KType, RT> rope_kernel;

    falcon_attention_executor() : m_query_emb(true) {
        register_inputs(qkv_proj, past_key, past_value, past_kv_shape, attn_causal_mask, cos_tab, sin_tab);
        register_outputs(output_emb, present_key, present_value);
    }

    PlainTensor<RT> m_query_emb;  // query with embed

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        auto B = past_key.size(0);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        auto L1 = qkv_proj.size(1);
        auto G = (qkv_proj.size(2) / S - H) / 2;  // group of broadcastable KV
        auto gH = H / G;
        auto max_position_embeddings = cos_tab.size(1);
        auto rotary_dims = cos_tab.size(2);

        {
            PROFILE(prof, "redefineOutputMemory");
            node->redefineOutputMemory({{B, L1, H * S}, {B * H, L0 + L1, S}, {B * H, L0 + L1, S}});
            update_outputs(node);
        }

        attn_causal_mask.assert_dims({B, 1, L1, L0 + L1});
        past_key.assert_dims({B, H, L0, S});
        past_value.assert_dims({B, H, L0, S});
        qkv_proj.assert_dims({B, L1, G * (gH + 2) * S});
        cos_tab.assert_dims({1, max_position_embeddings, rotary_dims});
        sin_tab.assert_dims({1, max_position_embeddings, rotary_dims});
        present_key.assert_dims({B * H, L0 + L1, S});
        present_value.assert_dims({B * H, L0 + L1, S});

        m_query_emb.resize({B, H, L1, S});

        auto qkv5d = qkv_proj.reshape({B, L1, G, (gH + 2), S});
        auto q_5d = qkv5d.slice(3, 0, gH);
        auto k_5d = qkv5d.slice(3, gH, gH + 1);
        auto v_5d = qkv5d.slice(3, gH + 1, gH + 2);

        // falcon pattern can only cover 3D present kv, but kernels require 4D
        present_key = present_key.reshape({B, H, L0 + L1, S});
        present_value = present_value.reshape({B, H, L0 + L1, S});

        auto cos_4d = cos_tab.reshape({1, 1, max_position_embeddings, rotary_dims});
        auto sin_4d = sin_tab.reshape({1, 1, max_position_embeddings, rotary_dims});
        rope_kernel(q_5d, k_5d, v_5d, past_key, past_value, m_query_emb, present_key, present_value, cos_4d, sin_4d);

        kernel(m_query_emb, present_key, present_value, {}, attn_causal_mask, output_emb);
    }
};

template <KernelTypes KType, typename RT>
struct llama2_attention_executor : public vnode_executor {
    EXECUTOR_SIGNATURE("llama2_attention");
    PlainTensor<RT> q_input;              // f32[B, L1, H*S]
    PlainTensor<RT> k_input;              // f32[B, L1, H*S]
    PlainTensor<RT> v_input;              // f32[B, L1, H*S]
    PlainTensor<RT> past_key;             // f32[B, H, L0, S]
    PlainTensor<RT> past_value;           // f32[B, H, L0, S]
    PlainTensor<float> attn_causal_mask;  // f32[B, 1, L1, L0 + L1]
    PlainTensor<float> rotary_emb_cos;    // f32[1,1,4096,128]
    PlainTensor<float> rotary_emb_sin;    // f32[1,1,4096,128]
    PlainTensor<int32_t> gather_pos_idx;  // i32[B, L1]

    PlainTensor<RT> output_emb;     // f32[B, L1, H*S]
    PlainTensor<RT> present_key;    // f32[B, H, L0+L1, S]
    PlainTensor<RT> present_value;  // f32[B, H, L0+L1, S]

    MHA_kernel<KType, RT> kernel;
    RoPE_kernel<KT_REF, RT> rope_kernel; // gather_pos_idx only implemented in REF so far

    llama2_attention_executor() : m_query_emb(true) {
        register_inputs(q_input,
                        k_input,
                        v_input,
                        past_key,
                        past_value,
                        attn_causal_mask,
                        rotary_emb_cos,
                        rotary_emb_sin,
                        gather_pos_idx);
        register_outputs(output_emb, present_key, present_value);
    }

    PlainTensor<RT> m_query_emb;  // query with embed

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);

        auto B = past_key.size(0);
        auto H = past_key.size(1);
        auto L0 = past_key.size(2);
        auto S = past_key.size(3);
        auto L1 = q_input.size(1);
        auto rotary_dims = rotary_emb_cos.size(3);
        auto max_position_embeddings = rotary_emb_cos.size(2);

        {
            PROFILE(prof, "redefineOutputMemory");
            node->redefineOutputMemory({{B, L1, H * S}, {B, H, L0 + L1, S}, {B, H, L0 + L1, S}});
            update_outputs(node);
        }

        attn_causal_mask.assert_dims({B, 1, L1, L0 + L1});
        past_key.assert_dims({B, H, L0, S});
        past_value.assert_dims({B, H, L0, S});
        q_input.assert_dims({B, L1, H * S});
        k_input.assert_dims({B, L1, H * S});
        v_input.assert_dims({B, L1, H * S});

        rotary_emb_cos.assert_dims({1, 1, max_position_embeddings, rotary_dims});
        rotary_emb_sin.assert_dims({1, 1, max_position_embeddings, rotary_dims});
        // gather_pos_idx.assert_dims({B, L1});

        present_key.assert_dims({B, H, L0 + L1, S});
        present_value.assert_dims({B, H, L0 + L1, S});

        m_query_emb.resize({B, H, L1, S});

        // set gather_pos_idx
        auto rope_q = q_input.reshape({B, L1, H, S});
        auto rope_k = k_input.reshape({B, L1, H, S});
        auto rope_v = v_input.reshape({B, L1, H, S});
        rope_kernel.gather_pos_idx = gather_pos_idx;
        rope_kernel(rope_q,
                    rope_k,
                    rope_v,
                    past_key,
                    past_value,
                    m_query_emb,
                    present_key,
                    present_value,
                    rotary_emb_cos,
                    rotary_emb_sin);

        kernel(m_query_emb, present_key, present_value, {}, attn_causal_mask, output_emb);
    }
};
template <KernelTypes KType, typename RT>
struct llama_RMSNorm_executor : public vnode_executor {
    EXECUTOR_SIGNATURE("llama_RMSNorm");
    PlainTensor<RT> input;   // f32[?, ?, 4096]
    PlainTensor<RT> weight;  // f32[1, 1, 4096]
    PlainTensor<RT> eps;     // f32[1, 1, 1]

    PlainTensor<RT> output;  // f32[?, ?, 4096]
    llama_RMSNorm_executor() {
        register_inputs(input, weight, eps);
        register_outputs(output);
    }

    void exec(Node* node, dnnl::stream strm, std::map<std::string, double>& symbol2value) override {
        update_inputs(node);
        auto data_shape = input.shape();
        auto last_dim = input.size(-1);
        auto batch_size = shape_size(data_shape) / last_dim;
        {
            PROFILE(prof, "redefineOutputMemory");
            node->redefineOutputMemory({data_shape});
            update_outputs(node);
        }

        input = input.reshape({batch_size, last_dim});
        weight = weight.reshape({last_dim});
        output = output.reshape({batch_size, last_dim});
        auto esp_value = *eps.data();

        parallel_for(batch_size, [&](size_t b) {
            auto* psrc = &input.at({b, 0});
            auto* pwei = &weight.at({0});
            auto* pdst = &output.at({b, 0});
            InferenceEngine::Extensions::Cpu::XARCH::rms_norm(pdst, psrc, esp_value, pwei, last_dim);
        });
    }
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
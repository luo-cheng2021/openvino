// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "gpt_neox_attn.h"
#include <ngraph/opsets/opset10.hpp>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include <cpu/x64/jit_generator.hpp>
#include "emitters/jit_dnnl_emitters.hpp"
#include "emitters/jit_load_store_emitters.hpp"

using namespace InferenceEngine;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;
using namespace dnnl::impl::cpu::x64;
using namespace Xbyak;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

template <cpu_isa_t isa>
struct jit_rotary_kernel : public jit_uni_rotary_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_rotary_kernel)

    explicit jit_rotary_kernel(const jit_rotary_compile_params& jcp) : jit_uni_rotary_kernel(jcp), jit_generator(jit_name()) {
        vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    }
    virtual ~jit_rotary_kernel() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41, Xmm, isa == cpu_isa_t::avx2, Ymm, Zmm>::type;

    void generate() override {
        this->preamble();

#define GET_OFF(field) offsetof(jit_rotary_call_args, field)
        mov(reg_q_src, ptr[reg_params + GET_OFF(q_src)]);
        mov(reg_k_src, ptr[reg_params + GET_OFF(k_src)]);
        mov(reg_cos, ptr[reg_params + GET_OFF(cos)]);
        mov(reg_sin, ptr[reg_params + GET_OFF(sin)]);
        mov(reg_q_dst, ptr[reg_params + GET_OFF(q_dst)]);
        mov(reg_k_dst, ptr[reg_params + GET_OFF(k_dst)]);

        uni_vpxor(vmm_q_src, vmm_q_src, vmm_q_src);
        uni_vpxor(vmm_k_src, vmm_k_src, vmm_k_src);
        uni_vpxor(vmm_cos, vmm_cos, vmm_cos);
        uni_vpxor(vmm_sin, vmm_sin, vmm_sin);

        auto half_rotary_ndims = jcp_.rotary_ndims / 2;
        for (size_t k = 0; k < jcp_.head_num; k++) {
            mov(reg_q_src_aux, reg_q_src);
            mov(reg_k_src_aux, reg_k_src);
            mov(reg_q_dst_aux, reg_q_dst);
            mov(reg_k_dst_aux, reg_k_dst);
            mov(reg_cos_aux, reg_cos);
            mov(reg_sin_aux, reg_sin);
            for (size_t i = 0; i < half_rotary_ndims / vec_size; i++) {
                rotary_first_half(vec_size);
            }
            if (half_rotary_ndims % vec_size != 0) {
                rotary_first_half(half_rotary_ndims % vec_size);
            }
            for (size_t i = 0; i < half_rotary_ndims / vec_size; i++) {
                rotary_last_half(vec_size);
            }
            if (half_rotary_ndims % vec_size != 0) {
                rotary_last_half(half_rotary_ndims % vec_size);
            }
            add(reg_q_src, jcp_.size_per_head * 3 * jcp_.src_prc.size());
            add(reg_k_src, jcp_.size_per_head * 3 * jcp_.src_prc.size());
            add(reg_q_dst, jcp_.q_seq_len * jcp_.size_per_head * jcp_.src_prc.size());
            add(reg_k_dst, jcp_.max_seq_len * jcp_.size_per_head * jcp_.src_prc.size());
        }

        this->postamble();

        for (const auto& emitter : emitters) {
            if (emitter.second)
                emitter.second->emit_data();
        }
    }

    void rotary_first_half(size_t step) {
        //bool is_tail = step < vec_size;

        // q_src[i + halfRotaryNdims]
        mov(reg_tmp, reg_q_src_aux);
        add(reg_tmp, jcp_.rotary_ndims / 2 * jcp_.src_prc.size());
        load(vmm_q_src, reg_tmp, jcp_.src_prc, step, false);
        // k_src[i + halfRotaryNdims]
        mov(reg_tmp, reg_k_src_aux);
        add(reg_tmp, jcp_.rotary_ndims / 2 * jcp_.src_prc.size());
        load(vmm_k_src, reg_tmp, jcp_.src_prc, step, false);
        load(vmm_cos, reg_cos_aux, Precision::FP32, step, false);
        load(vmm_sin, reg_sin_aux, Precision::FP32, step, false);
        // q_src[i + halfRotaryNdims] * sin[i]
        uni_vmulps(vmm_q_dst, vmm_q_src, vmm_sin);
        // k_src[i + halfRotaryNdims] * sin[i]
        uni_vmulps(vmm_k_dst, vmm_k_src, vmm_sin);

        load(vmm_q_src, reg_q_src_aux, jcp_.src_prc, step, false);
        load(vmm_k_src, reg_k_src_aux, jcp_.src_prc, step, false);
        // TODO: sse4
        // q_src[i] * cos[i] - q_src[i + halfRotaryNdims] * sin[i]
        vfmsub231ps(vmm_q_dst, vmm_q_src, vmm_cos);
        // k_src[i] * cos[i] - k_src[i + halfRotaryNdims] * sin[i]
        vfmsub231ps(vmm_k_dst, vmm_k_src, vmm_cos);

        store(reg_q_dst_aux, vmm_q_dst, jcp_.src_prc, step);
        store(reg_k_dst_aux, vmm_k_dst, jcp_.src_prc, step);

        add(reg_q_src_aux, jcp_.src_prc.size() * step);
        add(reg_k_src_aux, jcp_.src_prc.size() * step);
        add(reg_q_dst_aux, jcp_.src_prc.size() * step);
        add(reg_k_dst_aux, jcp_.src_prc.size() * step);
        add(reg_cos_aux, sizeof(float) * step);
        add(reg_sin_aux, sizeof(float) * step);
    }
    void rotary_last_half(size_t step) {
        bool is_tail = step < vec_size;

        // q_src[i - halfRotaryNdims]
        mov(reg_tmp, reg_q_src_aux);
        sub(reg_tmp, jcp_.rotary_ndims / 2 * jcp_.src_prc.size());
        load(vmm_q_src, reg_tmp, jcp_.src_prc, step, false);
        // k_src[i - halfRotaryNdims]
        mov(reg_tmp, reg_k_src_aux);
        sub(reg_tmp, jcp_.rotary_ndims / 2 * jcp_.src_prc.size());
        load(vmm_k_src, reg_tmp, jcp_.src_prc, step, false);
        load(vmm_cos, reg_cos_aux, Precision::FP32, step, false);
        load(vmm_sin, reg_sin_aux, Precision::FP32, step, false);
        // q_src[i - halfRotaryNdims] * sin[i]
        uni_vmulps(vmm_q_dst, vmm_q_src, vmm_sin);
        // k_src[i - halfRotaryNdims] * sin[i]
        uni_vmulps(vmm_k_dst, vmm_k_src, vmm_sin);

        load(vmm_q_src, reg_q_src_aux, jcp_.src_prc, step, false);
        load(vmm_k_src, reg_k_src_aux, jcp_.src_prc, step, false);
        // q_src[i] * cos[i] + q_src[i - halfRotaryNdims] * sin[i]
        vfmadd231ps(vmm_q_dst, vmm_q_src, vmm_cos);
        // k_src[i] * cos[i] + k_src[i - halfRotaryNdims] * sin[i]
        vfmadd231ps(vmm_k_dst, vmm_k_src, vmm_cos);

        store(reg_q_dst_aux, vmm_q_dst, jcp_.src_prc, step);
        store(reg_k_dst_aux, vmm_k_dst, jcp_.src_prc, step);

        if (!is_tail) {
            add(reg_q_src_aux, jcp_.src_prc.size() * step);
            add(reg_k_src_aux, jcp_.src_prc.size() * step);
            add(reg_q_dst_aux, jcp_.src_prc.size() * step);
            add(reg_k_dst_aux, jcp_.src_prc.size() * step);
            add(reg_cos_aux, sizeof(float) * step);
            add(reg_sin_aux, sizeof(float) * step);
        }
    }
#undef GET_OFF

    inline void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, Precision src_prc, const int& elt_num, bool fill) {
        const auto seed = load_emitter_params(src_prc, Precision::FP32, elt_num, fill, "float_min").hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_load_emitter(this, isa, src_prc, Precision::FP32, elt_num, Precision::FP32, fill, "float_min"));
        }

        emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), 0}, {static_cast<size_t>(vmm_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }
    inline void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, Precision dst_prc, const int& elt_num) {
        const auto seed = store_emitter_params(Precision::FP32, dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_store_emitter(this, isa, Precision::FP32, dst_prc, elt_num));
        }

        emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx()), 0}, {static_cast<size_t>(reg_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }

    size_t vec_size;

    Vmm vmm_q_src = Vmm(0);
    Vmm vmm_k_src = Vmm(1);
    Vmm vmm_cos = Vmm(2);
    Vmm vmm_sin = Vmm(3);
    Vmm vmm_q_dst = Vmm(4);
    Vmm vmm_k_dst = Vmm(5);

    Reg64 reg_q_src = r8;
    Reg64 reg_k_src = r9;
    Reg64 reg_cos = r10;
    Reg64 reg_sin = r11;
    Reg64 reg_q_dst = r12;
    Reg64 reg_k_dst = r13;
    Reg64 reg_q_src_aux = r14;
    Reg64 reg_k_src_aux = r15;
    Reg64 reg_q_dst_aux = rax;
    Reg64 reg_k_dst_aux = rbx;
    Reg64 reg_cos_aux = rsi;
    Reg64 reg_sin_aux = rbp;
    Reg64 reg_tmp = rdx;

    Reg64 reg_params = abi_param1;
    Reg64 reg_not_params = abi_not_param1;

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
    const std::vector<size_t> pool_aux_gpr_idxs = { static_cast<size_t>(reg_params.getIdx()), static_cast<size_t>(reg_not_params.getIdx()) };
    const std::vector<size_t> pool_aux_vmm_idxs = { 6 };
};

class GlobalContext {
public:
    static GlobalContext& getInstance() {
        static GlobalContext instance;
        return instance;
    }

    std::vector<uint8_t>& getOrCreateStore(const std::string& key, size_t new_size) {
        auto& store = simpleStore[key];
        if (store.size() < new_size) {
            std::vector<uint8_t> new_store(new_size);
            memcpy(new_store.data(), store.data(), store.size());
            store = std::move(new_store);
        }
        return store;
    }

private:
    std::unordered_map<std::string, std::vector<uint8_t>> simpleStore;
};

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
    maxPositionEmbeddings = attn_op->m_max_position_embeddings;
    rotaryEmbBase = attn_op->m_rotary_emb_base;
    rotaryPct = attn_op->m_rotary_pct;
    maxSeqLen = attn_op->m_max_seq_len;
    normalFactor = 1.0f / sqrtf(static_cast<float>(sizePerHead));

    rotaryNdims = static_cast<int>(sizePerHead * rotaryPct);
}

void GPTNeoxAttn::initSupportedPrimitiveDescriptors() {
    dataPrecision = getOriginalInputPrecisionAtPort(IN_QKV);
    dataTypeSize = dataPrecision.size();

    addSupportedPrimDesc({{LayoutType::ncsp, dataPrecision},
                          {LayoutType::ncsp, Precision::I32}},
                         {{LayoutType::ncsp, dataPrecision}},
                          impl_desc_type::ref_any);
}

void GPTNeoxAttn::createPrimitive() {
    Node::createPrimitive();
}

void GPTNeoxAttn::prepareParams() {
    const auto& qkv_dims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    const auto batch = qkv_dims[0];
    const auto seq_len = qkv_dims[1];

    // init rotary embeddings
    initRotery(maxPositionEmbeddings);
    // attention_mask shape: [batch, seq_len/maxSeqLen], real length = seq_len + past_key[-2]
    attnMasks.resize(batch * maxSeqLen, 0.0f);
    // memory for query transpose destination
    if (queryTranspose.size() < batch * seq_len * hiddenSize * dataTypeSize) {
        queryTranspose.resize(batch * seq_len * hiddenSize * dataTypeSize);
    }
    // memory size for key/value past buffers, shape: [2, batch, headNum, maxSeqLen, sizePerHead]
    pastKVBufferSize = 2 * batch * headNum * maxSeqLen * sizePerHead * dataTypeSize;

    {
        jit_rotary_compile_params jcp;
        jcp.src_prc = dataPrecision;
        jcp.head_num = headNum;
        jcp.rotary_ndims = rotaryNdims;
        jcp.hidden_size = hiddenSize;
        jcp.q_seq_len = seq_len;
        jcp.max_seq_len = maxSeqLen;
        jcp.size_per_head = sizePerHead;
        jcp.src_stride = hiddenSize * 3 * dataTypeSize;
        jcp.q_dst_stride = seq_len * sizePerHead * dataTypeSize;
        // key will directly write to past_keys
        jcp.k_dst_stride = maxSeqLen * sizePerHead * dataTypeSize;
        if (mayiuse(cpu_isa_t::avx512_core)) {
            rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx512_core>(jcp));
        } else if (mayiuse(cpu_isa_t::avx2)) {
            rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::avx2>(jcp));
        } else if (mayiuse(cpu_isa_t::sse41)) {
            // TODO: vfmad not in sse
            assert(false);
            rotaryKernel.reset(new jit_rotary_kernel<cpu_isa_t::sse41>(jcp));
        } else {
            THROW_ERROR << "cannot create jit rotary kernel";
        }
        rotaryKernel->create_ker();
    }
}

void GPTNeoxAttn::initRotery(size_t max_seq_len) {
    std::vector<float> inv_freq;
    for (size_t i = 0; i < rotaryNdims; i += 2) {
        inv_freq.push_back(1.0f / (powf(rotaryEmbBase, static_cast<float>(i) / rotaryNdims)));
    }
    std::vector<float> t;
    for (size_t i = 0; i < max_seq_len * 2; i++) {
        t.push_back(static_cast<float>(i));
    }
    auto width = rotaryNdims / 2 * 2;
    auto height = max_seq_len * 2;
    cosCached.resize(height * width);
    sinCached.resize(height * width);
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width / 2; j++) {
            cosCached[i * width + j] = cosf(t[i] * inv_freq[j]);
            cosCached[i * width + j + width / 2] = cosf(t[i] * inv_freq[j]);
            sinCached[i * width + j] = sinf(t[i] * inv_freq[j]);
            sinCached[i * width + j + width / 2] = sinf(t[i] * inv_freq[j]);
        }
    }
}

void GPTNeoxAttn::reinitAttentionMask(size_t batch, size_t max_seq_len) {
    std::vector<float> new_attn_masks;
    new_attn_masks.resize(batch * max_seq_len * 2, 0.0f);
    memcpy(&new_attn_masks[0], &attnMasks[0], attnMasks.size() * sizeof(float));
    attnMasks = std::move(new_attn_masks);
}

// template <typename in1_type>
// void GPTNeoxAttn::applyRotaryPosEmbImpl(uint8_t* q_src, uint8_t* k_src, uint8_t* q_dst, uint8_t* k_dst, float* cos_cached,
//     float* sin_cached, size_t batch, size_t q_seq_len, size_t offset) {
//     auto halfRotaryNdims = rotaryNdims / 2;
//     for (size_t m = 0; m < batch; m ++) {
//         float* cos = cos_cached + offset * rotaryNdims;
//         float* sin = sin_cached + offset * rotaryNdims;
//         auto q_dst_batch = q_dst + m * headNum * q_seq_len * sizePerHead * dataTypeSize;
//         auto k_dst_batch = k_dst + m * headNum * maxSeqLen * sizePerHead * dataTypeSize;
//         for (size_t n = 0; n < q_seq_len; n++) {
//             auto q_dst_seq = q_dst_batch + n * sizePerHead * dataTypeSize;
//             auto k_dst_seq = k_dst_batch + n * sizePerHead * dataTypeSize;
//             for (size_t k = 0; k < headNum; k++) {
//                 for (size_t i = 0; i < halfRotaryNdims; i++) {
//                     q_dst_seq[i] = q_src[i] * cos[i] - q_src[i + halfRotaryNdims] * sin[i];
//                     k_dst_seq[i] = k_src[i] * cos[i] - k_src[i + halfRotaryNdims] * sin[i];
//                 }
//                 for (size_t i = halfRotaryNdims; i < rotaryNdims; i++) {
//                     q_dst_seq[i] = q_src[i] * cos[i] + q_src[i - halfRotaryNdims] * sin[i];
//                     k_dst_seq[i] = k_src[i] * cos[i] + k_src[i - halfRotaryNdims] * sin[i];
//                 }
//                 q_src += sizePerHead * 3 * dataTypeSize;
//                 k_src += sizePerHead * 3 * dataTypeSize;
//                 q_dst_seq += q_seq_len * sizePerHead * dataTypeSize;
//                 k_dst_seq += maxSeqLen * sizePerHead * dataTypeSize;
//             }
//             cos += rotaryNdims;
//             sin += rotaryNdims;
//         }
//     }
// }

// q_src, k_src: [batch, seq_len, num_heads, 3 * head_size]
// q_dst: [batch, num_heads, query_seq_len, head_size]
// k_dst: [batch, num_heads, maxSeqLen, head_size]
void GPTNeoxAttn::applyRotaryPosEmb(uint8_t* q_src, uint8_t* k_src, uint8_t* q_dst, uint8_t* k_dst,
                                    float* cos_cached, float* sin_cached, size_t batch, size_t q_seq_len, size_t offset) {
    jit_rotary_call_args call_args;
    for (size_t m = 0; m < batch; m ++) {
        float* cos = cos_cached + offset * rotaryNdims;
        float* sin = sin_cached + offset * rotaryNdims;
        auto q_dst_batch = q_dst + m * headNum * q_seq_len * sizePerHead * dataTypeSize;
        auto k_dst_batch = k_dst + m * headNum * maxSeqLen * sizePerHead * dataTypeSize;
        for (size_t n = 0; n < q_seq_len; n++) {
            auto q_dst_seq = q_dst_batch + n * sizePerHead * dataTypeSize;
            auto k_dst_seq = k_dst_batch + n * sizePerHead * dataTypeSize;
            call_args.q_src = q_src;
            call_args.k_src = k_src;
            call_args.cos = cos;
            call_args.sin = sin;
            call_args.q_dst = q_dst_seq;
            call_args.k_dst = k_dst_seq;
            (*rotaryKernel)(&call_args);
            q_src += hiddenSize * 3 * dataTypeSize;
            k_src += hiddenSize * 3 * dataTypeSize;
            cos += rotaryNdims;
            sin += rotaryNdims;
        }
    }
}

void GPTNeoxAttn::executeDynamicImpl(dnnl::stream strm) {
    auto qkv_data_dims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    qkv_data_dims.back() = qkv_data_dims.back() / 3;
    redefineOutputMemory({qkv_data_dims});

    execute(strm);
}

// typical use:
// 1, read from kv input and write to pask_keys, which needed by attention
// 2, read from q input and write to q temp buffer
// src: [batch, seq_len, num_attention_heads, 3, head_size]
// dst: [batch, num_attention_heads, max_seq_len/seq_len, head_size]
static void MemcpyStride(void* dst, void* src, size_t copy_head_size, size_t head_size, size_t head_num, size_t seq_len,
    size_t max_seq_len, size_t batch, size_t type_size) {
    for (size_t m = 0; m < batch; m++) {
        auto* dst_batch = static_cast<uint8_t*>(dst) + m * head_num * max_seq_len * head_size * type_size;
        auto* src_batch = static_cast<uint8_t*>(src) + m * head_num * seq_len * 3 * head_size * type_size;
        for (size_t n = 0; n < head_num; n++) {
            auto* dst_head = dst_batch + n * max_seq_len * head_size * type_size;
            auto* src_head = src_batch + n * 3 * head_size * type_size;
            for (size_t i = 0; i < seq_len; i++) {
                memcpy(dst_head, src_head, copy_head_size * type_size);
                dst_head += head_size * type_size;
                src_head += head_num * 3 * head_size * type_size;
            }
        }
    }
}

void GPTNeoxAttn::execute(dnnl::stream strm) {
    // [batch, seq_len, (num_heads * 3 * head_size)]
    auto* qkv = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_QKV)->getMemoryPtr()->GetPtr());
    const int* past_keys_num = reinterpret_cast<const int*>(getParentEdgeAt(IN_PAST_KEYS_NUM)->getMemoryPtr()->GetPtr());
    auto* dst_data = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto& qkv_dims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    const auto batch = qkv_dims[0];
    const auto seq_len = qkv_dims[1];
    // lower 16 bit means the number of past keys, higher 16 bit means the model id
    const auto new_seq_offset = static_cast<size_t>(past_keys_num[0]) & 0xffff;
    assert(new_seq_offset < maxSeqLen);
    // usage: each 1x300 sub model and 1x1 sub model will share the same model id
    const auto model_id = static_cast<size_t>(past_keys_num[0]) >> 16;
    // [2, batch, num_heads, maxSeqLen, head_size]
    auto* past_keys = GlobalContext::getInstance().getOrCreateStore(getName() + std::to_string(model_id), pastKVBufferSize).data();

    // the sentence is longer than maxSeqLen
    if (seq_len + new_seq_offset > cosCached.size()) {
        initRotery(seq_len + new_seq_offset);
        reinitAttentionMask(batch, seq_len + new_seq_offset);
    }

    // [batch, seq_len, (num_heads * 3 * head_size)]
    //   --> [batch, seq_len, num_heads, 3 * head_size]
    auto query = qkv;                                      // qkv[..., : self.head_size].permute(0, 2, 1, 3)
    auto key = qkv + sizePerHead * dataTypeSize;           // qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    auto value = qkv + 2 * sizePerHead * dataTypeSize;     // qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)
    auto new_past_key_ptr = past_keys + dataTypeSize * new_seq_offset * sizePerHead;
    auto new_past_value_ptr = new_past_key_ptr + pastKVBufferSize / 2;
    // first token will write to pastKeys offset 0
    bool first_token = new_seq_offset == 0;
    // transpose + rotary embbeding:
    // transpose: [batch, seq_len, num_attention_heads, 3 * head_size] -->
    //          3 [batch, num_attention_heads, seq_len, head_size]
    // rotary embbeding: part of key will write to past_key, part of query will write to tempory buffer
    applyRotaryPosEmb(query, key, queryTranspose.data(), new_past_key_ptr, &cosCached[0], &sinCached[0], batch, seq_len, new_seq_offset);
    // query pass part(temp buffer): query = torch.cat((query, query_pass), dim=-1)
    MemcpyStride(queryTranspose.data() + rotaryNdims * dataTypeSize, query + rotaryNdims * dataTypeSize, sizePerHead - rotaryNdims, sizePerHead, headNum,
        seq_len, seq_len, batch, dataTypeSize);
    // key pass part(past_key): key = torch.cat((key, key_pass), dim=-1)
    MemcpyStride(new_past_key_ptr + rotaryNdims * dataTypeSize, key + rotaryNdims * dataTypeSize, sizePerHead - rotaryNdims, sizePerHead, headNum, seq_len,
        maxSeqLen, batch, dataTypeSize);
    // value(pastKeys): value = torch.cat((past_value, value), dim=-2)
    MemcpyStride(new_past_value_ptr, value, sizePerHead, sizePerHead, headNum, seq_len, maxSeqLen, batch, dataTypeSize);
    // attn_output = _attn(query, key, value)
    // attn_output = _merge_heads(attn_output, self.num_attention_heads, self.head_size)
    auto& mha = mhaGPTs[(static_cast<size_t>(batch) << 32) + static_cast<size_t>(new_seq_offset + seq_len)];
    if (!mha) {
        gpt::MHAGPT::CreateParam param = {
            batch, headNum, seq_len, sizePerHead,
            new_seq_offset + seq_len, normalFactor, dataPrecision, first_token, new_seq_offset + 1
        };
        mha = std::make_shared<gpt::MHAGPT>();
        mha->create(param);
        getSelectedPrimitiveDescriptor()->setImplementationType(mha->get_impl_type());
    }
    auto head_stride_in_q = sizePerHead * seq_len;
    auto batch_stride_in_q = head_stride_in_q * headNum;
    auto head_stride_in_kv = sizePerHead * maxSeqLen;
    auto batch_stride_in_kv = head_stride_in_kv * headNum;
    // q: [batch, num_heads, query_seq_len, head_size]
    // k: [batch, num_heads, key_seq_len, head_size]
    // v: [batch, num_heads, value_seq_len, head_size]
    // attention_mask: [batch, 1, 1, key_seq_len]
    // attn_output: [batch, query_seq_len, num_heads * head_size]
    gpt::MHAGPT::ExecParam param = {
        queryTranspose.data(), past_keys, past_keys + pastKVBufferSize / 2,
        &attnMasks[0],
        dst_data,
        head_stride_in_q, batch_stride_in_q,    // q stride
        head_stride_in_kv, batch_stride_in_kv,  // kv stride
        maxSeqLen,                              // attn_mask stride
        sizePerHead, hiddenSize * seq_len,      // output stride
    };

    mha->exec(param);
}

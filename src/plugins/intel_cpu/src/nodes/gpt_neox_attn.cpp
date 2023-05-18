// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>

#include "eltwise.h"
#include "fake_quantize.h"
#include "gpt_neox_attn.h"
#include <ngraph/opsets/opset10.hpp>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include <cpu/x64/jit_generator.hpp>
#include "emitters/jit_dnnl_emitters.hpp"
#include "emitters/jit_load_store_emitters.hpp"
#include "ie_parallel.hpp"
#include "special/quant_i8_custom.hpp"
#include "special/gemm_custom.hpp"

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
        if (jcp_.dst_prc == Precision::I8) {
            mov(reg_tmp, ptr[reg_params + GET_OFF(q_quant)]);
            uni_vmovss(Xmm(vmm_q_scale.getIdx()), ptr[reg_tmp]);
            uni_vbroadcastss(vmm_q_scale, Xmm(vmm_q_scale.getIdx()));
            mov(reg_tmp, ptr[reg_params + GET_OFF(k_quant)]);
            uni_vmovss(Xmm(vmm_k_scale.getIdx()), ptr[reg_tmp]);
            uni_vbroadcastss(vmm_k_scale, Xmm(vmm_k_scale.getIdx()));
        }

        auto half_rotary_ndims = jcp_.rotary_ndims / 2;
        mov(reg_q_dst_aux, reg_q_dst);
        mov(reg_cos_aux, reg_cos);
        mov(reg_sin_aux, reg_sin);
        size_t steps = 0;
        for (size_t i = 0; i < half_rotary_ndims / vec_size; i++) {
            rotary_first_half(vec_size);
            steps += vec_size;
        }
        if (half_rotary_ndims % vec_size != 0) {
            rotary_first_half(half_rotary_ndims % vec_size);
            steps += half_rotary_ndims % vec_size;
        }
        for (size_t i = 0; i < half_rotary_ndims / vec_size; i++) {
            rotary_last_half(vec_size);
            steps += vec_size;
        }
        if (half_rotary_ndims % vec_size != 0) {
            rotary_last_half(half_rotary_ndims % vec_size);
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
        mov(reg_tmp, reg_q_src);
        add(reg_tmp, jcp_.rotary_ndims / 2 * jcp_.src_prc.size());
        load(vmm_q_src, reg_tmp, jcp_.src_prc, step, false);
        // k_src[i + halfRotaryNdims]
        mov(reg_tmp, reg_k_src);
        add(reg_tmp, jcp_.rotary_ndims / 2 * jcp_.src_prc.size());
        load(vmm_k_src, reg_tmp, jcp_.src_prc, step, false);
        load(vmm_cos, reg_cos_aux, Precision::FP32, step, false);
        load(vmm_sin, reg_sin_aux, Precision::FP32, step, false);
        // q_src[i + halfRotaryNdims] * sin[i]
        uni_vmulps(vmm_q_dst, vmm_q_src, vmm_sin);
        // k_src[i + halfRotaryNdims] * sin[i]
        uni_vmulps(vmm_k_dst, vmm_k_src, vmm_sin);

        load(vmm_q_src, reg_q_src, jcp_.src_prc, step, false);
        load(vmm_k_src, reg_k_src, jcp_.src_prc, step, false);
        // TODO: sse4
        // q_src[i] * cos[i] - q_src[i + halfRotaryNdims] * sin[i]
        vfmsub231ps(vmm_q_dst, vmm_q_src, vmm_cos);
        // k_src[i] * cos[i] - k_src[i + halfRotaryNdims] * sin[i]
        vfmsub231ps(vmm_k_dst, vmm_k_src, vmm_cos);
        if (jcp_.dst_prc == Precision::I8) {
            uni_vmulps(vmm_q_dst, vmm_q_dst, vmm_q_scale);
            uni_vmulps(vmm_k_dst, vmm_k_dst, vmm_k_scale);
        }

        store(reg_q_dst_aux, vmm_q_dst, jcp_.dst_prc, step);
        store(reg_k_dst, vmm_k_dst, jcp_.dst_prc, step);

        add(reg_q_src, jcp_.src_prc.size() * step);
        add(reg_k_src, jcp_.src_prc.size() * step);
        add(reg_q_dst_aux, jcp_.dst_prc.size() * step);
        add(reg_k_dst, jcp_.dst_prc.size() * step);
        add(reg_cos_aux, sizeof(float) * step);
        add(reg_sin_aux, sizeof(float) * step);
    }
    void rotary_last_half(size_t step) {
        bool is_tail = step < vec_size;

        // q_src[i - halfRotaryNdims]
        mov(reg_tmp, reg_q_src);
        sub(reg_tmp, jcp_.rotary_ndims / 2 * jcp_.src_prc.size());
        load(vmm_q_src, reg_tmp, jcp_.src_prc, step, false);
        // k_src[i - halfRotaryNdims]
        mov(reg_tmp, reg_k_src);
        sub(reg_tmp, jcp_.rotary_ndims / 2 * jcp_.src_prc.size());
        load(vmm_k_src, reg_tmp, jcp_.src_prc, step, false);
        load(vmm_cos, reg_cos_aux, Precision::FP32, step, false);
        load(vmm_sin, reg_sin_aux, Precision::FP32, step, false);
        // q_src[i - halfRotaryNdims] * sin[i]
        uni_vmulps(vmm_q_dst, vmm_q_src, vmm_sin);
        // k_src[i - halfRotaryNdims] * sin[i]
        uni_vmulps(vmm_k_dst, vmm_k_src, vmm_sin);

        load(vmm_q_src, reg_q_src, jcp_.src_prc, step, false);
        load(vmm_k_src, reg_k_src, jcp_.src_prc, step, false);
        // q_src[i] * cos[i] + q_src[i - halfRotaryNdims] * sin[i]
        vfmadd231ps(vmm_q_dst, vmm_q_src, vmm_cos);
        // k_src[i] * cos[i] + k_src[i - halfRotaryNdims] * sin[i]
        vfmadd231ps(vmm_k_dst, vmm_k_src, vmm_cos);
        if (jcp_.dst_prc == Precision::I8) {
            uni_vmulps(vmm_q_dst, vmm_q_dst, vmm_q_scale);
            uni_vmulps(vmm_k_dst, vmm_k_dst, vmm_k_scale);
        }

        store(reg_q_dst_aux, vmm_q_dst, jcp_.dst_prc, step);
        store(reg_k_dst, vmm_k_dst, jcp_.dst_prc, step);

        if (!is_tail) {
            add(reg_q_src, jcp_.src_prc.size() * step);
            add(reg_k_src, jcp_.src_prc.size() * step);
            add(reg_q_dst_aux, jcp_.dst_prc.size() * step);
            add(reg_k_dst, jcp_.dst_prc.size() * step);
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
    Vmm vmm_q_scale = Vmm(7);
    Vmm vmm_k_scale = Vmm(8);

    Reg64 reg_q_src = r8;
    Reg64 reg_k_src = r9;
    Reg64 reg_cos = r10;
    Reg64 reg_sin = r11;
    Reg64 reg_q_dst = r12;
    Reg64 reg_k_dst = r13;
    Reg64 reg_q_dst_stride = r14;
    Reg64 reg_q_dst_aux = rax;
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
    using buffer_t = std::shared_ptr<uint8_t>;
    using beam_buffers_t = std::vector<buffer_t>;

    struct PastKVStore {
        // real memory buffer
        beam_buffers_t key_buffer;
        beam_buffers_t value_buffer;
        std::vector<buffer_t*> current_k_bufs;
        std::vector<buffer_t*> current_v_bufs;
    };
    static GlobalContext& getInstance() {
        static GlobalContext instance;
        return instance;
    }
    void init(size_t head_num, size_t size_per_head, size_t size_per_head_aligned, size_t max_seq_len, size_t data_type_len) {
        headNum = head_num;
        sizePerHead = size_per_head;
        sizePerHeadAligned = size_per_head_aligned;
        maxSeqLen = max_seq_len;
        dataTypeLen = data_type_len;
    }
    void getOrCreateStore(const std::string& key, size_t new_size_per_key_per_beam, const int* beam_idx, size_t beam_idx_num,
        sv::small_vector<uint8_t*, 4>& current_k_bufs, sv::small_vector<uint8_t*, 4>& current_v_bufs, size_t valid_histroy_seq_len) {
        // expected buffer: [2, beam_num/batch, headNum, maxSeqLen, sizePerHead]
        auto& store = simpleKVStore[key];
        current_k_bufs.resize(beam_idx_num);
        current_v_bufs.resize(beam_idx_num);
        // new_size_per_key_per_beam = headNum * maxSeqLen * sizePerHead * inputDataTypeSize
        // not init
        if (store.key_buffer.size() < beam_idx_num) {
            store.key_buffer.resize(beam_idx_num);
            store.value_buffer.resize(beam_idx_num);
            store.current_k_bufs.resize(beam_idx_num);
            store.current_v_bufs.resize(beam_idx_num);
            for (auto i = 0; i < beam_idx_num; i++) {
                store.key_buffer[i] = std::shared_ptr<uint8_t>(
                            reinterpret_cast<uint8_t*>(aligned_alloc(64, new_size_per_key_per_beam)),
                            [](void * p) { ::free(p); });
                memset(store.key_buffer[i].get(), 0, new_size_per_key_per_beam);
                store.value_buffer[i] = std::shared_ptr<uint8_t>(
                            reinterpret_cast<uint8_t*>(aligned_alloc(64, new_size_per_key_per_beam)),
                            [](void * p) { ::free(p); });
                memset(store.value_buffer[i].get(), 0, new_size_per_key_per_beam);
                store.current_k_bufs[i] = &store.key_buffer[i];
                store.current_v_bufs[i] = &store.value_buffer[i];
                current_k_bufs[i] = store.current_k_bufs[i]->get();
                current_v_bufs[i] = store.current_v_bufs[i]->get();
            }
            sizePerKeyPerBeam = new_size_per_key_per_beam;
            return;
        }
        assert(store.key_buffer.size() == beam_idx_num);
        // max seq becomes larger
        if (sizePerKeyPerBeam < new_size_per_key_per_beam) {
            auto tmpSizePerKeyPerBeam = new_size_per_key_per_beam * 2;
            for (auto i = 0; i < beam_idx_num; i++) {
                auto new_k_store = std::shared_ptr<uint8_t>(
                            reinterpret_cast<uint8_t*>(aligned_alloc(64, tmpSizePerKeyPerBeam)),
                            [](void * p) { ::free(p); });
                memset(new_k_store.get(), 0, tmpSizePerKeyPerBeam);
                memcpy(new_k_store.get(), store.key_buffer[i].get(), sizePerKeyPerBeam);
                store.key_buffer[i] = std::move(new_k_store);
                auto new_v_store = std::shared_ptr<uint8_t>(
                            reinterpret_cast<uint8_t*>(aligned_alloc(64, tmpSizePerKeyPerBeam)),
                            [](void * p) { ::free(p); });
                memset(new_v_store.get(), 0, tmpSizePerKeyPerBeam);
                memcpy(new_v_store.get(), store.value_buffer[i].get(), sizePerKeyPerBeam);
                store.value_buffer[i] = std::move(new_v_store);
                store.current_k_bufs[i] = &store.key_buffer[i];
                store.current_v_bufs[i] = &store.value_buffer[i];
            }
            sizePerKeyPerBeam = tmpSizePerKeyPerBeam;
        }
        for (auto i = 0; i < beam_idx_num; i++) {
            current_k_bufs[i] = store.current_k_bufs[i]->get();
            current_v_bufs[i] = store.current_v_bufs[i]->get();
        }
        // for each 2x300 case, ignore the reorder
        if (beam_idx == nullptr)
            return;

        // beam_idx contains new index which is the index of current_k_bufs
        // special case: beam_idx contains sequence like 0, 1, 2, 3... which means no reorder
        bool need_reorder = false;
        for (size_t i = 0; i < beam_idx_num; i++) {
            if (i != static_cast<size_t>(beam_idx[i])) {
                need_reorder = true;
                break;
            }
        }
        if (!need_reorder)
            return;

        sv::small_vector<buffer_t*, 4> ptrs_k(beam_idx_num, nullptr), ptrs_v(beam_idx_num, nullptr);
        // not used buffers pointer(items should be small numbers, use vector to decrease memory alloction times)
        sv::small_vector<buffer_t*, 4> no_use_ptrs_k(store.current_k_bufs.size());
        std::memcpy(no_use_ptrs_k.data(), store.current_k_bufs.data(), store.current_k_bufs.size() * sizeof(buffer_t*));
        sv::small_vector<std::pair<size_t, size_t>, 4> copy_pairs;
        // first pass: no shared items, shared items first occurence
        for (auto i = 0; i < beam_idx_num; i++) {
            auto wanted_idx = beam_idx[i];
            if (no_use_ptrs_k[wanted_idx]) {
                ptrs_k[i] = store.current_k_bufs[wanted_idx];
                ptrs_v[i] = store.current_v_bufs[wanted_idx];
                no_use_ptrs_k[wanted_idx] = nullptr;
            }
        }
        // second pass: shared items
        for (auto i = 0; i < beam_idx_num; i++) {
            if (ptrs_k[i] == nullptr) {
                auto wanted_idx = beam_idx[i];
                for (size_t j = 0; j < no_use_ptrs_k.size(); j++) {
                    if (no_use_ptrs_k[j]) {
                        copy_pairs.push_back({wanted_idx, j});
                        ptrs_k[i] = no_use_ptrs_k[j];
                        ptrs_v[i] = store.current_v_bufs[j];
                        no_use_ptrs_k[j] = nullptr;
                        break;
                    }
                }
            }

            current_k_bufs[i] = ptrs_k[i]->get();
            current_v_bufs[i] = ptrs_v[i]->get();
        }
        // third pass: copy, only first layer does the copy
        if (!copy_pairs.empty() && key.find("layers.0") != std::string::npos) {
            int layers = simpleKVStore.size();
            int works = layers * headNum * valid_histroy_seq_len;
            // same size memory will cost different time on different cores(Test on SPR HBM). use load balance one
            tbb::parallel_for(0, works, [&](int cur_work) {
                int layer_idx;
                int h;
                int s;

                parallel_it_init(cur_work, layer_idx, layers, h, headNum, s, valid_histroy_seq_len);

                auto it = simpleKVStore.begin();
                for (size_t i = 0; i < layer_idx; i++) ++it;
                auto& layer = (*it).second;
                for (auto& item: copy_pairs) {
                    auto* src = layer.current_k_bufs[item.first]->get();
                    auto* dst = layer.current_k_bufs[item.second]->get();
                    auto sub_src = src + (h * maxSeqLen + s) * sizePerHeadAligned * dataTypeLen;
                    auto sub_dst = dst + (h * maxSeqLen + s) * sizePerHeadAligned * dataTypeLen;
                    memcpy(sub_dst, sub_src, sizePerHead * dataTypeLen);
                    src = layer.current_v_bufs[item.first]->get();
                    dst = layer.current_v_bufs[item.second]->get();
                    sub_src = src + (h * maxSeqLen + s) * sizePerHeadAligned * dataTypeLen;
                    sub_dst = dst + (h * maxSeqLen + s) * sizePerHeadAligned * dataTypeLen;
                    memcpy(sub_dst, sub_src, sizePerHead * dataTypeLen);
                }
            });
        }

        for (auto i = 0; i < beam_idx_num; i++) {
            store.current_k_bufs[i] = ptrs_k[i];
            store.current_v_bufs[i] = ptrs_v[i];
        }
    }

private:
    std::unordered_map<std::string, PastKVStore> simpleKVStore;
    size_t headNum;
    size_t sizePerHead;
    size_t sizePerHeadAligned;
    size_t maxSeqLen;
    size_t dataTypeLen;
    size_t sizePerKeyPerBeam;
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

    q_quant = attn_op->m_q_quant;
    k_quant = attn_op->m_k_quant;
    qk_quant = attn_op->m_qk_quant;
    v_quant = attn_op->m_v_quant;
    useInt8 = q_quant != 0.0f;
    if (useInt8) {
        sizePerHeadAligned = rnd_up(sizePerHead, 64);
    } else {
        sizePerHeadAligned = rnd_up(sizePerHead, 32);
    }
}

void GPTNeoxAttn::extractQuantParam() {
    for (int i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];
        bool isLastPostOp = (i == (fusedWith.size() - 1));

        if (auto* eltwiseNode = dynamic_cast<Eltwise*>(node.get())) {
            continue;
        }

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            qkv_quant = fakeQuantizeNode->getInputScale();
            continue;
        }
    }
}

void GPTNeoxAttn::initSupportedPrimitiveDescriptors() {
    inputDataType = getOriginalInputPrecisionAtPort(IN_QKV);
    inputDataTypeSize = inputDataType.size();
    outputDataType = inputDataType;

    extractQuantParam();

    if (useInt8) {
        mhaInputDataType = Precision::I8;
        mhaInputDataTypeSize = sizeof(int8_t);
    } else {
        mhaInputDataType = inputDataType;
        mhaInputDataTypeSize = inputDataTypeSize;
    }

    if (!fusedWith.empty()) {
        outputDataType = fusedWith[fusedWith.size() - 1]->getOriginalOutputPrecisionAtPort(0);
        if (outputDataType == Precision::I8) {
            assert(!qkv_quant.empty());
        }
    }

    addSupportedPrimDesc({{LayoutType::ncsp, inputDataType},
                          {LayoutType::ncsp, Precision::I32},
                          {LayoutType::ncsp, Precision::I32},
                          {LayoutType::ncsp, Precision::I32}},
                         {{LayoutType::ncsp, outputDataType}},
                          impl_desc_type::ref_any);
    GlobalContext::getInstance().init(headNum, sizePerHead, sizePerHeadAligned, maxSeqLen, mhaInputDataTypeSize);
}

void GPTNeoxAttn::createPrimitive() {
    Node::createPrimitive();
}

// test only
int g_seq_offset = -1;
int g_beam_idx[] = {0, 1, 2, 3, 4, 5, 6, 7};
void GPTNeoxAttn::prepareParams() {
    const auto& qkv_dims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    const auto batch = qkv_dims[0];

    // init rotary embeddings
    if (!cosCached)
        initRotery(maxPositionEmbeddings);
    // attention_mask shape: [batch, seq_len/maxSeqLen], real length = seq_len + past_key[-2]
    if (!attnMasks) {
        auto capacity = batch * maxSeqLen * sizeof(float);
        attnMasks = std::shared_ptr<float>(
                            reinterpret_cast<float*>(aligned_alloc(64, capacity)),
                            [](void * p) { ::free(p); });
        memset(attnMasks.get(), 0, capacity);
    }
    // memory for query transpose destination
    if (!queryTranspose) {
        auto capacity = batch * maxSeqLen * (headNum * sizePerHeadAligned) * mhaInputDataTypeSize;
        queryTranspose = std::shared_ptr<uint8_t>(
                            reinterpret_cast<uint8_t*>(aligned_alloc(64, capacity)),
                            [](void * p) { ::free(p); });
        memset(queryTranspose.get(), 0, capacity);
    }

    if (!rotaryKernel) {
        jit_rotary_compile_params jcp;
        jcp.src_prc = inputDataType;
        jcp.dst_prc = mhaInputDataType;
        jcp.head_num = headNum;
        jcp.rotary_ndims = rotaryNdims;
        jcp.hidden_size = hiddenSize;
        jcp.max_seq_len = maxSeqLen;
        jcp.size_per_head = sizePerHead;
        jcp.size_per_head_aligned = sizePerHeadAligned;
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
    if (!mhaGPT) {
        gpt::MHAGPT::CreateParam param = {
            headNum, sizePerHead, sizePerHeadAligned,
            normalFactor, mhaInputDataType, outputDataType, maxSeqLen, qkv_quant.size() == 1
        };
        mhaGPT = std::move(std::unique_ptr<gpt::MHAGPT>(new gpt::MHAGPT()));
        mhaGPT->create(param);
        getSelectedPrimitiveDescriptor()->setImplementationType(mhaGPT->get_impl_type());
    }

    auto env = std::getenv("USE_OFFSET");
    if (env) {
        g_seq_offset = std::stoi(env);
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
    auto capacity = height * width * sizeof(float);
    cosCached = std::shared_ptr<float>(
                        reinterpret_cast<float*>(aligned_alloc(64, capacity)),
                        [](void * p) { ::free(p); });
    sinCached = std::shared_ptr<float>(
                        reinterpret_cast<float*>(aligned_alloc(64, capacity)),
                        [](void * p) { ::free(p); });

    auto* cos_p = cosCached.get();
    auto* sin_p = sinCached.get();
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width / 2; j++) {
            cos_p[i * width + j] = cosf(t[i] * inv_freq[j]);
            cos_p[i * width + j + width / 2] = cosf(t[i] * inv_freq[j]);
            sin_p[i * width + j] = sinf(t[i] * inv_freq[j]);
            sin_p[i * width + j + width / 2] = sinf(t[i] * inv_freq[j]);
        }
    }
}

void GPTNeoxAttn::reinitAttentionMask(size_t batch, size_t max_seq_len) {
    // std::vector<float> new_attn_masks;
    // new_attn_masks.resize(batch * max_seq_len * 2, 0.0f);
    // memcpy(&new_attn_masks[0], &attnMasks[0], attnMasks.size() * sizeof(float));
    // attnMasks = std::move(new_attn_masks);
}

// void GPTNeoxAttn::applyRotaryPosEmb(uint8_t* q_src, uint8_t* k_src, uint8_t* q_dst, const std::vector<uint8_t*>& k_dst, size_t k_start,
//                                     float* cos_cached, float* sin_cached, size_t batch, size_t q_seq_len, size_t offset) {
//     auto halfRotaryNdims = rotaryNdims / 2;
//     for (size_t m = 0; m < batch; m ++) {
//         float* cos = cos_cached + offset * rotaryNdims;
//         float* sin = sin_cached + offset * rotaryNdims;
//         auto q_dst_batch = q_dst + m * headNum * q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
//         auto k_dst_batch = k_dst[m] + k_start;
//         for (size_t n = 0; n < q_seq_len; n++) {
//             auto q_dst_seq = q_dst_batch + n * sizePerHeadAligned * mhaInputDataTypeSize;
//             auto k_dst_seq = k_dst_batch + n * sizePerHeadAligned * mhaInputDataTypeSize;
//             for (size_t k = 0; k < headNum; k++) {
//                 for (size_t i = 0; i < halfRotaryNdims; i++) {
//                     q_dst_seq[i] = q_src[i] * cos[i] - q_src[i + halfRotaryNdims] * sin[i];
//                     k_dst_seq[i] = k_src[i] * cos[i] - k_src[i + halfRotaryNdims] * sin[i];
//                 }
//                 for (size_t i = halfRotaryNdims; i < rotaryNdims; i++) {
//                     q_dst_seq[i] = q_src[i] * cos[i] + q_src[i - halfRotaryNdims] * sin[i];
//                     k_dst_seq[i] = k_src[i] * cos[i] + k_src[i - halfRotaryNdims] * sin[i];
//                 }
//                 q_src += sizePerHead * 3 * inputDataTypeSize;
//                 k_src += sizePerHead * 3 * inputDataTypeSize;
//                 q_dst_seq += q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
//                 k_dst_seq += maxSeqLen * sizePerHeadAligned * mhaInputDataTypeSize;
//             }
//             cos += rotaryNdims;
//             sin += rotaryNdims;
//         }
//     }
// }

// q_src, k_src: [batch, seq_len, num_heads, 3 * head_size]
// q_dst: [batch, num_heads, query_seq_len, head_size]
// k_dst: [batch, num_heads, maxSeqLen, head_size]
void GPTNeoxAttn::applyRotaryPosEmb(uint8_t* q_src, uint8_t* k_src, uint8_t* q_dst, const std::vector<uint8_t*>& k_dst, size_t k_start,
                                    float* cos_cached, float* sin_cached, size_t batch, size_t q_seq_len, size_t offset) {
    // jit_rotary_call_args call_args;
    // call_args.q_quant = &q_quant;
    // call_args.k_quant = &k_quant;
    // for (size_t m = 0; m < batch; m ++) {
    //     float* cos = cos_cached + offset * rotaryNdims;
    //     float* sin = sin_cached + offset * rotaryNdims;
    //     auto q_dst_batch = q_dst + m * headNum * q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
    //     auto k_dst_batch = k_dst[m] + k_start;
    //     for (size_t n = 0; n < q_seq_len; n++) {
    //         auto q_dst_seq = q_dst_batch + n * sizePerHeadAligned * mhaInputDataTypeSize;
    //         auto k_dst_seq = k_dst_batch + n * sizePerHeadAligned * mhaInputDataTypeSize;
    //         call_args.q_src = q_src;
    //         call_args.k_src = k_src;
    //         call_args.cos = cos;
    //         call_args.sin = sin;
    //         call_args.q_dst = q_dst_seq;
    //         call_args.k_dst = k_dst_seq;
    //         //call_args.q_dst_stride = q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;

    //         for (size_t s = 0; s < headNum; s++) {
    //             (*rotaryKernel)(&call_args);
    //             q_src += sizePerHead * 3 * inputDataTypeSize;
    //             k_src += sizePerHead * 3 * inputDataTypeSize;
    //             q_dst_seq += q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
    //             k_dst_seq += maxSeqLen * sizePerHeadAligned * mhaInputDataTypeSize;
    //             call_args.q_src = q_src;
    //             call_args.k_src = k_src;
    //             call_args.q_dst = q_dst_seq;
    //             call_args.k_dst = k_dst_seq;
    //         }
    //         cos += rotaryNdims;
    //         sin += rotaryNdims;
    //     }
    // }
    cos_cached += offset * rotaryNdims;
    sin_cached += offset * rotaryNdims;
    parallel_for3d(batch, headNum, q_seq_len, [&](size_t b, size_t h, size_t s) {
        auto q_dst_batch = q_dst + b * headNum * q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
        auto k_dst_batch = k_dst[b] + k_start;
        auto q_src_batch = q_src + b * hiddenSize * 3 * q_seq_len * inputDataTypeSize;
        auto k_src_batch = k_src + b * hiddenSize * 3 * q_seq_len * inputDataTypeSize;
        auto q_dst_seq = q_dst_batch + s * sizePerHeadAligned * mhaInputDataTypeSize;
        auto k_dst_seq = k_dst_batch + s * sizePerHeadAligned * mhaInputDataTypeSize;
        auto q_src_seq = q_src_batch + s * hiddenSize * 3 * inputDataTypeSize;
        auto k_src_seq = k_src_batch + s * hiddenSize * 3 * inputDataTypeSize;
        jit_rotary_call_args call_args;
        call_args.q_quant = &q_quant;
        call_args.k_quant = &k_quant;
        call_args.q_src = q_src_seq + h * sizePerHead * 3 * inputDataTypeSize;
        call_args.k_src = k_src_seq + h * sizePerHead * 3 * inputDataTypeSize;
        call_args.cos = cos_cached + s * rotaryNdims;
        call_args.sin = sin_cached + s * rotaryNdims;
        call_args.q_dst = q_dst_seq + h * q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
        call_args.k_dst = k_dst_seq + h * maxSeqLen * sizePerHeadAligned * mhaInputDataTypeSize;
        (*rotaryKernel)(&call_args);
    });
}

void GPTNeoxAttn::applyRotaryPosEmbMemcpy(uint8_t* q_src, uint8_t* k_src, uint8_t* q_dst, const sv::small_vector<uint8_t*, 4>& k_dst, size_t k_start,
    float* cos_cached, float* sin_cached, size_t batch, size_t q_seq_len, size_t offset, uint8_t* v_src, const sv::small_vector<uint8_t*, 4>& v_dst) {
    cos_cached += offset * rotaryNdims;
    sin_cached += offset * rotaryNdims;
    parallel_for3d(batch, headNum, q_seq_len, [&](size_t b, size_t h, size_t s) {
        // q, k rotary encoding
        auto q_dst_batch = q_dst + b * headNum * q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
        auto k_dst_batch = k_dst[b] + k_start;
        auto v_dst_batch = v_dst[b] + k_start;
        auto q_src_batch = q_src + b * hiddenSize * 3 * q_seq_len * inputDataTypeSize;
        auto k_src_batch = k_src + b * hiddenSize * 3 * q_seq_len * inputDataTypeSize;
        auto v_src_batch = v_src + b * hiddenSize * 3 * q_seq_len * inputDataTypeSize;
        auto q_dst_seq = q_dst_batch + s * sizePerHeadAligned * mhaInputDataTypeSize;
        auto k_dst_seq = k_dst_batch + s * sizePerHeadAligned * mhaInputDataTypeSize;
        auto v_dst_seq = v_dst_batch + s * sizePerHeadAligned * mhaInputDataTypeSize;
        auto q_src_seq = q_src_batch + s * hiddenSize * 3 * inputDataTypeSize;
        auto k_src_seq = k_src_batch + s * hiddenSize * 3 * inputDataTypeSize;
        auto v_src_seq = v_src_batch + s * hiddenSize * 3 * inputDataTypeSize;
        jit_rotary_call_args call_args;
        call_args.q_quant = &q_quant;
        call_args.k_quant = &k_quant;
        call_args.q_src = q_src_seq + h * sizePerHead * 3 * inputDataTypeSize;
        call_args.k_src = k_src_seq + h * sizePerHead * 3 * inputDataTypeSize;
        call_args.cos = cos_cached + s * rotaryNdims;
        call_args.sin = sin_cached + s * rotaryNdims;
        call_args.q_dst = q_dst_seq + h * q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
        call_args.k_dst = k_dst_seq + h * maxSeqLen * sizePerHeadAligned * mhaInputDataTypeSize;
        (*rotaryKernel)(&call_args);
        // q, k concat
        memcpy(static_cast<uint8_t*>(call_args.q_dst) + rotaryNdims * mhaInputDataTypeSize, static_cast<uint8_t*>(call_args.q_src) + rotaryNdims * inputDataTypeSize, mhaInputDataTypeSize * (sizePerHead - rotaryNdims));
        memcpy(static_cast<uint8_t*>(call_args.k_dst) + rotaryNdims * mhaInputDataTypeSize, static_cast<uint8_t*>(call_args.k_src) + rotaryNdims * inputDataTypeSize, mhaInputDataTypeSize * (sizePerHead - rotaryNdims));
        // v concat
        memcpy(static_cast<uint8_t*>(v_dst_seq) + h * maxSeqLen * sizePerHeadAligned * mhaInputDataTypeSize,
            static_cast<uint8_t*>(v_src_seq) + h * sizePerHead * 3 * inputDataTypeSize,
            sizePerHead * mhaInputDataTypeSize);
    });
}

void GPTNeoxAttn::applyRotaryPosEmbMemcpyQuant(uint8_t* q_src, uint8_t* k_src, uint8_t* q_dst, const sv::small_vector<uint8_t*, 4>& k_dst, size_t k_start,
    float* cos_cached, float* sin_cached, size_t batch, size_t q_seq_len, size_t offset, uint8_t* v_src, const sv::small_vector<uint8_t*, 4>& v_dst) {
    cos_cached += offset * rotaryNdims;
    sin_cached += offset * rotaryNdims;
    parallel_for3d(batch, headNum, q_seq_len, [&](size_t b, size_t h, size_t s) {
        // q, k rotary encoding
        auto q_dst_batch = q_dst + b * headNum * q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
        auto k_dst_batch = k_dst[b] + k_start;
        auto v_dst_batch = v_dst[b] + k_start;
        auto q_src_batch = q_src + b * hiddenSize * 3 * q_seq_len * inputDataTypeSize;
        auto k_src_batch = k_src + b * hiddenSize * 3 * q_seq_len * inputDataTypeSize;
        auto v_src_batch = v_src + b * hiddenSize * 3 * q_seq_len * inputDataTypeSize;
        auto q_dst_seq = q_dst_batch + s * sizePerHeadAligned * mhaInputDataTypeSize;
        auto k_dst_seq = k_dst_batch + s * sizePerHeadAligned * mhaInputDataTypeSize;
        auto v_dst_seq = v_dst_batch + s * sizePerHeadAligned * mhaInputDataTypeSize;
        auto q_src_seq = q_src_batch + s * hiddenSize * 3 * inputDataTypeSize;
        auto k_src_seq = k_src_batch + s * hiddenSize * 3 * inputDataTypeSize;
        auto v_src_seq = v_src_batch + s * hiddenSize * 3 * inputDataTypeSize;
        jit_rotary_call_args call_args;
        call_args.q_quant = &q_quant;
        call_args.k_quant = &k_quant;
        call_args.q_src = q_src_seq + h * sizePerHead * 3 * inputDataTypeSize;
        call_args.k_src = k_src_seq + h * sizePerHead * 3 * inputDataTypeSize;
        call_args.cos = cos_cached + s * rotaryNdims;
        call_args.sin = sin_cached + s * rotaryNdims;
        call_args.q_dst = q_dst_seq + h * q_seq_len * sizePerHeadAligned * mhaInputDataTypeSize;
        call_args.k_dst = k_dst_seq + h * maxSeqLen * sizePerHeadAligned * mhaInputDataTypeSize;
        (*rotaryKernel)(&call_args);
        // q, k concat
        quant_i8(static_cast<uint8_t*>(call_args.q_dst) + rotaryNdims, static_cast<uint8_t*>(call_args.q_src) + rotaryNdims * inputDataTypeSize, sizePerHead - rotaryNdims, q_quant);
        quant_i8(static_cast<uint8_t*>(call_args.k_dst) + rotaryNdims, static_cast<uint8_t*>(call_args.k_src) + rotaryNdims * inputDataTypeSize, sizePerHead - rotaryNdims, k_quant);
        // v concat
        quant_i8(static_cast<uint8_t*>(v_dst_seq) + h * maxSeqLen * sizePerHeadAligned * mhaInputDataTypeSize,
            static_cast<uint8_t*>(v_src_seq) + h * sizePerHead * 3 * inputDataTypeSize,
            sizePerHead, v_quant);
    });
}

void GPTNeoxAttn::updateAttnMask(const int* attn_mask, size_t batch, size_t seq_len) {
    for (size_t m = 0; m < batch; m ++) {
        auto* mask = attnMasks.get() + m * maxSeqLen;
        for (size_t n = 0; n < seq_len; n++) {
            mask[n] = attn_mask[n] ? 0 : -FLT_MAX;
        }
        attn_mask += maxSeqLen;
    }
}

void GPTNeoxAttn::executeDynamicImpl(dnnl::stream strm) {
    auto qkv_data_dims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    qkv_data_dims.back() = qkv_data_dims.back() / 3;
    redefineOutputMemory({qkv_data_dims});

    execute(strm);
}


void GPTNeoxAttn::execute(dnnl::stream strm) {
    // [batch, seq_len, (num_heads * 3 * head_size)]
    auto* qkv = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_QKV)->getMemoryPtr()->GetPtr());
    const int* past_keys_num = reinterpret_cast<const int*>(getParentEdgeAt(IN_PAST_KEYS_NUM)->getMemoryPtr()->GetPtr());
    int* beam_idx = reinterpret_cast<int*>(getParentEdgeAt(IN_BEAM_IDX)->getMemoryPtr()->GetPtr());
    const int* attn_mask = reinterpret_cast<const int*>(getParentEdgeAt(ATTN_MASK_IDX)->getMemoryPtr()->GetPtr());
    auto* dst_data = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto& qkv_dims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    const auto batch = qkv_dims[0];
    const auto seq_len = qkv_dims[1];
    // lower 16 bit means the number of past keys, higher 16 bit means the model id
    auto new_seq_offset = static_cast<size_t>(past_keys_num[0]) & 0xffff;
    if (g_seq_offset != -1) {
        new_seq_offset = g_seq_offset;
        beam_idx = g_beam_idx;
        // auto name = getName();
        // if (name.find("layers.31") != std::string::npos)
        //     g_seq_offset++;
    }
    assert(new_seq_offset < maxSeqLen);
    updateAttnMask(attn_mask, batch, seq_len + new_seq_offset);
    // usage: each 1x300 sub model and 1x1 sub model will share the same model id
    // const auto model_id = static_cast<size_t>(past_keys_num[0]) >> 16;
    // first token will write to pastKeys offset 0
    bool first_token = new_seq_offset == 0;
    // [2, batch, num_heads, maxSeqLen, head_size]
    auto size_per_key_per_beam = headNum * maxSeqLen * sizePerHeadAligned * mhaInputDataTypeSize;
    sv::small_vector<uint8_t*, 4> current_k_bufs, current_v_bufs;
    GlobalContext::getInstance().getOrCreateStore(getName(), size_per_key_per_beam, first_token ? nullptr : beam_idx, batch,
        current_k_bufs, current_v_bufs, new_seq_offset);

    // TODO: support the sentence length is longer than maxSeqLen
    // if (seq_len + new_seq_offset > cosCached.size()) {
    //     initRotery(seq_len + new_seq_offset);
    //     reinitAttentionMask(batch, seq_len + new_seq_offset);
    // }

    // [batch, seq_len, (num_heads * 3 * head_size)]
    //   --> [batch, seq_len, num_heads, 3 * head_size]
    auto query = qkv;                                      // qkv[..., : self.head_size].permute(0, 2, 1, 3)
    auto key = qkv + sizePerHead * inputDataTypeSize;           // qkv[..., self.head_size : 2 * self.head_size].permute(0, 2, 1, 3)
    auto value = qkv + 2 * sizePerHead * inputDataTypeSize;     // qkv[..., 2 * self.head_size :].permute(0, 2, 1, 3)
    // transpose + rotary embbeding:
    // transpose: [batch, seq_len, num_attention_heads, 3 * head_size] -->
    //          3 [batch, num_attention_heads, seq_len, head_size]
    // rotary embbeding: part of key will write to past_key, part of query will write to tempory buffer
    if (useInt8) {
        // query pass part(temp buffer): query = torch.cat((query, query_pass), dim=-1)
        // key pass part(past_key): key = torch.cat((key, key_pass), dim=-1)
        // value(pastKeys): value = torch.cat((past_value, value), dim=-2)
        applyRotaryPosEmbMemcpyQuant(query, key, queryTranspose.get(), current_k_bufs, mhaInputDataTypeSize * new_seq_offset * sizePerHeadAligned,
            cosCached.get(), sinCached.get(), batch, seq_len, new_seq_offset, value, current_v_bufs);
    } else {
        // query pass part(temp buffer): query = torch.cat((query, query_pass), dim=-1)
        // key pass part(past_key): key = torch.cat((key, key_pass), dim=-1)
        // value(pastKeys): value = torch.cat((past_value, value), dim=-2)
        applyRotaryPosEmbMemcpy(query, key, queryTranspose.get(), current_k_bufs, mhaInputDataTypeSize * new_seq_offset * sizePerHeadAligned,
            cosCached.get(), sinCached.get(), batch, seq_len, new_seq_offset, value, current_v_bufs);
    }
    // attn_output = _attn(query, key, value)
    // attn_output = _merge_heads(attn_output, self.num_attention_heads, self.head_size)
    auto head_stride_in_q = sizePerHeadAligned * seq_len;
    auto batch_stride_in_q = head_stride_in_q * headNum;
    auto head_stride_in_kv = sizePerHeadAligned * maxSeqLen;
    // q: [batch, num_heads, query_seq_len, head_size]
    // k: [batch, num_heads, key_seq_len, head_size]
    // v: [batch, num_heads, value_seq_len, head_size]
    // attention_mask: [batch, 1, 1, key_seq_len]
    // attn_output: [batch, query_seq_len, num_heads * head_size]
    gpt::MHAGPT::ExecParam param = {
        batch, seq_len, seq_len + new_seq_offset, new_seq_offset + 1,
        queryTranspose.get(), current_k_bufs, current_v_bufs,
        attnMasks.get(),
        dst_data,
        head_stride_in_q, batch_stride_in_q,    // q stride
        head_stride_in_kv,                      // kv stride
        maxSeqLen,                              // attn_mask stride
        sizePerHead, hiddenSize * seq_len,      // output stride
        q_quant != 0.0f ? 1.0f / q_quant : 0.0f,
        k_quant != 0.0f ? 1.0f / k_quant : 0.0f,
        qk_quant,
        v_quant != 0.0f ? 1.0f / v_quant : 0.0f,
        qkv_quant,
    };

    mhaGPT->exec(param);
}

bool GPTNeoxAttn::canFuse(const NodePtr& node) const {
    if (q_quant != 0.0f)
        return canFuseSimpleOperation(node);
    return false;
}

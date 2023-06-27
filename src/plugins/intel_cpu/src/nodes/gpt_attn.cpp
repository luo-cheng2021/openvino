// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>

#include "ie_parallel.hpp"
#include "eltwise.h"
#include "fake_quantize.h"
#include "gpt_attn.h"
#include <ngraph/opsets/opset10.hpp>
#include <utils/shape_inference/shape_inference_internal_dyn.hpp>
#include "llm_emb_gpt.hpp"
#include "llm_mha_gpt.hpp"

using namespace InferenceEngine;
using namespace ov::intel_cpu;
using namespace ov::intel_cpu::node;

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

namespace ov::intel_cpu::node {
class attn_gpt {
public:
    struct create_param {
        size_t num_heads;
        size_t head_size;
        size_t head_size_aligned;       // better to aligned to 64 bytes for best performance, apply for qkv
        size_t max_seq_len;             // max seq length for computing the size of matmul tmp result
        // supported (qkv, dst): (bf16, bf16)
        llmdnn::data_type_t qkv_precision;
        llmdnn::data_type_t dst_precision;
        size_t rotary_emb_base;
        float normal_factor;
        float rotary_pct;
        bool use_position2d;
    };
    struct exec_param {
        size_t batch;
        size_t query_seq_len;
        size_t past_seq_len;
        bool is_causal_in_attention;        // causal mask is fused in attention mask: chatglm uses it.
        uint8_t* qkv;
        uint8_t** layer_past_key_dst;
        uint8_t** layer_past_value_dst;
        int* position2d_ids;                // shape: [batch, 2, query_seq_len]
        float* attention_mask;              // attention mask, attention_mask[0] shape:
                                            //      [batch, 1, 1, key_seq_len], when is_causal_in_attention is false
                                            //      [batch, 1, query_seq_len, key_seq_len], when is_causal_in_attention is true
        uint8_t* attn_output;
        size_t head_stride_in_kv;
    };

    attn_gpt();
    bool create(const create_param& param);
    void exec(const exec_param& param);

private:
    create_param _create_param;
    std::shared_ptr<llmdnn::emb_gpt> _emb_gpt;
    std::shared_ptr<llmdnn::mha_gpt> _mha_gpt;
    std::shared_ptr<uint8_t> _query_dst;
    size_t _query_cached_batch = 0;
};

attn_gpt::attn_gpt(): _emb_gpt(std::make_shared<llmdnn::emb_gpt>()),
                      _mha_gpt(std::make_shared<llmdnn::mha_gpt>()) {
}

bool attn_gpt::create(const attn_gpt::create_param& param) {
    _create_param = param;
    llmdnn::emb_gpt::create_param emb_param;
    emb_param.num_heads = param.num_heads;
    emb_param.head_size = param.head_size;
    emb_param.head_size_aligned = param.head_size_aligned;
    emb_param.qkv_precision = param.qkv_precision;
    emb_param.dst_precision = param.dst_precision;
    emb_param.max_seq_len = param.max_seq_len;
    emb_param.rotary_emb_base = param.rotary_emb_base;
    emb_param.rotary_pct = param.rotary_pct;
    emb_param.use_position2d = param.use_position2d;

    if (!_emb_gpt->create(emb_param))
        return false;

    llmdnn::mha_gpt::create_param mha_param;
    mha_param.num_heads = param.num_heads;
    mha_param.head_size = param.head_size;
    mha_param.head_size_aligned = param.head_size_aligned;
    mha_param.normal_factor = param.normal_factor;
    mha_param.qkv_precision = param.qkv_precision;
    mha_param.dst_precision = param.dst_precision;
    mha_param.max_seq_len = param.max_seq_len;

    return _mha_gpt->create(mha_param);
}

void attn_gpt::exec(const attn_gpt::exec_param& param) {
    if (_query_cached_batch < param.batch) {
        auto precision_size = _create_param.qkv_precision == llmdnn::dnnl_bf16 ? 2 : 1;
        auto capacity = param.batch * _create_param.max_seq_len * (_create_param.num_heads * _create_param.head_size_aligned) *
            precision_size;
        _query_dst = std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(aligned_alloc(64, capacity)),
            [](void * p) { ::free(p); });
        memset(_query_dst.get(), 0, capacity);
        _query_cached_batch = param.batch;
    }

    llmdnn::emb_gpt::exec_param emb_param;
    emb_param.batch = param.batch;
    emb_param.query_seq_len = param.query_seq_len;
    emb_param.past_seq_len = param.past_seq_len;
    emb_param.qkv = param.qkv;
    emb_param.query_dst = _query_dst.get();
    emb_param.layer_past_key_src = param.layer_past_key_dst;
    emb_param.layer_past_value_src = param.layer_past_value_dst;
    emb_param.layer_past_key_dst = param.layer_past_key_dst;
    emb_param.layer_past_value_dst = param.layer_past_value_dst;
    emb_param.position2d_ids = param.position2d_ids;
    emb_param.head_stride_in_kv = param.head_stride_in_kv;
    _emb_gpt->exec(emb_param);

    llmdnn::mha_gpt::exec_param mha_param;
    mha_param.batch = param.batch;
    mha_param.query_seq_len = param.query_seq_len;
    mha_param.key_seq_len = param.query_seq_len + param.past_seq_len;
    mha_param.q = emb_param.query_dst;
    mha_param.attn_output = param.attn_output;
    mha_param.head_stride_in_kv = param.head_stride_in_kv;
    mha_param.is_causal_in_attention = param.is_causal_in_attention;
    mha_param.attention_mask = param.attention_mask;
    mha_param.k = emb_param.layer_past_key_dst;
    mha_param.v = emb_param.layer_past_value_dst;
    _mha_gpt->exec(mha_param);
}
} // namespace ov::intel_cpu::node

class GlobalContext {
public:
    using buffer_t = std::shared_ptr<uint8_t>;
    using beam_buffers_t = std::vector<buffer_t>;

    struct PastKVStore {
        // real memory buffer
        beam_buffers_t key_buffer;
        beam_buffers_t value_buffer;
        std::vector<uint8_t*> current_k_bufs;
        std::vector<uint8_t*> current_v_bufs;
    };
    static GlobalContext& getInstance() {
        static GlobalContext instance;
        return instance;
    }
    void init(size_t head_num, size_t size_per_head, size_t size_per_head_aligned, size_t max_seq_len, size_t data_type_len, size_t layer_num) {
        headNum = head_num;
        sizePerHead = size_per_head;
        sizePerHeadAligned = size_per_head_aligned;
        maxSeqLen = max_seq_len;
        dataTypeLen = data_type_len;
        layerNum = layer_num;
        simpleKVStore.resize(layerNum);
    }
    void getOrCreateStore(size_t layer_idx, size_t new_size_per_key_per_beam, const int* beam_idx, size_t beam_idx_num,
        uint8_t** current_k_bufs, uint8_t** current_v_bufs, size_t valid_histroy_seq_len) {
        // expected buffer: [2, beam_num/batch, headNum, maxSeqLen, sizePerHead]
        auto& store = simpleKVStore[layer_idx];
        // new_size_per_key_per_beam = headNum * maxSeqLen * sizePerHead * inputDataTypeSize
        // not init
        if (store.key_buffer.size() < beam_idx_num) {
            int works = layerNum;
            tbb::parallel_for(0, works, [&](int cur_idx) {
                auto& layer = simpleKVStore[cur_idx];
                layer.key_buffer.resize(beam_idx_num);
                layer.value_buffer.resize(beam_idx_num);
                layer.current_k_bufs.resize(beam_idx_num);
                layer.current_v_bufs.resize(beam_idx_num);
                for (size_t i = 0; i < beam_idx_num; i++) {
                    layer.key_buffer[i] = std::shared_ptr<uint8_t>(
                                reinterpret_cast<uint8_t*>(aligned_alloc(64, new_size_per_key_per_beam)),
                                [](void * p) { ::free(p); });
                    memset(layer.key_buffer[i].get(), 0, new_size_per_key_per_beam);
                    layer.value_buffer[i] = std::shared_ptr<uint8_t>(
                                reinterpret_cast<uint8_t*>(aligned_alloc(64, new_size_per_key_per_beam)),
                                [](void * p) { ::free(p); });
                    memset(layer.value_buffer[i].get(), 0, new_size_per_key_per_beam);
                    layer.current_k_bufs[i] = layer.key_buffer[i].get();
                    layer.current_v_bufs[i] = layer.value_buffer[i].get();
                }
            }, tbb::static_partitioner());
            for (size_t i = 0; i < beam_idx_num; i++) {
                current_k_bufs[i] = store.current_k_bufs[i];
                current_v_bufs[i] = store.current_v_bufs[i];
            }
            sizePerKeyPerBeam = new_size_per_key_per_beam;
            return;
        }
        assert(beam_idx_num <= store.key_buffer.size());
        for (size_t i = 0; i < beam_idx_num; i++) {
            current_k_bufs[i] = store.current_k_bufs[i];
            current_v_bufs[i] = store.current_v_bufs[i];
        }
        // first token or no beam search, ignore the reorder
        if (beam_idx == nullptr || beam_idx_num <= 1)
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

        uint8_t** ptrs_k = reinterpret_cast<uint8_t**>(alloca(beam_idx_num * sizeof(uint8_t*)));
        uint8_t** ptrs_v = reinterpret_cast<uint8_t**>(alloca(beam_idx_num * sizeof(uint8_t*)));
        memset(ptrs_k, 0, beam_idx_num * sizeof(uint8_t*));
        memset(ptrs_v, 0, beam_idx_num * sizeof(uint8_t*));
        // not used buffers pointer(items should be small numbers, use vector to decrease memory alloction times)
        uint8_t** no_use_ptrs_k = reinterpret_cast<uint8_t**>(alloca(beam_idx_num * sizeof(uint8_t*)));
        std::memcpy(no_use_ptrs_k, store.current_k_bufs.data(), beam_idx_num * sizeof(uint8_t*));
        std::pair<size_t, size_t>* copy_pairs = reinterpret_cast<std::pair<size_t, size_t>*>(alloca(beam_idx_num * sizeof(std::pair<size_t, size_t>)));
        int copy_count = 0;
        // first pass: no shared items, shared items first occurence
        for (size_t i = 0; i < beam_idx_num; i++) {
            auto wanted_idx = beam_idx[i];
            if (no_use_ptrs_k[wanted_idx]) {
                ptrs_k[i] = store.current_k_bufs[wanted_idx];
                ptrs_v[i] = store.current_v_bufs[wanted_idx];
                no_use_ptrs_k[wanted_idx] = nullptr;
            }
        }
        // second pass: shared items
        for (size_t i = 0; i < beam_idx_num; i++) {
            if (ptrs_k[i] == nullptr) {
                auto wanted_idx = beam_idx[i];
                for (size_t j = 0; j < beam_idx_num; j++) {
                    if (no_use_ptrs_k[j]) {
                        copy_pairs[copy_count++] ={wanted_idx, j};
                        ptrs_k[i] = no_use_ptrs_k[j];
                        ptrs_v[i] = store.current_v_bufs[j];
                        no_use_ptrs_k[j] = nullptr;
                        break;
                    }
                }
            }

            current_k_bufs[i] = ptrs_k[i];
            current_v_bufs[i] = ptrs_v[i];
        }
        // third pass: copy, only first layer does the copy
        if (copy_count && layer_idx == 0) {
            int works = layerNum;
            tbb::parallel_for(0, works, [&](int cur_work) {
                auto& layer = simpleKVStore[cur_work];
                for (int i = 0; i < copy_count; i++) {
                    auto& item = copy_pairs[i];
                    auto* src_k = layer.current_k_bufs[item.first];
                    auto* dst_k = layer.current_k_bufs[item.second];
                    auto* src_v = layer.current_v_bufs[item.first];
                    auto* dst_v = layer.current_v_bufs[item.second];
                    for (size_t h = 0; h < headNum; h++) {
                        auto sub_src_k = src_k + (h * maxSeqLen) * sizePerHeadAligned * dataTypeLen;
                        auto sub_dst_k = dst_k + (h * maxSeqLen) * sizePerHeadAligned * dataTypeLen;
                        memcpy(sub_dst_k, sub_src_k, sizePerHeadAligned * dataTypeLen * valid_histroy_seq_len);
                        auto* sub_src_v = src_v + (h * maxSeqLen) * sizePerHeadAligned * dataTypeLen;
                        auto* sub_dst_v = dst_v + (h * maxSeqLen) * sizePerHeadAligned * dataTypeLen;
                        memcpy(sub_dst_v, sub_src_v, sizePerHeadAligned * dataTypeLen * valid_histroy_seq_len);
                    }
                }
            }, tbb::static_partitioner());
        }

        for (size_t i = 0; i < beam_idx_num; i++) {
            store.current_k_bufs[i] = ptrs_k[i];
            store.current_v_bufs[i] = ptrs_v[i];
        }
    }

private:
    std::vector<PastKVStore> simpleKVStore;
    size_t headNum;
    size_t sizePerHead;
    size_t sizePerHeadAligned;
    size_t maxSeqLen;
    size_t dataTypeLen;
    size_t layerNum;
    size_t sizePerKeyPerBeam;
};

bool GPTAttn::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!ov::is_type<op::v10::GPTAttn>(op)) {
            errorMessage = "Not supported GPTAttn operation version. CPU plug-in supports only 10th version.";
            return false;
        }
    } catch (...) {
        return false;
    }

    return true;
}

GPTAttn::GPTAttn(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, InternalDynShapeInferFactory()) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    auto attn_op = ov::as_type_ptr<ov::op::v10::GPTAttn>(op);
    layerNum = attn_op->m_layer_num;
    headNum = attn_op->m_head_num;
    sizePerHead = attn_op->m_size_per_head;
    rotaryEmbBase = attn_op->m_rotary_emb_base;
    rotaryPct = attn_op->m_rotary_pct;
    curLayerNum = attn_op->m_cur_layer_num;
    maxSeqLen = attn_op->m_max_seq_len;
    normalFactor = 1.0f / sqrtf(static_cast<float>(sizePerHead));
    q_quant = attn_op->m_q_quant;
    k_quant = attn_op->m_k_quant;
    qk_quant = attn_op->m_qk_quant;
    v_quant = attn_op->m_v_quant;
    usePosition2d = attn_op->m_use_position2d;
    useInt8 = q_quant != 0.0f;
    if (useInt8) {
        sizePerHeadAligned = rnd_up(sizePerHead, 64);
    } else {
        sizePerHeadAligned = rnd_up(sizePerHead, 32);
    }
}

void GPTAttn::extractQuantParam() {
    for (size_t i = 0; i < fusedWith.size(); ++i) {
        auto& node = fusedWith[i];

        if (dynamic_cast<Eltwise*>(node.get())) {
            continue;
        }

        if (auto* fakeQuantizeNode = dynamic_cast<FakeQuantize*>(node.get())) {
            qkv_quant = fakeQuantizeNode->getInputScale();
            continue;
        }
    }
}

void GPTAttn::initSupportedPrimitiveDescriptors() {
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
                          {LayoutType::ncsp, Precision::FP32},
                          {LayoutType::ncsp, Precision::I32}},
                         {{LayoutType::ncsp, outputDataType}},
                          impl_desc_type::ref_any);
    GlobalContext::getInstance().init(headNum, sizePerHead, sizePerHeadAligned, maxSeqLen, mhaInputDataTypeSize, layerNum);
}

void GPTAttn::createPrimitive() {
    Node::createPrimitive();
}

void GPTAttn::prepareParams() {
    if (!attnGPT) {
        attnGPT = std::make_shared<attn_gpt>();
        attn_gpt::create_param param;
        param.num_heads = headNum;
        param.head_size = sizePerHead;
        param.head_size_aligned = sizePerHeadAligned;
        param.normal_factor = normalFactor;
        param.qkv_precision = llmdnn::dnnl_bf16;
        param.dst_precision = useInt8 ? llmdnn::dnnl_s8 : llmdnn::dnnl_bf16;
        param.max_seq_len = maxSeqLen;
        param.rotary_emb_base = rotaryEmbBase;
        param.rotary_pct = rotaryPct;
        param.use_position2d = usePosition2d;
        if (!attnGPT->create(param))
            THROW_ERROR << "create attnGPT failed.";
    }
}

void GPTAttn::executeDynamicImpl(dnnl::stream strm) {
    auto qkv_data_dims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    qkv_data_dims.back() = qkv_data_dims.back() / 3;
    redefineOutputMemory({qkv_data_dims});

    execute(strm);
}

void GPTAttn::execute(dnnl::stream strm) {
    // [batch, seq_len, (num_heads * 3 * head_size)]
    auto* qkv = reinterpret_cast<uint8_t*>(getParentEdgeAt(IN_QKV)->getMemoryPtr()->GetPtr());
    const int* past_keys_num = reinterpret_cast<const int*>(getParentEdgeAt(IN_PAST_KEYS_NUM)->getMemoryPtr()->GetPtr());
    int* beam_idx = reinterpret_cast<int*>(getParentEdgeAt(IN_BEAM_IDX)->getMemoryPtr()->GetPtr());
    float* attn_mask = reinterpret_cast<float*>(getParentEdgeAt(IN_ATTN_MASK)->getMemoryPtr()->GetPtr());
    int* position_ids = reinterpret_cast<int*>(getParentEdgeAt(IN_POSITION_IDS)->getMemoryPtr()->GetPtr());
    auto* dst_data = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());
    const auto& qkv_dims = getParentEdgeAt(IN_QKV)->getMemoryPtr()->getStaticDims();
    const auto& attn_dims = getParentEdgeAt(IN_ATTN_MASK)->getMemoryPtr()->getStaticDims();
    const auto batch = qkv_dims[0];
    const auto query_seq_len = qkv_dims[1];
    auto past_seq_len = static_cast<size_t>(past_keys_num[0]);
    assert(past_seq_len < maxSeqLen);
    // first token will write to pastKeys offset 0
    bool first_token = past_seq_len == 0;
    // [2, batch, num_heads, maxSeqLen, head_size]
    auto size_per_key_per_beam = headNum * maxSeqLen * sizePerHeadAligned * mhaInputDataTypeSize;
    uint8_t** current_k_bufs = reinterpret_cast<uint8_t**>(alloca(batch * sizeof(uint8_t*)));
    uint8_t** current_v_bufs = reinterpret_cast<uint8_t**>(alloca(batch * sizeof(uint8_t*)));
    GlobalContext::getInstance().getOrCreateStore(curLayerNum, size_per_key_per_beam, first_token ? nullptr : beam_idx, batch,
        current_k_bufs, current_v_bufs, past_seq_len);

    attn_gpt::exec_param param;
    param.batch = batch;
    param.query_seq_len = query_seq_len;
    param.past_seq_len = past_seq_len;
    param.qkv = qkv;
    param.layer_past_key_dst = current_k_bufs;
    param.layer_past_value_dst = current_v_bufs;
    param.position2d_ids = position_ids;

    param.is_causal_in_attention = attn_dims[2] != 1;
    param.attention_mask = attn_mask;
    param.head_stride_in_kv = maxSeqLen * sizePerHeadAligned;
    param.attn_output = dst_data;

    attnGPT->exec(param);
}

bool GPTAttn::canFuse(const NodePtr& node) const {
    if (q_quant != 0.0f)
        return canFuseSimpleOperation(node);
    return false;
}

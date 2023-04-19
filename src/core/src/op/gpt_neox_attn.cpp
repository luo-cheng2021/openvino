// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gpt_neox_attn.hpp"

#include "itt.hpp"

namespace ov {
op::v10::GPTNeoxAttn::GPTNeoxAttn(const Output<Node>& qkv, const Output<Node>& past_keys_num,
        const Output<Node>& beam_idx, const Output<Node>& attn_mask,
        int layer_num, int head_num, int size_per_head, int hidden_size, int max_position_embeddings,
        int rotary_emb_base, float rotary_pct, int max_seq_len,
        float q_quant, float k_quant, float qk_quant, float v_quant) :
        op::Op({qkv, past_keys_num, beam_idx, attn_mask}),
        m_layer_num(layer_num),
        m_head_num(head_num),
        m_size_per_head(size_per_head),
        m_hidden_size(hidden_size),
        m_max_position_embeddings(max_position_embeddings),
        m_rotary_emb_base(rotary_emb_base),
        m_rotary_pct(rotary_pct),
        m_max_seq_len(max_seq_len),
        m_q_quant(q_quant),
        m_k_quant(k_quant),
        m_qk_quant(qk_quant),
        m_v_quant(v_quant) {
    constructor_validate_and_infer_types();
}

bool op::v10::GPTNeoxAttn::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_GPTNeoxAttn_visit_attributes);
    visitor.on_attribute("layer_num", m_layer_num);
    visitor.on_attribute("head_num", m_head_num);
    visitor.on_attribute("size_per_head", m_size_per_head);
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("max_position_embeddings", m_max_position_embeddings);
    visitor.on_attribute("rotary_emb_base", m_rotary_emb_base);
    visitor.on_attribute("rotary_pct", m_rotary_pct);
    visitor.on_attribute("max_seq_len", m_max_seq_len);
    visitor.on_attribute("q_quant", m_q_quant);
    visitor.on_attribute("k_quant", m_k_quant);
    visitor.on_attribute("qk_quant", m_qk_quant);
    visitor.on_attribute("v_quant", m_v_quant);
    return true;
}

void op::v10::GPTNeoxAttn::validate_and_infer_types() {
    OV_OP_SCOPE(v10_GPTNeoxAttn_validate_and_infer_types);
    // NODE_VALIDATION_CHECK(this,
    //                       get_input_element_type(0).is_dynamic() || get_input_element_type(0).is_real(),
    //                       "The element type of the input tensor must be a floating point number.");
    // set_output_type(0, element::boolean, get_input_partial_shape(0));
    auto ps = get_input_partial_shape(0);
    ps.end()[-1] = m_hidden_size;
    set_output_type(0, element::f32, ps);
}

std::shared_ptr<Node> op::v10::GPTNeoxAttn::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_GPTNeoxAttn_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v10::GPTNeoxAttn>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3),
            m_layer_num,
            m_head_num,
            m_size_per_head,
            m_hidden_size,
            m_max_position_embeddings,
            m_rotary_emb_base,
            m_rotary_pct,
            m_max_seq_len,
            m_q_quant,
            m_k_quant,
            m_qk_quant,
            m_v_quant
            );
}
}  // namespace ov

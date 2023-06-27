// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gpt_attn.hpp"

#include "itt.hpp"

namespace ov {
op::v10::GPTAttn::GPTAttn(const Output<Node>& qkv,
        const Output<Node>& past_keys_num,
        const Output<Node>& beam_idx,
        const Output<Node>& attn_mask,
        const Output<Node>& position_ids,
        int layer_num,
        int head_num,
        int size_per_head,
        int rotary_emb_base,
        float rotary_pct,
        int cur_layer_num,
        int max_seq_len,
        bool use_position2d,
        float q_quant,
        float k_quant,
        float qk_quant,
        float v_quant) :
        op::Op({qkv, past_keys_num, beam_idx, attn_mask, position_ids}),
        m_layer_num(layer_num),
        m_head_num(head_num),
        m_size_per_head(size_per_head),
        m_rotary_emb_base(rotary_emb_base),
        m_rotary_pct(rotary_pct),
        m_cur_layer_num(cur_layer_num),
        m_max_seq_len(max_seq_len),
        m_use_position2d(use_position2d),
        m_q_quant(q_quant),
        m_k_quant(k_quant),
        m_qk_quant(qk_quant),
        m_v_quant(v_quant) {
    constructor_validate_and_infer_types();
}

bool op::v10::GPTAttn::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_GPTAttn_visit_attributes);
    visitor.on_attribute("layer_num", m_layer_num);
    visitor.on_attribute("head_num", m_head_num);
    visitor.on_attribute("size_per_head", m_size_per_head);
    visitor.on_attribute("rotary_emb_base", m_rotary_emb_base);
    visitor.on_attribute("rotary_pct", m_rotary_pct);
    visitor.on_attribute("cur_layer_num", m_cur_layer_num);
    visitor.on_attribute("max_seq_len", m_max_seq_len);
    visitor.on_attribute("use_position2d", m_use_position2d);
    visitor.on_attribute("q_quant", m_q_quant);
    visitor.on_attribute("k_quant", m_k_quant);
    visitor.on_attribute("qk_quant", m_qk_quant);
    visitor.on_attribute("v_quant", m_v_quant);
    return true;
}

void op::v10::GPTAttn::validate_and_infer_types() {
    OV_OP_SCOPE(v10_GPTAttn_validate_and_infer_types);
    // NODE_VALIDATION_CHECK(this,
    //                       get_input_element_type(0).is_dynamic() || get_input_element_type(0).is_real(),
    //                       "The element type of the input tensor must be a floating point number.");
    // set_output_type(0, element::boolean, get_input_partial_shape(0));
    auto ps = get_input_partial_shape(0);
    ps.end()[-1] /= 3;
    set_output_type(0, element::f32, ps);
}

std::shared_ptr<Node> op::v10::GPTAttn::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_GPTAttn_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v10::GPTAttn>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4),
        m_layer_num,
        m_head_num,
        m_size_per_head,
        m_rotary_emb_base,
        m_rotary_pct,
        m_cur_layer_num,
        m_max_seq_len,
        m_use_position2d,
        m_q_quant,
        m_k_quant,
        m_qk_quant,
        m_v_quant);
}
}  // namespace ov

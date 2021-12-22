// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "paddlepaddle_frontend/exceptions.hpp"

#include "internal/op/conditional_block.hpp"
#include "internal/op/select_input.hpp"

#include "internal/pass/transform_if.hpp"

#include "../../default_opset.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace opset8;

static std::shared_ptr<opset8::If> 
build_if_node(const std::shared_ptr<ov::op::internal::ConditionalBlock> cond0,
              const std::shared_ptr<ov::op::internal::ConditionalBlock> cond1,
              const std::shared_ptr<ov::op::internal::SelectInput> mask,
              const std::shared_ptr<opset8::Convert> cast,
              const std::shared_ptr<opset8::LogicalNot> logicalnot,
              std::vector<std::shared_ptr<Function>> functions) { 
    auto if_node = std::make_shared<opset8::If>(logicalnot);

/*     // then <-> cond1 as we switched the condition.
    if_node->set_then_body(then_branch);

    auto nodes = node.node_dict;
    for (const auto& then_param : then_branch->inputs()) {
        auto then_param2 = then_param.get_node_shared_ptr();
        auto in_tensor_name = then_param2->get_friendly_name();

        auto node_it = nodes.find(in_tensor_name);
        FRONT_END_GENERAL_CHECK(node_it != nodes.end(),
                                "Input ",
                                in_tensor_name,
                                " wasn't found. It may happen if model was cut incorrectly.");     
        auto param_node = ov::as_type_ptr<default_opset::Parameter>(then_param2);
        if_node->set_input(node_it->second, param_node, nullptr);
    }
 
    for (const auto& then_result : then_branch->outputs()) {  
        auto result_node = ov::as_type_ptr<default_opset::Result>(then_result.get_node_shared_ptr());   
        if_node->set_output(result_node, nullptr);
    }

    // else <-> cond0 */


    if_node->validate_and_infer_types();

    return if_node;
}

ov::frontend::pdpd::pass::TransformIf::TransformIf(std::vector<std::shared_ptr<Function>> functions) {
    auto false_label = ngraph::pattern::wrap_type<opset8::LogicalNot>(); // TODO: false_label, cond1_label, cast_label has the same producer.
    auto cond0_label = ngraph::pattern::wrap_type<ov::op::internal::ConditionalBlock>();
    auto cond1_label = ngraph::pattern::wrap_type<ov::op::internal::ConditionalBlock>({false_label});    
    auto cast_label = ngraph::pattern::wrap_type<opset8::Convert>();
    auto select_label =
        ngraph::pattern::wrap_type<ov::op::internal::SelectInput>({cond0_label, cond1_label, cast_label});    

    matcher_pass_callback callback = [functions](pattern::Matcher& m) -> bool {
        const auto& select_input = std::dynamic_pointer_cast<ov::op::internal::SelectInput>(m.get_match_root());
        const auto& conditional_block0 =
            std::dynamic_pointer_cast<ov::op::internal::ConditionalBlock>(select_input->get_input_node_shared_ptr(0));
        const auto& conditional_block1 = 
            std::dynamic_pointer_cast<ov::op::internal::ConditionalBlock>(select_input->get_input_node_shared_ptr(1));
        const auto& cast =
            std::dynamic_pointer_cast<opset8::Convert>(select_input->get_input_node_shared_ptr(2));
        const auto mask_idx = conditional_block1->get_input_size() - 1;
        const auto& logicalnot =
            std::dynamic_pointer_cast<opset8::LogicalNot>(conditional_block1->get_input_node_shared_ptr(mask_idx));     

        if (!select_input || !conditional_block0 || !conditional_block1 || !cast || !logicalnot)
            return false;

        auto if_node = build_if_node(conditional_block0, conditional_block1, select_input, cast, logicalnot, functions);

        replace_node(select_input, if_node);
        if_node->set_friendly_name(if_node->get_friendly_name());
        copy_runtime_info(select_input, if_node);        

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(select_label, "condtionalblock_select_to_If");
    this->register_matcher(m, callback);
}
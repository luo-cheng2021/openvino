// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/pass/transform_if.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "../../default_opset.hpp"
#include "internal/op/conditional_block.hpp"
#include "internal/op/select_input.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "paddlepaddle_frontend/exceptions.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace opset8;

static std::shared_ptr<opset8::If> build_if_node(
    const std::shared_ptr<ov::op::internal::ConditionalBlock> cond0 /*False branch*/,
    const std::shared_ptr<ov::op::internal::ConditionalBlock> cond1 /*True branch*/,
    const std::shared_ptr<ov::op::internal::SelectInput> mask,
    const std::shared_ptr<opset8::Convert> cast,
    const std::shared_ptr<opset8::LogicalNot> logicalnot,
    std::vector<std::shared_ptr<Function>> functions) {
    // then <-> cond0, else <-> cond1 as we switched the condition.
    const int32_t then_idx = cond0->get_subblock_index();
    const auto& then_branch = functions[then_idx];

    const int32_t else_idx = cond1->get_subblock_index();
    const auto& else_branch = functions[else_idx];

    const auto& then_params = then_branch->get_parameters();
    const auto& else_params = else_branch->get_parameters();

    auto if_node = std::make_shared<opset8::If>(logicalnot);
    if_node->set_then_body(then_branch);
    if_node->set_else_body(else_branch);

    const auto then_branch_inputs_from_parent = cond0->get_inputs_from_parent();
    NGRAPH_CHECK(then_branch_inputs_from_parent.size() == then_params.size(),
                 "Number of inputs to 'then_branch' is invalid. Expected " +
                     std::to_string(then_branch_inputs_from_parent.size()) + ", actual " +
                     std::to_string(then_params.size()));
    auto then_param = then_params.cbegin();
    for (const auto& from_parent : then_branch_inputs_from_parent) {
        if_node->set_input(from_parent, *then_param, nullptr);
        then_param++;
    }

    const auto else_branch_inputs_from_parent = cond1->get_inputs_from_parent();
    NGRAPH_CHECK(else_branch_inputs_from_parent.size() == else_params.size(),
                 "Number of inputs to 'else_branch' is invalid. Expected " +
                     std::to_string(else_branch_inputs_from_parent.size()) + ", actual " +
                     std::to_string(else_params.size()));
    auto else_param = else_params.cbegin();
    for (const auto& from_parent : else_branch_inputs_from_parent) {
        if_node->set_input(from_parent, nullptr, *else_param);
        else_param++;
    }
    NGRAPH_CHECK(then_branch->get_results().size() == else_branch->get_results().size(),
                 "'then' and 'else' branches have to have the same number of outputs");
    auto else_result = else_branch->get_results().cbegin();
    for (const auto& then_result : then_branch->get_results()) {
        if_node->set_output(then_result, *else_result);
        else_result++;
    }
    if_node->validate_and_infer_types();

    return if_node;
}

ov::frontend::pdpd::pass::TransformIf::TransformIf(std::vector<std::shared_ptr<Function>> functions) {
    // auto false_label = ngraph::pattern::wrap_type<opset8::LogicalNot>(); // TODO: false_label, cond1_label,
    // cast_label has the same producer. auto cond0_label =
    // ngraph::pattern::wrap_type<ov::op::internal::ConditionalBlock>(); auto cond1_label =
    // ngraph::pattern::wrap_type<ov::op::internal::ConditionalBlock>({false_label}); auto cast_label =
    // ngraph::pattern::wrap_type<opset8::Convert>(); auto select_label =
    //     ngraph::pattern::wrap_type<ov::op::internal::SelectInput>({cond0_label, cond1_label, cast_label});
    auto select_label = ngraph::pattern::wrap_type<ov::op::internal::SelectInput>();

    matcher_pass_callback callback = [functions](pattern::Matcher& m) -> bool {
        std::cout << "HERE! I CALL YOU!!!" << std::endl;
        const auto& select_input = std::dynamic_pointer_cast<ov::op::internal::SelectInput>(m.get_match_root());
        const auto& conditional_block0 =
            std::dynamic_pointer_cast<ov::op::internal::ConditionalBlock>(select_input->get_input_node_shared_ptr(0));
        const auto& conditional_block1 =
            std::dynamic_pointer_cast<ov::op::internal::ConditionalBlock>(select_input->get_input_node_shared_ptr(1));
        const auto& cast = std::dynamic_pointer_cast<opset8::Convert>(select_input->get_input_node_shared_ptr(2));
        const auto mask_idx = conditional_block0->get_input_size() - 1;  // False branch
        const auto& logicalnot =
            std::dynamic_pointer_cast<opset8::LogicalNot>(conditional_block0->get_input_node_shared_ptr(mask_idx));

        if (!select_input || !conditional_block0 || !conditional_block1 || !cast || !logicalnot) {
            std::cout << "Sorry! I EXIT HERE!!!" << std::endl;
            return false;
        }

        auto if_node = build_if_node(conditional_block0, conditional_block1, select_input, cast, logicalnot, functions);

        replace_node(select_input, if_node);
        if_node->set_friendly_name(if_node->get_friendly_name());
        copy_runtime_info(select_input, if_node);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(select_label, "condtionalblock_select_to_If");
    this->register_matcher(m, callback);
}
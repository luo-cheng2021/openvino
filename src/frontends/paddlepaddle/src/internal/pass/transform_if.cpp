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
#include "internal/op/tensorarray_length.hpp"
#include "internal/op/tensorarray_write.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "paddlepaddle_frontend/exceptions.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace opset8;

ov::frontend::pdpd::pass::TransformIf::TransformIf(std::vector<std::shared_ptr<Function>> functions) {
    // auto false_label = ngraph::pattern::wrap_type<opset8::LogicalNot>(); // TODO: false_label,
    // conditional_block1_label, cast_label has the same producer. auto conditional_block0_label =
    // ngraph::pattern::wrap_type<ov::op::internal::ConditionalBlock>(); auto conditional_block1_label =
    // ngraph::pattern::wrap_type<ov::op::internal::ConditionalBlock>({false_label}); auto cast_label =
    // ngraph::pattern::wrap_type<opset8::Convert>(); auto select_label =
    //     ngraph::pattern::wrap_type<ov::op::internal::SelectInput>({conditional_block0_label,
    //     conditional_block1_label, cast_label});
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

        /* build_if_node */

        // then <-> conditional_block0, else <-> conditional_block1 as we switched the condition.
        const int32_t then_idx = conditional_block0->get_subblock_index();
        const auto& then_branch = functions[then_idx];

        const int32_t else_idx = conditional_block1->get_subblock_index();
        const auto& else_branch = functions[else_idx];

        const auto& then_params = then_branch->get_parameters();
        const auto& else_params = else_branch->get_parameters();

        auto if_node = std::make_shared<opset8::If>(logicalnot);
        if_node->set_then_body(then_branch);
        if_node->set_else_body(else_branch);

        const auto then_branch_inputs_from_parent = conditional_block0->get_inputs_from_parent();
        NGRAPH_CHECK(then_branch_inputs_from_parent.size() == then_params.size(),
                     "Number of inputs to 'then_branch' is invalid. Expected " +
                         std::to_string(then_branch_inputs_from_parent.size()) + ", actual " +
                         std::to_string(then_params.size()));
        auto then_param = then_params.cbegin();
        for (const auto& from_parent : then_branch_inputs_from_parent) {
            auto node = from_parent;
            if (node.get_node_shared_ptr() == conditional_block1) {
                const auto& inputs = conditional_block1->input_values();
                for (auto &&input : inputs) {
                    if (input.get_node_shared_ptr()->get_friendly_name() == (*then_param)->get_friendly_name()) {
                        node = input;
                        break;
                    }
                }
            }

            if_node->set_input(node, *then_param, nullptr);
            then_param++;
        }

        const auto else_branch_inputs_from_parent = conditional_block1->get_inputs_from_parent();
        NGRAPH_CHECK(else_branch_inputs_from_parent.size() == else_params.size(),
                     "Number of inputs to 'else_branch' is invalid. Expected " +
                         std::to_string(else_branch_inputs_from_parent.size()) + ", actual " +
                         std::to_string(else_params.size()));
        auto else_param = else_params.cbegin();
        for (const auto& from_parent : else_branch_inputs_from_parent) {
            auto node = from_parent;
            if (node.get_node_shared_ptr() == conditional_block0) {
                const auto& inputs = conditional_block0->input_values();
                for (auto &&input : inputs) {
                    if (input.get_node_shared_ptr()->get_friendly_name() == (*else_param)->get_friendly_name()) {
                        node = input;
                        break;
                    }
                }
            }

            if_node->set_input(node, nullptr, *else_param);
            else_param++;
        }
        // NGRAPH_CHECK(then_branch->get_results().size() == else_branch->get_results().size(),
        //              "'then' and 'else' branches have to have the same number of outputs");
        auto else_results = else_branch->get_results();
        auto then_results = then_branch->get_results();
        /* replace conditional_block and select_input nodes. */

        NodeVector select_nodes;
        int if_outputs_idx = 0;
        for (auto i = 0; i < conditional_block0->outputs().size(); i++) {
            for (auto& cond0_consumer : conditional_block0->outputs()[i].get_target_inputs()) {
                if (is_type<ov::op::internal::SelectInput>(cond0_consumer.get_node())) {
                    const auto select_node =
                        dynamic_cast<ov::op::internal::SelectInput*>(cond0_consumer.get_node())->shared_from_this();
                    std::cout << "HERE GOT select_input !!" << std::endl;
                    auto then_out_idx = select_node->get_input_source_output(0).get_index();
                    auto else_out_idx = select_node->get_input_source_output(1).get_index();
                    if_node->set_output(then_results[then_out_idx], else_results[else_out_idx]);
                    if_node->validate_and_infer_types();

                    // replace each output of the select_input node
                    for (auto& output : select_node->outputs()) {
                        for (auto& consumer : output.get_target_inputs()) {
                            std::cout << "HERE GOT a consumer !!" << std::endl;
                            consumer.replace_source_output(if_node->outputs()[if_outputs_idx]);
                        }
                    }
                    if_outputs_idx++;

                    if_node->add_node_control_dependents(select_node);
                    if_node->add_node_control_dependencies(select_node);
                    select_node->clear_control_dependents();

                    select_nodes.emplace_back(select_node);
                } else {
                    FRONT_END_GENERAL_CHECK(
                        "Only select_input allowed to be consumer of conditional_block in this pattern.");
                }
            }
        }
        for (auto i = 0; i < conditional_block1->outputs().size(); i++) {
            for (auto& cond0_consumer : conditional_block1->outputs()[i].get_target_inputs()) {
                if (is_type<opset8::Result>(cond0_consumer.get_node())) {
                    std::cout << "HERE GOT result !!" << std::endl;
                    auto result_node =
                        std::dynamic_pointer_cast<opset8::Result>(cond0_consumer.get_node()->shared_from_this());

                    result_node->input(0).replace_source_output(Constant::create(element::f32, {1}, {0})->get_default_output());
                }
            }
        }

        copy_runtime_info(select_nodes, if_node);
        if_node->set_friendly_name(if_node->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(select_label, "condtionalblock_select_to_If");
    this->register_matcher(m, callback);
}

ov::frontend::pdpd::pass::TransformCond::TransformCond(std::vector<std::shared_ptr<Function>> funcs) {
    auto cond_label = ngraph::pattern::wrap_type<ov::op::internal::ConditionalBlock>();

    matcher_pass_callback callback = [funcs](pattern::Matcher& m) -> bool {
        std::vector<std::shared_ptr<Function>> functions = funcs;
        std::cout << "HERE! I CALL YOU!!!" << std::endl;
        auto conditional_block =
            std::dynamic_pointer_cast<ov::op::internal::ConditionalBlock>(m.get_match_root());
        size_t mask_idx = conditional_block->get_input_size() - 1;  // False branch
        std::shared_ptr<Node> cond = conditional_block->get_input_node_shared_ptr(mask_idx);
        
        if (!conditional_block || !cond) {
            std::cout << "Sorry! I EXIT HERE!!!" << std::endl;
            return false;
        }

        /* build_if_node */

        const int32_t then_idx = conditional_block->get_subblock_index();
        const auto& then_branch = functions[then_idx];
        const auto& then_params = then_branch->get_parameters();
        
        // make the else body, just pass through
        ParameterVector params;
        ResultVector results;
        for (auto i = 0; i < then_branch->get_output_size(); i++) {
            const auto param = std::make_shared<Parameter>(then_branch->get_output_element_type(i), then_branch->get_output_partial_shape(i));
            param->set_friendly_name(then_branch->get_output_op(i)->get_output_tensor(0).get_any_name());
            params.push_back(param);
            const auto result = std::make_shared<Result>(param);
            results.push_back(result);
        }
        const auto else_branch = std::make_shared<Function>(results, params);
        const auto& else_params = else_branch->get_parameters();

        auto if_node = std::make_shared<opset8::If>(cond);
        if_node->set_then_body(then_branch);
        if_node->set_else_body(else_branch);

        const auto then_branch_inputs_from_parent = conditional_block->get_inputs_from_parent();
        NGRAPH_CHECK(then_branch_inputs_from_parent.size() == then_params.size(),
                     "Number of inputs to 'then_branch' is invalid. Expected " +
                         std::to_string(then_branch_inputs_from_parent.size()) + ", actual " +
                         std::to_string(then_params.size()));
        auto then_param = then_params.cbegin();
        for (const auto& from_parent : then_branch_inputs_from_parent) {
            if_node->set_input(from_parent, *then_param, nullptr);
            then_param++;
        }
        // set_input may change the type and shape, update else first
        // for (auto i = 0; i < then_branch->get_output_size(); i++) {
        //     params[i]->set_partial_shape(then_branch->get_output_partial_shape(i));
        //     params[i]->set_element_type(then_branch->get_output_element_type(i));
        // }
        // else_branch->validate_nodes_and_infer_types();
        auto else_param = else_params.cbegin();
        for (const auto &else_param : else_params) {
            bool found = false;
            for (const auto& from_parent : then_branch_inputs_from_parent) {
                if (from_parent.get_any_name() == else_param->get_friendly_name()) {
                    if_node->set_input(from_parent, nullptr, else_param);
                    found = true;
                    break;
                }
            }
            // the output generate from the body, make a default value
            if (!found) {
                auto ps = else_param->get_partial_shape();
                const auto placeholder = Constant::create(else_param->get_element_type(), ps.get_min_shape(), {0});
                if_node->set_input(placeholder, nullptr, else_param);
            }
        }

        auto else_results = else_branch->get_results();
        auto then_results = then_branch->get_results();
        for (auto i = 0; i < else_results.size(); i++) {
            if_node->set_output(then_results[i], else_results[i]);
        }
        /* replace conditional_block and if nodes. */
        replace_node(conditional_block, if_node);
        if_node->set_friendly_name(conditional_block->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(cond_label, "condtionalblock_If");
    this->register_matcher(m, callback);
}

ov::frontend::pdpd::pass::ConditionalBlockTensorArrayOutputSlice::ConditionalBlockTensorArrayOutputSlice(std::vector<std::shared_ptr<Function>> functions) {
    // pattern: slice -> conditional_block -> const    
    auto conditionalblock_label = ngraph::pattern::wrap_type<ov::op::internal::ConditionalBlock>();
    auto slice_label = ngraph::pattern::wrap_type<opset8::Slice>({conditionalblock_label, ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input(), ngraph::pattern::any_input()});

    // auto slice_label = ngraph::pattern::wrap_type<opset8::Slice>();

    matcher_pass_callback callback = [functions](pattern::Matcher& m) -> bool {
        std::cout << "HERE! I CALL YOU!!!" << std::endl;
        const auto& slice_node = std::dynamic_pointer_cast<opset8::Slice>(m.get_match_root());
        auto conditionalblock_node = std::dynamic_pointer_cast<ov::op::internal::ConditionalBlock>(slice_node->get_input_node_shared_ptr(0));
      
        if (!conditionalblock_node || !slice_node) {
            std::cout << "Sorry! I EXIT HERE!!!" << std::endl;
            return false;
        }

        const auto& inputs = conditionalblock_node->input_values();
        const auto trip_count = Constant::create(element::i64, {1}, {1}); //FIXME?
        const auto& cond = inputs.back();
        const auto cond_name = cond.get_node_shared_ptr()->get_friendly_name();
        auto loop = std::make_shared<Loop>(trip_count, cond);
        const int32_t subblock_idx = conditionalblock_node->get_subblock_index();
        const auto& body_graph = functions[subblock_idx];
        loop->set_function(body_graph);

        // find_subgraph_match_to_pattern
        // pattern: TensorArrayWrite->TensorArrayLength, and the corresponding tensorarray is one of the output of body_graph.
        {
            ov::pass::Manager manager;
            manager.register_pass<ov::frontend::pdpd::pass::TensorArrayWriteConcatenation>(body_graph);      
            manager.run_passes(body_graph);
        }  

        const auto& parameters = body_graph->get_parameters();
        for (size_t i = 0; i < parameters.size(); i++) {
            bool marker = false;
            for (const auto& input : inputs) {
                if (input.get_node()->get_friendly_name() == parameters[i]->get_friendly_name()) {
                    loop->set_invariant_input(parameters[i], input);
                    marker=true;
                    break;
                }
            }
            FRONT_END_GENERAL_CHECK(marker, "could not find matching external input for internal parameter ", parameters[i]->get_friendly_name());            
        }

        loop->set_special_body_ports(Loop::SpecialBodyPorts{-1, -1});
        
        // replace output
        const auto& results = body_graph->get_results();
        OutputVector outputs(results.size());
        for (size_t i = 0; i < results.size(); i++) {
            auto out = loop->get_iter_value(results[i], -1);
            // auto out = loop->get_concatenated_slices(results[i], 0/*start*/, 1 /*stride*/, 1/*part_size*/, -1/*end*/, 0/*axis*/);
            conditionalblock_node->output(i).replace(out);
        }

        loop->add_node_control_dependents(conditionalblock_node);
        loop->add_node_control_dependencies(conditionalblock_node);
        conditionalblock_node->clear_control_dependents();

        loop->set_friendly_name(loop->get_friendly_name());
        copy_runtime_info(conditionalblock_node, loop);

        // bypass slice (FIXME: only works when slice[0], and tensorarray length is one.)
        for (auto& output : slice_node->outputs()) {
            for (auto& consumer : output.get_target_inputs()) {
                std::cout << "HERE GOT a consumer !!" << std::endl;
                consumer.replace_source_output(slice_node->input_values()[0]);
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(slice_label, "condtionalblock_tensorarray_slice");
    this->register_matcher(m, callback);
}

ov::frontend::pdpd::pass::TensorArrayWriteConcatenation::TensorArrayWriteConcatenation(std::shared_ptr<Function> func) {
    // pattern: tensorarraywrite -> tensorarraylength   
    // auto ta_length_label = ngraph::pattern::wrap_type<ov::op::internal::TensorArrayLength>();
    // auto ta_write_label = ngraph::pattern::wrap_type<ov::op::internal::TensorArrayWrite>({ta_length_label, ngraph::pattern::any_input(), ngraph::pattern::any_input()});

    auto ta_write_label = ngraph::pattern::wrap_type<ov::op::internal::TensorArrayWrite>();

    std::cout << "HERE! I ENTER (TensorArrayWriteConcatenation)!!!" << std::endl;

    matcher_pass_callback callback = [func](pattern::Matcher& m) -> bool {
        std::cout << "HERE! I CALL YOU (TensorArrayWriteConcatenation)!!!" << std::endl;
        const auto& ta_write_node = std::dynamic_pointer_cast<ov::op::internal::TensorArrayWrite>(m.get_match_root());
        auto ta_length_node = std::dynamic_pointer_cast<ov::op::internal::TensorArrayLength>(ta_write_node->get_input_node_shared_ptr(1));
      
        if (!ta_write_node || !ta_length_node) {
            std::cout << "Sorry! I EXIT HERE (TensorArrayWriteConcatenation)!!!" << std::endl;
            return false;
        }

        // bypass tensorarray_write as well as its tensortensor_length.
        const auto& ta_input = ta_write_node->get_input_source_output(0);
        ta_write_node->output(0).replace(ta_input);

        // remove tensorarray parameter
        const auto& ta_param = std::dynamic_pointer_cast<ngraph::op::Parameter>(ta_length_node->get_input_node_shared_ptr(0));
        func->remove_parameter(ta_param);
        func->validate_nodes_and_infer_types();

        ta_write_node->clear_control_dependents();

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(ta_write_label, "tensorarray_write_concatenation");
    this->register_matcher(m, callback);
}
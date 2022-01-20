// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/pass/transform_while.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "../../default_opset.hpp"
#include "internal/op/conditional_block.hpp"
#include "internal/op/tensorarray_length.hpp"
#include "internal/op/while.hpp"
#include "internal/op/tensorarray_to_tensor.hpp"
#include "internal/op/tensorarray_write.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "paddlepaddle_frontend/exceptions.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace opset8;

std::set<std::shared_ptr<Node>> try_trans_append_pattern(std::shared_ptr<ov::op::internal::While> while_node, std::shared_ptr<Function> child_model) {
    std::set<std::shared_ptr<Node>> new_results;
    // pattern: TensorArrayLength->TensorArrayWrite->Result, here we will ignore the parameter
    std::shared_ptr<Result> result;
    std::shared_ptr<Node> result_parent;
    for (const auto &node : child_model->get_results()) {
        const auto& tensor_write = node->get_input_node_shared_ptr(0);
        if (std::dynamic_pointer_cast<ov::op::internal::TensorArrayWrite>(tensor_write)) {
            if (std::dynamic_pointer_cast<ov::op::internal::TensorArrayLength>(tensor_write->get_input_node_shared_ptr(0))) {
                result_parent = tensor_write->get_input_node_shared_ptr(1);
                result = node;
            }
            if (std::dynamic_pointer_cast<ov::op::internal::TensorArrayLength>(tensor_write->get_input_node_shared_ptr(1))) {
                result_parent = tensor_write->get_input_node_shared_ptr(0);
                result = node;
            }
        }
    }
    // find append pattern, check concat in parent model
    std::shared_ptr<Node> concat_node;
    if (result) {
        for (auto i = 0; i < while_node->outputs().size(); i++) {
            for (auto& node : while_node->outputs()[i].get_target_inputs()) {
                if (is_type<ov::op::internal::TensorArrayToTensor>(node.get_node())) {
                    concat_node =
                        dynamic_cast<ov::op::internal::TensorArrayToTensor*>(node.get_node())->shared_from_this();
                }
            }
        }
    }

    if (concat_node) {
        // remove TensorArrayLength->TensorArrayWrite
        auto results = child_model->get_results();
        auto idx = child_model->get_result_index(result);
        for (const auto& result : results) {
            child_model->remove_result(result);
        }
        // special case: Parameter->TensorArrayLength->TensorArrayWrite
        if (std::dynamic_pointer_cast<Parameter>(result_parent)) {
            result_parent = std::make_shared<Convert>(result_parent->output(0), result_parent->get_output_element_type(0));
        }
        const auto new_result = result->clone_with_new_inputs(result_parent->outputs());
        new_result->set_friendly_name(result->get_friendly_name());
        new_result->output(0).set_names(result->output(0).get_names());
        results[idx] = std::dynamic_pointer_cast<Result>(new_result);
        child_model->add_results(results);
        new_results.insert(new_result);
        // remove TensorArrayToTensor
        const auto new_convert = std::make_shared<Convert>(concat_node->input_value(0), concat_node->get_input_element_type(0));
        new_convert->set_friendly_name(concat_node->get_friendly_name());
        replace_node(concat_node, new_convert);
    }

    return new_results;
}

ov::frontend::pdpd::pass::TransformWhile::TransformWhile(std::vector<std::shared_ptr<Function>> functions) {
    auto while_label = ngraph::pattern::wrap_type<ov::op::internal::While>();

    matcher_pass_callback callback = [functions](pattern::Matcher& m) -> bool {
        const auto& while_node = std::dynamic_pointer_cast<ov::op::internal::While>(m.get_match_root());
        if (!while_node)
            return false;
        const auto& inputs = while_node->input_values();
        const auto trip_count = Constant::create(element::i64, {1}, {-1});
        const auto& cond = inputs.back();
        const auto cond_name = cond.get_node_shared_ptr()->get_friendly_name();
        auto loop = std::make_shared<Loop>(trip_count, cond);
        auto sub_model = functions[while_node->m_sub_block];
        auto new_results = try_trans_append_pattern(while_node, sub_model);
        loop->set_function(sub_model);

        const auto& parameters = sub_model->get_parameters();
        const auto submodel_outputs = sub_model->outputs();
        for (size_t i = 0; i < parameters.size(); i++) {
            const auto& param_name = inputs[i].get_node()->get_friendly_name();
            auto out_node = sub_model->output(param_name);
            if (new_results.find(out_node.get_node_shared_ptr()) == new_results.end()) {
                loop->set_merged_input(parameters[i], inputs[i], out_node);
            } else {
                // we will ignore the tensorarray initialize value
                loop->set_invariant_input(parameters[i], inputs[i]);
            }
        }
        int64_t idx = -1;
        for (size_t i = 0; i < sub_model->get_results().size(); i++) {
            if (sub_model->output(i).get_tensor().get_any_name() == cond_name)
                idx = static_cast<int64_t>(i);
        }
        FRONT_END_GENERAL_CHECK(idx != -1, "could not find condition node in outputs.");

        loop->set_special_body_ports(Loop::SpecialBodyPorts{-1, idx});

        // replace output
        const auto& results = sub_model->get_results();
        OutputVector outputs(results.size());
        for (size_t i = 0; i < results.size(); i++) {
            if (new_results.find(results[i]) != new_results.end()) {
                // tensorarray value will be concat
                auto out = loop->get_concatenated_slices(results[i], 0, 1, 1, -1, 0);
                while_node->output(i).replace(out);
            } else {
                auto out = loop->get_iter_value(results[i], -1);
                while_node->output(i).replace(out);
            }
        }

        loop->add_node_control_dependents(while_node);
        loop->add_node_control_dependencies(while_node);
        while_node->clear_control_dependents();

        loop->set_friendly_name(loop->get_friendly_name());
        copy_runtime_info(while_node, loop);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(while_label, "while_loop");
    this->register_matcher(m, callback);
}
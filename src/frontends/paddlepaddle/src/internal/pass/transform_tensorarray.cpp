// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/pass/transform_tensorarray.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/variant.hpp>

#include "default_opset.hpp"
#include "internal/op/conditional_block.hpp"
#include "internal/op/tensorarray_length.hpp"
#include "internal/op/while.hpp"
#include "internal/op/tensorarray_to_tensor.hpp"
#include "internal/op/tensorarray_write.hpp"
#include "internal/op/tensorarray_append.hpp"
#include "internal/op/tensorarray_create.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "paddlepaddle_frontend/exceptions.hpp"
#include <ngraph/log.hpp>

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace frontend::pdpd::op::default_opset;

ov::frontend::pdpd::pass::TransformTensorArray::TransformTensorArray(std::vector<std::shared_ptr<Function>> functions) {
    auto length_label = ngraph::pattern::wrap_type<ov::op::internal::TensorArrayLength>();
    auto write_label = ngraph::pattern::wrap_type<ov::op::internal::TensorArrayWrite>({ngraph::pattern::any_input(), length_label});

    matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        const auto& write_node = opsMap.at(write_label).get_node_shared_ptr();
        const auto& length_node = opsMap.at(length_label).get_node_shared_ptr();
        if (!write_node || !length_node)
            return false;
        const auto& new_item = write_node->get_input_node_shared_ptr(0);
        const auto& list = length_node->get_input_node_shared_ptr(0);
        // remove TensorArrayLength->TensorArrayWrite
        const auto tensor_array = std::make_shared<ov::op::internal::TensorArrayAppend>(list->output(0), new_item->output(0));
        replace_node(write_node, tensor_array);
        tensor_array->set_friendly_name(write_node->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(write_label, "tensorarray");
    this->register_matcher(m, callback);
}

ov::frontend::pdpd::pass::TransformEliminateConvert::TransformEliminateConvert() {
    auto convert_pattern = ngraph::pattern::wrap_type<Convert>();

    matcher_pass_callback callback = [](ngraph::pattern::Matcher& m) {
        auto convert = std::dynamic_pointer_cast<Convert>(m.get_match_root());
        if (!convert) {
            return false;
        }
        if (convert->get_input_element_type(0) == convert->get_element_type()) {
            convert->output(0).replace(convert->input_value(0));
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(convert_pattern, "nop_convert");
    this->register_matcher(m, callback);
}

bool ov::frontend::pdpd::pass::TransformMarkupTensorArray::run_on_function(std::shared_ptr<ov::Function> f) {
    for (const auto& op : f->get_ordered_ops()) {
        // op is if or loop
        if (const auto& sub_graph_op = dynamic_pointer_cast<ngraph::op::util::MultiSubGraphOp>(op)) {
            const auto func_size = sub_graph_op->get_internal_subgraphs_size();
            for (auto n = 0; n < func_size; n++){
                const auto& func = sub_graph_op->get_function(n);
                // pass the tensor array to the parameter
                for (const auto& in : sub_graph_op->get_input_descriptions(n)) {
                    if (const auto& in_desc =
                            dynamic_pointer_cast<ngraph::op::util::MultiSubGraphOp::InputDescription>(in)) {

                        const auto& input = sub_graph_op->get_input_node_shared_ptr(in_desc->m_input_index);
                        // TensorArrayToTensor will produce tensor only
                        if (std::dynamic_pointer_cast<ov::op::internal::TensorArrayToTensor>(input))
                            continue;
                        if (op->get_input_source_output(in_desc->m_input_index).get_rt_info().count("TensorArray")) {
                            const auto& param = func->get_parameters().at(in_desc->m_body_parameter_index);
                            param->get_output_tensor(0).get_rt_info()["TensorArray"] = op->get_input_source_output(in_desc->m_input_index).get_rt_info()["TensorArray"];
                        }
                    }
                }
                // markup subgraph
                run_on_function(func);
                // get the result from subgraph
                for (const auto& out : sub_graph_op->get_output_descriptions(n)) {
                    if (const auto& out_desc =
                            dynamic_pointer_cast<ngraph::op::util::MultiSubGraphOp::OutputDescription>(out)) {
                        const auto& result =
                            sub_graph_op->get_function(n)->get_results().at(out_desc->m_body_value_index);
                        if (result->get_output_tensor(0).get_rt_info().count("TensorArray")) {
                            op->get_output_tensor(out_desc->m_output_index).get_rt_info()["TensorArray"] = result->get_output_tensor(0).get_rt_info()["TensorArray"];
                        }
                    }
                }
            }
        } else {
            // check if the parent.output has tensor array information
            for (auto i = 0; i < op->get_input_size(); i++) {
                const auto& input = op->get_input_node_shared_ptr(i);
                if (std::dynamic_pointer_cast<ov::op::internal::TensorArrayToTensor>(input))
                    continue;
                if (op->get_input_source_output(i).get_rt_info().count("TensorArray")) {
                    for (auto n = 0; n < op->get_output_size(); n++) {
                        if (op->get_output_tensor(n).get_rt_info().count("TensorArray")) {
                            NGRAPH_DEBUG << op->get_friendly_name() << ".output[" << n << "]" << " already has tensorarray, skip this one";
                            continue;
                        }
                        // copy the input tensorarray to all output ports
                        op->get_output_tensor(n).get_rt_info()["TensorArray"] = op->get_input_source_output(i).get_rt_info()["TensorArray"];
                    }
                }
            }
        }
    }
    return true;
}

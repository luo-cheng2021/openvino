// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/frontend.hpp"

#include <fstream>
#include <map>
#include <ngraph/pass/visualize_tree.hpp>
#include <string>
#include <vector>

#include "decoder_proto.hpp"
#include "framework.pb.h"
#include "input_model.hpp"
#include "internal/pass/transform_if.hpp"
#include "internal/pass/transform_while.hpp"
#include "internal/pass/transform_tensorarray.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/frontend/paddle/exception.hpp"
#include "internal/op/tensorarray_create.hpp"
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "op_table.hpp"
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset7.hpp"
#include "openvino/util/common_util.hpp"
#include "paddle_fw_node.hpp"
#include "paddle_utils.hpp"
#include "place.hpp"
#include "so_extension.hpp"

using namespace ov::opset7;
using namespace ov;
using namespace ov::frontend;

namespace ov {
namespace frontend {
namespace paddle {
namespace {

ov::Output<ov::Node> make_fake_dyn_out_node(const int axis, const PartialShape& partial_shape, const ov::Output<ov::Node>& input_node) {
    // data movement model
    const auto const_false = Constant::create(element::i32, {1}, {false});
    const auto param_input = std::make_shared<Parameter>(input_node.get_element_type(), partial_shape);
    const auto param_cond = std::make_shared<Parameter>(element::i32, PartialShape{1});
    const auto cond_result = std::make_shared<Result>(std::make_shared<Convert>(param_cond, element::boolean));
    const auto result = std::make_shared<Result>(param_input);
    auto sub_model = std::make_shared<Model>(ResultVector{result, cond_result}, ParameterVector{param_input, param_cond});

    const auto trip_count = Constant::create(element::i64, {1}, {-1});
    const auto cond = Constant::create(element::boolean, {1}, {true});
    auto loop = std::make_shared<Loop>(trip_count, cond);
    ov::pass::disable_constant_folding(loop);
    
    // iter, condition output idx
    loop->set_special_body_ports(Loop::SpecialBodyPorts{-1, 1});
    loop->set_function(sub_model);
    const auto& parameters = sub_model->get_parameters();
    const auto& results = sub_model->get_results();
    loop->set_invariant_input(parameters[0], input_node);
    loop->set_invariant_input(parameters[1], const_false);

    // the dynamic output
    const auto out = loop->get_concatenated_slices(results[0], 0, 1, 1, -1, axis);
    // the condition output
    loop->get_iter_value(results[1], -1);

    return out;
}

NamedOutputs make_ng_node(const std::map<paddle::TensorName, Output<Node>>& nodes,
                          const std::shared_ptr<OpPlace>& op_place,
                          const std::map<std::string, CreatorFunction>& CREATORS_MAP,
                          const std::shared_ptr<InputModel>& model) {
    const auto& op_desc = op_place->get_desc();

    auto creator_it = CREATORS_MAP.find(op_desc.type());
    FRONT_END_OP_CONVERSION_CHECK(creator_it != CREATORS_MAP.end(), "No creator found for ", op_desc.type(), " node.");
    NamedInputs named_inputs;
    for (const auto& input_port : op_desc.inputs()) {
        for (const auto& in_tensor_name : input_port.arguments()) {
            auto node_it = nodes.find(in_tensor_name);

            // TensorArray may create from scratch.
            // The node with tensorarray as its input may be created before the node with this tensorarray
            // as its output. e.g. the tensorarray is both the input and output of the same node.
            // So we have to create a fake empty node here.
            // Problem is, we have no idea which axis should be 0.
            // Since the models (faster/mask rcnn) are either concating tensors in tensorarray along the dynamic dimension,
            // or concating static shape tensors.
            // So we make the dynamic dimension to be 0. In case of static shape, we simply the the first dimension be 0.
            if (node_it == nodes.end()) {
                const auto& in_tensor = std::dynamic_pointer_cast<TensorPlace>(model->get_place_by_tensor_name(in_tensor_name));
                const auto& var_desc = in_tensor->get_desc();
                if (var_desc.type().has_tensor_array()) {
                    const auto& tensor_ps = in_tensor->get_partial_shape();
                    const auto& type = in_tensor->get_element_type();
                    Shape shape(tensor_ps.size());
                    for (auto i = 0; i < tensor_ps.size(); i++) {
                        const auto &dim = tensor_ps[i];
                        if (dim.is_static()) {
                            shape[i] = dim.get_length();
                        }
                    }

                    if (tensor_ps.is_static()) {
                        shape[1] = 0;
                    }

                    std::cout << "tensorarray ps " << tensor_ps << "fakenode " << shape << std::endl;

                    auto init = ov::opset6::Constant::create(type, shape, {0}); // FIXME
                    //auto fakenode = std::make_shared<ov::op::internal::TensorArrayCreate>(init);
                    auto fakenode = init;
                    // PartialShape ps(tensor_ps);
                    // ps[1] = Dimension();
                    // auto fakenode = make_fake_dyn_out_node(1, ps, init->output(0)).get_node_shared_ptr();
                    fakenode->set_friendly_name(in_tensor_name);
                    fakenode->output(0).get_tensor().add_names({in_tensor_name}); // ??
                    named_inputs[input_port.parameter()].push_back(fakenode);
                    continue; // ignore, handle it in its write_to_array
                }
            }

            // general check, because in case of error partial conversion should fail
            FRONT_END_GENERAL_CHECK(node_it != nodes.end(),
                                    "Input ",
                                    in_tensor_name,
                                    " for node with type ",
                                    op_desc.type(),
                                    " wasn't found. It may happen if model was cut incorrectly.");
            named_inputs[input_port.parameter()].push_back(node_it->second);
        }
    }
    NamedOutputs outputs;
    // In case the conversion function throws exception
    try {
        outputs = creator_it->second(NodeContext(DecoderProto(op_place), named_inputs));
    } catch (std::exception& ex) {
        FRONT_END_OP_CONVERSION_CHECK(false, "Fail to convert " + op_desc.type() + " Exception " + ex.what());
    }

    return outputs;
}

NamedOutputs make_framework_node(const std::map<paddle::TensorName, Output<Node>>& nodes,
                                 const std::shared_ptr<OpPlace>& op_place) {
    const auto& op_desc = op_place->get_desc();

    OutputVector inputs_vector;
    std::vector<std::string> inputs_names;
    NamedOutputs named_outputs;
    for (const auto& input_port : op_desc.inputs()) {
        for (const auto& in_tensor_name : input_port.arguments()) {
            auto it = nodes.find(in_tensor_name);
            // general check, because in case of error partial conversion should fail
            FRONT_END_GENERAL_CHECK(it != nodes.end(),
                                    "Input ",
                                    in_tensor_name,
                                    " for node with type ",
                                    op_desc.type(),
                                    " wasn't found. It may happen if model was cut incorrectly.");
            inputs_vector.push_back(it->second);
            inputs_names.push_back(in_tensor_name);
        }
    }

    auto node = std::make_shared<FrameworkNode>(DecoderProto(op_place), inputs_vector, inputs_names);

    return node->return_named_outputs();
}

bool normalize_framework_node(const std::shared_ptr<FrameworkNode>& node,
                              const std::map<std::string, CreatorFunction>& CREATORS_MAP) {
    auto type = node->get_op_type();
    auto creator_it = CREATORS_MAP.find(type);
    FRONT_END_OP_CONVERSION_CHECK(creator_it != CREATORS_MAP.end(), "No creator found for ", type, " node.");

    auto new_node_outputs = creator_it->second(NodeContext(node->get_decoder(), node->get_named_inputs()));
    auto new_node = new_node_outputs.begin()->second[0].get_node_shared_ptr();
    new_node->set_friendly_name(node->get_friendly_name());
    auto node_outputs = node->return_named_outputs();

    auto new_ports = new_node_outputs.begin();
    auto old_ports = node_outputs.begin();
    for (; new_ports != new_node_outputs.end() && old_ports != node_outputs.end(); ++new_ports, ++old_ports) {
        FRONT_END_OP_CONVERSION_CHECK(new_ports->first == old_ports->first,
                                      "Node outputs inconsistent after normalization: ",
                                      node->get_friendly_name());
        auto new_output = new_ports->second.begin();
        auto old_output = old_ports->second.begin();
        for (; new_output != new_ports->second.end() && old_output != old_ports->second.end();
             ++old_output, ++new_output) {
            old_output->replace(*new_output);
        }
    }
    return true;
}

std::istream* variant_to_stream_ptr(const ov::Any& variant, std::ifstream& ext_stream) {
    if (variant.is<std::istream*>()) {
        return variant.as<std::istream*>();
    } else if (variant.is<std::string>()) {
        const auto& model_path = variant.as<std::string>();
        ext_stream.open(model_path, std::ios::in | std::ifstream::binary);
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variant.is<std::wstring>()) {
        const auto& model_path = variant.as<std::wstring>();
        ext_stream.open(model_path, std::ios::in | std::ifstream::binary);
    }
#endif
    FRONT_END_INITIALIZATION_CHECK(ext_stream && ext_stream.is_open(), "Cannot open model file.");
    return &ext_stream;
}
}  // namespace

FrontEnd::FrontEnd() : m_op_translators(paddle::get_supported_ops()) {}

std::vector<std::shared_ptr<ov::Model>> FrontEnd::convert_each_node(
    const std::shared_ptr<ov::frontend::InputModel>& frontend_model,
    std::function<std::map<std::string, OutputVector>(const std::map<std::string, Output<Node>>&,
                                                      const std::shared_ptr<OpPlace>&)> func) {
    auto model = std::dynamic_pointer_cast<InputModel>(frontend_model);
    FRONT_END_GENERAL_CHECK(model, "Invalid input model");
    std::vector<std::shared_ptr<TensorPlace>> input_tensors;
    std::vector<std::shared_ptr<TensorPlace>> output_tensors;
    for (const auto& _inp_place : model->get_inputs()) {
        const auto& inp_place = std::dynamic_pointer_cast<TensorPlace>(_inp_place);
        input_tensors.emplace_back(inp_place);
    }
    for (const auto& _outp_place : model->get_outputs()) {
        const auto& outp_place = std::dynamic_pointer_cast<TensorPlace>(_outp_place);
        output_tensors.emplace_back(outp_place);
    }
    auto funcs = convert_each_node_recursive(model, 0, input_tensors, output_tensors, func);
    std::vector<std::shared_ptr<Model>> funcs_vec;
    for (auto &&item : funcs) {
        funcs_vec.emplace_back(item.second);
    }
    
    return funcs_vec;
}

std::map<int32_t, std::shared_ptr<ov::Model>> FrontEnd::convert_each_node_recursive(
    const std::shared_ptr<ov::frontend::InputModel>& frontend_model,
    const int32_t block_idx,
    const std::vector<std::shared_ptr<TensorPlace>>& input_tensors,
    const std::vector<std::shared_ptr<TensorPlace>>& output_tensors,
    std::function<std::map<std::string, OutputVector>(const std::map<std::string, Output<Node>>&,
                                                      const std::shared_ptr<OpPlace>&)> func) {
    auto model = std::dynamic_pointer_cast<InputModel>(frontend_model);
    FRONT_END_GENERAL_CHECK(model, "Invalid input model");
    auto nodes_dict(model->get_tensor_values());  // std::map<paddle::TensorName, Output<Node>>
    ParameterVector parameter_nodes;
    ResultVector result_nodes;
    OutputVector output_nodes;

    using PaddleOpType = std::string;
    std::map<int32_t, std::tuple<PaddleOpType,
                           std::vector<std::shared_ptr<TensorPlace>>,
                           std::vector<std::shared_ptr<TensorPlace>>>>
        subblock_inputs_outputs;  // keep info of controlflow ops

    for (const auto& _inp_place : input_tensors) {
        const auto& inp_place = std::dynamic_pointer_cast<TensorPlace>(_inp_place);
        const auto& var = inp_place->get_desc();
        const auto& shape = inp_place->get_partial_shape();
        const auto& type = inp_place->get_element_type();
        auto param = std::make_shared<Parameter>(type, shape);
        param->set_friendly_name(var.name());
        param->output(0).get_tensor().add_names({var.name()});
        nodes_dict[var.name()] = param;
        parameter_nodes.push_back(param);
    }

    const auto& op_places = model->get_op_places(block_idx);
    for (const auto& op_place : op_places) {
        const auto& op_desc = op_place->get_desc();
        if (op_desc.type() == "feed" || op_desc.type() == "fetch") {
            // inputs and outputs are stored in the model already
            continue;
        } else {
            if (op_desc.type() == "conditional_block") {
                std::vector<std::shared_ptr<TensorPlace>> outp_tensors;
                std::vector<std::shared_ptr<TensorPlace>> inp_tensors;

                auto outp_ports = op_place->get_output_ports();
                for (auto outp_port : outp_ports["Out"]) {
                    auto outp_tensor = outp_port->get_target_tensor_paddle();  // get_target_tensor();
                    outp_tensors.push_back(outp_tensor);
                }
                FRONT_END_GENERAL_CHECK(outp_tensors.size() > 0, "Port has no tensors connected.");

                auto inp_ports = op_place->get_input_ports();
                for (auto inp_port : inp_ports["Input"]) {
                    auto inp_tensor = inp_port->get_source_tensor_paddle();
                    inp_tensors.push_back(inp_tensor);
                }
                // FRONT_END_GENERAL_CHECK(inp_tensors.size() > 0, "Port has no tensors connected.");

                auto tmp_node = paddle::NodeContext(DecoderProto(op_place), paddle::NamedInputs());
                auto block_idx = tmp_node.get_attribute<int32_t>("sub_block");

                subblock_inputs_outputs[block_idx] = std::make_tuple(op_desc.type(), inp_tensors, outp_tensors);
            } else if (op_desc.type() == "while") {
                std::vector<std::shared_ptr<TensorPlace>> outp_tensors;
                std::vector<std::shared_ptr<TensorPlace>> inp_tensors;

                auto outp_ports = op_place->get_output_ports();
                for (auto outp_port : outp_ports["Out"]) {
                    auto outp_tensor = outp_port->get_target_tensor_paddle();
                    outp_tensors.push_back(outp_tensor);
                }
                FRONT_END_GENERAL_CHECK(outp_tensors.size() > 0, "Port has no tensors connected.");

                auto inp_ports = op_place->get_input_ports();
                for (auto inp_port : inp_ports["X"]) {
                    auto inp_tensor = inp_port->get_source_tensor_paddle();
                    inp_tensors.push_back(inp_tensor);
                }
                FRONT_END_GENERAL_CHECK(inp_tensors.size() > 0, "Port has no tensors connected.");

                auto tmp_node = paddle::NodeContext(DecoderProto(op_place), paddle::NamedInputs());
                auto block_idx = tmp_node.get_attribute<int32_t>("sub_block");

                subblock_inputs_outputs[block_idx] = std::make_tuple(op_desc.type(), inp_tensors, outp_tensors);
            }
            
            paddle::NamedOutputs named_outputs = func(nodes_dict, op_place);

            if (!named_outputs.empty()) {
                if (!op_desc.outputs().begin()->arguments().empty()) {
                    const auto& tensor_name = op_desc.outputs().begin()->arguments()[0];
                    auto node = named_outputs.begin()->second[0].get_node_shared_ptr();
                    node->set_friendly_name(tensor_name);
                }

                const auto& out_ports = op_desc.outputs();
                for (const auto& port : out_ports) {
                    // TODO: figure a way to safely handle unused outputs
                    if (named_outputs.count(port.parameter())) {
                        const auto& ng_outputs = named_outputs.at(port.parameter());
                        FRONT_END_OP_CONVERSION_CHECK(ng_outputs.size() == port.arguments_size(),
                                                      "The number of output tensors must be equal to "
                                                      "the number of outputs of the OV node.");
                        for (size_t idx = 0; idx < ng_outputs.size(); ++idx) {
                            const auto& var_name = port.arguments()[idx];
                            ng_outputs[idx].get_tensor().set_names({var_name});
                            // if nodes_dict already has node mapped to this tensor name it
                            // usually means that it was overwritten using setTensorValue
                            if (!nodes_dict.count(var_name))
                                nodes_dict[var_name] = ng_outputs[idx];
                            else {
                                // std::cout << "mainblock duplicated name " << var_name << std::endl;
                                nodes_dict[var_name] = ng_outputs[idx];
                            }
                        }
                    }
                }
            }
        }
    }

    for (const auto& _outp_place : output_tensors) {
        const auto& outp_place = std::dynamic_pointer_cast<TensorPlace>(_outp_place);
        auto var = outp_place->get_desc();
        auto input_var_name = var.name();
        auto result = std::make_shared<Result>(nodes_dict.at(input_var_name));
        result->set_friendly_name(input_var_name + "/Result");
        result_nodes.push_back(result);
        output_nodes.push_back(nodes_dict.at(input_var_name));
    }

    std::shared_ptr<ov::Model> main_block_func;
    if (parameter_nodes.size() > 0) {
        main_block_func = std::make_shared<ov::Model>(result_nodes, parameter_nodes);
    } else {
        main_block_func = std::make_shared<ov::Model>(output_nodes);
    }

    /* convert each sub block */
    std::map<int32_t, std::shared_ptr<ov::Model>> block_funcs;
    block_funcs.insert({block_idx, main_block_func});

    for (auto& item: subblock_inputs_outputs) {
        auto ctl_op_info = item.second;
        auto sub_block_func = convert_each_node_recursive(model, item.first, std::get<1>(ctl_op_info), std::get<2>(ctl_op_info), func);
        block_funcs.insert(sub_block_func.begin(), sub_block_func.end());
    }

    return block_funcs;
}

void FrontEnd::normalize(const std::vector<std::shared_ptr<Model>>& models) const {
    auto block_idx = 0;
    for (auto &model : models) {
        ov::pass::Manager manager;
        manager.register_pass<ov::pass::VisualizeTree>("pre_normalize"+std::to_string(block_idx)+".png");
        manager.register_pass<ov::frontend::paddle::pass::TransformEliminateConvert>();
        manager.register_pass<ov::frontend::paddle::pass::TransformTensorArray>(models);
        manager.register_pass<ov::frontend::paddle::pass::TransformIf>(models);
        manager.register_pass<ov::frontend::paddle::pass::TransformWhile>(models);
        manager.register_pass<ov::pass::VisualizeTree>("post_normalize"+std::to_string(block_idx)+".png");
        manager.run_passes(model);
        block_idx++;
    }
}

void FrontEnd::normalize(const std::shared_ptr<ov::Model>& model) const {
    std::vector<std::shared_ptr<Model>> models;
    models.push_back(model);
    normalize(models);
}

bool FrontEnd::supported_impl(const std::vector<ov::Any>& variants) const {
    // FrontEnd can only load model specified by one path, one file or two files.
    if (variants.empty() || variants.size() > 2)
        return false;

    // Validating first path, it must contain a model
    if (variants[0].is<std::string>()) {
        std::string suffix = ".pdmodel";
        std::string model_path = variants[0].as<std::string>();
        if (!ov::util::ends_with(model_path, suffix)) {
            model_path += paddle::get_path_sep<char>() + "__model__";
        }
        std::ifstream model_str(model_path, std::ios::in | std::ifstream::binary);
        // It is possible to validate here that protobuf can read model from the stream,
        // but it will complicate the check, while it should be as quick as possible
        return model_str && model_str.is_open();
    }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    else if (variants[0].is<std::wstring>()) {
        std::wstring suffix = L".pdmodel";
        std::wstring model_path = variants[0].as<std::wstring>();
        if (!ov::util::ends_with(model_path, suffix)) {
            model_path += paddle::get_path_sep<wchar_t>() + L"__model__";
        }
        std::ifstream model_str(model_path, std::ios::in | std::ifstream::binary);
        // It is possible to validate here that protobuf can read model from the stream,
        // but it will complicate the check, while it should be as quick as possible
        return model_str && model_str.is_open();
    }
#endif
    else if (variants[0].is<std::istream*>()) {
        // Validating first stream, it must contain a model
        auto p_model_stream = variants[0].as<std::istream*>();
        ::paddle::framework::proto::ProgramDesc fw;
        return fw.ParseFromIstream(p_model_stream);
    }
    return false;
}

InputModel::Ptr FrontEnd::load_impl(const std::vector<ov::Any>& variants) const {
    if (variants.size() == 1) {
        // The case when folder with __model__ and weight files is provided or .pdmodel file
        if (variants[0].is<std::string>()) {
            std::string m_path = variants[0].as<std::string>();
            return std::make_shared<InputModel>(m_path, m_telemetry);
        }
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
        else if (variants[0].is<std::wstring>()) {
            std::wstring m_path = variants[0].as<std::wstring>();
            return std::make_shared<InputModel>(m_path, m_telemetry);
        }
#endif
        // The case with only model stream provided and no weights. This means model has
        // no learnable weights
        else if (variants[0].is<std::istream*>()) {
            auto p_model_stream = variants[0].as<std::istream*>();
            return std::make_shared<InputModel>(std::vector<std::istream*>{p_model_stream}, m_telemetry);
        }
    } else if (variants.size() == 2) {
        // The case when .pdmodel and .pdparams files are provided
        std::ifstream model_stream;
        std::ifstream weights_stream;
        std::istream* p_model_stream = paddle::variant_to_stream_ptr(variants[0], model_stream);
        std::istream* p_weights_stream = paddle::variant_to_stream_ptr(variants[1], weights_stream);
        if (p_model_stream && p_weights_stream) {
            return std::make_shared<InputModel>(std::vector<std::istream*>{p_model_stream, p_weights_stream},
                                                m_telemetry);
        }
    }
    FRONT_END_THROW("Model can be loaded either from 1 or 2 files/streams");
}

std::shared_ptr<ov::Model> FrontEnd::convert(const InputModel::Ptr& model) const {
    auto paddle_model = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(paddle_model != nullptr, "Invalid input model");

    if (!m_transformation_extensions.empty()) {
        auto function = decode(model);

        ov::pass::Manager manager;
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(function);
        convert(function);
        return function;
    }

    auto f = convert_each_node(
        paddle_model,
        [&](const std::map<std::string, Output<Node>>& nodes_dict, const std::shared_ptr<OpPlace>& op_place) {
            return paddle::make_ng_node(nodes_dict, op_place, m_op_translators, paddle_model);
        });

    normalize(f);
    return ngraph::clone_function(*f[0]);
}

void FrontEnd::convert(const std::shared_ptr<ov::Model>& partiallyConverted) const {
    for (const auto& node : partiallyConverted->get_ordered_ops()) {
        if (ov::is_type<FrameworkNode>(node)) {
            paddle::normalize_framework_node(std::dynamic_pointer_cast<FrameworkNode>(node), m_op_translators);
        }
    }
    for (const auto& result : partiallyConverted->get_results()) {
        result->validate_and_infer_types();
    }

    normalize(partiallyConverted);
}

std::shared_ptr<ov::Model> FrontEnd::convert_partially(const InputModel::Ptr& model) const {
    auto paddle_model = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(paddle_model != nullptr, "Invalid input model");

    if (!m_transformation_extensions.empty()) {
        auto function = decode(model);

        ov::pass::Manager manager;
        for (const auto& transformation : m_transformation_extensions) {
            transformation->register_pass(manager);
        }
        manager.run_passes(function);
        convert(function);
        return function;
    }

    auto f = convert_each_node(
        paddle_model,
        [&](const std::map<std::string, Output<Node>>& nodes_dict, const std::shared_ptr<OpPlace>& op_place) {
            paddle::NamedOutputs named_outputs;
            try {
                named_outputs = paddle::make_ng_node(nodes_dict, op_place, m_op_translators, paddle_model);
            } catch (const OpConversionFailure&) {
                named_outputs = paddle::make_framework_node(nodes_dict, op_place);
            }
            return named_outputs;
        });

    normalize(f);

    return ngraph::clone_function(*f[0]);
}

std::shared_ptr<ov::Model> FrontEnd::decode(const InputModel::Ptr& model) const {
    auto paddle_model = std::dynamic_pointer_cast<InputModel>(model);
    FRONT_END_GENERAL_CHECK(paddle_model != nullptr, "Invalid input model");

    auto f = convert_each_node(paddle_model, paddle::make_framework_node);
    return f[0];
}

std::string FrontEnd::get_name() const {
    return "paddle";
}

void FrontEnd::add_extension(const std::shared_ptr<ov::Extension>& extension) {
    if (auto telemetry = std::dynamic_pointer_cast<TelemetryExtension>(extension)) {
        m_telemetry = telemetry;
    } else if (auto transformation = std::dynamic_pointer_cast<DecoderTransformationExtension>(extension)) {
        m_transformation_extensions.push_back(transformation);
    } else if (const auto& so_ext = std::dynamic_pointer_cast<ov::detail::SOExtension>(extension)) {
        add_extension(so_ext->extension());
        m_extensions.push_back(so_ext);
    } else if (auto common_conv_ext = std::dynamic_pointer_cast<ov::frontend::ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(common_conv_ext);
        m_op_translators[common_conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return common_conv_ext->get_converter_named()(context);
        };
    } else if (const auto& paddle_conv_ext = std::dynamic_pointer_cast<ConversionExtension>(extension)) {
        m_conversion_extensions.push_back(paddle_conv_ext);
        m_op_translators[paddle_conv_ext->get_op_type()] = [=](const NodeContext& context) {
            return paddle_conv_ext->get_converter()(context);
        };
    }
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov

PADDLE_C_API FrontEndVersion GetAPIVersion() {
    return OV_FRONTEND_API_VERSION;
}

PADDLE_C_API void* GetFrontEndData() {
    FrontEndPluginInfo* res = new FrontEndPluginInfo();
    res->m_name = "paddle";
    res->m_creator = []() {
        return std::make_shared<ov::frontend::paddle::FrontEnd>();
    };
    return res;
}

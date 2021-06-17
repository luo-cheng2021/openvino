// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/matrix_nms.hpp"
#include <cstring>
#include <ngraph/validation_util.hpp>
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/matrix_nms.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::v8::MatrixNms, "MatrixNms", 8);

op::v8::MatrixNms::MatrixNms(const Output<Node>& boxes,
                             const Output<Node>& scores,
                             const SortResultType sort_result_type,
                             const bool sort_result_across_batch,
                             const ngraph::element::Type& output_type,
                             const float score_threshold,
                             const int nms_top_k,
                             const int keep_top_k,
                             const int background_class,
                             const DecayFunction decay_function,
                             const float gaussian_sigma,
                             const float post_threshold)
    : NmsBase(boxes, scores, sort_result_type, output_type, nms_top_k, keep_top_k)
    , m_sort_result_across_batch{sort_result_across_batch}
    , m_score_threshold{score_threshold}
    , m_background_class{background_class}
    , m_decay_function{decay_function}
    , m_gaussian_sigma{gaussian_sigma}
    , m_post_threshold{post_threshold}
{
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::MatrixNms::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v8_MatrixNms_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2, "Number of inputs must be 2");

    return std::make_shared<op::v8::MatrixNms>(new_args.at(0),
                                               new_args.at(1),
                                               m_sort_result_type,
                                               m_sort_result_across_batch,
                                               m_output_type,
                                               m_score_threshold,
                                               m_nms_top_k,
                                               m_keep_top_k,
                                               m_background_class,
                                               m_decay_function,
                                               m_gaussian_sigma,
                                               m_post_threshold);
}

void op::v8::MatrixNms::validate()
{
    NmsBase::validate();

    NODE_VALIDATION_CHECK(this,
                          m_background_class >= -1,
                          "The 'background_class' must be great or equal -1. Got:",
                          m_background_class);
}

bool ngraph::op::v8::MatrixNms::visit_attributes(AttributeVisitor& visitor)
{
    NGRAPH_OP_SCOPE(v8_MatrixNms_visit_attributes);
    NmsBase::visit_attributes(visitor);

    visitor.on_attribute("sort_result_across_batch", m_sort_result_across_batch);
    visitor.on_attribute("score_threshold", m_score_threshold);
    visitor.on_attribute("background_class", m_background_class);
    visitor.on_attribute("decay_function", m_decay_function);
    visitor.on_attribute("gaussian_sigma", m_gaussian_sigma);
    visitor.on_attribute("post_threshold", m_post_threshold);

    return true;
}

bool ngraph::op::v8::MatrixNms::has_evaluate() const
{
    NGRAPH_OP_SCOPE(v8_MatrixNms_visit_attributes);
    switch (get_input_element_type(0))
    {
    case ngraph::element::f32: return true;
    default: break;
    }
    return false;
}

namespace
{
    std::vector<float> get_floats(const std::shared_ptr<HostTensor>& input, const Shape& shape)
    {
        size_t input_size = shape_size(shape);
        std::vector<float> result(input_size);

        switch (input->get_element_type())
        {
            case element::Type_t::bf16:
            {
                bfloat16* p = input->get_data_ptr<bfloat16>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = float(p[i]);
                }
            }
                break;
            case element::Type_t::f16:
            {
                float16* p = input->get_data_ptr<float16>();
                for (size_t i = 0; i < input_size; ++i)
                {
                    result[i] = float(p[i]);
                }
            }
                break;
            case element::Type_t::f32:
            {
                float* p = input->get_data_ptr<float>();
                memcpy(result.data(), p, input_size * sizeof(float));
            }
                break;
            default: throw std::runtime_error("Unsupported data type."); break;
        }

        return result;
    }

} // namespace

namespace matrix_nms_v8
{
    using SortResultType = op::v8::MatrixNms::SortResultType;
    struct InfoForNMS
    {
        Shape selected_outputs_shape;
        Shape selected_indices_shape;
        Shape boxes_shape;
        Shape scores_shape;
        std::vector<float> boxes_data;
        std::vector<float> scores_data;
        size_t selected_outputs_shape_size;
        size_t selected_indices_shape_size;
    };

    constexpr size_t boxes_port = 0;
    constexpr size_t scores_port = 1;

    PartialShape infer_selected_outputs_shape(
        const std::vector<std::shared_ptr<HostTensor>>& inputs, int nms_top_k, int keep_top_k)
    {
        const auto boxes_ps = inputs[boxes_port]->get_partial_shape();
        const auto scores_ps = inputs[scores_port]->get_partial_shape();

        PartialShape result = {Dimension::dynamic(), 6};

        if (boxes_ps.rank().is_static() && scores_ps.rank().is_static())
        {
            const auto num_boxes_boxes = boxes_ps[1];
            if (num_boxes_boxes.is_static() && scores_ps[0].is_static() && scores_ps[1].is_static())
            {
                const auto num_boxes = num_boxes_boxes.get_length();
                const auto num_classes = scores_ps[1].get_length();
                int64_t max_output_boxes_per_class = 0;
                if (nms_top_k >= 0)
                    max_output_boxes_per_class = std::min(num_boxes, (int64_t)nms_top_k);
                else
                    max_output_boxes_per_class = num_boxes;

                auto max_output_boxes_per_batch = max_output_boxes_per_class * num_classes;
                if (keep_top_k >= 0)
                    max_output_boxes_per_batch =
                        std::min(max_output_boxes_per_batch, (int64_t)keep_top_k);

                result[0] = max_output_boxes_per_batch * scores_ps[0].get_length();
            }
        }

        return result;
    }

    std::vector<float> prepare_boxes_data(const std::shared_ptr<HostTensor>& boxes,
                                          const Shape& boxes_shape)
    {
        auto result = get_floats(boxes, boxes_shape);
        return result;
    }

    std::vector<float> prepare_scores_data(const std::shared_ptr<HostTensor>& scores,
                                           const Shape& scores_shape)
    {
        auto result = get_floats(scores, scores_shape);
        return result;
    }

    InfoForNMS get_info_for_nms_eval(const op::v8::MatrixNms* nms,
                                     const std::vector<std::shared_ptr<HostTensor>>& inputs)
    {
        InfoForNMS result;

        auto selected_outputs_shape =
            infer_selected_outputs_shape(inputs, nms->get_nms_top_k(), nms->get_keep_top_k());
        result.selected_outputs_shape = selected_outputs_shape.to_shape();
        result.selected_indices_shape = {result.selected_outputs_shape[0], 1};

        result.boxes_shape = inputs[boxes_port]->get_shape();
        result.scores_shape = inputs[scores_port]->get_shape();

        result.boxes_data = prepare_boxes_data(inputs[boxes_port], result.boxes_shape);
        result.scores_data = prepare_scores_data(inputs[scores_port], result.scores_shape);

        result.selected_outputs_shape_size = shape_size(result.selected_outputs_shape);
        result.selected_indices_shape_size = shape_size(result.selected_indices_shape);

        return result;
    }
} // namespace matrix_nms_v8

bool ngraph::op::v8::MatrixNms::evaluate(const HostTensorVector& outputs,
                                         const HostTensorVector& inputs) const
{
    if (inputs[0]->get_element_type() != element::f32)
        return false;
    auto info = matrix_nms_v8::get_info_for_nms_eval(this, inputs);

    std::vector<float> selected_outputs(info.selected_outputs_shape_size);
    std::vector<int64_t> selected_indices(info.selected_indices_shape_size);
    std::vector<int64_t> valid_outputs(info.boxes_shape[0]);

    runtime::reference::matrix_nms(info.boxes_data.data(),
                                   info.boxes_shape,
                                   info.scores_data.data(),
                                   info.scores_shape,
                                   this->get_sort_result_type(),
                                   this->get_sort_result_across_batch(),
                                   this->get_score_threshold(),
                                   this->get_nms_top_k(),
                                   this->get_keep_top_k(),
                                   this->get_background_class(),
                                   this->get_decay_function(),
                                   this->get_gaussian_sigma(),
                                   this->get_post_threshold(),
                                   selected_outputs.data(),
                                   info.selected_outputs_shape,
                                   selected_indices.data(),
                                   info.selected_indices_shape,
                                   valid_outputs.data());

    runtime::reference::matrix_nms_postprocessing(
        outputs, this->get_output_type(), selected_outputs, selected_indices, valid_outputs);
    return true;
}

namespace ngraph
{
    template <>
    EnumNames<op::v8::MatrixNms::DecayFunction>& EnumNames<op::v8::MatrixNms::DecayFunction>::get()
    {
        static auto enum_names = EnumNames<op::v8::MatrixNms::DecayFunction>(
            "op::v8::MatrixNms::DecayFunction",
            {{"gaussian", op::v8::MatrixNms::DecayFunction::GAUSSIAN},
             {"linear", op::v8::MatrixNms::DecayFunction::LINEAR}});
        return enum_names;
    }

    constexpr DiscreteTypeInfo AttributeAdapter<op::v8::MatrixNms::DecayFunction>::type_info;

    std::ostream& operator<<(std::ostream& s, const op::v8::MatrixNms::DecayFunction& type)
    {
        return s << as_string(type);
    }
} // namespace ngraph

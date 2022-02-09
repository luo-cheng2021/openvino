// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/tensorarray_to_tensor.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "ngraph/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ov;

BWDCMP_RTTI_DEFINITION(op::internal::TensorArrayToTensor);

op::internal::TensorArrayToTensor::TensorArrayToTensor(const Output<Node>& arg0, const ConcatParam& param)
    : Op({arg0}),
    m_is_concat(true),
    m_concat_param(param) {
    constructor_validate_and_infer_types();
}

op::internal::TensorArrayToTensor::TensorArrayToTensor(const Output<Node>& arg0, const SliceParam& param)
    : Op({arg0}),
    m_is_concat(false),
    m_slice_param(param) {
    constructor_validate_and_infer_types();
}

op::internal::TensorArrayToTensor::TensorArrayToTensor(const Output<Node>& arg0, const bool is_concat, const ConcatParam& concat_param, const SliceParam& slice_param)
    : Op({arg0}),
    m_is_concat(is_concat),
    m_concat_param(concat_param),
    m_slice_param(slice_param) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::TensorArrayToTensor::clone_with_new_inputs(const OutputVector& new_args) const {
    auto clone = make_shared<TensorArrayToTensor>(new_args[0], m_is_concat, m_concat_param, m_slice_param);
    if (get_output_tensor(0).get_rt_info().count("TensorArray")) {
        clone->get_output_tensor(0).get_rt_info()["TensorArray"] = get_output_tensor(0).get_rt_info()["TensorArray"];
    }
    return clone;
}

bool op::internal::TensorArrayToTensor::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

void op::internal::TensorArrayToTensor::validate_and_infer_types() {
    auto ps = get_input_partial_shape(0);
    const auto type = get_input_element_type(0);
    if (m_is_concat) {
        if (ps.rank().is_static()) {
            // remove the 0 dim
            ov::PartialShape new_ps;
            new_ps.resize(ps.size() - 1);
            std::copy(ps.begin() + 1, ps.end(), new_ps.begin());
            new_ps[m_concat_param.axis] = ov::Dimension();
            ps = new_ps;
        }
    } else {
        if (ps.rank().is_static()) {
            // remove the 0 dim
            ov::PartialShape new_ps;
            new_ps.resize(ps.size() - 1);
            ps = new_ps;
        }
    }
    set_output_type(0, type, ps);
}

namespace {
std::vector<size_t> calculate_shape_sizes(const std::vector<Shape>& in_shapes) {
    std::vector<size_t> sizes;
    sizes.reserve(in_shapes.size());
    std::transform(begin(in_shapes), end(in_shapes), std::back_inserter(sizes), [](const Shape& shape) {
        return shape_size(shape);
    });
    return sizes;
}

void concat(const std::vector<const char*>& args,
            char* out,
            const std::vector<Shape>& in_shapes,
            const Shape& out_shape,
            int64_t concatenation_axis,
            size_t elem_size) {
    size_t steps = 1;
    for (int i = 0; i < concatenation_axis; ++i) {
        steps *= out_shape[i];
    }

    const auto& shape_sizes = calculate_shape_sizes(in_shapes);

    size_t out_offset = 0;
    for (size_t step = 0; step < steps; ++step) {
        for (size_t in_index = 0; in_index < args.size(); ++in_index) {
            const size_t size = shape_sizes[in_index] / steps;
            const size_t in_offset = step * size;

            std::memcpy(&out[out_offset * elem_size], &args[in_index][in_offset * elem_size], size * elem_size);

            out_offset += size;
        }
    }
}
}

bool op::internal::TensorArrayToTensor::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    const auto val = get_output_tensor(0).get_rt_info().at("TensorArray");
    const auto tensor_array = val.as<std::shared_ptr<std::vector<ov::runtime::Tensor>>>();
    outputs[0]->set_element_type(inputs[0]->get_element_type());
    if (m_is_concat) {
        Shape out_shape;
        auto input_shape = inputs[0]->get_partial_shape().get_shape();
        out_shape.resize(input_shape.size() - 1, 0);
        std::vector<const char*> srcs;
        std::vector<Shape> shapes_to_concat(tensor_array->size());
        // compute concat param
        for (auto i = 0; i < (*tensor_array).size(); i++) {
            const auto &tensor = (*tensor_array)[i];
            const auto& tensor_shape = tensor.get_shape();
            NODE_VALIDATION_CHECK(this,
                tensor_shape.size() == out_shape.size(),
                "rank of each tensor should be same, got: ",
                tensor_shape.size());
            if (i == 0) {
                out_shape = tensor_shape;
            } else {
                for (auto i = 0; i < out_shape.size(); i++) {
                    if (i != m_concat_param.axis) {
                        NODE_VALIDATION_CHECK(this,
                            tensor_shape[i] == out_shape[i],
                            "dim of each tensor should be same, got: ",
                            tensor_shape[i]);
                    }
                }
                out_shape[m_concat_param.axis] += tensor_shape[m_concat_param.axis];
            }
            shapes_to_concat[i] = tensor_shape;
            srcs.push_back(static_cast<char*>(tensor.data()));
        }
        outputs[0]->set_shape(out_shape);
        concat(srcs, static_cast<char*>(outputs[0]->get_data_ptr()),
            shapes_to_concat, out_shape, m_concat_param.axis, inputs[0]->get_element_type().size());

    } else {
        if (!tensor_array->empty()) {
            NODE_VALIDATION_CHECK(this,
                                m_slice_param.index < tensor_array->size(),
                                "slice param: ", m_slice_param.index,
                                " exceeds the max index, max index is: ",
                                tensor_array->size() - 1);
            Shape out_shape = (*tensor_array).at(m_slice_param.index).get_shape();
            outputs[0]->set_shape(out_shape);
            memcpy(outputs[0]->get_data_ptr(), (*tensor_array).at(m_slice_param.index).data(), outputs[0]->get_size_in_bytes());
        } else {
            Shape out_shape = get_input_partial_shape(0).get_min_shape();
            outputs[0]->set_shape(out_shape);
            outputs[0]->get_data_ptr();
        }
    }

    return true;
}

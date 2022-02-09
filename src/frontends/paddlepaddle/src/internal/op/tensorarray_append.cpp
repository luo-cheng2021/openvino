// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/tensorarray_append.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "ngraph/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

BWDCMP_RTTI_DEFINITION(op::internal::TensorArrayAppend);

op::internal::TensorArrayAppend::TensorArrayAppend(const Output<Node>& input, const Output<Node>& index)
    : Op({input, index}) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::internal::TensorArrayAppend::clone_with_new_inputs(const OutputVector& new_args) const {
    auto clone = make_shared<TensorArrayAppend>(new_args[0], new_args[1]);
    if (get_output_tensor(0).get_rt_info().count("TensorArray")) {
        clone->get_output_tensor(0).get_rt_info()["TensorArray"] = get_output_tensor(0).get_rt_info()["TensorArray"];
    }
    return clone;
}

bool op::internal::TensorArrayAppend::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

void op::internal::TensorArrayAppend::validate_and_infer_types() {
    auto ps = get_input_partial_shape(0);
    for (auto i = 0; i < ps.size(); i++) {
        ps[i] = ov::Dimension();
    }
    set_output_type(0, get_input_element_type(0), ps);
}

bool op::internal::TensorArrayAppend::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    // cache the data inside the runtime var
    ov::runtime::Tensor tensor(inputs[1]->get_element_type(), inputs[1]->get_shape());
    memcpy(tensor.data(), inputs[1]->get_data_ptr(), inputs[1]->get_size_in_bytes());
    const auto val = get_output_tensor(0).get_rt_info().at("TensorArray");
    const auto tensor_array = val.as<std::shared_ptr<std::vector<ov::runtime::Tensor>>>();
    tensor_array->emplace_back(tensor);

    // set the output type and shape, the data is not copied
    outputs[0]->set_element_type(inputs[0]->get_element_type());
    auto shape = inputs[0]->get_partial_shape().get_shape();
    shape[0] = tensor_array->size();    // make the dim[0] is exactly the size of the list
    outputs[0]->set_shape(shape);
    // force the buffer allocated
    outputs[0]->get_data_ptr();

    return true;
}

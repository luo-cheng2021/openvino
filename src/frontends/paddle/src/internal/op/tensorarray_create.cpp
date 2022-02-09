// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "internal/op/tensorarray_create.hpp"

#include <algorithm>
#include <ngraph/validation_util.hpp>

#include "ngraph/op/constant.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace std;
using namespace ov;

BWDCMP_RTTI_DEFINITION(op::internal::TensorArrayCreate);

op::internal::TensorArrayCreate::TensorArrayCreate(const Output<Node>& args0) : Op({args0}){
    constructor_validate_and_infer_types();
    get_output_tensor(0).get_rt_info()["TensorArray"] = std::make_shared<std::vector<ov::runtime::Tensor>>();
}

std::shared_ptr<Node> op::internal::TensorArrayCreate::clone_with_new_inputs(const OutputVector& new_args) const {
    auto clone = make_shared<TensorArrayCreate>(new_args[0]);
    if (get_output_tensor(0).get_rt_info().count("TensorArray")) {
        clone->get_output_tensor(0).get_rt_info()["TensorArray"] = get_output_tensor(0).get_rt_info()["TensorArray"];
    }
    return clone;
}

void op::internal::TensorArrayCreate::validate_and_infer_types() {
    auto ps = get_input_partial_shape(0);

    for (auto i = 0; i < ps.size(); i++) {
        ps[i] = ov::Dimension();
    }
    set_output_type(0, get_input_element_type(0), ps);
}

bool op::internal::TensorArrayCreate::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    outputs[0]->set_unary(inputs[0]);
    outputs[0]->write(inputs[0]->get_data_ptr(), outputs[0]->get_size_in_bytes());
    const auto val = get_output_tensor(0).get_rt_info().at("TensorArray");
    const auto tensor_array = val.as<std::shared_ptr<std::vector<ov::runtime::Tensor>>>();
    tensor_array->clear();
    return true;
}

bool op::internal::TensorArrayCreate::constant_fold(OutputVector& output_values, const OutputVector& inputs_values) {
    return false;
}
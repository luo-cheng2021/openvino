// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/ops/mha.hpp"

#include "openvino/op/op.hpp"
#include "openvino/op/mha.hpp"
#include "pyopenvino/core/common.hpp"
#include "../dict_attribute_visitor.hpp"

namespace py = pybind11;

void regclass_graph_op_MultiHeadAttention(py::module m) {
    using ov::op::internal::MultiHeadAttention;
    py::class_<MultiHeadAttention, std::shared_ptr<MultiHeadAttention>, ov::Node> cls(
        m,
        "_MultiHeadAttention");
    cls.doc() = "Experimental extention for MultiHeadAttention operation. Use with care: no backward compatibility is "
                "guaranteed in future releases.";

    cls.def(py::init([](const ov::OutputVector& inputs, const py::dict& attributes) {
            std::unordered_map<std::string, std::shared_ptr<ov::op::util::Variable>> variables;
            util::DictAttributeDeserializer visitor(attributes, variables);
            auto node = std::make_shared<MultiHeadAttention>();
            node->set_arguments(inputs);
            if (node->visit_attributes(visitor)) {
                node->constructor_validate_and_infer_types();
            }
            return node;
        }),
        py::arg("inputs"),
        py::arg("attributes"));
}

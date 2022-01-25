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

#include "../../default_opset.hpp"
#include "internal/op/conditional_block.hpp"
#include "internal/op/tensorarray_length.hpp"
#include "internal/op/while.hpp"
#include "internal/op/tensorarray_to_tensor.hpp"
#include "internal/op/tensorarray_write.hpp"
#include "internal/op/unary_dyn.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "paddlepaddle_frontend/exceptions.hpp"

using namespace std;
using namespace ov;
using namespace ov::pass;
using namespace opset8;

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
        const auto& new_item_unsqueeze = std::make_shared<Unsqueeze>(new_item->output(0), Constant::create(element::i32, {1}, {0}));
        // remove TensorArrayLength->TensorArrayWrite
        const auto concat = std::make_shared<Concat>(OutputVector{list->output(0), new_item_unsqueeze->output(0)}, 1/*HARDCODE axis*/); // TODO: which axis should be concat?
        const auto concat_dyn = std::make_shared<ov::op::internal::UnaryDyn>(concat);
        replace_node(write_node, concat_dyn);
        concat_dyn->set_friendly_name(write_node->get_friendly_name());

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(write_label, "tensorarray");
    this->register_matcher(m, callback);
}
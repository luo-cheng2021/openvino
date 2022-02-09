// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits.h>

#include <node_context.hpp>

#include "default_opset.hpp"
#include "internal/op/tensorarray_to_tensor.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
using namespace default_opset;
NamedOutputs slice(const NodeContext& node) {
    auto data = node.get_ng_input("Input");
    auto decrease_axis = node.get_attribute<std::vector<int32_t>>("decrease_axis");

    // // check if there are any TensorArray inputs.
    const auto inputs_names = node.get_input_var_names("Input");
    std::vector<TensorName> tensorarray_inputs;
    for (const auto& inputname : inputs_names) {
        // TODO: add support for 'StartsTensor'
        if (node.is_tensorarray(inputname, 1) && decrease_axis.size() == 1) {
            auto starts = node.get_attribute<std::vector<int32_t>>("starts");
            auto ends = node.get_attribute<std::vector<int32_t>>("ends");
            PDPD_OP_VALIDATION_CHECK(node,
                                     starts.size() == 1,
                                     "tensor array 'starts' size should be 1, got: ",
                                     starts.size());
            PDPD_OP_VALIDATION_CHECK(node,
                                     ends.size() == 1,
                                     "tensor array 'ends' size should be 1, got: ",
                                     ends.size());
            PDPD_OP_VALIDATION_CHECK(node,
                                     starts[0] + 1 == ends[0],
                                     "tensor array 'ends[0]' should equal ' starts[0] + 1', got starts[0], ends[0]: ",
                                     starts[0], ends[0]);
            ov::op::internal::TensorArrayToTensor::SliceParam param{starts[0]};
            auto placeholder = std::make_shared<ov::op::internal::TensorArrayToTensor>(data, param);

            return node.default_single_output_mapping({placeholder}, {"Out"});
        }
    }
    // if (tensorarray_inputs.size()>0) {
    //     auto start = Constant::create(element::i32, {1}, {0});
    //     auto stop = Constant::create(element::i32, {1}, {1});
    //     auto step = Constant::create(element::i32, {1}, {1});
    //     auto axes = Constant::create(element::i32, {1}, {0});
    //     const auto slice_node = std::make_shared<Slice>(data, start, stop, step, axes);
    //     return node.default_single_output_mapping({slice_node}, {"Out"});
    // }

    auto axes = node.get_attribute<std::vector<int32_t>>("axes");
    Output<Node> start_idx_node, end_idx_node;
    if (node.has_ng_input("StartsTensor")) {
        start_idx_node = node.get_ng_input("StartsTensor");
    } else if (node.has_ng_input("StartsTensorList")) {
        auto inputs = node.get_ng_inputs("StartsTensorList");
        start_idx_node = std::make_shared<Concat>(inputs, 0);
    } else {
        auto starts = node.get_attribute<std::vector<int32_t>>("starts");
        start_idx_node = Constant::create(element::i32, {starts.size()}, starts);
    }

    if (node.has_ng_input("EndsTensor")) {
        end_idx_node = node.get_ng_input("EndsTensor");
    } else if (node.has_ng_input("EndsTensorList")) {
        auto inputs = node.get_ng_inputs("EndsTensorList");
        end_idx_node = std::make_shared<Concat>(inputs, 0);
    } else {
        auto ends = node.get_attribute<std::vector<int32_t>>("ends");
        end_idx_node = Constant::create(element::i32, {ends.size()}, ends);
    }

    // The following process is:
    // Given:
    // data = [ [1, 2, 3, 4], [5, 6, 7, 8], ] // shape is: [2, 4]
    // axes = [0]
    // starts = [1]
    // ends = [2]
    // Our process is:
    //  1. Get 'axes': [0, 1], 'starts', 'ends'
    //  2. Get data shape: [2,4] and dims: 2
    //  3. Create two tensor t1 and t2, shape is the dims from step2: 2. t1: [0, 0], t2: [INT_MAX, INT_MAX]
    //  4. Use 'ScatterNDUpdate' to update some elements in t1, the updated indexes are coming from 'axes', the contents
    //  are coming from 'starts', t1: [1, 0]; apply the similar process to t2
    //  5. Call 'StrideSlice' with t1 and t2
    // Why using ScatterNDUpdate is that 'axes' may be discontinuous.

    // the shape of input, such as [2, 4]
    auto shape_node = std::make_shared<ShapeOf>(data, element::Type_t::i32);
    // the input dim, such as [2]
    auto shape_shape_node = std::make_shared<ShapeOf>(shape_node, element::i32);
    auto const_0_node = Constant::create(element::i32, {}, {0});
    auto const_max_node = Constant::create(element::i32, {}, {INT_MAX});
    // t1: [0, 0]
    auto start_node = std::make_shared<Broadcast>(const_0_node, shape_shape_node);
    // t2: [INT_MAX, INT_MAX]
    auto end_node = std::make_shared<Broadcast>(const_max_node, shape_shape_node);
    auto axes_node = Constant::create(element::i32, {axes.size(), 1}, axes);
    // update t1
    auto fixed_start_node = std::make_shared<ScatterNDUpdate>(start_node, axes_node, start_idx_node);
    // update t2
    auto fixed_end_node = std::make_shared<ScatterNDUpdate>(end_node, axes_node, end_idx_node);

    auto stride_slice_node = std::make_shared<StridedSlice>(data,
                                                            fixed_start_node,
                                                            fixed_end_node,
                                                            std::vector<int64_t>{0},
                                                            std::vector<int64_t>{0});

    if (decrease_axis.size() > 0) {
        // according to paddle slice_op, when all axes are decreased, output shape is [1], instead of scalar.
        // Ref: paddle/fluid/operators/slice_op.h
        PartialShape input_shape = data.get_partial_shape();
        PDPD_OP_VALIDATION_CHECK(node,
                                 input_shape.rank().is_static(),
                                 "input rank of slice must be static when decrease_axis is set.");

        auto squeeze_index_node = Constant::create(element::i32, {decrease_axis.size()}, decrease_axis);
        auto decreased_node = std::make_shared<Squeeze>(stride_slice_node, squeeze_index_node);

        auto input_rank = input_shape.rank().get_length();
        if (input_rank == decrease_axis.size()) {
            auto restore_node = std::make_shared<Reshape>(decreased_node,
                                                          std::make_shared<Constant>(element::i64, Shape{1}, 1),
                                                          false);  // restore to shape (1,)
            return node.default_single_output_mapping({restore_node}, {"Out"});
        }

        return node.default_single_output_mapping({decreased_node}, {"Out"});
    }

    return node.default_single_output_mapping({stride_slice_node}, {"Out"});
}
}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov

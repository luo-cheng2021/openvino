// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs multiclass_nms(const NodeContext& node) {
    using namespace default_opset;
    using namespace element;

    auto bboxes = node.get_input("BBoxes");
    auto scores = node.get_input("Scores");

    // a new style of parameters:
    // https://github.com/PaddlePaddle/PaddleDetection/blob/417537702bf8a8b25b41d48651fff0a0b1c8a4b2/ppdet/modeling/ops.py#L873
    const auto scores_ps = scores.get_partial_shape();
    if (scores_ps.rank().is_static() && scores_ps.rank().get_length() == 2) {
        // bboxes: [M, C, 4] -> [N=1, M(M*C), 4]
        // scores: [M, C] -> [N=1, C, M(M*C)]
        bboxes = std::make_shared<Reshape>(bboxes, Constant::create(element::i32, {3}, {1, -1, 4}), false);
        scores = std::make_shared<Transpose>(scores, Constant::create(element::i32, {2}, {1, 0}));
        // TODO: remove static shape
        const auto scores_1d = std::make_shared<Reshape>(scores, Constant::create(element::i32, {1}, {-1}), false);
        const auto shape = std::make_shared<Concat>(
            NodeVector{Constant::create(element::i32, {2}, std::vector<int64_t>{1, scores_ps[1].get_length()}),
                       std::make_shared<ShapeOf>(scores_1d, element::i32)},
            0);
        scores = std::make_shared<Broadcast>(scores_1d, shape);
    }

    auto score_threshold = node.get_attribute<float>("score_threshold");
    auto iou_threshold = node.get_attribute<float>("nms_threshold");
    auto nms_top_k = node.get_attribute<int>("nms_top_k");
    auto keep_top_k = node.get_attribute<int>("keep_top_k");
    auto background_class = node.get_attribute<int>("background_label");
    auto nms_eta = node.get_attribute<float>("nms_eta");

    auto out_names = node.get_output_names();
    PADDLE_OP_CHECK(node, out_names.size() == 3, "Unexpected number of outputs of MulticlassNMS");

    auto type_index = node.get_out_port_type("Index");
    auto type_num = node.get_out_port_type("NmsRoisNum");
    PADDLE_OP_CHECK(node,
                    (type_index == i32 || type_index == i64) && (type_num == i32 || type_num == i64),
                    "Unexpected data type of outputs of MulticlassNMS: " + std::to_string(out_names.size()));

    auto normalized = node.get_attribute<bool>("normalized");

    NamedOutputs named_outputs;
    std::vector<Output<Node>> nms_outputs;
    MulticlassNms::Attributes attrs;
    attrs.nms_top_k = nms_top_k;
    attrs.iou_threshold = iou_threshold;
    attrs.score_threshold = score_threshold;
    attrs.sort_result_type = MulticlassNms::SortResultType::CLASSID;
    attrs.keep_top_k = keep_top_k;
    attrs.background_class = background_class;
    attrs.nms_eta = nms_eta;
    attrs.normalized = normalized;
    attrs.output_type = type_index;
    attrs.sort_result_across_batch = false;

    nms_outputs = std::make_shared<MulticlassNms>(bboxes, scores, attrs)->outputs();

    named_outputs["Out"] = {nms_outputs[0]};
    named_outputs["Index"] = {nms_outputs[1]};
    named_outputs["NmsRoisNum"] = {nms_outputs[2]};

    if (type_num != type_index) {
        // adapter
        auto node_convert = std::make_shared<Convert>(nms_outputs[2], type_num);
        named_outputs["NmsRoisNum"] = {node_convert};
    }

    return named_outputs;
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov

// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_fusion.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/x64/op/rope.hpp"
#include "utils/gen_pattern.hpp"

#ifdef CPU_DEBUG_CAPS
#define CALLBACK_LOG(m) std::cout << matcher_name << " " << m.get_match_root()->get_friendly_name() << std::endl;
#else
#define CALLBACK_LOG(m)
#endif

namespace ov {
namespace intel_cpu {

class RoPEFusionGPTNEOX : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionGPTNEOX", "0");
    RoPEFusionGPTNEOX() {
        MATCHER_SCOPE(RoPEFusionGPTNEOX);

        // rope pattern matching triggers a little design flaw:
        //   y1 = mul(x, cos)
        //   y2 = mul(x, sin)
        //   y = add(y1, y2)
        // there is a chance that in 'y1' branch, pattern x is mapped to actual value of cos (mul is commutable)
        // this leads to the matching failure of 'y2' branch, because cos didn't appear in that
        // branch.
        // so here we use a WA, only match the path of rotate_hal(x)*sin and check the x*cos path
        // in the callback
        auto x = GenPattern(ov::Rank(4));
        auto x_or_cos1 = GenPattern(ov::Rank(4));
        auto x_or_cos2 = GenPattern(ov::Rank(4));
        auto t_sin = GenPattern(ov::Rank(4));

        x->set_friendly_name("x");

        auto half_ndims = Symbol("half_ndims");
        auto int32_max = std::numeric_limits<std::int32_t>::max();

        // rotate half : [-x2, x1]
        auto x2 = GenSlice(x, half_ndims, int32_max, 1, 3);
        auto x2neg = GenPattern<opset1::Multiply>({x2, {-1}}, nullptr, {{"auto_broadcast", "numpy"}});
        auto x1 = GenSlice(x, 0, half_ndims, 1, 3);
        auto x_rotate_half = GenPattern<opset1::Concat>({x2neg, x1}, nullptr, {{"axis", -1}});

        auto mul_cos = GenPattern<opset1::Multiply>({x_or_cos1, x_or_cos2}, nullptr, {{"auto_broadcast", "numpy"}});
        auto mul_sin = GenPattern<opset1::Multiply>({x_rotate_half, t_sin}, nullptr, {{"auto_broadcast", "numpy"}});

        // [x1, x2]*cos + [-x2, x1]*sin
        auto result = GenPattern<opset1::Add>({mul_cos, mul_sin}, nullptr, {{"auto_broadcast", "numpy"}});

        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            CALLBACK_LOG(m);

            const auto& pattern_map = m.get_pattern_value_map();
            auto root = m.get_match_root();
            std::map<std::string, double> symbol_name2value;
            if (!validate_matched_symbols(m, symbol_name2value)) {
                return false;
            }

            // check mul(x, cos) exists
            Output<Node> v_cos;
            if (pattern_map.at(x_or_cos1) == pattern_map.at(x)) {
                v_cos = pattern_map.at(x_or_cos2);
            } else if (pattern_map.at(x_or_cos2) == pattern_map.at(x)) {
                v_cos = pattern_map.at(x_or_cos1);
            } else {
                // not a RoPE
                return false;
            }

            RoPENode::Config config;
            OutputVector new_args;
            config.ndims = 2 * symbol_name2value["half_ndims"];

            new_args.push_back(pattern_map.at(x));
            new_args.push_back(v_cos);
            new_args.push_back(pattern_map.at(t_sin));

            auto old_node = root;
            auto new_node = std::make_shared<RoPENode>(new_args, config);
            new_node->set_friendly_name(old_node->get_friendly_name());
            ov::replace_node(old_node, new_node);

            // this new node may match following additional matchers
            register_new_node(new_node);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
        this->register_matcher(m, callback);
    }
};

// only a fraction of head_size is rotary-embedded
class RoPEFusionIOSlicing : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionIOSlicing", "0");

    RoPEFusionIOSlicing() {
        MATCHER_SCOPE(RoPEFusionIOSlicing);
        auto int32_max = std::numeric_limits<std::int32_t>::max();
        auto data = GenPattern(ov::Rank(4));

        auto ndims = Symbol("ndims");
        auto x = GenSlice(data, 0, ndims, 1, 3);
        auto y = GenSlice(data, ndims, int32_max, 1, 3);
        auto x_emb = GenPattern<RoPENode>({x, {}, {}}) | GenPattern<RoPENode>({x, {}, {}, {}});
        auto result = GenPattern<opset1::Concat>({x_emb, y}, nullptr, {{"axis", -1}});

        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            CALLBACK_LOG(m);

            const auto& pattern_map = m.get_pattern_value_map();
            auto root = m.get_match_root();

            auto rope_node = as_type_ptr<RoPENode>(root->input_value(0).get_node_shared_ptr());
            if (!rope_node) {
                return false;
            }

            std::map<std::string, double> symbol_name2value;
            if (!validate_matched_symbols(m, symbol_name2value)) {
                return false;
            }
            auto ndims = symbol_name2value["ndims"];

            auto& config = rope_node->get_config();
            if (config.ndims != 0 && config.ndims != ndims) {
                return false;
            }

            config.ndims = ndims;

            // remove slice & concat
            rope_node->set_argument(0, pattern_map.at(data));
            rope_node->set_friendly_name(root->get_friendly_name());
            ov::replace_node(root, rope_node);

            rope_node->validate_and_infer_types();

            register_new_node(rope_node);
            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
        this->register_matcher(m, callback);
    }
};

class RoPEFusionPreprocess : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionPreprocess", "0");
    RoPEFusionPreprocess() {
        MATCHER_SCOPE(RoPEFusionPreprocess);

        // gptneox-preprocess of input data
        auto input_to_reshape = GenPattern(ov::Rank(3));
        auto input_to_slice = GenPattern(ov::Rank(4));
        auto input_to_trans = GenPattern(ov::Rank(4));  // no need to slice from 3S

        // input needs to be reshaped from 3d to 4d (by split last dim into [H,X])
        auto head_cnt = Symbol("H");
        auto head_size_mx = Symbol("X");

        // reshape split last dimension into 2
        auto shape_3d = GenPattern("i32[3]");
        auto shape_BL = GenPattern<opset8::Gather>({shape_3d, {0, 1}, {0}}, "i32[2]", {{"batch_dims", 0}});
        auto shape_BLHS = GenPattern<opset1::Concat>({shape_BL, {head_cnt}, {head_size_mx}}, "i32[4]", {{"axis", 0}});
        auto input_4d = GenPattern<opset1::Reshape>({input_to_reshape, shape_BLHS}, nullptr, {{"special_zero", 0}});

        // in some model qkv prejection is combined, slice one of them out before RoPE
        auto slice_start = Symbol("slice_start");
        auto slice_stop = Symbol("slice_stop");
        auto input_slice = GenSlice(input_4d | input_to_slice, slice_start, slice_stop, 1, 3);

        // some model will transpose from [B,L,H,S] to [B,H,L,S] before RoPE
        auto input_trans = GenPattern<opset1::Transpose>({input_slice | input_4d | input_to_trans, {0, 2, 1, 3}});
        auto x = input_trans;
        auto result = GenPattern<RoPENode>({x, {}, {}}) | GenPattern<RoPENode>({x, {}, {}, {}});

        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            CALLBACK_LOG(m);

            const auto& pattern_map = m.get_pattern_value_map();
            auto root = m.get_match_root();
            auto rope_node = as_type_ptr<RoPENode>(root);
            if (!rope_node)
                return false;
            std::map<std::string, double> symbol_name2value;
            if (!validate_matched_symbols(m, symbol_name2value)) {
                return false;
            }
            auto& config = rope_node->get_config();

            if (pattern_map.count(input_slice)) {
                config.slice_start = symbol_name2value["slice_start"];
                config.slice_stop = symbol_name2value["slice_stop"];
            }
            if (pattern_map.count(input_trans))
                config.input_trans0213 = true;

            if (pattern_map.count(input_to_reshape)) {
                config.reshape_H = symbol_name2value["H"];
                rope_node->set_argument(0, pattern_map.at(input_to_reshape));
            } else if (pattern_map.count(input_to_slice)) {
                rope_node->set_argument(0, pattern_map.at(input_to_slice));
            } else if (pattern_map.count(input_to_trans)) {
                rope_node->set_argument(0, pattern_map.at(input_to_trans));
            } else {
                return false;
            }
            rope_node->validate_and_infer_types();
            register_new_node(rope_node);
            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
        this->register_matcher(m, callback);
    }
};

class RoPEFusionPostprocess : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionPostprocess", "0");

    RoPEFusionPostprocess() {
        MATCHER_SCOPE(RoPEFusionPostprocess);

        // gptneox-preprocess of input data
        auto rope3 = GenPattern<RoPENode>({{}, {}, {}});
        auto rope4 = GenPattern<RoPENode>({{}, {}, {}, {}});
        auto output_trans = GenPattern<opset1::Transpose>({rope3 | rope4, {0, 2, 1, 3}});
        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            CALLBACK_LOG(m);
            auto root = m.get_match_root();
            auto rope_node = as_type_ptr<RoPENode>(root->input_value(0).get_node_shared_ptr());
            if (!rope_node)
                return false;
            auto& config = rope_node->get_config();
            config.output_trans0213 = true;
            ov::replace_node(root, rope_node);
            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(output_trans, matcher_name);
        this->register_matcher(m, callback);
    }
};

class RoPEFusionConcatPast : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionConcatPast", "0");
    RoPEFusionConcatPast();
};

class RoPEFusionCosSinPreprocess : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionCosSinPreprocess", "0");

    RoPEFusionCosSinPreprocess() {
        MATCHER_SCOPE(RoPEFusionCosSinPreprocess);

        auto cos_const = GenPattern<opset1::Constant>({});  // "f32[1,1,2048,24]"
        auto sin_const = GenPattern<opset1::Constant>({});  // "f32[1,1,2048,24]"

        auto node_batch_size = GenPattern("i32[1]");
        auto tile_batch = GenPattern("i32[1]");
        auto gather_positions = GenPattern("i32[?,?,?,?]");
        auto gather_positions_2d = GenPattern("i32[?,?]");

        auto tile_repeats = GenPattern<opset1::Concat>({tile_batch, {1}, {1}, {1}}, "i32[4]", {{"axis", 0}});

        auto prepare_cos_sin_gptneox = [&](std::shared_ptr<Node> const_tab) {
            auto slice1 = GenPattern<opset1::StridedSlice>({const_tab, {0}, node_batch_size, {1}},
                                                           nullptr,
                                                           {{"begin_mask", {0}},
                                                            {"end_mask", {0}},
                                                            {"new_axis_mask", {}},
                                                            {"shrink_axis_mask", {}},
                                                            {"ellipsis_mask", {}}});  //"f32[..1,1,2048,24]",
            auto slice2 = GenPattern<opset8::Slice>({const_tab, {0}, node_batch_size, {1}, {0}});
            auto tiled = GenPattern<opset1::Tile>({slice1 | slice2, tile_repeats});  // "f32[?,1,2048,24]"
            return GenPattern<opset6::GatherElements>({tiled, gather_positions}, nullptr, {{"axis", 2}});
        };

        auto prepare_cos_sin_llama = [&](std::shared_ptr<Node> const_tab) {
            auto seq_len = GenPattern("i32[1]");
            auto Slice_1 = GenPattern<opset8::Slice>({const_tab, {0}, seq_len, {1}, {2}});  // "f32[1, 1,..2048,128]"
            auto Squeeze_2 = GenPattern<opset1::Squeeze>({Slice_1, {1}});                   // "f32[1,..2048,128]"
            auto Squeeze_3 = GenPattern<opset1::Squeeze>({Squeeze_2, {0}});                 // "f32[..2048,128]"
            auto Gather_8 = GenPattern<opset8::Gather>({Squeeze_3, gather_positions_2d, 0},
                                                       nullptr,
                                                       {{"batch_dims", 0}});    // "f32[B,L,128]"
            auto Unsqueeze_7 = GenPattern<opset1::Unsqueeze>({Gather_8, {1}});  //  "f32[?,1,?,128]"
            return Unsqueeze_7;                                                 // B,H,L,S (H is broadcasted)
        };

        auto prepare_cos_sin_llama_v2 = [&](std::shared_ptr<Node> const_tab) {
            auto slice_end = GenPattern("i32[3]");
            auto slice_Slice = GenPattern<opset1::StridedSlice>({const_tab, {0, 0, 0}, slice_end, {1, 1, 1}},
                                                                nullptr,
                                                                {{"begin_mask", {1, 1, 0}},
                                                                 {"end_mask", {1, 1, 0}},
                                                                 {"new_axis_mask", {}},
                                                                 {"shrink_axis_mask", {}},
                                                                 {"ellipsis_mask", {}}});
            auto squeeze_1 = GenPattern<opset1::Reshape>({slice_Slice, {-1, 128}}, nullptr, {{"special_zero", 0}});
            auto gather =
                GenPattern<opset8::Gather>({squeeze_1, gather_positions_2d, {0}}, nullptr, {{"batch_dims", 0}});
            auto unsqueeze = GenPattern<opset1::Unsqueeze>({gather, {1}});
            return unsqueeze;
        };

        auto cos_tab =
            prepare_cos_sin_gptneox(cos_const) | prepare_cos_sin_llama(cos_const) | prepare_cos_sin_llama_v2(cos_const);
        auto sin_tab =
            prepare_cos_sin_gptneox(sin_const) | prepare_cos_sin_llama(sin_const) | prepare_cos_sin_llama_v2(sin_const);

        auto x = GenPattern(ov::Rank(4));
        auto rope = GenPattern<RoPENode>({x, cos_tab, sin_tab});

        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            CALLBACK_LOG(m);
            const auto& pattern_map = m.get_pattern_value_map();
            auto root = m.get_match_root();
            auto rope_node = as_type_ptr<RoPENode>(pattern_map.at(rope).get_node_shared_ptr());
            if (!rope_node)
                return false;

            if (pattern_map.count(cos_const)) {
                rope_node->set_argument(1, pattern_map.at(cos_const));
            }
            if (pattern_map.count(sin_const)) {
                rope_node->set_argument(2, pattern_map.at(sin_const));
            }

            auto& config = rope_node->get_config();
            if (pattern_map.count(gather_positions)) {
                auto arg_id = rope_node->get_input_size();
                rope_node->set_argument(arg_id, pattern_map.at(gather_positions));
                config.gather_position_arg_id = arg_id;
            } else if (pattern_map.count(gather_positions_2d)) {
                auto arg_id = rope_node->get_input_size();
                rope_node->set_argument(arg_id, pattern_map.at(gather_positions_2d));
                config.gather_position_arg_id = arg_id;
            }
            register_new_node(rope_node);
            return true;
        };
        auto m = std::make_shared<ngraph::pattern::Matcher>(rope, matcher_name);
        this->register_matcher(m, callback);
    }
};

// remove stridedslice from 0 to int32_max with stride 1
class EliminateStridedSlice : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("EliminateStridedSlice", "0");
    EliminateStridedSlice() {
        MATCHER_SCOPE(EliminateStridedSlice);
        auto data = ov::pass::pattern::any_input(ngraph::pattern::has_static_rank());
        auto begin = ov::pass::pattern::wrap_type<opset1::Constant>(ngraph::pattern::type_matches(ov::element::i32));
        auto end = ov::pass::pattern::wrap_type<opset1::Constant>(ngraph::pattern::type_matches(ov::element::i32));
        auto stride = ov::pass::pattern::wrap_type<opset1::Constant>(ngraph::pattern::type_matches(ov::element::i32));

        auto strided_slice = ov::pass::pattern::wrap_type<opset1::StridedSlice>(
            {data, begin, end, stride},
            [](const Output<Node>& value) {
                auto s1 = as_type_ptr<opset1::StridedSlice>(value.get_node_shared_ptr());
                if (!s1->get_new_axis_mask().empty() || !s1->get_shrink_axis_mask().empty() ||
                    !s1->get_ellipsis_mask().empty()) {
                    return false;
                }

                auto inputs = s1->input_values();

                auto begin = as_type_ptr<opset1::Constant>(inputs[1].get_node_shared_ptr());
                auto end = as_type_ptr<opset1::Constant>(inputs[2].get_node_shared_ptr());
                // stride is all 1
                auto stride = as_type_ptr<opset1::Constant>(inputs[3].get_node_shared_ptr());

                if (!begin)
                    return false;
                if (!end)
                    return false;
                if (!stride)
                    return false;

                auto v_stride = stride->cast_vector<int32_t>();
                for (auto& v : v_stride) {
                    if (v != 1)
                        return false;
                }

                auto v_begin = begin->cast_vector<int32_t>();
                auto v_end = end->cast_vector<int32_t>();

                auto& begin_mask = s1->get_begin_mask();
                auto& end_mask = s1->get_end_mask();
                auto mask_size = begin_mask.size();
                if (begin_mask.size() != end_mask.size()) {
                    return false;
                }

                auto int32_max = std::numeric_limits<std::int32_t>::max();
                for (size_t i = 0; i < mask_size; i++) {
                    if (begin_mask[i] != end_mask[i])
                        return false;
                    // all valid [begin, end] are [0, int32_max]
                    if (begin_mask[i] == 0 && end_mask[i] == 0) {
                        if (v_begin[i] != 0 || v_end[i] != int32_max)
                            return false;
                    }
                }
                return true;
            });

        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            CALLBACK_LOG(m);
            auto root = m.get_match_root();
            return replace_output_update_name(root->output(0), root->input_value(0));
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(strided_slice, matcher_name);
        this->register_matcher(m, callback);
    }
};

RoPEFusionConcatPast::RoPEFusionConcatPast() {
    MATCHER_SCOPE(RoPEFusionConcatPast);

    auto x = GenPattern(ov::Rank(4));
    auto past = GenPattern(ov::Rank(4));
    auto rope = GenPattern<RoPENode>({x, {}, {}}) | GenPattern<RoPENode>({x, {}, {}, {}});
    auto result = GenPattern<opset1::Concat>({past, rope}, nullptr, {{"axis", -2}}) |
                  GenPattern<opset1::Concat>({past, rope}, nullptr, {{"axis", 2}});

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto rope_node = as_type_ptr<RoPENode>(root->input_value(1).get_node_shared_ptr());
        if (!rope_node)
            return false;
        auto& config = rope_node->get_config();
        if (config.concat_with_past_arg_id)
            return false;

        // append past to input args
        auto arg_id = rope_node->get_input_size();
        rope_node->set_argument(arg_id, pattern_map.at(past));
        config.concat_with_past_arg_id = arg_id;

        rope_node->set_friendly_name(root->get_friendly_name());
        ov::replace_node(root, rope_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

template <typename T>
std::shared_ptr<Node> InterleaveDuplicateConst() {
    auto pnode = std::make_shared<GenericPattern>();
    pnode->set_predicate([](const Output<Node>& value) {
        auto s1 = as_type_ptr<opset1::Constant>(value.get_node_shared_ptr());
        if (!s1) {
            _VERBOSE_LOG("*mismatched InterleaveDuplicateConst op type: opset1::Constant vs", value);
            return false;
        }

        // ignore higher dimensions, require lowerst 2D to be lower triangular
        auto shape = s1->get_output_shape(0);
        if (shape.size() != 1) {
            _VERBOSE_LOG("*unexpected InterleaveDuplicateConst shape : ", value);
            return false;
        }

        // NxN const matrix
        bool values_matched = true;
        auto N = shape[0];
        std::vector<T> output_vector = s1->cast_vector<T>();
        for (size_t i = 0; i < N; i += 2) {
            auto expected = i / 2;
            if (output_vector[i] != expected || output_vector[i + 1] != expected) {
                values_matched = false;
                break;
            }
        }
        if (!values_matched) {
            _VERBOSE_LOG("*mismatched InterleaveDuplicateConst values : ", value);
        }

        return values_matched;
    });
    return pnode;
}

class RoPEFusionGPTJ : public ngraph::pass::MatcherPass {
public:
    OPENVINO_RTTI("RoPEFusionGPTJ", "0");
    RoPEFusionGPTJ() {
        MATCHER_SCOPE(RoPEFusionGPTJ);

        auto linear_MatMul_90 = GenPattern("f32[?,?,?]");       // B, L, H*S
        auto gather_GatherElements = GenPattern("f32[?,?,?]");  // B, L, rotary_ndims cos/sin table

        auto head_cnt = Symbol("head_cnt");
        auto head_size = Symbol("head_size");
        auto rotary_ndims = Symbol("rotary_ndims");

        auto ShapeOf_182700 = GenPattern<opset1::ShapeOf>({linear_MatMul_90}, "i32[3]");
        auto Gather_97571 = GenPattern<opset8::Gather>({ShapeOf_182700, {0, 1}, {0}}, "i32[2]", {{"batch_dims", 0}});
        auto shape4d_BLHS =
            GenPattern<opset1::Concat>({Gather_97571, {head_cnt}, {head_size}}, "i32[4]", {{"axis", 0}});

        auto view_Reshape_105 =
            GenPattern<opset1::Reshape>({linear_MatMul_90, shape4d_BLHS}, "f32[?,?,16,256]", {{"special_zero", 0}});

        auto slice_Slice_156 = GenSlice(view_Reshape_105, 0, rotary_ndims, 1, 3);
        auto ListUnpack_135_VariadicSplit =
            GenPattern<opset1::VariadicSplit>({gather_GatherElements, {-1}, {rotary_ndims / 2, -1}},
                                              "f32[?,?,32] f32[?,?,32]");
        auto unsqueeze_Unsqueeze_230 =
            GenPattern<opset1::Unsqueeze>({ListUnpack_135_VariadicSplit->output(1), {2}}, "f32[?,?,1,32]");
        auto repeat_interleave_Gather_251 =
            GenPattern<opset8::Gather>({unsqueeze_Unsqueeze_230, InterleaveDuplicateConst<int32_t>(), {3}},
                                       "f32[?,?,1,64]",
                                       {{"batch_dims", 0}});
        auto mul_Cos = GenPattern<opset1::Multiply>({slice_Slice_156, repeat_interleave_Gather_251},
                                                    "f32[?,?,16,64]",
                                                    {{"auto_broadcast", "numpy"}});

        // [x0, -x1,    x2, -x3,    x4, -x5, ....]
        auto slice_Slice_281 = GenSlice(slice_Slice_156, 1, 2147483647, 2, 3);
        auto neg_Multiply =
            GenPattern<opset1::Multiply>({slice_Slice_281, {-1.0f}}, "f32[?,?,16,32]", {{"auto_broadcast", "numpy"}});
        auto Unsqueeze_65504 = GenPattern<opset1::Unsqueeze>({neg_Multiply, {-1}}, "f32[?,?,16,32,1]");
        auto slice_Slice_275 = GenSlice(slice_Slice_156, 0, 2147483647, 2, 3);
        auto Unsqueeze_65505 = GenPattern<opset1::Unsqueeze>({slice_Slice_275, {-1}}, "f32[?,?,16,32,1]");
        auto stack_285 =
            GenPattern<opset1::Concat>({Unsqueeze_65504, Unsqueeze_65505}, "f32[?,?,16,32,2]", {{"axis", -1}});
        auto ShapeOf_182921 = GenPattern<opset1::ShapeOf>({stack_285}, "i32[5]");
        auto flatten_Slice = GenSlice(ShapeOf_182921, 0, 3, 1, 0);
        auto flatten_Concat = GenPattern<opset1::Concat>({flatten_Slice, {-1}}, "i32[4]", {{"axis", 0}});
        auto flatten_Reshape =
            GenPattern<opset1::Reshape>({stack_285, flatten_Concat}, "f32[?,?,16,?]", {{"special_zero", 1}});

        // *sin
        auto unsqueeze_Unsqueeze_205 =
            GenPattern<opset1::Unsqueeze>({ListUnpack_135_VariadicSplit->output(0), {2}}, "f32[?,?,1,32]");
        auto repeat_interleave_Gather_217 =
            GenPattern<opset8::Gather>({unsqueeze_Unsqueeze_205, InterleaveDuplicateConst<int32_t>(), {3}},
                                       "f32[?,?,1,64]",
                                       {{"batch_dims", 0}});
        auto mul_Sin = GenPattern<opset1::Multiply>({flatten_Reshape, repeat_interleave_Gather_217},
                                                    "f32[?,?,16,64]",
                                                    {{"auto_broadcast", "numpy"}});

        auto add_Add = GenPattern<opset1::Add>({mul_Cos, mul_Sin}, "f32[?,?,16,64]", {{"auto_broadcast", "numpy"}});

        // concat with non-rotary embedded dimensions
        auto slice_Slice_162 = GenSlice(view_Reshape_105, rotary_ndims, 2147483647, 1, 3);
        auto cat_Concat = GenPattern<opset1::Concat>({add_Add, slice_Slice_162}, "f32[?,?,16,256]", {{"axis", -1}});
        auto permute_Transpose_434 = GenPattern<opset1::Transpose>({cat_Concat, {0, 2, 1, 3}}, "f32[?,16,?,256]");
        auto result = permute_Transpose_434;

        matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
            CALLBACK_LOG(m);

            const auto& pattern_map = m.get_pattern_value_map();
            auto root = m.get_match_root();
            std::map<std::string, double> symbol_name2value;
            if (!validate_matched_symbols(m, symbol_name2value)) {
                return false;
            }

            RoPENode::Config config;
            OutputVector new_args;
            config.reshape_H = symbol_name2value["head_cnt"];
            config.ndims = symbol_name2value["rotary_ndims"];
            config.is_interleaved = true;
            config.output_trans0213 = true;
            config.is_cos_sin_combined = true;

            new_args.push_back(pattern_map.at(linear_MatMul_90));
            new_args.push_back(pattern_map.at(gather_GatherElements));

            auto old_node = root;
            auto new_node = std::make_shared<RoPENode>(new_args, config);
            new_node->set_friendly_name(old_node->get_friendly_name());
            ov::replace_node(old_node, new_node);

            // this new node may match following additional matchers
            register_new_node(new_node);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(result, matcher_name);
        this->register_matcher(m, callback);
    }
};

RoPEFusion::RoPEFusion() {
#ifdef CPU_DEBUG_CAPS
    if (!std::getenv("USE_ROPE") || !atoi(std::getenv("USE_ROPE")))
        return;
    std::cout << "[USE_ROPE] RoPEFusion is enabled" << std::endl;
#endif
    add_matcher<EliminateStridedSlice>();
    add_matcher<RoPEFusionGPTNEOX>();

    // RoPEFusionGPTJ is a very special type requires its own specilization
    add_matcher<RoPEFusionGPTJ>();

    add_matcher<RoPEFusionCosSinPreprocess>();
    add_matcher<RoPEFusionIOSlicing>();
    add_matcher<RoPEFusionPreprocess>();
    add_matcher<RoPEFusionPostprocess>();
    // add_matcher<RoPEFusionConcatPast>();
}

}  // namespace intel_cpu
}  // namespace ov
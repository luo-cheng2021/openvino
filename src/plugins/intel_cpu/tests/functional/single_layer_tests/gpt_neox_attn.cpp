/// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include <openvino/opsets/opset10.hpp>
#include "ngraph/type/bfloat16.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace {
    std::vector<InputShape> inputShape;
}  // namespace
using GPTNeoxAttnSpecificParams = std::tuple<
        std::vector<InputShape>>;        // pooled vector

using GPTNeoxAttnLayerTestParams = std::tuple<
        GPTNeoxAttnSpecificParams,
        ElementType,         // Net precision
        size_t,              // head number
        size_t,              // size per head
        size_t,              // past key number
        size_t,              // max seq length
        float,               // rotary_pct
        TargetDevice>;       // Device name

using GPTNeoxAttnCPUTestParamsSet = std::tuple<
        CPULayerTestsDefinitions::GPTNeoxAttnLayerTestParams,
        CPUSpecificParams>;

class GPTNeoxAttnCPUTest : public testing::WithParamInterface<GPTNeoxAttnCPUTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<GPTNeoxAttnCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::GPTNeoxAttnLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPr;
        GPTNeoxAttnSpecificParams shapes;
        size_t head_num;
        size_t size_per_head;
        size_t past_key_number;
        size_t max_seq_length;
        float rotary_pct;

        std::tie(shapes, netPr, head_num, size_per_head, past_key_number, max_seq_length, rotary_pct, td) = basicParamsSet;
        std::tie(inputShape) = shapes;
        std::ostringstream result;
        result << "GPTNeoxAttnTest_";
        result << "IS=(";
        for (const auto& shape : inputShape) {
            result << CommonTestUtils::partialShape2str({shape.first}) << "_";
        }
        result << ")_TS=(";
        for (const auto& shape : inputShape) {
            for (const auto& item : shape.second) {
                result << CommonTestUtils::vec2str(item) << "_";
            }
        }
        result << netPr << "_";
        result << head_num << "_";
        result << size_per_head << "_";
        result << past_key_number << "_";
        result << max_seq_length << "_";
        result << rotary_pct << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << std::to_string(obj.index);
        return result.str();
    }
protected:
    size_t head_num;
    size_t size_per_head;
    size_t past_key_number;
    size_t max_seq_length;
    float rotary_pct;
    ElementType netPrecision;

    void SetUp() override {
        CPULayerTestsDefinitions::GPTNeoxAttnLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::GPTNeoxAttnSpecificParams GPTNeoxAttnParams;

        std::tie(GPTNeoxAttnParams, netPrecision, head_num, size_per_head, past_key_number, max_seq_length, rotary_pct, targetDevice) = basicParamsSet;
        std::tie(inputShape) = GPTNeoxAttnParams;
        if (netPrecision == ElementType::bf16) {
            selectedType = "unknown_FP32";
            rel_threshold = 1.f;
            configuration[InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16] = InferenceEngine::PluginConfigParams::YES;
        }

        init_input_shapes(inputShape);

        selectedType = std::string("unknown_FP32");
        function = createFunction();
    }

    std::shared_ptr<ngraph::Function> createFunction() {
        auto input_ids = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputDynamicShapes[0]);
        input_ids->set_friendly_name("input_ids");
        auto num = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, inputDynamicShapes[1]);
        num->set_friendly_name("past_num");

        auto gpt = std::make_shared<ov::opset10::GPTNeoxAttn>(input_ids, num, 32, head_num, size_per_head,
            head_num * size_per_head, 2048, 10000, rotary_pct, max_seq_length);
        gpt->set_friendly_name("gpt");
        gpt->get_rt_info() = getCPUInfo();

        auto function = std::make_shared<ov::Model>(gpt->outputs(), ov::ParameterVector{input_ids, num}, "gpt");
        return function;
    }
    // pattern is:
    //                                           qkv_rotary:[batch, query_seq_len, num_heads * 3 * head_size]
    //                                                                |
    //                                           Reshape0:[batch, query_seq_len, num_heads, 3 * head_size]
    //                                                                |
    //                                                  Split:[batch, query_seq_len, num_heads, head_size] * 3
    //                                                   |            |                                                        |
    // Transpose-query[batch, num_heads, query_seq_len, head_size]  Transpose-key[batch, num_heads, query_seq_len, head_size] Transpose-value
    //    |                                                           |                                                        |
    //    |                                                     Concat_pastkey:[batch, num_heads, key_seq_len, head_size]     Concat_pastvalue
    //    \                                                           |                                                        |
    //     \                                           Transpose0: [batch, num_heads, head_size, key_seq_len]                 to_value
    //      \                                                         /
    //       \                                                       /
    //        \                                                     /
    //        MatMul0: [batch, num_heads, query_seq_len, key_seq_len]
    //          |
    //          |   norm_factor(const): [1]
    //          |       /
    //       Multiply: [batch, num_heads, query_seq_len, key_seq_len]
    //          |
    //          |   causal_mask: [1, 1, query_seq_len, key_seq_len]
    //          |       /
    //       Select(only for 1x300): [batch, num_heads, query_seq_len, key_seq_len]
    //          |
    //          |   attention_mask:[batch, 1, 1, key_seq_len]
    //          |       /
    //       Add: [batch, num_heads, query_seq_len, key_seq_len]
    //          |
    //       SoftMax: [batch, num_heads, query_seq_len, key_seq_len]
    //          |
    //           \  value:[batch, num_heads, key_seq_len, head_size]
    //            \     /
    //             MatMul1: [batch, num_heads, query_seq_len, head_size]
    //               |
    //            Transpose1(only for 1x300): [batch, query_seq_len, num_heads * head_size]
    void init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) override {
        using namespace ov;
        auto batch = targetInputStaticShapes[0][0];
        auto query_seq_len = targetInputStaticShapes[0][1];

        // [batch, query_seq_len, head_num * 3 * head_size]
        auto input_ids = std::make_shared<ngraph::opset1::Parameter>(netPrecision, targetInputStaticShapes[0]);
        input_ids->set_friendly_name("input_ids");
        std::shared_ptr<ov::Node> input_ids_convert = input_ids;
        if (netPrecision == ngraph::element::bf16) {
            input_ids_convert = std::make_shared<opset10::Convert>(input_ids, ngraph::element::f32);
        }
        auto num = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, targetInputStaticShapes[1]);
        num->set_friendly_name("past_num");
        // [batch, query_seq_len, head_num, 3 * head_size]
        auto qkv_reshape = std::make_shared<opset10::Reshape>(input_ids_convert, opset10::Constant::create(element::i32, Shape{4},
            {batch, query_seq_len, head_num, 3 * size_per_head}), false);
        // [batch, query_seq_len, head_num, head_size] * 3
        auto axis = opset10::Constant::create(element::i32, {}, {-1});
        auto split = std::make_shared<opset10::Split>(qkv_reshape, axis, 3);
        auto query = split->output(0);
        auto key = split->output(1);
        auto value = split->output(2);
        // [batch, head_num, query_seq_len, head_size]
        auto query_transpose = std::make_shared<opset10::Transpose>(query, opset10::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
        // [batch, head_num, query_seq_len+past_key_num, head_size]
        auto key_transpose = std::make_shared<opset10::Transpose>(key, opset10::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
        auto value_transpose = std::make_shared<opset10::Transpose>(value, opset10::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
        if (past_key_number) {
            // concat query_seq_len+past_key
            assert(false);
        }
        // [batch, head_num, query_seq_len, query_seq_len+past_key_num]
        auto matmul0 = std::make_shared<opset10::MatMul>(query_transpose, key_transpose, false, true);
        auto norm_factor = opset10::Constant::create(element::f32, Shape{1}, {1.0f / std::sqrt(size_per_head)});
        auto multiply = std::make_shared<opset10::Multiply>(matmul0, norm_factor);
        std::vector<uint8_t> mask(query_seq_len * (query_seq_len + past_key_number));
        for (auto k = 0; k < query_seq_len; k++) {
            for (auto l = 0; l < query_seq_len + past_key_number; l++) {
                mask[k * (query_seq_len + past_key_number) + l] = (l <= k + past_key_number) ? 1 : 0;
            }
        }
        // [1, 1, query_seq_len, query_seq_len + past_key_number]
        auto causal_mask = std::make_shared<opset10::Constant>(element::boolean, Shape{1, 1, query_seq_len, query_seq_len + past_key_number}, mask);
        auto minus = std::make_shared<opset10::Constant>(element::f32, Shape{1}, -FLT_MAX);
        auto select = std::make_shared<opset10::Select>(causal_mask, multiply, minus);
        // [batch, 1, 1, query_seq_len+past_key_num]
        auto attention_mask = std::make_shared<opset10::Constant>(element::f32, Shape{batch, 1, 1, query_seq_len + past_key_number}, 0.0f);
        auto add = std::make_shared<opset10::Add>(select, attention_mask);
        // [batch, head_num, query_seq_len, query_seq_len + past_key_number]
        auto softmax = std::make_shared<opset10::Softmax>(add, -1);
        // [batch, head_num, query_seq_len, head_size]
        auto matmul1 = std::make_shared<opset10::MatMul>(softmax, value_transpose);
        // [batch, query_seq_len, head_num, head_size]
        auto transpose1 = std::make_shared<opset10::Transpose>(matmul1, opset10::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
        // [batch, query_seq_len, head_num * head_size]
        auto reshape1 = std::make_shared<opset10::Reshape>(transpose1, opset10::Constant::create(element::i32, Shape{3},
            {batch, query_seq_len, head_num * size_per_head}), false);
        funcRef = std::make_shared<ov::Model>(NodeVector{reshape1}, ParameterVector{input_ids, num});

        ngraph::helpers::resize_function(funcRef, targetInputStaticShapes);
    }

    void validate() override {
        auto actualOutputs = get_plugin_outputs();
        auto expectedOutputs = calculate_refs();
        if (expectedOutputs.empty()) {
            return;
        }
        ASSERT_EQ(actualOutputs.size(), expectedOutputs.size())
                << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

        compare(expectedOutputs, actualOutputs);
    }
    void initRotery(size_t max_position_embeddings, size_t rotary_dims, std::vector<float>& cos_cached, std::vector<float>& sin_cached) {
        std::vector<float> inv_freq;
        for (size_t i = 0; i < rotary_dims; i += 2) {
            inv_freq.push_back(1.0f / (powf(10000, static_cast<float>(i) / rotary_dims)));
        }
        std::vector<float> t;
        for (size_t i = 0; i < max_position_embeddings * 2; i++) {
            t.push_back(static_cast<float>(i));
        }
        auto width = rotary_dims / 2 * 2;
        auto height = max_position_embeddings * 2;
        cos_cached.resize(height * width);
        sin_cached.resize(height * width);
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width / 2; j++) {
                cos_cached[i * width + j] = cosf(t[i] * inv_freq[j]);
                cos_cached[i * width + j + width / 2] = cosf(t[i] * inv_freq[j]);
                sin_cached[i * width + j] = sinf(t[i] * inv_freq[j]);
                sin_cached[i * width + j + width / 2] = sinf(t[i] * inv_freq[j]);
            }
        }
    }
    // qkv:[batch, query_seq_len, num_heads * 3 * head_size]
    void rotary(float* qkv, size_t batch, size_t seq_len, size_t offset, size_t head_num, size_t size_per_head, int rotary_dims) {
        std::vector<float> cos_cached;
        std::vector<float> sin_cached;
        initRotery(2048, rotary_dims, cos_cached, sin_cached);
        int half_rotary_dims = rotary_dims / 2;
        auto* query = qkv;
        std::vector<float> copy_q(rotary_dims), copy_k(rotary_dims);
        for (size_t m = 0; m < batch; m ++) {
            float* cos = &cos_cached[offset * rotary_dims];
            float* sin = &sin_cached[offset * rotary_dims];
            auto q_batch = query + m * head_num * seq_len * size_per_head * 3;
            for (size_t n = 0; n < seq_len; n++) {
                auto q_seq = q_batch + n * head_num * size_per_head * 3;
                auto k_seq = q_seq + size_per_head;
                for (size_t j = 0; j < head_num; j++) {
                    memcpy(copy_q.data(), q_seq, rotary_dims * sizeof(float));
                    memcpy(copy_k.data(), k_seq, rotary_dims * sizeof(float));
                    for (size_t i = 0; i < half_rotary_dims; i++) {
                        q_seq[i] = copy_q[i] * cos[i] - copy_q[i + half_rotary_dims] * sin[i];
                        k_seq[i] = copy_k[i] * cos[i] - copy_k[i + half_rotary_dims] * sin[i];
                    }
                    for (size_t i = half_rotary_dims; i < rotary_dims; i++) {
                        q_seq[i] = copy_q[i] * cos[i] + copy_q[i - half_rotary_dims] * sin[i];
                        k_seq[i] = copy_k[i] * cos[i] + copy_k[i - half_rotary_dims] * sin[i];
                    }
                    q_seq += size_per_head * 3;
                    k_seq += size_per_head * 3;
                }
                cos += rotary_dims;
                sin += rotary_dims;
            }
        }
    }
    std::map<std::shared_ptr<ov::Node>, ov::Tensor> inputs_ref;
    std::vector<ov::Tensor> calculate_refs() override {
        inputs = inputs_ref;
        return SubgraphBaseTest::calculate_refs();
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (int i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;

            if (i == 1) {
                tensor = ov::Tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                auto *dataPtr = tensor.data<int32_t>();
                dataPtr[0] = past_key_number;
                inputs_ref.insert({funcInput.get_node_shared_ptr(), tensor});
            } else {
                tensor = ov::test::utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], 2560, 0, 256);
                if (funcInput.get_element_type() == ov::element::f32) {
                    auto *dataSrcPtr = tensor.data<float>();
                    ov::Tensor tensor_ref(tensor.get_element_type(), tensor.get_shape());
                    auto *dataDstPtr = tensor_ref.data<float>();
                    memcpy(dataDstPtr, dataSrcPtr, tensor.get_byte_size());
                    rotary(dataDstPtr, targetInputStaticShapes[0][0], targetInputStaticShapes[0][1], past_key_number,
                        head_num, size_per_head, static_cast<int>(size_per_head * rotary_pct));
                    inputs_ref.insert({funcInput.get_node_shared_ptr(), tensor_ref});
                } else {
                    std::vector<ov::bfloat16> bf16Tmp(tensor.get_size());
                    auto *dataSrcPtr = tensor.data<ov::bfloat16>();
                    memcpy(bf16Tmp.data(), dataSrcPtr, tensor.get_byte_size());
                    auto f32Tmp = ov::bfloat16::to_float_vector(bf16Tmp);
                    rotary(f32Tmp.data(), targetInputStaticShapes[0][0], targetInputStaticShapes[0][1], past_key_number,
                        head_num, size_per_head, static_cast<int>(size_per_head * rotary_pct));
                    auto bf16Rotary = ov::bfloat16::from_float_vector(f32Tmp);
                    ov::Tensor tensor_ref(tensor.get_element_type(), tensor.get_shape());
                    auto *dataDstPtr = tensor_ref.data<ov::bfloat16>();
                    memcpy(dataDstPtr, bf16Rotary.data(), tensor.get_byte_size());
                    inputs_ref.insert({funcInput.get_node_shared_ptr(), tensor_ref});
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_P(GPTNeoxAttnCPUTest, CompareWithRefs) {
    if (!InferenceEngine::with_cpu_x86_avx512_core())
        GTEST_SKIP();

    run();
    //CheckPluginRelatedResults(compiledModel, "GPTNeoxAttn");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    //resCPUParams.push_back(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}});  // i.e. two equal output layouts
    //resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc, ncdhw}, {}, {}});
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{abc, x}, {abc}, {}, {}});
    }

    return resCPUParams;
}

const std::vector<ElementType> netPrecisions = {
    ElementType::f32,
    ElementType::bf16
};

std::vector<std::vector<ov::Shape>> staticInputShapeVector = {{{2, 300, 7680}, {1}}, {{2, 301, 7680}, {1}}};

const auto staticGPTNeoxAttnParams = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapeVector))      // feature map shape
);

INSTANTIATE_TEST_SUITE_P(smoke_GPTNeoxAttnTest, GPTNeoxAttnCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticGPTNeoxAttnParams,
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(32),         // head number
                                         ::testing::Values(80),         // size per head
                                         ::testing::Values(0),          // past key number
                                         ::testing::Values(400),        // max seq length
                                         ::testing::Values(0.25f),      // rotary_pct
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice())),
                         GPTNeoxAttnCPUTest::getTestCaseName);


} // namespace
} // namespace CPULayerTestsDefinitions

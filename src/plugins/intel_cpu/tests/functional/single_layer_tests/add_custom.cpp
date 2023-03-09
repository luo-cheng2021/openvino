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
using AddCustomSpecificParams = std::tuple<
        std::vector<InputShape>>;        // pooled vector

using AddCustomLayerTestParams = std::tuple<
        AddCustomSpecificParams,
        ElementType,         // Net precision
        TargetDevice>;       // Device name

using AddCustomCPUTestParamsSet = std::tuple<
        CPULayerTestsDefinitions::AddCustomLayerTestParams,
        CPUSpecificParams>;

class AddCustomCPUTest : public testing::WithParamInterface<AddCustomCPUTestParamsSet>,
                            virtual public SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<AddCustomCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::AddCustomLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        std::string td;
        ElementType netPr;
        AddCustomSpecificParams shapes;

        std::tie(shapes, netPr, td) = basicParamsSet;
        std::tie(inputShape) = shapes;
        std::ostringstream result;
        result << "AddCustomTest_";
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
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << std::to_string(obj.index);
        return result.str();
    }
protected:
    ElementType netPrecision;

    void SetUp() override {
        CPULayerTestsDefinitions::AddCustomLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::AddCustomSpecificParams AddCustomParams;

        std::tie(AddCustomParams, netPrecision, targetDevice) = basicParamsSet;
        std::tie(inputShape) = AddCustomParams;
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
        auto node0 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputDynamicShapes[0]);
        node0->set_friendly_name("node0");
        auto node1 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputDynamicShapes[1]);
        node1->set_friendly_name("node1");
        auto node2 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputDynamicShapes[2]);
        node2->set_friendly_name("node2");

        auto add = std::make_shared<ov::opset10::AddCustom>(node0, node1, node2);
        add->set_friendly_name("addcustom");
        add->get_rt_info() = getCPUInfo();

        auto function = std::make_shared<ov::Model>(add->outputs(), ov::ParameterVector{node0, node1, node2}, "add");
        return function;
    }
    void init_ref_function(std::shared_ptr<ov::Model> &funcRef, const std::vector<ov::Shape>& targetInputStaticShapes) override {
        using namespace ov;
        auto node0 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputDynamicShapes[0]);
        node0->set_friendly_name("node0");
        auto node1 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputDynamicShapes[1]);
        node1->set_friendly_name("node1");
        auto node2 = std::make_shared<ngraph::opset1::Parameter>(netPrecision, inputDynamicShapes[2]);
        node2->set_friendly_name("node2");

        auto add0 = std::make_shared<ov::opset10::Add>(node0, node1);
        add0->set_friendly_name("add0");
        add0->get_rt_info() = getCPUInfo();
        auto add1 = std::make_shared<ov::opset10::Add>(add0, node2);
        add1->set_friendly_name("add1");
        add1->get_rt_info() = getCPUInfo();

        funcRef = std::make_shared<ov::Model>(add1->outputs(), ov::ParameterVector{node0, node1, node2}, "add");
        ngraph::helpers::resize_function(funcRef, targetInputStaticShapes);
    }
};

TEST_P(AddCustomCPUTest, CompareWithRefs) {
    if (!InferenceEngine::with_cpu_x86_avx512_core())
        GTEST_SKIP();
    if (!InferenceEngine::with_cpu_x86_bfloat16() && netPrecision == ElementType::bf16)
        GTEST_SKIP();
    run();
    //CheckPluginRelatedResults(compiledModel, "AddCustom");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice() {
    std::vector<CPUSpecificParams> resCPUParams;
    //resCPUParams.push_back(CPUSpecificParams{{ncdhw, x}, {ncdhw}, {}, {}});  // i.e. two equal output layouts
    //resCPUParams.push_back(CPUSpecificParams{{ndhwc, x}, {ndhwc, ncdhw}, {}, {}});
    if (with_cpu_x86_avx512f()) {
        resCPUParams.push_back(CPUSpecificParams{{abc, abc, abc}, {abc}, {}, {}});
    }

    return resCPUParams;
}

const std::vector<ElementType> netPrecisions = {
    //ElementType::f32,
    ElementType::bf16
};

std::vector<std::vector<ov::Shape>> staticInputShapeVector = {{{2, 1, 2560}, {2, 1, 2560}, {2, 1, 2560}},
    {{2, 301, 2560}, {2, 301, 2560}, {2, 301, 2560}}};

const auto staticAddCustomParams = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(staticInputShapeVector))      // feature map shape
);

INSTANTIATE_TEST_SUITE_P(smoke_AddCustomTest, AddCustomCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         staticAddCustomParams,
                                         ::testing::ValuesIn(netPrecisions),
                                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice())),
                         AddCustomCPUTest::getTestCaseName);


} // namespace
} // namespace CPULayerTestsDefinitions

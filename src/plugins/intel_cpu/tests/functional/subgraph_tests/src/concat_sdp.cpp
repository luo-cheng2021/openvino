// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/opsets/opset13.hpp>

#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ngraph;

namespace SubgraphTestsDefinitions {
// Subgraph:
/*                            Parameter
 *                                |
 *       Parameter    ReadValue   |    ReadValue  Parameter
 *           \           /        |       \          /
 *            \         /         |        \        /
 *               Concat           |          Concat
 *                / \             |            / \
 *               /   \            |           /   \
 *              /     \           |          /     \
 *          Assign     ScaledDotProductAttention  Assign
 *                                |
 *                              Result
 */

class ConcatSDPTest : virtual public LayerTestsUtils::LayerTestsCommon {
public:
    void SetUp() override {
        const std::vector<size_t> qkvShape = {1, 8, 0, 64};
        const std::vector<size_t> pastKVShape = {1, 8, 1, 64};
        auto pastk = std::make_shared<ov::op::v3::ReadValue>(
            ngraph::builder::makeConstant<float>(element::Type_t::f32, pastKVShape, {}, true),
            "pastk");
        auto pastv = std::make_shared<ov::op::v3::ReadValue>(
            ngraph::builder::makeConstant<float>(element::Type_t::f32, pastKVShape, {}, true),
            "pastv");
        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(qkvShape)),
                                        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(qkvShape)),
                                        std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(qkvShape))};
        auto concatK = builder::makeConcat(OutputVector{pastk, inputParams[1]}, 2);
        auto concatV = builder::makeConcat(OutputVector{pastv, inputParams[2]}, 2);
        auto sdp = std::make_shared<ov::opset13::ScaledDotProductAttention>(inputParams[0], concatK, concatV, false);
        auto pastk_assign = std::make_shared<op::v3::Assign>(concatK, "pastk");
        auto pastv_assign = std::make_shared<op::v3::Assign>(concatV, "pastv");
        // pastk_assign->add_control_dependency(pastk);
        // pastv_assign->add_control_dependency(pastv);

        ResultVector results{std::make_shared<ov::op::v0::Result>(sdp)};
        SinkVector sinks{pastk_assign, pastv_assign};
        function = std::make_shared<Function>(results, sinks, inputParams, "ConcatSDP");
        targetDevice = ov::test::utils::DEVICE_CPU;
    }
};

namespace {
TEST_F(ConcatSDPTest, smoke_ConcatSDP_CPU) {
    Run();
}
}  // namespace
}  // namespace SubgraphTestsDefinitions

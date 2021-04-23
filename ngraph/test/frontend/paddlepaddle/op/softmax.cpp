// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using softmaxTestParam = FrontendOpTestParam;
using softmaxTest = FrontendOpTest;

static softmaxTestParam softmax() {
    softmaxTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "softmax"; //TODO: compact model/decomposited

    //Inputs inputs;
    res.inputs.emplace_back(test::NDArray<float, 4>{{{{2.0, 3.0, 4.0, 5.0 },
                                                    {3.0, 4.0, 5.0, 6.0 },
                                                    {7.0, 8.0, 8.0, 9.0 }},
                                                    {{1.0, 2.0, 3.0, 4.0 },
                                                    {5.0, 6.0, 7.0, 8.0 },
                                                    {6.0, 7.0, 8.0, 9.0 }}}}
                            .get_vector());

    // 
    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{0.0065732626, 0.0065732626, 0.017147826, 0.017147826 },
                                                                {0.01786798, 0.01786798, 0.04661262, 0.04661262 },
                                                                {0.9755587, 0.9755587, 0.93623954, 0.93623954 }},
                                                                {{0.004901689, 0.004901689, 0.004901689, 0.004901689 },
                                                                {0.26762316, 0.26762316, 0.26762316, 0.26762316 },
                                                                {0.72747517, 0.72747517, 0.72747517, 0.72747517 }}}})
                               .get_vector());

    return res;
}

TEST_P(softmaxTest, test_softmax) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, softmaxTest,
                        ::testing::Values(
                            softmax()
                        ),                        
                        softmaxTest::getTestCaseName);                                                 

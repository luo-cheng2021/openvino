// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using matmulTestParam = FrontendOpTestParam;
using matmulTest = FrontendOpTest;

static matmulTestParam matmul() {
    matmulTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "matmul"; //TODO: compact model/decomposited

    //Inputs inputs;
    // data (1, 1, 7, 5) input tensor
    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                   {5.f, 6.f, 7.f, 8.f, 9.f},
                                                   {10.f, 11.f, 12.f, 13.f, 14.f},
                                                   {15.f, 16.f, 17.f, 18.f, 19.f},
                                                   {20.f, 21.f, 22.f, 23.f, 24.f},
                                                   {25.f, 26.f, 27.f, 28.f, 29.f},
                                                   {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                            .get_vector());

    // (1, 1, 4, 3)
    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{12.f, 27.f, 24.f},
                                                      {63.f, 108.f, 81.f},
                                                      {123.f, 198.f, 141.f},
                                                      {112.f, 177.f, 124.f}}}})
                               .get_vector());

    return res;
}

TEST_P(matmulTest, test_matmul) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, matmulTest,
                        ::testing::Values(
                            matmul()
                        ),                        
                        matmulTest::getTestCaseName);                                                 

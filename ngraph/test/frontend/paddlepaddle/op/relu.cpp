// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using reluTestParam = FrontendOpTestParam;
using reluTest = FrontendOpTest;

static reluTestParam relu() {
    reluTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "relu"; //TODO: compact model/decomposited

    //Inputs inputs;
    res.inputs.emplace_back(test::NDArray<float, 2>{{-2.0, 0.0, 1.0 }}
                            .get_vector());

    // 
    res.expected_outputs.emplace_back(test::NDArray<float, 2>({{0.0, 0.0, 1.0 }})
                               .get_vector());

    return res;
}

TEST_P(reluTest, test_relu) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, reluTest,
                        ::testing::Values(
                            relu()
                        ),                        
                        reluTest::getTestCaseName);                                                 

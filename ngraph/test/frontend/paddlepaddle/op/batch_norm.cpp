// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using batchNormTestParam = FrontendOpTestParam;
using batchNormTest = FrontendOpTest;

static batchNormTestParam batchNorm() {
    batchNormTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "batch_norm";  //TODO: compact model/decomposited

    //Inputs inputs;
    // data (1, 1, 4, 4) input tensor
    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{-1.0, 0.0, 1.0 }},
                                                       {{2.0, 3.0, 4.0 }}}}}
                              .get_vector());

    // (1, 1, 2, 2)
    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{-0.999995, 0.0, 0.999995 }},
                                                                 {{-0.22474074, 1.0, 2.2247407 }}}}})
                               .get_vector());

    return res;
}

TEST_P(batchNormTest, test_batchNorm) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, batchNormTest,
                        ::testing::Values(
                            batchNorm()
                        ),                        
                        batchNormTest::getTestCaseName);                                               

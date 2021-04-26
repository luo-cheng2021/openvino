// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using bmmTestParam = FrontendOpTestParam;
using bmmTest = FrontendOpTest;

static bmmTestParam bmm() {
    bmmTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "paddle_bmm";

    res.inputs.emplace_back(test::NDArray<float, 3>{{{{1., 1., 1., 1., 1., 1., 1.},
                                                             {1., 1., 1., 1., 1., 1., 1.},
                                                             {1., 1., 1., 1., 1., 1., 1.},
                                                             {1., 1., 1., 1., 1., 1., 1.},
                                                             {1., 1., 1., 1., 1., 1., 1.}}}}.get_vector());

    res.inputs.emplace_back(test::NDArray<float, 3>{{{{0., 1., 2., 3., 4.},
                                                    {5., 6., 7., 8., 9.},
                                                    {10., 11., 12., 13., 14.},
                                                    {15., 16., 17., 18., 19.},
                                                    {20., 21., 22., 23., 24.},
                                                    {25., 26., 27., 28., 29.},
                                                    {30., 31., 32., 33., 34.}}}}.get_vector());



    res.expected_outputs.emplace_back(test::NDArray<float, 3>{{{{ 10.,  10.,  10.,  10.,  10.,  10.,  10.},
                                                                { 35.,  35.,  35.,  35.,  35.,  35.,  35.},
                                                                { 60.,  60.,  60.,  60.,  60.,  60.,  60.},
                                                                { 85.,  85.,  85.,  85.,  85.,  85.,  85.},
                                                                {110., 110., 110., 110., 110., 110., 110.},
                                                                {135., 135., 135., 135., 135., 135., 135.},
                                                                {160., 160., 160., 160., 160., 160., 160.}}}}.get_vector());
    return res;
}

TEST_P(bmmTest, test_bmm) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, bmmTest,
                        ::testing::Values(
                                bmm()
                        ),
                        bmmTest::getTestCaseName);

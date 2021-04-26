// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using interpolateParam = FrontendOpTestParam;
using interpolateTest = FrontendOpTest;

static interpolateParam bilinear_downsample_false_1() {
    interpolateParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "bilinear_downsample_false_1.pdmodel";


    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f},
                                                              {9.f, 10.f, 11.f, 12.f},
                                                              {13.f, 14.f, 15.f, 16.f}}}}}.get_vector());

    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{1.f, 3.f, 5.f, 7.f},
                                                                        {9.f, 11.f, 13.f, 15.f}}}}).get_vector());
    return res;
}

static interpolateParam bilinear_downsample_false_0() {
    interpolateParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "bilinear_downsample_false_0";


    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.f, 2.f, 3.f, 4.f},
                                                       {5.f, 6.f, 7.f, 8.f},
                                                       {9.f, 10.f, 11.f, 12.f},
                                                       {13.f, 14.f, 15.f, 16.f}}}}}.get_vector());

    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{1.5f, 3.5f, 5.5f, 7.5f},
                                                                 {9.5f, 11.5f, 13.5f, 15.5f}}}}).get_vector());
    return res;
}

static interpolateParam bilinear_downsample_true_0() {
    interpolateParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "bilinear_downsample_true_0";


    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f},
                                                              {9.f, 10.f, 11.f, 12.f},
                                                              {13.f, 14.f, 15.f, 16.f}}}}}.get_vector());

    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{1.0f, 3.333333f, 5.6666665f, 8.0f},
                                                                        {9.0f, 11.33333f, 13.666666f, 16.0f}}}}).get_vector());
    return res;
}

static interpolateParam bilinear_upsample_false_1() {
    interpolateParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "bilinear_upsample_false_1";


    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f},
                                                              {9.f, 10.f, 11.f, 12.f},
                                                              {13.f, 14.f, 15.f, 16.f}}}}}.get_vector());

    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{1.0f, 1.5f, 2.f, 2.5f, 3.f, 3.5f, 4.f, 4.f},
                                                                        {3.f, 3.5f, 4.f, 4.5f, 5.f, 5.5f, 6.f, 6.f},
                                                                        {5.f, 5.5f, 6.f, 6.5f, 7.f, 7.5f, 8.f, 8.f},
                                                                        {7.f, 7.5f, 8.f, 8.5f, 9.f, 9.5f, 10.f, 10.f},
                                                                        {9.f, 9.5f, 10.f, 10.5f, 11.f, 11.5f, 12.f, 12.f},
                                                                        {11.f, 11.5f, 12.f, 12.5f, 13.f, 13.5f, 14.f, 14.f},
                                                                        {13, 13.5f, 14.f, 14.5f, 15.f, 15.5f, 16.f, 16.f},
                                                                        {13., 13.5f, 14.f, 14.5f, 15.f, 15.5f, 16.f, 16.f}}}}).get_vector());
    return res;
}

static interpolateParam bilinear_upsample_false_0() {
    interpolateParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "bilinear_upsample_false_0";


    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f},
                                                              {9.f, 10.f, 11.f, 12.f},
                                                              {13.f, 14.f, 15.f, 16.f}}}}}.get_vector());

    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{1.0f, 1.25f, 1.75f, 2.25f, 2.75f, 3.25f, 3.75f, 4.f},
                                                                        {2.f, 2.25f, 2.75f, 3.25f, 3.75f, 4.25f, 4.75f, 5.f},
                                                                        {4.f, 4.25f, 4.75f, 5.25f, 5.75f, 6.25f, 6.75f, 7.f},
                                                                        {6.f, 6.25f, 6.75f, 7.25f, 7.75f, 8.25f, 8.75f, 9.f},
                                                                        {8.f, 8.25f, 8.75f, 9.25f, 9.75f, 10.25f, 10.75f, 11.f},
                                                                        {10.f, 10.25f, 10.75f, 11.25f, 11.75f, 12.25f, 12.75f, 13.f},
                                                                        {12, 12.25f, 12.75f, 13.25f, 13.75f, 14.25f, 14.75f, 15.f},
                                                                        {13., 13.25f, 13.75f, 14.25f, 14.75f, 15.25f, 15.75f, 16.f}}}}).get_vector());
    return res;
}

static interpolateParam paddle_bilinear_upsample_true_0() {
    interpolateParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "bilinear_upsample_true_0";


    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f},
                                                              {9.f, 10.f, 11.f, 12.f},
                                                              {13.f, 14.f, 15.f, 16.f}}}}}.get_vector());

    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{1.f, 1.4285715f, 1.8571429f, 2.2857141f, 2.7142856f, 3.142857f, 3.5714285f, 4.f},
                                                                        {2.7142856f, 3.1428568f, 3.5714283f, 4.f, 4.428571f, 4.8571424f, 5.285714f, 5.714286f},
                                                                        {4.428571f, 4.8571424f, 5.285714f, 5.714286f, 6.1428566f, 6.57142837f, 7.f, 7.4285717f},
                                                                        {6.142857f, 6.5714283f, 6.9999995f, 7.428571f, 7.8571424f, 8.285714f, 8.714286f, 9.142857f},
                                                                        {7.857143f, 8.285714f, 8.714286f, 9.142857f, 9.571428f, 10.f, 10.428572f, 10.857142f},
                                                                        {9.571428f, 10.f, 10.428572f, 10.857142f, 11.285713f, 11.714285f, 12.142857f, 12.571428f},
                                                                        {11.285714f, 11.714285f, 12.142857f, 12.571428f, 13.f, 13.428572, 13.857143f, 14.285714f},
                                                                        {13.f, 13.428571f, 13.857142f, 14.285714f, 14.714286f, 15.142857f, 15.571428f, 16.f}}}}).get_vector());
    return res;
}

static interpolateParam paddle_nearest_downsample_false_0() {
    interpolateParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "nearest_downsample_false_0";


    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f},
                                                              {9.f, 10.f, 11.f, 12.f},
                                                              {13.f, 14.f, 15.f, 16.f}}}}}.get_vector());

    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{1.f, 3.f, 5.f, 7.f},
                                                                        {9.f, 11.f, 13.f, 15.f}}}}).get_vector());
    return res;
}

static interpolateParam paddle_nearest_upsample_false_0() {
    interpolateParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "nearest_upsample_false_0";


    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f},
                                                              {9.f, 10.f, 11.f, 12.f},
                                                              {13.f, 14.f, 15.f, 16.f}}}}}.get_vector());

    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f},
                                                                        {1.f, 1.f, 2.f, 2.f, 3.f, 3.f, 4.f, 4.f},
                                                                        {5.f, 5.f, 6.f, 6.f, 7.f, 7.f, 8.f, 8.f},
                                                                        {5.f, 5.f, 6.f, 6.f, 7.f, 7.f, 8.f, 8.f},
                                                                        {9.f, 9.f, 10.f, 10.f, 11.f, 11.f, 12.f, 12.f},
                                                                        {9.f, 9.f, 10.f, 10.f, 11.f, 11.f, 12.f, 12.f},
                                                                        {13, 13.f, 14.f, 14.f, 15.f, 15.f, 16.f, 16.f},
                                                                        {13, 13.f, 14.f, 14.f, 15.f, 15.f, 16.f, 16.f}}}}).get_vector());
    return res;
}

TEST_P(interpolateTest, test_interpolate) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, interpolateTest,
                        ::testing::Values(
                                bilinear_downsample_false_1(),
                                bilinear_downsample_false_0(),
                                bilinear_downsample_true_0(),
                                bilinear_upsample_false_1(),
                                bilinear_upsample_false_0(),
                                bilinear_upsample_true_0(),
                                nearest_downsample_false_0()
                        ),
                        interpolateTest::getTestCaseName);

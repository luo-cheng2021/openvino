// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using conv2dTestParam = FrontendOpTestParam;
using conv2dTest = FrontendOpTest;

static conv2dTestParam paddle_conv2d_SAME_padding() {
    conv2dTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS + "paddle_conv2d_SAME_padding/";
    res.m_modelName =    "paddle_conv2d_SAME_padding.pdmodel";

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                          {5.f, 6.f, 7.f, 8.f, 9.f},
                                                          {10.f, 11.f, 12.f, 13.f, 14.f},
                                                          {15.f, 16.f, 17.f, 18.f, 19.f},
                                                          {20.f, 21.f, 22.f, 23.f, 24.f},
                                                          {25.f, 26.f, 27.f, 28.f, 29.f},
                                                          {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
            .get_vector());


    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{12.f, 27.f, 24.f},
                                                             {63.f, 108.f, 81.f},
                                                             {123.f, 198.f, 141.f},
                                                             {112.f, 177.f, 124.f}}}}).get_vector());

    return res;
}

static conv2dTestParam paddle_conv2d_VALID_padding() {
    conv2dTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS + "paddle_conv2d_VALID_padding/";
    res.m_modelName =    "paddle_conv2d_VALID_padding.pdmodel";

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f, 9.f},
                                                              {10.f, 11.f, 12.f, 13.f, 14.f},
                                                              {15.f, 16.f, 17.f, 18.f, 19.f},
                                                              {20.f, 21.f, 22.f, 23.f, 24.f},
                                                              {25.f, 26.f, 27.f, 28.f, 29.f},
                                                              {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                                    .get_vector());


    res.expected_outputs.emplace_back(test::NDArray<float, 3>({{{{54.f, 72.f},
                                                                        {144.f, 162.f},
                                                                        {234.f, 252.f}}}}).get_vector());

    return res;
}

static conv2dTestParam paddle_conv2d_strides_padding() {
    conv2dTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS + "paddle_conv2d_strides_padding/";
    res.m_modelName =    "paddle_conv2d_strides_padding.pdmodel";

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f, 9.f},
                                                              {10.f, 11.f, 12.f, 13.f, 14.f},
                                                              {15.f, 16.f, 17.f, 18.f, 19.f},
                                                              {20.f, 21.f, 22.f, 23.f, 24.f},
                                                              {25.f, 26.f, 27.f, 28.f, 29.f},
                                                              {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                                    .get_vector());


    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{12.f, 27.f, 24.f},
                                                                        {63.f, 108.f, 81.f},
                                                                        {123.f, 198.f, 141.f},
                                                                        {112.f, 177.f, 124.f}}}}).get_vector());

    return res;
}

static conv2dTestParam paddle_conv2d_strides_no_padding() {
    conv2dTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS + "paddle_conv2d_strides_no_padding/";
    res.m_modelName =    "paddle_conv2d_strides_no_padding.pdmodel";

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f, 9.f},
                                                              {10.f, 11.f, 12.f, 13.f, 14.f},
                                                              {15.f, 16.f, 17.f, 18.f, 19.f},
                                                              {20.f, 21.f, 22.f, 23.f, 24.f},
                                                              {25.f, 26.f, 27.f, 28.f, 29.f},
                                                              {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                                    .get_vector());


    res.expected_outputs.emplace_back(test::NDArray<float, 3>({{{{54.f, 72.f},
                                                                        {144.f, 162.f},
                                                                        {234.f, 252.f}}}}).get_vector());

    return res;
}

static conv2dTestParam paddle_conv2d_strides_assymetric_padding() {
    conv2dTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS + "paddle_conv2d_strides_assymetric_padding/";
    res.m_modelName =    "paddle_conv2d_strides_assymetric_padding.pdmodel";

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f, 9.f},
                                                              {10.f, 11.f, 12.f, 13.f, 14.f},
                                                              {15.f, 16.f, 17.f, 18.f, 19.f},
                                                              {20.f, 21.f, 22.f, 23.f, 24.f},
                                                              {25.f, 26.f, 27.f, 28.f, 29.f},
                                                              {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                                    .get_vector());


    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{21.f, 33.f},
                                                                        {99.f, 117.f},
                                                                        {189.f, 207.f},
                                                                        {171.f, 183.f}}}}).get_vector());

    return res;
}

static conv2dTestParam paddle_conv2d_dilation_assymetric_pads_strides() {
    conv2dTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS + "paddle_conv2d_dilation_assymetric_pads_strides/";
    res.m_modelName =    "paddle_conv2d_dilation_assymetric_pads_strides.pdmodel";

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f, 3.f, 4.f},
                                                              {5.f, 6.f, 7.f, 8.f, 9.f},
                                                              {10.f, 11.f, 12.f, 13.f, 14.f},
                                                              {15.f, 16.f, 17.f, 18.f, 19.f},
                                                              {20.f, 21.f, 22.f, 23.f, 24.f},
                                                              {25.f, 26.f, 27.f, 28.f, 29.f},
                                                              {30.f, 31.f, 32.f, 33.f, 34.f}}}}}
                                    .get_vector());


    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{12.f, 21.f, 27.f, 33.f, 24.f, 13.f},
                                                                 {93.f, 144.f, 153.f, 162.f, 111.f, 57.f},
                                                                 {112.f, 171.f, 177.f, 183.f, 124.f, 63.f}}}}).get_vector());

    return res;
}

static conv2dTestParam paddle_depthwise_conv2d_convolotuin() {
    conv2dTestParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS + "paddle_depthwise_conv2d_convolotuin/";
    res.m_modelName =    "paddle_depthwise_conv2d_convolotuin.pdmodel";

    res.inputs.emplace_back(test::NDArray<float, 4>{{{{{0.f, 1.f, 2.f},
                                                              {3.f, 4.f, 5.f},
                                                              {6.f, 7.f, 8.f},
                                                              {9.f, 10.f, 11.f},
                                                              {12.f, 13.f, 14.f},
                                                              {15.f, 16.f, 17.f},
                                                              {18.f, 19.f, 20.f},
                                                              {21.f, 22.f, 23.f},
                                                              {24.f, 25.f, 26.f}}}}}.get_vector());


    res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{8.f, 15.f, 12.f},
                                                                        {21.f, 36.f, 27.f},
                                                                        {20.f, 33.f, 24.f},
                                                                        {44.f, 69.f, 48.f},
                                                                        {75.f, 117.f, 81.f},
                                                                        {56.f, 87.f, 60.f},
                                                                        {80.f, 123.f, 84.f},
                                                                        {129.f, 198.f, 135.f},
                                                                        {92.f, 141.f, 96.f}}}}).get_vector());

    return res;
}

TEST_P(conv2dTest, test_conv2d) {
ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, conv2dTest,
        ::testing::Values(
                paddle_conv2d_SAME_padding(),
                paddle_conv2d_VALID_padding(),
                paddle_conv2d_strides_padding(),
                paddle_conv2d_strides_no_padding(),
                paddle_conv2d_strides_assymetric_padding(),
                paddle_conv2d_dilation_assymetric_pads_strides(),
                paddle_depthwise_conv2d_convolotuin()
                ),
        conv2dTest::getTestCaseName);

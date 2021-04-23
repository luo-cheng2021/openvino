// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

auto shared_input_NCHW = test::NDArray<float, 4>{{{{{0.0, 1.0, 2.0, 3.0 },
                                                    {4.0, 5.0, 6.0, 7.0 },
                                                    {8.0, 9.0, 10.0, 11.0 },
                                                    {12.0, 13.0, 14.0, 15.0 }},
                                                    {{16.0, 17.0, 18.0, 19.0 },
                                                    {20.0, 21.0, 22.0, 23.0 },
                                                    {24.0, 25.0, 26.0, 27.0 },
                                                    {28.0, 29.0, 30.0, 31.0 }},
                                                    {{32.0, 33.0, 34.0, 35.0 },
                                                    {36.0, 37.0, 38.0, 39.0 },
                                                    {40.0, 41.0, 42.0, 43.0 },
                                                    {44.0, 45.0, 46.0, 47.0 }}},
                                                    {{{48.0, 49.0, 50.0, 51.0 },
                                                    {52.0, 53.0, 54.0, 55.0 },
                                                    {56.0, 57.0, 58.0, 59.0 },
                                                    {60.0, 61.0, 62.0, 63.0 }},
                                                    {{64.0, 65.0, 66.0, 67.0 },
                                                    {68.0, 69.0, 70.0, 71.0 },
                                                    {72.0, 73.0, 74.0, 75.0 },
                                                    {76.0, 77.0, 78.0, 79.0 }},
                                                    {{80.0, 81.0, 82.0, 83.0 },
                                                    {84.0, 85.0, 86.0, 87.0 },
                                                    {88.0, 89.0, 90.0, 91.0 },
                                                    {92.0, 93.0, 94.0, 95.0 }}}}}
                            .get_vector();

auto shared_input_NHWC = test::NDArray<float, 4>{{{{{0.0, 1.0, 2.0 },
                                                    {3.0, 4.0, 5.0 },
                                                    {6.0, 7.0, 8.0 },
                                                    {9.0, 10.0, 11.0 }},
                                                    {{12.0, 13.0, 14.0 },
                                                    {15.0, 16.0, 17.0 },
                                                    {18.0, 19.0, 20.0 },
                                                    {21.0, 22.0, 23.0 }},
                                                    {{24.0, 25.0, 26.0 },
                                                    {27.0, 28.0, 29.0 },
                                                    {30.0, 31.0, 32.0 },
                                                    {33.0, 34.0, 35.0 }},
                                                    {{36.0, 37.0, 38.0 },
                                                    {39.0, 40.0, 41.0 },
                                                    {42.0, 43.0, 44.0 },
                                                    {45.0, 46.0, 47.0 }}},
                                                    {{{48.0, 49.0, 50.0 },
                                                    {51.0, 52.0, 53.0 },
                                                    {54.0, 55.0, 56.0 },
                                                    {57.0, 58.0, 59.0 }},
                                                    {{60.0, 61.0, 62.0 },
                                                    {63.0, 64.0, 65.0 },
                                                    {66.0, 67.0, 68.0 },
                                                    {69.0, 70.0, 71.0 }},
                                                    {{72.0, 73.0, 74.0 },
                                                    {75.0, 76.0, 77.0 },
                                                    {78.0, 79.0, 80.0 },
                                                    {81.0, 82.0, 83.0 }},
                                                    {{84.0, 85.0, 86.0 },
                                                    {87.0, 88.0, 89.0 },
                                                    {90.0, 91.0, 92.0 },
                                                    {93.0, 94.0, 95.0 }}}}}
                            .get_vector();                            

/* maxPool2D */
namespace maxPool2D {

    using maxPool2DTestParam = FrontendOpTestParam;
    using maxPool2DTest = FrontendOpTest;

    static maxPool2DTestParam maxPool_test1() {
        maxPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "maxPool_test1";  //TODO: compact model/decomposited

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 2, 2)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{1.0, 3.0 },
                                                                    {13.0, 15.0 }},
                                                                    {{17.0, 19.0 },
                                                                    {29.0, 31.0 }},
                                                                    {{33.0, 35.0 },
                                                                    {45.0, 47.0 }}},
                                                                    {{{49.0, 51.0 },
                                                                    {61.0, 63.0 }},
                                                                    {{65.0, 67.0 },
                                                                    {77.0, 79.0 }},
                                                                    {{81.0, 83.0 },
                                                                    {93.0, 95.0 }}}}})
                                .get_vector());

        return res;
    }
    static maxPool2DTestParam maxPool_test2() {
        maxPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "maxPool_test2";  //TODO: compact model/decomposited

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 2, 2)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({})
                                .get_vector());

        return res;
    }
    static maxPool2DTestParam maxPool_test3() {
        maxPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "maxPool_test3";  //TODO: compact model/decomposited

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 2, 2)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{5.00, 7.00 },
                                                                    {13.00, 15.00 }},
                                                                    {{21.00, 23.00 },
                                                                    {29.00, 31.00 }},
                                                                    {{37.00, 39.00 },
                                                                    {45.00, 47.00 }}},
                                                                    {{{53.00, 55.00 },
                                                                    {61.00, 63.00 }},
                                                                    {{69.00, 71.00 },
                                                                    {77.00, 79.00 }},
                                                                    {{85.00, 87.00 },
                                                                    {93.00, 95.00 }}}}})
                                .get_vector());

        return res;
    }
    static maxPool2DTestParam maxPool_test4() {
        maxPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "maxPool_test4";  //TODO: compact model/decomposited

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 1, 1)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{10.00 }},
                                                                    {{26.00 }},
                                                                    {{42.00 }}},
                                                                    {{{58.00 }},
                                                                    {{74.00 }},
                                                                    {{90.00 }}}}})
                                .get_vector());

        return res;
    }
    static maxPool2DTestParam maxPool_test5() {
        maxPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "maxPool_test5";  //TODO: compact model/decomposited

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 1, 1)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{15.00 }},
                                                                    {{31.00 }},
                                                                    {{47.00 }}},
                                                                    {{{63.00 }},
                                                                    {{79.00 }},
                                                                    {{95.00 }}}}})
                                .get_vector());

        return res;
    }
    static maxPool2DTestParam maxPool_test6() {
        maxPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "maxPool_test6";  //TODO: compact model/decomposited

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NHWC);

        // (2, 2, 2, 3)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{3.00, 4.00, 5.00 },
                                                                    {9.00, 10.00, 11.00 }},
                                                                    {{39.00, 40.00, 41.00 },
                                                                    {45.00, 46.00, 47.00 }}},
                                                                    {{{51.00, 52.00, 53.00 },
                                                                    {57.00, 58.00, 59.00 }},
                                                                    {{87.00, 88.00, 89.00 },
                                                                    {93.00, 94.00, 95.00 }}}}})
                                .get_vector());

        return res;
    }                    

    TEST_P(maxPool2DTest, test_pool2d) {
        validateOp();
    }

    INSTANTIATE_TEST_CASE_P(FrontendOpTest, maxPool2DTest,
                            ::testing::Values(
                                maxPool_test1(),
                                //maxPool_test2(),
                                maxPool_test3(),
                                maxPool_test4(),
                                maxPool_test5(),
                                maxPool_test6()
                            ),                        
                            maxPool2DTest::getTestCaseName); 
}

/* avgPool2D */
namespace avgPool2D {
    using avgPool2DTestParam = FrontendOpTestParam;
    using avgPool2DTest = FrontendOpTest;

    static avgPool2DTestParam avgPool_test1() {
        avgPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "avgPool_test1";

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 2, 2)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{0.50, 2.50 },
                                                                    {8.50, 10.50 }},
                                                                    {{16.50, 18.50 },
                                                                    {24.50, 26.50 }},
                                                                    {{32.50, 34.50 },
                                                                    {40.50, 42.50 }}},
                                                                    {{{48.50, 50.50 },
                                                                    {56.50, 58.50 }},
                                                                    {{64.50, 66.50 },
                                                                    {72.50, 74.50 }},
                                                                    {{80.50, 82.50 },
                                                                    {88.50, 90.50 }}}}})
                                .get_vector());

        return res;
    }

    static avgPool2DTestParam avgPool_test2() {
        avgPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "avgPool_test2";

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (1, 1, 4, 4)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{7.5 }},
                                                                    {{23.5 }},
                                                                    {{39.5 }}},
                                                                    {{{55.5 }},
                                                                    {{71.5 }},
                                                                    {{87.5 }}}}})
                                .get_vector());

        return res;
    }

    static avgPool2DTestParam avgPool_test3() {
        avgPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "avgPool_test3";

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 2, 2)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{2.50, 4.50 },
                                                                    {10.50, 12.50 }},
                                                                    {{18.50, 20.50 },
                                                                    {26.50, 28.50 }},
                                                                    {{34.50, 36.50 },
                                                                    {42.50, 44.50 }}},
                                                                    {{{50.50, 52.50 },
                                                                    {58.50, 60.50 }},
                                                                    {{66.50, 68.50 },
                                                                    {74.50, 76.50 }},
                                                                    {{82.50, 84.50 },
                                                                    {90.50, 92.50 }}}}})
                                .get_vector());

        return res;
    }

    static avgPool2DTestParam avgPool_test4() {
        avgPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "avgPool_test4";

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 1, 1)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{5.00 }},
                                                                    {{21.00 }},
                                                                    {{37.00 }}},
                                                                    {{{53.00 }},
                                                                    {{69.00 }},
                                                                    {{85.00 }}}}})
                                .get_vector());

        return res;
    }                

    static avgPool2DTestParam avgPool_test5() {
        avgPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "avgPool_test5";

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 1, 1)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{7.5 }},
                                                                    {{23.5 }},
                                                                    {{39.5 }}},
                                                                    {{{55.5 }},
                                                                    {{71.5 }},
                                                                    {{87.5 }}}}})
                                .get_vector());

        return res;
    }

    static avgPool2DTestParam avgPool_test6() {
        avgPool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "avgPool_test6";  //TODO: compact model/decomposited

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NHWC);

        // (2, 2, 2, 3)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{1.5, 2.5, 3.5 },
                                                                    {7.5, 8.5, 9.5 }},
                                                                    {{25.5, 26.5, 27.5 },
                                                                    {31.5, 32.5, 33.5 }}},
                                                                    {{{49.5, 50.5, 51.5 },
                                                                    {55.5, 56.5, 57.5 }},
                                                                    {{73.5, 74.5, 75.5 },
                                                                    {79.5, 80.5, 81.5 }}}}})
                                .get_vector());

        return res;
    }

    TEST_P(avgPool2DTest, test_pool2d) {
        validateOp();
    }

    INSTANTIATE_TEST_CASE_P(FrontendOpTest, avgPool2DTest,
                            ::testing::Values(
                                avgPool_test1(),
                                //avgPool_test2(),
                                avgPool_test3(),
                                avgPool_test4(),
                                avgPool_test5(),
                                avgPool_test6()
                            ),                        
                            avgPool2DTest::getTestCaseName); 
}

/* maxAaptivePool2D */
namespace maxAaptivePool2D {
    using maxAaptivePool2DTestParam = FrontendOpTestParam;
    using maxAaptivePool2DTest = FrontendOpTest;

    static maxAaptivePool2DTestParam maxAdaptivePool_test1() {
        maxAaptivePool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "maxAdaptivePool2D_test1";

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 2, 2)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{5.00, 6.00, 7.00 },
                                                                    {9.00, 10.00, 11.00 },
                                                                    {13.00, 14.00, 15.00 }},
                                                                    {{21.00, 22.00, 23.00 },
                                                                    {25.00, 26.00, 27.00 },
                                                                    {29.00, 30.00, 31.00 }},
                                                                    {{37.00, 38.00, 39.00 },
                                                                    {41.00, 42.00, 43.00 },
                                                                    {45.00, 46.00, 47.00 }}},
                                                                    {{{53.00, 54.00, 55.00 },
                                                                    {57.00, 58.00, 59.00 },
                                                                    {61.00, 62.00, 63.00 }},
                                                                    {{69.00, 70.00, 71.00 },
                                                                    {73.00, 74.00, 75.00 },
                                                                    {77.00, 78.00, 79.00 }},
                                                                    {{85.00, 86.00, 87.00 },
                                                                    {89.00, 90.00, 91.00 },
                                                                    {93.00, 94.00, 95.00 }}}}})
                                .get_vector());

        return res;
    }

    TEST_P(maxAaptivePool2DTest, test_adaptive_pool2d) {
        validateOp();
    }

    INSTANTIATE_TEST_CASE_P(FrontendOpTest, maxAaptivePool2DTest,
                            ::testing::Values(
                                maxAdaptivePool_test1()
                            ),                        
                            maxAaptivePool2DTest::getTestCaseName); 
}

/* avgAaptivePool2D */
namespace avgAaptivePool2D {
    using avgAaptivePool2DTestParam = FrontendOpTestParam;
    using avgAaptivePool2DTest = FrontendOpTest;

    static avgAaptivePool2DTestParam avgAdaptivePool_test1() {
        avgAaptivePool2DTestParam res;
        res.m_frontEndName = PDPD;
        res.m_modelsPath =   PATH_TO_MODELS;
        res.m_modelName =    "avgAdaptivePool2D_test1";

        // data (2, 3, 4, 4) input tensor
        res.inputs.emplace_back(shared_input_NCHW);

        // (2, 3, 2, 2)
        res.expected_outputs.emplace_back(test::NDArray<float, 4>({{{{{2.50, 3.50, 4.50 },
                                                                    {6.50, 7.50, 8.50 },
                                                                    {10.50, 11.50, 12.50 }},
                                                                    {{18.50, 19.50, 20.50 },
                                                                    {22.50, 23.50, 24.50 },
                                                                    {26.50, 27.50, 28.50 }},
                                                                    {{34.50, 35.50, 36.50 },
                                                                    {38.50, 39.50, 40.50 },
                                                                    {42.50, 43.50, 44.50 }}},
                                                                    {{{50.50, 51.50, 52.50 },
                                                                    {54.50, 55.50, 56.50 },
                                                                    {58.50, 59.50, 60.50 }},
                                                                    {{66.50, 67.50, 68.50 },
                                                                    {70.50, 71.50, 72.50 },
                                                                    {74.50, 75.50, 76.50 }},
                                                                    {{82.50, 83.50, 84.50 },
                                                                    {86.50, 87.50, 88.50 },
                                                                    {90.50, 91.50, 92.50 }}}}})
                                .get_vector());

        return res;
    }

    TEST_P(avgAaptivePool2DTest, test_adaptive_pool2d) {
        validateOp();
    }

    INSTANTIATE_TEST_CASE_P(FrontendOpTest, avgAaptivePool2DTest,
                            ::testing::Values(
                                avgAdaptivePool_test1()
                            ),                        
                            avgAaptivePool2DTest::getTestCaseName); 
}

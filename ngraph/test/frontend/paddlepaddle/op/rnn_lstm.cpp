// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "../../shared/include/op.hpp"

using namespace ngraph;
using namespace ngraph::frontend;

static const std::string PDPD = "pdpd";
static const std::string PATH_TO_MODELS = "/paddlepaddle/models/";

using concatTestParam = FrontendOpTestParam;
using rnnLSTM = FrontendOpTest;

static rnnLSTMParam rnn_lstm_layer_1_forward() {
    rnnLSTMParam res;
    res.m_frontEndName = PDPD;
    res.m_modelsPath =   PATH_TO_MODELS;
    res.m_modelName =    "rnn_lstm_layer_1_forward";

    res.inputs.emplace_back(test::NDArray<float, 3>{{{{1.00, 1.00 },
                                                      {1.00, 1.00 },
                                                      {1.00, 1.00 }},
                                                     {{1.00, 1.00 },
                                                      {1.00, 1.00 },
                                                      {1.00, 1.00 }},
                                                      {{1.00, 1.00 },
                                                       {1.00, 1.00 },
                                                       {1.00, 1.00 }},
                                                       {{1.00, 1.00 },
                                                        {1.00, 1.00 },
                                                        {1.00, 1.00 }}}
                                                }.get_vector());


    res.expected_outputs.emplace_back(test::NDArray<float, 3>{
            {
                    {
                            {0.94405496, 0.94405496}
                            {0.9393652,  0.9393652},
                            {0.9396501,  0.9396501}
                    },
                    {
                            {0.94405496, 0.94405496},
                            {0.91908807, 0.91908807},
                            {0.9143285,  0.9143285 }
                    },
                    {
                            {0.8271083, 0.8271083 },
                            {0.66429245, 0.66429245},
                            {0.61485946, 0.61485946}
                    },
                    {
                            {0.8271083, 0.8271083},
                            {0.84379673, 0.84379673]},
                            {0.8524127, 0.8524127}
                    }

            }}
    }.get_vector());
    return res;
}

TEST_P(rnnLSTM, test_LSTM) {
    ASSERT_NO_THROW(validateOp());
}

INSTANTIATE_TEST_CASE_P(FrontendOpTest, rnnLSTM,
                        ::testing::Values(
                                rnn_lstm_layer_1_forward()
                        ),
                        rnnLSTM::getTestCaseName);

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <mkldnn_extension_utils.h>
#include "opt/mkldnn_adaptive_pooling_opt.h"

namespace MKLDNNPlugin {

class MKLDNNAdaptivePoolingNode : public MKLDNNNode {
public:
  MKLDNNAdaptivePoolingNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    int spatialDimsCount;
    InferenceEngine::Precision precision = InferenceEngine::Precision::FP32;
    AvgFunc _avg;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin

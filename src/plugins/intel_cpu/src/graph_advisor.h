// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp/ie_cnn_network.h"
#include "config.h"
#include "cpu_memory.h"
#include "normalize_preprocess.h"
#include "node.h"
#include "edge.h"
#include "cache/multi_cache.h"
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <atomic>

namespace ov {
namespace intel_cpu {

class Advisor {
public:
    static bool AdviseBrgconv(dnnl::engine& engine, std::shared_ptr<const ov::Model> func);
};

}   // namespace intel_cpu
}   // namespace ov

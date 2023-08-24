// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <array>
#include <cstddef>
#include <vector>
#include <cstdint>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {

void rms_norm(float* dst, float* src, float eps, float* weight, size_t len);

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
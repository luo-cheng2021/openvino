// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/rms.hpp"
#include "program_node.h"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/rms.hpp"
#endif

#if OV_GPU_WITH_SYCL
    #include "impls/sycl/rms.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<rms>::get_implementations() {
#if OV_GPU_WITH_SYCL
    static const std::vector<std::shared_ptr<ImplementationManager>> impls_sycl = {
        OV_GPU_CREATE_INSTANCE_SYCL(sycl::RMSImplementationManagerSYCL, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_SYCL(sycl::RMSImplementationManagerSYCL, shape_types::dynamic_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::RMSImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::RMSImplementationManager, shape_types::dynamic_shape)
    };
    auto p = std::getenv("USE_SYCL");
    if (p && p[0] == '1') {
        return impls_sycl;
    }
#endif
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::RMSImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::RMSImplementationManager, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu

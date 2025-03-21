// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms_inst.h"
#include "registry/implementation_manager.hpp"

#include <memory>

namespace cldnn {
namespace sycl {

struct RMSImplementationManagerSYCL : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("RMSImplementationManagerSYCL")
    RMSImplementationManagerSYCL(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::sycl, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<rms>());

        static const std::vector<format::type> supported_formats = {
            format::bfyx,
        };

        const auto& rms_node = node.as<rms>();
        const auto& in_layout = rms_node.get_input_layout(0);
        const auto& out_layout = rms_node.get_output_layout(0);
        auto in0_dt = in_layout.data_type;
        auto out_dt = out_layout.data_type;

        bool is_float = one_of(in0_dt, {data_types::f16, data_types::f32}) &&
                        one_of(out_dt, {data_types::f16, data_types::f32});
        if (!is_float)
            return false;


        if (!one_of(in_layout.format.value, supported_formats) || !one_of(out_layout.format.value, supported_formats))
            return false;

        if (in_layout.data_padding || out_layout.data_padding)
            return false;

        return true;
    }
};

}  // namespace sycl
}  // namespace cldnn

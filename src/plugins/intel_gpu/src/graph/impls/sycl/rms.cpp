// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rms.hpp"
#include "rms_inst.h"
#include "ocl/sycl_engine.hpp"
#include "ocl/sycl_stream.hpp"
#include "openvino/core/type/element_type.hpp"
#include "primitive_sycl_base.h"

#include "impls/ocl/kernel_selector_helper.h"

#include "rms_kernel.hpp"

#include <memory>

#ifdef __SYCL_DEVICE_ONLY__
          #define CONSTANT __attribute__((opencl_constant))
#else
          #define CONSTANT
#endif

namespace cldnn {
namespace sycl {

struct rms_sycl : typed_primitive_sycl_impl<rms> {
    using parent = typed_primitive_sycl_impl<rms>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::sycl::rms_sycl)

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<rms_sycl>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& /* events */, typed_primitive_inst<rms>& instance) override {
        auto& network = instance.get_network();
        const auto& desc = instance.get_typed_desc<rms>();

        auto& stream = downcast<ocl::sycl_stream>(network.get_stream());
        auto& engine = downcast<ocl::sycl_engine>(network.get_engine());
        ::sycl::context sycl_context = engine.get_sycl_context();
        ::sycl::queue& sycl_queue = stream.get_sycl_queue();

        const auto& params = instance.get_impl_params();
        auto out_shape = params->output_layouts[0].get_shape();

        auto output = instance.output_memory_ptr(0);

        std::vector<memory::ptr> inputs = { instance.input_memory_ptr(0), instance.input_memory_ptr(1) };

        ov::element::Type_t data_t = params->input_layouts[0].data_type;
        ov::element::Type_t weight_t = params->input_layouts[1].data_type;
        ov::element::Type_t out_t = params->output_layouts[0].data_type;
        const auto& primitive = params->typed_desc<cldnn::rms>();
        auto eps = primitive->epsilon;

        // batch * seq_len
        size_t bs = std::accumulate(out_shape.begin(), out_shape.end() - 1, size_t{1}, std::multiplies<size_t>());
        size_t channel_num = *(out_shape.end() - 1);

        bool barrier = stream.get_queue_type() == QueueTypes::out_of_order;
        if (barrier) {
            sycl_queue.submit([=](::sycl::handler& cgh) {
                cgh.ext_oneapi_barrier();
            });
        }
        #define CASE(InputType, WeightType, DstType) \
            data_t == ov::element::InputType && \
            weight_t == ov::element::WeightType && \
            out_t == ov::element::DstType

        if (CASE(f32, f32, f32)) {
            auto data = static_cast<const float*>(inputs[0]->buffer_ptr());
            auto weight = static_cast<const float*>(inputs[1]->buffer_ptr());
            auto out = static_cast<float*>(output->buffer_ptr());
            auto event = details::rms_kernel(sycl_queue, data, weight, out, bs, channel_num, eps);

            return to_ocl_event(stream, event);
        } else if (CASE(f16, f16, f16)) {
            auto data = static_cast<::sycl::half*>(inputs[0]->buffer_ptr());
            auto weight = static_cast<::sycl::half*>(inputs[1]->buffer_ptr());
            auto out = static_cast<::sycl::half*>(output->buffer_ptr());

            auto event = details::rms_kernel(sycl_queue, data, weight, out, bs, channel_num, eps);

            return to_ocl_event(stream, event);
        } else {
            OPENVINO_THROW("No instance for given types found: ", data_t, " ", out_t);
        }
    }

    static std::unique_ptr<primitive_impl> create(const rms_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        return std::make_unique<rms_sycl>(engine, config);
    }
};

std::unique_ptr<primitive_impl> RMSImplementationManagerSYCL::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<rms>());
    return sycl::rms_sycl::create(static_cast<const rms_node&>(node), params);
}

}  // namespace sycl
}  // namespace cldnn

// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/except.hpp"
#include "openvino/core/partial_shape.hpp"
#include "stub_opt.hpp"

#include <cctype>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_ocl.hpp>
#include <sstream>
#include <string_view>
#include <tuple>
#include <utility>

#include "cm/utils/kernel_generator.hpp"
#include "cm/utils/kernels_db.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "debug_helper.hpp"
#include "intel_gpu/primitives/stub.hpp"
#include "intel_gpu/runtime/lru_cache.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "stub_inst.h"
#include "ocl_v2/utils/fused_ops_jitter.hpp"
#include "ocl_v2/utils/jitter.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "utils/kernel_generator.hpp"
#include "cm/utils/kernel_generator.hpp"
#include "stub_helper.hpp"

namespace ov::intel_gpu::ocl {

class BiAttention : public cm::KernelGenerator {
public:
    static constexpr const char* m_name = "stub_biattention";
    BiAttention() : KernelGenerator(m_name, "_1") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<stub>();
        const auto& param = desc->m_params;
        for (auto&& kv : param) {
            auto&& key = kv.first;
            if (std::all_of(key.begin(), key.end(), [](unsigned char c){ return !std::isalpha(c) || std::isupper(c); })) {
                jit.make(kv.first, kv.second);
            }
        }

        jit.make("KERNEL_NAME", get_entry_point(params));
        jit.make("STAGE", "1");

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }
};

class BiAttention2 : public cm::KernelGenerator {
public:
    static constexpr const char* m_name = "stub_biattention";
    BiAttention2() : KernelGenerator(m_name, "_2") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        auto desc = params.typed_desc<stub>();
        const auto& param = desc->m_params;
        for (auto&& kv : param) {
            auto&& key = kv.first;
            if (std::all_of(key.begin(), key.end(), [](unsigned char c){ return !std::isalpha(c) || std::isupper(c); })) {
                jit.make(kv.first, kv.second);
            }
        }

        jit.make("KERNEL_NAME", get_entry_point(params));
        jit.make("STAGE", "2");

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }
};

struct OPENVINO_CORE_EXPORTS biattention_opt : cldnn::custom_kernel {
    std::shared_ptr<stub> m_prim;
    program_node& m_prog_node;
    size_t m_num_heads;

    biattention_opt(std::shared_ptr<stub> prim, program_node& prog) : m_prim(prim), m_prog_node(prog) {
        m_num_heads = std::stoi(prim->m_params["NUM_HEADS"]);
    }
    virtual std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>> create_kernels() override {
        return {
            std::make_shared<Stage>(std::make_shared<BiAttention>()),
            std::make_shared<Stage>(std::make_shared<BiAttention2>())
        };
    }
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        auto cur_stubop = params.typed_desc<stub>();
        std::vector<BufferDescriptor> internal_buffers;
        auto query_layout = params.input_layouts[0];
        auto key_layout = params.input_layouts[1];
        auto query_ps = query_layout.get_partial_shape();
        auto key_ps = key_layout.get_partial_shape();
        ov::PartialShape ps = {query_ps[0], static_cast<int>(m_num_heads), key_ps[1], (query_ps[1].get_length() + 15) / 16 * 16};
        query_layout.set_partial_shape(ps);
        query_layout.data_type = ov::element::f32;
        internal_buffers.emplace_back(query_layout, false);
        return internal_buffers;
    }

    virtual cldnn::event::ptr execute(const std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>>&kernels, const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        // node input:
        //  query
        //  key
        //  vison_values
        //  lang_values
        //  attn_mask_l
        // first kernel input:
        //  int seqlen
        //  int kv_seq_len
        //  int attn_weights_stride  unit is float
        //  half* query,        [batch, v_seq_len, HEAD_NUM * HEAD_DIM]
        //  half* key,          [batch, L_SEQ_LEN, HEAD_NUM * HEAD_DIM]
        //  half* value,        [batch, L_SEQ_LEN, HEAD_NUM * HEAD_DIM], lang_values
        //  half* attn_mask     [batch, L_SEQ_LEN]
        //  float* attn_weights [batch, HEAD_NUM, L_SEQ_LEN, v_seq_len'], v_seq_len' aligned to 64 bytes
        //  half* output        [batch, v_seq_len, HEAD_NUM * HEAD_DIM], outputs[0]

        auto query_layout = ins.input_memory_ptr(0)->get_layout();
        auto key_layout = ins.input_memory_ptr(1)->get_layout();
        // wg_size = 16
        // q_step = CM_GRF_WIDTH//32 # or 8 on Xe1
        // wg_seq_len = wg_size * q_step
        // wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
        // GWS = [1, self.num_heads, wg_count * wg_size]
        // LWS = [1, 1, wg_size]
        auto query_ps = query_layout.get_partial_shape();
        auto key_ps = key_layout.get_partial_shape();
        size_t batch = query_ps[0].get_length();
        size_t seq_len = query_ps[1].get_length();
        size_t key_len = key_ps[1].get_length();
        size_t wg_size = 16;
        auto CM_GRF_WIDTH = 512;
        auto q_step = CM_GRF_WIDTH / 32;
        auto wg_seq_len = wg_size * q_step;
        auto wg_count = (seq_len + wg_seq_len - 1) / wg_seq_len;
        std::vector<size_t> global = {batch, m_num_heads, static_cast<size_t>(wg_count * wg_size)};
        std::vector<size_t> local = {1, 1, wg_size};
        cldnn::kernel_arguments_desc desc;
        cldnn::kernel_arguments_data args;

        scalars_desc scalars_desc;
        scalar_desc sdesc;
        desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        sdesc.t = scalar_desc::Types::INT32;
        sdesc.v.s32 = seq_len;
        scalars_desc.push_back(sdesc);

        desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 1});
        sdesc.t = scalar_desc::Types::INT32;
        sdesc.v.s32 = key_len;
        scalars_desc.push_back(sdesc);

        desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 2});
        sdesc.t = scalar_desc::Types::INT32;
        sdesc.v.s32 = (seq_len + 15) / 16 * 16;
        scalars_desc.push_back(sdesc);

        args.scalars = &scalars_desc;

        desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        args.inputs.push_back(ins.input_memory_ptr(0));

        desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
        args.inputs.push_back(ins.input_memory_ptr(1));

        desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
        args.inputs.push_back(ins.input_memory_ptr(3));

        desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 3});
        args.inputs.push_back(ins.input_memory_ptr(4));

        const auto& intermediates_memories = ins.get_intermediates_memories();
        desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.outputs.push_back(intermediates_memories[0]);

        desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
        args.outputs.push_back(ins.output_memory_ptr(0));

        desc.workGroups.global = global;
        desc.workGroups.local = local;

        return execute_stage(events,
                             ins,
                             *kernels[0],
                             desc,
                             args,
                             ins.needs_completion_event());
    }
    
    virtual layout calc_output_layout(const program_node& node, const kernel_impl_params& params) const override {
        auto layout = params.input_layouts[0];
        return layout;
    }

    virtual std::vector<layout> calc_output_layouts(const program_node& node, const kernel_impl_params& impl_param) const override {
        return {impl_param.input_layouts[0], impl_param.input_layouts[1]};
    }
};

static std::shared_ptr<custom_kernel> create_biattention_kernel(std::shared_ptr<stub> prim, program_node& prog) {
    return std::make_shared<ov::intel_gpu::ocl::biattention_opt>(prim, prog);
}

}

namespace cldnn {
DEFINE_REG_CUSTOM_KERNEL(BiAttention, ov::intel_gpu::ocl::create_biattention_kernel);
}
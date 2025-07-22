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
    static constexpr const char* m_name = "stub_biattention_v";
    BiAttention() : KernelGenerator(m_name) {}

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

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }

    std::string get_build_options(const RuntimeParams& params) const override {
        // -mdump_asm
        return " -cmc -Qxcm_jit_option=\"-abortonspill\" -Qxcm_register_file_size=256 -g2 ";
    }
};

class BiAttention2 : public cm::KernelGenerator {
public:
    static constexpr const char* m_name = "stub_biattention_max";
    BiAttention2() : KernelGenerator(m_name) {}

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

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }

    std::string get_build_options(const RuntimeParams& params) const override {
        // -mdump_asm
        return " -cmc -Qxcm_jit_option=\"-abortonspill\" -Qxcm_register_file_size=256 -g2 ";
    }
};

class BiAttention3 : public cm::KernelGenerator {
public:
    static constexpr const char* m_name = "stub_biattention_exp";
    BiAttention3() : KernelGenerator(m_name) {}

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

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }

    std::string get_build_options(const RuntimeParams& params) const override {
        // -mdump_asm
        return " -cmc -Qxcm_jit_option=\"-abortonspill\" -Qxcm_register_file_size=256 -g2 ";
    }
};

class BiAttention4 : public cm::KernelGenerator {
public:
    static constexpr const char* m_name = "stub_biattention_l";
    BiAttention4() : KernelGenerator(m_name) {}

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

        return jit;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{nullptr};
    }

    std::string get_build_options(const RuntimeParams& params) const override {
        // -mdump_asm
        return " -cmc -Qxcm_jit_option=\"-abortonspill\" -Qxcm_register_file_size=256 -g2 ";
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
            std::make_shared<Stage>(std::make_shared<BiAttention2>()),
            std::make_shared<Stage>(std::make_shared<BiAttention3>()),
            std::make_shared<Stage>(std::make_shared<BiAttention4>()),
        };
    }
    std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        auto cur_stubop = params.typed_desc<stub>();
        std::vector<BufferDescriptor> internal_buffers;
        auto query_layout = params.input_layouts[0];
        auto key_layout = params.input_layouts[1];
        auto query_ps = query_layout.get_partial_shape();
        auto key_ps = key_layout.get_partial_shape();
        {
            // attn_weights
            ov::PartialShape ps = {query_ps[0], static_cast<int>(m_num_heads), key_ps[1], (query_ps[1].get_length() + 31) / 32 * 32};
            query_layout.set_partial_shape(ps);
            internal_buffers.emplace_back(query_layout, false);
        }
        {
            // max, half
            ov::PartialShape ps = {key_ps[0] * static_cast<int>(m_num_heads), (key_ps[1].get_length() + 31) / 32 * 32, 1};
            key_layout.set_partial_shape(ps);
            internal_buffers.emplace_back(key_layout, false);
        }
        {
            // sum, float
            ov::PartialShape ps = {key_ps[0] * static_cast<int>(m_num_heads), (key_ps[1].get_length() + 31) / 32 * 32, 1};;
            //key_layout.data_type = ov::element::f32;
            key_layout.set_partial_shape(ps);
            internal_buffers.emplace_back(key_layout, false);
        }
        return internal_buffers;
    }

    virtual cldnn::event::ptr execute(const std::vector<std::shared_ptr<ov::intel_gpu::ocl::Stage>>&kernels, const std::vector<cldnn::event::ptr>& events, cldnn::primitive_inst& ins) override {
        // node input:
        //  query: vision feature
        //  key: text feature
        //  vison_values
        //  lang_values
        //  attn_mask_l
        // first(vision attn) kernel input:
        //  int seqlen
        //  int kv_seq_len
        //  int attn_weights_stride  unit is half
        //  half* query,        [batch, v_seq_len, HEAD_NUM * HEAD_DIM]
        //  half* key,          [batch, L_SEQ_LEN, HEAD_NUM * HEAD_DIM]
        //  half* value,        [batch, L_SEQ_LEN, HEAD_NUM * HEAD_DIM], lang_values
        //  half* attn_mask     [batch, L_SEQ_LEN]
        //  half* attn_weights  [batch, HEAD_NUM, L_SEQ_LEN, v_seq_len'], v_seq_len' aligned to 64 bytes
        //  half* output        [batch, v_seq_len, HEAD_NUM * HEAD_DIM], it's outputs[0]
        // second(max) kernel input:
        //  int l_pic
        //  int l_pic_stride    unit is float, l_pic_stride = v_seq_len'
        //  int l_text_stride   unit is half
        //  half* attn          [batch, HEAD_NUM, L_SEQ_LEN, v_seq_len']
        //  half* output        [batch, L_SEQ_LEN', 1], L_SEQ_LEN' aligned to 64 bytes
        // third(exp) kernel input:
        //  int l_pic
        //  int l_pic_stride    unit is float, l_pic_stride = v_seq_len'
        //  int l_text_stride   unit is half
        //  half* attn          [batch, HEAD_NUM, L_SEQ_LEN, v_seq_len']
        //  half* max_buf       [batch, L_SEQ_LEN', 1], L_SEQ_LEN' aligned to 64 bytes
        //  float* sum_buf      [batch, L_SEQ_LEN', 1], L_SEQ_LEN' aligned to 64 bytes
        // fourth(text attn) kernel input:
        //  int seqlen          --> L_SEQ_LEN
        //  int kv_seq_len      --> v_seq_len
        //  int attn_weights_stride  unit is half
        //  int sum_buf_stride       unit is float
        //  half* attn_weights  [batch, HEAD_NUM, L_SEQ_LEN, v_seq_len'], v_seq_len' aligned to 64 bytes
        //  half* value,        [batch, v_seq_len, HEAD_NUM * HEAD_DIM], vison_values
        //  float* sum_buf      [batch*HEAD_NUM, L_SEQ_LEN', 1], L_SEQ_LEN' aligned to 64 bytes
        //  half* output        [batch, L_SEQ_LEN, HEAD_NUM * HEAD_DIM], it's outputs[1]

        auto query_layout = ins.get_impl_params()->get_input_layout(0);
        auto key_layout = ins.get_impl_params()->get_input_layout(1);
        // wg_size = 16
        // q_step = CM_GRF_WIDTH//32 # or 8 on Xe1
        // wg_seq_len = wg_size * q_step
        // wg_count = (seq_len + wg_seq_len - 1) // wg_seq_len
        // GWS = [1, self.num_heads, wg_count * wg_size]
        // LWS = [1, 1, wg_size]
        auto query_ps = query_layout.get_partial_shape();
        auto key_ps = key_layout.get_partial_shape();
        const size_t batch = query_ps[0].get_length();
        const size_t seq_len = query_ps[1].get_length();
        const size_t key_len = key_ps[1].get_length();
        cldnn::event::ptr v_attn_event;
        const auto& intermediates_memories = ins.get_intermediates_memories();
        {
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
            sdesc.v.s32 = (seq_len + 31) / 32 * 32;
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

            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
            args.outputs.push_back(intermediates_memories[0]);

            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
            args.outputs.push_back(ins.output_memory_ptr(0));

            desc.workGroups.global = global;
            desc.workGroups.local = local;

            v_attn_event = execute_stage(events,
                                ins,
                                *kernels[0],
                                desc,
                                args,
                                true);
        }
        cldnn::event::ptr max_event;
        {
            // kernel: max
            auto l_p = seq_len;
            auto l_t = key_len;
            auto items = static_cast<size_t>((l_p + 255) / 256);
            items = (items + 7) / 8 * 8;
            std::vector<size_t> global = {items,static_cast<size_t>(l_t), static_cast<size_t>(batch * m_num_heads)};
            std::vector<size_t> local = {8, 1, 1};
            cldnn::kernel_arguments_desc desc;
            cldnn::kernel_arguments_data args;
            auto max_v_mem = intermediates_memories[1];

            scalars_desc scalars_desc;
            args.scalars = &scalars_desc;
            {
                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
                scalar_desc sdesc;
                sdesc.t = scalar_desc::Types::INT32;
                sdesc.v.s32 = l_p;
                scalars_desc.push_back(sdesc);

                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 1});
                sdesc.t = scalar_desc::Types::INT32;
                sdesc.v.s32 = (l_p + 31) / 32 * 32;
                scalars_desc.push_back(sdesc);

                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 2});
                sdesc.t = scalar_desc::Types::INT32;
                sdesc.v.s32 = (l_t + 31) / 32 * 32;
                scalars_desc.push_back(sdesc);
            }

            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
            args.inputs.push_back(intermediates_memories[0]);
            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
            args.outputs.push_back(max_v_mem);

            desc.workGroups.global = global;
            desc.workGroups.local = local;
            std::vector<cldnn::event::ptr> events_new({v_attn_event});
            // 0xfafa-> -57152
            events_new.push_back(max_v_mem->fill(ins.get_network().get_stream(), 0xfa, false));
            max_event =  execute_stage(events_new,
                                ins,
                                *kernels[1],
                                desc,
                                args,
                                true);
        }
        cldnn::event::ptr exp_event;
        {
            // kernel: exp
            auto l_p = seq_len;
            auto l_t = key_len;
            auto items = static_cast<size_t>((l_p + 255) / 256);
            items = (items + 7) / 8 * 8;
            std::vector<size_t> global = {items,static_cast<size_t>(l_t), static_cast<size_t>(batch * m_num_heads)};
            std::vector<size_t> local = {8, 1, 1};
            cldnn::kernel_arguments_desc desc;
            cldnn::kernel_arguments_data args;

            scalars_desc scalars_desc;
            args.scalars = &scalars_desc;
            {
                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
                scalar_desc sdesc;
                sdesc.t = scalar_desc::Types::INT32;
                sdesc.v.s32 = l_p;
                scalars_desc.push_back(sdesc);

                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 1});
                sdesc.t = scalar_desc::Types::INT32;
                sdesc.v.s32 = (l_p + 31) / 32 * 32;;
                scalars_desc.push_back(sdesc);

                desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 2});
                sdesc.t = scalar_desc::Types::INT32;
                sdesc.v.s32 = (l_t + 31) / 32 * 32;;
                scalars_desc.push_back(sdesc);
            }

            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
            args.inputs.push_back(intermediates_memories[0]);
            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
            args.inputs.push_back(intermediates_memories[1]);
            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
            args.outputs.push_back(intermediates_memories[2]);

            desc.workGroups.global = global;
            desc.workGroups.local = local;
            std::vector<cldnn::event::ptr> events_new({max_event});
            events_new.push_back(intermediates_memories[2]->fill(ins.get_network().get_stream(), 0, false));
            exp_event = execute_stage(events_new,
                                ins,
                                *kernels[2],
                                desc,
                                args,
                                true);
        }
        {
            // kernel: text attn
            size_t wg_size = 4;
            auto wg_seq_len = 512;
            auto wg_count = (seq_len + wg_seq_len - 1) / wg_seq_len;
            std::vector<size_t> global = {batch, m_num_heads, static_cast<size_t>(wg_count * wg_size)};
            std::vector<size_t> local = {1, 1, wg_size};
            cldnn::kernel_arguments_desc desc;
            cldnn::kernel_arguments_data args;

            scalars_desc scalars_desc;
            scalar_desc sdesc;
            desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
            sdesc.t = scalar_desc::Types::INT32;
            sdesc.v.s32 = key_len;
            scalars_desc.push_back(sdesc);

            desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 1});
            sdesc.t = scalar_desc::Types::INT32;
            sdesc.v.s32 = seq_len;
            scalars_desc.push_back(sdesc);

            desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 2});
            sdesc.t = scalar_desc::Types::INT32;
            sdesc.v.s32 = (seq_len + 31) / 32 * 32;
            scalars_desc.push_back(sdesc);

            desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 3});
            sdesc.t = scalar_desc::Types::INT32;
            sdesc.v.s32 = (key_len + 31) / 32 * 32;
            scalars_desc.push_back(sdesc);

            args.scalars = &scalars_desc;

            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
            args.inputs.push_back(intermediates_memories[0]);

            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});
            args.inputs.push_back(ins.input_memory_ptr(2));

            desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
            args.inputs.push_back(intermediates_memories[2]);

            desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
            args.outputs.push_back(ins.output_memory_ptr(1));

            desc.workGroups.global = global;
            desc.workGroups.local = local;
            std::vector<cldnn::event::ptr> events_new({exp_event});
            events_new.push_back(ins.output_memory_ptr(1)->fill(ins.get_network().get_stream(), 0, false));

            return execute_stage(events_new,
                                ins,
                                *kernels[3],
                                desc,
                                args,
                                ins.needs_completion_event());
        }
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
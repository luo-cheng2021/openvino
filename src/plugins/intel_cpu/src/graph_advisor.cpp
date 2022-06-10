// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <map>
#include <vector>
#include <tuple>
#include <unordered_set>
#include <limits>
#include <fstream>
#include <unordered_map>
#include <memory>
#include <utility>

#include "graph_advisor.h"
#include "graph_dumper.h"
#include "graph_optimizer.h"
#include "dnnl_extension_utils.h"
#include "extension_mngr.h"
#include "memory_solver.hpp"
#include "itt.h"
#include "infer_request.h"
#include "nodes/input.h"
#include <nodes/reorder.h>
#include "nodes/convert.h"

#include <ie_algorithm.hpp>
#include <blob_factory.hpp>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"

#include "precision_utils.h"
#include <ie_plugin_config.hpp>

#include "utils/general_utils.h"
#include "utils/debug_capabilities.h"
#include "utils/node_dumper.h"
#include "utils/ngraph_utils.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/verbose.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include <cpu/x64/cpu_isa_traits.hpp>

#include <ngraph/node.hpp>
#include <ngraph/function.hpp>
#include <ngraph/variant.hpp>
#include <ngraph/ops.hpp>
#include <transformations/utils/utils.hpp>
#include <low_precision/low_precision.hpp>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <cpu/cpu_primitive.hpp>

using namespace dnnl;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_cpu {
using tag = memory::format_tag;
using dt = memory::data_type;

static tag getTag(int shape_size, bool channel_first) {
    tag t = tag::any;
    switch (shape_size) {
    case 3:
        t = channel_first ? tag::nwc : tag::nCw16c;
        break;
    case 4:
        t = channel_first ? tag::nhwc : tag::nChw16c;
        break;
    case 5:
        t = channel_first ? tag::ndhwc : tag::nCdhw16c;
        break;
    default:
        assert(false);
        break;
    }
    return t;
}

struct ConvKernel {
    std::shared_ptr<convolution_forward> conv_brg;
    std::shared_ptr<std::unordered_map<int, memory>> conv_args_brg;
    std::shared_ptr<convolution_forward> conv_jit;
    std::shared_ptr<std::unordered_map<int, memory>> conv_args_jit;
};

static std::tuple<std::shared_ptr<convolution_forward>, std::shared_ptr<std::unordered_map<int, memory>>>
makeConv(engine& eng, std::shared_ptr<ov::Node> op, float* src, float* dst, float* weights, bool channel_first) {
    auto conv_ov = std::dynamic_pointer_cast<ngraph::op::v1::Convolution>(op);
    ov::CoordinateDiff paddingL, paddingR;
    std::vector<ptrdiff_t> dilation;
    ov::Strides strides;
    if (conv_ov) {
        paddingL = conv_ov->get_pads_begin();
        paddingR = conv_ov->get_pads_end();
        for (int i = 0; i < conv_ov->get_dilations().size(); i++) {
            dilation.push_back(static_cast<ptrdiff_t>(conv_ov->get_dilations()[i]) - 1);
        }
        strides = conv_ov->get_strides();
    } else {
        auto conv_ov = std::dynamic_pointer_cast<ngraph::op::v1::GroupConvolution>(op);
        if (!conv_ov)
            return {nullptr, nullptr};
        paddingL = conv_ov->get_pads_begin();
        paddingR = conv_ov->get_pads_end();
        for (int i = 0; i < conv_ov->get_dilations().size(); i++) {
            dilation.push_back(static_cast<ptrdiff_t>(conv_ov->get_dilations()[i]) - 1);
        }
        strides = conv_ov->get_strides();
    }

    auto shape = op->get_input_shape(0);
    auto src_dims = DnnlExtensionUtils::convertToDnnlDims(shape);
    auto weights_dims = DnnlExtensionUtils::convertToDnnlDims(op->get_input_shape(1));
    auto dst_dims = DnnlExtensionUtils::convertToDnnlDims(op->get_output_shape(0));
    auto src_tag = getTag(shape.size(), channel_first);
    auto conv_src_md = memory::desc(src_dims, dt::f32, src_tag);
    auto conv_weights_md = memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_dst_md = memory::desc(dst_dims, dt::f32, src_tag);

    auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
            algorithm::convolution_direct, conv_src_md, conv_weights_md,
            conv_dst_md,
            dnnl::memory::dims(strides.begin(), strides.end()),
            dnnl::memory::dims(dilation.begin(), dilation.end()),
            dnnl::memory::dims(paddingL.begin(), paddingL.end()),
            dnnl::memory::dims(paddingR.begin(), paddingR.end()));
    primitive_attr conv_attr;
    // Create primitive descriptor.
    auto conv_pd
            = convolution_forward::primitive_desc(conv_desc, conv_attr, eng);
    impl_desc_type impl_type = parse_impl_name(conv_pd.impl_info_str());
    if (channel_first && !(impl_type & brgconv))
        return {nullptr, nullptr};
    if (!channel_first && !(impl_type & jit_avx512))
        return {nullptr, nullptr};
    auto conv_args = std::make_shared<std::unordered_map<int, memory>>();
    auto conv_src_mem = memory(conv_pd.src_desc(), eng, src);
    auto conv_weights_mem = memory(conv_pd.weights_desc(), eng, weights);
    auto conv_dst_mem = memory(conv_pd.dst_desc(), eng, dst);
    conv_args->insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args->insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args->insert({DNNL_ARG_DST, conv_dst_mem});
    auto conv = std::make_shared<convolution_forward>(conv_pd);
    return {conv, conv_args};
}

static ConvKernel makeConvs(engine& eng, std::shared_ptr<ov::Node> op, float* src, float* dst, float* weights) {
    auto brg = makeConv(eng, op, src, dst, weights, true);
    auto jit = makeConv(eng, op, src, dst, weights, false);
    return {std::get<0>(brg), std::get<1>(brg), std::get<0>(jit), std::get<1>(jit)};
}

static size_t align16(size_t s) {
    return (s + 15) / 16 * 16;
}

static size_t getEstimateElementSize(const ov::Shape& shape) {
    size_t size = shape[0];
    for (auto i = 1; i < shape.size(); i++) {
        size *= align16(shape[i]);
    }
    return size;
}

bool Advisor::AdviseBrgconv(dnnl::engine& engine, std::shared_ptr<const ov::Model> func) {
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::nanoseconds ns;
    // begin time
    auto beg_time = Time::now();
    // brgconv needs avx512 support
    if (!impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core)) {
        return false;
    }
    // if (impl::cpu::x64::mayiuse(impl::cpu::x64::avx512_core_amx)) {
    //     return false;
    // }
    dnnl::stream engine_stream(engine);

    auto orderedOps = func->get_ordered_ops();
    NodeVector nodes;
    std::vector<std::pair<std::shared_ptr<convolution_forward>, std::unordered_map<int, memory>>> convs;
    size_t src_size = 0;
    size_t dst_size = 0;
    size_t weights_size = 0;
    int dynamic_nodes_num = 0;
    // find conv nodes
    for (const auto& op : orderedOps) {
        if (ngraph::is_type<ngraph::op::v1::Convolution>(op) || ngraph::is_type<ngraph::op::v1::GroupConvolution>(op)) {
            nodes.push_back(op);
            const auto ps = op->get_input_partial_shape(0);
            // does not measure dynamic shape
            if (ps.is_dynamic()) {
                dynamic_nodes_num++;
                continue;
            }
            auto size = getEstimateElementSize(ps.get_shape());
            if (size > src_size) {
                src_size = size;
            }
            size = getEstimateElementSize(op->get_output_partial_shape(0).get_shape());
            if (size > dst_size) {
                dst_size = size;
            }
            size = getEstimateElementSize(op->get_input_partial_shape(1).get_shape());
            if (size > weights_size) {
                weights_size = size;
            }
        }
    }

    if (nodes.empty() || dynamic_nodes_num > nodes.size()) {
        return false;
    }

    std::vector<float> src(src_size, 0);
    std::vector<float> dst(dst_size, 0);
    std::vector<float> weights(weights_size, 0);

    std::vector<ConvKernel> kernels;
    // create conv nodes
    for (const auto& op : nodes) {
        kernels.emplace_back(makeConvs(engine, op, &src[0], &dst[0], &weights[0]));
    }

    // first iteration
    for (const auto& ker : kernels) {
        if (ker.conv_brg && ker.conv_jit) {
            ker.conv_brg->execute(engine_stream, *ker.conv_args_brg);
            ker.conv_jit->execute(engine_stream, *ker.conv_args_jit);
        }
    }
    // brg time
    auto start_time = Time::now();
    for (const auto& ker : kernels) {
        if (ker.conv_brg && ker.conv_jit) {
            ker.conv_brg->execute(engine_stream, *ker.conv_args_brg);
        }
    }
    auto end_time_brg = Time::now();
    // jit time
    for (const auto& ker : kernels) {
        if (ker.conv_brg && ker.conv_jit) {
            ker.conv_jit->execute(engine_stream, *ker.conv_args_jit);
        }
    }
    auto end_time = Time::now();

    auto exec_time_brg = std::chrono::duration_cast<ns>(end_time_brg - start_time).count();
    auto exec_time_jit = std::chrono::duration_cast<ns>(end_time - end_time_brg).count();
    auto exec_time = std::chrono::duration_cast<ns>(end_time - beg_time).count();
    if (exec_time_jit >= exec_time_brg) {
        printf("########################### should use brg (%ld, %ld, %ld) \n", (int64_t)exec_time_brg, (int64_t)exec_time_jit, (int64_t)exec_time);
    } else {
        printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@ should use jit (%ld, %ld, %ld) \n", (int64_t)exec_time_brg, (int64_t)exec_time_jit, (int64_t)exec_time);
    }

    return false;
}

}   // namespace intel_cpu
}   // namespace ov

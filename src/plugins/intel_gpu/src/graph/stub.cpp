// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stub_inst.h"
#include "openvino/core/except.hpp"
#include "program_node.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include "primitive_type_base.h"
#include "openvino/core/parallel.hpp"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(stub)

// TODO: automatic register
// https://stackoverflow.com/questions/9459980/c-global-variable-not-initialized-when-linked-through-static-libraries-but-ok
struct custom_kernel_reference {
    std::unordered_map<std::string, custom_kernel_info> kernel_map;
    custom_kernel_reference() {
        CALL_REG_CUSTOM_KERNEL(WSAttention)
        CALL_REG_CUSTOM_KERNEL(SWSAttention)
        CALL_REG_CUSTOM_KERNEL(PadRollPermute)
        CALL_REG_CUSTOM_KERNEL(PermuteRollCrop)
        CALL_REG_CUSTOM_KERNEL(AddInDeformable)
        CALL_REG_CUSTOM_KERNEL(PermuteAddConcat)
        CALL_REG_CUSTOM_KERNEL(MaxSubClip)
        CALL_REG_CUSTOM_KERNEL(BiAttention)
    };
} g_custom_kernel_reference;

void register_custom_kernel(const std::string& type, const custom_kernel_info& info) {
    g_custom_kernel_reference.kernel_map[type] = info;
}

typed_program_node<stub>::typed_program_node(std::shared_ptr<stub> prim, program& prog) : parent(prim, prog) {
    OPENVINO_ASSERT(g_custom_kernel_reference.kernel_map.count(prim->m_params["type"]), "custom kernel '", prim->m_params["type"], "' not support");
    const auto& info = g_custom_kernel_reference.kernel_map.at(prim->m_params["type"]);
    m_kernel = info.make_custom_kernel(prim, *this);
}

/*
    Calc_output_layout method is called only when output layout is invalidated.
    It means, that it is called when:
    1) It has never been called.
    2) Dependency has changed output layout.
    In this both cases, we need to recalc branch_true and branch_false.
    !* We can be sure, that this method was called AT LEAST once during graph compilation.*!
*/
layout stub_inst::calc_output_layout(stub_node const& node, kernel_impl_params const& impl_param) {
    return node.m_kernel->calc_output_layout(node, impl_param);
}

template<typename ShapeType>
std::vector<layout> stub_inst::calc_output_layouts(stub_node const& node, kernel_impl_params const& impl_param) {
    return node.m_kernel->calc_output_layouts(node, impl_param);
}

template std::vector<layout> stub_inst::calc_output_layouts<ov::PartialShape>(stub_node const& node, const kernel_impl_params& impl_param);

std::string stub_inst::to_string(stub_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    json_composite stubop_info;

    node_info->add("stub info", stubop_info);

    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

/*
stub primitive is reusing memory with the input.
*/
stub_inst::typed_primitive_inst(network& network, stub_node const& node)
    : parent(network, node) {
}

}  // namespace cldnn
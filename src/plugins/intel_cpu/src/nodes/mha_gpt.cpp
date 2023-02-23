// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "ie_parallel.hpp"
#include "mha_gpt.h"
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"
#include <utils/general_utils.h>
#include <cpu/x64/jit_generator.hpp>
#include "emitters/jit_dnnl_emitters.hpp"
#include "emitters/jit_load_store_emitters.hpp"
#include "common/cpu_convert.h"
#include "ngraph_transformations/op/mha.hpp"
#include "dnnl_extension_utils.h"
#include <ie_ngraph_utils.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace dnnl::impl::cpu::x64;
using namespace dnnl::impl::cpu::x64::matmul;
using namespace Xbyak;

#define THROW_ERROR IE_THROW() << " in MHAGPT"

namespace ov {
namespace intel_cpu {
namespace gpt {

struct jit_mul_add_softmax_compile_params {
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    bool with_mul_scales;
    bool is_mul_first;
    bool with_scales0;
    bool broadcast_scales0;
    bool with_scales1;
    bool broadcast_scales1;
    size_t tail_size;
};

struct jit_mul_add_softmax_call_args {
    const void *p_in0;
    const void *p_mul_in1;
    const void *p_add_in1;
    void *p_out;
    void *p_buffer;
    const void *p_scales0;
    const void *p_scales1;
    size_t work_amount;
};

struct jit_uni_mul_add_softmax_kernel {
    void (*ker_)(const jit_mul_add_softmax_call_args*);

    void operator()(const jit_mul_add_softmax_call_args* call_args) {
        assert(ker_);
        ker_(call_args);
    }

    explicit jit_uni_mul_add_softmax_kernel(const jit_mul_add_softmax_compile_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_mul_add_softmax_kernel() {}

    virtual void create_ker() = 0;

    jit_mul_add_softmax_compile_params jcp_;
};

struct jit_convert_reorder_compile_params {
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    size_t inner_work_amount;
    bool with_scales;
    bool broadcast_scales;
    size_t src_stride;
    size_t dst_stride;
};

struct jit_convert_reorder_call_args {
    const void *p_in;
    void *p_out;
    const void *p_scales;
    size_t outter_work_amount;
};

struct jit_uni_convert_reorder_kernel {
    void (*ker_)(const jit_convert_reorder_call_args*);

    void operator()(const jit_convert_reorder_call_args* call_args) {
        assert(ker_);
        ker_(call_args);
    }

    explicit jit_uni_convert_reorder_kernel(const jit_convert_reorder_compile_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_convert_reorder_kernel() {}

    virtual void create_ker() = 0;

    jit_convert_reorder_compile_params jcp_;
};

struct jit_convert_transpose_compile_params {
    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision dst_prc;
    size_t inner_work_amount;
    size_t outter_work_amount;
    bool with_scales;
    bool broadcast_scales;
    size_t inner_src_stride;
    size_t outter_src_stride;
    size_t outter_dst_stride;
};

struct jit_convert_transpose_call_args {
    const void *p_in;
    void *p_out;
    const void *p_scales;
};

struct jit_uni_convert_transpose_kernel {
    void (*ker_)(const jit_convert_transpose_call_args*);

    void operator()(const jit_convert_transpose_call_args* call_args) {
        assert(ker_);
        ker_(call_args);
    }

    explicit jit_uni_convert_transpose_kernel(const jit_convert_transpose_compile_params& jcp) : ker_(nullptr), jcp_(jcp) {}
    virtual ~jit_uni_convert_transpose_kernel() {}

    virtual void create_ker() = 0;

    jit_convert_transpose_compile_params jcp_;
};

#define MHA_BRGEMM_KERNELS_NUM 8

template <cpu_isa_t isa>
struct jit_mul_add_softmax_kernel : public jit_uni_mul_add_softmax_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_mul_add_softmax_kernel)

    explicit jit_mul_add_softmax_kernel(const jit_mul_add_softmax_compile_params& jcp) : jit_uni_mul_add_softmax_kernel(jcp), jit_generator(jit_name()) {
        exp_emitter = std::make_shared<jit_dnnl_aux_emitter>(this, isa, dnnl_eltwise_exp, 0.f, 0.f);

        vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    }
    virtual ~jit_mul_add_softmax_kernel() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41, Xmm, isa == cpu_isa_t::avx2, Ymm, Zmm>::type;

    void generate() override {
        this->preamble();

#define GET_OFF(field) offsetof(jit_mul_add_softmax_call_args, field)
        mov(reg_in0, ptr[reg_params + GET_OFF(p_in0)]);
        mov(reg_add_in1, ptr[reg_params + GET_OFF(p_add_in1)]);
        mov(reg_out, ptr[reg_params + GET_OFF(p_out)]);
        mov(reg_buffer, ptr[reg_params + GET_OFF(p_buffer)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(work_amount)]);

        Xbyak::Label mul_add_max_loop_label;
        Xbyak::Label mul_add_max_end_label;
        Xbyak::Label sub_exp_reduce_loop_label;
        Xbyak::Label sub_exp_reduce_end_label;
        Xbyak::Label mul_loop_label;
        Xbyak::Label mul_end_label;

        size_t tail_size = jcp_.tail_size;

        mov(reg_buffer_aux, reg_buffer);
        mov(reg_work_amount_aux, reg_work_amount);
        mov(reg_tmp, dnnl::impl::float2int(-FLT_MAX));
        vmovq(xmm_tmp, reg_tmp);
        vbroadcastss(get_vmm_max(0), xmm_tmp);

        // mul1 input is const and always float
        if (jcp_.with_mul_scales) {
            mov(reg_mul_in1, ptr[reg_params + GET_OFF(p_mul_in1)]);
            uni_vmovss(Xmm(get_vmm_in(1).getIdx()), ptr[reg_mul_in1]);
            uni_vbroadcastss(get_vmm_in(1), Xmm(get_vmm_in(1).getIdx()));
        }

        if (jcp_.with_scales0) {
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales0)]);

            mov(reg_tmp, dnnl::impl::float2int(-128.0f));
            vmovq(xmm_tmp, reg_tmp);
            vbroadcastss(vmm_crop_low, xmm_tmp);

            mov(reg_tmp, dnnl::impl::float2int(127.0f));
            vmovq(xmm_tmp, reg_tmp);
            vbroadcastss(vmm_crop_high, xmm_tmp);
        }

        if (jcp_.with_scales0 && jcp_.broadcast_scales0) {
            uni_vmovss(Xmm(vmm_scales.getIdx()), ptr[reg_scales]);
            uni_vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
        }

        L(mul_add_max_loop_label);
        {
            cmp(reg_work_amount_aux, vec_size);
            jl(mul_add_max_end_label, T_NEAR);

            mul_add_max(vec_size);

            sub(reg_work_amount_aux, vec_size);

            jmp(mul_add_max_loop_label, T_NEAR);
        }
        L(mul_add_max_end_label);
        if (tail_size) {
            mul_add_max(tail_size);
        }

        sub(rsp, sizeof(float) * vec_size);
        uni_vmovups(ptr[rsp], get_vmm_max(0));
        mov(reg_tmp, dnnl::impl::float2int(-FLT_MAX));
        vmovq(xmm_tmp, reg_tmp);
        vbroadcastss(get_vmm_max(0), xmm_tmp);
        for (size_t i = 0; i < vec_size; i++) {
            mov(reg_tmp_32, ptr[rsp + i * sizeof(float)]);
            vmovq(xmm_tmp, reg_tmp);
            uni_vmaxps(get_xmm_max(0), get_xmm_max(0), xmm_tmp);
        }
        uni_vbroadcastss(get_vmm_max(0), get_xmm_max(0));
        add(rsp, sizeof(float) * vec_size);

        uni_vpxor(get_vmm_denom(0), get_vmm_denom(0), get_vmm_denom(0));
        mov(reg_work_amount_aux, reg_work_amount);
        mov(reg_buffer_aux, reg_buffer);
        L(sub_exp_reduce_loop_label);
        {
            cmp(reg_work_amount_aux, vec_size);
            jl(sub_exp_reduce_end_label, T_NEAR);

            sub_exp_reduce(vec_size);

            sub(reg_work_amount_aux, vec_size);

            jmp(sub_exp_reduce_loop_label, T_NEAR);
        }
        L(sub_exp_reduce_end_label);
        if (tail_size) {
            sub_exp_reduce(tail_size);
        }

        sub(rsp, sizeof(float) * vec_size);
        uni_vmovups(ptr[rsp], get_vmm_denom(0));
        uni_vpxor(get_vmm_aux(0), get_vmm_aux(0), get_vmm_aux(0));
        for (size_t i = 0; i < vec_size; i++) {
            mov(reg_tmp_32, ptr[rsp + i * sizeof(float)]);
            vmovq(xmm_tmp, reg_tmp);
            uni_vaddps(get_xmm_aux(0), get_xmm_aux(0), xmm_tmp);
        }
        vbroadcastss(get_vmm_aux(0), get_xmm_aux(0));
        add(rsp, sizeof(float) * vec_size);

        mov(reg_tmp, dnnl::impl::float2int(1.0f));
        vmovq(xmm_tmp, reg_tmp);
        vbroadcastss(get_vmm_denom(0), xmm_tmp);
        uni_vdivps(get_vmm_denom(0), get_vmm_denom(0), get_vmm_aux(0));

        if (jcp_.with_scales1)
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales1)]);

        if (jcp_.with_scales1 && jcp_.broadcast_scales1) {
            uni_vmovss(Xmm(vmm_scales.getIdx()), ptr[reg_scales]);
            uni_vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
        }

        mov(reg_work_amount_aux, reg_work_amount);
        L(mul_loop_label);
        {
            cmp(reg_work_amount_aux, vec_size);
            jl(mul_end_label, T_NEAR);

            mul_loop(vec_size);

            sub(reg_work_amount_aux, vec_size);

            jmp(mul_loop_label, T_NEAR);
        }
        L(mul_end_label);
        if (tail_size) {
            mul_loop(tail_size);
        }

        this->postamble();

        for (const auto& emitter : emitters) {
            if (emitter.second)
                emitter.second->emit_data();
        }

        exp_emitter->emit_data();
    }

    void mul_add_max(size_t step) {
        bool is_tail = step < vec_size;

        load(get_vmm_in(0), reg_in0, jcp_.src_prc, step, is_tail);
        load(get_vmm_in(2), reg_add_in1, Precision::FP32, step, is_tail);

        if (jcp_.with_scales0) {
            if (!jcp_.broadcast_scales0) {
                load(vmm_scales, reg_scales, Precision::FP32, step, is_tail);
                add(reg_scales,  sizeof(float) * step);
            }
            uni_vmulps(get_vmm_in(0), get_vmm_in(0), vmm_scales);
            uni_vmaxps(get_vmm_in(0), get_vmm_in(0), vmm_crop_low);
            uni_vminps(get_vmm_in(0), get_vmm_in(0), vmm_crop_high);
        }

        if (jcp_.with_mul_scales) {
            if (jcp_.is_mul_first) {
                uni_vmulps(get_vmm_in(0), get_vmm_in(0), get_vmm_in(1));
                uni_vaddps(get_vmm_in(0), get_vmm_in(0), get_vmm_in(2));
            } else {
                uni_vaddps(get_vmm_in(0), get_vmm_in(0), get_vmm_in(2));
                uni_vmulps(get_vmm_in(0), get_vmm_in(0), get_vmm_in(1));
            }
        } else {
            uni_vaddps(get_vmm_in(0), get_vmm_in(0), get_vmm_in(2));
        }

        uni_vmaxps(get_vmm_max(0), get_vmm_max(0), get_vmm_in(0));

        store(reg_buffer_aux, get_vmm_in(0), Precision::FP32, step);

        if (!is_tail) {
            add(reg_in0, jcp_.src_prc.size() * step);
            add(reg_add_in1, sizeof(float) * step);
            add(reg_buffer_aux, sizeof(float) * step);
        }
    }

    void sub_exp_reduce(size_t step) {
        bool is_tail = step < vec_size;

        load(get_vmm_in(0), reg_buffer_aux, Precision::FP32, step, is_tail);

        uni_vsubps(get_vmm_in(0), get_vmm_in(0), get_vmm_max(0));

        auto vmm_exp_idx = static_cast<size_t>(get_vmm_in(0).getIdx());
        exp_emitter->emit_code({vmm_exp_idx}, {vmm_exp_idx}, pool_aux_vmm_idxs, pool_aux_gpr_idxs);

        uni_vaddps(get_vmm_denom(0), get_vmm_denom(0), get_vmm_in(0));

        store(reg_buffer_aux, get_vmm_in(0), Precision::FP32, step);

        if (!is_tail) {
            add(reg_buffer_aux, sizeof(float) * step);
        }
    }

    void mul_loop(size_t step) {
        bool is_tail = step < vec_size;

        load(get_vmm_in(0), reg_buffer, Precision::FP32, step, is_tail);

        uni_vmulps(get_vmm_in(0), get_vmm_in(0), get_vmm_denom(0));

        if (jcp_.src_prc == Precision::I32) {
            if (jcp_.with_scales1) {
                if (!jcp_.broadcast_scales1) {
                    load(vmm_scales, reg_scales, Precision::FP32, step, is_tail);
                    add(reg_scales,  sizeof(float) * step);
                }
                uni_vmulps(get_vmm_in(0), get_vmm_in(0), vmm_scales);
            }
        }

        store(reg_out, get_vmm_in(0), jcp_.dst_prc, step);

        if (!is_tail) {
            add(reg_buffer, sizeof(float) * step);
            add(reg_out, jcp_.dst_prc.size() * step);
        }
#undef GET_OFF
    }

    inline void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, Precision src_prc, const int& elt_num, bool fill) {
        const auto seed = load_emitter_params(src_prc, Precision::FP32, elt_num, fill, "float_min").hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_load_emitter(this, isa, src_prc, Precision::FP32, elt_num, Precision::FP32, fill, "float_min"));
        }

        emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), 0}, {static_cast<size_t>(vmm_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }
    inline void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, Precision dst_prc, const int& elt_num) {
        const auto seed = store_emitter_params(Precision::FP32, dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_store_emitter(this, isa, Precision::FP32, dst_prc, elt_num));
        }

        emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx()), 0}, {static_cast<size_t>(reg_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }

    size_t unroll_factor = 3;
    size_t vec_size;

    Vmm get_vmm_in(int idx) {
        return Vmm(1 + 0 * unroll_factor + idx);
    }

    Vmm get_vmm_aux(int idx) {
        return Vmm(1 + 1 * unroll_factor + idx);
    }
    Xmm get_xmm_aux(int idx) {
        return Xmm(1 + 1 * unroll_factor + idx);
    }

    Vmm get_vmm_max(int idx) {
        return Vmm(1 + 2 * unroll_factor + idx);
    }
    Xmm get_xmm_max(int idx) {
        return Xmm(1 + 2 * unroll_factor + idx);
    }


    Vmm get_vmm_denom(int idx) {
        return Vmm(1 + 3 * unroll_factor + idx);
    }

    Xmm xmm_tmp = Xmm(0);

    Vmm vmm_scales = Vmm(0);
    Vmm vmm_crop_low = Vmm(14);
    Vmm vmm_crop_high = Vmm(15);

    Reg64 reg_in0 = r8;
    Reg64 reg_mul_in1 = r9;
    Reg64 reg_add_in1 = r10;
    Reg64 reg_out = r11;
    Reg64 reg_scales = r12;
    Reg64 reg_work_amount = r13;
    Reg64 reg_work_amount_aux = r14;
    Reg64 reg_buffer = r15;
    Reg64 reg_buffer_aux = rax;
    Reg64 reg_tmp = rbx;
    Reg32 reg_tmp_32 = Reg32(rbx.getIdx());
    Reg64 reg_max = rdx;
    Reg32 reg_max_32 = Reg32(rdx.getIdx());
    Reg64 reg_params = abi_param1;

    const std::vector<size_t> pool_aux_gpr_idxs = { static_cast<size_t>(rsi.getIdx()), static_cast<size_t>(rbp.getIdx()) };
    const std::vector<size_t> pool_aux_vmm_idxs = { 12, 13, 14, 15 };

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;

    std::shared_ptr<jit_dnnl_aux_emitter> exp_emitter = nullptr;
    std::unique_ptr<jit_load_emitter> load_emitter = nullptr;
    std::unique_ptr<jit_store_emitter> store_emitter = nullptr;
};

template <cpu_isa_t isa>
struct jit_convert_reorder_kernel : public jit_uni_convert_reorder_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_convert_reorder_kernel)

    explicit jit_convert_reorder_kernel(const jit_convert_reorder_compile_params& jcp) : jit_uni_convert_reorder_kernel(jcp), jit_generator(jit_name()) {
        vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / sizeof(float);
    }
    virtual ~jit_convert_reorder_kernel() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41, Xmm, isa == cpu_isa_t::avx2, Ymm, Zmm>::type;

    void generate() override {
        this->preamble();

#define GET_OFF(field) offsetof(jit_convert_reorder_call_args, field)
        mov(reg_in, ptr[reg_params + GET_OFF(p_in)]);
        mov(reg_out, ptr[reg_params + GET_OFF(p_out)]);
        mov(reg_outter_work_amount, ptr[reg_params + GET_OFF(outter_work_amount)]);

        if (jcp_.with_scales) {
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales)]);
        }

        Xbyak::Label convert_reorder_inner_loop_label;
        Xbyak::Label convert_reorder_inner_end_label;
        Xbyak::Label convert_reorder_outter_loop_label;
        Xbyak::Label convert_reorder_outter_end_label;

        if (jcp_.with_scales && jcp_.broadcast_scales) {
            uni_vmovss(Xmm(vmm_scales.getIdx()), ptr[reg_scales]);
            uni_vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
        }

        L(convert_reorder_outter_loop_label);
        {
            cmp(reg_outter_work_amount, 1);
            jl(convert_reorder_outter_end_label, T_NEAR);

            size_t tail_size = jcp_.inner_work_amount % vec_size;
            mov(reg_inner_work_amount, jcp_.inner_work_amount);
            mov(reg_in_aux, reg_in);
            mov(reg_out_aux, reg_out);
            if (jcp_.with_scales && !jcp_.broadcast_scales) {
                mov(reg_scales, ptr[reg_params + GET_OFF(p_scales)]);
            }

            L(convert_reorder_inner_loop_label);
            {
                cmp(reg_inner_work_amount, vec_size);
                jl(convert_reorder_inner_end_label, T_NEAR);

                convert_reorder(vec_size);

                sub(reg_inner_work_amount, vec_size);

                jmp(convert_reorder_inner_loop_label, T_NEAR);
            }
            L(convert_reorder_inner_end_label);
            if (tail_size) {
                convert_reorder(tail_size);
            }

            dec(reg_outter_work_amount);
            add(reg_in, jcp_.src_prc.size() * jcp_.src_stride);
            add(reg_out, jcp_.dst_prc.size() * jcp_.dst_stride);

            jmp(convert_reorder_outter_loop_label, T_NEAR);
        }
        L(convert_reorder_outter_end_label);

        this->postamble();

        for (const auto& emitter : emitters) {
            if (emitter.second)
                emitter.second->emit_data();
        }
    }

    void convert_reorder(size_t step) {
        bool is_tail = step < vec_size;

        load(vmm_in, reg_in_aux, jcp_.src_prc, step, is_tail);

        if (jcp_.with_scales) {
            if (!jcp_.broadcast_scales) {
                load(vmm_scales, reg_scales, Precision::FP32, step, is_tail);
                add(reg_scales,  sizeof(float) * step);
            }
            uni_vmulps(vmm_in, vmm_in, vmm_scales);
        }

        store(reg_out_aux, vmm_in, jcp_.dst_prc, step);

        if (!is_tail) {
            add(reg_in_aux, jcp_.src_prc.size() * step);
            add(reg_out_aux, jcp_.dst_prc.size() * step);
        }
    }
#undef GET_OFF

    inline void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, Precision src_prc, const int& elt_num, bool fill) {
        const auto seed = load_emitter_params(src_prc, Precision::FP32, elt_num, fill, "float_min").hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_load_emitter(this, isa, src_prc, Precision::FP32, elt_num, Precision::FP32, fill, "float_min"));
        }

        emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), 0}, {static_cast<size_t>(vmm_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }
    inline void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, Precision dst_prc, const int& elt_num) {
        const auto seed = store_emitter_params(Precision::FP32, dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_store_emitter(this, isa, Precision::FP32, dst_prc, elt_num));
        }

        emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx()), 0}, {static_cast<size_t>(reg_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }

    size_t vec_size;

    Xmm xmm_tmp = Xmm(2);
    Vmm vmm_scales = Vmm(0);
    Vmm vmm_in = Vmm(1);

    Reg64 reg_in = r8;
    Reg64 reg_in_aux = r9;
    Reg64 reg_out = r10;
    Reg64 reg_out_aux = r11;
    Reg64 reg_scales = r12;
    Reg64 reg_inner_work_amount = r14;
    Reg64 reg_outter_work_amount = r15;
    Reg64 reg_params = abi_param1;

    const std::vector<size_t> pool_aux_gpr_idxs = { static_cast<size_t>(rsi.getIdx()), static_cast<size_t>(rbp.getIdx()) };
    const std::vector<size_t> pool_aux_vmm_idxs = { static_cast<size_t>(xmm_tmp.getIdx()) };

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
};

template <cpu_isa_t isa>
struct jit_convert_transpose_kernel : public jit_uni_convert_transpose_kernel, public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_convert_transpose_kernel)

    explicit jit_convert_transpose_kernel(const jit_convert_transpose_compile_params& jcp) : jit_uni_convert_transpose_kernel(jcp), jit_generator(jit_name()) {
        interm_prc = jcp_.with_scales ? Precision(Precision::FP32) : jcp_.src_prc;
        vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<isa>::vlen / interm_prc.size();
    }
    virtual ~jit_convert_transpose_kernel() {}

    void create_ker() override {
        jit_generator::create_kernel();
        ker_ = (decltype(ker_))jit_ker();
    }

private:
    using Vmm = typename dnnl::impl::utils::conditional3<isa == cpu_isa_t::sse41, Xmm, isa == cpu_isa_t::avx2, Ymm, Zmm>::type;

    void generate() override {
        this->preamble();

#define GET_OFF(field) offsetof(jit_convert_transpose_call_args, field)
        mov(reg_in, ptr[reg_params + GET_OFF(p_in)]);
        mov(reg_out, ptr[reg_params + GET_OFF(p_out)]);
        if (jcp_.with_scales) {
            mov(reg_scales, ptr[reg_params + GET_OFF(p_scales)]);
        }

        Xbyak::Label convert_transpose_inner_loop_label;
        Xbyak::Label convert_transpose_inner_end_label;
        Xbyak::Label convert_transpose_outter_loop_label;
        Xbyak::Label convert_transpose_outter_end_label;

        if (jcp_.with_scales && jcp_.broadcast_scales) {
            uni_vmovss(Xmm(vmm_scales.getIdx()), ptr[reg_scales]);
            uni_vbroadcastss(vmm_scales, Xmm(vmm_scales.getIdx()));
        }

        mov(reg_outter_work_amount, jcp_.outter_work_amount);
        L(convert_transpose_outter_loop_label);
        {
            cmp(reg_outter_work_amount, 1);
            jl(convert_transpose_outter_end_label, T_NEAR);

            size_t tail_size = jcp_.inner_work_amount % vec_size;
            mov(reg_inner_work_amount, jcp_.inner_work_amount);
            mov(reg_in_aux, reg_in);
            mov(reg_out_aux, reg_out);
            if (jcp_.with_scales && !jcp_.broadcast_scales) {
                mov(reg_scales, ptr[reg_params + GET_OFF(p_scales)]);
            }

            L(convert_transpose_inner_loop_label);
            {
                cmp(reg_inner_work_amount, vec_size);
                jl(convert_transpose_inner_end_label, T_NEAR);

                convert_transpose(vec_size);

                sub(reg_inner_work_amount, vec_size);

                jmp(convert_transpose_inner_loop_label, T_NEAR);
            }
            L(convert_transpose_inner_end_label);
            if (tail_size) {
                convert_transpose(tail_size);
            }

            dec(reg_outter_work_amount);
            add(reg_in, jcp_.src_prc.size() * jcp_.outter_src_stride);
            add(reg_out, jcp_.dst_prc.size() * jcp_.outter_dst_stride);

            jmp(convert_transpose_outter_loop_label, T_NEAR);
        }
        L(convert_transpose_outter_end_label);

        this->postamble();

        for (const auto& emitter : emitters) {
            if (emitter.second)
                emitter.second->emit_data();
        }
    }

    void convert_transpose(size_t step) {
        bool is_tail = step < vec_size;

        sub(rsp, jcp_.src_prc.size() * vec_size);
        for (size_t i = 0; i < step; i++) {
            if (jcp_.src_prc.size() == 4) {
                mov(reg_tmp_32, ptr[reg_in_aux + i * jcp_.inner_src_stride * jcp_.src_prc.size()]);
                mov(ptr[rsp + i * jcp_.src_prc.size()], reg_tmp_32);
            } else if (jcp_.src_prc.size() == 2) {
                mov(reg_tmp_16, ptr[reg_in_aux + i * jcp_.inner_src_stride * jcp_.src_prc.size()]);
                mov(ptr[rsp + i * jcp_.src_prc.size()], reg_tmp_16);
            } else if (jcp_.src_prc.size() == 1) {
                mov(reg_tmp_8, ptr[reg_in_aux + i * jcp_.inner_src_stride * jcp_.src_prc.size()]);
                mov(ptr[rsp + i * jcp_.src_prc.size()], reg_tmp_8);
            }
        }
        load(vmm_in, rsp, jcp_.src_prc, interm_prc, vec_size, false);
        add(rsp, jcp_.src_prc.size() * vec_size);

        if (jcp_.with_scales) {
            if (!jcp_.broadcast_scales) {
                load(vmm_scales, reg_scales, Precision::FP32, Precision::FP32, step, false);
                add(reg_scales, sizeof(float) * step);
            }
            uni_vmulps(vmm_in, vmm_in, vmm_scales);
        }

        store(reg_out_aux, vmm_in, interm_prc, jcp_.dst_prc, step);

        if (!is_tail) {
            add(reg_in_aux, jcp_.src_prc.size() * step * jcp_.inner_src_stride);
            add(reg_out_aux, jcp_.dst_prc.size() * step);
        }
    }
#undef GET_OFF
    inline void load(const Vmm& vmm_dst, const Xbyak::Reg64& reg_src, Precision src_prc, Precision dst_prc, const int& elt_num, bool fill) {
        const auto seed = load_emitter_params(src_prc, dst_prc, elt_num, fill, "float_min").hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_load_emitter(this, isa, src_prc, dst_prc, elt_num, Precision::FP32, fill, "float_min"));
        }

        emitters[seed]->emit_code({static_cast<size_t>(reg_src.getIdx()), 0}, {static_cast<size_t>(vmm_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }
    inline void store(const Xbyak::Reg64& reg_dst, const Vmm& vmm_src, Precision src_prc, Precision dst_prc, const int& elt_num) {
        const auto seed = store_emitter_params(src_prc, dst_prc, elt_num).hash();
        if (!emitters[seed]) {
            emitters[seed].reset(new jit_store_emitter(this, isa, src_prc, dst_prc, elt_num));
        }

        emitters[seed]->emit_code({static_cast<size_t>(vmm_src.getIdx()), 0}, {static_cast<size_t>(reg_dst.getIdx())},
                                  pool_aux_vmm_idxs, pool_aux_gpr_idxs);
    }

    size_t vec_size;
    Precision interm_prc;

    Xmm xmm_tmp = Xmm(2);
    Vmm vmm_scales = Vmm(0);
    Vmm vmm_in = Vmm(1);

    Reg64 reg_in = r8;
    Reg64 reg_in_aux = r9;
    Reg64 reg_out = r10;
    Reg64 reg_out_aux = r11;
    Reg64 reg_scales = r12;
    Reg8 reg_tmp_8 = Reg8(r13.getIdx());
    Reg16 reg_tmp_16 = Reg16(r13.getIdx());
    Reg32 reg_tmp_32 = Reg32(r13.getIdx());
    Reg64 reg_inner_work_amount = r14;
    Reg64 reg_outter_work_amount = r15;
    Reg64 reg_params = abi_param1;

    const std::vector<size_t> pool_aux_gpr_idxs = { static_cast<size_t>(rsi.getIdx()), static_cast<size_t>(rbp.getIdx()) };
    const std::vector<size_t> pool_aux_vmm_idxs = { static_cast<size_t>(xmm_tmp.getIdx()) };

    std::unordered_map<size_t, std::unique_ptr<jit_emitter>> emitters;
};

struct MHAGPT::Impl {
    void create(const CreateParam& param);
    void exec(const ExecParam& param);
    impl_desc_type get_impl_type() const {
        return _impl_desc_type;
    }

    CreateParam _create_param;
    impl_desc_type _impl_desc_type;

    // copy from mha.h/cpp begin
    struct brgemmCtx {
        size_t M, N, K, LDA, LDB, LDC;
        dnnl_data_type_t dt_in0, dt_in1;
        char palette[64];
        bool is_with_amx;
        bool is_with_comp;
        float beta;
    };

    template <typename in1_type>
    void mhaImpl(const ExecParam& param);

    void init_brgemm(brgemmCtx& ctx, std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel, bool use_amx) const;
    void init_brgemm_copy_a(std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t>& brgCopyKernel,
        size_t K, size_t K_blk, size_t K_tail, size_t LDA, dnnl_data_type_t dt_in0) const;
    void init_brgemm_copy_b(std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t>& brgCopyKernel,
        size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K, bool is_with_amx, dnnl_data_type_t dt_in0,
        dnnl_data_type_t dt_in1) const;

    void callBrgemm(brgemmCtx& ctx, std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t>& brgKernel,
                    const void* pin0, const void* pin1, void* pout, void* wsp) const;

    size_t getBrgIdx(size_t mIdx, size_t kIdx, size_t nIdx) const {
        return mIdx * 4 + kIdx * 2 + nIdx;
    }
    InferenceEngine::Precision accPrecision0;
    InferenceEngine::Precision accPrecision1;

    VectorDims dimsTranspose0In0;
    VectorDims dimsTranspose1In0;
    VectorDims dimsMulIn1;
    VectorDims dimsAddIn1;
    VectorDims dimsTranspose2In0;
    VectorDims dimsOut;

    // VectorDims strTranspose0In0;
    // VectorDims strTranspose1In0;
    // VectorDims strMulIn1;
    // VectorDims strAddIn1;
    // VectorDims strTranspose2In0;
    // VectorDims strOut;

    VectorDims dimsMatMul0In0;
    VectorDims dimsMatMul0In1;
    VectorDims dimsMatMul0Out;
    VectorDims dimsMatMul1In1;
    VectorDims dimsMatMul1Out;

    size_t batch0, batch1;
    size_t M_q_seq_len, M_blk, M_tail;
    size_t K0_head_size, K0_blk, K0_tail, N0_key_seq_len, N0_blk, N0_tail;
    size_t K1_key_seq_len, K1_blk, K1_tail, N1_head_size, N1_blk, N1_tail;

    size_t bufferMatMul0In0Size;
    size_t bufferMatMul0In1Size;
    size_t bufferMatMul0OutSize;
    size_t bufferMatMul1In1Size;
    size_t bufferMatMul1OutSize;
    size_t bufferCompensation0Size;
    size_t bufferCompensation1Size;
    size_t wsp_size_per_thread = 4 * 1024;

    std::vector<uint8_t> bufferMatMul0In0;
    std::vector<uint8_t> bufferMatMul0In1;
    std::vector<uint8_t> bufferMatMul0Out;
    std::vector<uint8_t> bufferMatMul1In1;
    std::vector<uint8_t> bufferMatMul1Out;
    std::vector<int32_t> bufferCompensation0;
    std::vector<int32_t> bufferCompensation1;
    std::vector<size_t> wsp;

    InferenceEngine::Precision fqPrc2;

    std::vector<float> mulScales;
    std::vector<float> fqScales0;
    std::vector<float> fqScales1;
    std::vector<float> fqScales2;
    std::vector<float> fqScales3;

    size_t brg0VnniFactor;
    brgemmCtx brgCtxs0[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgKernels0[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_a_t> brgCopyAKernel0;
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t> brgCopyBKernel0;

    size_t brg1VnniFactor;
    brgemmCtx brgCtxs1[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::x64::brgemm_kernel_t> brgKernels1[MHA_BRGEMM_KERNELS_NUM];
    std::unique_ptr<dnnl::impl::cpu::x64::matmul::jit_brgemm_matmul_copy_b_t> brgCopyBKernel1;

    //std::unique_ptr<jit_uni_mul_add_softmax_kernel> mulAddSoftmaxKernel;
    std::vector<std::shared_ptr<jit_uni_mul_add_softmax_kernel>> mulAddSoftmaxKernel;
    std::unique_ptr<jit_uni_convert_reorder_kernel> convertReorderKernel;
    std::unique_ptr<jit_uni_convert_transpose_kernel> convertTransposeKernel;
    // copy from mha.h/cpp end

    size_t vec_size = 1;
};

void MHAGPT::Impl::init_brgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, bool use_amx) const {
    brgemm_t brgDesc;
    brgemm_strides_t strides {static_cast<dnnl_dim_t>(ctx.M * ctx.K), static_cast<dnnl_dim_t>(ctx.K * ctx.N)};

    const bool is_int8 = utils::one_of(ctx.dt_in0, data_type::u8, data_type::s8) && utils::one_of(ctx.dt_in1, data_type::u8, data_type::s8);
    auto isa = use_amx ? isa_any
        : ctx.dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : (is_int8 ? avx512_core_vnni : avx512_core);
    auto status = brgemm_desc_init(&brgDesc, isa, brgemm_strd, ctx.dt_in0, ctx.dt_in1,
            false, false, brgemm_row_major, 1.f, ctx.beta, ctx.LDA, ctx.LDB, ctx.LDC, ctx.M, ctx.N, ctx.K, &strides);
    if (status != dnnl_success) {
        THROW_ERROR << "cannot be executed due to invalid brgconv params";
    }

    ctx.is_with_amx = use_amx;
    status = brgemm_init_tiles(brgDesc, ctx.palette);
    if (use_amx) {
        amx_tile_configure(ctx.palette);
    }

    ctx.is_with_comp = ctx.dt_in0 == dnnl_data_type_t::dnnl_s8 && !ctx.is_with_amx;

    brgemm_kernel_t* brgKernel_ = nullptr;
    status = brgemm_kernel_create(&brgKernel_, brgDesc);
    if (status != dnnl_success) {
        THROW_ERROR << "cannot be executed due to invalid brgconv params";
    }
    brgKernel.reset(brgKernel_);
}

void MHAGPT::Impl::init_brgemm_copy_a(std::unique_ptr<jit_brgemm_matmul_copy_a_t>& brgCopyKernel, size_t K, size_t K_blk, size_t K_tail,
        size_t LDA, dnnl_data_type_t dt_in0) const {
    brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_tag = dnnl_abcd;
    brgCopyKernelConf.K = K;
    brgCopyKernelConf.K_tail = K_tail;
    brgCopyKernelConf.K_blk = K_blk;
    brgCopyKernelConf.use_buffer_a_tail_only = false;
    brgCopyKernelConf.LDA = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.s8s8_compensation_required = false;
    brgCopyKernelConf.wei_zp_type = dnnl::impl::cpu::x64::none;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.a_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(dt_in0));
    brgCopyKernelConf.transposed_A = false;

    create_brgemm_matmul_copy_a(brgCopyKernel, &brgCopyKernelConf);
}

void MHAGPT::Impl::init_brgemm_copy_b(std::unique_ptr<jit_brgemm_matmul_copy_b_t>& brgCopyKernel, size_t N, size_t N_blk, size_t N_tail, size_t LDB, size_t K,
        bool is_with_amx, dnnl_data_type_t dt_in0, dnnl_data_type_t dt_in1) const {
    brgemm_matmul_conf_t brgCopyKernelConf;
    brgCopyKernelConf.src_dt = dt_in0;
    brgCopyKernelConf.wei_dt = dt_in1;
    brgCopyKernelConf.wei_n_blk = N_blk;
    brgCopyKernelConf.wei_tag = dnnl_abcd;
    brgCopyKernelConf.copy_B_wei_stride = 0;
    brgCopyKernelConf.LDB = LDB;
    brgCopyKernelConf.N = N;
    brgCopyKernelConf.N_tail = N_tail;
    brgCopyKernelConf.N_blk = N_blk;
    brgCopyKernelConf.K = K;
    brgCopyKernelConf.K_blk = K;
    brgCopyKernelConf.N_chunk_elems = brgCopyKernelConf.N_blk;
    brgCopyKernelConf.b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.tr_b_dt_sz = DnnlExtensionUtils::sizeOfDataType(static_cast<dnnl::memory::data_type>(brgCopyKernelConf.src_dt));
    brgCopyKernelConf.req_wei_vnni_downconvert = false;

    if (is_with_amx) {
        brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16_amx_bf16 : avx512_core_bf16_amx_int8;
        brgCopyKernelConf.s8s8_compensation_required = false;
    } else {
        brgCopyKernelConf.isa = dt_in0 == dnnl_data_type_t::dnnl_bf16 ? avx512_core_bf16 : avx512_core_vnni;
        brgCopyKernelConf.s8s8_compensation_required = dt_in0 == dnnl_data_type_t::dnnl_s8;
    }

    brgCopyKernelConf.has_zero_point_a = false;
    brgCopyKernelConf.has_zero_point_b = false;
    brgCopyKernelConf.src_zp_type = dnnl::impl::cpu::x64::none;

    create_brgemm_matmul_copy_b(brgCopyKernel, &brgCopyKernelConf);
}

void MHAGPT::Impl::create(const CreateParam& param) {
    _create_param = param;
    mulScales.push_back(_create_param.normal_factor);

    // q: [batch, num_heads, query_seq_len, head_size]
    // k: [batch, num_heads, maxSeqLen(valid: key_seq_len), head_size]
    // v: [batch, num_heads, maxSeqLen(valid: value_seq_len), head_size]
    // attention_mask: [batch, 1, 1, maxSeqLen(valid: key_seq_len)]
    // matmul1: [batch, num_heads, query_seq_len, head_size]
    // attn_output: [batch, query_seq_len, num_heads * head_size]
    // q shape
    dimsMatMul0In0 = { _create_param.batch, _create_param.num_heads, _create_param.query_seq_len, _create_param.head_size };
    // k transposed shape
    dimsMatMul0In1 = { _create_param.batch, _create_param.num_heads, _create_param.head_size, _create_param.key_seq_len };
    // q*k' shape
    dimsMatMul0Out = {dimsMatMul0In0[0], dimsMatMul0In0[1], dimsMatMul0In0[2], dimsMatMul0In1[3]};
    // v shape: key_seq_len == value_seq_len
    dimsMatMul1In1 = { _create_param.batch, _create_param.num_heads, _create_param.key_seq_len, _create_param.head_size };

    bool isAMXSupported = mayiuse(avx512_core_bf16_amx_int8) || mayiuse(avx512_core_bf16_amx_bf16);

    size_t numThreads = parallel_get_max_threads();

    size_t matmulOptimalM = 32;

    batch0 = dimsMatMul0Out[0];
    batch1 = dimsMatMul0Out[1];

    M_q_seq_len = dimsMatMul0In0[2];
    M_blk = matmulOptimalM;
    M_tail = M_q_seq_len % M_blk;

    N0_key_seq_len = dimsMatMul0In1[3];
    K0_head_size = dimsMatMul0In0[3];

    auto brg0Prc = param.qkv_precision;
    brg0VnniFactor = 4 / brg0Prc.size();
    bool brg0WithAMX = isAMXSupported && brg0Prc != Precision::FP32 && (K0_head_size % brg0VnniFactor == 0);// && (N0_key_seq_len % brg0VnniFactor == 0);

    N0_blk = brg0Prc == Precision::FP32 ? N0_key_seq_len :
             brg0Prc == Precision::BF16 ? 32 : 64;
    N0_tail = N0_key_seq_len % N0_blk;
    K0_blk = brg0WithAMX ? brg0Prc == Precision::BF16 ? 32 : 64
                         : K0_head_size;
    K0_tail = K0_head_size % K0_blk;

    accPrecision0 = brg0Prc == Precision::I8 ? Precision::I32 : Precision::FP32;

    size_t brg0BaseIdx = -1;
    for (size_t m = 0; m < 2; m++) {
        for (size_t k = 0; k < 2; k++) {
            for (size_t n = 0; n < 2; n++) {
                auto& brgemmCtx = brgCtxs0[getBrgIdx(m, k, n)];

                auto M_ = m ? M_tail
                            : M_q_seq_len < M_blk ? 0 : M_blk;
                auto N_ = n ? N0_tail : N0_key_seq_len - N0_tail;
                auto K_ = k ? K0_tail : K0_head_size - K0_tail;
                auto beta = k && brgCtxs0[getBrgIdx(m, 0, n)].K != 0 ? 1.0f : 0.0f;
                // q:      [batch, num_heads, query_seq_len, head_size]
                // k':     [batch, num_heads, head_size, key_seq_len]
                // q * k': [batch, num_heads, query_seq_len, key_seq_len]
                brgemmCtx.M = M_;
                brgemmCtx.N = N_;
                brgemmCtx.K = K_;
                brgemmCtx.LDA = K0_head_size;
                brgemmCtx.LDB = rnd_up(N0_key_seq_len, N0_blk);
                brgemmCtx.LDC = N0_key_seq_len;
                brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg0Prc));
                brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg0Prc));
                brgemmCtx.beta = beta;

                // don't create brgemm kernels for empty tiles
                if (M_ != 0 && K_ != 0 && N_ != 0) {
                    if (brg0BaseIdx == -1)
                        brg0BaseIdx = getBrgIdx(m, k, n);
                    init_brgemm(brgemmCtx, brgKernels0[getBrgIdx(m, k, n)], brg0WithAMX);
                }
            }
        }
    }

    auto& brgemmCtx0 = brgCtxs0[brg0BaseIdx];

    // TODO: matrix A copy should be performed to enable AMX matmuls for arbitrary shapes
    // if (brgemmCtx0.is_with_amx && K0_tail) {
    //     init_brgemm_copy_a(brgCopyAKernel0, K0, K0_blk, K0_tail, brgemmCtx0.LDA, brgemmCtx0.dt_in0);
    // }

    // B matrix, reshape to VNNI if using amx/vnni
    if (brgemmCtx0.is_with_amx || brg0Prc == Precision::I8 || brg0Prc == Precision::BF16) {
        // k': [batch, num_heads, head_size, key_seq_len]
        init_brgemm_copy_b(brgCopyBKernel0, N0_key_seq_len, N0_blk, N0_tail, brgemmCtx0.LDB, brgemmCtx0.K,
            brgemmCtx0.is_with_amx, brgemmCtx0.dt_in0, brgemmCtx0.dt_in1);
    }
    // [batch, num_heads, query_seq_len, head_size]
    dimsMatMul1Out = {dimsMatMul0Out[0], dimsMatMul0Out[1], dimsMatMul0Out[2], dimsMatMul1In1[3]};

    N1_head_size = dimsMatMul1Out[3];
    K1_key_seq_len = dimsMatMul0Out[3];

    auto brg1PrcIn0 = !fqScales2.empty() ? fqPrc2 : param.qkv_precision;
    auto brg1PrcIn1 = param.qkv_precision;
    brg1VnniFactor = 4 / brg1PrcIn0.size();
    bool brg1WithAMX = isAMXSupported && brg1PrcIn0 != Precision::FP32 && (K1_key_seq_len % brg1VnniFactor == 0) && (N1_head_size % brg1VnniFactor == 0);

    N1_blk = brg1PrcIn1 == Precision::FP32 ? N1_head_size :
             brg1PrcIn1 == Precision::BF16 ? 32 : 64;
    N1_tail = N1_head_size % N1_blk;
    K1_blk = brg1WithAMX ? brg1PrcIn0 == Precision::BF16 ? 32 : 64
                         : K1_key_seq_len;
    K1_tail = K1_key_seq_len % K1_blk;

    accPrecision1 = one_of(brg1PrcIn0, Precision::U8, Precision::I8) ? Precision::I32 : Precision::FP32;

    size_t brg1BaseIdx = -1;
    for (size_t m = 0; m < 2; m++) {
        for (size_t k = 0; k < 2; k++) {
            for (size_t n = 0; n < 2; n++) {
                auto& brgemmCtx = brgCtxs1[getBrgIdx(m, k, n)];

                auto M_ = m ? M_tail
                            : M_q_seq_len < M_blk ? 0 : M_blk;
                auto N_ = n ? N1_tail : N1_head_size - N1_tail;
                auto K_ = k ? K1_tail : K1_key_seq_len - K1_tail;

                // in0: [batch, num_heads, query_seq_len, key_seq_len]
                // value: [batch, num_heads, key_seq_len, head_size]
                // out: [batch, num_heads, query_seq_len, head_size]
                // transposed out: [batch, query_seq_len, num_heads * head_size]
                auto beta = k && brgCtxs1[getBrgIdx(m, 0, n)].K != 0 ? 1.0f : 0.0f;
                brgemmCtx.M = M_;
                brgemmCtx.N = N_;
                brgemmCtx.K = K_;
                brgemmCtx.LDA = K1_key_seq_len;
                brgemmCtx.LDB = brg1PrcIn1 == Precision::FP32 ? N1_head_size : rnd_up(N1_head_size, N1_blk);
                brgemmCtx.LDC = accPrecision1 == param.qkv_precision ? batch1 * N1_head_size : N1_head_size;
                brgemmCtx.dt_in0 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg1PrcIn0));
                brgemmCtx.dt_in1 = static_cast<dnnl_data_type_t>(DnnlExtensionUtils::IEPrecisionToDataType(brg1PrcIn1));
                brgemmCtx.beta = beta;

                // don't create brgemm kernels for empty tiles
                if (M_ != 0 && K_ != 0 && N_ != 0) {
                    if (brg1BaseIdx == -1)
                        brg1BaseIdx = getBrgIdx(m, k, n);

                    init_brgemm(brgemmCtx, brgKernels1[getBrgIdx(m, k, n)], brg1WithAMX);
                }
            }
        }
    }

    auto& brgemmCtx1 = brgCtxs1[brg1BaseIdx];
    if (brgemmCtx1.is_with_amx || brg1PrcIn1 == Precision::I8 || brg1PrcIn1 == Precision::BF16) {
        // value: [batch, num_heads, key_seq_len, head_size]
        init_brgemm_copy_b(brgCopyBKernel1, N1_head_size, N1_blk, N1_tail, brgemmCtx1.LDB, brgemmCtx1.K,
            brgemmCtx1.is_with_amx, brgemmCtx1.dt_in0, brgemmCtx1.dt_in1);
    }

    bufferMatMul0In0Size = M_blk * rnd_up(K0_head_size, K0_blk) * brg0Prc.size();
    bufferMatMul0In1Size = rnd_up(K0_head_size, brg0VnniFactor) * rnd_up(N0_key_seq_len, N0_blk) * brg0Prc.size();
    bufferMatMul0OutSize = brgemmCtx0.M * N0_key_seq_len * accPrecision0.size();
    bufferMatMul1In1Size = rnd_up(K1_key_seq_len, brg1VnniFactor) * rnd_up(N1_head_size, N1_blk) * std::max(brg0Prc.size(), brg1PrcIn1.size());
    bufferMatMul1OutSize = brgemmCtx1.M * N1_head_size * accPrecision1.size();
    bufferCompensation0Size = rnd_up(N0_key_seq_len, N0_blk);
    bufferCompensation1Size = rnd_up(N1_head_size, N1_blk);

    if (brgCopyAKernel0) {
        bufferMatMul0In0.resize(numThreads * bufferMatMul0In0Size);
    }
    bufferMatMul0In1.resize(numThreads * bufferMatMul0In1Size);
    bufferMatMul0Out.resize(numThreads * bufferMatMul0OutSize);
    bufferMatMul1In1.resize(numThreads * bufferMatMul1In1Size);
    bufferMatMul1Out.resize(numThreads * bufferMatMul1OutSize);
    if (brgemmCtx0.is_with_comp) {
        bufferCompensation0.resize(numThreads * bufferCompensation0Size);
    }
    if (brgemmCtx1.is_with_comp) {
        bufferCompensation1.resize(numThreads * bufferCompensation1Size);
    }

    if (brgemmCtx0.is_with_amx || brgemmCtx1.is_with_amx) {
        wsp.resize(numThreads * wsp_size_per_thread);
    }

    {
        if (mayiuse(cpu_isa_t::avx512_core)) {
            vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<cpu_isa_t::avx512_core>::vlen / sizeof(float);
        } else if (mayiuse(cpu_isa_t::avx2)) {
            vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<cpu_isa_t::avx512_core>::vlen / sizeof(float);
        } else if (mayiuse(cpu_isa_t::sse41)) {
            vec_size = dnnl::impl::cpu::x64::cpu_isa_traits<cpu_isa_t::avx512_core>::vlen / sizeof(float);
        } else {
            THROW_ERROR << "cannot create jit eltwise kernel";
        }
        mulAddSoftmaxKernel.resize(vec_size);

        for (auto i = 0; i < vec_size; i++) {
            jit_mul_add_softmax_compile_params jcp;
            jcp.src_prc = accPrecision0;
            jcp.dst_prc = brg1PrcIn0;
            jcp.with_mul_scales = !mulScales.empty();
            jcp.is_mul_first = true;
            jcp.with_scales0 = !fqScales1.empty();
            jcp.broadcast_scales0 = fqScales1.size() == 1;
            jcp.with_scales1 = !fqScales2.empty();
            jcp.broadcast_scales1 = fqScales2.size() == 1;
            jcp.tail_size = i;

            if (mayiuse(cpu_isa_t::avx512_core)) {
                mulAddSoftmaxKernel[i].reset(new jit_mul_add_softmax_kernel<cpu_isa_t::avx512_core>(jcp));
            } else if (mayiuse(cpu_isa_t::avx2)) {
                mulAddSoftmaxKernel[i].reset(new jit_mul_add_softmax_kernel<cpu_isa_t::avx2>(jcp));
            } else if (mayiuse(cpu_isa_t::sse41)) {
                mulAddSoftmaxKernel[i].reset(new jit_mul_add_softmax_kernel<cpu_isa_t::sse41>(jcp));
            }
        }
    }

    // acc precision convert: for matmul1 may be different from output precision
    if (accPrecision1 != param.qkv_precision) {
        // matmul1: [batch, num_heads, query_seq_len, head_size]
        // attn_output: [batch, query_seq_len, num_heads * head_size]
        jit_convert_reorder_compile_params jcp;
        jcp.src_prc = accPrecision1;
        jcp.dst_prc = param.qkv_precision;
        jcp.inner_work_amount = N1_head_size;     // head size(feature)
        jcp.with_scales = !fqScales3.empty();
        jcp.broadcast_scales = fqScales3.size() == 1;
        jcp.src_stride = N1_head_size;
        jcp.dst_stride = batch1 * N1_head_size;   // num_heads * head_size

        if (mayiuse(cpu_isa_t::avx512_core)) {
            convertReorderKernel.reset(new jit_convert_reorder_kernel<cpu_isa_t::avx512_core>(jcp));
        } else if (mayiuse(cpu_isa_t::avx2)) {
            convertReorderKernel.reset(new jit_convert_reorder_kernel<cpu_isa_t::avx2>(jcp));
        } else if (mayiuse(cpu_isa_t::sse41)) {
            convertReorderKernel.reset(new jit_convert_reorder_kernel<cpu_isa_t::sse41>(jcp));
        } else {
            THROW_ERROR << "cannot create jit eltwise kernel";
        }
    }

    // k: [batch, num_heads, key_seq_len, head_size]
    // k': [batch, num_heads, head_size, key_seq_len]
    if (!fqScales0.empty()) {
        jit_convert_transpose_compile_params jcp;
        jcp.src_prc = brg0Prc;
        jcp.dst_prc = brg0Prc;
        jcp.inner_work_amount = N0_key_seq_len;
        jcp.outter_work_amount = K0_head_size;
        jcp.with_scales = !fqScales0.empty();
        jcp.broadcast_scales = fqScales0.size() == 1;
        jcp.inner_src_stride = K0_head_size;          // head_size
        jcp.outter_src_stride = N0_key_seq_len * K0_head_size;    // head_size * key_seq_len
        jcp.outter_dst_stride = N0_key_seq_len;

        if (mayiuse(cpu_isa_t::avx512_core)) {
            convertTransposeKernel.reset(new jit_convert_transpose_kernel<cpu_isa_t::avx512_core>(jcp));
        } else if (mayiuse(cpu_isa_t::avx2)) {
            convertTransposeKernel.reset(new jit_convert_transpose_kernel<cpu_isa_t::avx2>(jcp));
        } else if (mayiuse(cpu_isa_t::sse41)) {
            convertTransposeKernel.reset(new jit_convert_transpose_kernel<cpu_isa_t::sse41>(jcp));
        } else {
            THROW_ERROR << "cannot create jit eltwise kernel";
        }
    }

    for (auto i = 0; i < vec_size; i++) {
        mulAddSoftmaxKernel[i]->create_ker();
    }

    if (convertReorderKernel)
        convertReorderKernel->create_ker();

    if (convertTransposeKernel)
        convertTransposeKernel->create_ker();

    if (brgemmCtx0.is_with_amx || brgemmCtx1.is_with_amx) {
        _impl_desc_type = jit_avx512_amx;
    } else {
        if (mayiuse(cpu_isa_t::avx512_core)) {
            _impl_desc_type = jit_avx512;
        } else if (mayiuse(cpu_isa_t::avx2)) {
            _impl_desc_type = jit_avx2;
        } else if (mayiuse(cpu_isa_t::sse41)) {
            _impl_desc_type = jit_sse42;
        }
    }
}

template<typename srcT, typename dstT>
static void reorder2D(const srcT* pin, dstT* pout, const std::vector<size_t>& dimsOut,
               const std::vector<size_t>& stridesOut, const std::vector<size_t>& stridesIn) {
    for (int i0 = 0; i0 < dimsOut[0]; i0++) {
        for (int i1 = 0; i1 < dimsOut[1]; i1++) {
            pout[i0 * stridesOut[0] + i1 * stridesOut[1]] = static_cast<dstT>(pin[i0 * stridesIn[0] + i1 * stridesIn[1]]);
        }
    }
}

void MHAGPT::Impl::callBrgemm(brgemmCtx& ctx, std::unique_ptr<brgemm_kernel_t>& brgKernel, const void* pin0, const void* pin1, void* pout, void* wsp) const {
    if (ctx.is_with_amx)
        amx_tile_configure(ctx.palette);
    if (ctx.is_with_comp) {
        brgemm_post_ops_data_t post_ops_data;
        brgemm_kernel_execute_postops(brgKernel.get(), 1, pin0, pin1, nullptr, pout, pout, post_ops_data, wsp);
    } else {
        brgemm_kernel_execute(brgKernel.get(), 1, pin0, pin1, nullptr, pout, wsp);
    }
}

template <typename in1_type>
void MHAGPT::Impl::mhaImpl(const ExecParam& param) {
    const uint8_t* pQIn0 = param.q;
    const auto& pKIn0 = param.k;
    const float* pAddIn1 = param.attention_mask;
    const auto& pVIn0 = param.v;
    uint8_t* pout = param.attn_output;

    auto outPrcSize = _create_param.qkv_precision.size();
    parallel_for2d(dimsMatMul0Out[0], dimsMatMul0Out[1], [&](size_t i0, size_t i1) {
        size_t threadNum = parallel_get_thread_num();

        auto pQIn0_aux = pQIn0 + (i0 * param.batch_stride_in_q + i1 * param.head_stride_in_q) * _create_param.qkv_precision.size();
        auto pKIn0_aux = pKIn0[i0] + i1 * param.head_stride_in_kv * _create_param.qkv_precision.size();
        auto pVIn0_aux = pVIn0[i0] + i1 * param.head_stride_in_kv * _create_param.qkv_precision.size();

        auto pAddIn1_aux = pAddIn1 + i0 * param.batch_stride_in_attn_mask;

        auto bufferMatMul0In1_local = reinterpret_cast<uint8_t*>(bufferMatMul0In1.data() + threadNum * bufferMatMul0In1Size);
        auto bufferMatMul0Out_local = reinterpret_cast<uint8_t*>(bufferMatMul0Out.data() + threadNum * bufferMatMul0OutSize);
        auto bufferMatMul1In1_local = reinterpret_cast<uint8_t*>(bufferMatMul1In1.data() + threadNum * bufferMatMul1In1Size);
        auto bufferMatMul1Out_local = reinterpret_cast<uint8_t*>(bufferMatMul1Out.data() + threadNum * bufferMatMul1OutSize);

        auto pTranspose1Out_aux = brgCopyBKernel0 ? bufferMatMul1In1_local
                                                  : bufferMatMul0In1_local;

        if (convertTransposeKernel) {
            jit_convert_transpose_call_args call_args;
            call_args.p_in = pKIn0_aux;
            call_args.p_out = pTranspose1Out_aux;
            call_args.p_scales = fqScales0.data();

            (*convertTransposeKernel)(&call_args);
        } else {
            // k: [batch, num_heads, key_seq_len, head_size]
            // k': [batch, num_heads, head_size, key_seq_len]
            reorder2D(reinterpret_cast<const in1_type*>(pKIn0_aux), reinterpret_cast<in1_type*>(pTranspose1Out_aux),
                {K0_head_size, N0_key_seq_len}, {N0_key_seq_len, 1}, {1, K0_head_size});
        }

        auto bufferCompensation0_aux = !bufferCompensation0.empty()
            ? bufferCompensation0.data() + threadNum * bufferCompensation0Size
            : nullptr;
        auto bufferCompensation1_aux = !bufferCompensation1.empty()
            ? bufferCompensation1.data() + threadNum * bufferCompensation1Size
            : nullptr;

        auto wsp_local = !wsp.empty() ? wsp.data() + threadNum * wsp_size_per_thread : nullptr;

        // k': [batch, num_heads, head_size, key_seq_len]
        //   N0 == key_seq_len
        auto pMatMul0In1 = reinterpret_cast<uint8_t*>(pTranspose1Out_aux);
        if (brgCopyBKernel0) {
            for (size_t nb = 0; nb < div_up(N0_key_seq_len, N0_blk); nb++) {
                auto pCopyKernel0In = pMatMul0In1 + nb * N0_blk * _create_param.qkv_precision.size();
                auto pCopyKernel0Out = bufferMatMul0In1_local + nb * N0_blk * brg0VnniFactor * _create_param.qkv_precision.size();

                auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();

                const bool is_N_tail = (N0_key_seq_len - nb * N0_blk < N0_blk);
                ctx.current_N_blk = is_N_tail ? N0_tail : N0_blk;
                ctx.src = pCopyKernel0In;
                ctx.tr_src = pCopyKernel0Out;
                ctx.compensation_ptr = bufferCompensation0_aux + nb * N0_blk;
                ctx.zp_a_compensation_ptr = nullptr;
                ctx.zp_a_neg_value_ptr = nullptr;
                ctx.current_K_start = 0;
                ctx.current_K_iters = K0_head_size;

                (*brgCopyBKernel0)(&ctx);
            }

            pMatMul0In1 = bufferMatMul0In1_local;
        }

        // v layout: [batch, num_heads, key_seq_len, head_size]
        //   N1 == head_size
        auto pMatMul1In1 = pVIn0_aux;
        if (brgCopyBKernel1) {
            for (size_t nb = 0; nb < div_up(N1_head_size, N1_blk); nb++) {
                auto pCopyKernel1In = pMatMul1In1 + nb * N1_blk * _create_param.qkv_precision.size();
                auto pCopyKernel1Out = reinterpret_cast<uint8_t*>(bufferMatMul1In1_local) + nb * N1_blk * brg1VnniFactor * _create_param.qkv_precision.size();

                auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();

                const bool is_N_tail = (N1_head_size - nb * N1_blk < N1_blk);
                ctx.current_N_blk = is_N_tail ? N1_tail : N1_blk;
                ctx.src = pCopyKernel1In;
                ctx.tr_src = pCopyKernel1Out;
                ctx.compensation_ptr = bufferCompensation1_aux + nb * N1_blk;
                ctx.zp_a_compensation_ptr = nullptr;
                ctx.zp_a_neg_value_ptr = nullptr;
                ctx.current_K_start = 0;
                ctx.current_K_iters = K1_key_seq_len;

                (*brgCopyBKernel1)(&ctx);
            }

            pMatMul1In1 = reinterpret_cast<uint8_t*>(bufferMatMul1In1_local);
        }

        for (size_t mb = 0; mb < div_up(M_q_seq_len, M_blk); mb++) {
            const bool is_M_tail = (M_q_seq_len - mb * M_blk < M_blk);
            auto cur_M_blk = is_M_tail ? M_tail : M_blk;
            // q layout: [batch, num_heads, query_seq_len, head_size]
            auto pMatMul0In0 = pQIn0_aux + (mb * M_blk * K0_head_size) * _create_param.qkv_precision.size();

            // TODO: matrix A copy should be performed to enable AMX matmuls for arbitrary shapes
            // if (brgCopyAKernel0) {
            //     auto bufferMatMul0In0_local = reinterpret_cast<void*>(bufferMatMul0In0.data() + threadNum * bufferMatMul0In0Size);

            //     auto pCopyKernel0In = pMatMul0In0;
            //     auto pCopyKernel0Out = reinterpret_cast<data_type*>(bufferMatMul0In0_local);

            //     auto ctx = jit_brgemm_matmul_copy_a_t::ctx_t();

            //     ctx.current_M_blk = cur_M_blk;
            //     ctx.zp_b_compensation_buffer_ptr = nullptr;
            //     ctx.zp_a_compensation_result_ptr = nullptr;
            //     ctx.zp_b_neg_value_ptr = nullptr;
            //     ctx.zp_ab_comp_ptr = nullptr;
            //     ctx.src = pCopyKernel0In;
            //     ctx.tr_src = pCopyKernel0Out;
            //     ctx.current_K_start = 0;
            //     ctx.current_K_blk = K0;

            //     (*brgCopyAKernel0)(&ctx);

            //     pMatMul0In0 = reinterpret_cast<const data_type*>(bufferMatMul0In0_local);
            // }

            auto pMatMul0Out = bufferMatMul0Out_local;

            size_t brgIdx0 = getBrgIdx(0, 0, 0);
            size_t K0_step0 = brgCtxs0[brgIdx0].K;
            size_t K0_step1 = brgCtxs0[brgIdx0].K * brgCtxs0[brgIdx0].LDB;
            size_t N0_step0 = brgCtxs0[brgIdx0].N * brg0VnniFactor;
            size_t N0_step1 = brgCtxs0[brgIdx0].N;
            for (size_t n = 0; n < 2; n++) {
                for (size_t k = 0; k < 2; k++) {
                    size_t mIdx = is_M_tail ? 1 : 0;
                    auto& brgemmCtx = brgCtxs0[getBrgIdx(mIdx, k, n)];

                    auto wsp = brgemmCtx.is_with_comp
                        ? reinterpret_cast<void*>(bufferCompensation0_aux + n * N0_step1)
                        : reinterpret_cast<void*>(wsp_local);

                    if (brgemmCtx.K != 0 && brgemmCtx.N != 0) {
                        callBrgemm(brgemmCtx, brgKernels0[getBrgIdx(mIdx, k, n)],
                            pMatMul0In0 + (k * K0_step0) * _create_param.qkv_precision.size(),
                            pMatMul0In1 + (k * K0_step1 + n * N0_step0) * _create_param.qkv_precision.size(),
                            pMatMul0Out + (n * N0_step1) * accPrecision0.size(), wsp);
                    }
                }
            }

            auto pMulIn1 = reinterpret_cast<float*>(mulScales.empty() ? nullptr : mulScales.data());
            // loop along K dimension
            auto valid_softmax_items = _create_param.first_valid_softmax_items + mb * M_blk;
            for (size_t m = 0; m < cur_M_blk; m++) {
                jit_mul_add_softmax_call_args call_args;
                call_args.p_in0 = pMatMul0Out + m * N0_key_seq_len * accPrecision0.size();
                call_args.p_mul_in1 = mulScales.size() > 1 ? pMulIn1 + i1 : pMulIn1;
                call_args.p_add_in1 = pAddIn1_aux;
                call_args.p_out = pMatMul0Out + m * N0_key_seq_len * _create_param.qkv_precision.size();
                call_args.p_buffer = pMatMul0Out + m * N0_key_seq_len * accPrecision0.size();
                call_args.p_scales0 = fqScales1.data();
                call_args.p_scales1 = fqScales2.data();
                call_args.work_amount = valid_softmax_items;

                (*mulAddSoftmaxKernel[valid_softmax_items % vec_size])(&call_args);
                // attn_scores = torch.where(causal_mask, attn_scores, mask_value)
                if (_create_param.need_select_transpose) {
                    void *invalidPtr = pMatMul0Out + (m * N0_key_seq_len + valid_softmax_items) * _create_param.qkv_precision.size();
                    memset(invalidPtr, 0, (N0_key_seq_len - valid_softmax_items) * _create_param.qkv_precision.size());
                    valid_softmax_items = std::min(valid_softmax_items + 1, N0_key_seq_len);
                }
            }

            auto pMatMul1In0 = bufferMatMul0Out_local;
            // transposed shape: [bs, seq_len, num_attention_heads, attn_head_size]
            auto pOut_aux = pout + (i0 * param.batch_stride_in_attn + i1 * param.head_stride_in_attn) * outPrcSize;

            auto pMatMul1Out = _create_param.qkv_precision == Precision::FP32
                ? pOut_aux + (mb * M_blk * batch1 * N1_head_size) * outPrcSize
                : bufferMatMul1Out_local;

            size_t brgIdx1 = getBrgIdx(0, 0, 0);
            size_t K1_step0 = brgCtxs1[brgIdx1].K;
            size_t K1_step1 = brgCtxs1[brgIdx1].K * brgCtxs1[brgIdx1].LDB;
            size_t N1_step0 = brgCtxs1[brgIdx1].N * brg1VnniFactor;
            size_t N1_step1 = brgCtxs1[brgIdx1].N;
            for (size_t n = 0; n < 2; n++) {
                for (size_t k = 0; k < 2; k++) {
                    size_t mIdx = is_M_tail ? 1 : 0;
                    auto& brgemmCtx = brgCtxs1[getBrgIdx(mIdx, k, n)];

                    auto wsp = brgemmCtx.is_with_comp
                        ? reinterpret_cast<void*>(bufferCompensation1_aux + n * N1_step1)
                        : reinterpret_cast<void*>(wsp_local);

                    if (brgemmCtx.K != 0 && brgemmCtx.N != 0) {
                        callBrgemm(brgemmCtx, brgKernels1[getBrgIdx(mIdx, k, n)],
                            pMatMul1In0 + (k * K1_step0) * _create_param.qkv_precision.size(),
                            pMatMul1In1 + (k * K1_step1 + n * N1_step0) * _create_param.qkv_precision.size(),
                            pMatMul1Out + (n * N1_step1) * accPrecision1.size(), wsp);
                    }
                }
            }

            if (convertReorderKernel) {
                // matmul1: [batch, num_heads, query_seq_len, head_size]
                // attn_output: [batch, query_seq_len, num_heads * head_size]
                jit_convert_reorder_call_args call_args;
                call_args.p_in = pMatMul1Out;
                call_args.p_out = pOut_aux + (mb * M_blk * batch1 * N1_head_size) * outPrcSize;
                call_args.p_scales = fqScales3.data();
                call_args.outter_work_amount = cur_M_blk;

                (*convertReorderKernel)(&call_args);
            }
        }
    });
}

void MHAGPT::Impl::exec(const ExecParam& param) {
    if (_create_param.qkv_precision == Precision::FP32) {
        mhaImpl<float>(param);
    } else if (_create_param.qkv_precision == Precision::BF16) {
        mhaImpl<bfloat16_t>(param);
    } else if (_create_param.qkv_precision == Precision::I8) {
        mhaImpl<int8_t>(param);
    } else {
        THROW_ERROR << "doesn't support provided input precisions";
    }
}

// interface
MHAGPT::MHAGPT(): impl(std::make_shared<Impl>()) {
}

int MHAGPT::query_scratch_size(const CreateParam& param) {
    return 0;
}

void MHAGPT::create(const CreateParam& param) {
    impl->create(param);
}

void MHAGPT::exec(const ExecParam& param) {
    impl->exec(param);
}

impl_desc_type MHAGPT::get_impl_type() {
    return impl->get_impl_type();
}

}   // namespace gpt
}   // namespace intel_cpu
}   // namespace ov

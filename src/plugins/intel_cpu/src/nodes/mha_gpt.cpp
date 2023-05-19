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
#include "common/dnnl_thread.hpp"
#include "special/gemm_custom.hpp"
#include "special/quant_i8_custom.hpp"

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

        //if (jcp_.src_prc == Precision::I32) 
        {
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

struct MHAGPT::Impl {
    void create(const CreateParam& param);
    void exec(const ExecParam& param);
    impl_desc_type get_impl_type() const {
        return _impl_desc_type;
    }

    CreateParam _create_param;
    impl_desc_type _impl_desc_type;

    void mha_bf16(const ExecParam &param);
    void mha_i8(const ExecParam &param);

    size_t bufferMatMul0OutSize;
    size_t bufferMatMul1OutSize;

    std::shared_ptr<uint8_t> bufferMatMul0Out;
    std::shared_ptr<uint8_t> bufferMatMul1Out;

    std::vector<float> mulScales;
    std::vector<float> fqScales0;
    std::vector<float> fqScales1;
    std::vector<float> fqScales2;
    std::vector<float> fqScales3;

    std::vector<std::shared_ptr<jit_uni_mul_add_softmax_kernel>> mulAddSoftmaxKernel;
    std::unique_ptr<jit_uni_convert_reorder_kernel> convertReorderKernel;

    size_t vec_size = 1;
    std::vector<std::shared_ptr<amx_kernel::MatmulVector<ov::bfloat16, ov::bfloat16>>> gemAvB_BF16xBF16;
    std::vector<std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>> qKtrGemm_BF16xBF16;
    std::vector<std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>> qKVGemm_BF16xBF16;

    std::vector<std::shared_ptr<amx_kernel::Matmul<int8_t, int8_t>>> qKtrGemm_i8xi8;
    std::vector<std::shared_ptr<amx_kernel::Matmul<uint8_t, int8_t>>> qKVGemm_u8xi8;
    std::vector<std::shared_ptr<amx_kernel::MatmulVector<int8_t, int8_t>>> gemAvB_i8xi8;
};

void MHAGPT::Impl::create(const CreateParam& param) {
    _create_param = param;
    mulScales.push_back(_create_param.normal_factor);

    // q: [batch, num_heads, query_seq_len, head_size]
    // k: [batch, num_heads, maxSeqLen(valid: key_seq_len), head_size]
    // v: [batch, num_heads, maxSeqLen(valid: value_seq_len), head_size]
    // attention_mask: [batch, 1, 1, maxSeqLen(valid: key_seq_len)]
    // matmul1: [batch, num_heads, query_seq_len, head_size]
    // attn_output: [batch, query_seq_len, num_heads * head_size]
    bool isAMXSupported = mayiuse(avx512_core_bf16_amx_int8) || mayiuse(avx512_core_bf16_amx_bf16);

    size_t numThreads = parallel_get_max_threads();
    if (_create_param.qkv_precision == Precision::I8) {
        qKtrGemm_i8xi8.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            qKtrGemm_i8xi8[i] = std::make_shared<amx_kernel::Matmul<int8_t, int8_t>>(false, true);
        }
        qKVGemm_u8xi8.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            qKVGemm_u8xi8[i] = std::make_shared<amx_kernel::Matmul<uint8_t, int8_t>>(false, false);
        }
        gemAvB_i8xi8.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            gemAvB_i8xi8[i] = std::make_shared<amx_kernel::MatmulVector<int8_t, int8_t>>();
        }
    } else {
        gemAvB_BF16xBF16.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            gemAvB_BF16xBF16[i] = std::make_shared<amx_kernel::MatmulVector<ov::bfloat16, ov::bfloat16>>();
        }
        qKtrGemm_BF16xBF16.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            qKtrGemm_BF16xBF16[i] = std::make_shared<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>(false, true);
        }
        qKVGemm_BF16xBF16.resize(numThreads);
        for (size_t i = 0; i < numThreads; i++) {
            qKVGemm_BF16xBF16[i] = std::make_shared<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>>(false, false);
        }
    }

    bufferMatMul0OutSize = _create_param.max_seq_len * rnd_up(_create_param.max_seq_len * sizeof(float), 64);
    bufferMatMul1OutSize = _create_param.max_seq_len * _create_param.head_size_aligned * sizeof(float);

    bufferMatMul0Out = std::shared_ptr<uint8_t>(
                            reinterpret_cast<uint8_t*>(aligned_alloc(64, numThreads * bufferMatMul0OutSize)),
                            [](void * p) { ::free(p); });
    bufferMatMul1Out = std::shared_ptr<uint8_t>(
                            reinterpret_cast<uint8_t*>(aligned_alloc(64, numThreads * bufferMatMul1OutSize)),
                            [](void * p) { ::free(p); });

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
            jcp.src_prc = Precision::FP32;
            jcp.dst_prc = _create_param.qkv_precision;
            // softmax always generates u8
            if (_create_param.qkv_precision == Precision::I8)
                jcp.dst_prc = Precision::U8;
            jcp.with_mul_scales = !mulScales.empty();
            jcp.is_mul_first = true;
            jcp.with_scales0 = !fqScales1.empty();
            jcp.broadcast_scales0 = fqScales1.size() == 1;
            jcp.with_scales1 = _create_param.qkv_precision == Precision::I8;  // !fqScales2.empty();
            jcp.broadcast_scales1 = true;                                     // fqScales2.size() == 1;
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

    // acc precision convert: for matmul1 may be different from output precision. TODO: fused
    {
        // matmul1: [batch, num_heads, query_seq_len, head_size]
        // attn_output: [batch, query_seq_len, num_heads * head_size]
        jit_convert_reorder_compile_params jcp;
        jcp.src_prc = Precision::FP32;
        jcp.dst_prc = param.dst_precision;
        jcp.inner_work_amount = _create_param.head_size;     // head size(feature)
        jcp.with_scales = _create_param.dst_precision == Precision::I8;
        jcp.broadcast_scales = _create_param.is_qkv_quant_per_tensor;
        jcp.src_stride = _create_param.head_size_aligned;
        jcp.dst_stride = _create_param.num_heads * _create_param.head_size;   // num_heads * head_size

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

    for (auto i = 0; i < vec_size; i++) {
        mulAddSoftmaxKernel[i]->create_ker();
    }

    if (convertReorderKernel)
        convertReorderKernel->create_ker();

    if (isAMXSupported) {
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

void MHAGPT::Impl::mha_bf16(const ExecParam &param) {
    uint8_t* pQIn0 = param.q;
    auto& pKIn0 = param.k;
    float* pAddIn1 = param.attention_mask;
    auto& pVIn0 = param.v;
    uint8_t* pout = param.attn_output;

    auto outPrcSize = _create_param.qkv_precision.size();
    auto& gemAvB_ops = gemAvB_BF16xBF16;
    auto& qKtrGemm_ops = qKtrGemm_BF16xBF16;
    auto& qKVGemm_ops = qKVGemm_BF16xBF16;
    bool is_vector = param.query_seq_len == 1;

    if (is_vector) {
        parallel_for2d(param.batch, _create_param.num_heads, [&](size_t i0, size_t i1) {
            size_t threadNum = parallel_get_thread_num();

            auto pQIn0_aux = pQIn0 + (i0 * param.batch_stride_in_q + i1 * param.head_stride_in_q) * _create_param.qkv_precision.size();
            auto pKIn0_aux = pKIn0[i0] + i1 * param.head_stride_in_kv * _create_param.qkv_precision.size();
            auto pVIn0_aux = pVIn0[i0] + i1 * param.head_stride_in_kv * _create_param.qkv_precision.size();

            auto pAddIn1_aux = pAddIn1 + i0 * param.batch_stride_in_attn_mask;

            auto bufferMatMul0Out_local = reinterpret_cast<uint8_t*>(bufferMatMul0Out.get() + threadNum * bufferMatMul0OutSize);
            auto bufferMatMul1Out_local = reinterpret_cast<uint8_t*>(bufferMatMul1Out.get() + threadNum * bufferMatMul1OutSize);
            
            tensor2D<ov::bfloat16> matK(param.key_seq_len, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pKIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
            // N: key_seq_len, K: head_size
            // q[1, K] * transpose(k[N, K])        ==>
            //     k[N, K] * transpose(q[1, K])    ==>
            //     k[N, K] * q[K, 1]
            (*gemAvB_ops[threadNum])(matK, reinterpret_cast<ov::bfloat16*>(pQIn0_aux), reinterpret_cast<float*>(bufferMatMul0Out_local));

            auto pMulIn1 = reinterpret_cast<float*>(mulScales.empty() ? nullptr : mulScales.data());
            auto pMatMul0Out = bufferMatMul0Out_local;
            // loop along K dimension
            auto valid_softmax_items = param.first_valid_softmax_items;
            {
                jit_mul_add_softmax_call_args call_args;
                call_args.p_in0 = pMatMul0Out;
                call_args.p_mul_in1 = mulScales.size() > 1 ? pMulIn1 + i1 : pMulIn1;
                call_args.p_add_in1 = pAddIn1_aux;
                call_args.p_out = pMatMul0Out;
                call_args.p_buffer = pMatMul0Out;
                call_args.p_scales0 = fqScales1.data();
                call_args.p_scales1 = fqScales2.data();
                call_args.work_amount = valid_softmax_items;

                (*mulAddSoftmaxKernel[valid_softmax_items % vec_size])(&call_args);
                // attn_scores = torch.where(causal_mask, attn_scores, mask_value)
            }
            auto pOut_aux = pout + (i0 * param.batch_stride_in_attn + i1 * param.head_stride_in_attn) * outPrcSize;
            tensor2D<ov::bfloat16> matQK(param.query_seq_len, param.key_seq_len, reinterpret_cast<ov::bfloat16*>(bufferMatMul0Out_local), rnd_up(param.key_seq_len * sizeof(bfloat16), 64));
            tensor2D<ov::bfloat16> matV(param.key_seq_len, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pVIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
            tensor2D<float> matQKV(param.query_seq_len, _create_param.head_size, reinterpret_cast<float*>(bufferMatMul1Out_local), _create_param.head_size_aligned * sizeof(float));
            amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQKV);
            (*qKVGemm_ops[threadNum])(matQK, matV, 0, _create_param.head_size, pp);
            if (convertReorderKernel) {
                // matmul1: [batch, num_heads, query_seq_len, head_size]
                // attn_output: [batch, query_seq_len, num_heads * head_size]
                jit_convert_reorder_call_args call_args;
                call_args.p_in = bufferMatMul1Out_local;
                call_args.p_out = pOut_aux;
                call_args.p_scales = fqScales3.data();
                call_args.outter_work_amount = param.query_seq_len;

                (*convertReorderKernel)(&call_args);
            }
        });
    } else {
        int seq_cout_all = rnd_up(param.query_seq_len, 32) / 32;
        int work_amount = param.batch * _create_param.num_heads * seq_cout_all;
        int numThreads = parallel_get_max_threads();
        parallel_for(numThreads, [&](int threadNum) {
            int i0;
            int i1;
            int seq;
            int start {0}, end {0};
            splitter(work_amount, numThreads, threadNum, start, end);

            parallel_it_init(start, i0, param.batch, i1, _create_param.num_heads, seq, seq_cout_all);
            uint8_t* prev_k = nullptr;
            uint8_t* prev_v = nullptr;
            for (int iwork = start; iwork < end; ++iwork) {
                int seq_start = seq * 32;
                int seq_end = std::min(static_cast<size_t>(seq_start) + 32, param.query_seq_len);
                int seq_cout = seq_end - seq_start;
                // q: [batch, num_heads, query_seq_len, head_size]
                // k: [batch, num_heads, key_seq_len, head_size]
                // v: [batch, num_heads, value_seq_len, head_size]
                auto pQIn0_aux = pQIn0 + (i0 * param.batch_stride_in_q + i1 * param.head_stride_in_q + seq_start * _create_param.head_size_aligned) * _create_param.qkv_precision.size();
                auto pKIn0_aux = pKIn0[i0] + i1 * param.head_stride_in_kv * _create_param.qkv_precision.size();
                auto pVIn0_aux = pVIn0[i0] + i1 * param.head_stride_in_kv * _create_param.qkv_precision.size();

                auto pAddIn1_aux = pAddIn1 + i0 * param.batch_stride_in_attn_mask;

                auto bufferMatMul0Out_local = reinterpret_cast<uint8_t*>(bufferMatMul0Out.get() + threadNum * bufferMatMul0OutSize);
                auto bufferMatMul1Out_local = reinterpret_cast<uint8_t*>(bufferMatMul1Out.get() + threadNum * bufferMatMul1OutSize);
                
                tensor2D<ov::bfloat16> matQ(seq_cout, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pQIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
                tensor2D<ov::bfloat16> matK(param.key_seq_len, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pKIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
                tensor2D<float> matQK(seq_cout, param.key_seq_len, reinterpret_cast<float*>(bufferMatMul0Out_local), rnd_up(param.key_seq_len * sizeof(float), 64));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQK);
                (*qKtrGemm_ops[threadNum])(matQ, matK, 0, param.key_seq_len, pp, pKIn0_aux == prev_k);
                prev_k = pKIn0_aux;

                auto pMulIn1 = reinterpret_cast<float*>(mulScales.empty() ? nullptr : mulScales.data());
                auto pMatMul0Out = bufferMatMul0Out_local;
                // loop along K dimension
                auto valid_softmax_items = param.first_valid_softmax_items + seq_start;
                for (size_t m = 0; m < seq_cout; m++) {
                    jit_mul_add_softmax_call_args call_args;
                    call_args.p_in0 = pMatMul0Out + m * rnd_up(param.key_seq_len * sizeof(float), 64);
                    call_args.p_mul_in1 = mulScales.size() > 1 ? pMulIn1 + i1 : pMulIn1;
                    call_args.p_add_in1 = pAddIn1_aux;
                    call_args.p_out = pMatMul0Out + m * rnd_up(param.key_seq_len * sizeof(bfloat16), 64);
                    call_args.p_buffer = pMatMul0Out + m * rnd_up(param.key_seq_len * sizeof(float), 64);
                    call_args.p_scales0 = fqScales1.data();
                    call_args.p_scales1 = fqScales2.data();
                    call_args.work_amount = valid_softmax_items;

                    (*mulAddSoftmaxKernel[valid_softmax_items % vec_size])(&call_args);
                    // attn_scores = torch.where(causal_mask, attn_scores, mask_value)
                    if (param.key_seq_len > valid_softmax_items) {
                        auto *invalidPtr = static_cast<bfloat16*>(call_args.p_out) + valid_softmax_items;
                        memset(invalidPtr, 0, (param.key_seq_len - valid_softmax_items) * _create_param.qkv_precision.size());
                        valid_softmax_items = std::min(valid_softmax_items + 1, param.key_seq_len);
                    }
                }
                auto pOut_aux = pout + (i0 * param.batch_stride_in_attn + i1 * param.head_stride_in_attn
                    + seq_start * param.head_stride_in_attn * _create_param.num_heads) * outPrcSize;
                tensor2D<ov::bfloat16> matQKBF16(seq_cout, param.key_seq_len, reinterpret_cast<ov::bfloat16*>(bufferMatMul0Out_local), rnd_up(param.key_seq_len * sizeof(bfloat16), 64));
                tensor2D<ov::bfloat16> matV(param.key_seq_len, _create_param.head_size, reinterpret_cast<ov::bfloat16*>(pVIn0_aux), _create_param.head_size_aligned * sizeof(ov::bfloat16));
                tensor2D<float> matQKV(seq_cout, _create_param.head_size, reinterpret_cast<float*>(bufferMatMul1Out_local), _create_param.head_size_aligned * sizeof(float));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp2(matQKV);
                (*qKVGemm_ops[threadNum])(matQKBF16, matV, 0, _create_param.head_size, pp2, prev_v == pVIn0_aux);
                prev_v = pVIn0_aux;
                if (convertReorderKernel) {
                    // matmul1: [batch, num_heads, query_seq_len, head_size]
                    // attn_output: [batch, query_seq_len, num_heads * head_size]
                    jit_convert_reorder_call_args call_args;
                    call_args.p_in = bufferMatMul1Out_local;
                    call_args.p_out = pOut_aux;
                    call_args.p_scales = fqScales3.data();
                    call_args.outter_work_amount = seq_cout;

                    (*convertReorderKernel)(&call_args);
                }
                parallel_it_step(i0, param.batch, i1, _create_param.num_heads, seq, seq_cout_all);
            }
        });
    }
}

void MHAGPT::Impl::mha_i8(const ExecParam &param) {
    uint8_t* pQIn0 = param.q;
    auto& pKIn0 = param.k;
    float* pAddIn1 = param.attention_mask;
    auto& pVIn0 = param.v;
    uint8_t* pout = param.attn_output;

    auto outPrcSize = _create_param.dst_precision.size();
    auto& gemAvB_ops = gemAvB_i8xi8;
    auto& qKtrGemm_ops = qKtrGemm_i8xi8;
    auto& qKVGemm_ops = qKVGemm_u8xi8;
    bool is_vector = param.query_seq_len == 1;
    // dequant param
    auto mul_scales = mulScales[0] * param.q_dequant * param.k_dequant;
    auto qkv_quant = param.qkv_quant;
    for (size_t i = 0; i < param.qkv_quant.size(); i++) {
        qkv_quant[i] *= param.v_dequant / param.qk_quant;
    }

    if (is_vector) {
        parallel_for2d(param.batch, _create_param.num_heads, [&](size_t i0, size_t i1) {
            size_t threadNum = parallel_get_thread_num();

            auto pQIn0_aux = pQIn0 + (i0 * param.batch_stride_in_q + i1 * param.head_stride_in_q) * _create_param.qkv_precision.size();
            auto pKIn0_aux = pKIn0[i0] + i1 * param.head_stride_in_kv * _create_param.qkv_precision.size();
            auto pVIn0_aux = pVIn0[i0] + i1 * param.head_stride_in_kv * _create_param.qkv_precision.size();

            auto pAddIn1_aux = pAddIn1 + i0 * param.batch_stride_in_attn_mask;

            auto bufferMatMul0Out_local = reinterpret_cast<uint8_t*>(bufferMatMul0Out.get() + threadNum * bufferMatMul0OutSize);
            auto bufferMatMul1Out_local = reinterpret_cast<uint8_t*>(bufferMatMul1Out.get() + threadNum * bufferMatMul1OutSize);
            
            tensor2D<int8_t> matK(param.key_seq_len, _create_param.head_size, reinterpret_cast<int8_t*>(pKIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
            // N: key_seq_len, K: head_size
            // q[1, K] * transpose(k[N, K])        ==>
            //     k[N, K] * transpose(q[1, K])    ==>
            //     k[N, K] * q[K, 1]
            (*gemAvB_ops[threadNum])(matK, reinterpret_cast<int8_t*>(pQIn0_aux), reinterpret_cast<int32_t*>(bufferMatMul0Out_local));
            cvt_i32_f32(reinterpret_cast<float*>(bufferMatMul0Out_local), reinterpret_cast<int32_t*>(bufferMatMul0Out_local), param.key_seq_len);

            auto pMulIn1 = reinterpret_cast<float*>(mulScales.empty() ? nullptr : &mul_scales);
            auto pMatMul0Out = bufferMatMul0Out_local;
            // loop along K dimension
            auto valid_softmax_items = param.first_valid_softmax_items;
            {
                jit_mul_add_softmax_call_args call_args;
                call_args.p_in0 = pMatMul0Out;
                call_args.p_mul_in1 = mulScales.size() > 1 ? pMulIn1 + i1 : pMulIn1;
                call_args.p_add_in1 = pAddIn1_aux;
                call_args.p_out = pMatMul0Out;
                call_args.p_buffer = pMatMul0Out;
                call_args.p_scales0 = fqScales1.data();
                call_args.p_scales1 = &param.qk_quant;
                call_args.work_amount = valid_softmax_items;

                (*mulAddSoftmaxKernel[valid_softmax_items % vec_size])(&call_args);
            }
            auto pOut_aux = pout + (i0 * param.batch_stride_in_attn + i1 * param.head_stride_in_attn) * outPrcSize;
            tensor2D<uint8_t> matQK(param.query_seq_len, param.key_seq_len, reinterpret_cast<uint8_t*>(bufferMatMul0Out_local), rnd_up(param.key_seq_len * sizeof(uint8_t), 64));
            tensor2D<int8_t> matV(param.key_seq_len, _create_param.head_size, reinterpret_cast<int8_t*>(pVIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
            tensor2D<float> matQKV(param.query_seq_len, _create_param.head_size, reinterpret_cast<float*>(bufferMatMul1Out_local), _create_param.head_size_aligned * sizeof(float));
            amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQKV);
            (*qKVGemm_ops[threadNum])(matQK, matV, 0, _create_param.head_size, pp);
            if (convertReorderKernel) {
                // matmul1: [batch, num_heads, query_seq_len, head_size]
                // attn_output: [batch, query_seq_len, num_heads * head_size]
                jit_convert_reorder_call_args call_args;
                call_args.p_in = bufferMatMul1Out_local;
                call_args.p_out = pOut_aux;
                call_args.p_scales = qkv_quant.data();
                call_args.outter_work_amount = param.query_seq_len;

                (*convertReorderKernel)(&call_args);
            }
        });
    } else {
        int seq_cout_all = rnd_up(param.query_seq_len, 32) / 32;
        int work_amount = param.batch * _create_param.num_heads * seq_cout_all;
        int numThreads = parallel_get_max_threads();
        parallel_for(numThreads, [&](int threadNum) {
            int i0;
            int i1;
            int seq;
            int start {0}, end {0};
            splitter(work_amount, numThreads, threadNum, start, end);

            parallel_it_init(start, i0, param.batch, i1, _create_param.num_heads, seq, seq_cout_all);
            uint8_t* prev_k = nullptr;
            uint8_t* prev_v = nullptr;
            for (int iwork = start; iwork < end; ++iwork) {
                int seq_start = seq * 32;
                int seq_end = std::min(static_cast<size_t>(seq_start) + 32, param.query_seq_len);
                int seq_cout = seq_end - seq_start;
                // q: [batch, num_heads, query_seq_len, head_size]
                // k: [batch, num_heads, key_seq_len, head_size]
                // v: [batch, num_heads, value_seq_len, head_size]
                auto pQIn0_aux = pQIn0 + (i0 * param.batch_stride_in_q + i1 * param.head_stride_in_q + seq_start * _create_param.head_size_aligned);
                auto pKIn0_aux = pKIn0[i0] + i1 * param.head_stride_in_kv;
                auto pVIn0_aux = pVIn0[i0] + i1 * param.head_stride_in_kv;

                auto pAddIn1_aux = pAddIn1 + i0 * param.batch_stride_in_attn_mask;

                auto bufferMatMul0Out_local = reinterpret_cast<uint8_t*>(bufferMatMul0Out.get() + threadNum * bufferMatMul0OutSize);
                auto bufferMatMul1Out_local = reinterpret_cast<uint8_t*>(bufferMatMul1Out.get() + threadNum * bufferMatMul1OutSize);
                
                tensor2D<int8_t> matQ(seq_cout, _create_param.head_size, reinterpret_cast<int8_t*>(pQIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
                tensor2D<int8_t> matK(param.key_seq_len, _create_param.head_size, reinterpret_cast<int8_t*>(pKIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
                tensor2D<float> matQK(seq_cout, param.key_seq_len, reinterpret_cast<float*>(bufferMatMul0Out_local), rnd_up(param.key_seq_len * sizeof(float), 64));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(matQK);
                (*qKtrGemm_ops[threadNum])(matQ, matK, 0, param.key_seq_len, pp, prev_k == pKIn0_aux);
                prev_k = pKIn0_aux;

                auto pMulIn1 = reinterpret_cast<float*>(mulScales.empty() ? nullptr : &mul_scales);
                auto pMatMul0Out = bufferMatMul0Out_local;
                // loop along K dimension
                auto valid_softmax_items = param.first_valid_softmax_items + seq_start;
                for (size_t m = 0; m < seq_cout; m++) {
                    jit_mul_add_softmax_call_args call_args;
                    call_args.p_in0 = pMatMul0Out + m * rnd_up(param.key_seq_len * sizeof(float), 64);
                    call_args.p_mul_in1 = mulScales.size() > 1 ? pMulIn1 + i1 : pMulIn1;
                    call_args.p_add_in1 = pAddIn1_aux;
                    call_args.p_out = pMatMul0Out + m * rnd_up(param.key_seq_len * sizeof(int8_t), 64);
                    call_args.p_buffer = pMatMul0Out + m * rnd_up(param.key_seq_len * sizeof(float), 64);
                    call_args.p_scales0 = fqScales1.data();
                    call_args.p_scales1 = &param.qk_quant;
                    call_args.work_amount = valid_softmax_items;

                    (*mulAddSoftmaxKernel[valid_softmax_items % vec_size])(&call_args);
                    // attn_scores = torch.where(causal_mask, attn_scores, mask_value)
                    if (param.key_seq_len > valid_softmax_items) {
                        auto *invalidPtr = static_cast<int8_t*>(call_args.p_out) + valid_softmax_items;
                        memset(invalidPtr, 0, (param.key_seq_len - valid_softmax_items) * _create_param.qkv_precision.size());
                        valid_softmax_items = std::min(valid_softmax_items + 1, param.key_seq_len);
                    }
                }
                // attn_output: [batch, query_seq_len, num_heads * head_size]
                auto pOut_aux = pout + (i0 * param.batch_stride_in_attn + i1 * param.head_stride_in_attn
                    + seq_start * param.head_stride_in_attn * _create_param.num_heads) * outPrcSize;
                tensor2D<uint8_t> matQKI8(seq_cout, param.key_seq_len, reinterpret_cast<uint8_t*>(bufferMatMul0Out_local), rnd_up(param.key_seq_len * sizeof(uint8_t), 64));
                tensor2D<int8_t> matV(param.key_seq_len, _create_param.head_size, reinterpret_cast<int8_t*>(pVIn0_aux), _create_param.head_size_aligned * sizeof(int8_t));
                tensor2D<float> matQKV(seq_cout, _create_param.head_size, reinterpret_cast<float*>(bufferMatMul1Out_local), _create_param.head_size_aligned * sizeof(float));
                amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp2(matQKV);
                (*qKVGemm_ops[threadNum])(matQKI8, matV, 0, _create_param.head_size, pp2, prev_v == pVIn0_aux);
                prev_v = pVIn0_aux;
                if (convertReorderKernel) {
                    // matmul1: [batch, num_heads, query_seq_len, head_size]
                    // attn_output: [batch, query_seq_len, num_heads * head_size]
                    jit_convert_reorder_call_args call_args;
                    call_args.p_in = bufferMatMul1Out_local;
                    call_args.p_out = pOut_aux;
                    call_args.p_scales = qkv_quant.data();
                    call_args.outter_work_amount = seq_cout;

                    (*convertReorderKernel)(&call_args);
                }
                parallel_it_step(i0, param.batch, i1, _create_param.num_heads, seq, seq_cout_all);
            }
        });
    }
}

void MHAGPT::Impl::exec(const ExecParam& param) {
    if (_create_param.qkv_precision == Precision::FP32) {
        assert(false);
    } else if (_create_param.qkv_precision == Precision::BF16) {
        mha_bf16(param);
    } else if (_create_param.qkv_precision == Precision::I8) {
        mha_i8(param);
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

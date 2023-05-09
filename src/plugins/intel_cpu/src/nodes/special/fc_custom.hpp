#pragma once
#include <stdint.h>
#include <stddef.h>
#include <memory>

class FC {
public:
    enum FCType {
        FCType_S8,
        FCType_BF16,
        FCType_BF16_W8
    };
    FC();
    void init(size_t threads, FCType t);
    // s8 s8
    void fc_s8s8s8_dq_gelu_q(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* q);
    void fc_s8s8s8_dq_q(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* q);
    void fc_s8s8s8_dq_bias_gelu_q(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias, float* q);
    void fc_s8s8s8_dq_bias_q(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias, float* q);

    void fc_s8s8bf16_dq_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq);
    void fc_s8s8bf16_dq(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq);
    void fc_s8s8bf16_dq_bias_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias);
    void fc_s8s8bf16_dq_bias(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias);

    void fc_s8s8f32_dq_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq);
    void fc_s8s8f32_dq(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq);
    void fc_s8s8f32_dq_bias_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias);
    void fc_s8s8f32_dq_bias(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias);

    // bf16 s8
    void set_q_dq(float q, float dq);
    static void get_min_max(int8_t* weight, size_t dim0, size_t dim1, size_t stride, float& min, float& max);
    void fc_bf16s8bf16_dq_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq);
    void fc_bf16s8bf16_dq(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq);
    void fc_bf16s8bf16_dq_bias_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias);
    void fc_bf16s8bf16_dq_bias(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias);

    void fc_bf16s8f32_dq_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq);
    void fc_bf16s8f32_dq(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq);
    void fc_bf16s8f32_dq_bias_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias);
    void fc_bf16s8f32_dq_bias(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* dq, float* bias);

    // bf16 bf16
    void fc_bf16bf16bf16_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K);
    void fc_bf16bf16bf16(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K);
    void fc_bf16bf16bf16_bias_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* bias);
    void fc_bf16bf16bf16_bias(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* bias);

    void fc_bf16bf16f32_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K);
    void fc_bf16bf16f32(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K);
    void fc_bf16bf16f32_bias_gelu(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* bias);
    void fc_bf16bf16f32_bias(int8_t* src, int8_t* weight, int8_t* dst, size_t M, size_t N, size_t K, float* bias);
private:
    struct Impl;
    std::shared_ptr<Impl> _impl;
};

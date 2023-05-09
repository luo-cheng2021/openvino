#pragma once
#include <stdint.h>
#include <stddef.h>
#include <memory>

class Matmul {
public:
    enum Type {
        Type_S8,
        Type_S8_v,
        Type_U8,
        Type_BF16,
        Type_BF16_v
    };
    Matmul(Type t, bool transpose);
    void matmul_s8s8f32(int8_t* A, int8_t* B, float* C, size_t lda, size_t ldb, size_t ldc, size_t M, size_t N, size_t K);
    void matmul_u8s8f32(uint8_t* A, int8_t* B, float* C, size_t lda, size_t ldb, size_t ldc, size_t M, size_t N, size_t K);
    void gemAvB_s8s8f32(int8_t* A, int8_t* B, float* C, size_t lda, size_t M, size_t K);

    void matmul_bf16bf16f32(int8_t* A, int8_t* B, float* C, size_t ldb, size_t ldc, size_t M, size_t N, size_t K);
    void gemAvB_bf16bf16f32(int8_t* A, int8_t* B, float* C, size_t ldb, size_t ldc, size_t M, size_t N, size_t K);
private:
    struct Impl;
    std::shared_ptr<Impl> _impl;
};

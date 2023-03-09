#pragma once

#include <stdint.h>
#include <openvino/core/type/bfloat16.hpp>

namespace ov {
namespace intel_cpu {

#define rndup(x, n) (((x + n - 1)/n)*n)

template<typename T>
struct tensor2D {
    size_t dims[2];
    std::shared_ptr<T> data;
    size_t stride;
    size_t padded_dim1;
    tensor2D() {
        dims[0] = 0;
        dims[1] = 0;
    }

    tensor2D(size_t d0, size_t d1) {
        resize(d0, d1);
        //fill_rnd();
    }

    tensor2D(size_t d0, size_t d1, T* p) {
        dims[0] = d0;
        dims[1] = d1;
        stride = d1 * sizeof(T);
        // if (stride % 64) {
        //     auto stride_fix = rndup(stride, 64);
        //     std::cout << "\tWarnning: stride " << stride << " is not aligned to cache line, will increase to " << stride_fix
        //               << " (" << stride_fix/64 << " cache lines)\n";
        //     stride = stride_fix;
        // }
        padded_dim1 = stride / sizeof(T);

        // align begin address to cache line is vital, so tile load can
        // use all bandwidth (L1D/L2 only deliver data in unit of 64-byte aligned cache-line)
        data = std::shared_ptr<T>(p, [](void * p) { });
    }

    tensor2D(size_t d0, size_t d1, T* p, size_t _stride) {
        dims[0] = d0;
        dims[1] = d1;
        stride = _stride;
        // if (stride % 64) {
        //     auto stride_fix = rndup(stride, 64);
        //     std::cout << "\tWarnning: stride " << stride << " is not aligned to cache line, will increase to " << stride_fix
        //               << " (" << stride_fix/64 << " cache lines)\n";
        //     stride = stride_fix;
        // }
        padded_dim1 = stride / sizeof(T);

        // align begin address to cache line is vital, so tile load can
        // use all bandwidth (L1D/L2 only deliver data in unit of 64-byte aligned cache-line)
        data = std::shared_ptr<T>(p, [](void * p) { });
    }

    tensor2D<T> Tr() {
        tensor2D<T> ret(dims[1], dims[0]);
        for(size_t c0=0; c0 < dims[0]; ++c0) {
            for(size_t c1=0; c1 < dims[1]; ++c1) {
                ret(c1, c0) = (*this)(c0, c1);
            }
        }
        return ret;
    }

    void resize(size_t d0, size_t d1) {
        dims[0] = d0;
        dims[1] = d1;
        stride = d1 * sizeof(T);
        if (stride % 64) {
            auto stride_fix = rndup(stride, 64);
            std::cout << "\tWarnning: stride " << stride << " is not aligned to cache line, will increase to " << stride_fix
                      << " (" << stride_fix/64 << " cache lines)\n";
            stride = stride_fix;
        }
        padded_dim1 = stride / sizeof(T);

        // align begin address to cache line is vital, so tile load can
        // use all bandwidth (L1D/L2 only deliver data in unit of 64-byte aligned cache-line)
        data = std::shared_ptr<T>(
                    reinterpret_cast<T*>(aligned_alloc(64, dims[0] * stride)),
                    [](void * p) { free(p); });
    }

    T & operator[](int i) {
        return data.get()[i];
    }

    const T & operator[](int i) const {
        return data.get()[i];
    }

    //https://stackoverflow.com/questions/1936399/c-array-operator-with-multiple-arguments
    T & operator()(int i0, int i1) {
        return (*this)[i0 * padded_dim1 + i1];
    }

    const T & operator()(int i0, int i1) const {
        return (*this)[i0 * padded_dim1 + i1];
    }

    void fill_rnd() {
        for(size_t i = 0; i<dims[0]*padded_dim1; i++) {
            // lower mantissa can help to avoid small errors in accuracy comparison
            (*this)[i] = (rand() & 1) - 0.5;
        }
    }

    void operator=(const T & v) {
        for(size_t k = 0; k<dims[0]*padded_dim1; k++)
            (*this)[k] = v;
    }

    void operator=(const tensor2D<T> & t2) {
        assert(dims[0]*dims[1] == t2.dims[0] * t2.dims[1]);
        for(size_t c0 = 0; c0 < dims[0]; c0++)
        for(size_t c1 = 0; c1 < dims[1]; c1++) {
            size_t k = c0*dims[1] + c1;
            auto c2 = k / t2.dims[1];
            auto c3 = k % t2.dims[1];
            (*this)(c0, c1) = t2(c2, c3);
        }
    }

    bool operator==(const tensor2D<T> & rhs) {
        bool ok = true;
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        for(size_t i0=0; i0<dims[0]; i0++)
        for(size_t i1=0; i1<dims[1]; i1++) {
            if ((*this)(i0,i1) != rhs(i0,i1)) {
                std::cout << " operator== failed at (" << i0 << ", " << i1 << ")  value "
                          << (*this)(i0,i1) << "!=" << rhs(i0,i1) << std::endl;
                ok = false;
                return ok;
            }
        }
        return ok;
    }
    friend std::ostream& operator<<(std::ostream& out, const tensor2D<T>& obj) {
        int i0;
        auto showline = [&](size_t i) {
            out << "[" << i << "," << 0 << "]: ";
            int i1;
            for(i1=0; i1<obj.dims[1] && i1 < 8; i1++) {
                out << obj(i0,i1) << ",";
            }
            if (i1 < obj.dims[1]) out << "...";
            out << std::endl;
        };
        for(i0=0; i0 < obj.dims[0] && i0 < 32; i0++) {
            showline(i0);
        }
        if (i0 < obj.dims[0]) {
            out << "... ... ... ..." << std::endl;
            showline(obj.dims[0] - 1);
        }
        return out;
    }
};

// KpackedB is B matrix in block of 32x32 arranged in column-major
// each 32x32 block is composed of 2 horizontal neighboring tiles
// of 32x16(further repacked as 16x16x2)
// 
//  +---+---+-----
//  |B0 |B1 |
//  |   |   |
//  +---+---+
//  |   |   | 
// 
struct KpackedB {
    std::shared_ptr<bfloat16> data;
    int K;
    int N;
    int Kblocks;
    int Nblocks;
    KpackedB(tensor2D<bfloat16> & matB) {
        K = matB.dims[0];
        N = matB.dims[1];
        Kblocks = (K + 31)/32;
        Nblocks = (N + 31)/32;
        int total_size = Kblocks * Nblocks * 32 * 32 * sizeof(bfloat16);
        data = std::shared_ptr<bfloat16>(
                    reinterpret_cast<bfloat16*>(aligned_alloc(64, rndup(total_size, 64))),
                    [](void * p){ ::free(p); });
        
        for (int k = 0; k < Kblocks*32; k++)
        for (int n = 0; n < Nblocks*32; n++) {
            if (k < K && n < N)
                (*this)(k, n) = matB(k, n);
            else
                (*this)(k, n) = 0; // padding zero
        }
    }

    bfloat16 & operator()(int k, int n) {
        int kb = k/32;
        int nb = n/32;
        int block_offset = (nb*Kblocks + kb)*(32*32);
        int kr = k % 32;
        int nr = n % 32;
        int offset = block_offset;
        
        if (nr >= 16) {
            //B1
            offset += 32*16;
            nr -= 16;
        }
        // (kr,nr) is coordinate in 32x16 submatrix
        // after repack it becomes offset in 16x16x2
        offset += (kr/2)*32 + 2*nr + (kr&1);
        return data.get()[offset];
    }
};

class Gemm {
public:
    // pack function
    // pack to 16*(16*2), amx bf16 required
    // NOTE: packedB must be pre-alloced, size: rndup(K, 32)/32 * rndup(N, 32)/32 * 32 * 32 * sizeof(bfloat16)
    //KpackedB packB_16x16x2_bf16(tensor2D<bfloat16>& matB);

    // Gemm:   [M, K] * [K, N], amx bf16 version
    // GevAmB: [1, K] * [K, N], amx bf16 version, TBD
    // FC: Gemm+GevAmB
    // matB: packed to 16*(16*2), amx bf16 required
    void gemm(tensor2D<bfloat16> & matA, KpackedB & matB, tensor2D<bfloat16> & matC);

    // GemAvB: [M, K] * [K, 1], avx512 bf16 version, B will be packed inside the function.
    // special case: A[1, K] * transpose(B[N, K]) = C[1, N] equals
    //               A[N, K] * B[K, 1] = C[N, 1]
    // Matmul: Gemm+GemAvB
    void gemAvB(tensor2D<bfloat16> & matA, bfloat16 * vecB, bfloat16 * vecC, tensor2D<bfloat16>& Bpadded, bool outFP32 = true);

    // GevAmB: [1, K] * [K, N], avx512 bf16 version, TBD
    // Matmul: Gemm+GevAmB
};

}
}
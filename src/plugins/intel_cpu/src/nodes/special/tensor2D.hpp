#pragma once

#include <memory>
#include <iostream>
#include <functional>
#ifdef ENABLE_NUMA
#include "numa.h"
#endif

#define rndup(x, n) (((x + n - 1)/n)*n)

template<typename T>
struct tensor2D {
    int dims[2];
    std::shared_ptr<T> data;
    int capacity;
    int stride;
    int padded_dim1;
    tensor2D() {
        dims[0] = 0;
        dims[1] = 0;
        capacity = 0;
    }

    operator bool() {
        return dims[0] * dims[1] > 0;
    }

    tensor2D(int d0, int d1) {
        capacity = 0;
        resize(d0, d1);
        //fill_rnd();
    }

    tensor2D(int d0, int d1, T * ext, int _stride) {
        capacity = 1;
        data = std::shared_ptr<T>(ext, [](void *) {});
        dims[0] = d0;
        dims[1] = d1;
        stride = _stride;
        padded_dim1 = stride / sizeof(T);
    }

    tensor2D<T> Tr() {
        tensor2D<T> ret(dims[1], dims[0]);
        for(int c0=0; c0 < dims[0]; ++c0) {
            for(int c1=0; c1 < dims[1]; ++c1) {
                ret(c1, c0) = (*this)(c0, c1);
            }
        }
        return ret;
    }
    tensor2D<T> clone() {
        tensor2D<T> ret;
        ret.resize(dims[0], dims[1]);
        if (ret.stride == stride) {
            memcpy(ret.data.get(), data.get(), dims[0] * stride);
        }else{
            for(int i=0;i<dims[0];i++) {
                memcpy(&ret(i,0), &(*this)(i,0), ret.stride);
            }
        }
        return ret;
    }
    void resize(int d0, int d1) {
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

        // resize method never shrink capacity
        auto need_capacity = dims[0] * stride;
        if (capacity < need_capacity) {
            capacity = need_capacity;
            // align begin address to cache line is vital, so tile load can
            // use all bandwidth (L1D/L2 only deliver data in unit of 64-byte aligned cache-line)

#ifdef ENABLE_NUMA
            if (USE_NUMA) {
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(numa_alloc_local(capacity)),
                            [need_capacity](void * p){ numa_free(p, need_capacity); });
            } else {
#else
            {
#endif
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(aligned_alloc(64, capacity)),
                            [](void * p) { ::free(p); });
            }
            if (reinterpret_cast<uintptr_t>(data.get()) % 64)
                std::cout << "WARNING: resize(), data is not cache-line aligned!" << std::endl;
        }
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
        for(int i = 0; i<dims[0]*padded_dim1; i++) {
            // lower mantissa can help to avoid small errors in accuracy comparison
            (*this)[i] = (rand() & 1) - 0.5;
        }
    }

    void operator=(const T & v) {
        for(int k = 0; k<dims[0]*padded_dim1; k++)
            (*this)[k] = v;
    }

    void operator=(const tensor2D<T> & t2) {
        assert(dims[0]*dims[1] == t2.dims[0] * t2.dims[1]);
        for(int c0 = 0; c0 < dims[0]; c0++)
        for(int c1 = 0; c1 < dims[1]; c1++) {
            int k = c0*dims[1] + c1;
            auto c2 = k / t2.dims[1];
            auto c3 = k % t2.dims[1];
            (*this)(c0, c1) = t2(c2, c3);
        }
    }

    bool operator==(const tensor2D<T> & rhs) {
        bool ok = true;
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            if ((*this)(i0,i1) != rhs(i0,i1)) {
                std::cout << " operator== failed at (" << i0 << ", " << i1 << ")  value "
                          << (*this)(i0,i1) << "!=" << rhs(i0,i1) << std::endl;
                ok = false;
                return ok;
            }
        }
        return ok;
    }
    bool compare(const tensor2D<T> & rhs, float tolerance) {
        float max_abs_diff = 0;
        float max_rel_diff = 0;
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            auto diff = std::fabs((*this)(i0,i1) - rhs(i0,i1));
            auto rel_diff = diff/std::fabs((*this)(i0,i1));
            max_abs_diff = std::max(max_abs_diff, diff);
            if (std::fabs((*this)(i0,i1) > 0) && diff > 0)
                max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
        std::cout << "max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff;
        return tolerance > max_abs_diff;
    }
    friend std::ostream& operator<<(std::ostream& out, const tensor2D<T>& obj) {
        int i0;
        auto showline = [&](int i) {
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

using func_act = std::function<float(float)>;

template<typename T>
void matmul(tensor2D<T> & A,
            tensor2D<T> & B,
            tensor2D<T> & C,
            float * bias = nullptr,
            func_act act = func_act()) {
    int M = C.dims[0];
    int N = C.dims[1];
    int K = A.dims[1];
    assert(B.dims[0] == K);
    assert(B.dims[1] == N);
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            float sum = C(m,n);
            int k;
            for (k = 0; (k + 32) <= K; k += 32) {
                float psum0 = 0;
                float psum1 = 0;
                for(int p = 0; p < 32; p+=2) {
                    psum0 += static_cast<float>(A(m,k+p)) * static_cast<float>(B(k+p,n));
                    psum1 += static_cast<float>(A(m,k+p+1)) * static_cast<float>(B(k+p+1,n));
                }
                sum += (psum0 + psum1);
            }
            for(; k < K; k++) {
                sum += static_cast<float>(A(m,k)) * static_cast<float>(B(k,n));
            }
            if (bias) {
                sum += bias[n];
            }
            if (act) {
                sum = act(sum);
            }
            C(m,n) = sum;
        }
    }
}

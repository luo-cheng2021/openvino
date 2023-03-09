#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
#include <thread>

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include "gemm_custom.hpp"

namespace ov {
namespace intel_cpu {

/*
https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-amx-instructions/intrinsics-for-amx-tile-instructions/tile-loadconfig.html

void _tile_loadconfig (const void * mem_addr)
	format of memory payload. each field is a byte.
		 0: palette_id
		 1: startRow (8b)
	 2-15: reserved (must be zero)
	16-17: tile0.colsb -- bytes_per_row
	18-19: tile1.colsb
	20-21: tile2.colsb
			...
	46-47: tile15.colsb
		48: tile0.rows
		49: tile1.rows
		50: tile2.rows
			 ...
		63: tile15.rows

void _tile_storeconfig (void * mem_addr)
    Stores the current tile configuration to a 64-byte memory location specified by "mem_addr".
    The tile configuration format is specified below, and includes the tile type pallette,
    the number of bytes per row, and the number of rows. If tiles are not configured,
    all zeroes will be stored to memory.
*/
struct tileconfig_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    tileconfig_t() = default;
    tileconfig_t(int palette, int _startRow, int numTiles, int _rows, int columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        for(int i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        for(int i = 0; i < numTiles; i++) {
            cols[i] = columnsBytes;
            rows[i] = _rows;
        }
        for(int i = numTiles; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
        load();
    }
    ~tileconfig_t() {
        _tile_release();
    }
    void load() {
        std::cout << "\ttile load config ... " << std::flush;
        _tile_loadconfig(this);
        std::cout << *this << std::flush << std::endl;
    }
    void store() {
        _tile_storeconfig(this);
    }
    friend std::ostream& operator<<(std::ostream& out, const tileconfig_t& cfg) {
        out << " palette_id=" << static_cast<int>(cfg.palette_id);
        out << " startRow=" << static_cast<int>(cfg.startRow);
        out << " row x colsb=(";
        for (int i = 0; i < 16;i++) {
            if (cfg.rows[i] == 0 && cfg.cols[i] == 0)
                continue;
            if (i > 0) out << ",";
            out << static_cast<int>(cfg.rows[i]) << "x" << static_cast<int>(cfg.cols[i]);
        }
        out << ")";
        return out;
    }
} __attribute__ ((__packed__));

//================================================================================
// fc layer:  B is const and can be arranged into best sequential format
//------------------
// register blocking:
// A bfloat16_16x32
// B bfloat16_32x16 (layout: 16x16x4)
// C    float_16x16
//
//         B0 B1
//         ...
//         B0 B1
//A0 : A0   C C
//A1 : A1   C C
//------------------
// cache blocking:
//                Bb:     Kx32
//   Ab:  m0*32xK Cb:  m0*32x32
//
// (Ab + Bb) should fit in L2 cache
//    (m0*32xK*elesz + Kx32*elesz) < L2
//     m0 < L2/(32*K*elesz) - 1
//
struct FC {
    static constexpr int tC00 = 0;
    static constexpr int tC01 = 1;
    static constexpr int tC10 = 2;
    static constexpr int tC11 = 3;
    static constexpr int tA0 = 4;
    static constexpr int tA1 = 5;
    static constexpr int tB0 = 6;
    static constexpr int tB1 = 7;

    // for processing tails
    tensor2D<bfloat16> Atails;

    FC() {}

    // post process kernels, tC00 ~ tC11
    struct PP2bf16 {
        tensor2D<float> buffC;
        PP2bf16() : buffC(16, 2*16) {}
        void postProcess16x32(int8_t * pdst, int stride, int valid_m, int valid_n) {
            float * psrc = &buffC(0,0);
            if (valid_m >= 16 && valid_n >= 32) {
                for(int i = 0; i < 16; i ++) {
                    auto b = _mm512_loadu_epi16(psrc);
                    auto a = _mm512_loadu_epi16(psrc + 16);
                    auto c = _mm512_cvtne2ps_pbh(a, b);
                    _mm512_storeu_epi16(pdst, c);   // 32 bf16
                    pdst += stride;
                    psrc += 32;
                }
            } else {
                __mmask32 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
                for(int i = 0; i < valid_m; i ++) {
                    auto b = _mm512_loadu_epi16(psrc);
                    auto a = _mm512_loadu_epi16(psrc + 16);
                    auto c = _mm512_cvtne2ps_pbh(a, b);
                    _mm512_mask_storeu_epi16(pdst, k, c);   // 32 bf16
                    pdst += stride;
                    psrc += 32;
                }
            }
        }
        void operator()(bfloat16 * pC, int stride, int valid_m, int valid_n) {
            _tile_stored(tC00, &buffC(0,0), buffC.stride);
            _tile_stored(tC01, &buffC(0,16), buffC.stride);
            postProcess16x32(reinterpret_cast<int8_t*>(pC), stride, valid_m, valid_n);

            if (valid_m > 16) {
                _tile_stored(tC10, &buffC(0,0), buffC.stride);
                _tile_stored(tC11, &buffC(0,16), buffC.stride);
                postProcess16x32(reinterpret_cast<int8_t*>(pC) + 16*stride, stride, valid_m-16, valid_n);
            }
        }
    };

    // matB has been pre k-packed
    template<typename PP>
    void operator()(tensor2D<bfloat16> & matA,
                    KpackedB & matB,
                    tensor2D<bfloat16> & matC,
                    PP ppkernel) {
        int M = matC.dims[0];
        int N = matC.dims[1];
        int K = matA.dims[1];
        assert(K == matB.K);
        assert(N == matB.N);

        int elesz = sizeof(uint16_t);
        int L2 = 2048*1024; // 2MB
        int slice_size = 32*K*elesz;
        int mc = L2/slice_size - 1;
        assert(mc > 0);

        int mtails = M % 32;

        if (mtails > 0) {
            if (K > Atails.dims[1])
                Atails.resize(32, rndup(K, 32));
            // copy tails into Atails (in unit of 32x32)
            for (int m = 0; m < mtails; m++) {
                memcpy(&Atails(m, 0), &matA(M - mtails + m, 0), matA.stride);
                if (Atails.stride > matA.stride) {
                    memset(reinterpret_cast<int8_t*>(&Atails(m, 0)) + matA.stride,
                           0,
                           Atails.stride - matA.stride);
                }
            }
        }

        for (int m0 = 0; m0 < M; m0 += mc*32) { // loop m:
            int m1 = std::min(m0 + mc*32, M);
            for(int n = 0; n < N; n+=32) {   // loop n: reuse Ab in L2
                // (m0*32xK) * (Kx32) => m0*32x32
                int valid_n = std::min(N - n, 32);
                for (int m = m0; m < m1; m+=32) { // loop mi: reuse Bb in L2
                    int valid_m = std::min(M - m, 32);
                    auto * pA0 = &matA(m, 0);
                    auto * pA1 = &matA(m + 16, 0);
                    auto strideA = matA.stride;
                    auto * pB = &matB(0, n);
                    if (valid_m < 32) {
                        // use Atails buffer to prevent memory read segmentfault
                        pA0 = &Atails(0, 0);
                        pA1 = &Atails(16, 0);
                        strideA = Atails.stride;
                    }
                    _tile_zero(tC00);
                    _tile_zero(tC01);
                    _tile_zero(tC10);
                    _tile_zero(tC11);
                    for (int k = 0; k < K; k += 32) {
                        _tile_loadd(tA0, pA0 + k, strideA);
                        _tile_loadd(tB0, pB, 64); pB += (16*32);
                        _tile_dpbf16ps(tC00, tA0, tB0);
                        _tile_loadd(tA1, pA1 + k, strideA);
                        _tile_dpbf16ps(tC10, tA1, tB0);
                        _tile_loadd(tB1, pB, 64); pB += (16*32);
                        _tile_dpbf16ps(tC01, tA0, tB1);
                        _tile_dpbf16ps(tC11, tA1, tB1);
                    }
                    // post processing the accumulator tiles
                    //  - add bias
                    //  - do activations
                    //  - convert into bfloat16
                    //  - store into C matrix
                    (ppkernel)(&matC(m, n), matC.stride, valid_m, valid_n);
                }
            }
        }
    }
};

void Gemm::gemm(tensor2D<bfloat16> & matA, KpackedB & matB, tensor2D<bfloat16> & matC) {
    FC fc;
    FC::PP2bf16 pp;

    tileconfig_t tfg(1, 0, 8, 16, 64);
    fc(matA, matB, matC, pp);
}

void Gemm::gemAvB(tensor2D<bfloat16> & matA, bfloat16 * vecB, bfloat16 * vecC, tensor2D<bfloat16>& Bpadded, bool outFP32) {
    int M = matA.dims[0];
    int K = matA.dims[1];

    if (K % 32) {
        if (K > Bpadded.dims[1])
            Bpadded.resize(1, rndup(K, 32));
        auto newB = &Bpadded(0, 0);
        memset(newB, 0, Bpadded.stride);
        memcpy(newB, vecB, K * sizeof(bfloat16));
        vecB = newB;
    }

    auto nstride = matA.stride/sizeof(bfloat16);
    for(int m = 0; m < M; m += 16) {
        auto * pA = &matA(m, 0);
        auto * pBi32 = reinterpret_cast<int32_t*>(vecB);
        __m512 regC0 = _mm512_setzero();
        __m512 regC1 = _mm512_setzero();
        for(int k = 0; k < K; k += 32, pA += 32, pBi32 += 16) {
            // handle Ab: 16x32
            // transposed in register as 16x16x2
            //   r0: (a0,a1)(b0,b1)....
            //   r1: (a2,a3)(b2,b3)....
            //      ...
            //   rf: (a30,a31),(b30,b31)....
            // 
            __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
            __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
            r0 = _mm512_loadu_epi32(pA + 0*nstride);
            r1 = _mm512_loadu_epi32(pA + 1*nstride);
            r2 = _mm512_loadu_epi32(pA + 2*nstride);
            r3 = _mm512_loadu_epi32(pA + 3*nstride);
            r4 = _mm512_loadu_epi32(pA + 4*nstride);
            r5 = _mm512_loadu_epi32(pA + 5*nstride);
            r6 = _mm512_loadu_epi32(pA + 6*nstride);
            r7 = _mm512_loadu_epi32(pA + 7*nstride);
            r8 = _mm512_loadu_epi32(pA + 8*nstride);
            r9 = _mm512_loadu_epi32(pA + 9*nstride);
            ra = _mm512_loadu_epi32(pA + 10*nstride);
            rb = _mm512_loadu_epi32(pA + 11*nstride);
            rc = _mm512_loadu_epi32(pA + 12*nstride);
            rd = _mm512_loadu_epi32(pA + 13*nstride);
            re = _mm512_loadu_epi32(pA + 14*nstride);
            rf = _mm512_loadu_epi32(pA + 15*nstride);

            t0 = _mm512_unpacklo_epi32(r0,r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
            t1 = _mm512_unpackhi_epi32(r0,r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
            t2 = _mm512_unpacklo_epi32(r2,r3); //  32  48  33  49 ...
            t3 = _mm512_unpackhi_epi32(r2,r3); //  34  50  35  51 ...
            t4 = _mm512_unpacklo_epi32(r4,r5); //  64  80  65  81 ...  
            t5 = _mm512_unpackhi_epi32(r4,r5); //  66  82  67  83 ...
            t6 = _mm512_unpacklo_epi32(r6,r7); //  96 112  97 113 ...
            t7 = _mm512_unpackhi_epi32(r6,r7); //  98 114  99 115 ...
            t8 = _mm512_unpacklo_epi32(r8,r9); // 128 ...
            t9 = _mm512_unpackhi_epi32(r8,r9); // 130 ...
            ta = _mm512_unpacklo_epi32(ra,rb); // 160 ...
            tb = _mm512_unpackhi_epi32(ra,rb); // 162 ...
            tc = _mm512_unpacklo_epi32(rc,rd); // 196 ...
            td = _mm512_unpackhi_epi32(rc,rd); // 198 ...
            te = _mm512_unpacklo_epi32(re,rf); // 228 ...
            tf = _mm512_unpackhi_epi32(re,rf); // 230 ...

            r0 = _mm512_unpacklo_epi64(t0,t2); //   0  16  32  48 ...
            r1 = _mm512_unpackhi_epi64(t0,t2); //   1  17  33  49 ...
            r2 = _mm512_unpacklo_epi64(t1,t3); //   2  18  34  49 ...
            r3 = _mm512_unpackhi_epi64(t1,t3); //   3  19  35  51 ...
            r4 = _mm512_unpacklo_epi64(t4,t6); //  64  80  96 112 ...  
            r5 = _mm512_unpackhi_epi64(t4,t6); //  65  81  97 114 ...
            r6 = _mm512_unpacklo_epi64(t5,t7); //  66  82  98 113 ...
            r7 = _mm512_unpackhi_epi64(t5,t7); //  67  83  99 115 ...
            r8 = _mm512_unpacklo_epi64(t8,ta); // 128 144 160 176 ...  
            r9 = _mm512_unpackhi_epi64(t8,ta); // 129 145 161 178 ...
            ra = _mm512_unpacklo_epi64(t9,tb); // 130 146 162 177 ... 
            rb = _mm512_unpackhi_epi64(t9,tb); // 131 147 163 179 ...
            rc = _mm512_unpacklo_epi64(tc,te); // 192 208 228 240 ... 
            rd = _mm512_unpackhi_epi64(tc,te); // 193 209 229 241 ...
            re = _mm512_unpacklo_epi64(td,tf); // 194 210 230 242 ...
            rf = _mm512_unpackhi_epi64(td,tf); // 195 211 231 243 ...

            t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
            t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
            t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
            t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
            t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
            t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
            t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
            t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
            t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
            t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
            ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
            tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
            tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
            td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
            te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
            tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...

            r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
            r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
            r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
            r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
            r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
            r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
            r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
            r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
            r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
            r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
            ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
            rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
            rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
            rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
            re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
            rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255

            // vdpbf16ps
            regC0 = _mm512_dpbf16_ps(regC0, r0, _mm512_set1_epi32(pBi32[0]));
            regC1 = _mm512_dpbf16_ps(regC1, r1, _mm512_set1_epi32(pBi32[1]));
            regC0 = _mm512_dpbf16_ps(regC0, r2, _mm512_set1_epi32(pBi32[2]));
            regC1 = _mm512_dpbf16_ps(regC1, r3, _mm512_set1_epi32(pBi32[3]));
            regC0 = _mm512_dpbf16_ps(regC0, r4, _mm512_set1_epi32(pBi32[4]));
            regC1 = _mm512_dpbf16_ps(regC1, r5, _mm512_set1_epi32(pBi32[5]));
            regC0 = _mm512_dpbf16_ps(regC0, r6, _mm512_set1_epi32(pBi32[6]));
            regC1 = _mm512_dpbf16_ps(regC1, r7, _mm512_set1_epi32(pBi32[7]));
            regC0 = _mm512_dpbf16_ps(regC0, r8, _mm512_set1_epi32(pBi32[8]));
            regC1 = _mm512_dpbf16_ps(regC1, r9, _mm512_set1_epi32(pBi32[9]));
            regC0 = _mm512_dpbf16_ps(regC0, ra, _mm512_set1_epi32(pBi32[10]));
            regC1 = _mm512_dpbf16_ps(regC1, rb, _mm512_set1_epi32(pBi32[11]));
            regC0 = _mm512_dpbf16_ps(regC0, rc, _mm512_set1_epi32(pBi32[12]));
            regC1 = _mm512_dpbf16_ps(regC1, rd, _mm512_set1_epi32(pBi32[13]));
            regC0 = _mm512_dpbf16_ps(regC0, re, _mm512_set1_epi32(pBi32[14]));
            regC1 = _mm512_dpbf16_ps(regC1, rf, _mm512_set1_epi32(pBi32[15]));
        }
        regC0 = _mm512_add_ps(regC0, regC1);
        if (outFP32) {
            _mm512_store_ps(reinterpret_cast<float*>(vecC) + m, regC0);
        } else {
            auto regOut = _mm512_cvtne2ps_pbh(regC0, regC0); // only 16 bfloat16 results in lower 256bits 
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(vecC + m), _mm512_extracti64x4_epi64(regOut, 0));
        }
    }
}

}
}
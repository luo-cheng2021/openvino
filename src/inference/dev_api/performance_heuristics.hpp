// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include <cfloat>

#include "ngraph/ngraph.hpp"
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset10.hpp>

namespace ov {
struct MemBandwidthPressure {
    float max_mem_tolerance = UNKNOWN;
    float ratio_compute_convs = 0;
    float ratio_mem_limited_convs = 0;
    float ratio_compute_deconvs = 0;

    static constexpr float UNKNOWN = FLT_MAX;
    static constexpr float ALL = 1.0f;
    static constexpr float NONE = 0.0f;
    static constexpr float LIMITED = 0.5f;  // conservatively assume 1/2 utilization of the cache
};

struct LineSetHash {
    size_t operator()(const std::pair<void*, size_t>& p) const {
        return reinterpret_cast<size_t>(p.first) + p.second;
    }
};

struct SimMemory {
    size_t L1DSize = 32 * 1024;
    size_t L2Size = 2 * 1024 * 1024;
    size_t L3Size = 0;//1.875 * 1024 * 1024;  // 1.875 * 1024 * 1024 * 3 / 2;
    // latency https://stackoverflow.com/a/33065382
    float L1DLatency = 4;             // L1D hit, 2 ns //1ns
    float L2Latency = 16;             // L2 hit, 8 ns //4ns
    float L3Latency = 64;             // L3 hit, 32ns //16ns
    float DDRLatency = 200;           // hit local ddr, 100ns
    float DDRLatencybackground = 0.0f;// average additional latency caused by other streams
    float parrelMemAccess = 12;
    size_t free_size;
    float Latencys[5];
    size_t threadNum;
    float threadCostRatio = 1.0f;
    // only consider output
    std::unordered_map<void*, std::pair<size_t, size_t>> blocks;
    SimMemory(size_t thread_num) {
        resetCache();
        Latencys[0] = L1DLatency;
        Latencys[1] = L2Latency;
        Latencys[2] = L3Latency;
        Latencys[3] = DDRLatency;
        Latencys[4] = DDRLatency * 2;
        threadNum = thread_num;
        switch(thread_num) {
            case 4:
                threadCostRatio = 1 / 0.90f;
                break;
            case 8:
                threadCostRatio = 1 / 0.80f;
                break;
            default:
                break;
        }
    }
    void tryCache(void *ptr, size_t size) {
        // find room for new item
        while (free_size < size) {
            if (blocks.size() > 0) {
                auto it = blocks.begin();
                auto item_size = (*it).second.second - (*it).second.first;
                if (item_size > size) {
                    (*it).second.first += size;
                    free_size += size;
                    break;
                }
                blocks.erase(it);
                free_size += item_size;
            } else {
                break;
            }
        }
        // only cache fit in cache
        if (free_size > size) {
            blocks.insert({ptr, {0, size}});
            free_size -= size;
        }
    }
    void resetCache() {
        free_size = L2Size + L3Size;
        blocks.clear();
    }
    std::unordered_map<std::pair<void*, size_t>, bool, LineSetHash> lines;
    std::deque<std::pair<void*, size_t>> linesOrder;
    int accessLine(void* p, size_t offset, bool dirty = false) {
        // cache hit
        if (lines.find(std::make_pair(p, offset)) != lines.end()) {
            return 1;
        }
        bool force_wb = false;
        if (free_size < 64) {
            auto& item = linesOrder.front();
            if (lines[item]) {
                // if a new write alloc cause writing back, mean writing alloc is an update
                if (dirty && item.first == p)
                    force_wb = true;
            }
            lines.erase(item);
            linesOrder.pop_front();
            free_size += 64;
        }
        lines[std::make_pair(p, offset)] = dirty;
        linesOrder.push_back(std::make_pair(p, offset));
        free_size -= 64;
        return dirty ? (force_wb ? 3 : 1) : (force_wb ? 4 : 3);
    }
    float access1(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                  void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size) {
        float cost = 0;
        std::vector<size_t> in_curr_offsets(inputs_size, 0), out_curr_offsets(outputs_size, 0);
        int hits[5] = {0};
        int in_hits[5] = {0};
        int out_hits[5] = {0};
        while (1) {
            bool accessed = false;
            for (size_t i = 0; i < inputs_size; i++) {
                auto& offset = in_curr_offsets[i];
                void* input_mem = inputs_mem[i];
                size_t input_mem_size = inputs_mem_size[i] / threadNum;
                if (offset < input_mem_size) {
                    int level = accessLine(input_mem, offset);
                    offset += 64;
                    cost += Latencys[level];
                    accessed = true;
                    hits[level]++;
                    in_hits[level]++;
                }
            }
            for (size_t i = 0; i < outputs_size; i++) {
                auto& offset = out_curr_offsets[i];
                void* output_mem = outputs_mem[i];
                size_t output_mem_size = outputs_mem_size[i] / threadNum;
                if (offset < output_mem_size) {
                    int level = accessLine(output_mem, offset, true);
                    offset += 64;
                    cost += Latencys[level];
                    accessed = true;
                    hits[level]++;
                    out_hits[level]++;
                }
            }
            // all memory touched
            if (!accessed)
                break;
        }
        return cost / parrelMemAccess * threadNum * threadCostRatio;
    }
    float accessGEMM(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                  void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size) {
        float cost = 0;
        std::vector<size_t> in_curr_offsets(inputs_size, 0), out_curr_offsets(outputs_size, 0);
        int hits[5] = {0};
        int in_hits[5] = {0};
        int out_hits[5] = {0};
        while (1) {
            bool accessed = false;
            for (size_t i = 0; i < inputs_size; i++) {
                auto& offset = in_curr_offsets[i];
                void* input_mem = inputs_mem[i];
                size_t input_mem_size = inputs_mem_size[i] / threadNum;
                if (offset < input_mem_size) {
                    int level = accessLine(input_mem, offset);
                    offset += 64;
                    cost += Latencys[level];
                    accessed = true;
                    hits[level]++;
                    in_hits[level]++;
                }
            }
            // all memory touched
            if (!accessed)
                break;
        }
        while (1) {
            bool accessed = false;
            for (size_t i = 0; i < outputs_size; i++) {
                auto& offset = out_curr_offsets[i];
                void* output_mem = outputs_mem[i];
                size_t output_mem_size = outputs_mem_size[i] / threadNum;
                if (offset < output_mem_size) {
                    int level = accessLine(output_mem, offset, true);
                    offset += 64;
                    cost += Latencys[level];
                    accessed = true;
                    hits[level]++;
                    out_hits[level]++;
                }
            }
            // all memory touched
            if (!accessed)
                break;
        }

        return cost / parrelMemAccess * threadNum * threadCostRatio;
    }
    float accessConv(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                  void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size) {
        float cost = 0;
        std::vector<size_t> in_curr_offsets(inputs_size, 0), out_curr_offsets(outputs_size, 0);
        int hits[5] = {0};
        int in_hits[5] = {0};
        int out_hits[5] = {0};
        // convolution loop order is decided by minimum size of A or B
        //  In multithread, the minimum block will be loaded in each thread
        //  Need to consider: if A>L2 && B>L2 mean the minimum block will be loaded multiple times
        size_t thread_num_A = inputs_mem_size[0] >= inputs_mem_size[1] ? threadNum : 1;
        size_t thread_num_B = inputs_mem_size[0] < inputs_mem_size[1] ? threadNum : 1;
        while (1) {
            bool accessed = false;
            for (size_t i = 0; i < inputs_size; i++) {
                auto& offset = in_curr_offsets[i];
                void* input_mem = inputs_mem[i];
                size_t input_mem_size = inputs_mem_size[i] / (i == 0 ? thread_num_A : thread_num_B);
                if (offset < input_mem_size) {
                    int level = accessLine(input_mem, offset);
                    offset += 64;
                    cost += Latencys[level];
                    accessed = true;
                    hits[level]++;
                    in_hits[level]++;
                }
            }
            // all memory touched
            if (!accessed)
                break;
        }
        while (1) {
            bool accessed = false;
            for (size_t i = 0; i < outputs_size; i++) {
                auto& offset = out_curr_offsets[i];
                void* output_mem = outputs_mem[i];
                size_t output_mem_size = outputs_mem_size[i] / threadNum;
                if (offset < output_mem_size) {
                    int level = accessLine(output_mem, offset, true);
                    offset += 64;
                    cost += Latencys[level];
                    accessed = true;
                    hits[level]++;
                    out_hits[level]++;
                }
            }
            // all memory touched
            if (!accessed)
                break;
        }

        return cost / parrelMemAccess * threadNum * threadCostRatio;
    }
    int last_hit_level = 3;
    float access1Simple(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                  void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size) {
        float cost = 0;
        auto input_size_all =
            std::accumulate(inputs_mem_size, inputs_mem_size + inputs_size, size_t(0), std::plus<size_t>());
        auto output_size_all =
            std::accumulate(outputs_mem_size, outputs_mem_size + outputs_size, size_t(0), std::plus<size_t>());
        // auto size_all_per_thread = (input_size_all + output_size_all) / threadNum;
        // int cur_hit_level;
        // if (size_all_per_thread >= L2Size + L3Size) {
        //     // hit ddr
        //     cur_hit_level = std::max(3, last_hit_level);
        //     last_hit_level = 3;
        // } else if (size_all_per_thread <= L2Size) {
        //     // hit L2
        //     cur_hit_level = std::max(1, last_hit_level);
        //     last_hit_level = 1;
        // } else {
        //     // hit L2 + L3
        //     cur_hit_level = std::max(2, last_hit_level);
        //     last_hit_level = 2;
        // }
        // cost = size_all_per_thread / 64 * Latencys[cur_hit_level];

        // return cost / parrelMemAccess * threadNum * threadCostRatio;
        auto size_a_per_thread = inputs_mem_size[0] / threadNum;
        auto size_b_per_thread = (input_size_all - inputs_mem_size[0]) / threadNum;
        auto size_out_per_thread = output_size_all / threadNum;
        auto size_all_per_thread = size_a_per_thread + size_b_per_thread + size_out_per_thread;
        int cur_hit_level;
        if (size_all_per_thread >= L2Size + L3Size) {
            cur_hit_level = std::max(3, last_hit_level);
            last_hit_level = 3;
            cost = (size_a_per_thread + size_out_per_thread) / 64 * Latencys[cur_hit_level];
        } else if (size_all_per_thread <= L2Size) {
            // hit L2
            if (last_hit_level == 3) {
                cost = size_a_per_thread / 64 * Latencys[3];
            } else if (last_hit_level == 1) {
                cost = size_a_per_thread / 64 * Latencys[1];
            } else {
                cost = size_a_per_thread / 64 * Latencys[2];
            }
            cost += size_out_per_thread / 64 * Latencys[1];
            last_hit_level = 1;
        } else {
            // hit L2 + L3
            if (last_hit_level == 3) {
                cost = size_a_per_thread / 64 * Latencys[3];
            } else {
                // could not determine in L2 or L3, use the bad case
                cost = size_a_per_thread / 64 * Latencys[2];
            }
            cost += size_out_per_thread / 64 * Latencys[2];
            last_hit_level = 2;
        }
        cost += size_b_per_thread / 64 * Latencys[3];

        return cost / parrelMemAccess * threadNum * threadCostRatio;        
    }
    float accessGEMMSimple(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                  void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size) {
        float cost = 0;
        auto input_size_all =
            std::accumulate(inputs_mem_size, inputs_mem_size + inputs_size, size_t(0), std::plus<size_t>());
        auto output_size_all =
            std::accumulate(outputs_mem_size, outputs_mem_size + outputs_size, size_t(0), std::plus<size_t>());
        // auto size_all_per_thread = (input_size_all + output_size_all) / threadNum;
        // int cur_hit_level;
        // if (size_all_per_thread >= L2Size + L3Size) {
        //     // hit ddr
        //     cur_hit_level = std::max(3, last_hit_level);
        //     last_hit_level = 3;
        // } else if (size_all_per_thread <= L2Size) {
        //     // hit L2
        //     cur_hit_level = std::max(1, last_hit_level);
        //     last_hit_level = 1;
        // } else {
        //     // hit L2 + L3
        //     cur_hit_level = std::max(2, last_hit_level);
        //     last_hit_level = 2;
        // }
        // cost = size_all_per_thread / 64 * Latencys[cur_hit_level];
        auto size_a_per_thread = inputs_mem_size[0] / threadNum;
        auto size_b_per_thread = inputs_mem_size[1] / threadNum;
        auto size_c_per_thread = outputs_mem_size[0] / threadNum;
        auto size_all_per_thread = size_a_per_thread + size_b_per_thread + size_c_per_thread;
        int cur_hit_level;
        if (size_all_per_thread >= L2Size + L3Size) {
            // simply treat it will hit ddr:
            // if size_all_per_thread > L2 + L3, means it will use sub-blocks. Assume splitting A,
            //  the first A/C block a1/c1 may hit L2, from second block a2/c2 it will miss L2.
            cur_hit_level = std::max(3, last_hit_level);
            last_hit_level = 3;
            // Matrix a, c will be decided by previous node and current node size
            cost = (size_a_per_thread + size_c_per_thread) / 64 * Latencys[cur_hit_level];
        } else if (size_all_per_thread <= L2Size) {
            // hit L2
            // Matrix a
            if (last_hit_level == 3) {
                cost = size_a_per_thread / 64 * Latencys[3];
            } else if (last_hit_level == 1) {
                cost = size_a_per_thread / 64 * Latencys[1];
            } else {
                cost = size_a_per_thread / 64 * Latencys[2];
            }
            // Matrix c
            cost += size_c_per_thread / 64 * Latencys[1];
            last_hit_level = 1;
        } else {
            // hit L2 + L3
            // Matrix a
            if (last_hit_level == 3) {
                cost = size_a_per_thread / 64 * Latencys[3];
            } else {
                // could not determine in L2 or L3, use the bad case
                cost = size_a_per_thread / 64 * Latencys[2];
            }
            // Matrix c
            cost += size_c_per_thread / 64 * Latencys[2];
            last_hit_level = 2;
        }
        // Matrix b assumes not in cache
        cost += size_b_per_thread / 64 * Latencys[3];

        return cost / parrelMemAccess * threadNum * threadCostRatio;
    }
    float accessConvSimple(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                  void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size) {
        float cost = 0;
        // convolution loop order is decided by minimum size of A or B
        //  In multithread, the minimum block will be loaded in each thread
        //  Need to consider: if A>L2 && B>L2 mean the minimum block will be loaded multiple times
        size_t thread_num_A = inputs_mem_size[0] >= inputs_mem_size[1] ? threadNum : 1;
        size_t thread_num_B = inputs_mem_size[0] < inputs_mem_size[1] ? threadNum : 1;
        auto size_a_per_thread = inputs_mem_size[0] / thread_num_A;
        auto size_b_per_thread = inputs_mem_size[1] / thread_num_B;
        auto size_c_per_thread = outputs_mem_size[0] / threadNum;
        auto size_all_per_thread = size_a_per_thread + size_b_per_thread + size_c_per_thread;
        int cur_hit_level;
        if (size_all_per_thread >= L2Size + L3Size) {
            // simply treat it will hit ddr:
            // if size_all_per_thread > L2 + L3, means it will use sub-blocks. Assume splitting A,
            //  the first A/C block a1/c1 may hit L2, from second block a2/c2 it will miss L2.
            cur_hit_level = std::max(3, last_hit_level);
            last_hit_level = 3;
            // Matrix a, c will be decided by previous node and current node size
            cost = (size_a_per_thread + size_c_per_thread) / 64 * Latencys[cur_hit_level];
        } else if (size_all_per_thread <= L2Size) {
            // hit L2
            // Matrix a
            if (last_hit_level == 3) {
                cost = size_a_per_thread / 64 * Latencys[3];
            } else if (last_hit_level == 1) {
                cost = size_a_per_thread / 64 * Latencys[1];
            } else {
                cost = size_a_per_thread / 64 * Latencys[2];
            }
            // Matrix c
            cost += size_c_per_thread / 64 * Latencys[1];
            last_hit_level = 1;
        } else {
            // hit L2 + L3
            // Matrix a
            if (last_hit_level == 3) {
                cost = size_a_per_thread / 64 * Latencys[3];
            } else {
                // could not determine in L2 or L3, use the bad case
                cost = size_a_per_thread / 64 * Latencys[2];
            }
            // Matrix c
            cost += size_c_per_thread / 64 * Latencys[2];
            last_hit_level = 2;
        }
        // Matrix b assumes not in cache
        cost += size_b_per_thread / 64 * Latencys[3];

        return cost / parrelMemAccess * threadNum * threadCostRatio;
    }
    float access2(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                    void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size) {
        float cost = 0;
        size_t input_total = 0, output_total = 0;
        for (size_t i = 0; i < inputs_size; i++) {
            input_total += inputs_mem_size[i];
        }
        for (size_t i = 0; i < outputs_size; i++) {
            output_total += outputs_mem_size[i];
        }
        input_total /= threadNum;
        if (input_total > L2Size + L3Size) {
            cost += (L2Size + L3Size) / 64.0f * L2Latency;
            cost += (input_total - L2Size - L3Size) / 64.0f * DDRLatency;
        } else {
            cost += input_total / 64 * L2Latency;
        }
        output_total /= threadNum;
        if (output_total > L2Size + L3Size) {
            cost += (L2Size + L3Size) / 64.0f * L2Latency;
            cost += (output_total - L2Size - L3Size) / 64.0f * DDRLatency;
        } else {
            cost += output_total / 64 * L2Latency;
        }

        return cost / parrelMemAccess * threadNum * threadCostRatio;
    }
    float access0(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                    void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size, size_t threads_num = 1) {
        float cost = 0;
        // access input cost
        for (size_t i = 0; i < inputs_size; i++) {
            void* input_mem = inputs_mem[i];
            size_t input_mem_size = inputs_mem_size[i] / threads_num;
            auto it = blocks.find(input_mem);
            // cached and not overided
            if (it != blocks.end() && (*it).first == 0) {
                auto mem_size = input_mem_size;
                if (mem_size > L2Size) {
                    cost += (mem_size - L2Size) / 64 / parrelMemAccess * L3Latency;
                    mem_size = L2Size;
                }
                cost += mem_size / 64 / parrelMemAccess * L2Latency;
            } else {
                cost += input_mem_size / 64 / parrelMemAccess * DDRLatency;
                tryCache(input_mem, input_mem_size);
            }
        }

        // access output cost
        for (size_t i = 0; i < outputs_size; i++) {
            void* output_mem = outputs_mem[i];
            size_t output_mem_size = outputs_mem_size[i] / threads_num;
            if (output_mem_size < L2Size + L3Size) {
                auto mem_size = output_mem_size;
                if (mem_size > L2Size) {
                    cost += (mem_size - L2Size) / 64 / parrelMemAccess * L3Latency;
                    mem_size = L2Size;
                }
                cost += mem_size / 64 / parrelMemAccess * L2Latency;
                tryCache(output_mem, output_mem_size);
            } else {
                cost += output_mem_size / 64 / parrelMemAccess * DDRLatency;
                // too big
                resetCache();
            }
        }

        return cost;
    }
};

struct SimGraph {
    SimMemory memSubSystem1, memSubSystem4, memSubSystem8;
    SimGraph() : memSubSystem1(1), memSubSystem4(4), memSubSystem8(8) {}

    bool isLowPrecision(ngraph::element::Type type) {
        return (type == ngraph::element::i8) || (type == ngraph::element::u8);
    }
    bool isHalfPrecision(ngraph::element::Type type) {
        return (type == ngraph::element::bf16) || (type == ngraph::element::f16);
    }
    size_t getTypeSize(ngraph::element::Type type) {
        const bool isINT8 = isLowPrecision(type);
        const bool isBF16orFP16 = isHalfPrecision(type);
        return isINT8 ? 1 : isBF16orFP16 ? 2 : 4; 
    }
    struct Ret {
        float cost1 = 0.0f;
        float memCost1 = 0.0f;
        float computeCost1 = 0.0f;
        float cost4 = 0.0f;
        float memCost4 = 0.0f;
        float computeCost4 = 0.0f;
        float cost8 = 0.0f;
        float memCost8 = 0.0f;
        float computeCost8 = 0.0f;        
    };
    void accessMem(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                   void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size, Ret& ret) {
        ret.memCost1 = memSubSystem1.access1Simple(inputs_mem, inputs_mem_size, inputs_size,
            outputs_mem, outputs_mem_size, outputs_size);
        ret.memCost4 = memSubSystem4.access1Simple(inputs_mem, inputs_mem_size, inputs_size,
            outputs_mem, outputs_mem_size, outputs_size);
        ret.memCost8 = memSubSystem8.access1Simple(inputs_mem, inputs_mem_size, inputs_size,
            outputs_mem, outputs_mem_size, outputs_size);
    }
    void accessMemGEMM(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                   void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size, Ret& ret) {
        ret.memCost1 = memSubSystem1.accessGEMMSimple(inputs_mem, inputs_mem_size, inputs_size,
            outputs_mem, outputs_mem_size, outputs_size);
        ret.memCost4 = memSubSystem4.accessGEMMSimple(inputs_mem, inputs_mem_size, inputs_size,
            outputs_mem, outputs_mem_size, outputs_size);
        ret.memCost8 = memSubSystem8.accessGEMMSimple(inputs_mem, inputs_mem_size, inputs_size,
            outputs_mem, outputs_mem_size, outputs_size);
    }
    void accessMemConv(void** inputs_mem, size_t* inputs_mem_size, size_t inputs_size,
                   void** outputs_mem, size_t* outputs_mem_size, size_t outputs_size, Ret& ret) {
        ret.memCost1 = memSubSystem1.accessConvSimple(inputs_mem, inputs_mem_size, inputs_size,
            outputs_mem, outputs_mem_size, outputs_size);
        ret.memCost4 = memSubSystem4.accessConvSimple(inputs_mem, inputs_mem_size, inputs_size,
            outputs_mem, outputs_mem_size, outputs_size);
        ret.memCost8 = memSubSystem8.accessConvSimple(inputs_mem, inputs_mem_size, inputs_size,
            outputs_mem, outputs_mem_size, outputs_size);
    }
    Ret runFake(const std::shared_ptr<ngraph::opset1::MatMul>& node, bool lastIsFQ) {
        Ret ret;

        std::vector<void*> inputs_mem(2);
        std::vector<size_t> inputs_size(2);
        std::vector<void*> outputs_mem(1);
        std::vector<size_t> outputs_size(1);
        size_t ops;
        const auto input0 = node->input(0);
        const auto input1 = node->input(1);
        const auto output = node->output(0);

        // Check that input and output shape a fully defined (not dynamic)
        if (input0.get_partial_shape().is_static() && input1.get_partial_shape().is_static() &&
            output.get_partial_shape().is_static()) {
            const auto& shapeInput0 = input0.get_shape();
            const auto& shapeInput1 = input1.get_shape();
            const auto& shapeOutput = output.get_shape();
            size_t IC;
            size_t OC;
            const auto dataSizeInput0 =
                std::accumulate(shapeInput0.begin(), shapeInput0.end(), size_t(1), std::multiplies<size_t>());
            const auto dataSizeInput1 =
                std::accumulate(shapeInput1.begin(), shapeInput1.end(), size_t(1), std::multiplies<size_t>());
            if (node->get_transpose_b()) {
                ops = dataSizeInput0;// * shapeInput1[shapeInput1.size() - 2];
                OC = shapeInput1[shapeInput1.size() - 2];
                IC = shapeInput1[shapeInput1.size() - 1];
            } else {
                ops = dataSizeInput0;// * shapeInput1[shapeInput1.size() - 1];
                OC = shapeInput1[shapeInput1.size() - 1];
                IC = shapeInput1[shapeInput1.size() - 2];
            }
            const auto dataSizeOutput =
                std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());

            auto typeSize = getTypeSize(input1.get_element_type());
            inputs_size[0] = dataSizeInput0 * typeSize;
            inputs_mem[0] = node->input_value(0).get_node();
            inputs_size[1] = dataSizeInput1 * typeSize;
            inputs_mem[1] = node->input_value(1).get_node();
            outputs_mem[0] = output.get_node();
            outputs_size[0] = dataSizeOutput * (lastIsFQ ? typeSize : getTypeSize(output.get_element_type()));

            if (typeSize == 1) {
                ops = ops / IC * ((IC + 63) / 64 * 64) * ((OC + 15) / 16 * 16);
                ret.computeCost1 = static_cast<float>(ops) / 1024 * 3 / 2;
                ret.computeCost1 += dataSizeOutput / (16 * 16) * (52 - 16);
            } else if (typeSize == 2) {
                ops = ops / IC * ((IC + 31) / 32 * 32) * ((OC + 15) / 16 * 16);
                ret.computeCost1 = static_cast<float>(ops) / 512 * 3 / 2;
                ret.computeCost1 += dataSizeOutput / (16 * 16) * (52 - 16);
            } else {
                ops = ops / IC * ((IC + 15) / 16 * 16) * ((OC + 15) / 16 * 16);
                ret.computeCost1 = static_cast<float>(ops) / 32;
            }
            ret.computeCost4 = ret.computeCost1 * memSubSystem4.threadCostRatio;
            ret.computeCost8 = ret.computeCost1 * memSubSystem8.threadCostRatio;
            if (shapeInput0.size() == shapeInput1.size() && shapeInput0.size() >= 3) {
                // ND x ND, B matrix has batch dimension
                accessMemGEMM(&inputs_mem[0],
                              &inputs_size[0],
                              inputs_mem.size(),
                              &outputs_mem[0],
                              &outputs_size[0],
                              outputs_mem.size(),
                              ret);
            } else {
                // 3D x 2D, 3D x 1D, 2D x 2D, B matrix has no batch dimension
                accessMemConv(&inputs_mem[0],
                              &inputs_size[0],
                              inputs_mem.size(),
                              &outputs_mem[0],
                              &outputs_size[0],
                              outputs_mem.size(),
                              ret);
            }
            ret.cost1 = std::max(ret.memCost1, ret.computeCost1);
            ret.cost4 = std::max(ret.memCost4, ret.computeCost4);
            ret.cost8 = std::max(ret.memCost8, ret.computeCost8);
        }
        
        return ret;
    }
    Ret runFake(const std::shared_ptr<ngraph::opset1::Convolution>& node, bool lastIsFQ) {
        Ret ret;

        std::vector<void*> inputs_mem(2);
        std::vector<size_t> inputs_size(2);
        std::vector<void*> outputs_mem(1);
        std::vector<size_t> outputs_size(1);
        size_t ops;
        const auto input0 = node->input(0);
        const auto input1 = node->input_value(1);
        const auto output = node->output(0);

        // Check that input and output shape a fully defined (not dynamic)
        if (input0.get_partial_shape().is_static() && input1.get_partial_shape().is_static() &&
            output.get_partial_shape().is_static()) {
            const auto& shapeInput0 = input0.get_shape();
            const auto& weightDims = input1.get_shape();
            const auto& shapeOutput = output.get_shape();
            const auto OC = weightDims[0];
            const auto IC = weightDims[1];
            const auto dataSizeInput0 =
                std::accumulate(shapeInput0.begin(), shapeInput0.end(), size_t(1), std::multiplies<size_t>());
            const auto dataSizeInput1 =
                std::accumulate(weightDims.begin(), weightDims.end(), size_t(1), std::multiplies<size_t>());
            const auto dataSizeOutput =
                std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());
            ops = dataSizeInput1 / weightDims[0] * dataSizeOutput;

            auto typeSize = getTypeSize(input1.get_element_type());
            inputs_size[0] = dataSizeInput0 * typeSize;
            inputs_mem[0] = node->input_value(0).get_node();
            inputs_size[1] = dataSizeInput1 * typeSize;
            inputs_mem[1] = node->input_value(1).get_node();
            outputs_mem[0] = output.get_node();
            outputs_size[0] = dataSizeOutput * (lastIsFQ ? typeSize : getTypeSize(output.get_element_type()));

            if (typeSize == 1) {
                ops = ops / IC / OC * ((IC + 63) / 64 * 64) * ((OC + 15) / 16 * 16);
                // our implementation almost 1 amx ops : 1 amx load, and most load will be from L2 and the throughput will be
                //   at least 23 cycles(from SOM), almost 1.5 times tdp* throuphput. Using pure memory latency calculation will not be accurate
                ret.computeCost1 = static_cast<float>(ops) / 1024 * 3 / 2;
                // if sequence is tilezero(x1)->tdp*(x1, x2, x3) it seems the execution will serialize, count the latency here
                ret.computeCost1 += dataSizeOutput / (16 * 16) * (52 - 16);
            } else if (typeSize == 2) {
                ops = ops / IC / OC * ((IC + 31) / 32 * 32) * ((OC + 15) / 16 * 16);
                ret.computeCost1 = static_cast<float>(ops) / 512 * 3 / 2;
                ret.computeCost1 += dataSizeOutput / (16 * 16) * (52 - 16);
            } else {
                ops = ops / IC / OC * ((IC + 15) / 16 * 16) * ((OC + 15) / 16 * 16);
                ret.computeCost1 = static_cast<float>(ops) / 32;
            }
            ret.computeCost4 = ret.computeCost1 * memSubSystem4.threadCostRatio;
            ret.computeCost8 = ret.computeCost1 * memSubSystem8.threadCostRatio;
            accessMemConv(&inputs_mem[0],
                      &inputs_size[0],
                      inputs_mem.size(),
                      &outputs_mem[0],
                      &outputs_size[0],
                      outputs_mem.size(),
                      ret);
            ret.cost1 = std::max(ret.memCost1, ret.computeCost1);
            ret.cost4 = std::max(ret.memCost4, ret.computeCost4);
            ret.cost8 = std::max(ret.memCost8, ret.computeCost8);
        }

        return ret;
    }
    Ret runFake(const std::shared_ptr<ngraph::opset1::ConvolutionBackpropData>& node, bool lastIsFQ) {
        Ret ret;
        std::cout << "ConvolutionBackpropData" << "\n";
        return ret;
    }
    Ret runFake(const std::shared_ptr<ov::Node>& node, bool firstItemInSubgraph) {
        Ret ret;
        const auto inputs = node->input_values();
        const auto outputs = node->outputs();

        bool canEstimated = true;
        std::vector<void*> inputs_mem(inputs.size());
        std::vector<size_t> inputs_size(inputs.size());
        size_t ops = 0;
        int data_size = 4;
        for (size_t i = 0; i < inputs.size(); i++) {
            if (inputs[i].get_partial_shape().is_static()) {
                const int data_type_size = getTypeSize(inputs[i].get_element_type());
                const auto& shapeInput = inputs[i].get_shape();
                const auto dataSizeInput =
                    std::accumulate(shapeInput.begin(), shapeInput.end(), size_t(1), std::multiplies<size_t>());
                inputs_size[i] = dataSizeInput * data_type_size;
                inputs_mem[i] = inputs[i].get_node();
                ops += dataSizeInput;
                data_size = std::min(data_type_size, data_size);
            } else {
                canEstimated = false;
            }
        }
        std::vector<void*> outputs_mem(outputs.size());
        std::vector<size_t> outputs_size(outputs.size());
        for (size_t i = 0; i < outputs.size(); i++) {
            if (outputs[i].get_partial_shape().is_static()) {
                const int data_type_size = getTypeSize(outputs[i].get_element_type());
                const auto& shapeOutput = outputs[i].get_shape();
                const auto dataSizeOutput =
                    std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());
                outputs_size[i] = dataSizeOutput * data_type_size;
                outputs_mem[i] = node.get();   // a node with multiple output?
            } else {
                canEstimated = false;
            }
        }
        if (canEstimated) {
            if (node->get_type_info() == ngraph::opset10::Mish::get_type_info_static()) {
                ret.computeCost1 = static_cast<float>(ops) / 16 * 30;
            } else {
                if (data_size == 1) {
                    ret.computeCost1 = static_cast<float>(ops) / 128;
                } else if (data_size == 2) {
                    ret.computeCost1 = static_cast<float>(ops) / 64;
                } else {
                    ret.computeCost1 = static_cast<float>(ops) / 32;
                }
            }
            ret.computeCost4 = ret.computeCost1 * memSubSystem4.threadCostRatio;
            ret.computeCost8 = ret.computeCost1 * memSubSystem8.threadCostRatio;

            if (firstItemInSubgraph) {
                accessMem(&inputs_mem[0],
                          &inputs_size[0],
                          inputs_mem.size(),
                          &outputs_mem[0],
                          &outputs_size[0],
                          outputs_mem.size(),
                          ret);
            }
            ret.cost1 = std::max(ret.memCost1, ret.computeCost1);
            ret.cost4 = std::max(ret.memCost4, ret.computeCost4);
            ret.cost8 = std::max(ret.memCost8, ret.computeCost8);
        }

        return ret;
    }
    bool fuseNodeTo(std::vector<std::shared_ptr<ov::Node>>& subgraph, std::shared_ptr<ov::Node>& new_node) {
        if (subgraph.empty()) {
            subgraph.emplace_back(new_node);
            return true;
        }
        // pattern:
        // 1, eltwise -> ... -> [eltwise] (snippets already handled)
        // 2, Mat/Conv -> [eltwise]
        // 3, FakeQuant -> ([Sub] -> Mat/Conv -> [eltwise] -> [FakeQuant])
        const auto& first_node = subgraph[0];
        const auto& last_node = subgraph.back();
        // new_node must follow last node
        bool is_parent = false;
        for (size_t i = 0; i < new_node->get_input_size(); i++) {
            if (new_node->input_value(i).get_node_shared_ptr() == last_node) {
                is_parent = true;
                break;
            }
        }
        if (!is_parent) {
            return false;
        }
        const auto& first_node_type_info = first_node->get_type_info();
        const auto& new_node_type_info = new_node->get_type_info();
        static const size_t elt_types[] = {
            ngraph::opset1::Abs::get_type_info_static().hash(),
            ngraph::opset1::Acos::get_type_info_static().hash(),
            ngraph::opset1::Add::get_type_info_static().hash(),
            ngraph::opset1::Asin::get_type_info_static().hash(),
            ngraph::opset1::Atan::get_type_info_static().hash(),
            // ngraph::opset1::BatchNormInference::get_type_info_static().hash(),
            // ngraph::opset1::Broadcast::get_type_info_static().hash(),
            ngraph::opset1::Ceiling::get_type_info_static().hash(),
            ngraph::opset1::Clamp::get_type_info_static().hash(),
            ngraph::opset1::Convert::get_type_info_static().hash(),
            ngraph::opset1::ConvertLike::get_type_info_static().hash(),
            ngraph::opset1::Cos::get_type_info_static().hash(),
            ngraph::opset1::Cosh::get_type_info_static().hash(),
            ngraph::opset1::Divide::get_type_info_static().hash(),
            ngraph::opset1::Elu::get_type_info_static().hash(),
            ngraph::opset1::Erf::get_type_info_static().hash(),
            ngraph::opset1::Exp::get_type_info_static().hash(),
            ngraph::opset1::FakeQuantize::get_type_info_static().hash(),
            ngraph::opset1::Floor::get_type_info_static().hash(),
            ngraph::opset1::FloorMod::get_type_info_static().hash(),
            ngraph::opset1::HardSigmoid::get_type_info_static().hash(),
            ngraph::opset1::Mod::get_type_info_static().hash(),
            ngraph::opset1::Multiply::get_type_info_static().hash(),
            ngraph::opset1::Negative::get_type_info_static().hash(),
            ngraph::opset1::PRelu::get_type_info_static().hash(),
            ngraph::opset1::Relu::get_type_info_static().hash(),
            ngraph::opset1::Sigmoid::get_type_info_static().hash(),
            ngraph::opset1::Sin::get_type_info_static().hash(),
            ngraph::opset1::Sinh::get_type_info_static().hash(),
            ngraph::opset1::Sqrt::get_type_info_static().hash(),
            ngraph::opset1::ShapeOf::get_type_info_static().hash(),
            ngraph::opset1::Squeeze::get_type_info_static().hash(),
            ngraph::opset1::Subtract::get_type_info_static().hash(),
            ngraph::opset1::Tan::get_type_info_static().hash(),
            ngraph::opset1::Tanh::get_type_info_static().hash(),
            ngraph::opset1::Unsqueeze::get_type_info_static().hash(),
            ngraph::opset1::Xor::get_type_info_static().hash(),
            ngraph::opset10::Mish::get_type_info_static().hash(),
            ngraph::opset10::Swish::get_type_info_static().hash(),
            ngraph::opset10::Gelu::get_type_info_static().hash(),
            ngraph::opset10::HSwish::get_type_info_static().hash(),
            ngraph::opset10::Acosh::get_type_info_static().hash(),
            ngraph::opset10::Asinh::get_type_info_static().hash(),
            ngraph::opset10::Atanh::get_type_info_static().hash(),
            ngraph::opset10::HSigmoid::get_type_info_static().hash(),
            ngraph::opset10::LogSoftmax::get_type_info_static().hash(),
            ngraph::opset10::Round::get_type_info_static().hash(),
            ngraph::opset10::Gelu::get_type_info_static().hash(),
            ngraph::opset10::Softmax::get_type_info_static().hash(),
        };

        static const size_t compute_types [] = {
            ngraph::opset1::Convolution::get_type_info_static().hash(),
            ngraph::opset1::GroupConvolution::get_type_info_static().hash(),
            ngraph::opset1::ConvolutionBackpropData::get_type_info_static().hash(),
            ngraph::opset1::GroupConvolutionBackpropData::get_type_info_static().hash(),
            ngraph::opset1::MatMul::get_type_info_static().hash(),
            ngraph::opset1::ConvolutionBackpropData::get_type_info_static().hash(),
        };

        auto new_node_is_compute = std::any_of(&compute_types[0],
                                  &compute_types[sizeof(compute_types) / sizeof(compute_types[0])],
                                  [&new_node_type_info](const size_t t) {
            return new_node_type_info.hash() == t;
        });
        if (new_node_is_compute) {
            // fq->subtract->conv
            if (first_node_type_info == ngraph::opset1::Subtract::get_type_info_static() && subgraph.size() == 1) {
                auto type = first_node->get_input_element_type(0);
                if (type == ov::element::i8 || type == ov::element::u8) {
                    // subtract will be fused as zp, ignore it
                    subgraph[0] = new_node;
                    return true;
                }
            }
            // computational op, should be a new subgraph
            return false;
        }
        auto new_node_is_elt = std::any_of(&elt_types[0],
                                  &elt_types[sizeof(elt_types) / sizeof(elt_types[0])],
                                  [&new_node_type_info](const size_t t) {
            return new_node_type_info.hash() == t;
        });
        if (!new_node_is_elt) {
            return false;
        }
        auto first_node_is_compute = std::any_of(&compute_types[0],
                                                 &compute_types[sizeof(compute_types) / sizeof(compute_types[0])],
                                                 [&first_node_type_info](const size_t t) {
                                                     return first_node_type_info.hash() == t;
                                                 });
        if (first_node_is_compute) {
            // case: Mat/Conv -> [eltwise] -> [FakeQuant] -> [Subtract]
            if (subgraph.back()->get_type_info() == ngraph::opset1::FakeQuantize::get_type_info_static() &&
                new_node_type_info == ngraph::opset1::Subtract::get_type_info_static())
                return true;
            // case: Mat/Conv -> [eltwise] -> [FakeQuant]
            subgraph.emplace_back(new_node);
            return true;
        } else {
            auto first_node_is_elt = std::any_of(&elt_types[0],
                                                 &elt_types[sizeof(elt_types) / sizeof(elt_types[0])],
                                                 [&first_node_type_info](const size_t t) {
                                                     return first_node_type_info.hash() == t;
                                                 });
            if (first_node_is_elt) {
                // case: eltwise -> ... -> [eltwise]
                subgraph.emplace_back(new_node);
                return true;
            }
        }
        return false;
    }
    Ret runFake(const std::shared_ptr<ngraph::Function>& nGraphFunc) {
        Ret ret;

        std::vector<std::shared_ptr<ov::Node>> subgraph;
        for (auto& node : nGraphFunc->get_ordered_ops()) {
            const auto& type_info = node->get_type_info();
            if (type_info == ngraph::opset1::Parameter::get_type_info_static() ||
                type_info == ngraph::opset1::Result::get_type_info_static() ||
                type_info == ngraph::opset1::ShapeOf::get_type_info_static() ||
                type_info == ngraph::opset1::Reshape::get_type_info_static() ||
                type_info == ngraph::opset1::Gather::get_type_info_static() ||
                type_info == ngraph::opset10::Gather::get_type_info_static() ||
                type_info == ngraph::opset10::LSTMSequence::get_type_info_static() || // TODO: text-recognition-0012
                type_info == ngraph::opset1::Constant::get_type_info_static())
                continue;

            if (node->get_friendly_name() == "conv2d_7/Conv2D" || node->get_friendly_name() == "MatMul_853")
                std::cout << "xxx";
            if (fuseNodeTo(subgraph, node))
                continue;
            bool last_is_fq = subgraph.back()->get_type_info() == ngraph::opset1::FakeQuantize::get_type_info_static();
            for (size_t i = 0; i < subgraph.size(); i++) {
                auto& nodeInSubgraph = subgraph[i];
                bool first = i == 0;
                Ret cur;

                if (nodeInSubgraph->get_friendly_name() == "conv2d_7/Conv2D" ||
                    nodeInSubgraph->get_friendly_name() == "MatMul_853")
                    std::cout << "xxx";
                const auto& type_info = nodeInSubgraph->get_type_info();
                if (type_info == ngraph::opset1::MatMul::get_type_info_static()) {
                    cur = runFake(std::dynamic_pointer_cast<ngraph::opset1::MatMul>(nodeInSubgraph), last_is_fq);
                } else if (type_info == ngraph::opset1::Convolution::get_type_info_static()) {
                    cur = runFake(std::dynamic_pointer_cast<ngraph::opset1::Convolution>(nodeInSubgraph), last_is_fq);
                } else if (type_info == ngraph::opset1::ConvolutionBackpropData::get_type_info_static()) {
                    cur = runFake(std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(nodeInSubgraph),
                                  last_is_fq);
                } else {
                    cur = runFake(nodeInSubgraph, first);
                    continue;
                }
                std::cout << "name: " << nodeInSubgraph->get_friendly_name() << " " << nodeInSubgraph->get_type_name() << " "
                        << cur.memCost1 << " " << cur.computeCost1 << " " << cur.memCost4 << " " << cur.computeCost4 << " " << cur.memCost8 << " " << cur.computeCost8 << "\n";

                ret.cost1 += cur.cost1;
                ret.cost4 += cur.cost4;
                ret.cost8 += cur.cost8;
                ret.memCost1 += cur.memCost1;
                ret.memCost4 += cur.memCost4;
                ret.memCost8 += cur.memCost8;
                ret.computeCost1 += cur.computeCost1;
                ret.computeCost4 += cur.computeCost4;
                ret.computeCost8 += cur.computeCost8;
            }
            subgraph.clear();
            subgraph.emplace_back(node);
        }
        std::cout << "cost1 " << ret.cost1 << " cost4 " << ret.cost4 << " cost8 " << ret.cost8 << "\n";
        std::cout << "cost1Compute " << ret.computeCost1 << " cost4Compute " << ret.computeCost4 << " cost8Compute " << ret.computeCost8 << "\n";
        std::cout << "cost1Mem " << ret.memCost1 << " cost4Mem " << ret.memCost4 << " cost8Mem " << ret.memCost8 << " zziiuu\n";

        return ret;
    }
};

static MemBandwidthPressure MemBandwidthPressureTolerance(
    const std::shared_ptr<ngraph::Function> nGraphFunc,
    const float cache_size,
    const float memThresholdAssumeLimited = MemBandwidthPressure::LIMITED) {
    int total_convs = 0, mem_limited_convs = 0, compute_convs = 0, total_gemms = 0, mem_limited_gemms = 0,
        total_deconvs = 0, compute_deconvs = 0, mem_limited_deconvs = 0;
    auto memLimitedFactor = [&](int size_data_moved, int datatype_size = 4) -> float {
        return (cache_size / (size_data_moved * datatype_size));
    };
    auto isLowPrecision = [&](ngraph::element::Type type) -> bool {
        return (type == ngraph::element::i8) || (type == ngraph::element::u8);
    };
    auto isHalfPrecision = [&](ngraph::element::Type type) -> bool {
        return (type == ngraph::element::bf16) || (type == ngraph::element::f16);
    };

    float worst_case = MemBandwidthPressure::UNKNOWN;

    SimGraph sim;
    sim.runFake(nGraphFunc);
    // Traverse nGraph Function in topological order
    for (auto& node : nGraphFunc->get_ordered_ops()) {
        const auto node_name = node->get_type_info().name;
        if (std::strcmp("MatMul", node_name) && std::strcmp("Convolution", node_name) &&
            std::strcmp("ConvolutionBackpropData", node_name)) {
            if (!std::strcmp("GRUSequence", node_name) || !std::strcmp("TensorIterator", node_name)) {
                MemBandwidthPressure res;
                res.max_mem_tolerance = MemBandwidthPressure::UNKNOWN;
                return res;
            }
            continue;
        }
        auto type1 = node->input_value(1).get_element_type();  // weights
        const bool isINT8 = isLowPrecision(type1);
        const bool isBF16orFP16 = isHalfPrecision(type1);
        const int data_type_size = isINT8 ? 1 : isBF16orFP16 ? 2 : 4;

        int dataSizeInput = 0, dataSizeOutput = 0;
        if (!std::strcmp("MatMul", node_name)) {
            const auto input0 = node->input(0);
            const auto input1 = node->input(1);
            const auto output = node->output(0);
            // Check that input and output shape a fully defined (not dynamic)
            if (input0.get_partial_shape().is_static() && input1.get_partial_shape().is_static() &&
                output.get_partial_shape().is_static()) {
                const auto& shapeInput0 = input0.get_shape();
                const auto& shapeInput1 = input1.get_shape();
                const auto non_const = !get_constant_from_source(node->input_value(1));
                const auto& shapeOutput = output.get_shape();
                const auto dataSizeInput0 =
                    std::accumulate(shapeInput0.begin(), shapeInput0.end(), size_t(1), std::multiplies<size_t>());
                const auto dataSizeInput1 =
                    std::accumulate(shapeInput1.begin(), shapeInput1.end(), size_t(1), std::multiplies<size_t>());
                dataSizeOutput =
                    std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());
                const auto total_data = dataSizeInput0 + non_const * dataSizeInput1 + dataSizeOutput;
                total_gemms++;
                const auto factor = memLimitedFactor(total_data, data_type_size);
                mem_limited_gemms += factor < memThresholdAssumeLimited;
                worst_case = std::min(factor, worst_case);
            }
        } else if (!std::strcmp("Convolution", node_name)) {
            // Check that input and output shape a fully defined (not dynamic)
            const auto input = node->input(0);
            const auto output = node->output(0);
            const auto kernels = node->input(1);
            const auto& shape = kernels.get_shape();
            total_convs++;
            if (shape.size() >= 4 /* conventional 2D/3D conv */ && shape[2] >= 3 && shape[3] >= 3) {
                compute_convs++;
                continue;
            }
            if (input.get_partial_shape().is_static() && output.get_partial_shape().is_static()) {
                const auto& shapeInput = input.get_shape();
                const auto& shapeOutput = output.get_shape();
                if (shapeInput.size() > 4 /*5D*/ && isINT8) {
                    compute_convs++;
                    continue;
                }
                dataSizeInput =
                    std::accumulate(shapeInput.begin(), shapeInput.end(), size_t(1), std::multiplies<size_t>());
                dataSizeOutput =
                    std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());
                const auto factor = memLimitedFactor(dataSizeInput + dataSizeOutput, data_type_size);
                mem_limited_convs += factor < memThresholdAssumeLimited;
                worst_case = std::min(factor, worst_case);
            }
        } else if (!std::strcmp("ConvolutionBackpropData", node_name)) {
            const auto input = node->input(0);
            const auto output = node->output(0);
            total_deconvs++;

            // Check that input and output shape a fully defined (not dynamic)
            if (input.get_partial_shape().is_static() && output.get_partial_shape().is_static()) {
                const auto shapeInput = input.get_shape();
                const auto shapeOutput = output.get_shape();
                if (shapeInput.size() > 4 /*5D*/ && isINT8) {
                    compute_deconvs++;
                    continue;
                }
                dataSizeInput =
                    std::accumulate(shapeInput.begin(), shapeInput.end(), size_t(1), std::multiplies<size_t>());
                dataSizeOutput =
                    std::accumulate(shapeOutput.begin(), shapeOutput.end(), size_t(1), std::multiplies<size_t>());
                const auto factor = memLimitedFactor(dataSizeInput + dataSizeOutput, data_type_size);
                mem_limited_deconvs += factor < memThresholdAssumeLimited;
                worst_case = std::min(factor, worst_case);
            }
        }
    }
    MemBandwidthPressure res;
    res.max_mem_tolerance = worst_case;
    res.ratio_mem_limited_convs = total_convs ? static_cast<float>(mem_limited_convs) / total_convs : 0;
    res.ratio_compute_convs = total_convs ? static_cast<float>(compute_convs) / total_convs : 0;
    res.ratio_compute_deconvs = total_deconvs ? static_cast<float>(compute_deconvs) / total_deconvs : 0;
    return res;
}

}  // namespace ov
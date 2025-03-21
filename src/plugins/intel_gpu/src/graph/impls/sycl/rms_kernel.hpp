#pragma once

#include "sycl/sycl.hpp"
#include "sycl/ext/oneapi/experimental/builtins.hpp"

namespace cldnn::sycl::details {

template<typename Type>
inline ::sycl::event rms_kernel(::sycl::queue& queue,
    const Type* in_buf, const Type* weight, Type* out_buf, const size_t bs, const size_t channel_num, float epsilon) {
    const size_t WG_SIZE = 256;
    return queue.submit([=](::sycl::handler& h) {
            // __local float variance[16];
            ::sycl::local_accessor<float, 1> variance(::sycl::range(WG_SIZE / 32), h);
            ::sycl::stream out(65536, 128, h);

            h.parallel_for(::sycl::nd_range<1>(bs * WG_SIZE, WG_SIZE), [=](::sycl::nd_item<1> index) [[intel::reqd_sub_group_size(32)]] {
            auto g = index.get_group();
            int row = g[0];
            int id_local = index.get_local_id();
            auto sg = index.get_sub_group();
            int id_sg = sg.get_group_id();
            int id_sg_local = sg.get_local_id();
            auto input = in_buf + row * channel_num;
            auto output = out_buf + row * channel_num;
            float local_var = 0;

            for (int i = id_local; i < channel_num; i += WG_SIZE) {
                // half/fp16 has very limited range: ±65,504
                // which may cause square result to overflow,
                // the square must be done in fp32
                float x = (float)input[i];
                local_var += x * x;
            }

            local_var = ::sycl::reduce_over_group(sg, local_var, ::sycl::plus<>());
            // local_var = sub_group_reduce_add(local_var);
            if (id_sg_local == 0)
                variance[id_sg] = local_var;
            // barrier(CLK_LOCAL_MEM_FENCE);
            index.barrier(::sycl::access::fence_space::local_space);
            if (id_sg_local < WG_SIZE / 32)
                local_var = variance[id_sg_local];
            else
                local_var = 0;
            // float all_variance = sub_group_reduce_add(local_var);
            float all_variance = ::sycl::reduce_over_group(sg, local_var, ::sycl::plus<>());
            float scale = ::sycl::rsqrt((all_variance / channel_num) + epsilon);

            for (int i = id_local; i < channel_num; i += WG_SIZE) {
                output[i] = input[i] * scale * weight[i];
            }
        });
    });
}

}
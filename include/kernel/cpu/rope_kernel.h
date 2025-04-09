#ifndef _ROPE_CPU_H_
#define _ROPE_CPU_H_
#include "tensor.h"
namespace kernel {
    void rope_cache_cal(int head_size, int max_seq_len, const mem::Tensor sin_cache, const mem::Tensor cos_cache, float rope_theta);
    
    void rope_kernel_cpu(const mem::Tensor& input_q, const mem::Tensor& input_k, const mem::Tensor& pos_now, 
        const mem::Tensor& sin_cache, const mem::Tensor& cos_cache, int32_t hidden_dim_size, int32_t head_dim);
};
#endif
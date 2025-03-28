#ifndef _ROPE_CUDA_H_
#define _ROPE_CUDA_H_
#include "tensor.h"
namespace kernel {
    void rope_cache_cu_cal(int head_size, int max_seq_len, mem::Tensor sin_cache, mem::Tensor cos_cache);
    
    void rope_kernel_cuda(const mem::Tensor& input_q, const mem::Tensor& input_k, const mem::Tensor& pos_now, 
        const mem::Tensor& sin_cache, const mem::Tensor& cos_cache, int32_t dim, int32_t head_size);
};

#endif
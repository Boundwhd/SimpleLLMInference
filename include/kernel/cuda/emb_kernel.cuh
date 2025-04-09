#ifndef _EMB_CUDA_H_
#define _EMB_CUDA_H_
#include "tensor.h"

namespace kernel {
    void emb_kernel_cuda(const mem::Tensor& input, const mem::Tensor& weight,
        const mem::Tensor& output, int32_t vocab_size, int32_t hidden_dim_size);
}

#endif
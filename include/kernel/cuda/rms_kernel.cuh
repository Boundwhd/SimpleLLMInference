#ifndef _RMS_CUDA_H_
#define _RMS_CUDA_H_
#include "tensor.h"

namespace kernel {
    void rmsnorm_kernel_cuda(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor&output,
        int32_t hidden_dim_size, float eps);
};

#endif
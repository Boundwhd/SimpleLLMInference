#ifndef _MATMUL_CUDA_H_
#define _MATMUL_CUDA_H_
#include "tensor.h"
namespace kernel {
    void matmul_kernel_cuda(const mem::Tensor& input1, const mem::Tensor& weight, 
        const mem::Tensor& output);
};
#endif
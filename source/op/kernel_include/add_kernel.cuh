#ifndef _ADD_CUDA_H_
#define _ADD_CUDA_H_
#include "tensor.h"
namespace kernel {
    void add_kernel_cuda(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& output);
};
#endif
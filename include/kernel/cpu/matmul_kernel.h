#ifndef _MATMUL_CPU_H_
#define _MATMUL_CPU_H_
#include "tensor.h"

namespace kernel {
    void matmul_kernel_cpu(const mem::Tensor& input, const mem::Tensor& weight,
        const mem::Tensor& output, int32_t dim0, int32_t dim1, float scale = 1.0f);
}

#endif
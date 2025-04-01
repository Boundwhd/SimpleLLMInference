#ifndef _RMS_CPU_H_
#define _RMS_CPU_H_
#include "tensor.h"

namespace kernel {
    void rmsnorm_kernel_cpu(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor&output,
        int32_t hidden_dim_size, float eps);
};

#endif
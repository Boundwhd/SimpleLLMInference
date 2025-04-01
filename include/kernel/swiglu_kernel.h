#ifndef _SWIGLU_CPU_H_
#define _SWIGLU_CPU_H_
#include "tensor.h"
namespace kernel {
    void swiglu_kernel_cpu(const mem::Tensor& up, const mem::Tensor& gate, const mem::Tensor& output, int32_t intermediate_size);
}

#endif
#ifndef _EMB_CPU_H_
#define _EMB_CPU_H_
#include "tensor.h"
namespace kernel {
    void emb_kernel_cpu(const mem::Tensor& input, const mem::Tensor& weight,
        const mem::Tensor& output, int32_t vocab_size);
}
#endif
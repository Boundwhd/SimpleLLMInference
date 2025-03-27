#include "matmul_kernel.h"

namespace kernel {
    void matmul_kernel_cpu(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor& output) {
        if (input.get_dim(0) != weight.get_dim(1)) {
            LOG("Tensor with Wrong Dim!");
        }

        const uint row = weight.get_dim(0);
        const uint col = weight.get_dim(1);

        auto cpu_alloc = mem::CPUDeviceAllocatorFactory::get_instance();
        cpu_alloc->memset_zero(const_cast<float*>(output.ptr<float>()), output.get_dim(0) * DaraTypeSize);

        const float* in_ptr = input.ptr<float>();
        for (int i = 0; i < row; i++) {
            float* out_ptr = const_cast<float*>(output.ptr<float>() + i);
            const float* wei_ptr = weight.ptr<float>() + i * col;
            float sum = 0;
            for (int j = 0; j < col; j++) {
                sum += in_ptr[j] * wei_ptr[j];
            }
            *out_ptr = sum;
        }
    }
}
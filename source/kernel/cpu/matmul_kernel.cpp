#include "matmul_kernel.h"

// 待优化
namespace kernel {
    void matmul_kernel_cpu(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor& output, 
        int32_t dim0, int32_t dim1, float scale) {

        if (input.get_dim(0) != dim1) {
            LOG("Tensor with Wrong Dim!");
        }
        
        const uint row = dim0;
        const uint col = dim1;

        auto cpu_alloc = mem::CPUDeviceAllocatorFactory::get_instance();
        cpu_alloc->memset_zero(const_cast<float*>(output.ptr<float>()), output.get_dim(0) * DataTypeSize);

        const float* in_ptr = input.ptr<float>();
        for (int i = 0; i < row; i++) {
            float* out_ptr = const_cast<float*>(output.ptr<float>() + i);
            const float* wei_ptr = weight.ptr<float>() + i * col;
            float sum = 0;
            for (int j = 0; j < col; j++) {
                sum += in_ptr[j] * wei_ptr[j];
            }
            *out_ptr = sum * scale;
        }
    }
}
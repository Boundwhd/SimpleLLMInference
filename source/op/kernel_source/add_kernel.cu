#include "add_kernel.cuh"

namespace kernel {
    __global__ void add_kernel_cu(size_t size, const float* in1, const float* in2, float* out) {
        const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < size) {
            out[tid] = in1[tid] + in2[tid];
        }
    }
    void add_kernel_cuda(const mem::Tensor& input1, const mem::Tensor& input2, const mem::Tensor& output) {
        size_t size = input1.size();
        const uint block_size = 32;
        const uint grid_size = (size + block_size - 1) / block_size;
        add_kernel_cu<<<grid_size, block_size>>>(size, input1.ptr<float>(), input2.ptr<float>(), 
            const_cast<float*>(output.ptr<float>()));
    };
};
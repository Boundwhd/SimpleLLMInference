#include "emb_kernel.cuh"

namespace kernel {
    __global__ void emb_kernel_cu(float* in_weight, float* out, int32_t weight_dim) {
        const uint tid = threadIdx.x + blockDim.x * blockIdx.x;
        if (tid < weight_dim) {
            out[tid] = in_weight[tid];
        }
    }

    void emb_kernel_cuda(const mem::Tensor& input1, const mem::Tensor& weight, 
        const mem::Tensor& output, int32_t vocab_size) {
        const int32_t weight_dim = weight.get_dim(1);
        int32_t token = 0;
        if (input1.device_type() == base::DeviceType::kDeviceCUDA) {
            mem::Tensor input_cpu = input1.clone();
            input_cpu.to_cpu();
            token = *(input_cpu.ptr<int32_t>());
        } else {
            token = *(input1.ptr<int32_t>());
        }
        float* src_ptr = const_cast<float*>(weight.ptr<float>(token * weight_dim));
        float* dst_ptr = const_cast<float*>(output.ptr<float>());

        const uint block_size = 512;
        const uint grid_size = (weight_dim + block_size - 1) / block_size;
        emb_kernel_cu<<<grid_size, block_size>>>(src_ptr, dst_ptr, weight_dim);
    }
}
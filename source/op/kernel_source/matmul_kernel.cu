#include "matmul_kernel.cuh"

namespace kernel {
    __global__ void matmul_kernel_cu(const float* in, const float* weight, float* out, int dim) {
        int lane_id = threadIdx.x;
        int row = blockIdx.x;

        float sum = 0.0f;
        for (int i = lane_id; i < dim; i += warpSize) {
            sum += in[i] * weight[row * dim + i];
        }
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            out[row] = sum;
        }
    }

    void matmul_kernel_cuda(const mem::Tensor& input1, const mem::Tensor& weight, const mem::Tensor& output) {
        if (input1.get_dim(0) != weight.get_dim(1)) {
            LOG("Tensor with Wrong Dim!");
        }

        const uint row = weight.get_dim(0);
        const uint col = weight.get_dim(1);

        const float* in_ptr = input1.ptr<float>();
        const float* wei_ptr = weight.ptr<float>();
        float* out_ptr = const_cast<float*>(output.ptr<float>());

        const uint block_size = 32;
        const uint grid_size = row;
        matmul_kernel_cu<<<grid_size, block_size>>>(in_ptr, wei_ptr, out_ptr, col);
    }
}
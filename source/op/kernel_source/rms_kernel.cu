#include "rms_kernel.cuh"

namespace kernel {
    __global__ void rms_kernel_cu(const float* input, const float* weight, float* output, int32_t dim) {
        int laneId = threadIdx.x;

        float sum = 0;
        for (int i = laneId; i < dim; i += warpSize) {
            sum += (input[i] * input[i]);
        }

        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        __syncthreads();

        __shared__ float smem[1];
        if (laneId == 0) {
            smem[0] = sum;
        }

        sum = smem[0];
        __syncthreads();
        
        float norm_factor = rsqrtf(sum / dim + 1e-6f);
        for (int i = laneId; i < dim; i += warpSize) {
            output[i] = norm_factor * input[i] * weight[i];
        }
    }

    void rmsnorm_kernel_cuda(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor&output) {
        const float* in_ptr = input.ptr<float>();
        const float* wei_ptr = weight.ptr<float>();
        float* out_ptr = const_cast<float*>(output.ptr<float>());

        int32_t dim = input.size();
        const uint block_size = 32;
        const uint grid_size = 1;
        rms_kernel_cu<<<grid_size, block_size>>>(in_ptr, wei_ptr, out_ptr, dim);
    }
}
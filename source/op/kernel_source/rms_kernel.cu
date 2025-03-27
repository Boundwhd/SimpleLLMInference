#include "rms_kernel.cuh"

namespace kernel {
    __global__ void rms_kernel_cu(const float* input, const float* weight, float* output, int32_t dim) {
        const int laneId = threadIdx.x % warpSize;
        const int warpId = threadIdx.x / warpSize;  
        __shared__ float warp_sums[32];            

        float sum = 0.0f;
        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            sum += input[i] * input[i];
        }

        
        unsigned mask = __ballot_sync(0xffffffff, threadIdx.x < dim);
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(mask, sum, offset);
        }

        if (laneId == 0) {
            warp_sums[warpId] = sum;  // 每个Warp存一个局部和
        }
        __syncthreads();

        
        if (warpId == 0) {
            sum = (laneId < blockDim.x / warpSize) ? warp_sums[laneId] : 0.0f;
            
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            if (laneId == 0) {
                warp_sums[0] = sum;
            }
        }
        __syncthreads();

        const float total_sum = warp_sums[0];
        const float norm_factor = rsqrtf(total_sum / dim + 1e-6f);

        for (int i = threadIdx.x; i < dim; i += blockDim.x) {
            output[i] = norm_factor * input[i] * weight[i];
        }
    }

    void rmsnorm_kernel_cuda(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor&output) {
        const float* in_ptr = input.ptr<float>();
        const float* wei_ptr = weight.ptr<float>();
        float* out_ptr = const_cast<float*>(output.ptr<float>());

        int32_t dim = input.size();
        const uint block_size = 256;
        const uint grid_size = (dim + block_size - 1) / block_size;
        rms_kernel_cu<<<grid_size, block_size>>>(in_ptr, wei_ptr, out_ptr, dim);
    }
}
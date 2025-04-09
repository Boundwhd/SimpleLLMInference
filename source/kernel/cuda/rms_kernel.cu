#include "rms_kernel.cuh"
#define WARP_SIZE 32

namespace kernel {
    template<const int NUM_THREADS=256>
    __global__ void rmsnorm_f32_kernel(float* x, float* y, float* weight, float* total, int32_t hidden_dim, float eps) {
        int tid = threadIdx.x;
        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        float value = idx < hidden_dim ? x[idx] : 0.0f;
        float sum = value * value;
        for (int mask = 16; mask >= 1; mask >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
        }

        int warp = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        __shared__ float shared_mem[NUM_THREADS/32];
        if (lane == 0) {
            shared_mem[warp] = sum;
        }
        __syncthreads();
        
        sum = (lane < NUM_THREADS/32) ? shared_mem[lane] : 0.0f;
        for (int mask = 16; mask >= 1; mask >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, mask);
        }
        if (tid == 0) {
            atomicAdd(total, sum);
        }
        __threadfence();

        float mean = __fdividef(*total, hidden_dim);
        float scale = rsqrtf(mean + eps);  

        if (idx < hidden_dim) {
            y[idx] = scale * x[idx] * weight[idx];
        }
    }

    void rmsnorm_kernel_cuda(const mem::Tensor& input, const mem::Tensor& weight, const mem::Tensor&output,
        int32_t hidden_dim_size, float eps) {

        float* in_ptr = const_cast<float*>(input.ptr<float>());
        float* wei_ptr = const_cast<float*>(weight.ptr<float>());
        float* out_ptr = const_cast<float*>(output.ptr<float>());
        
        auto alloc_cuda = mem::CUDADeviceAllocatorFactory::get_instance();
        const mem::Tensor total_sum({1}, true, alloc_cuda);
        float* total_ptr = const_cast<float*>(total_sum.ptr<float>());
        alloc_cuda->memset_zero(total_ptr, 4);

        const int block_size = 256;
        const int grid_size = (hidden_dim_size + block_size - 1) / block_size;
        rmsnorm_f32_kernel<<<grid_size, block_size>>>(in_ptr, out_ptr, wei_ptr, total_ptr, hidden_dim_size, eps);
    }
}